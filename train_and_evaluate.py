import os
import logging
from tqdm import tqdm
from torchvision.utils import save_image
import torch
import utils
import scipy.io as io
import numpy as np

import fmmax_solver

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def evaluate(generator, eng, numImgs, params):
    generator.eval()

    # generate images
    z = sample_z(numImgs, params)
    images_raw = generator(z, params)
    logging.info('Generation is done. \n')

    # Evaluate efficiency before AND after sign(.) so we can detect
    # under-binarised generators (large gap = binary_amp cap too low).
    effs_raw = compute_effs(images_raw, eng, params)
    images = torch.sign(images_raw)
    effs = compute_effs(images, eng, params)

    drop = (effs_raw.mean() - effs.mean()).item()
    logging.info(f'eff_mean raw={effs_raw.mean().item():.4f}  '
                 f'sign={effs.mean().item():.4f}  drop={drop:+.4f}')

    # save images
    filename = 'imgs_w' + str(params.wavelength) +'_a' + str(params.angle) +'deg.mat'
    file_path = os.path.join(params.output_dir,'outputs',filename)
    io.savemat(file_path, mdict={'imgs': images.cpu().detach().numpy(),
                                 'effs': effs.cpu().detach().numpy(),
                                 'effs_raw': effs_raw.cpu().detach().numpy()})

    # plot histogram
    fig_path = params.output_dir + '/figures/Efficiency.png'
    utils.plot_histogram(effs.data.cpu().numpy().reshape(-1), params.numIter, fig_path)




def train(generator, optimizer, scheduler, eng, params, pca=None):

    generator.train()

    # initialization
    if params.restore_from is None:
        effs_mean_history = []
        effs_sign_history = []
        binarization_history = []
        diversity_history = []
        iter0 = 0
    else:
        effs_mean_history = params.checkpoint['effs_mean_history']
        effs_sign_history = params.checkpoint.get('effs_sign_history', [])
        binarization_history = params.checkpoint['binarization_history']
        diversity_history = params.checkpoint['diversity_history']
        iter0 = params.checkpoint['iter']

    # training loop
    with tqdm(total=params.numIter) as t:
        it = 0
        while True:
            it +=1
            params.iter = it + iter0

            # normalized iteration number
            normIter = params.iter / params.numIter

            # specify current batch size
            params.batch_size = int(params.batch_size_start +  (params.batch_size_end - params.batch_size_start) * (1 - (1 - normIter)**params.batch_size_power))

            # sigma decay
            params.sigma = params.sigma_start + (params.sigma_end - params.sigma_start) * normIter

            # binarization amplitude in the tanh function
            amp_sched = getattr(params, 'binary_amp_schedule', 'two_phase')
            if amp_sched == 'stepwise':
                # Original GLOnet: floor(iter/100) + 1, capped at 10
                params.binary_amp = min(int(params.iter / 100) + 1, 10)
            elif amp_sched == 'linear':
                # Single linear ramp: 1 → binary_amp_max over full training
                amp_max = float(getattr(params, 'binary_amp_max', 10))
                params.binary_amp = 1.0 + (amp_max - 1.0) * params.iter / params.numIter
            elif amp_sched == 'delayed_linear':
                # Delayed ramp: amp=1 until binary_step_iter, then 1→amp_max
                # until binary_amp_ramp_end, then hold at amp_max
                amp_max = float(getattr(params, 'binary_amp_max', 10))
                delay_iter = int(getattr(params, 'binary_step_iter', 400))
                ramp_end = int(getattr(params, 'binary_amp_ramp_end', params.numIter))
                if params.iter < delay_iter:
                    params.binary_amp = 1.0
                elif params.iter <= ramp_end:
                    frac = (params.iter - delay_iter) / max(ramp_end - delay_iter, 1)
                    params.binary_amp = 1.0 + (amp_max - 1.0) * frac
                else:
                    params.binary_amp = amp_max
            else:
                # Two-phase linear (default, V7 compatible)
                amp_lo = float(getattr(params, 'binary_amp_max', 6))
                amp_hi = float(getattr(params, 'binary_amp_max_phase_b', amp_lo))
                phase_iter = int(getattr(params, 'bin_phase_iter', params.numIter))
                if params.iter < phase_iter:
                    params.binary_amp = min(
                        1.0 + (amp_lo - 1.0) * params.iter / max(phase_iter, 1), amp_lo)
                else:
                    progress_b = (params.iter - phase_iter) / max(params.numIter - phase_iter, 1)
                    params.binary_amp = min(amp_lo + (amp_hi - amp_lo) * progress_b, amp_hi)

            # save model
            if it % 5000 == 0 or it > params.numIter:
                model_dir = os.path.join(params.output_dir, 'model','iter{}'.format(it+iter0))
                os.makedirs(model_dir, exist_ok = True)
                utils.save_checkpoint({'iter': it + iter0 - 1,
                                       'gen_state_dict': generator.state_dict(),
                                       'optim_state_dict': optimizer.state_dict(),
                                       'scheduler_state_dict': scheduler.state_dict(),
                                       'effs_mean_history': effs_mean_history,
                                       'effs_sign_history': effs_sign_history,
                                       'binarization_history': binarization_history,
                                       'diversity_history': diversity_history
                                       },
                                       checkpoint=model_dir)

            # terminate the loop
            if it > params.numIter:
                return


            # sample  z
            z = sample_z(params.batch_size, params)

            # generate a batch of iamges
            gen_imgs = generator(z, params)


            # calculate efficiencies and gradients using EM solver
            effs, gradients = compute_effs_and_gradients(gen_imgs, eng, params)

            # free optimizer buffer
            optimizer.zero_grad()

            # construct the loss function
            p_start = params.binary_penalty_start
            p_end = params.binary_penalty_end
            step_iter = int(params.binary_step_iter)
            ramp_end = int(getattr(params, 'binary_penalty_ramp_end', step_iter))
            if params.iter < step_iter:
                binary_penalty = p_start
            elif ramp_end > step_iter and params.iter <= ramp_end:
                frac = (params.iter - step_iter) / (ramp_end - step_iter)
                binary_penalty = p_start + (p_end - p_start) * frac
            else:
                binary_penalty = p_end
            g_loss = global_loss_function(gen_imgs, effs, gradients, params.sigma, binary_penalty)

            # train the generator
            g_loss.backward()
            optimizer.step()
            scheduler.step()


            # evaluate
            if it % params.plot_iter == 0:
                generator.eval()

                # vilualize generated images at various conditions
                visualize_generated_images(generator, params)

                # evaluate the performance of current generator
                effs_mean, effs_sign, binarization, diversity = evaluate_training_generator(generator, eng, params)

                # add to history
                effs_mean_history.append(effs_mean)
                effs_sign_history.append(effs_sign)
                binarization_history.append(binarization)
                diversity_history.append(diversity)

                # plot current history
                utils.plot_loss_history((effs_mean_history, diversity_history, binarization_history,
                                         effs_sign_history), params)
                generator.train()

            t.update()



def sample_z(batch_size, params):
    '''
    smaple noise vector z
    '''
    return (torch.rand(batch_size, params.noise_dims).type(Tensor)*2.-1.) * params.noise_amplitude


def compute_effs_and_gradients(gen_imgs, eng, params):
    '''
    Args:
        gen_imgs: N x C x H  (C=1)
        eng: unused (kept for API compatibility with the original MATLAB interface)
        params: parameters

    Returns:
        effs: N (1-D tensor)
        gradients: N x C x H
    '''
    imgs_np = gen_imgs.detach().cpu().numpy()
    if imgs_np.ndim == 3:  # (N, 1, H)
        imgs_np = imgs_np[:, 0, :]
    N, H = imgs_np.shape
    wavelengths = np.full(N, float(params.wavelength))
    angles = np.full(N, float(params.angle))
    effs_and_grads = fmmax_solver.gradient_from_solver_batch(
        imgs_np, wavelengths, angles,
        periods_nm=getattr(params, 'period_nm', None),
        nn=int(getattr(params, 'fourier_nn', fmmax_solver.DEFAULT_FOURIER_NN)),
    )
    effs = Tensor(effs_and_grads[:, 0])
    gradients = Tensor(effs_and_grads[:, 1:]).unsqueeze(1)  # (N, 1, H)
    return effs, gradients


def compute_effs(imgs, eng, params):
    '''
    Args:
        imgs: N x C x H  (C=1)
        eng: unused
        params: parameters

    Returns:
        effs: N x 1 tensor
    '''
    imgs_np = imgs.detach().cpu().numpy()
    if imgs_np.ndim == 3:
        imgs_np = imgs_np[:, 0, :]
    N = imgs_np.shape[0]
    wavelengths = np.full(N, float(params.wavelength))
    angles = np.full(N, float(params.angle))
    effs = fmmax_solver.eval_eff_batch(
        imgs_np, wavelengths, angles,
        periods_nm=getattr(params, 'period_nm', None),
        nn=int(getattr(params, 'fourier_nn', fmmax_solver.DEFAULT_FOURIER_NN)),
    )
    return Tensor(effs[:, None])



def global_loss_function(gen_imgs, effs, gradients, sigma=0.5, binary_penalty=0):
    '''
    Args:
        gen_imgs: N x C x H (x W)
        effs: N x 1
        gradients: N x C x H (x W)
        max_effs: N x 1
        sigma: scalar
        binary_penalty: scalar
    '''
    # efficiency loss
    exp_w = torch.exp(effs / sigma)
    exp_w = exp_w / torch.mean(exp_w)
    eff_loss_tensor = -gen_imgs * gradients * (1. / sigma) * exp_w.view(-1, 1, 1)
    eff_loss = torch.sum(torch.mean(eff_loss_tensor, dim=0).view(-1))

    # binarization loss
    binary_loss = - torch.mean(torch.abs(gen_imgs.view(-1)) * (2.0 - torch.abs(gen_imgs.view(-1)))) 

    # total loss
    loss = eff_loss + binary_loss * binary_penalty

    return loss



def visualize_generated_images(generator, params, n_row = 4, n_col = 4):
    # generate images and save
    fig_path = params.output_dir +  '/figures/deviceSamples/Iter{}.png'.format(params.iter)

    was_training = generator.training
    generator.eval()
    with torch.no_grad():
        z = sample_z(n_col * n_row, params)
        imgs = generator(z, params)
        imgs_2D = imgs.unsqueeze(2).repeat(1, 1, 64, 1)
        save_image(imgs_2D, fig_path, nrow=n_row, value_range=(-1, 1))
    if was_training:
        generator.train()
    


def evaluate_training_generator(generator, eng, params, num_imgs = 100):

    # generate images
    z = sample_z(num_imgs, params)
    imgs_raw = generator(z, params)

    # Raw efficiency (continuous-valued patterns).
    effs_raw = compute_effs(imgs_raw, eng, params)
    # Sign-binarised efficiency — the metric the paper actually reports.
    imgs_sign = torch.sign(imgs_raw)
    effs_sign = compute_effs(imgs_sign, eng, params)

    effs_mean_raw = torch.mean(effs_raw.view(-1)).item()
    effs_mean_sign = torch.mean(effs_sign.view(-1)).item()

    # binarization of generated images
    binarization = torch.mean(torch.abs(imgs_raw.view(-1))).cpu().detach().numpy()

    # diversity of generated images
    diversity = torch.mean(torch.std(imgs_raw, dim=0)).cpu().detach().numpy()

    # plot histogram (raw, matching legacy behaviour)
    fig_path = params.output_dir +  '/figures/histogram/Iter{}.png'.format(params.iter)
    utils.plot_histogram(effs_raw.data.cpu().numpy().reshape(-1), params.iter, fig_path)

    logging.info(f"iter {params.iter}: eff_raw={effs_mean_raw:.4f} "
                 f"eff_sign={effs_mean_sign:.4f} bin={float(binarization):.3f}")

    return effs_mean_raw, effs_mean_sign, binarization, diversity

