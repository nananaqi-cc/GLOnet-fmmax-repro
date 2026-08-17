"""
Multi-wavelength training for GLOnet × FMMAX.

Extends train_and_evaluate.py to optimise across several wavelengths
simultaneously.  The loss at each step is the (optionally weighted) mean
of the single-wavelength losses.

Usage:  see main_multi_wl.py
"""
from __future__ import annotations

import os
import logging
import numpy as np
import torch
import scipy.io as io

import fmmax_solver
import utils
from train_and_evaluate import (
    sample_z,
    visualize_generated_images,
)
from tqdm import tqdm

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


# ---------------------------------------------------------------------------
# Multi-wavelength gradient & loss
# ---------------------------------------------------------------------------
def compute_effs_and_gradients_multi(gen_imgs, params):
    """Return (all_effs, all_grads) — one entry per wavelength.

    all_effs[i]:  Tensor (N,)     efficiency at wavelength i
    all_grads[i]: Tensor (N,1,H)  per-pixel gradient at wavelength i
    """
    imgs_np = gen_imgs.detach().cpu().numpy()
    if imgs_np.ndim == 3:
        imgs_np = imgs_np[:, 0, :]
    N, H = imgs_np.shape
    wavelengths = [float(w) for w in params.wavelengths]
    angle = float(params.angle)
    period_nm = getattr(params, "period_nm", None)
    fourier_nn = int(getattr(params, "fourier_nn", fmmax_solver.DEFAULT_FOURIER_NN))

    all_effs = []
    all_grads = []
    for wl in wavelengths:
        wl_arr = np.full(N, wl)
        ang_arr = np.full(N, angle)
        eg = fmmax_solver.gradient_from_solver_batch(
            imgs_np, wl_arr, ang_arr,
            no_gradnorm=getattr(params, "no_gradnorm", False),
            periods_nm=period_nm, nn=fourier_nn,
        )
        all_effs.append(Tensor(eg[:, 0]))
        all_grads.append(Tensor(eg[:, 1:]).unsqueeze(1))  # (N, 1, H)
    return all_effs, all_grads


def global_loss_function_multi(gen_imgs, all_effs, all_grads,
                                sigma=0.5, binary_penalty=0, params=None):
    """Loss = mean over wavelengths of single-wavelength losses.

    Optionally weighted by params.wavelength_weights.
    """
    weights = getattr(params, "wavelength_weights", None)
    if weights is None:
        weights = [1.0] * len(all_effs)

    total_loss = 0.0
    for effs, grad, w in zip(all_effs, all_grads, weights):
        exp_w = torch.exp(effs / sigma)
        exp_w = exp_w / torch.mean(exp_w)
        eff_loss_tensor = (-gen_imgs * grad * (1.0 / sigma)
                           * exp_w.view(-1, 1, 1))
        eff_loss = torch.sum(torch.mean(eff_loss_tensor, dim=0).view(-1))
        total_loss = total_loss + float(w) * eff_loss

    total_loss = total_loss / sum(weights)

    # Binarisation loss (shared across wavelengths — only depends on gen_imgs)
    binary_loss = -torch.mean(torch.abs(gen_imgs.view(-1))
                              * (2.0 - torch.abs(gen_imgs.view(-1))))
    return total_loss + binary_loss * binary_penalty


# ---------------------------------------------------------------------------
# Multi-wavelength evaluation helpers
# ---------------------------------------------------------------------------
def compute_effs_multi(imgs, params):
    """Evaluate sign-binarised efficiency at every wavelength.

    Returns:
        effs_by_wl: list of Tensors (N, 1), one per wavelength
    """
    imgs_np = imgs.detach().cpu().numpy()
    if imgs_np.ndim == 3:
        imgs_np = imgs_np[:, 0, :]
    N = imgs_np.shape[0]
    angle = float(params.angle)
    period_nm = getattr(params, "period_nm", None)
    fourier_nn = int(getattr(params, "fourier_nn", fmmax_solver.DEFAULT_FOURIER_NN))
    effs_by_wl = []
    for wl in params.wavelengths:
        wl_arr = np.full(N, float(wl))
        ang_arr = np.full(N, angle)
        e = fmmax_solver.eval_eff_batch(
            imgs_np, wl_arr, ang_arr, periods_nm=period_nm, nn=fourier_nn,
        )
        effs_by_wl.append(Tensor(e[:, None]))
    return effs_by_wl


def evaluate_training_generator_multi(generator, eng, params, num_imgs=100):
    """Evaluate at all wavelengths and report mean + per-wavelength stats."""
    generator.eval()
    with torch.no_grad():
        z = sample_z(num_imgs, params)
        imgs_raw = generator(z, params)
        imgs_sign = torch.sign(imgs_raw)

        # Per-wavelength sign efficiency
        effs_sign_by_wl = compute_effs_multi(imgs_sign, params)

        # Overall mean (across patterns AND wavelengths) — the headline metric
        effs_sign_all = torch.cat([e.view(-1) for e in effs_sign_by_wl])
        effs_mean_sign = effs_sign_all.mean().item()

        # Per-wavelength mean (for logging)
        per_wl_means = [e.mean().item() for e in effs_sign_by_wl]
        per_wl_max = [e.max().item() for e in effs_sign_by_wl]

        # Track raw (continuous) efficiency across all wavelengths
        effs_raw_by_wl = compute_effs_multi(imgs_raw, params)
        effs_mean_raw = torch.cat([e.view(-1) for e in effs_raw_by_wl]).mean().item()

        binarization = torch.mean(torch.abs(imgs_raw.view(-1))).cpu().item()
        diversity = torch.mean(torch.std(imgs_raw, dim=0)).cpu().item()

        # Logging
        wl_str = "  ".join(
            f"wl{wl:.0f}={per_wl_means[i]:.4f}(max={per_wl_max[i]:.4f})"
            for i, wl in enumerate(params.wavelengths)
        )
        logging.info(
            f"iter {params.iter}: eff_raw(mean)={effs_mean_raw:.4f}  "
            f"eff_sign(mean)={effs_mean_sign:.4f}  bin={binarization:.3f}  |  {wl_str}"
        )

        # Plot histograms for ALL wavelengths (not just reference)
        for i, wl in enumerate(params.wavelengths):
            fig_path = os.path.join(params.output_dir, "figures", "histogram",
                                    f"Iter{params.iter}_wl{int(wl)}.png")
            utils.plot_histogram(effs_raw_by_wl[i].data.cpu().numpy().reshape(-1),
                                 params.iter, fig_path, title_suffix=f"{int(wl)} nm")

        # Save per-device efficiencies to .mat for later Origin plotting
        eval_dir = os.path.join(params.output_dir, "figures", "eval_data")
        os.makedirs(eval_dir, exist_ok=True)
        eval_mdict = {}
        for i, wl in enumerate(params.wavelengths):
            eval_mdict[f"effs_raw_wl{int(wl)}"] = \
                effs_raw_by_wl[i].data.cpu().numpy().reshape(-1)
            eval_mdict[f"effs_sign_wl{int(wl)}"] = \
                effs_sign_by_wl[i].data.cpu().numpy().reshape(-1)
        io.savemat(os.path.join(eval_dir, f"Iter{params.iter}.mat"), mdict=eval_mdict)

    generator.train()
    return effs_mean_raw, effs_mean_sign, binarization, diversity, per_wl_means


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_multi(generator, optimizer, scheduler, eng, params):
    generator.train()

    if params.restore_from is None:
        effs_mean_history = []
        effs_sign_history = []
        binarization_history = []
        diversity_history = []
        per_wl_history = []  # list of lists
        iter0 = 0
    else:
        ckpt = params.checkpoint
        effs_mean_history = ckpt["effs_mean_history"]
        effs_sign_history = ckpt.get("effs_sign_history", [])
        binarization_history = ckpt["binarization_history"]
        diversity_history = ckpt["diversity_history"]
        per_wl_history = ckpt.get("per_wl_history", [])
        iter0 = ckpt["iter"]

    with tqdm(total=params.numIter) as t:
        it = 0
        while True:
            it += 1
            params.iter = it + iter0

            normIter = params.iter / params.numIter

            params.batch_size = int(
                params.batch_size_start
                + (params.batch_size_end - params.batch_size_start)
                * (1.0 - (1.0 - normIter) ** params.batch_size_power)
            )

            params.sigma = params.sigma_start + (params.sigma_end - params.sigma_start) * normIter

            # Binary-amp schedule (identical to single-wavelength)
            amp_sched = getattr(params, 'binary_amp_schedule', 'two_phase')
            if amp_sched == 'stepwise':
                params.binary_amp = min(int(params.iter / 100) + 1, 10)
            elif amp_sched == 'linear':
                amp_max = float(getattr(params, 'binary_amp_max', 10))
                params.binary_amp = 1.0 + (amp_max - 1.0) * params.iter / params.numIter
            elif amp_sched == 'delayed_linear':
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

            # Periodic resumable checkpoints. The default interval is short
            # enough to limit recovery loss without materially affecting runtime.
            checkpoint_iter = int(getattr(params, "checkpoint_iter", 250))
            if it % checkpoint_iter == 0 or it > params.numIter:
                model_dir = os.path.join(params.output_dir, "model",
                                         f"iter{it + iter0}")
                os.makedirs(model_dir, exist_ok=True)
                utils.save_checkpoint(
                    {
                        "iter": it + iter0 - 1,
                        "gen_state_dict": generator.state_dict(),
                        "optim_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "effs_mean_history": effs_mean_history,
                        "effs_sign_history": effs_sign_history,
                        "binarization_history": binarization_history,
                        "diversity_history": diversity_history,
                        "per_wl_history": per_wl_history,
                    },
                    checkpoint=model_dir,
                )

            if it > params.numIter:
                return

            # --- One training step ---
            z = sample_z(params.batch_size, params)
            gen_imgs = generator(z, params)

            all_effs, all_grads = compute_effs_and_gradients_multi(gen_imgs, params)

            optimizer.zero_grad()

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
            g_loss = global_loss_function_multi(
                gen_imgs, all_effs, all_grads, params.sigma, binary_penalty, params
            )
            g_loss.backward()
            optimizer.step()
            scheduler.step()

            # --- Evaluation ---
            if it % params.plot_iter == 0:
                visualize_generated_images(generator, params)
                effs_mean, effs_sign, binarization, diversity, per_wl = \
                    evaluate_training_generator_multi(generator, eng, params)

                effs_mean_history.append(effs_mean)
                effs_sign_history.append(effs_sign)
                binarization_history.append(binarization)
                diversity_history.append(diversity)
                per_wl_history.append(per_wl)

                utils.plot_loss_history(
                    (effs_mean_history, diversity_history, binarization_history, effs_sign_history),
                    params,
                    per_wl_history=per_wl_history,
                    wavelengths=params.wavelengths,
                )

            t.update()


# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------
def _eval_imgs_multi_sub_batch(imgs_np, params, sub_batch=50):
    """Evaluate multi-wavelength efficiency in sub-batches to avoid GPU OOM."""
    N = imgs_np.shape[0]
    angle = float(params.angle)
    period_nm = getattr(params, "period_nm", None)
    fourier_nn = int(getattr(params, "fourier_nn", fmmax_solver.DEFAULT_FOURIER_NN))
    effs_by_wl = [[] for _ in range(len(params.wavelengths))]
    for s in range(0, N, sub_batch):
        e = min(s + sub_batch, N)
        for j, wl in enumerate(params.wavelengths):
            wl_arr = np.full(e - s, float(wl))
            ang_arr = np.full(e - s, angle)
            chunk = fmmax_solver.eval_eff_batch(
                imgs_np[s:e], wl_arr, ang_arr,
                periods_nm=period_nm, nn=fourier_nn,
            )
            effs_by_wl[j].append(chunk)
    return [np.concatenate(e) for e in effs_by_wl]


def evaluate_multi(generator, eng, numImgs, params):
    generator.eval()
    with torch.no_grad():
        z = sample_z(numImgs, params)
        images_raw = generator(z, params)
        logging.info("Generation done.\n")

        images = torch.sign(images_raw)

        imgs_raw_np = images_raw.detach().cpu().numpy()
        if imgs_raw_np.ndim == 3:
            imgs_raw_np = imgs_raw_np[:, 0, :]
        imgs_sign_np = np.sign(imgs_raw_np)

        effs_raw_arrays = _eval_imgs_multi_sub_batch(imgs_raw_np, params)
        effs_sign_arrays = _eval_imgs_multi_sub_batch(imgs_sign_np, params)

        effs_raw_by_wl = [Tensor(e[:, None]) for e in effs_raw_arrays]
        effs_sign_by_wl = [Tensor(e[:, None]) for e in effs_sign_arrays]

        # Overall mean across wavelengths
        eff_raw_all = torch.cat([e.view(-1) for e in effs_raw_by_wl])
        eff_sign_all = torch.cat([e.view(-1) for e in effs_sign_by_wl])

        logging.info(
            f"eff_mean raw={eff_raw_all.mean().item():.4f}  "
            f"sign={eff_sign_all.mean().item():.4f}  "
            f"drop={eff_raw_all.mean().item() - eff_sign_all.mean().item():+.4f}"
        )
        for i, wl in enumerate(params.wavelengths):
            er = effs_raw_by_wl[i].view(-1)
            es = effs_sign_by_wl[i].view(-1)
            logging.info(
                f"  wl={wl:.0f}: raw_mean={er.mean().item():.4f} max={er.max().item():.4f}  "
                f"sign_mean={es.mean().item():.4f} max={es.max().item():.4f}"
            )

        # Save as .mat (store sign images + all wavelength efficiencies)
        mdict = {
            "imgs": images.cpu().numpy(),
            "imgs_raw": images_raw.cpu().numpy(),
        }
        for i, wl in enumerate(params.wavelengths):
            mdict[f"effs_raw_wl{int(wl)}"] = effs_raw_by_wl[i].cpu().numpy()
            mdict[f"effs_sign_wl{int(wl)}"] = effs_sign_by_wl[i].cpu().numpy()

        filename = f"imgs_multi_wl_a{int(params.angle)}deg.mat"
        file_path = os.path.join(params.output_dir, "outputs", filename)
        io.savemat(file_path, mdict=mdict)

        # Histograms for ALL wavelengths
        for i, wl in enumerate(params.wavelengths):
            fig_path = os.path.join(params.output_dir, "figures",
                                    f"Efficiency_wl{int(wl)}.png")
            utils.plot_histogram(
                effs_sign_by_wl[i].data.cpu().numpy().reshape(-1),
                params.numIter, fig_path, title_suffix=f"{int(wl)} nm",
            )
