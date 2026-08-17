import os
import logging
import random
import argparse
import numpy as np
import torch

from train_and_evaluate import evaluate, train
from net import Generator
import utils


# parser
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='results',
                                        help="Results folder")
parser.add_argument('--wavelength', default=None)
parser.add_argument('--angle', default=None)
parser.add_argument('--restore_from', default=None,
                                        help="Optional, directory or file containing weights to reload before training")
parser.add_argument('--seed', default=None, type=int,
                    help="Random seed (auto-generated if not given)")
parser.add_argument('--params_file', default=None,
                    help="Parameter JSON template; copied into output_dir as Params.json")
parser.add_argument('--period_nm', default=None, type=float,
                    help="Fixed physical period in nm")
parser.add_argument('--fourier_nn', default=None, type=int,
                    help="Explicit Fourier truncation parameter")
parser.add_argument('--num_iter', default=None, type=int,
                    help="Override training iterations (useful for smoke tests)")
parser.add_argument('--batch_size', default=None, type=int,
                    help="Override constant training batch size")
parser.add_argument('--num_eval', default=500, type=int,
                    help="Number of generated patterns in final evaluation")


if __name__ == '__main__':
    # Load the directory from commend line
    args = parser.parse_args()

    # Seed control: determinism for reproducibility
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Set the logger
    utils.set_logger(os.path.join(args.output_dir, 'train.log'))

    # Load parameters from json file
    json_path = args.params_file or os.path.join(args.output_dir,'Params.json')
    assert os.path.isfile(json_path), "No json file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Add attributes to params
    params.output_dir = args.output_dir
    params.seed = seed
    params.cuda = torch.cuda.is_available()
    params.restore_from = args.restore_from
    params.numIter = int(params.numIter)
    params.noise_dims = int(params.noise_dims)
    params.gkernlen = int(params.gkernlen)
    params.step_size = int(params.step_size)    

    if args.wavelength is not None:
        params.wavelength = int(args.wavelength)
    if args.angle is not None:
        params.angle = int(args.angle)
    if args.period_nm is not None:
        params.period_nm = float(args.period_nm)
    if args.fourier_nn is not None:
        params.fourier_nn = int(args.fourier_nn)
    if args.num_iter is not None:
        params.numIter = int(args.num_iter)
    if args.batch_size is not None:
        params.batch_size_start = int(args.batch_size)
        params.batch_size_end = int(args.batch_size)

    params.save(os.path.join(args.output_dir, 'Params.json'))


    # make directory
    os.makedirs(args.output_dir + '/outputs', exist_ok = True)
    os.makedirs(args.output_dir + '/model', exist_ok = True)
    os.makedirs(args.output_dir + '/figures/histogram', exist_ok = True)
    os.makedirs(args.output_dir + '/figures/deviceSamples', exist_ok = True)

    # Define the models 
    generator = Generator(params)
        
    # Move to gpu if possible
    if params.cuda:
        generator.cuda()


    # Define the optimizer
    optimizer = torch.optim.Adam(generator.parameters(), lr=params.lr, betas=(params.beta1, params.beta2))
    
    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.step_size, gamma = params.gamma)


    # Load model data
    if args.restore_from is not None :
        params.checkpoint = utils.load_checkpoint(restore_from, generator, optimizer, scheduler)
        logging.info('Model data loaded')

    
    # Train the model and save
    if params.numIter != 0 :
        logging.info(f"Start training  wavelength={params.wavelength}nm  "
                     f"angle={params.angle}°  seed={params.seed}")
        train(generator, optimizer, scheduler, None, params)

    # Generate images and save
    logging.info('Start generating devices')
    evaluate(generator, None, numImgs=args.num_eval, params=params)

