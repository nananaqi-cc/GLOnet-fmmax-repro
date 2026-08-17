"""Entry point for multi-wavelength GLOnet training."""
import os
import logging
import random
import argparse
import numpy as np
import torch

from net import Generator
import utils
from train_multi_wl import train_multi, evaluate_multi
import fmmax_solver


parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default="results/multi_wl_v1",
                    help="Results folder")
parser.add_argument("--wavelengths", default="700,800,900",
                    help="Comma-separated wavelengths in nm")
parser.add_argument("--angle", default=None, type=float)
parser.add_argument("--restore_from", default=None)
parser.add_argument("--wl_weights", default=None,
                    help="Comma-separated wavelength weights (e.g. '3,1,1')")
parser.add_argument("--seed", default=None, type=int,
                    help="Random seed (auto-generated if not given)")
parser.add_argument("--params_file", default=None,
                    help="Parameter JSON template; copied into output_dir as Params.json")
parser.add_argument("--period_nm", default=None, type=float,
                    help="Fixed physical period in nm; omit for legacy wavelength-scaled periods")
parser.add_argument("--fourier_nn", default=None, type=int,
                    help="Explicit Fourier truncation parameter")
parser.add_argument("--num_iter", default=None, type=int,
                    help="Override training iterations (useful for smoke tests)")
parser.add_argument("--batch_size", default=None, type=int,
                    help="Override constant training batch size")
parser.add_argument("--num_eval", default=500, type=int,
                    help="Number of generated patterns in final evaluation")


if __name__ == "__main__":
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

    # Load params from json
    json_path = args.params_file or os.path.join(args.output_dir, "Params.json")
    assert os.path.isfile(json_path), f"No json file found at {json_path}"
    params = utils.Params(json_path)

    params.output_dir = args.output_dir
    params.seed = seed
    params.cuda = torch.cuda.is_available()
    params.restore_from = args.restore_from
    params.numIter = int(params.numIter)
    params.noise_dims = int(params.noise_dims)
    params.gkernlen = int(params.gkernlen)
    params.step_size = int(params.step_size)

    # Multi-wavelength specific
    params.wavelengths = [float(x.strip()) for x in args.wavelengths.split(",")]
    # Reference wavelength (for histograms, backward compat)
    params.wavelength = params.wavelengths[0]
    if args.angle is not None:
        params.angle = float(args.angle)
    if args.period_nm is not None:
        params.period_nm = float(args.period_nm)
    if args.fourier_nn is not None:
        params.fourier_nn = int(args.fourier_nn)
    if args.num_iter is not None:
        params.numIter = int(args.num_iter)
    if args.batch_size is not None:
        params.batch_size_start = int(args.batch_size)
        params.batch_size_end = int(args.batch_size)

    # Default: equal weights
    if args.wl_weights is not None:
        params.wavelength_weights = [float(x.strip()) for x in args.wl_weights.split(",")]
    elif not hasattr(params, "wavelength_weights"):
        params.wavelength_weights = [1.0] * len(params.wavelengths)

    # Set logger
    utils.set_logger(os.path.join(args.output_dir, "train.log"))
    logging.info(f"Multi-wavelength training: {params.wavelengths} nm  "
                 f"angle={params.angle}°  seed={params.seed}")
    logging.info(f"Wavelength weights: {params.wavelength_weights}")
    if hasattr(params, "period_nm"):
        output_angles = [
            fmmax_solver.diffraction_angle_deg(w, params.period_nm)
            for w in params.wavelengths
        ]
        logging.info(
            f"Fixed period: {params.period_nm:.6f} nm; "
            f"+1 output angles: {[round(a, 4) for a in output_angles]}"
        )
    logging.info(f"Fourier nn: {getattr(params, 'fourier_nn', 'default')}")
    params.save(os.path.join(args.output_dir, "Params.json"))

    # Create directories
    for sub in ["outputs", "model", "figures/histogram", "figures/deviceSamples",
                "figures/eval_data"]:
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    # Build model
    generator = Generator(params)
    if params.cuda:
        generator.cuda()

    optimizer = torch.optim.Adam(
        generator.parameters(), lr=params.lr,
        betas=(params.beta1, params.beta2),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=params.step_size, gamma=params.gamma,
    )

    # Restore checkpoint if requested
    if args.restore_from is not None:
        params.checkpoint = utils.load_checkpoint(
            args.restore_from, generator, optimizer, scheduler,
        )
        logging.info("Model data loaded from checkpoint")

    # Train
    if params.numIter != 0:
        logging.info("Start multi-wavelength training")
        train_multi(generator, optimizer, scheduler, None, params)

    # Final evaluation
    logging.info("Start generating devices")
    evaluate_multi(generator, None, numImgs=args.num_eval, params=params)
