"""Evaluate saved GLOnet checkpoints on fixed, per-seed noise banks.

The production output files use the random state left after training.  This
script instead applies one deterministic 500-vector noise bank to every saved
checkpoint of a seed, so checkpoint differences are not confounded by a new
Monte Carlo sample.  It does not retrain or alter any checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from net import Generator
import fmmax_solver
import utils


WAVELENGTHS = np.asarray([700.0, 800.0, 900.0])


def binary_amplitude(iteration: int, params: utils.Params) -> float:
    schedule = getattr(params, "binary_amp_schedule", "delayed_linear")
    if schedule != "delayed_linear":
        raise ValueError(f"This audit expects delayed_linear, found {schedule!r}")
    start = int(getattr(params, "binary_step_iter", 500))
    end = int(getattr(params, "binary_amp_ramp_end", params.numIter))
    maximum = float(getattr(params, "binary_amp_max", 10.0))
    if iteration < start:
        return 1.0
    if iteration <= end:
        fraction = (iteration - start) / max(end - start, 1)
        return 1.0 + (maximum - 1.0) * fraction
    return maximum


def checkpoint_paths(seed_dir: Path) -> list[Path]:
    paths = sorted(
        (seed_dir / "model").glob("iter*/model.pth.tar"),
        key=lambda path: int(path.parent.name.removeprefix("iter")),
    )
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {seed_dir / 'model'}")
    return paths


def generate_patterns(seed_dir: Path, seed: int, n_candidates: int):
    params = utils.Params(str(seed_dir / "Params.json"))
    params.cuda = torch.cuda.is_available()
    params.wavelengths = WAVELENGTHS.tolist()
    params.numIter = int(params.numIter)
    params.noise_dims = int(params.noise_dims)
    params.gkernlen = int(params.gkernlen)

    rng = np.random.default_rng(100_000 + seed)
    noise_np = rng.uniform(
        -float(params.noise_amplitude),
        float(params.noise_amplitude),
        size=(n_candidates, params.noise_dims),
    ).astype(np.float32)
    device = torch.device("cuda" if params.cuda else "cpu")
    noise = torch.from_numpy(noise_np).to(device)

    patterns, steps, source_dirs = [], [], []
    model = Generator(params).to(device).eval()
    for path in checkpoint_paths(seed_dir):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["gen_state_dict"])
        iteration = int(checkpoint["iter"])
        params.binary_amp = binary_amplitude(iteration, params)
        with torch.no_grad():
            images = model(noise, params)
        pattern = np.sign(images[:, 0, :].cpu().numpy())
        patterns.append(pattern)
        steps.append(iteration)
        source_dirs.append(path.parent.name)
    del model, noise
    if params.cuda:
        torch.cuda.empty_cache()
    return np.asarray(patterns), np.asarray(steps), source_dirs, params


def evaluate_patterns(patterns: np.ndarray, params: utils.Params,
                      sub_batch: int) -> np.ndarray:
    n_checkpoints, n_candidates, _ = patterns.shape
    values = np.zeros((n_checkpoints, n_candidates, len(WAVELENGTHS)))
    for checkpoint in range(n_checkpoints):
        for start in range(0, n_candidates, sub_batch):
            stop = min(start + sub_batch, n_candidates)
            for j, wavelength in enumerate(WAVELENGTHS):
                values[checkpoint, start:stop, j] = fmmax_solver.eval_eff_batch(
                    patterns[checkpoint, start:stop],
                    np.full(stop - start, wavelength),
                    periods_nm=float(params.period_nm),
                    nn=int(params.fourier_nn),
                )
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/revision_fixed"))
    parser.add_argument("--output_dir", type=Path,
                        default=Path("results/revision_fixed/glonet_checkpoint_audit"))
    parser.add_argument("--n_candidates", type=int, default=500)
    parser.add_argument("--sub_batch", type=int, default=25)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "noise_policy": "fixed NumPy noise bank per seed, RNG seed 100000 + training seed",
        "n_candidates": args.n_candidates,
        "seeds": [],
    }
    for seed in range(1, 6):
        seed_dir = args.root / f"multi_seed{seed}"
        patterns, steps, source_dirs, params = generate_patterns(
            seed_dir, seed, args.n_candidates
        )
        efficiencies = evaluate_patterns(patterns, params, args.sub_batch)
        candidate_means = efficiencies.mean(axis=2)
        checkpoint_means = candidate_means.mean(axis=1)
        best_index = int(np.argmax(checkpoint_means))
        final_index = int(np.argmax(steps))
        output = args.output_dir / f"seed{seed}_checkpoints.npz"
        np.savez_compressed(
            output,
            steps=steps,
            source_dirs=np.asarray(source_dirs),
            patterns=patterns,
            efficiencies=efficiencies,
            checkpoint_candidate_mean=checkpoint_means,
            best_checkpoint_index=best_index,
            final_checkpoint_index=final_index,
        )
        summary["seeds"].append({
            "seed": seed,
            "available_steps": steps.tolist(),
            "source_dirs": source_dirs,
            "checkpoint_candidate_means": checkpoint_means.tolist(),
            "best_step": int(steps[best_index]),
            "best_mean": float(checkpoint_means[best_index]),
            "best_candidate": float(candidate_means[best_index].max()),
            "final_step": int(steps[final_index]),
            "final_mean": float(checkpoint_means[final_index]),
            "final_best_candidate": float(candidate_means[final_index].max()),
        })
        print(json.dumps(summary["seeds"][-1], indent=2))

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
