"""Matched fixed-period multi-wavelength direct pixel optimization.

This baseline optimizes independent 256-pixel variables with Adam.  It uses
the same fixed period, wavelengths, Gaussian smoothing, delayed-linear tanh
projection, material model, FMMAX formulation, and Fourier truncation as the
fixed-period GLOnet revision protocol.  It is a direct pixel-optimization
baseline, not a state-of-the-art density-based topology optimizer.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax
import scipy.io as io

import fmmax_solver


def gaussian_kernel(kernlen: int = 19, sigma: float = 6.0) -> jnp.ndarray:
    x = jnp.arange(float(kernlen))
    mean = (kernlen - 1) / 2.0
    kernel = jnp.exp(-((x - mean) ** 2) / (2.0 * sigma**2))
    return kernel / jnp.sum(kernel)


def periodic_filter(x: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    pad = len(kernel) - 1
    padded = jnp.concatenate([x[..., -pad // 2:], x, x[..., :pad // 2]], axis=-1)
    nx = x.shape[-1]
    result = jnp.zeros_like(x)
    for i in range(len(kernel)):
        result = result + kernel[i] * padded[..., i:i + nx]
    return result


def binary_amplitude(step: int, total_steps: int, amp_max: float = 10.0) -> float:
    delay_end = int(total_steps / 3)
    ramp_end = int(2 * total_steps / 3)
    if step < delay_end:
        return 1.0
    if step <= ramp_end:
        fraction = (step - delay_end) / max(ramp_end - delay_end, 1)
        return 1.0 + (amp_max - 1.0) * fraction
    return amp_max


class FixedPeriodObjective:
    def __init__(self, wavelengths: tuple[float, ...], period_nm: float, nn: int,
                 kernel_len: int = 19, kernel_sigma: float = 6.0):
        self.wavelengths = tuple(float(w) for w in wavelengths)
        self.period_nm = float(period_nm)
        self.nn = int(nn)
        self.kernel = gaussian_kernel(kernel_len, kernel_sigma)
        self.lattice, self.expansion, self.m0, self.m1 = fmmax_solver._expansion_for(
            self.period_nm, self.nn
        )

    def pixels(self, raw: jnp.ndarray, amplitude: float) -> jnp.ndarray:
        return jnp.tanh(periodic_filter(raw, self.kernel) * amplitude) * 1.05

    def efficiencies_from_pixels(self, pixels: jnp.ndarray) -> jnp.ndarray:
        values = []
        for wavelength in self.wavelengths:
            n_si = fmmax_solver.si_index(wavelength)
            img01 = jnp.clip(pixels / 2.0 + 0.5, 0.0, 1.0)
            nvec = img01 * (n_si - fmmax_solver.N_AIR) + fmmax_solver.N_AIR
            values.append(
                fmmax_solver._forward_efficiency(
                    nvec,
                    wavelength,
                    self.lattice,
                    self.expansion,
                    self.m0,
                    self.m1,
                    0.51 * wavelength,
                )
            )
        return jnp.stack(values)

    def mean_efficiency(self, raw: jnp.ndarray, amplitude: float) -> jnp.ndarray:
        return jnp.mean(self.efficiencies_from_pixels(self.pixels(raw, amplitude)))

    @functools.cached_property
    def value_and_grad_batch(self):
        fn = jax.value_and_grad(self.mean_efficiency)
        return jax.jit(jax.vmap(fn, in_axes=(0, None)))

    @functools.cached_property
    def sign_efficiencies_batch(self):
        def fn(raw):
            binary = jnp.sign(periodic_filter(raw, self.kernel))
            return self.efficiencies_from_pixels(binary)
        return jax.jit(jax.vmap(fn, in_axes=0))


def optimize_batch(objective: FixedPeriodObjective, raw_init: np.ndarray,
                   total_steps: int, learning_rate: float,
                   log_interval: int = 25) -> dict:
    raw = jnp.asarray(raw_init, dtype=jnp.float64)
    optimizer = optax.adam(learning_rate)
    state = jax.vmap(optimizer.init)(raw)

    def update_one(parameters, opt_state, gradient):
        updates, new_state = optimizer.update(gradient, opt_state, parameters)
        return optax.apply_updates(parameters, updates), new_state

    update_batch = jax.jit(jax.vmap(update_one))
    best_score = np.full(raw.shape[0], -np.inf)
    best_per_wavelength = np.zeros((raw.shape[0], len(objective.wavelengths)))
    best_patterns = np.zeros(np.asarray(raw).shape)
    best_checkpoint_step = np.full(raw.shape[0], -1, dtype=int)
    history_steps, history_mean = [], []
    checkpoint_efficiencies, checkpoint_patterns = [], []
    checkpoint_raw, checkpoint_continuous_patterns = [], []
    start = time.perf_counter()

    def capture(update_count: int) -> None:
        nonlocal best_score, best_per_wavelength, best_patterns, best_checkpoint_step
        amplitude = binary_amplitude(max(update_count - 1, 0), total_steps)
        per_wavelength = np.asarray(objective.sign_efficiencies_batch(raw))
        score = np.mean(per_wavelength, axis=1)
        raw_np = np.asarray(raw)
        patterns = np.sign(np.asarray(periodic_filter(raw, objective.kernel)))
        continuous = np.asarray(objective.pixels(raw, amplitude))
        improved = score > best_score
        best_score[improved] = score[improved]
        best_per_wavelength[improved] = per_wavelength[improved]
        best_patterns[improved] = patterns[improved]
        best_checkpoint_step[improved] = update_count
        history_steps.append(update_count)
        history_mean.append(float(np.mean(score)))
        checkpoint_efficiencies.append(per_wavelength)
        checkpoint_patterns.append(patterns)
        checkpoint_raw.append(raw_np)
        checkpoint_continuous_patterns.append(continuous)
        if update_count % (4 * log_interval) == 0 or update_count == total_steps:
            print(
                f"update {update_count:4d}/{total_steps}: sign mean={score.mean():.4f}, "
                f"max={score.max():.4f}, elapsed={time.perf_counter()-start:.1f}s",
                flush=True,
            )

    # Capture the true initialization, then exactly `total_steps` Adam updates.
    # This removes the legacy extra gradient evaluation and makes checkpoint
    # labels equal to the number of completed updates.
    capture(0)
    for update_count in range(1, total_steps + 1):
        amplitude = binary_amplitude(update_count - 1, total_steps)
        _, gradients = objective.value_and_grad_batch(raw, amplitude)
        raw, state = update_batch(raw, state, -gradients)
        if update_count % log_interval == 0 or update_count == total_steps:
            capture(update_count)

    elapsed_seconds = time.perf_counter() - start

    return {
        "score": best_score,
        "efficiencies": best_per_wavelength,
        "patterns": best_patterns,
        "best_checkpoint_step": best_checkpoint_step,
        "history_steps": np.asarray(history_steps),
        "history_mean": np.asarray(history_mean),
        "checkpoint_efficiencies": np.asarray(checkpoint_efficiencies),
        "checkpoint_patterns": np.asarray(checkpoint_patterns),
        "checkpoint_raw": np.asarray(checkpoint_raw),
        "checkpoint_continuous_patterns": np.asarray(checkpoint_continuous_patterns),
        "initial_raw": np.asarray(checkpoint_raw[0]),
        "final_raw": np.asarray(checkpoint_raw[-1]),
        "final_efficiencies": np.asarray(checkpoint_efficiencies[-1]),
        "final_patterns": np.asarray(checkpoint_patterns[-1]),
        "elapsed_seconds": np.asarray(elapsed_seconds),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavelengths", default="700,800,900")
    parser.add_argument("--period_nm", type=float, default=fmmax_solver.FIXED_PERIOD_NM)
    parser.add_argument("--fourier_nn", type=int, default=fmmax_solver.DEFAULT_FOURIER_NN)
    parser.add_argument("--n_seeds", type=int, default=500)
    parser.add_argument("--n_iter", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--output_dir", default="results/revision_fixed/direct_multi")
    args = parser.parse_args()

    wavelengths = tuple(float(x) for x in args.wavelengths.split(","))
    objective = FixedPeriodObjective(wavelengths, args.period_nm, args.fourier_nn)
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_steps_planned = [0] + list(range(25, args.n_iter + 1, 25))
    if checkpoint_steps_planned[-1] != args.n_iter:
        checkpoint_steps_planned.append(args.n_iter)
    gradient_solve_count = args.n_seeds * args.n_iter * len(wavelengths)
    checkpoint_solve_count = args.n_seeds * len(checkpoint_steps_planned) * len(wavelengths)
    metadata = vars(args) | {
        "wavelengths": list(wavelengths),
        "output_angles_deg": [
            fmmax_solver.diffraction_angle_deg(w, args.period_nm) for w in wavelengths
        ],
        "schema_version": 2,
        "gradient_wavelength_solver_evaluations": gradient_solve_count,
        "checkpoint_wavelength_solver_evaluations": checkpoint_solve_count,
        "total_wavelength_solver_evaluations": gradient_solve_count + checkpoint_solve_count,
        "checkpoint_steps": checkpoint_steps_planned,
        "baseline_type": "Adam-based direct pixel optimization",
    }
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    batch_dir = os.path.join(args.output_dir, "batches")
    os.makedirs(batch_dir, exist_ok=True)
    all_results = []
    for start in range(0, args.n_seeds, args.batch_size):
        count = min(args.batch_size, args.n_seeds - start)
        stop = start + count
        batch_path = os.path.join(batch_dir, f"batch_{start:06d}_{stop - 1:06d}.npz")
        if os.path.isfile(batch_path):
            with np.load(batch_path) as saved:
                result = {key: saved[key] for key in saved.files}
            if len(result["score"]) != count:
                raise ValueError(f"Unexpected saved batch size in {batch_path}")
            required = {
                "checkpoint_efficiencies", "checkpoint_patterns",
                "checkpoint_raw", "checkpoint_continuous_patterns",
                "initial_raw", "final_raw", "final_efficiencies", "final_patterns",
                "best_checkpoint_step", "elapsed_seconds",
            }
            missing = sorted(required.difference(result))
            if missing:
                raise ValueError(
                    f"Saved batch {batch_path} predates trajectory capture and is "
                    f"missing {missing}. Use a new --output_dir to preserve old data."
                )
            print(f"batch seeds {start}-{stop - 1}: loaded saved result")
            all_results.append(result)
            continue
        rng = np.random.default_rng(start)
        raw_init = rng.uniform(-2.0, 2.0, (count, 256))
        print(f"batch seeds {start}-{stop - 1}")
        result = optimize_batch(objective, raw_init, args.n_iter, args.learning_rate)
        temp_batch_path = batch_path + f".tmp.{os.getpid()}.npz"
        np.savez_compressed(temp_batch_path, **result)
        os.replace(temp_batch_path, batch_path)
        all_results.append(result)

    scores = np.concatenate([r["score"] for r in all_results])
    efficiencies = np.concatenate([r["efficiencies"] for r in all_results])
    patterns = np.concatenate([r["patterns"] for r in all_results])
    final_efficiencies = np.concatenate([r["final_efficiencies"] for r in all_results])
    final_patterns = np.concatenate([r["final_patterns"] for r in all_results])
    best_checkpoint_step = np.concatenate([r["best_checkpoint_step"] for r in all_results])
    checkpoint_efficiencies = np.concatenate(
        [r["checkpoint_efficiencies"] for r in all_results], axis=1
    )
    checkpoint_patterns = np.concatenate(
        [r["checkpoint_patterns"] for r in all_results], axis=1
    )
    checkpoint_raw = np.concatenate([r["checkpoint_raw"] for r in all_results], axis=1)
    checkpoint_continuous_patterns = np.concatenate(
        [r["checkpoint_continuous_patterns"] for r in all_results], axis=1
    )
    initial_raw = np.concatenate([r["initial_raw"] for r in all_results])
    final_raw = np.concatenate([r["final_raw"] for r in all_results])
    checkpoint_steps = np.asarray(all_results[0]["history_steps"])
    bottleneck = np.min(efficiencies, axis=1)
    results_path = os.path.join(args.output_dir, "direct_multi_results.mat")
    temp_results_path = results_path + f".tmp.{os.getpid()}"
    io.savemat(temp_results_path, {
        "patterns": patterns,
        "efficiencies": efficiencies,
        "best_checkpoint_step": best_checkpoint_step,
        "final_patterns": final_patterns,
        "final_efficiencies": final_efficiencies,
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_patterns": checkpoint_patterns,
        "checkpoint_efficiencies": checkpoint_efficiencies,
        "checkpoint_raw": checkpoint_raw,
        "checkpoint_continuous_patterns": checkpoint_continuous_patterns,
        "initial_raw": initial_raw,
        "final_raw": final_raw,
        "overall_mean_per_device": scores,
        "bottleneck_per_device": bottleneck,
        "wavelengths_nm": np.asarray(wavelengths),
        "period_nm": args.period_nm,
        "fourier_nn": args.fourier_nn,
        "seeds": np.arange(args.n_seeds),
    }, appendmat=False)
    os.replace(temp_results_path, results_path)
    summary = {
        "n": int(args.n_seeds),
        "mean": float(np.mean(scores)),
        "sample_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "median": float(np.median(scores)),
        "maximum": float(np.max(scores)),
        "mean_bottleneck": float(np.mean(bottleneck)),
        "per_wavelength_mean": np.mean(efficiencies, axis=0).tolist(),
        "per_wavelength_max": np.max(efficiencies, axis=0).tolist(),
        "selection_policy": "best sign-binary checkpoint per trajectory",
        "schema_version": 2,
        "checkpoint_semantics": "true initialization at 0, then completed Adam updates",
        "checkpoint_steps": checkpoint_steps.tolist(),
        "batch_elapsed_seconds": [float(r["elapsed_seconds"]) for r in all_results],
        "total_batch_elapsed_seconds": float(sum(float(r["elapsed_seconds"]) for r in all_results)),
        "final_iterate": {
            "mean": float(np.mean(final_efficiencies.mean(axis=1))),
            "sample_std": float(np.std(final_efficiencies.mean(axis=1), ddof=1)),
            "median": float(np.median(final_efficiencies.mean(axis=1))),
            "maximum": float(np.max(final_efficiencies.mean(axis=1))),
            "mean_bottleneck": float(np.mean(final_efficiencies.min(axis=1))),
            "maximum_bottleneck": float(np.max(final_efficiencies.min(axis=1))),
        },
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
