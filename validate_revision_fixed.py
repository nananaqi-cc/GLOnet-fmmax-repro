"""Numerical validation for the corrected fixed-period transmission solver."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat

import fmmax_solver


def representative_patterns(count: int = 5) -> np.ndarray:
    data = loadmat(
        "results/multi_w700_w800_w900_a60_original_v1_seed5/outputs/"
        "imgs_multi_wl_a60deg.mat"
    )
    patterns = np.sign(data["imgs"][:, 0, :])
    efficiencies = np.column_stack([
        data[f"effs_sign_wl{w}"].reshape(-1) for w in (700, 800, 900)
    ])
    objectives = [
        np.mean(efficiencies, axis=1),
        np.min(efficiencies, axis=1),
        efficiencies[:, 0], efficiencies[:, 1], efficiencies[:, 2],
    ]
    selected = []
    for values in objectives:
        for index in np.argsort(values)[::-1]:
            if int(index) not in selected:
                selected.append(int(index))
                break
    return patterns[selected[:count]]


def full_fd_validation(patterns: np.ndarray, wavelength_nm: float,
                       period_nm: float, nn: int, h: float,
                       batch_size: int) -> tuple[list[dict], np.ndarray, np.ndarray]:
    n_si = fmmax_solver.si_index(wavelength_nm)
    value_grad = fmmax_solver._eff_grad_jit(
        wavelength_nm, period_nm, nn, 0.51 * wavelength_nm
    )
    forward_batch = fmmax_solver._eff_batch_jit(
        wavelength_nm, period_nm, nn, 0.51 * wavelength_nm
    )
    all_ad, all_fd, records = [], [], []
    for device_index, pattern in enumerate(patterns):
        nvec = np.where(pattern > 0, n_si, fmmax_solver.N_AIR).astype(np.float64)
        efficiency, gradient = value_grad(jnp.asarray(nvec))
        gradient = np.asarray(gradient)
        perturbed = []
        for pixel in range(len(nvec)):
            plus, minus = nvec.copy(), nvec.copy()
            plus[pixel] += h
            minus[pixel] -= h
            perturbed.extend((plus, minus))
        perturbed = np.asarray(perturbed)
        values = []
        for start in range(0, len(perturbed), batch_size):
            values.append(np.asarray(forward_batch(jnp.asarray(perturbed[start:start+batch_size]))))
        values = np.concatenate(values)
        finite_difference = (values[0::2] - values[1::2]) / (2.0 * h)
        error = np.abs(gradient - finite_difference)
        denominator = np.linalg.norm(finite_difference)
        correlation = float(np.corrcoef(gradient, finite_difference)[0, 1])
        record = {
            "device": device_index,
            "efficiency": float(efficiency),
            "mae": float(np.mean(error)),
            "median_ae": float(np.median(error)),
            "p95_ae": float(np.percentile(error, 95)),
            "max_ae": float(np.max(error)),
            "relative_l2": float(np.linalg.norm(gradient-finite_difference) / denominator),
            "pearson_r": correlation,
        }
        print("gradient", record)
        records.append(record)
        all_ad.append(gradient)
        all_fd.append(finite_difference)
    return records, np.asarray(all_ad), np.asarray(all_fd)


def convergence(patterns: np.ndarray, wavelengths: tuple[float, ...],
                period_nm: float, nn_values: tuple[int, ...]) -> np.ndarray:
    values = np.zeros((len(patterns), len(wavelengths), len(nn_values)))
    for wi, wavelength in enumerate(wavelengths):
        for ni, nn in enumerate(nn_values):
            values[:, wi, ni] = fmmax_solver.eval_eff_batch(
                patterns, np.full(len(patterns), wavelength),
                periods_nm=period_nm, nn=nn,
            )
            print("convergence", wavelength, nn, values[:, wi, ni])
    return values


def make_figure(ad: np.ndarray, fd: np.ndarray, conv: np.ndarray,
                nn_values: tuple[int, ...], output: Path) -> None:
    error = np.abs(ad-fd).reshape(-1)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))
    axes[0].scatter(fd.reshape(-1), ad.reshape(-1), s=4, alpha=0.35)
    limits = [min(fd.min(), ad.min()), max(fd.max(), ad.max())]
    axes[0].plot(limits, limits, "k--", linewidth=1)
    axes[0].set_xlabel("finite-difference gradient")
    axes[0].set_ylabel("AD gradient")
    axes[0].set_title("All 256 pixels x 5 devices")
    axes[1].hist(error, bins=40)
    axes[1].set_xlabel("absolute gradient error")
    axes[1].set_ylabel("count")
    axes[1].set_yscale("log")
    reference = conv[:, :, -1, None]
    delta = np.abs(conv-reference)
    for wi, wavelength in enumerate((700, 800, 900)):
        max_delta = np.max(delta[:, wi, :], axis=0)
        axes[2].plot(nn_values, np.maximum(max_delta, 1e-12), "o-", label=f"{wavelength} nm")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Fourier truncation nn")
    axes[2].set_ylabel("max |efficiency - nn=80 reference|")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period_nm", type=float, default=fmmax_solver.FIXED_PERIOD_NM)
    parser.add_argument("--fourier_nn", type=int, default=fmmax_solver.DEFAULT_FOURIER_NN)
    parser.add_argument("--fd_h", type=float, default=1e-4)
    parser.add_argument("--fd_batch_size", type=int, default=16)
    parser.add_argument("--output_dir", default="results/revision_fixed/validation")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    patterns = representative_patterns(5)
    records, ad, fd = full_fd_validation(
        patterns, 800.0, args.period_nm, args.fourier_nn, args.fd_h, args.fd_batch_size
    )
    nn_values = (10, 14, 20, 30, 40, 60, 80)
    conv = convergence(patterns, (700.0, 800.0, 900.0), args.period_nm, nn_values)
    make_figure(ad, fd, conv, nn_values, output / "numerical_validation.png")
    summary = {
        "solver_output": "transmitted +1 order via FMMAX s11",
        "period_nm": args.period_nm,
        "fourier_nn_gradient": args.fourier_nn,
        "fd_step": args.fd_h,
        "devices": records,
        "aggregate": {
            "mae": float(np.mean(np.abs(ad-fd))),
            "median_ae": float(np.median(np.abs(ad-fd))),
            "p95_ae": float(np.percentile(np.abs(ad-fd), 95)),
            "max_ae": float(np.max(np.abs(ad-fd))),
            "pearson_r": float(np.corrcoef(ad.reshape(-1), fd.reshape(-1))[0, 1]),
        },
        "convergence_nn": list(nn_values),
        "max_abs_delta_vs_nn80": np.max(np.abs(conv-conv[:, :, -1, None]), axis=0).tolist(),
    }
    with open(output / "validation_summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    savemat(output / "validation_data.mat", {
        "patterns": patterns, "gradient_ad": ad, "gradient_fd": fd,
        "convergence_efficiencies": conv, "convergence_nn": np.asarray(nn_values),
        "period_nm": args.period_nm,
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
