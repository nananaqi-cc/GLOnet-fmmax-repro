"""Preliminary fixed-period re-evaluation, fabricability, and Fourier analysis.

The script analyzes already generated binary patterns.  These results are
diagnostic and do not replace fixed-period retraining, because the source
generators were trained with wavelength-scaled periods.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat

import fmmax_solver


WAVELENGTHS = np.asarray([700.0, 800.0, 900.0])


def periodic_runs(pattern: np.ndarray) -> list[tuple[int, int]]:
    """Return (material sign, run length) with first/last periodic runs merged."""
    x = np.sign(np.asarray(pattern).reshape(-1)).astype(int)
    starts = np.r_[0, np.flatnonzero(x[1:] != x[:-1]) + 1]
    runs = [(int(x[s]), int(e - s)) for s, e in zip(starts, np.r_[starts[1:], len(x)])]
    if len(runs) > 1 and runs[0][0] == runs[-1][0]:
        runs[0] = (runs[0][0], runs[0][1] + runs[-1][1])
        runs.pop()
    return runs


def structure_metrics(pattern: np.ndarray, period_nm: float) -> dict:
    x = np.sign(np.asarray(pattern).reshape(-1))
    runs = periodic_runs(x)
    pixel_nm = float(period_nm) / len(x)
    si_runs = [length for sign, length in runs if sign > 0]
    air_runs = [length for sign, length in runs if sign < 0]
    min_si_px = min(si_runs) if si_runs else 0
    min_air_px = min(air_runs) if air_runs else 0
    min_feature_px = min(v for v in (min_si_px, min_air_px) if v > 0)
    return {
        "pixel_nm": pixel_nm,
        "duty_cycle_si": float(np.mean(x > 0)),
        "transitions_periodic": len(runs),
        "min_si_px": min_si_px,
        "min_air_px": min_air_px,
        "min_feature_px": min_feature_px,
        "min_si_nm": min_si_px * pixel_nm,
        "min_air_nm": min_air_px * pixel_nm,
        "min_feature_nm": min_feature_px * pixel_nm,
        "max_aspect_ratio_325nm": 325.0 / (min_feature_px * pixel_nm),
    }


def morphology_1d(pattern: np.ndarray, radius: int, dilate_si: bool) -> np.ndarray:
    x = np.asarray(pattern).reshape(-1) > 0
    shifted = [np.roll(x, shift) for shift in range(-radius, radius + 1)]
    result = np.logical_or.reduce(shifted) if dilate_si else np.logical_and.reduce(shifted)
    return np.where(result, 1.0, -1.0)


def evaluate(patterns: np.ndarray, period_nm: float, nn: int) -> np.ndarray:
    output = np.zeros((len(patterns), len(WAVELENGTHS)))
    for j, wavelength in enumerate(WAVELENGTHS):
        output[:, j] = fmmax_solver.eval_eff_batch(
            patterns,
            np.full(len(patterns), wavelength),
            periods_nm=period_nm,
            nn=nn,
        )
    return output


def load_multi_candidates() -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    path = (
        Path("results/multi_w700_w800_w900_a60_original_v1_seed5/outputs")
        / "imgs_multi_wl_a60deg.mat"
    )
    data = loadmat(path)
    patterns = np.sign(data["imgs"][:, 0, :])
    efficiencies = np.column_stack([
        data[f"effs_sign_wl{int(w)}"].reshape(-1) for w in WAVELENGTHS
    ])
    objectives = [
        ("best_overall_mean", np.mean(efficiencies, axis=1)),
        ("best_bottleneck", np.min(efficiencies, axis=1)),
        ("best_700nm", efficiencies[:, 0]),
        ("best_800nm", efficiencies[:, 1]),
        ("best_900nm", efficiencies[:, 2]),
    ]
    selected, labels = [], []
    used = set()
    for label, values in objectives:
        for index in np.argsort(values)[::-1]:
            if int(index) not in used:
                used.add(int(index))
                selected.append(int(index))
                labels.append(label)
                break
    return patterns[selected], efficiencies[selected], labels, selected


def load_fourier_comparison(multi_pattern: np.ndarray) -> tuple[np.ndarray, list[str]]:
    paths = [
        ("single 700 nm", "results/single_w700_a60_original_v3_seed4/outputs/imgs_w700_a60deg.mat"),
        ("single 800 nm", "results/single_w800_a60_original_v3_seed4/outputs/imgs_w800_a60deg.mat"),
        ("single 900 nm", "results/single_w900_a60_original_v3_seed3/outputs/imgs_w900_a60deg.mat"),
    ]
    patterns, labels = [], []
    for label, path in paths:
        data = loadmat(path)
        index = int(np.argmax(data["effs"].reshape(-1)))
        patterns.append(np.sign(data["imgs"][index, 0, :]))
        labels.append(label)
    patterns.append(np.sign(multi_pattern))
    labels.append("multi seed 5, best overall")
    return np.asarray(patterns), labels


def save_fourier_figure(patterns: np.ndarray, labels: list[str], output: Path) -> None:
    fig, axes = plt.subplots(len(patterns), 2, figsize=(8.0, 1.75 * len(patterns)),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    for row, (pattern, label) in enumerate(zip(patterns, labels)):
        axes[row, 0].step(np.arange(len(pattern)), pattern, where="mid", linewidth=0.8)
        axes[row, 0].set_ylim(-1.25, 1.25)
        axes[row, 0].set_ylabel(label, fontsize=8)
        spectrum = np.abs(np.fft.rfft(pattern) / len(pattern)) ** 2
        axes[row, 1].stem(np.arange(1, min(41, len(spectrum))), spectrum[1:41],
                          linefmt="C1-", markerfmt="C1.", basefmt=" ")
        axes[row, 1].set_yscale("log")
        axes[row, 1].set_ylim(1e-6, max(1e-5, spectrum[1:41].max() * 2))
    axes[-1, 0].set_xlabel("pixel index")
    axes[-1, 1].set_xlabel("spatial harmonic |m|")
    axes[0, 0].set_title("Binary pattern")
    axes[0, 1].set_title("Fourier power")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period_nm", type=float, default=fmmax_solver.FIXED_PERIOD_NM)
    parser.add_argument("--fourier_nn", type=int, default=fmmax_solver.DEFAULT_FOURIER_NN)
    parser.add_argument("--output_dir", default="results/revision_fixed/diagnostics")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    patterns, old_efficiencies, labels, indices = load_multi_candidates()
    fixed_efficiencies = evaluate(patterns, args.period_nm, args.fourier_nn)
    records = []
    for i, label in enumerate(labels):
        record = {
            "label": label,
            "source_index": indices[i],
            **structure_metrics(patterns[i], args.period_nm),
            "old_scaled_period_efficiencies": old_efficiencies[i].tolist(),
            "fixed_period_efficiencies": fixed_efficiencies[i].tolist(),
            "fixed_period_mean": float(np.mean(fixed_efficiencies[i])),
            "fixed_period_bottleneck": float(np.min(fixed_efficiencies[i])),
        }
        tolerance = {}
        for radius in (1, 2, 3):
            for mode, dilate in (("si_dilation", True), ("si_erosion", False)):
                modified = morphology_1d(patterns[i], radius, dilate)
                tolerance[f"{mode}_{radius}px"] = evaluate(
                    modified[None, :], args.period_nm, args.fourier_nn
                )[0].tolist()
        record["tolerance_efficiencies"] = tolerance
        records.append(record)

    fourier_patterns, fourier_labels = load_fourier_comparison(patterns[0])
    save_fourier_figure(fourier_patterns, fourier_labels, output / "fourier_comparison.png")
    with open(output / "candidate_metrics.json", "w", encoding="utf-8") as stream:
        json.dump({
            "diagnostic_only": True,
            "note": "Source generators used wavelength-scaled periods; fixed-period retraining is required.",
            "period_nm": args.period_nm,
            "fourier_nn": args.fourier_nn,
            "output_angles_deg": [
                fmmax_solver.diffraction_angle_deg(w, args.period_nm) for w in WAVELENGTHS
            ],
            "candidates": records,
        }, stream, indent=2)
    savemat(output / "candidate_analysis.mat", {
        "patterns": patterns,
        "old_scaled_period_efficiencies": old_efficiencies,
        "fixed_period_efficiencies": fixed_efficiencies,
        "period_nm": args.period_nm,
        "fourier_nn": args.fourier_nn,
        "source_indices": np.asarray(indices),
    })
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
