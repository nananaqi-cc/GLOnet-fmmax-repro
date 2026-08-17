"""Summarize only corrected fixed-period production runs.

This script intentionally refuses to fall back to legacy wavelength-scaled
outputs. It creates the cross-seed, baseline, manufacturability, tolerance,
and Fourier-domain artifacts needed for the revised manuscript.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat

import analyze_revision_candidates as arc


WAVELENGTHS = np.asarray([700.0, 800.0, 900.0])


def one_mat(directory: Path) -> Path:
    matches = sorted((directory / "outputs").glob("*.mat"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one production MAT file in {directory / 'outputs'}, found {len(matches)}")
    return matches[0]


def load_multi(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    data = loadmat(one_mat(directory))
    patterns = np.sign(np.asarray(data["imgs"]).reshape(len(data["imgs"]), -1))
    efficiencies = np.column_stack(
        [np.asarray(data[f"effs_sign_wl{int(w)}"]).reshape(-1) for w in WAVELENGTHS]
    )
    return patterns, efficiencies


def load_single(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    data = loadmat(one_mat(directory))
    patterns = np.sign(np.asarray(data["imgs"]).reshape(len(data["imgs"]), -1))
    return patterns, np.asarray(data["effs"]).reshape(-1)


def distribution_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "maximum": float(values.max()),
        "success_ge_0.7": float(np.mean(values >= 0.7)),
        "success_ge_0.8": float(np.mean(values >= 0.8)),
    }


def plot_method_comparison(seed_rows: list[dict], direct_efficiencies: np.ndarray,
                           output: Path) -> None:
    seed_mean = np.asarray([row["candidate_mean"]["mean"] for row in seed_rows])
    seed_std = np.asarray([row["candidate_mean"]["sample_std"] for row in seed_rows])
    seed_best = np.asarray([row["candidate_mean"]["maximum"] for row in seed_rows])
    seed_bottleneck = np.asarray([row["bottleneck"]["mean"] for row in seed_rows])
    direct_mean = direct_efficiencies.mean(axis=1)
    direct_bottleneck = direct_efficiencies.min(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.7))
    x = np.arange(1, 6)
    axes[0].errorbar(x, seed_mean, yerr=seed_std, fmt="o", capsize=3,
                     label="GLOnet seed: candidate mean +/- SD")
    axes[0].axhline(direct_mean.mean(), color="C1", ls="--",
                    label="direct: candidate mean")
    axes[0].set(ylabel="three-wavelength mean efficiency", xlabel="GLOnet seed",
                xticks=x, ylim=(0.35, 0.95))
    axes[0].legend(fontsize=6, loc="lower right")

    axes[1].bar(x - 0.18, seed_best, width=0.36, label="GLOnet")
    axes[1].bar(x + 0.18, np.repeat(direct_mean.max(), 5), width=0.36,
                label="direct best-of-500")
    axes[1].set(ylabel="best candidate mean efficiency", xlabel="GLOnet seed",
                xticks=x, ylim=(0.55, 0.92))
    axes[1].legend(fontsize=6, loc="lower right")

    axes[2].bar(x - 0.18, seed_bottleneck, width=0.36, label="GLOnet")
    axes[2].bar(x + 0.18, np.repeat(direct_bottleneck.mean(), 5), width=0.36,
                label="direct")
    axes[2].set(ylabel="mean bottleneck efficiency", xlabel="GLOnet seed",
                xticks=x, ylim=(0.15, 0.78))
    axes[2].legend(fontsize=6, loc="upper left")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def select_candidates(patterns: np.ndarray, efficiencies: np.ndarray):
    criteria = [
        ("best_mean", efficiencies.mean(axis=1)),
        ("best_bottleneck", efficiencies.min(axis=1)),
        ("best_700nm", efficiencies[:, 0]),
        ("best_800nm", efficiencies[:, 1]),
        ("best_900nm", efficiencies[:, 2]),
    ]
    selected, labels = [], []
    used: set[int] = set()
    for label, score in criteria:
        index = next(int(i) for i in np.argsort(score)[::-1] if int(i) not in used)
        used.add(index)
        selected.append(index)
        labels.append(label)
    return patterns[selected], efficiencies[selected], labels, selected


def tolerance_records(pattern: np.ndarray, reference: np.ndarray, period_nm: float, nn: int) -> dict:
    records = {}
    for radius in (1, 2, 3):
        for mode, dilate in (("si_dilation", True), ("si_erosion", False)):
            changed = arc.morphology_1d(pattern, radius, dilate)
            efficiency = arc.evaluate(changed[None, :], period_nm, nn)[0]
            records[f"{mode}_{radius}px"] = {
                "efficiencies": efficiency.tolist(),
                "delta": (efficiency - reference).tolist(),
            }
    return records


def plot_fourier(patterns: np.ndarray, labels: list[str], output: Path) -> None:
    fig, axes = plt.subplots(len(patterns), 2, figsize=(7.2, 1.55 * len(patterns)),
                             gridspec_kw={"width_ratios": [1.2, 1.0]})
    for row, (pattern, label) in enumerate(zip(patterns, labels)):
        axes[row, 0].step(np.arange(pattern.size), pattern, where="mid", lw=0.75)
        axes[row, 0].set_ylim(-1.2, 1.2)
        axes[row, 0].set_ylabel(label, fontsize=7.5)
        power = np.abs(np.fft.rfft(pattern) / pattern.size) ** 2
        harmonics = np.arange(1, min(41, power.size))
        axes[row, 1].stem(harmonics, power[1:len(harmonics) + 1],
                          linefmt="C1-", markerfmt="C1.", basefmt=" ")
        axes[row, 1].set_yscale("log")
        axes[row, 1].set_ylim(1e-7, max(1e-5, power[1:len(harmonics) + 1].max() * 2))
    axes[-1, 0].set_xlabel("pixel index")
    axes[-1, 1].set_xlabel("spatial harmonic |m|")
    axes[0, 0].set_title("Binary pattern")
    axes[0, 1].set_title("Fourier power")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/revision_fixed"))
    parser.add_argument("--period_nm", type=float, default=1039.2304845413264)
    parser.add_argument("--fourier_nn", type=int, default=40)
    args = parser.parse_args()
    out = args.root / "production_summary"
    out.mkdir(parents=True, exist_ok=True)

    all_patterns, all_efficiencies, seed_rows = [], [], []
    for seed in range(1, 6):
        patterns, efficiencies = load_multi(args.root / f"multi_seed{seed}")
        all_patterns.append(patterns)
        all_efficiencies.append(efficiencies)
        seed_rows.append({
            "seed": seed,
            "candidate_mean": distribution_stats(efficiencies.mean(axis=1)),
            "bottleneck": distribution_stats(efficiencies.min(axis=1)),
            "per_wavelength": {
                str(int(w)): distribution_stats(efficiencies[:, j])
                for j, w in enumerate(WAVELENGTHS)
            },
        })
    patterns = np.concatenate(all_patterns)
    efficiencies = np.concatenate(all_efficiencies)

    singles = {}
    fourier_patterns, fourier_labels = [], []
    for wavelength in WAVELENGTHS:
        sp, se = load_single(args.root / f"single_w{int(wavelength)}_seed1")
        singles[str(int(wavelength))] = distribution_stats(se)
        best = int(np.argmax(se))
        fourier_patterns.append(sp[best])
        fourier_labels.append(f"single {int(wavelength)} nm")

    chosen_patterns, chosen_eff, chosen_labels, chosen_indices = select_candidates(patterns, efficiencies)
    candidate_rows = []
    for pattern, eff, label, index in zip(chosen_patterns, chosen_eff, chosen_labels, chosen_indices):
        candidate_rows.append({
            "label": label,
            "pooled_index": index,
            "source_seed": index // len(all_patterns[0]) + 1,
            "source_candidate_index": index % len(all_patterns[0]),
            "efficiencies": eff.tolist(),
            "mean_efficiency": float(eff.mean()),
            "bottleneck_efficiency": float(eff.min()),
            **arc.structure_metrics(pattern, args.period_nm),
            "tolerance": tolerance_records(pattern, eff, args.period_nm, args.fourier_nn),
        })

    direct_path = args.root / "direct_multi" / "summary.json"
    if not direct_path.exists():
        raise FileNotFoundError(f"Missing matched baseline summary: {direct_path}")
    direct = json.loads(direct_path.read_text(encoding="utf-8"))
    direct_mat = loadmat(args.root / "direct_multi" / "direct_multi_results.mat")
    direct_efficiencies = np.asarray(direct_mat["efficiencies"], dtype=float).reshape(-1, 3)
    direct["objective_distribution"] = distribution_stats(direct_efficiencies.mean(axis=1))
    direct["bottleneck_distribution"] = distribution_stats(direct_efficiencies.min(axis=1))

    fourier_patterns.append(chosen_patterns[0])
    fourier_labels.append("multi, best mean")
    plot_fourier(np.asarray(fourier_patterns), fourier_labels, out / "fourier_corrected.png")
    plot_method_comparison(seed_rows, direct_efficiencies, out / "method_comparison.png")

    summary = {
        "protocol": {
            "fixed_period_nm": args.period_nm,
            "wavelengths_nm": WAVELENGTHS.tolist(),
            "fourier_nn": args.fourier_nn,
            "multi_seeds": 5,
            "candidates_per_seed": int(len(all_patterns[0])),
        },
        "multi_cross_seed": {
            "seed_level_mean_of_candidate_means": distribution_stats(
                np.asarray([row["candidate_mean"]["mean"] for row in seed_rows])
            ),
            "pooled_candidate_mean": distribution_stats(efficiencies.mean(axis=1)),
            "pooled_bottleneck": distribution_stats(efficiencies.min(axis=1)),
            "per_seed": seed_rows,
        },
        "single_wavelength": singles,
        "matched_direct_pixel_optimization": direct,
        "selected_candidates": candidate_rows,
    }
    (out / "revision_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    savemat(out / "selected_candidates.mat", {
        "patterns": chosen_patterns,
        "efficiencies": chosen_eff,
        "pooled_indices": np.asarray(chosen_indices),
        "wavelengths_nm": WAVELENGTHS,
        "period_nm": args.period_nm,
        "fourier_nn": args.fourier_nn,
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
