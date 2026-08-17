"""Unified ensemble analysis for corrected GLOnet and direct optimization.

Outputs performance/checkpoint comparisons, three-objective Pareto fronts,
threshold pass rates, structural Hamming statistics, objective-space coverage,
bidirectional set coverage, and direct-candidate manufacturability records.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

import analyze_revision_candidates as arc
import summarize_revision_results as srr


WAVELENGTHS = np.asarray([700.0, 800.0, 900.0])
THRESHOLDS = (0.6, 0.7, 0.8)


def pareto_indices(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    keep = np.ones(len(values), dtype=bool)
    for i, point in enumerate(values):
        if not keep[i]:
            continue
        dominates_i = np.all(values >= point, axis=1) & np.any(values > point, axis=1)
        if np.any(dominates_i):
            keep[i] = False
    return np.flatnonzero(keep)


def hamming_stats(patterns: np.ndarray, max_pairs: int = 200_000) -> dict:
    binary = np.asarray(patterns) > 0
    n = len(binary)
    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        distances = np.asarray([], dtype=float)
        mode = "none"
    elif total_pairs <= max_pairs:
        i, j = np.triu_indices(n, 1)
        distances = np.mean(binary[i] != binary[j], axis=1)
        mode = "exact"
    else:
        rng = np.random.default_rng(20260816)
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n - 1, size=max_pairs)
        j = j + (j >= i)
        distances = np.mean(binary[i] != binary[j], axis=1)
        mode = f"sampled_{max_pairs}"
    packed = np.packbits(binary, axis=1)
    unique_count = len(np.unique(packed, axis=0))
    return {
        "unique_patterns": int(unique_count),
        "unique_fraction": float(unique_count / n),
        "pair_mode": mode,
        "pair_count": int(len(distances)),
        "mean": float(distances.mean()) if len(distances) else 0.0,
        "median": float(np.median(distances)) if len(distances) else 0.0,
        "p05": float(np.quantile(distances, 0.05)) if len(distances) else 0.0,
        "p95": float(np.quantile(distances, 0.95)) if len(distances) else 0.0,
    }


def objective_occupancy(efficiencies: np.ndarray, width: float = 0.05) -> dict:
    bins_per_axis = int(round(1.0 / width))
    cells = np.floor(np.clip(efficiencies, 0.0, 1.0 - 1e-12) / width).astype(int)
    occupied = len(np.unique(cells, axis=0))
    return {
        "bin_width": width,
        "occupied_cells": int(occupied),
        "total_cells": int(bins_per_axis**3),
        "occupied_fraction": float(occupied / bins_per_axis**3),
        "occupied_per_candidate": float(occupied / len(efficiencies)),
    }


def feature_coverage(patterns: np.ndarray, efficiencies: np.ndarray,
                     period_nm: float) -> dict:
    minimum_nm = np.asarray([
        arc.structure_metrics(pattern, period_nm)["min_feature_nm"]
        for pattern in patterns
    ])
    means = efficiencies.mean(axis=1)
    bottlenecks = efficiencies.min(axis=1)
    thresholds = {}
    for value in (20.0, 40.0, 50.0):
        mask = minimum_nm >= value
        thresholds[str(int(value))] = {
            "count": int(mask.sum()),
            "fraction": float(mask.mean()),
            "best_mean": float(means[mask].max()) if np.any(mask) else None,
            "best_bottleneck": float(bottlenecks[mask].max()) if np.any(mask) else None,
        }
    return {
        "minimum_feature_nm": {
            "mean": float(minimum_nm.mean()),
            "median": float(np.median(minimum_nm)),
            "minimum": float(minimum_nm.min()),
            "maximum": float(minimum_nm.max()),
        },
        "thresholds_nm": thresholds,
    }


def ensemble_record(name: str, patterns: np.ndarray, efficiencies: np.ndarray,
                    period_nm: float) -> dict:
    patterns = np.sign(np.asarray(patterns).reshape(len(patterns), -1))
    efficiencies = np.asarray(efficiencies, dtype=float).reshape(len(patterns), 3)
    means = efficiencies.mean(axis=1)
    bottlenecks = efficiencies.min(axis=1)
    pareto = pareto_indices(efficiencies)
    return {
        "name": name,
        "n": int(len(patterns)),
        "mean_objective": srr.distribution_stats(means),
        "bottleneck": srr.distribution_stats(bottlenecks),
        "per_wavelength_mean": efficiencies.mean(axis=0).tolist(),
        "per_wavelength_max": efficiencies.max(axis=0).tolist(),
        "best_mean_index": int(np.argmax(means)),
        "best_mean_efficiencies": efficiencies[np.argmax(means)].tolist(),
        "best_bottleneck_index": int(np.argmax(bottlenecks)),
        "best_bottleneck_efficiencies": efficiencies[np.argmax(bottlenecks)].tolist(),
        "pareto_count": int(len(pareto)),
        "pareto_fraction": float(len(pareto) / len(patterns)),
        "pareto_indices": pareto.tolist(),
        "thresholds": {
            str(threshold): {
                "mean_objective_ge": float(np.mean(means >= threshold)),
                "all_three_ge": float(np.mean(bottlenecks >= threshold)),
                "count_all_three_ge": int(np.sum(bottlenecks >= threshold)),
            }
            for threshold in THRESHOLDS
        },
        "hamming": hamming_stats(patterns),
        "objective_coverage": objective_occupancy(efficiencies),
        "feature_coverage": feature_coverage(patterns, efficiencies, period_nm),
    }


def set_coverage(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of B weakly dominated by at least one candidate in A."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    dominated = np.zeros(len(b), dtype=bool)
    for start in range(0, len(b), 100):
        chunk = b[start:start + 100]
        dominated[start:start + len(chunk)] = np.any(
            np.all(a[:, None, :] >= chunk[None, :, :], axis=2), axis=0
        )
    return float(np.mean(dominated))


def nearest_hamming(source: np.ndarray, reference: np.ndarray) -> dict:
    source = np.asarray(source) > 0
    reference = np.asarray(reference) > 0
    minima = np.empty(len(source), dtype=float)
    for start in range(0, len(source), 50):
        chunk = source[start:start + 50]
        distances = np.mean(chunk[:, None, :] != reference[None, :, :], axis=2)
        minima[start:start + len(chunk)] = distances.min(axis=1)
    return {
        "mean": float(minima.mean()),
        "median": float(np.median(minima)),
        "p95": float(np.quantile(minima, 0.95)),
        "exact_match_fraction": float(np.mean(minima == 0)),
    }


def pair_record(name_a: str, patterns_a: np.ndarray, eff_a: np.ndarray,
                name_b: str, patterns_b: np.ndarray, eff_b: np.ndarray) -> dict:
    return {
        "a": name_a,
        "b": name_b,
        "objective_set_coverage_C_A_B": set_coverage(eff_a, eff_b),
        "objective_set_coverage_C_B_A": set_coverage(eff_b, eff_a),
        "structural_nearest_A_to_B": nearest_hamming(patterns_a, patterns_b),
        "structural_nearest_B_to_A": nearest_hamming(patterns_b, patterns_a),
    }


def select_distinct(patterns: np.ndarray, efficiencies: np.ndarray, period_nm: float):
    minimum_nm = np.asarray([
        arc.structure_metrics(pattern, period_nm)["min_feature_nm"]
        for pattern in patterns
    ])
    mean_score = efficiencies.mean(axis=1)
    bottleneck_score = efficiencies.min(axis=1)
    def constrained(score: np.ndarray, threshold: float) -> np.ndarray:
        return np.where(minimum_nm >= threshold, score, -np.inf)
    criteria = [
        ("best_mean_unconstrained", mean_score),
        ("best_bottleneck_unconstrained", bottleneck_score),
        ("best_mean_min40nm", constrained(mean_score, 40.0)),
        ("next_best_bottleneck_min40nm_distinct", constrained(bottleneck_score, 40.0)),
        ("best_mean_min50nm", constrained(mean_score, 50.0)),
    ]
    selected, labels, used = [], [], set()
    for label, score in criteria:
        index = next(int(i) for i in np.argsort(score)[::-1] if int(i) not in used)
        used.add(index)
        selected.append(index)
        labels.append(label)
    return labels, selected


def direct_candidate_records(patterns: np.ndarray, efficiencies: np.ndarray,
                             period_nm: float, nn: int):
    labels, indices = select_distinct(patterns, efficiencies, period_nm)
    records = []
    for label, index in zip(labels, indices):
        pattern = patterns[index]
        reference = efficiencies[index]
        tolerance = {}
        for radius in (1, 2, 3):
            for mode, dilate in (("si_dilation", True), ("si_erosion", False)):
                changed = arc.morphology_1d(pattern, radius, dilate)
                values = arc.evaluate(changed[None, :], period_nm, nn)[0]
                tolerance[f"{mode}_{radius}px"] = {
                    "efficiencies": values.tolist(),
                    "delta": (values - reference).tolist(),
                    "mean": float(values.mean()),
                    "bottleneck": float(values.min()),
                }
        records.append({
            "label": label,
            "source_index": index,
            "efficiencies": reference.tolist(),
            "mean_efficiency": float(reference.mean()),
            "bottleneck_efficiency": float(reference.min()),
            **arc.structure_metrics(pattern, period_nm),
            "tolerance": tolerance,
        })
    return records, labels, indices


def write_report(summary: dict, output: Path) -> None:
    def fmt(value) -> str:
        if value is None:
            return "not applicable"
        if isinstance(value, (float, np.floating)):
            return f"{value:.6f}"
        return str(value)

    glonet_audit = summary["protocol"]["glonet_checkpoint_audit_available"]
    direct_history = summary["protocol"]["direct_candidate_history_available"]
    glonet_checkpoint_line = (
        "- GLOnet checkpoint values resample each saved model with one fixed "
        "noise bank per seed; seed 1 has only its final model."
        if glonet_audit else
        "- GLOnet checkpoint values are 100-candidate ensemble means logged every "
        "50 steps. Saved models were not resampled, so candidate-level Pareto and "
        "structure statistics are unavailable at those checkpoints."
    )
    direct_final_line = (
        "- Direct final values contain 500 binary structures after 300 Adam updates."
        if direct_history else
        "- The legacy direct output retains only the step-300 ensemble mean "
        "(0.63898), not the 500 final structures."
    )
    lines = [
        "# Unified GLOnet and direct-optimization candidate analysis",
        "",
        "> Automatically generated data report. All efficiencies use the fixed "
        "1039.23 nm period, transmitted +1 order, and nn=40.",
        "",
        "## Reporting conventions",
        "",
        "- GLOnet production-final values are the 500 candidates in each original "
        "production output.",
        glonet_checkpoint_line,
        direct_final_line,
        "- The direct aggregate-best checkpoint is the single logged checkpoint "
        "with the highest mean over all 500 trajectories.",
        "- Direct per-trajectory-best values select the highest mean objective "
        "separately from 13 checkpoints along each trajectory.",
        "",
        "## Ensemble statistics",
        "",
        "| Ensemble | N | Mean objective | Maximum objective | Mean bottleneck | Pareto count | Unique structures | Mean Hamming | All-three >=0.7 | All-three >=0.8 | Occupied 0.05 cells | Minimum feature >=40 nm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in summary["ensembles"]:
        lines.append(
            f"| {record['name']} | {record['n']} | "
            f"{record['mean_objective']['mean']:.4f} | "
            f"{record['mean_objective']['maximum']:.4f} | "
            f"{record['bottleneck']['mean']:.4f} | {record['pareto_count']} | "
            f"{record['hamming']['unique_patterns']} | {record['hamming']['mean']:.4f} | "
            f"{record['thresholds']['0.7']['all_three_ge']:.3f} | "
            f"{record['thresholds']['0.8']['all_three_ge']:.3f} | "
            f"{record['objective_coverage']['occupied_cells']} | "
            f"{record['feature_coverage']['thresholds_nm']['40']['fraction']:.3f} |"
        )
    lines += ["", "## Iteration and checkpoint comparison", ""]
    for row in summary["iteration_comparison"]:
        lines.append(
            f"- **{row['method']}**: final mean {fmt(row.get('final_mean'))}; "
            f"aggregate-best checkpoint {fmt(row.get('best_checkpoint_step'))} / "
            f"{fmt(row.get('best_checkpoint_mean'))}; per-trajectory-best mean "
            f"{fmt(row.get('per_trajectory_best_mean'))}; best candidate "
            f"{fmt(row.get('best_candidate_mean'))}."
        )
    lines += ["", "## Selected direct-optimization candidates", ""]
    for row in summary["selected_direct_candidates"]:
        dilation = row["tolerance"]["si_dilation_1px"]
        erosion = row["tolerance"]["si_erosion_1px"]
        lines.append(
            f"- `{row['label']}`: efficiencies "
            f"{np.round(row['efficiencies'], 4).tolist()}, mean "
            f"{row['mean_efficiency']:.4f}, bottleneck "
            f"{row['bottleneck_efficiency']:.4f}, minimum feature "
            f"{row['min_feature_nm']:.2f} nm, nominal maximum aspect ratio "
            f"{row['max_aspect_ratio_325nm']:.2f}; one-pixel dilation bottleneck "
            f"{dilation['bottleneck']:.4f}, one-pixel erosion bottleneck "
            f"{erosion['bottleneck']:.4f}."
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        (
            "- GLOnet checkpoint resampling uses a fixed noise bank and is not the "
            "same Monte Carlo sample as the post-training production distribution."
            if glonet_audit else
            "- The best logged GLOnet checkpoint provides only a 100-candidate "
            "ensemble mean; its sample differs from the 500-candidate production "
            "distribution."
        ),
        "- Seed 1 has no intermediate saved model, so its best intermediate "
        "checkpoint candidates cannot be recovered reproducibly.",
        (
            "- Direct final and aggregate-checkpoint structures are available for "
            "complete candidate-level ensemble statistics."
            if direct_history else
            "- Candidate structures are unavailable at the direct final and single "
            "aggregate checkpoint. Pareto, Hamming, and manufacturability statistics "
            "therefore use only the retained per-trajectory-best set."
        ),
        "- Pooled GLOnet contains five times the candidates and training cost; it "
        "describes aggregate coverage but does not support matched-budget claims.",
        "- Morphological perturbations are periodic one-dimensional uniform erosion "
        "or dilation tests, not full EBL/RIE process simulations.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/revision_fixed"))
    parser.add_argument("--direct_dir", type=Path,
                        default=Path("results/revision_fixed/direct_multi"))
    parser.add_argument("--checkpoint_dir", type=Path,
                        default=Path("results/revision_fixed/glonet_checkpoint_audit"))
    parser.add_argument("--output_dir", type=Path,
                        default=Path("results/revision_fixed/unified_comparison"))
    parser.add_argument("--period_nm", type=float, default=1039.2304845413264)
    parser.add_argument("--fourier_nn", type=int, default=40)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ensemble_data: list[tuple[str, np.ndarray, np.ndarray]] = []
    production = []
    for seed in range(1, 6):
        patterns, efficiencies = srr.load_multi(args.root / f"multi_seed{seed}")
        name = f"GLOnet seed {seed}: production final"
        ensemble_data.append((name, patterns, efficiencies))
        production.append((patterns, efficiencies))
    pooled_patterns = np.concatenate([item[0] for item in production])
    pooled_efficiencies = np.concatenate([item[1] for item in production])
    ensemble_data.append(("GLOnet pooled 5 seeds: production final", pooled_patterns,
                          pooled_efficiencies))

    checkpoint_meta = []
    checkpoint_audit_available = all(
        (args.checkpoint_dir / f"seed{seed}_checkpoints.npz").exists()
        for seed in range(1, 6)
    )
    if checkpoint_audit_available:
        for seed in range(1, 6):
            with np.load(args.checkpoint_dir / f"seed{seed}_checkpoints.npz") as data:
                patterns = data["patterns"]
                efficiencies = data["efficiencies"]
                steps = data["steps"]
                best = int(data["best_checkpoint_index"])
                final = int(data["final_checkpoint_index"])
            ensemble_data.append((
                f"GLOnet seed {seed}: common-noise final step {steps[final]}",
                patterns[final], efficiencies[final],
            ))
            if best != final or len(steps) > 1:
                ensemble_data.append((
                    f"GLOnet seed {seed}: common-noise best saved step {steps[best]}",
                    patterns[best], efficiencies[best],
                ))
            checkpoint_meta.append({
                "seed": seed,
                "available_steps": steps.tolist(),
                "best_step": int(steps[best]),
                "final_step": int(steps[final]),
                "candidate_level_available": True,
            })
    else:
        for seed in range(1, 6):
            history = loadmat(args.root / f"multi_seed{seed}" / "history.mat")
            sign_mean = np.asarray(history["effs_sign_history"]).reshape(-1)
            logged_steps = (np.arange(len(sign_mean)) + 1) * 50
            best = int(np.argmax(sign_mean))
            checkpoint_meta.append({
                "seed": seed,
                "available_logged_steps": logged_steps.tolist(),
                "best_logged_step": int(logged_steps[best]),
                "best_logged_mean": float(sign_mean[best]),
                "final_logged_mean": float(sign_mean[-1]),
                "candidate_level_available": False,
                "note": "Training log batch only; model/candidate recovery requires checkpoint audit.",
            })

    direct = loadmat(args.direct_dir / "direct_multi_results.mat")
    direct_best_patterns = np.asarray(direct["patterns"])
    direct_best_eff = np.asarray(direct["efficiencies"])
    direct_candidate_history_available = "final_patterns" in direct
    direct_sets = {}
    direct_checkpoint_summary = {}
    if direct_candidate_history_available:
        direct_final_patterns = np.asarray(direct["final_patterns"])
        direct_final_eff = np.asarray(direct["final_efficiencies"])
        direct_steps = np.asarray(direct["checkpoint_steps"]).reshape(-1).astype(int)
        direct_cp_patterns = np.asarray(direct["checkpoint_patterns"])
        direct_cp_eff = np.asarray(direct["checkpoint_efficiencies"])
        direct_cp_means = direct_cp_eff.mean(axis=2).mean(axis=1)
        global_best = int(np.argmax(direct_cp_means))
        ensemble_data.extend([
            ("Direct: final step 300", direct_final_patterns, direct_final_eff),
            (f"Direct: global best checkpoint step {direct_steps[global_best]}",
             direct_cp_patterns[global_best], direct_cp_eff[global_best]),
            ("Direct: per-trajectory best checkpoint", direct_best_patterns, direct_best_eff),
        ])
        direct_sets = {
            "final": (direct_final_patterns, direct_final_eff),
            "global_best": (direct_cp_patterns[global_best], direct_cp_eff[global_best]),
            "per_trajectory_best": (direct_best_patterns, direct_best_eff),
        }
        direct_checkpoint_summary = {
            "candidate_level_available": True,
            "steps": direct_steps.tolist(),
            "global_best_step": int(direct_steps[global_best]),
            "global_best_ensemble_mean": float(direct_cp_means[global_best]),
            "final_ensemble_mean": float(direct_final_eff.mean(axis=1).mean()),
        }
    else:
        batch_paths = sorted((args.direct_dir / "batches").glob("batch_*.npz"))
        histories, direct_steps = [], None
        for path in batch_paths:
            with np.load(path) as batch:
                steps = np.asarray(batch["history_steps"]).reshape(-1).astype(int)
                if direct_steps is None:
                    direct_steps = steps
                elif not np.array_equal(direct_steps, steps):
                    raise ValueError(f"Inconsistent history steps in {path}")
                histories.append(np.asarray(batch["history_mean"]).reshape(-1))
        history_mean = np.mean(np.stack(histories), axis=0)
        global_best = int(np.argmax(history_mean))
        ensemble_data.append(
            ("Direct: per-trajectory best checkpoint", direct_best_patterns, direct_best_eff)
        )
        direct_sets = {
            "per_trajectory_best": (direct_best_patterns, direct_best_eff),
        }
        direct_checkpoint_summary = {
            "candidate_level_available": False,
            "legacy_step_labels": direct_steps.tolist(),
            "ensemble_mean_by_checkpoint": history_mean.tolist(),
            "global_best_legacy_step_label": int(direct_steps[global_best]),
            "global_best_ensemble_mean": float(history_mean[global_best]),
            "final_ensemble_mean": float(history_mean[-1]),
            "note": (
                "Legacy files saved only aggregate checkpoint means and per-trajectory "
                "best candidates; final/global-checkpoint patterns cannot be recovered."
            ),
        }

    records = [ensemble_record(*item, args.period_nm) for item in ensemble_data]
    comparisons = []
    for seed, (patterns, efficiencies) in enumerate(production, start=1):
        for direct_name, (dp, de) in direct_sets.items():
            comparisons.append(pair_record(
                f"GLOnet seed {seed}: production final", patterns, efficiencies,
                f"Direct: {direct_name}", dp, de,
            ))

    selected, selected_labels, selected_indices = direct_candidate_records(
        direct_best_patterns, direct_best_eff, args.period_nm, args.fourier_nn
    )
    iteration_comparison = []
    for seed, (_, efficiencies) in enumerate(production, start=1):
        metadata = checkpoint_meta[seed - 1]
        iteration_comparison.append({
            "method": f"GLOnet seed {seed}",
            "final_mean": float(efficiencies.mean(axis=1).mean()),
            "best_checkpoint_step": metadata.get("best_step", metadata.get("best_logged_step")),
            "best_checkpoint_mean": metadata.get("best_mean", metadata.get("best_logged_mean")),
            "per_trajectory_best_mean": None,
            "best_candidate_mean": float(efficiencies.mean(axis=1).max()),
            "checkpoint_candidate_level_available": metadata["candidate_level_available"],
        })
    iteration_comparison.append({
        "method": "Direct optimization",
        "final_mean": direct_checkpoint_summary["final_ensemble_mean"],
        "best_checkpoint_step": direct_checkpoint_summary.get(
            "global_best_step", direct_checkpoint_summary.get("global_best_legacy_step_label")
        ),
        "best_checkpoint_mean": direct_checkpoint_summary["global_best_ensemble_mean"],
        "per_trajectory_best_mean": float(direct_best_eff.mean(axis=1).mean()),
        "best_candidate_mean": float(direct_best_eff.mean(axis=1).max()),
        "checkpoint_candidate_level_available": direct_candidate_history_available,
    })
    summary = {
        "protocol": {
            "wavelengths_nm": WAVELENGTHS.tolist(),
            "period_nm": args.period_nm,
            "fourier_nn": args.fourier_nn,
            "thresholds": list(THRESHOLDS),
            "objective_coverage_bin_width": 0.05,
            "direct_candidate_history_available": direct_candidate_history_available,
            "direct_checkpoint_summary": direct_checkpoint_summary,
            "glonet_checkpoint_audit_available": checkpoint_audit_available,
            "glonet_checkpoint_availability": checkpoint_meta,
        },
        "ensembles": records,
        "iteration_comparison": iteration_comparison,
        "pairwise_coverage": comparisons,
        "selected_direct_candidates": selected,
    }
    (args.output_dir / "unified_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    savemat(args.output_dir / "selected_direct_candidates.mat", {
        "patterns": direct_best_patterns[selected_indices],
        "efficiencies": direct_best_eff[selected_indices],
        "indices": np.asarray(selected_indices),
        "labels": np.asarray(selected_labels, dtype=object),
        "wavelengths_nm": WAVELENGTHS,
    })
    with (args.output_dir / "ensemble_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "ensemble", "n", "mean_objective", "max_objective", "mean_bottleneck",
            "pareto_count", "unique_patterns", "mean_hamming", "all3_ge_0.6",
            "all3_ge_0.7", "all3_ge_0.8", "occupied_0.05_cells",
            "feature_ge_20nm", "feature_ge_40nm", "feature_ge_50nm",
        ])
        for row in records:
            writer.writerow([
                row["name"], row["n"], row["mean_objective"]["mean"],
                row["mean_objective"]["maximum"], row["bottleneck"]["mean"],
                row["pareto_count"], row["hamming"]["unique_patterns"],
                row["hamming"]["mean"], row["thresholds"]["0.6"]["all_three_ge"],
                row["thresholds"]["0.7"]["all_three_ge"],
                row["thresholds"]["0.8"]["all_three_ge"],
                row["objective_coverage"]["occupied_cells"],
                row["feature_coverage"]["thresholds_nm"]["20"]["fraction"],
                row["feature_coverage"]["thresholds_nm"]["40"]["fraction"],
                row["feature_coverage"]["thresholds_nm"]["50"]["fraction"],
            ])
    write_report(summary, args.output_dir / "report.md")
    print(json.dumps({
        "direct_candidate_history_available": direct_candidate_history_available,
        "direct_checkpoint_summary": direct_checkpoint_summary,
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
