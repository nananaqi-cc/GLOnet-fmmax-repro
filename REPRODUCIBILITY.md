# Reproducibility guide

This package separates three activities: inspecting retained numerical data,
regenerating aggregate analyses, and rerunning the full optimization protocol.
The first two can be completed without retraining.

## Retained-data checks

The compact dataset under `results/revision_fixed/` contains final candidate
matrices for three single-wavelength GLOnet references, five multi-wavelength
GLOnet seeds, the retained direct-optimization candidates, and numerical
validation data. It intentionally omits draft documents, intermediate figures,
training logs, and most model checkpoints.

Regenerate the primary aggregate summary:

```bash
python summarize_revision_results.py
```

The expected seed-level mean of the five GLOnet candidate means is
approximately `0.624720`, with sample standard deviation `0.091070`. The
retained per-trajectory-best direct candidate mean is approximately `0.699054`.

Regenerate the unified candidate comparison:

```bash
python analyze_method_ensembles.py
```

The generated `results/revision_fixed/unified_comparison/` directory contains
`unified_summary.json`, `ensemble_metrics.csv`, `selected_direct_candidates.mat`,
and an English `report.md`. For the legacy direct run, the expected final
ensemble mean is `0.638979` and the best single aggregate checkpoint mean is
`0.653010` at the historical step label 200.

## Numerical validation

The retained validation summaries document the full 1280-derivative finite-
difference comparison, Fourier-order convergence, and lossless energy
conservation. To recompute the validation suite with FMMAX:

```bash
python validate_revision_fixed.py
python energy_conservation_fixed.py
```

These commands require the specified JAX/FMMAX environment and overwrite the
corresponding validation outputs. The reported maximum `nn=40` versus `nn=80`
efficiency difference is below `4.8e-4`, and the maximum lossless energy-balance
error is approximately `7.1e-7`.

## Full optimization rerun

The included final outputs trigger the stage-skipping checks in
`run_revision_experiments.sh`. Use a fresh clone or move the included
`results/revision_fixed/` directory before a clean rerun:

```bash
PYTHON_BIN=.venv/bin/python bash run_revision_experiments.sh
```

The production protocol contains three single-wavelength runs, five independent
three-wavelength GLOnet seeds, a 500-start direct Adam baseline, and validation.
It is computationally expensive and should be run on a CUDA-capable GPU.

## Checkpoint limitation

The historical direct run retained per-trajectory-best candidate structures and
aggregate checkpoint means, but not candidate-level final or aggregate-best
structures. Candidate-level Pareto, Hamming-distance, coverage, and
manufacturability comparisons therefore use the per-trajectory-best direct set.
This limitation is encoded in the machine-readable summaries and must remain
visible in downstream reporting.
