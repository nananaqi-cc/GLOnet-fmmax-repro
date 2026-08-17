# Numerical data

This directory contains a compact snapshot of the corrected fixed-period
production results.

- `single_w700_seed1`, `single_w800_seed1`, and `single_w900_seed1` contain the
  fixed-period single-wavelength reference candidates.
- `multi_seed1` through `multi_seed5` contain the final candidates from five
  independent three-wavelength GLOnet runs.
- `direct_multi` contains the retained direct-optimization candidates and the
  run summary.
- `validation` contains the full-pixel gradient, Fourier-order convergence,
  and energy-conservation summaries.
- `production_summary` contains machine-readable aggregate statistics and the
  selected representative GLOnet candidates.
- `unified_comparison_existing` contains the Pareto, threshold-success,
  Hamming-distance, candidate-coverage, minimum-feature, and selected direct
  candidate comparisons.

The direct baseline has three distinct reporting conventions:

1. Final iteration: the ensemble summary after 300 Adam updates.
2. Best aggregate checkpoint: one common logged checkpoint selected by the
   highest ensemble mean.
3. Per-trajectory best: a potentially different logged binary checkpoint is
   retained for each trajectory.

The compact historical run did not retain candidate-level structures for the
first two conventions. Candidate-level Pareto and manufacturability analyses
therefore use the per-trajectory-best direct set and state that limitation.
