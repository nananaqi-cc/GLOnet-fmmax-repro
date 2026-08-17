#!/usr/bin/env bash
set -euo pipefail

if [[ -x .venv-linux/bin/python ]]; then
  DEFAULT_PYTHON=.venv-linux/bin/python
else
  DEFAULT_PYTHON=.venv/bin/python
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
PERIOD_NM="${PERIOD_NM:-1039.2304845413264}"
FOURIER_NN="${FOURIER_NN:-40}"
CONFIG="configs/fixed_period_1039nm.json"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplconfig-linux}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

has_mat_output() {
  local output_dir="$1"
  compgen -G "$output_dir/outputs/*.mat" >/dev/null
}

run_glonet_stage() {
  local label="$1"
  local output_dir="$2"
  shift 2
  if has_mat_output "$output_dir"; then
    echo "[skip] $label: completed MAT output already exists"
    return
  fi
  echo "[run] $label"
  "$@"
}

for wavelength in 700 800 900; do
  output_dir="results/revision_fixed/single_w${wavelength}_seed1"
  run_glonet_stage "single wavelength ${wavelength} nm" "$output_dir" \
    "$PYTHON_BIN" main.py --params_file "$CONFIG" \
    --output_dir "$output_dir" \
    --wavelength "$wavelength" --period_nm "$PERIOD_NM" \
    --fourier_nn "$FOURIER_NN" --seed 1
done

for seed in 1 2 3 4 5; do
  output_dir="results/revision_fixed/multi_seed${seed}"
  run_glonet_stage "three-wavelength seed ${seed}" "$output_dir" \
    "$PYTHON_BIN" main_multi_wl.py --params_file "$CONFIG" \
    --output_dir "$output_dir" \
    --wavelengths 700,800,900 --period_nm "$PERIOD_NM" \
    --fourier_nn "$FOURIER_NN" --seed "$seed"
done

if [[ -f results/revision_fixed/direct_multi/summary.json ]]; then
  echo "[skip] matched direct optimization: summary already exists"
else
  echo "[run] matched direct optimization"
  "$PYTHON_BIN" direct_opt_multi.py --wavelengths 700,800,900 \
    --period_nm "$PERIOD_NM" --fourier_nn "$FOURIER_NN" \
    --n_seeds 500 --n_iter 300 --batch_size 25 \
    --output_dir results/revision_fixed/direct_multi
fi

if [[ ! -f results/revision_fixed/validation/validation_summary.json ]]; then
  "$PYTHON_BIN" validate_revision_fixed.py --period_nm "$PERIOD_NM" --fourier_nn "$FOURIER_NN"
else
  echo "[skip] validation suite: summary already exists"
fi
if [[ ! -f results/revision_fixed/validation/energy_conservation.json ]]; then
  "$PYTHON_BIN" energy_conservation_fixed.py --period_nm "$PERIOD_NM"
else
  echo "[skip] energy conservation: summary already exists"
fi
"$PYTHON_BIN" summarize_revision_results.py --period_nm "$PERIOD_NM" --fourier_nn "$FOURIER_NN"
