# GLOnet-FMMAX fixed-period reproducibility package

This repository contains the source code and compact numerical data needed to
reproduce the fixed-period, three-wavelength GLOnet-FMMAX calculations. The
physical structure is a one-dimensional Si-on-glass metagrating with a period
of 1039.2304845 nm. The target is the transmitted +1 diffraction order at
700, 800, and 900 nm.

This repository intentionally contains no manuscript source, reviewer
correspondence, submission forms, or draft PDFs.

## Included material

- GLOnet single- and multi-wavelength training code.
- Matched Adam direct pixel-optimization code.
- FMMAX transmission, gradient, convergence, and energy-conservation checks.
- Configuration for the fixed-period production protocol.
- Compact final candidate matrices and summary data used by the analysis.
- Scripts that regenerate the numerical summaries from the included data.

The optical-constant table in `solvers/p_Si.mat` is the silicon data distributed
with the original GLOnet implementation and attributed there to Green, *Solar
Energy Materials and Solar Cells* **92**, 1305-1310 (2008).

## Environment

The production calculations used Ubuntu 24.04 under WSL2, Python 3.12.3,
FMMAX 1.7.1, JAX/jaxlib 0.6.2 with the CUDA 12 plugin, PyTorch 2.8.0,
torchvision 0.23.0, and double-precision JAX calculations. An NVIDIA GPU with
at least 12 GB memory is recommended for the production configuration.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirm that JAX detects the GPU before starting a production run:

```bash
python -c "import jax; print(jax.devices())"
```

## Reproduce summaries from included data

```bash
python summarize_revision_results.py
python analyze_method_ensembles.py
```

The included data are under `results/revision_fixed/`. See
`results/README.md` for the scope and checkpoint conventions.

## Rerun the numerical protocol

`run_revision_experiments.sh` skips a stage when its expected output already
exists. To perform a clean rerun, use a fresh clone or move the included
`results/revision_fixed` directory outside the repository first.

```bash
PYTHON_BIN=.venv/bin/python bash run_revision_experiments.sh
```

The full protocol consists of three fixed-period single-wavelength references,
five independent three-wavelength GLOnet seeds, a 500-start direct baseline,
and the numerical validation suite. It requires substantial GPU time.

## Reproducibility scope

The repository distinguishes direct-optimization final-iterate statistics,
the best single aggregate checkpoint, and per-trajectory best checkpoints.
Only the latter retains a different checkpoint for each direct trajectory.
The GLOnet seed-level statistics and the best-of-five candidate are reported
separately.

## License

See `LICENSE`.
