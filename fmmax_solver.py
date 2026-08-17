"""
FMMAX-based RCWA solver for 1D TM metasurface deflector.

Replaces:
  solvers/Eval_Eff_1D.m / Eval_Eff_1D_parallel.m       -> eval_eff_1d / eval_eff_batch
  solvers/GradientFromSolver_1D.m / *_parallel.m       -> gradient_from_solver_1d / gradient_from_solver_batch

Design: FMMAX runs as a JAX function; gradients come from `jax.grad` end-to-end.
The PyTorch boundary uses numpy arrays for inputs/outputs.

Conventions (mirroring Eval_Eff_1D.m):
  - 1D grating periodic in x. TM polarization (H along y).
  - Incidence normal from glass substrate (matching `inc_bottom_transmitted`).
  - Target diffraction order m = +1 in air, emerging at angle = desired_angle_deg.
  - Layer stack glass / patterned Si / air with claddings of 0.51*wavelength.

Output is normalised so that:
  eff = (forward Poynting flux at order m=+1 in air) / (incident forward flux in glass).
"""

from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import functools
import numpy as np
import scipy.io as io

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import fmmax


# ---------------------------------------------------------------------------
# Material data
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_si_data = io.loadmat(os.path.join(_HERE, "solvers", "p_Si.mat"))
_SI_WL = _si_data["WL"].flatten().astype(np.float64)
_SI_N = _si_data["n"].flatten().astype(np.float64)


def si_index(wavelength_nm: float) -> float:
    """Real refractive index of Si at given wavelength (k dropped, matching MATLAB)."""
    return float(np.interp(wavelength_nm, _SI_WL, _SI_N))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_AIR = 1.0
N_GLASS = 1.45
THICKNESS_NM = 325.0
FORMULATION = fmmax.Formulation.JONES_DIRECT_FOURIER
DEFAULT_FOURIER_NN = 40

# Primary fixed-period revision protocol.  The period is chosen so that the
# longest design wavelength (900 nm) exits in the +1 order at 60 degrees;
# shorter wavelengths therefore remain propagating at smaller angles.
FIXED_PERIOD_NM = float(900.0 / np.sin(np.deg2rad(60.0)))


def period_from_angle(wavelength_nm: float, angle_deg: float) -> float:
    """Legacy wavelength-scaled period used by the original experiments."""
    return float(wavelength_nm) / np.sin(np.deg2rad(float(angle_deg)))


def diffraction_angle_deg(wavelength_nm: float, period_nm: float, order: int = 1) -> float:
    """Propagating diffraction angle in air for normal incidence.

    Raises ValueError when the requested order is evanescent.
    """
    argument = float(order) * float(wavelength_nm) / float(period_nm)
    if abs(argument) > 1.0:
        raise ValueError(
            f"Order m={order} is evanescent: |m*lambda/period|={abs(argument):.6f} > 1"
        )
    return float(np.rad2deg(np.arcsin(argument)))


def _resolve_period(wavelength_nm: float, desired_angle_deg: float | None,
                    period_nm: float | None) -> float:
    if period_nm is not None:
        return float(period_nm)
    if desired_angle_deg is None:
        raise ValueError("desired_angle_deg is required when period_nm is not supplied")
    return period_from_angle(wavelength_nm, desired_angle_deg)


# ---------------------------------------------------------------------------
# Cached expansion / order indices  (depend only on period and nn)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=64)
def _expansion_for(period_nm: float, nn: int):
    """1D-as-2D lattice with deeply evanescent y direction so CIRCULAR
    truncation reduces to a pure x expansion."""
    lattice = fmmax.LatticeVectors(
        u=float(period_nm) * fmmax.X,
        v=float(period_nm) * 0.01 * fmmax.Y,
    )
    expansion = fmmax.generate_expansion(
        primitive_lattice_vectors=lattice,
        approximate_num_terms=2 * nn + 1,
        truncation=fmmax.Truncation.CIRCULAR,
    )
    bc = np.asarray(expansion.basis_coefficients)
    m0 = int(np.where(bc[:, 0] == 0)[0][0])
    m1 = int(np.where(bc[:, 0] == 1)[0][0])
    return lattice, expansion, m0, m1


# ---------------------------------------------------------------------------
# Differentiable forward simulation
# ---------------------------------------------------------------------------
def _forward_efficiency(
    nvec: jnp.ndarray,
    wavelength_nm: float,
    lattice: fmmax.LatticeVectors,
    expansion: fmmax.Expansion,
    m0_idx: int,
    m1_idx: int,
    clad_t: float,
    thickness: float = THICKNESS_NM,
) -> jnp.ndarray:
    """JAX-pure efficiency function: nvec -> scalar efficiency at m=+1.

    `nvec` is the (Nx,) refractive index profile of the patterned middle layer.
    """
    num_terms = int(expansion.num_terms)

    eps_air = jnp.asarray([[N_AIR ** 2]], dtype=jnp.complex128)
    eps_si = (nvec ** 2).astype(jnp.complex128).reshape(-1, 1)
    eps_glass = jnp.asarray([[N_GLASS ** 2]], dtype=jnp.complex128)

    wl = jnp.asarray(float(wavelength_nm))
    in_plane_wavevector = jnp.asarray([0.0, 0.0])

    common = dict(
        wavelength=wl,
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=lattice,
        expansion=expansion,
        formulation=FORMULATION,
    )

    sol_glass = fmmax.eigensolve_isotropic_media(permittivity=eps_glass, **common)
    sol_si = fmmax.eigensolve_isotropic_media(permittivity=eps_si, **common)
    sol_air = fmmax.eigensolve_isotropic_media(permittivity=eps_air, **common)

    thicknesses = [
        jnp.asarray(float(clad_t)),
        jnp.asarray(float(thickness)),
        jnp.asarray(float(clad_t)),
    ]
    s_matrix = fmmax.stack_s_matrix([sol_glass, sol_si, sol_air], thicknesses)

    # TM block lives in indices [num_terms, 2*num_terms). One-hot at m=0.
    inc = jnp.zeros((2 * num_terms, 1), dtype=jnp.complex128)
    inc = inc.at[num_terms + m0_idx, 0].set(1.0 + 0.0j)
    # FMMAX convention: s11 maps forward-going amplitudes at the stack start
    # to forward-going amplitudes at the stack end.  s21 is reflection back at
    # the start and must not be evaluated using the air-side layer solution.
    transmitted = s_matrix.s11 @ inc

    flux_t, _ = fmmax.directional_poynting_flux(
        transmitted, jnp.zeros_like(transmitted), sol_air
    )
    flux_inc, _ = fmmax.directional_poynting_flux(
        inc, jnp.zeros_like(inc), sol_glass
    )

    total_inc = jnp.sum(flux_inc).real
    flux_m1 = flux_t[m1_idx, 0].real + flux_t[num_terms + m1_idx, 0].real

    # Avoid 0/0 when nn is too small
    return jnp.where(total_inc > 1e-15, flux_m1 / total_inc, 0.0)


# JIT compiled cache keyed on the static args (wavelength, period, nn, clad_t).
@functools.lru_cache(maxsize=64)
def _eff_jit(wavelength_nm: float, period_nm: float, nn: int, clad_t: float):
    """Returns a jitted (nvec) -> efficiency function."""
    lattice, expansion, m0, m1 = _expansion_for(period_nm, nn)
    return jax.jit(
        lambda nvec: _forward_efficiency(
            nvec, wavelength_nm, lattice, expansion, m0, m1, clad_t,
        )
    )


@functools.lru_cache(maxsize=64)
def _eff_grad_jit(wavelength_nm: float, period_nm: float, nn: int, clad_t: float):
    """Returns a jitted (nvec) -> (efficiency, d eff/d nvec) function."""
    lattice, expansion, m0, m1 = _expansion_for(period_nm, nn)
    fn = lambda nvec: _forward_efficiency(
        nvec, wavelength_nm, lattice, expansion, m0, m1, clad_t,
    )
    return jax.jit(jax.value_and_grad(fn))


@functools.lru_cache(maxsize=64)
def _eff_grad_batch_jit(wavelength_nm: float, period_nm: float, nn: int, clad_t: float):
    """Returns a jitted batched (nvec_batch) -> (efficiencies, gradients) function.

    Batches over the leading axis of nvec_batch via vmap, so an entire batch of
    devices is solved in a single JAX call. Compiles once per unique
    (wavelength, period, nn, clad_t, batch_shape) tuple.
    """
    lattice, expansion, m0, m1 = _expansion_for(period_nm, nn)
    fn = lambda nvec: _forward_efficiency(
        nvec, wavelength_nm, lattice, expansion, m0, m1, clad_t,
    )
    grad_fn = jax.value_and_grad(fn)
    return jax.jit(jax.vmap(grad_fn, in_axes=0))


@functools.lru_cache(maxsize=64)
def _eff_batch_jit(wavelength_nm: float, period_nm: float, nn: int, clad_t: float):
    """Returns a jitted batched (nvec_batch) -> efficiencies function (no gradient)."""
    lattice, expansion, m0, m1 = _expansion_for(period_nm, nn)
    fn = lambda nvec: _forward_efficiency(
        nvec, wavelength_nm, lattice, expansion, m0, m1, clad_t,
    )
    return jax.jit(jax.vmap(fn, in_axes=0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def eval_eff_1d(img: np.ndarray, wavelength_nm: float,
                desired_angle_deg: float | None = None,
                nn: int | None = None,
                period_nm: float | None = None) -> float:
    """Forward efficiency for a single device. Equivalent to `Eval_Eff_1D.m`."""
    img = np.asarray(img, dtype=np.float64).flatten()
    img_01 = img / 2.0 + 0.5
    n_si = si_index(float(wavelength_nm))
    nvec = img_01 * (n_si - N_AIR) + N_AIR

    period = _resolve_period(wavelength_nm, desired_angle_deg, period_nm)
    if nn is None:
        nn = DEFAULT_FOURIER_NN
    clad_t = 0.51 * float(wavelength_nm)

    fn = _eff_jit(float(wavelength_nm), period, int(nn), clad_t)
    eff = float(fn(jnp.asarray(nvec)))
    return float(np.clip(eff, 0.0, 1.0))


def gradient_from_solver_1d(img: np.ndarray, wavelength_nm: float,
                             desired_angle_deg: float | None = None,
                             no_gradnorm: bool = False,
                             period_nm: float | None = None,
                             nn: int | None = None) -> tuple[float, np.ndarray]:
    """Efficiency + per-pixel gradient via JAX autodiff.

    Equivalent in spirit to `GradientFromSolver_1D.m`. The MATLAB version
    used a manual TM adjoint; here we let `jax.grad` differentiate the
    efficiency end-to-end, then apply the same post-processing
    (edge clip, sign clip, normalize, scale by 2).
    """
    img = np.asarray(img, dtype=np.float64).flatten()
    Nx = img.size
    img_01 = img / 2.0 + 0.5
    n_si = si_index(float(wavelength_nm))
    nvec = img_01 * (n_si - N_AIR) + N_AIR

    period = _resolve_period(wavelength_nm, desired_angle_deg, period_nm)
    if nn is None:
        nn = DEFAULT_FOURIER_NN
    clad_t = 0.51 * float(wavelength_nm)

    fn = _eff_grad_jit(float(wavelength_nm), period, int(nn), clad_t)
    eff_jnp, grad_n_jnp = fn(jnp.asarray(nvec))
    efficiency = float(np.clip(float(eff_jnp), 0.0, 1.0))
    grad_n = np.asarray(grad_n_jnp)  # d eff / d nvec  (Nx,)

    # Chain rule to img:    nvec = img_01 * (n_si - 1) + 1
    #                       d nvec / d img_01 = (n_si - 1)
    #                       d img_01 / d img = 1/2
    grad_img = grad_n * (n_si - N_AIR) * 0.5  # d eff / d img

    # Match MATLAB output range: pre-MATLAB code applied a *nvec scaling
    # before normalisation, then *2 at the end. Here we go straight from
    # autograd: just normalise and scale by 2.
    gr = grad_img.copy()

    # Sign clip: cannot increase already-saturated index, can't decrease already-min
    gr[(img_01 >= 0.999) & (gr > 0)] = 0.0
    gr[(img_01 <= 0.001) & (gr < 0)] = 0.0

    if not no_gradnorm:
        gr_max = float(np.max(np.abs(gr)))
        if gr_max > 1e-15:
            gr = gr / gr_max
    gradient = (gr * 2.0).astype(np.float64)
    return efficiency, gradient


def eval_eff_batch(imgs: np.ndarray, wavelengths: np.ndarray,
                   angles: np.ndarray | None = None,
                   periods_nm: np.ndarray | float | None = None,
                   nn: int | None = None) -> np.ndarray:
    imgs = np.asarray(imgs, dtype=np.float64)
    if imgs.ndim == 3:
        imgs = imgs.squeeze()
    B = imgs.shape[0]
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    if angles is None:
        angles = np.full(B, np.nan, dtype=np.float64)
    else:
        angles = np.asarray(angles, dtype=np.float64)
    if periods_nm is None:
        periods = np.asarray([
            period_from_angle(wavelengths[i], angles[i]) for i in range(B)
        ], dtype=np.float64)
    else:
        periods = np.broadcast_to(np.asarray(periods_nm, dtype=np.float64), (B,))
    if nn is None:
        nn = DEFAULT_FOURIER_NN

    # Fast path: shared (wavelength, angle) → vmap over batch.
    if (np.ptp(wavelengths) < 1e-9) and (np.ptp(periods) < 1e-9):
        wl = float(wavelengths[0])
        period = float(periods[0])
        clad_t = 0.51 * wl
        n_si = si_index(wl)
        img_01 = imgs / 2.0 + 0.5
        nvec_batch = img_01 * (n_si - N_AIR) + N_AIR
        fn = _eff_batch_jit(wl, period, nn, clad_t)
        effs = np.asarray(fn(jnp.asarray(nvec_batch)))
        return np.clip(effs, 0.0, 1.0).astype(np.float64)

    out = np.zeros(B, dtype=np.float64)
    for i in range(B):
        out[i] = eval_eff_1d(
            imgs[i], float(wavelengths[i]),
            None if np.isnan(angles[i]) else float(angles[i]),
            nn=int(nn), period_nm=float(periods[i]),
        )
    return out


def gradient_from_solver_batch(imgs: np.ndarray, wavelengths: np.ndarray,
                                desired_angles: np.ndarray | None = None,
                                no_gradnorm: bool = False,
                                periods_nm: np.ndarray | float | None = None,
                                nn: int | None = None) -> np.ndarray:
    """Returns (B, 1+Nx) array: column 0 is efficiency, columns 1.. are the gradient."""
    imgs = np.asarray(imgs, dtype=np.float64)
    if imgs.ndim == 3:
        imgs = imgs.squeeze()
    B, Nx = imgs.shape
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    if desired_angles is None:
        desired_angles = np.full(B, np.nan, dtype=np.float64)
    else:
        desired_angles = np.asarray(desired_angles, dtype=np.float64)
    if periods_nm is None:
        periods = np.asarray([
            period_from_angle(wavelengths[i], desired_angles[i]) for i in range(B)
        ], dtype=np.float64)
    else:
        periods = np.broadcast_to(np.asarray(periods_nm, dtype=np.float64), (B,))
    if nn is None:
        nn = DEFAULT_FOURIER_NN

    # Fast path: shared (wavelength, angle) → vmap over batch in a single JIT call.
    if (np.ptp(wavelengths) < 1e-9) and (np.ptp(periods) < 1e-9):
        wl = float(wavelengths[0])
        period = float(periods[0])
        clad_t = 0.51 * wl
        n_si = si_index(wl)
        img_01 = imgs / 2.0 + 0.5
        nvec_batch = img_01 * (n_si - N_AIR) + N_AIR

        fn = _eff_grad_batch_jit(wl, period, nn, clad_t)
        effs_jnp, grad_n_jnp = fn(jnp.asarray(nvec_batch))
        effs = np.clip(np.asarray(effs_jnp), 0.0, 1.0)
        grad_n = np.asarray(grad_n_jnp)  # (B, Nx)
        grad_img = grad_n * (n_si - N_AIR) * 0.5  # d eff / d img

        # Vectorised post-processing matching gradient_from_solver_1d
        gr = grad_img.copy()
        gr[(img_01 >= 0.999) & (gr > 0)] = 0.0
        gr[(img_01 <= 0.001) & (gr < 0)] = 0.0
        # Per-row peak normalisation (skip if disabled)
        if not no_gradnorm:
            gr_max = np.max(np.abs(gr), axis=1, keepdims=True)
            gr_max = np.where(gr_max > 1e-15, gr_max, 1.0)
            gr = gr / gr_max
        gr = gr * 2.0

        out = np.zeros((B, 1 + Nx), dtype=np.float64)
        out[:, 0] = effs
        out[:, 1:] = gr
        return out

    # Fallback: per-sample loop when wavelengths/angles vary across the batch.
    out = np.zeros((B, 1 + Nx), dtype=np.float64)
    for i in range(B):
        eff, gr = gradient_from_solver_1d(
            imgs[i], float(wavelengths[i]),
            None if np.isnan(desired_angles[i]) else float(desired_angles[i]),
            no_gradnorm=no_gradnorm, period_nm=float(periods[i]), nn=int(nn),
        )
        out[i, 0] = eff
        out[i, 1:] = gr
    return out


# ---------------------------------------------------------------------------
# Validation utilities
# ---------------------------------------------------------------------------
def fd_check(img: np.ndarray, wavelength_nm: float,
             desired_angle_deg: float | None = None,
             pixels: list[int] | None = None, h: float = 1e-3,
             period_nm: float | None = None,
             nn: int | None = None) -> dict:
    """Compare AD gradient against finite-difference at sample pixels.

    Returns a dict of arrays for inspection. Note: this checks the RAW
    `d eff / d img` gradient (without MATLAB-style normalisation/clipping),
    so it can only be compared against a freshly-computed AD gradient
    using the same chain rule.
    """
    img = np.asarray(img, dtype=np.float64).flatten()
    Nx = img.size

    img_01 = img / 2.0 + 0.5
    n_si = si_index(float(wavelength_nm))
    nvec = img_01 * (n_si - N_AIR) + N_AIR

    period = _resolve_period(wavelength_nm, desired_angle_deg, period_nm)
    if nn is None:
        nn = DEFAULT_FOURIER_NN
    clad_t = 0.51 * float(wavelength_nm)

    fn = _eff_grad_jit(float(wavelength_nm), period, int(nn), clad_t)
    eff_ad, grad_n_ad = fn(jnp.asarray(nvec))
    eff_ad = float(eff_ad)
    grad_n_ad = np.asarray(grad_n_ad)
    grad_img_ad = grad_n_ad * (n_si - N_AIR) * 0.5  # d eff / d img

    eff_fn = _eff_jit(float(wavelength_nm), period, int(nn), clad_t)

    if pixels is None:
        rng = np.random.default_rng(0)
        pixels = sorted(rng.choice(Nx, 10, replace=False).tolist())

    fd_vals = np.zeros(len(pixels))
    ad_vals = np.zeros(len(pixels))
    for i, p in enumerate(pixels):
        img_p = img.copy(); img_p[p] = np.clip(img[p] + h, -1.0, 1.0)
        img_m = img.copy(); img_m[p] = np.clip(img[p] - h, -1.0, 1.0)
        nv_p = (img_p / 2.0 + 0.5) * (n_si - N_AIR) + N_AIR
        nv_m = (img_m / 2.0 + 0.5) * (n_si - N_AIR) + N_AIR
        e_p = float(eff_fn(jnp.asarray(nv_p)))
        e_m = float(eff_fn(jnp.asarray(nv_m)))
        fd_vals[i] = (e_p - e_m) / (img_p[p] - img_m[p])
        ad_vals[i] = grad_img_ad[p]

    return {
        "pixels": np.asarray(pixels),
        "fd": fd_vals,
        "ad": ad_vals,
        "eff": eff_ad,
        "max_abs_diff": float(np.max(np.abs(fd_vals - ad_vals))),
        "rel_diff": np.abs(fd_vals - ad_vals) / (np.abs(fd_vals) + 1e-12),
    }


def validate_against_reference(test_cases_path: str | None = None) -> list[dict]:
    """Run all 10 cases and report side-by-side with MATLAB Reticolo."""
    if test_cases_path is None:
        test_cases_path = os.path.join(_HERE, "test_cases.mat")
    data = io.loadmat(test_cases_path)
    cases = data["results"]
    n = cases.shape[0]
    print(f"Validating {n} cases from {test_cases_path}\n")
    print(f"{'i':>2}  {'lam':>7}  {'ang':>5}  {'eff_eval ref':>12}  {'fmmax':>8}  "
          f"{'eff_grad ref':>12}  {'fmmax':>8}  {'gr corr':>8}")
    results = []
    for i in range(n):
        c = cases[i, 0]
        img = np.asarray(c["img"][0, 0]).flatten()
        wavelength = float(np.asarray(c["wavelength"][0, 0]).item())
        angle = float(np.asarray(c["angle"][0, 0]).item())
        eff_eval_ref = float(np.asarray(c["eff_eval"][0, 0]).item())
        eff_grad_ref = float(np.asarray(c["eff_grad"][0, 0]).item())
        gr_ref = np.asarray(c["gradient"][0, 0]).flatten()

        eff_eval = eval_eff_1d(img, wavelength, angle)
        eff_grad, gr = gradient_from_solver_1d(img, wavelength, angle)

        if np.std(gr) > 1e-12 and np.std(gr_ref) > 1e-12:
            corr = float(np.corrcoef(gr, gr_ref)[0, 1])
        else:
            corr = float("nan")

        print(f"{i:>2}  {wavelength:>7.1f}  {angle:>5.1f}  "
              f"{eff_eval_ref:>12.4f}  {eff_eval:>8.4f}  "
              f"{eff_grad_ref:>12.4f}  {eff_grad:>8.4f}  "
              f"{corr:>8.4f}")
        results.append({
            "i": i, "wavelength": wavelength, "angle": angle,
            "eff_eval_ref": eff_eval_ref, "eff_eval": eff_eval,
            "eff_grad_ref": eff_grad_ref, "eff_grad": eff_grad,
            "gr_ref": gr_ref, "gr": gr, "gr_corr": corr,
        })

    eff_diffs = [abs(r["eff_eval"] - r["eff_eval_ref"]) for r in results]
    corrs = [r["gr_corr"] for r in results if not np.isnan(r["gr_corr"])]
    print()
    print(f"Summary:")
    print(f"  eff_eval |diff| mean={np.mean(eff_diffs):.3e}  max={np.max(eff_diffs):.3e}")
    if corrs:
        print(f"  gradient corr   mean={np.mean(corrs):.4f}  min={np.min(corrs):.4f}  max={np.max(corrs):.4f}")
    return results


if __name__ == "__main__":
    validate_against_reference()
