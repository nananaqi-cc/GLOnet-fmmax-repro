"""Export FMMAX validation data for Fig 2.

Generates:
  (a) Energy conservation (T+R across Fourier orders) for the best GLOnet device
  (b) AD vs FD gradient error distribution
  (c) Efficiency convergence vs nn for the best GLOnet device

The validation device is the best-performing sign-binarized pattern from the
single-wavelength 800 nm GLOnet generator (eff = 0.750).

Output: results/outputs/fmmax_validation.mat
"""
import os
import numpy as np
import scipy.io as io
import jax
import jax.numpy as jnp
import fmmax

from fmmax_solver import (
    _expansion_for, _forward_efficiency, si_index, eval_eff_1d,
    N_AIR, N_GLASS, THICKNESS_NM, FORMULATION,
)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
jax.config.update("jax_enable_x64", True)


def _load_best_device(wavelength=800.0):
    """Load the best sign-binarized device from single-wavelength 800 nm training.

    Returns the refractive-index profile nvec (Nx,) for the patterned Si layer,
    mapped as +1 -> n_si, -1 -> N_AIR (matching fmmax_solver convention).
    """
    mat_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", "single_w800_a60_original_v3_seed4",
        "outputs", "imgs_w800_a60deg.mat",
    )
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Best device not found: {mat_path}")
    data = io.loadmat(mat_path)
    imgs = data["imgs"]          # (N, 1, pixels)
    effs = data["effs"].flatten()
    best_idx = int(np.argmax(effs))
    best_img = np.sign(imgs[best_idx]).squeeze().astype(np.float64)
    n_si = si_index(float(wavelength))
    nvec = np.where(best_img > 0, n_si, N_AIR)
    print(f"  Loaded best device: idx={best_idx}, "
          f"eff={effs[best_idx]:.4f}, "
          f"Si pixels={(best_img > 0).sum()}, "
          f"edges={int(np.sum(best_img[1:] != best_img[:-1]))}")
    return nvec


def compute_ad_fd_error():
    """Compare AD gradient vs central finite difference on a binary grating."""
    import fmmax_solver

    wavelength = 800.0
    angle = 60.0
    period = wavelength / np.sin(np.deg2rad(angle))

    # Binary grating test pattern (representative of GLOnet output)
    pixels = 256
    n_si = si_index(wavelength)
    n_sio2 = 1.45
    nvec = np.ones(pixels, dtype=np.float64) * n_sio2
    nvec[: pixels // 2] = n_si

    # AD gradient
    nn = 80
    clad_t = 0.51 * wavelength
    grad_fn = fmmax_solver._eff_grad_jit(wavelength, period, nn, clad_t)
    eff_ad, grad_ad = grad_fn(jnp.asarray(nvec))
    grad_ad = np.asarray(grad_ad, dtype=np.float64)
    print(f"  AD efficiency = {float(eff_ad):.8f}")

    # Central finite difference (subsample every 16th pixel for speed)
    eps = 1e-6
    subsample = 16
    grad_fd = np.zeros(pixels, dtype=np.float64)
    for i in range(0, pixels, subsample):
        nvec_plus = nvec.copy()
        nvec_minus = nvec.copy()
        nvec_plus[i] += eps
        nvec_minus[i] -= eps
        e_plus = eval_eff_1d(nvec_plus, wavelength, angle)
        e_minus = eval_eff_1d(nvec_minus, wavelength, angle)
        grad_fd[i] = (e_plus - e_minus) / (2 * eps)

    mask = np.zeros(pixels, dtype=bool)
    mask[::subsample] = True
    error_abs = np.abs(grad_ad[mask] - grad_fd[mask])
    error_rel = error_abs / (np.abs(grad_ad[mask]) + 1e-15)

    print(f"  |grad_AD - grad_FD| ({len(error_abs)}/{pixels} pixels):")
    print(f"    Linf = {error_abs.max():.2e}")
    print(f"    mean absolute = {error_abs.mean():.2e}")
    print(f"    mean relative = {error_rel.mean():.1e}")

    return grad_ad, grad_fd, mask, error_abs, error_rel


def compute_nn_convergence(nvec):
    """Compute efficiency vs nn for the given refractive-index profile at 800nm."""
    wavelength = 800.0
    angle = 60.0
    period = wavelength / np.sin(np.deg2rad(angle))

    nn_list = [10, 15, 20, 30, 40, 60, 80, 100, 120]
    eff_list = []

    for nn in nn_list:
        clad_t = 0.51 * wavelength
        lattice, expansion, m0, m1 = _expansion_for(period, nn)

        @jax.jit
        def fn(nv):
            return _forward_efficiency(
                nv, wavelength, lattice, expansion, m0, m1, clad_t,
            )

        eff = float(fn(jnp.asarray(nvec)))
        eff_list.append(eff)
        print(f"  nn={nn:3d}: efficiency={eff:.8f}")

    return nn_list, eff_list


def compute_energy_conservation(nvec):
    """Compute T+R across all propagating orders for the given profile."""
    wavelength = 800.0
    angle = 60.0
    period = wavelength / np.sin(np.deg2rad(angle))
    clad_t = 0.51 * wavelength

    nn_list = [10, 15, 20, 30, 40, 60, 80, 100, 120]
    t_list, r_list, tr_list = [], [], []

    for nn in nn_list:
        lattice, expansion, m0, m1 = _expansion_for(period, nn)
        num_terms = int(expansion.num_terms)

        eps_air = jnp.asarray([[N_AIR ** 2]], dtype=jnp.complex128)
        eps_si = (jnp.asarray(nvec, dtype=jnp.float64) ** 2).astype(jnp.complex128).reshape(-1, 1)
        eps_glass = jnp.asarray([[N_GLASS ** 2]], dtype=jnp.complex128)

        wl = jnp.asarray(float(wavelength))
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

        thicknesses = [jnp.asarray(float(clad_t)),
                       jnp.asarray(float(THICKNESS_NM)),
                       jnp.asarray(float(clad_t))]
        s_matrix = fmmax.stack_s_matrix([sol_glass, sol_si, sol_air], thicknesses)

        inc = jnp.zeros((2 * num_terms, 1), dtype=jnp.complex128)
        inc = inc.at[num_terms + m0, 0].set(1.0 + 0.0j)

        # S-matrix convention: s11 = transmission (fwd at start -> fwd at end)
        #                    s21 = reflection   (fwd at start -> bwd at start)
        transmitted = s_matrix.s11 @ inc
        reflected = s_matrix.s21 @ inc

        flux_t_fwd, flux_t_bwd = fmmax.directional_poynting_flux(
            transmitted, jnp.zeros_like(transmitted), sol_air
        )
        flux_r_fwd, flux_r_bwd = fmmax.directional_poynting_flux(
            reflected, jnp.zeros_like(reflected), sol_glass
        )
        flux_inc_fwd, _ = fmmax.directional_poynting_flux(
            inc, jnp.zeros_like(inc), sol_glass
        )

        total_inc = float(jnp.sum(flux_inc_fwd).real)
        T_total = float(jnp.sum(flux_t_fwd).real + jnp.sum(flux_t_bwd).real)
        R_total = float(jnp.sum(flux_r_fwd).real + jnp.sum(flux_r_bwd).real)

        T = T_total / total_inc
        R = R_total / total_inc

        t_list.append(T)
        r_list.append(R)
        tr_list.append(T + R)
        print(f"  nn={nn:3d}: T={T:.8f}, R={R:.8f}, T+R={T+R:.10f}")

    return nn_list, t_list, r_list, tr_list


def main():
    out_dir = "results/outputs"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("Loading best device (800 nm GLOnet)")
    print("=" * 60)
    nvec = _load_best_device()

    print()
    print("=" * 60)
    print("(a) Energy conservation (T+R) -- best device")
    print("=" * 60)
    ec_nn_list, t_list, r_list, tr_list = compute_energy_conservation(nvec)

    print()
    print("=" * 60)
    print("(b) AD vs FD gradient error")
    print("=" * 60)
    grad_ad, grad_fd, fd_mask, error_abs, error_rel = compute_ad_fd_error()

    print()
    print("=" * 60)
    print("(c) Efficiency convergence vs nn -- best device")
    print("=" * 60)
    nn_list, eff_list = compute_nn_convergence(nvec)

    mdict = {
        "grad_ad": grad_ad,
        "grad_fd": grad_fd,
        "fd_mask": fd_mask,
        "grad_error_abs": error_abs,
        "grad_error_rel": error_rel,
        "nn_list": np.array(nn_list, dtype=np.float64),
        "eff_vs_nn": np.array(eff_list, dtype=np.float64),
        "ec_nn_list": np.array(ec_nn_list, dtype=np.float64),
        "T_list": np.array(t_list, dtype=np.float64),
        "R_list": np.array(r_list, dtype=np.float64),
        "TR_list": np.array(tr_list, dtype=np.float64),
    }
    io.savemat(os.path.join(out_dir, "fmmax_validation.mat"), mdict)
    print(f"\nSaved to {out_dir}/fmmax_validation.mat")


if __name__ == "__main__":
    main()
