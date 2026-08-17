"""Energy-conservation check for the corrected fixed-period FMMAX stack."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import fmmax
import jax.numpy as jnp
import numpy as np
from scipy.io import loadmat

import fmmax_solver


def energy_balance(pattern: np.ndarray, wavelength: float, period_nm: float,
                   nn: int) -> dict:
    n_si = fmmax_solver.si_index(wavelength)
    nvec = np.where(pattern > 0, n_si, fmmax_solver.N_AIR)
    lattice, expansion, m0, _ = fmmax_solver._expansion_for(period_nm, nn)
    num_terms = int(expansion.num_terms)
    common = dict(
        wavelength=jnp.asarray(wavelength),
        in_plane_wavevector=jnp.asarray([0.0, 0.0]),
        primitive_lattice_vectors=lattice,
        expansion=expansion,
        formulation=fmmax_solver.FORMULATION,
    )
    sol_glass = fmmax.eigensolve_isotropic_media(
        permittivity=jnp.asarray([[fmmax_solver.N_GLASS**2]], dtype=jnp.complex128), **common
    )
    sol_pattern = fmmax.eigensolve_isotropic_media(
        permittivity=(jnp.asarray(nvec)**2).astype(jnp.complex128).reshape(-1, 1), **common
    )
    sol_air = fmmax.eigensolve_isotropic_media(
        permittivity=jnp.asarray([[fmmax_solver.N_AIR**2]], dtype=jnp.complex128), **common
    )
    stack = fmmax.stack_s_matrix(
        [sol_glass, sol_pattern, sol_air],
        [0.51*wavelength, fmmax_solver.THICKNESS_NM, 0.51*wavelength],
    )
    incident = jnp.zeros((2*num_terms, 1), dtype=jnp.complex128)
    incident = incident.at[num_terms+m0, 0].set(1.0+0.0j)
    transmitted = stack.s11 @ incident
    reflected = stack.s21 @ incident
    incident_forward, _ = fmmax.directional_poynting_flux(
        incident, jnp.zeros_like(incident), sol_glass
    )
    transmitted_forward, transmitted_backward = fmmax.directional_poynting_flux(
        transmitted, jnp.zeros_like(transmitted), sol_air
    )
    reflected_forward, reflected_backward = fmmax.directional_poynting_flux(
        jnp.zeros_like(reflected), reflected, sol_glass
    )
    incident_power = float(jnp.sum(incident_forward))
    transmission = float(jnp.sum(transmitted_forward) / incident_power)
    reflection = float(-jnp.sum(reflected_backward) / incident_power)
    spurious = float(
        (jnp.sum(jnp.abs(transmitted_backward)) + jnp.sum(jnp.abs(reflected_forward)))
        / incident_power
    )
    return {
        "nn": nn,
        "transmission": transmission,
        "reflection": reflection,
        "T_plus_R": transmission+reflection,
        "directional_spurious_flux": spurious,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period_nm", type=float, default=fmmax_solver.FIXED_PERIOD_NM)
    parser.add_argument("--wavelength", type=float, default=800.0)
    parser.add_argument("--output", default="results/revision_fixed/validation/energy_conservation.json")
    args = parser.parse_args()
    data = loadmat(
        "results/multi_w700_w800_w900_a60_original_v1_seed5/outputs/"
        "imgs_multi_wl_a60deg.mat"
    )
    pattern = np.sign(data["imgs"][0, 0, :])
    records = [
        energy_balance(pattern, args.wavelength, args.period_nm, nn)
        for nn in (10, 14, 20, 30, 40, 60, 80)
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(records, stream, indent=2)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
