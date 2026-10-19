"""Benchmark against Joshi et al. (2026), arXiv:2604.07436.

Joshi et al. ran the structured Trotter decomposition on a trapped-ion computer
for the U(1) quantum link model with hardcore-boson matter on open lattices
(4x3 with J = 0, 5x4 with J = 2; kappa = 1, m = 3, g = 6 on resonance). This
module puts the circuits and classical references of this package on exactly
that problem, which validates the parts of the thesis that are carried over
from their work:

  resources       per-step gate counts of circuit.MatterTrotter, transpiled to
                  {cx, rz, sx, x} and to the native-ZZ basis {rzz, rz, sx, x},
                  next to Table S1 of Joshi et al. (all-to-all connectivity
                  in both cases, as on trapped ions)
  string_dynamics exact sector evolution of their string initial state
                  (matter_sector.MatterSector), with the Trotterised circuit
                  emulated in the same basis for chosen time steps
  plaquette_effect  the qualitative claim of their Fig. 2 and 4: with J != 0
                  pairs are created away from the string, with J = 0 they are
                  not; quantified by the particle density at distance >= 2
                  from the initial string.

    python joshi_benchmark.py            all three, results/joshi_benchmark.json + .png
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy.sparse.linalg import expm_multiply

from circuit import ZZ_BASIS_GATES, MatterTrotter, circuit_resources
from config import environment
from dynamics import SectorTrotter
from lattice import OpenLattice
from matter_sector import MatterSector
from QuantumFields.matter_field import staircase_string, string_state_bits

# Table S1 of Joshi et al.: gate counts for one Trotter step on Quantinuum H2.
JOSHI_TABLE_S1 = {
    (5, 4): {'J': 2.0, 'qubits': 51, 'one_qubit_gates': 682, 'two_qubit_gates': 385,
             'two_qubit_depth': 28, 'depth': 159},
    (4, 3): {'J': 0.0, 'qubits': 29, 'one_qubit_gates': 222, 'two_qubit_gates': 119,
             'two_qubit_depth': 28, 'depth': 137},
}
JOSHI_PARAMS = {'kappa': 1.0, 'm': 3.0, 'g': 6.0}


def resources(dt: float = 0.1, verbose: bool = True) -> list:
    """Our per-step resources against Table S1 of Joshi et al."""
    rows = []
    for (W, H), ref in JOSHI_TABLE_S1.items():
        lat = OpenLattice(W, H)
        step = MatterTrotter(lat, J=ref['J'], dt=dt, **JOSHI_PARAMS).build_step()
        row = {'lattice': f"{W}x{H}", 'J': ref['J'], 'joshi': ref,
               'cx_basis': circuit_resources(step),
               'zz_basis': circuit_resources(step, basis_gates=ZZ_BASIS_GATES),
               'hopping_terms': lat.n_links, 'plaquettes': lat.n_plaquettes}
        rows.append(row)
        if verbose:
            c, z = row['cx_basis'], row['zz_basis']
            print(f" {W}x{H} J={ref['J']:.0f}: {c['qubits']} qubits | two-qubit gates per step: "
                  f"this work {c['two_qubit_gates']} (native-ZZ basis {z['two_qubit_gates']}), "
                  f"Joshi et al. {ref['two_qubit_gates']} | two-qubit depth: this work "
                  f"{c['two_qubit_depth']}, Joshi et al. {ref['two_qubit_depth']}")
    if verbose:
        print("   (single-qubit counts are not compared: the native one-qubit gate sets differ)")
    return rows


def _site_distance_to_string(lattice, string_links) -> np.ndarray:
    """Manhattan distance of every site from the nearest site on the string."""
    on_string = set()
    for link in string_links:
        a, b = lattice.link_endpoints(link)
        on_string.update([a, b])
    dist = []
    for s in range(lattice.n_sites):
        i, j = lattice.site_coords(s)
        dist.append(min(abs(i - a) + abs(j - b) for (a, b) in on_string))
    return np.array(dist)


def string_dynamics(W=4, H=3, J=0.0, t_max=3.0, n_times=61, trotter=((0.1, 1), (0.1, 2)),
                    kappa=1.0, m=3.0, g=6.0, verbose=True) -> dict:
    """Exact dynamics of the string state, with emulated Trotter circuits."""
    lat = OpenLattice(W, H)
    t0 = time.time()
    sector = MatterSector(lat)
    Hm = sector.hamiltonian(kappa, m, g, J)
    string = staircase_string(lat)
    psi0 = sector.product_state(string_state_bits(lat, string))
    dist = _site_distance_to_string(lat, string)
    times = np.linspace(0.0, t_max, n_times)
    dt_out = times[1] - times[0]

    string_sz = sector.link_sz()[:, string]                    # -1/2 while the string is intact
    intact = np.all(string_sz < 0, axis=1).astype(float)

    def observables(psi):
        prob = np.abs(psi) ** 2
        dens = sector.particle_density(psi)
        return {'particles': float(dens.sum()),
                'particles_off_string': float(dens[dist >= 1].sum()),
                'particles_far': float(dens[dist >= 2].sum()),
                'string_intact': float(prob @ intact),
                'density': dens}

    psi = psi0.copy()
    series = {k: [] for k in ('particles', 'particles_off_string', 'particles_far',
                              'string_intact')}
    density = []
    for k in range(n_times):
        if k:
            psi = expm_multiply(-1j * dt_out * Hm, psi)
        obs = observables(psi)
        for key in series:
            series[key].append(obs[key])
        density.append(obs['density'].tolist())
    out = {'lattice': f"{W}x{H}", 'qubits': lat.n_qubits, 'sector_dim': sector.dim,
           'kappa': kappa, 'm': m, 'g': g, 'J': J, 'string_links': string,
           'times': times.tolist(), **series, 'density': density,
           'site_distance_to_string': dist.tolist(), 'trotter': [],
           'seconds_exact': time.time() - t0}

    for dt, order in trotter:
        n_steps = int(round(t_max / dt))
        emu = SectorTrotter.for_matter(sector, kappa, m, g, J, dt, order)
        psi_t, psi_e = psi0.copy(), psi0.copy()
        rec = {'dt': dt, 'order': order, 'times': [], 'fidelity': [], 'particles': []}
        for n in range(n_steps + 1):
            if n:
                psi_t = emu.step(psi_t)
                psi_e = expm_multiply(-1j * dt * Hm, psi_e)
            rec['times'].append(n * dt)
            rec['fidelity'].append(float(abs(np.vdot(psi_e, psi_t)) ** 2))
            rec['particles'].append(float(sector.particle_density(psi_t).sum()))
        out['trotter'].append(rec)
    if verbose:
        msg = (f" {W}x{H} J={J}: sector dim {sector.dim:,} ({lat.n_qubits} qubits), "
               f"max particles {max(series['particles']):.3f}, "
               f"string intact at t={t_max}: {series['string_intact'][-1]:.3f}, "
               f"time-averaged particles off the string {np.mean(series['particles_off_string']):.4f}"
               f" (at distance >= 2: {np.mean(series['particles_far']):.4f})")
        for rec in out['trotter']:
            msg += f" | Trotter dt={rec['dt']} o{rec['order']} F(t_max)={rec['fidelity'][-1]:.4f}"
        print(msg + f"  [{time.time() - t0:.0f} s]")
    return out


def plaquette_effect(W=5, H=4, Js=(0.0, 2.0), t_max=3.0, n_times=31, verbose=True) -> dict:
    """Pair creation away from the string with and without the plaquette term."""
    runs = [string_dynamics(W, H, J=J, t_max=t_max, n_times=n_times, trotter=(),
                            verbose=verbose) for J in Js]
    return {'lattice': f"{W}x{H}", 'runs': runs}


def plot(result, path="results/joshi_benchmark.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    rows = result['resources']
    labels = [r['lattice'] + f" (J={r['J']:.0f})" for r in rows]
    x = np.arange(len(rows))
    axes[0].bar(x - 0.2, [r['joshi']['two_qubit_gates'] for r in rows], 0.4,
                color="0.6", label="Joshi et al., Table S1")
    axes[0].bar(x + 0.2, [r['cx_basis']['two_qubit_gates'] for r in rows], 0.4,
                color="#1f4e79", label="this work (exact two-level gates)")
    for k, r in enumerate(rows):
        axes[0].text(k - 0.2, r['joshi']['two_qubit_gates'], str(r['joshi']['two_qubit_gates']),
                     ha="center", va="bottom", fontsize=8)
        axes[0].text(k + 0.2, r['cx_basis']['two_qubit_gates'],
                     f"{r['cx_basis']['two_qubit_gates']} (depth {r['cx_basis']['two_qubit_depth']})",
                     ha="center", va="bottom", fontsize=8)
    axes[0].set(xticks=x, xticklabels=labels, ylabel="two-qubit gates per Trotter step",
                title="(a) one Trotter step of the matter-coupled model")
    axes[0].legend(fontsize=8)

    sd = result['string_dynamics']
    axes[1].plot(sd['times'], sd['particles'], color="#c0392b", label="particles (exact)")
    axes[1].plot(sd['times'], sd['string_intact'], color="#1f4e79",
                 label="string fully intact (exact)")
    for rec in sd['trotter']:
        axes[1].plot(rec['times'], rec['particles'], 'o', ms=3,
                     label=f"particles, Trotter dt={rec['dt']} order {rec['order']}")
    axes[1].set(xlabel="time (1/kappa)", title=f"(b) string breaking, {sd['lattice']}, "
                f"m={sd['m']:.0f}, g={sd['g']:.0f}, J={sd['J']:.0f}")
    axes[1].legend(fontsize=7.5)
    axes[1].grid(alpha=0.3)

    for run, colour in zip(result['plaquette_effect']['runs'], ("#7f8c8d", "#8e44ad")):
        axes[2].plot(run['times'], run['particles_far'], color=colour, lw=2,
                     label=f"J = {run['J']:.0f}")
    axes[2].set(xlabel="time (1/kappa)", ylabel="particles at distance >= 2 from string",
                title=f"(c) pair creation away from the string, "
                      f"{result['plaquette_effect']['lattice']}")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def run_all(verbose=True, big_lattice=(5, 4)) -> dict:
    os.makedirs("results", exist_ok=True)
    out = {'environment': environment(), 'resources': resources(verbose=verbose),
           'string_dynamics': string_dynamics(4, 3, J=0.0, verbose=verbose),
           'plaquette_effect': plaquette_effect(*big_lattice, verbose=verbose)}
    with open("results/joshi_benchmark.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    plot(out)
    return out


if __name__ == "__main__":
    run_all()
    print(" -> results/joshi_benchmark.json, results/joshi_benchmark.png")
