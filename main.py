"""Core pipeline for "Towards Optimized Casimir Simulations".

Runs the reference experiments and writes every number they produce to
results/summary.json (plus figures):

    1  Hamiltonian construction and verification (commutators, colouring)
    2  Classical reference: sector ED against the penalised full-space
       Hamiltonian, and the Gauss-law penalty scan
    3  Structured Trotter step: error scaling and measured resources
    4  Single-ancilla ground-state filter (RQSVT) and its resources against
       a block-encoded QSVT baseline
    5  Casimir energy against plate separation
    6  DMRG cross-check

    python main.py            full run (about two minutes)
    python main.py --quick    small lattices only (about a minute)
    python main.py --gui      open the interactive explorer instead

The research studies built on this pipeline are in experiments.py, the
hardware experiments in hardware.py.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.sparse.linalg import eigsh

from casimir import plot_scan, run_casimir_scan
from circuit import trotter_error
from config import SimulationConfig, environment
from lattice import LatticeGrid
from QuantumFields.gauge_field import GaussLaw, QuantumLinkModel
from resources import ground_state_comparison, trotter_step_table
from rqsvt import RQSVTGroundState
from sector import GaugeSector


def banner(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def l1(op) -> float:
    return float(np.sum(np.abs(op.simplify(atol=1e-12).coeffs)))


# ---------------------------------------------------------------------------

def verify_hamiltonian(cfg) -> dict:
    """[G_j, H] = 0 at every site (plates included) and [H_E^uniform, H_B] = 0."""
    banner("1  Hamiltonian construction and verification")
    lattice = LatticeGrid(cfg.width, cfg.height)
    lattice.debug()
    lattice.visualize_ascii()
    out = {'qubits': lattice.n_qubits, 'qubits_with_ancilla': lattice.n_qubits + 1}

    model = QuantumLinkModel(lattice, g=1.3, J=cfg.J, rk=0.7, field_pattern="staggered")
    H = model.build_physical_hamiltonian()
    gauss = GaussLaw(lattice, plates="fixed",
                     plate_charges={(i, j): 0.5 for j in (0, lattice.height - 1)
                                    for i in range(lattice.width)})
    worst = 0.0
    for site in range(lattice.n_sites):
        G = gauss.divergence_operator(*lattice.site_to_coords(site))
        worst = max(worst, l1((G @ H) - (H @ G)))
    out['max_commutator_l1'] = worst
    print(f"\nmax_j ||[G_j, H_E + H_B + H_RK]||_1 over all {lattice.n_sites} sites = {worst:.2e}")

    uniform = QuantumLinkModel(lattice, g=1.0, J=1.0)
    HE, HB = uniform.build_electric_term(), uniform.build_magnetic_term()
    out['uniform_field_commutator_l1'] = l1((HE @ HB) - (HB @ HE))
    stag = QuantumLinkModel(lattice, g=1.0, J=1.0, field_pattern="staggered")
    HEs = stag.build_electric_term()
    out['staggered_field_commutator_l1'] = l1((HEs @ HB) - (HB @ HEs))
    print(f"||[H_E(uniform),   H_B]||_1 = {out['uniform_field_commutator_l1']:.2e}"
          "   (conserved: no electric dynamics in the pure-gauge model)")
    print(f"||[H_E(staggered), H_B]||_1 = {out['staggered_field_commutator_l1']:.2e}")
    print(f"plaquette sublayers: checkerboard valid = "
          f"{lattice.sublayers_are_disjoint(lattice.get_plaquette_sublayers('checkerboard'))}, "
          f"{lattice.n_plaquette_colours}-colouring valid = "
          f"{lattice.sublayers_are_disjoint(lattice.get_plaquette_sublayers())}")
    out['checkerboard_valid'] = lattice.sublayers_are_disjoint(
        lattice.get_plaquette_sublayers('checkerboard'))
    return out


def classical_reference(cfg) -> dict:
    """Sector ED against the penalised Hamiltonian; energy and violation against the penalty."""
    banner("2  Classical reference and Gauss-law penalty scan")
    out = {'lattices': [], 'penalty_scan': []}
    for (W, H) in [(2, 2), (cfg.width, cfg.height)]:
        lattice = LatticeGrid(W, H)
        sector = GaugeSector(lattice)
        gs = sector.ground_state(g=cfg.g, J=cfg.J, rk=cfg.rk,
                                 field_pattern=cfg.field_pattern)
        violation = GaussLaw(lattice).max_violation(sector.embed(gs['state']))
        energies = []
        for seed in range(3):                      # three random starts: eigensolver spread
            rng = np.random.default_rng(cfg.seed + seed)
            Hfull = QuantumLinkModel(lattice, g=cfg.g, J=cfg.J, rk=cfg.rk,
                                     gauss_penalty=cfg.penalty,
                                     field_pattern=cfg.field_pattern
                                     ).build_hamiltonian().to_matrix(sparse=True)
            energies.append(float(eigsh(Hfull, k=1, which='SA',
                                        v0=rng.standard_normal(Hfull.shape[0]),
                                        return_eigenvectors=False)[0]))
        row = {'W': W, 'H': H, 'qubits': lattice.n_qubits,
               'hilbert_dim': 2 ** lattice.n_qubits, 'gauge_sector_dim': sector.dim,
               'charge_sector_dim': int(len(gs['support'])),
               'E0_sector': gs['energy'], 'gap_in_sector': gs['gap'],
               'E0_full_space': energies[0],
               'eigsh_spread': float(np.ptp(energies)),
               'gauss_violation': violation,
               'plate_charges': None if gs['plate_charges'] is None
               else gs['plate_charges'].tolist()}
        out['lattices'].append(row)
        print(f"{W}x{H}: 2^{lattice.n_qubits} = {row['hilbert_dim']} qubit states, "
              f"{sector.dim} gauge invariant, {row['charge_sector_dim']} in the ground-state "
              f"plate-charge sector")
        print(f"     E0 sector {gs['energy']:.10f} | full space {energies[0]:.10f} | "
              f"eigsh spread {row['eigsh_spread']:.1e} | gap {gs['gap']:.6f} | "
              f"max ||G_j psi|| = {violation:.1e}")

    lattice = LatticeGrid(cfg.width, cfg.height)
    E_ref = out['lattices'][-1]['E0_sector']
    print(f"\nPenalty scan on {cfg.width}x{cfg.height} (sector E0 = {E_ref:.8f}):")
    for lam in cfg.penalty_scan:
        Hfull = QuantumLinkModel(lattice, g=cfg.g, J=cfg.J, rk=cfg.rk, gauss_penalty=lam,
                                 field_pattern=cfg.field_pattern).build_hamiltonian()
        w, v = eigsh(Hfull.to_matrix(sparse=True), k=1, which='SA')
        viol = GaussLaw(lattice).max_violation(v[:, 0])
        one_norm = float(np.sum(np.abs(Hfull.coeffs)))
        out['penalty_scan'].append({'penalty': lam, 'E0': float(w[0]),
                                    'violation': viol, 'pauli_one_norm': one_norm})
        print(f"  lambda={lam:5d}  E0={w[0]:+.8f}  diff={abs(w[0] - E_ref):.1e}  "
              f"max||G psi||={viol:.1e}  Pauli 1-norm={one_norm:9.1f}")
    return out


def trotter_analysis(cfg, quick) -> dict:
    """Error order of the structured step and its gate counts against lattice size."""
    banner("3  Structured Trotter decomposition")
    out = {'error_scaling': [], 'resources': []}
    lattice = LatticeGrid(2, 3)
    for order in (1, 2):
        errs = [trotter_error(lattice, 0.0, cfg.J, dt, order=order) for dt in cfg.dt_scan]
        rate = float(np.polyfit(np.log(cfg.dt_scan), np.log(errs), 1)[0])
        out['error_scaling'].append({'order': order, 'dt': list(cfg.dt_scan),
                                     'step_error': errs, 'fitted_power': rate})
        print(f"order {order}: one-step error {['%.2e' % e for e in errs]} "
              f"-> dt^{rate:.2f}  (global error dt^{rate - 1:.2f})")
    sizes = [(2, 2), (2, 3), (3, 3), (4, 3)] if quick else \
        [(2, 2), (2, 3), (3, 3), (4, 3), (3, 5), (4, 4), (6, 4), (8, 6)]
    print(f"\n{'lattice':>8} {'N_q':>4} {'layers':>6} | {'2q gates':>8} {'2q depth':>8} "
          f"{'depth':>6} | ladders 2q/depth | generic 2q/depth")
    for row in trotter_step_table(sizes, g=cfg.g, J=cfg.J):
        s, gen = row['structured'], row.get('generic')
        out['resources'].append(row)
        lad = row['structured_pauli']
        tail = f"{lad['two_qubit_gates']:6d} / {lad['two_qubit_depth']:4d}   | " + (
            f"{gen['two_qubit_gates']:6d} / {gen['two_qubit_depth']:5d}" if gen else "     -")
        print(f"{row['W']:>5}x{row['H']:<2} {row['qubits']:4d} {row['sublayers']:6d} | "
              f"{s['two_qubit_gates']:8d} {s['two_qubit_depth']:8d} {s['depth']:6d} | {tail}")
    return out


def rqsvt_analysis(cfg, quick) -> dict:
    """Filter accuracy (exact, Trotter, qDRIFT signal operators) and gate counts."""
    banner("4  Single-ancilla ground-state filter (RQSVT)")
    out = {}
    lattice = LatticeGrid(cfg.width, cfg.height)
    for kind in ("step", "gaussian"):
        rq = RQSVTGroundState(lattice, g=cfg.g, J=cfg.J, rk=cfg.rk,
                              field_pattern=cfg.field_pattern, filter_kind=kind)
        info = rq.summary()
        print(f"\n[{kind} filter]  {info['qubits']} qubits, lambda_H={info['lambda_H']:.1f}, "
              f"gap={info['gap']:.4f}, Fourier degree D={info['fourier_degree']}, "
              f"{info['signal_operators']} signal operators, "
              f"initial overlap={info['initial_overlap']:.4f}")
        exact = rq.run("exact")
        print(f"  exact signal op : E={exact['energy']:.8f}  err={exact['energy'] - rq.E0:+.1e}  "
              f"overlap={exact['overlap']:.6f}  P(success)={exact['success_prob']:.3f}")
        trot = rq.estimate_energy("trotter", (1, 2, 4))
        for r in trot['runs']:
            print(f"  trotter n={r['n']:<4d}: E={r['energy']:.8f}  err={r['energy'] - rq.E0:+.1e}  "
                  f"overlap={r['overlap']:.6f}")
        print(f"  Richardson      : E={trot['energy_richardson']:.8f}  err={trot['error']:+.1e}")
        schedule = (100, 200) if quick else (200, 400, 800)
        qd = rq.estimate_energy("qdrift", schedule, n_circuits=4 if quick else 8,
                                seed=cfg.seed)
        for r in qd['runs']:
            print(f"  qdrift  N={r['n']:<4d}: E={r['energy']:.8f}  err={r['energy'] - rq.E0:+.1e}  "
                  f"overlap={r['overlap']:.6f}")
        print(f"  Richardson      : E={qd['energy_richardson']:.8f}  err={qd['error']:+.1e}")
        comp = ground_state_comparison(rq, trotter_steps=1, qdrift_samples=schedule[-1])
        t, q, b = comp['gqsp_trotter'], comp['gqsp_qdrift'], comp['block_encoded_qsvt']
        print(f"  two-qubit gates : GQSP+Trotter(n=1) {t['two_qubit_gates']:,} "
              f"(Pauli ladders: {comp['gqsp_trotter_pauli_ladders']['two_qubit_gates']:,}) on {t['qubits']} qubits"
              f" | GQSP+qDRIFT(N={schedule[-1]}) {q['two_qubit_gates']:,} on {q['qubits']} qubits")
        print(f"                    block-encoded QSVT ~{b['two_qubit_gates']:,} on {b['qubits']} qubits"
              f" (unary iteration ~{b['two_qubit_gates_unary_estimate']:,} on {b['qubits_unary']})")
        out[kind] = {'summary': info,
                     'exact': {k: v for k, v in exact.items() if k != 'state'},
                     'trotter': trot, 'qdrift': qd, 'resources': comp}
    return out


def casimir_analysis(cfg, quick) -> dict:
    """E0(d) scans at every width in the configuration, plus lam_RK = 0.5 at W = 4."""
    banner("5  Casimir energy against plate separation")
    out = {}
    os.makedirs("results", exist_ok=True)
    for W, seps in cfg.casimir_widths.items():
        if quick:
            seps = tuple(d for d in seps if W * (2 * d + 1) <= 30)
        scan = run_casimir_scan(W, seps, g=cfg.g, J=1.0, rk=cfg.rk,
                                field_pattern=cfg.field_pattern)
        if 'analysis' in scan:
            plot_scan(scan, f"results/casimir_W{W}.png")
        out[f"W{W}"] = scan
    if not quick:
        scan = run_casimir_scan(4, tuple(range(1, 8)), J=1.0, rk=0.5)
        plot_scan(scan, "results/casimir_W4_rk0.5.png")
        out["W4_rk0.5"] = scan
    return out


def dmrg_analysis(cfg) -> dict:
    """DMRG against sector ED, and MPS compressibility of exact ground states."""
    banner("6  DMRG cross-check")
    from tensor_network import run_compression_scan, validate_dmrg
    out = {'validation': [validate_dmrg(LatticeGrid(2, 3), J=cfg.J, plates="fixed"),
                          validate_dmrg(LatticeGrid(cfg.width, cfg.height), J=cfg.J)]}
    out['compression'] = run_compression_scan([(2, 3), (3, 3)])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="small lattices only")
    parser.add_argument("--skip-dmrg", action="store_true", help="omit the DMRG cross-check")
    parser.add_argument("--gui", action="store_true", help="open the interactive explorer")
    args = parser.parse_args()

    if args.gui:
        from visualize import launch
        launch()
        return

    cfg = SimulationConfig()
    start = time.time()
    summary = {'environment': environment(),
               'config': {k: v for k, v in vars(cfg).items() if k != 'casimir_widths'}}
    summary['verification'] = verify_hamiltonian(cfg)
    summary['classical'] = classical_reference(cfg)
    summary['trotter'] = trotter_analysis(cfg, args.quick)
    summary['rqsvt'] = rqsvt_analysis(cfg, args.quick)
    summary['casimir'] = casimir_analysis(cfg, args.quick)
    if not args.skip_dmrg:
        summary['dmrg'] = dmrg_analysis(cfg)
    summary['runtime_seconds'] = time.time() - start

    os.makedirs("results", exist_ok=True)
    with open("results/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: o.tolist()
                  if hasattr(o, "tolist") else float(o))
    print(f"\nDone in {summary['runtime_seconds']:.0f} s. Numbers: results/summary.json")


if __name__ == "__main__":
    main()
