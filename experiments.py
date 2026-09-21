"""Research experiments beyond the core pipeline (thesis Sec. 4.3).

    python experiments.py rk-scan          decay of the Casimir residual against lam_RK
    python experiments.py filter-scaling   cost and accuracy of the filter against d
    python experiments.py noise            filter under two-qubit depolarising noise
    python experiments.py all

Each experiment prints a table, writes results/<name>.json and a figure.
All of them are functions, so they can also be driven from a notebook with
other parameters.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from casimir import analyse, ground_energy_sector
from lattice import LatticeGrid
from noise import noisy_filter_study
from rqsvt import RQSVTGroundState

XI_GRID_MAX = 12.0


def _save(name, payload):
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}.json", "w") as fh:
        json.dump(payload, fh, indent=2,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# 1. Rokhsar-Kivelson scan
# ---------------------------------------------------------------------------

def rk_scan(width=4, rks=(-0.5, -0.25, 0.0, 0.25, 0.5, 0.7, 0.8, 0.9),
            separations=tuple(range(1, 8)), J=1.0, verbose=True) -> dict:
    """E0(d) for each lam_RK; decay length xi, power-law exponent, D2 and gaps.

    lam_RK = J is the Rokhsar-Kivelson point, where E0 = 0 for every d and the
    gap closes; approaching it the correlation length, and with it the range of
    the Casimir residual, should grow.
    """
    rows = []
    for rk in rks:
        data = [ground_energy_sector(width, d + 1, J=J, rk=rk) for d in separations]
        an = analyse(separations, [r['energy'] for r in data])
        xi, alpha = an['exponential']['xi'], an['power_law']['alpha']
        d2 = np.array(an['second_difference'])
        row = {'rk': rk, 'xi': xi, 'xi_at_bound': bool(xi >= XI_GRID_MAX - 1e-6 or xi <= 0.11),
               'alpha': alpha, 'alpha_at_bound': bool(alpha <= 1.21 or alpha >= 8.99),
               'sse_exponential': an['exponential']['sse'], 'sse_power': an['power_law']['sse'],
               'D2': d2.tolist(), 'smooth_attractive': bool(np.all(d2 < 0)),
               'gap_min': data[-1]['gap'], 'gaps': [r['gap'] for r in data],
               'E0': [r['energy'] for r in data]}
        rows.append(row)
        if verbose:
            better = "exp" if row['sse_exponential'] < row['sse_power'] else "power"
            print(f" lam_RK={rk:+.2f}  xi={xi:5.2f}{'*' if row['xi_at_bound'] else ' '} "
                  f"alpha={alpha:4.2f}{'*' if row['alpha_at_bound'] else ' '} "
                  f"SSE exp/pow={row['sse_exponential']:.1e}/{row['sse_power']:.1e} ({better})  "
                  f"all D2<0: {row['smooth_attractive']}  gap(d={separations[-1]})={row['gap_min']:.3f}")
    if verbose:
        print(" (* = fit parameter at the edge of its search range: not determined)")
    out = {'width': width, 'J': J, 'separations': list(separations), 'rows': rows}
    _save("rk_scan", out)

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ds = np.array(separations[1:-1])
    for row in rows:
        d2 = -np.array(row['D2'])
        ok = d2 > 0
        axes[0].semilogy(ds[ok], d2[ok], 'o-', ms=4, label=f"$\\lambda_{{RK}}$={row['rk']:+.2f}")
    axes[0].set(xlabel="plate separation d", ylabel="$-D_2(d)$ (attractive part)",
                title=f"Second difference of $E_0(d)$, W={width}")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3, which="both")
    good = [r for r in rows if not r['xi_at_bound']]
    axes[1].plot([r['rk'] for r in good], [r['xi'] for r in good], 'o-', color="#c0392b",
                 label="decay length $\\xi$ (exponential fit)")
    ax2 = axes[1].twinx()
    ax2.plot([r['rk'] for r in rows], [1.0 / r['gap_min'] for r in rows], 's--',
             color="#1e8449", label=f"1 / gap at d={separations[-1]}")
    axes[1].set(xlabel="$\\lambda_{RK}$ / J", ylabel="$\\xi$ (lattice spacings)",
                title="Range of the Casimir residual")
    ax2.set_ylabel("1 / gap", color="#1e8449")
    axes[1].legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="lower right")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/rk_scan.png", dpi=160)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 2. Filter cost against plate separation
# ---------------------------------------------------------------------------

def filter_scaling(width=4, separations=(1, 2, 3, 4), J=1.0, rk=0.0,
                   schedule=(1, 2, 4), verbose=True) -> dict:
    """The quantum algorithm as a function of d: degree, overlap, accuracy, gates."""
    rows = []
    for d in separations:
        t = time.time()
        rq = RQSVTGroundState(LatticeGrid(width, d + 1), J=J, rk=rk, filter_kind="gaussian")
        info = rq.summary()
        est = rq.estimate_energy("trotter", schedule)
        res = rq.resources("trotter", n_steps=1)
        row = {'d': d, 'qubits': info['qubits'], 'sector_dim': info['sector_dim'],
               'gap': info['gap'], 'lambda_H': info['lambda_H'],
               'signal_operators': info['signal_operators'],
               'initial_overlap': info['initial_overlap'],
               'success_prob': est['runs'][0]['success_prob'],
               'errors': [r['energy'] - rq.E0 for r in est['runs']],
               'overlaps': [r['overlap'] for r in est['runs']],
               'richardson_error': est['error'], 'E0': rq.E0,
               'E_rqsvt': est['energy_richardson'],
               'two_qubit_gates_n1': res['two_qubit_gates'],
               'seconds': time.time() - t}
        rows.append(row)
        if verbose:
            print(f" d={d}  {row['qubits']:2d} qubits  gap={row['gap']:.3f}  "
                  f"signal ops={row['signal_operators']:4d}  overlap0={row['initial_overlap']:.4f}  "
                  f"P(success)={row['success_prob']:.4f}  err(n=1)={row['errors'][0]:+.1e}  "
                  f"Richardson={row['richardson_error']:+.1e}  2q gates={row['two_qubit_gates_n1']:,}"
                  f"  [{row['seconds']:.0f}s]")
    out = {'width': width, 'J': J, 'rk': rk, 'schedule': list(schedule), 'rows': rows}
    if len(rows) >= 3:
        x = np.log([r['d'] for r in rows[1:]])
        out['gate_growth_exponent'] = float(np.polyfit(
            x, np.log([r['two_qubit_gates_n1'] for r in rows[1:]]), 1)[0])
        if verbose:
            print(f" two-qubit gates grow as d^{out['gate_growth_exponent']:.2f} "
                  "(expected about d^3: lambda_H/gap ~ d^2 signal operators, each with ~ d plaquettes)")
    _save("filter_scaling", out)

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ds = [r['d'] for r in rows]
    axes[0].loglog(ds, [r['two_qubit_gates_n1'] for r in rows], 'o-', color="#1f4e79",
                   label="two-qubit gates (n = 1)")
    axes[0].loglog(ds, [1.0 / r['success_prob'] for r in rows], 's--', color="#ca6f1e",
                   label="1 / P(success): repetitions")
    axes[0].set(xlabel="plate separation d", title=f"Cost of the filter, W={width}")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")
    for k, n in enumerate(schedule):
        axes[1].semilogy(ds, [abs(r['errors'][k]) for r in rows], 'o-', label=f"Trotter n={n}")
    axes[1].semilogy(ds, [abs(r['richardson_error']) for r in rows], '*-', ms=10, color="k",
                     label="Richardson")
    axes[1].set(xlabel="plate separation d", ylabel="$|E - E_0|$", title="Accuracy")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("results/filter_scaling.png", dpi=160)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 3. Gate noise
# ---------------------------------------------------------------------------

def noise_study(lattices=((2, 2), (2, 3)), J=2.0,
                error_rates=(0.0, 1e-4, 3e-4, 1e-3, 3e-3), trajectories=400,
                verbose=True) -> dict:
    out = {'studies': []}
    for (W, H) in lattices:
        rq = RQSVTGroundState(LatticeGrid(W, H), J=J, plates="fixed", filter_kind="gaussian")
        shots = trajectories if rq.n <= 8 else max(60, trajectories // 5)
        study = noisy_filter_study(rq, error_rates, trajectories=shots)
        out['studies'].append(study)
        if verbose:
            print(f" {W}x{H}: {study['qubits']} qubits, {study['cx_gates']} CX, "
                  f"E0={study['E0']:.4f}, {shots} trajectories per point")
            for r in study['rows']:
                print(f"   p={r['p']:.0e}  errors/circuit={r['expected_errors']:5.2f}  "
                      f"dE raw={r['energy_raw'] - study['E0']:+.4f}  "
                      f"dE gauge-selected={r['energy_gauge'] - study['E0']:+.4f}  "
                      f"physical weight={r['gauge_weight']:.3f}")
    _save("noise_study", out)

    plt = _plt()
    fig, axes = plt.subplots(1, len(out['studies']), figsize=(5.5 * len(out['studies']), 4),
                             squeeze=False)
    for ax, study in zip(axes[0], out['studies']):
        ps = [r['p'] for r in study['rows'] if r['p'] > 0]
        raw = [abs(r['energy_raw'] - study['E0']) / abs(study['E0'])
               for r in study['rows'] if r['p'] > 0]
        sel = [abs(r['energy_gauge'] - study['E0']) / abs(study['E0'])
               for r in study['rows'] if r['p'] > 0]
        ax.loglog(ps, raw, 'o-', color="#c0392b", label="ancilla post-selection only")
        ax.loglog(ps, sel, 's-', color="#1f4e79", label="+ Gauss-law post-selection")
        floor = abs(study['rows'][0]['energy_raw'] - study['E0']) / abs(study['E0'])
        ax.axhline(floor, color="k", ls=":", label="noiseless (Trotter) floor")
        ax.set(xlabel="two-qubit error rate p", ylabel="relative energy error",
               title=f"{study['lattice']}: {study['qubits']} qubits, {study['cx_gates']} CX")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("results/noise_study.png", dpi=160)
    plt.close(fig)
    return out


EXPERIMENTS = {'rk-scan': rk_scan, 'filter-scaling': filter_scaling, 'noise': noise_study}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=list(EXPERIMENTS) + ["all"])
    args = parser.parse_args()
    for name in (EXPERIMENTS if args.experiment == "all" else [args.experiment]):
        print("=" * 72 + f"\n{name}\n" + "=" * 72)
        start = time.time()
        EXPERIMENTS[name]()
        print(f" [{time.time() - start:.0f} s]  -> results/{name.replace('-', '_')}*.json/.png")
