"""Research experiments: one function and one command per study.

    python experiments.py rk-scan             decay of the Casimir residual against lam_RK
    python experiments.py electric            control: staggered electric field (mass gap) scan
    python experiments.py width               finite-width extrapolation E0(d)/W, W = 2..8
    python experiments.py filter-scaling      cost and accuracy of the filter against d
    python experiments.py filter-design       filter sharpness and truncation against cost
    python experiments.py trotter-order       first against second order inside the filter
    python experiments.py noise               filter under two-qubit depolarising noise
    python experiments.py quench              real-time quench between the plates against d
    python experiments.py trotter-budget      Trotter fidelity and gate cost of the quench
    python experiments.py joshi               benchmark against Joshi et al. (2026)
    python experiments.py all [--quick]

Every command accepts options (python experiments.py <name> -h), prints a
table, and writes results/<name>[_tag].json and a .png figure. The functions
can also be called from a notebook with other parameters.

Fits whose best parameter sits at the edge of its search range are marked
"at bound" and should not be quoted as measurements.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from casimir import analyse, ground_energy_sector
from config import environment
from lattice import LatticeGrid
from QuantumFields.gauge_field import default_plates
from rqsvt import RQSVTGroundState
from sector import count_states

XI_GRID_MAX = 12.0
SECTOR_MAX_DIM = 6_000_000


def _save(name, payload):
    os.makedirs("results", exist_ok=True)
    payload = {'environment': environment(), **payload}
    with open(f"results/{name}.json", "w") as fh:
        json.dump(payload, fh, indent=1,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    return f"results/{name}.json"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _finish(fig, name):
    fig.tight_layout()
    fig.savefig(f"results/{name}.png", dpi=160)
    _plt().close(fig)


def _fit_flags(an) -> dict:
    xi, alpha = an['exponential']['xi'], an['power_law']['alpha']
    return {'xi': xi, 'xi_at_bound': bool(xi >= XI_GRID_MAX - 1e-6 or xi <= 0.11),
            'alpha': alpha, 'alpha_at_bound': bool(alpha <= 1.21 or alpha >= 8.99),
            'sse_exponential': an['exponential']['sse'], 'sse_power': an['power_law']['sse']}


# ---------------------------------------------------------------------------
# Casimir residual: coupling scans
# ---------------------------------------------------------------------------

def rk_scan(width=4, rks=(-0.5, -0.25, 0.0, 0.25, 0.5, 0.7, 0.8, 0.9),
            separations=tuple(range(1, 8)), J=1.0, verbose=True) -> dict:
    """E0(d) for each lam_RK: decay length xi, power-law exponent, D2 and gaps.

    lam_RK = J is the Rokhsar-Kivelson point, where E0 = 0 for every d and the
    gap closes; approaching it the correlation length, and with it the range
    of the Casimir residual, should grow.
    """
    rows = []
    for rk in rks:
        data = [ground_energy_sector(width, d + 1, J=J, rk=rk) for d in separations]
        an = analyse(separations, [r['energy'] for r in data])
        d2 = np.array(an['second_difference'])
        row = {'rk': rk, **_fit_flags(an), 'D2': d2.tolist(),
               'all_D2_negative': bool(np.all(d2 < 0)),
               'gap_min': data[-1]['gap'], 'gaps': [r['gap'] for r in data],
               'E0': [r['energy'] for r in data]}
        rows.append(row)
        if verbose:
            better = "exp" if row['sse_exponential'] < row['sse_power'] else "power"
            print(f" lam_RK={rk:+.2f}  xi={row['xi']:5.2f}{'*' if row['xi_at_bound'] else ' '} "
                  f"alpha={row['alpha']:4.2f}{'*' if row['alpha_at_bound'] else ' '} "
                  f"SSE exp/pow={row['sse_exponential']:.1e}/{row['sse_power']:.1e} ({better})  "
                  f"all D2<0: {row['all_D2_negative']}  gap(d={separations[-1]})={row['gap_min']:.3f}")
    if verbose:
        print(" (* = fit parameter at the edge of its search range: not determined)")
    out = {'width': width, 'J': J, 'separations': list(separations), 'rows': rows}
    _save(f"rk_scan_W{width}", out)

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
                 label="decay length $\\xi$ (determined fits only)")
    ax2 = axes[1].twinx()
    ax2.plot([r['rk'] for r in rows], [1.0 / r['gap_min'] for r in rows], 's--',
             color="#1e8449", label=f"1 / gap at d={separations[-1]}")
    axes[1].set(xlabel="$\\lambda_{RK}$ / J", ylabel="$\\xi$ (lattice spacings)",
                title="Range of the Casimir residual")
    ax2.set_ylabel("1 / gap", color="#1e8449")
    axes[1].legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="lower right")
    axes[1].grid(alpha=0.3)
    _finish(fig, f"rk_scan_W{width}")
    return out


def electric_scan(width=4, gs=(0.0, 0.5, 1.0, 2.0), separations=tuple(range(1, 8)), J=1.0,
                  rk=0.0, verbose=True) -> dict:
    """Control experiment: a staggered electric field gaps the photon.

    A mass term should shorten the range of the residual (smaller xi, faster
    decay of D2). This separates a genuine long-range effect from lattice
    artefacts in the lam_RK scan. Needs even W (fixed staggered plates).
    """
    rows = []
    for g in gs:
        data = [ground_energy_sector(width, d + 1, g=g, J=J, rk=rk, field_pattern="staggered")
                for d in separations]
        an = analyse(separations, [r['energy'] for r in data])
        row = {'g': g, **_fit_flags(an), 'D2': an['second_difference'],
               'gaps': [r['gap'] for r in data], 'E0': [r['energy'] for r in data]}
        rows.append(row)
        if verbose:
            print(f" g={g:4.2f}  xi={row['xi']:5.2f}{'*' if row['xi_at_bound'] else ' '} "
                  f"alpha={row['alpha']:4.2f}{'*' if row['alpha_at_bound'] else ' '} "
                  f"gap(d={separations[-1]})={row['gaps'][-1]:.3f}  "
                  f"D2: " + " ".join(f"{v:+.4f}" for v in row['D2']))
    out = {'width': width, 'J': J, 'rk': rk, 'separations': list(separations), 'rows': rows}
    _save(f"electric_scan_W{width}", out)

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ds = np.array(separations[1:-1])
    for row in rows:
        d2 = np.abs(np.array(row['D2']))
        axes[0].semilogy(ds, d2, 'o-', ms=4, label=f"g={row['g']:.2f}")
        axes[1].plot(separations, row['gaps'], 's-', ms=4, label=f"g={row['g']:.2f}")
    axes[0].set(xlabel="plate separation d", ylabel="$|D_2(d)|$",
                title=f"Casimir signal with a staggered field, W={width}")
    axes[1].set(xlabel="plate separation d", ylabel="in-sector gap", title="Gap")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    _finish(fig, f"electric_scan_W{width}")
    return out


def width_extrapolation(widths=(2, 4, 6, 8), d_max=7, J=1.0, rk=0.0,
                        max_dim=SECTOR_MAX_DIM, verbose=True) -> dict:
    """E0(d)/W and D2(d)/W against width, extrapolated in 1/W^2.

    Only even W (fixed staggered plates). For each W every separation whose
    sector fits in `max_dim` is solved (count_states decides beforehand). At
    every d reached by at least two widths, E0/W = e(d) + c(d)/W^2 is fitted
    on the two or more largest widths; the same is done for D2/W. This is the
    route from the narrow cylinders to a statement about the 2+1D limit, and
    it shows directly how far W = 2..4 are from it.
    """
    table = {}
    for W in widths:
        if W % 2:
            raise ValueError("width extrapolation uses fixed staggered plates: even W only")
        table[W] = {}
        for d in range(1, d_max + 1):
            dim = count_states(LatticeGrid(W, d + 1), "fixed")
            if dim > max_dim:
                if verbose:
                    print(f" W={W} d={d}: {dim:,} states > max_dim, stopping this width")
                break
            t = time.time()
            r = ground_energy_sector(W, d + 1, J=J, rk=rk, plates="fixed")
            table[W][d] = {'E0': r['energy'], 'gap': r['gap'], 'dim': r['dim']}
            if verbose:
                print(f" W={W} d={d}: {r['dim']:>9,} states  E0/W={r['energy'] / W:+.8f}  "
                      f"gap={r['gap']:.4f}  [{time.time() - t:.1f} s]")

    def d2_of(W, d):
        e = table[W]
        return e[d + 1]['E0'] - 2 * e[d]['E0'] + e[d - 1]['E0'] \
            if all(k in e for k in (d - 1, d, d + 1)) else None

    extrap = []
    for d in range(1, d_max + 1):
        ws = [W for W in widths if d in table[W]]
        if len(ws) < 2:
            continue
        x = np.array([1.0 / W ** 2 for W in ws])
        y = np.array([table[W][d]['E0'] / W for W in ws])
        fit = np.polyfit(x[-3:], y[-3:], 1) if len(ws) >= 3 else np.polyfit(x, y, 1)
        row = {'d': d, 'widths': ws, 'E0_per_W': y.tolist(), 'E0_per_W_extrapolated': float(fit[1]),
               'finite_width_coefficient': float(fit[0])}
        wd = [W for W in ws if d2_of(W, d) is not None]
        if wd:
            row['D2_per_W'] = {W: d2_of(W, d) / W for W in wd}
            if len(wd) >= 2:
                xd = np.array([1.0 / W ** 2 for W in wd])
                yd = np.array([d2_of(W, d) / W for W in wd])
                row['D2_per_W_extrapolated'] = float(np.polyfit(xd[-3:], yd[-3:], 1)[1])
        extrap.append(row)
        if verbose:
            msg = (f" d={d}: E0/W -> {row['E0_per_W_extrapolated']:+.8f} "
                   f"(from W={ws}, slope in 1/W^2 {row['finite_width_coefficient']:+.4f})")
            if 'D2_per_W_extrapolated' in row:
                msg += f"   D2/W -> {row['D2_per_W_extrapolated']:+.6f}"
            print(msg)
    out = {'J': J, 'rk': rk, 'widths': list(widths), 'table': table, 'extrapolation': extrap}
    _save("width_extrapolation", out)

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for row in extrap:
        x = [1.0 / W ** 2 for W in row['widths']]
        axes[0].plot(x, row['E0_per_W'], 'o-', label=f"d={row['d']}")
        axes[0].plot(0, row['E0_per_W_extrapolated'], 'k*', ms=7)
    axes[0].set(xlabel="$1/W^2$", ylabel="$E_0/W$", title="Ground-state energy per unit width")
    axes[0].legend(fontsize=7, ncol=2)
    for W in widths:
        pts = [(d, d2_of(W, d) / W) for d in range(2, d_max) if d2_of(W, d) is not None]
        if pts:
            ds, vals = zip(*pts)
            axes[1].plot(ds, vals, 'o-', label=f"W={W}")
    pts = [(r['d'], r['D2_per_W_extrapolated']) for r in extrap if 'D2_per_W_extrapolated' in r]
    if pts:
        axes[1].plot(*zip(*pts), 'k*--', ms=9, label="extrapolated $W\\to\\infty$")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set(xlabel="plate separation d", ylabel="$D_2(d)/W$",
                title="Casimir signal per unit width")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    _finish(fig, "width_extrapolation")
    return out


# ---------------------------------------------------------------------------
# Ground-state filter
# ---------------------------------------------------------------------------

def filter_scaling(width=4, separations=(1, 2, 3, 4), J=1.0, rk=0.0, schedule=(1, 2, 4),
                   filter_kind="gaussian", initial_state="max_flippable", order=1,
                   verbose=True) -> dict:
    """The quantum algorithm as a function of d: degree, overlap, accuracy, gates."""
    rows = []
    for d in separations:
        t = time.time()
        rq = RQSVTGroundState(LatticeGrid(width, d + 1), J=J, rk=rk, filter_kind=filter_kind,
                              initial_state=initial_state)
        info = rq.summary()
        est = rq.estimate_energy("trotter", schedule, order=order)
        res = rq.resources("trotter", n_steps=1, order=order)
        row = {'d': d, 'qubits': info['qubits'], 'sector_dim': info['sector_dim'],
               'gap': info['gap'], 'lambda_H': info['lambda_H'],
               'signal_operators': info['signal_operators'],
               'initial_overlap': info['initial_overlap'],
               'success_prob': est['runs'][0]['success_prob'],
               'errors': [r['energy'] - rq.E0 for r in est['runs']],
               'overlaps': [r['overlap'] for r in est['runs']],
               'richardson_error': est['error'], 'E0': rq.E0,
               'E_filter': est['energy_richardson'],
               'two_qubit_gates_n1': res['two_qubit_gates'],
               'seconds': time.time() - t}
        rows.append(row)
        if verbose:
            print(f" d={d}  {row['qubits']:2d} qubits  gap={row['gap']:.3f}  "
                  f"signal ops={row['signal_operators']:4d}  overlap0={row['initial_overlap']:.4f}  "
                  f"P(success)={row['success_prob']:.4f}  err(n=1)={row['errors'][0]:+.1e}  "
                  f"Richardson={row['richardson_error']:+.1e}  2q gates={row['two_qubit_gates_n1']:,}"
                  f"  [{row['seconds']:.0f}s]")
    out = {'width': width, 'J': J, 'rk': rk, 'schedule': list(schedule), 'filter': filter_kind,
           'initial_state': initial_state, 'order': order, 'rows': rows}
    if len(rows) >= 3:
        x = np.log([r['d'] for r in rows[1:]])
        out['gate_growth_exponent'] = float(np.polyfit(
            x, np.log([r['two_qubit_gates_n1'] for r in rows[1:]]), 1)[0])
        out['overlap_decay_per_d'] = float(np.exp(-np.polyfit(
            [r['d'] for r in rows], np.log([r['initial_overlap'] for r in rows]), 1)[0]))
        if verbose:
            print(f" two-qubit gates grow as d^{out['gate_growth_exponent']:.2f} (about d^3 "
                  f"expected: lambda_H/gap ~ d^2 signal operators with ~ d plaquettes each); "
                  f"initial overlap falls by {out['overlap_decay_per_d']:.2f}x per unit of d")
    tag = f"W{width}_{filter_kind}_{initial_state}_o{order}"
    _save(f"filter_scaling_{tag}", out)

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
    _finish(fig, f"filter_scaling_{tag}")
    return out


def filter_design(width=3, height=3, J=2.0, sharpness=(2.0, 3.0, 4.0),
                  truncation=(1e-2, 1e-3, 1e-4), kinds=("gaussian", "step"),
                  verbose=True) -> dict:
    """Overlap reached by the exact filter against its cost, over the design knobs.

    The goal is the cheapest filter (fewest signal operators) that reaches a
    target overlap such as 0.999, together with the Trotter error at n = 1.
    """
    rows = []
    lattice = LatticeGrid(width, height)
    for kind in kinds:
        for s in sharpness:
            for tr in truncation:
                rq = RQSVTGroundState(lattice, J=J, filter_kind=kind, sharpness=s, truncation=tr)
                exact = rq.run("exact")
                trot = rq.run("trotter", n_steps=1)
                row = {'filter': kind, 'sharpness': s, 'truncation': tr, 'degree': rq.D,
                       'signal_operators': rq.n_signal, 'overlap_exact': exact['overlap'],
                       'success_prob': exact['success_prob'],
                       'error_exact': exact['energy'] - rq.E0,
                       'error_trotter_n1': trot['energy'] - rq.E0}
                rows.append(row)
                if verbose:
                    print(f" {kind:8s} sharpness={s:.1f} truncation={tr:.0e}: D={rq.D:3d} "
                          f"({rq.n_signal} signal ops)  overlap={exact['overlap']:.6f}  "
                          f"P={exact['success_prob']:.3f}  err exact={row['error_exact']:+.1e}  "
                          f"err Trotter n=1={row['error_trotter_n1']:+.1e}")
    out = {'lattice': f"{width}x{height}", 'J': J, 'rows': rows}
    _save(f"filter_design_{width}x{height}", out)

    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for kind, marker in zip(kinds, "os"):
        sel = [r for r in rows if r['filter'] == kind]
        ax.semilogy([r['signal_operators'] for r in sel],
                    [max(1 - r['overlap_exact'], 1e-16) for r in sel], marker,
                    label=f"{kind} filter")
    ax.axhline(1e-3, color="k", ls=":", label="overlap 0.999")
    ax.set(xlabel="signal operators", ylabel="1 - overlap with the ground state",
           title=f"Filter design, {width}x{height}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    _finish(fig, f"filter_design_{width}x{height}")
    return out


def trotter_order(lattices=((2, 3), (3, 3)), J=2.0, schedules=((1, 2, 4, 8), (1, 2, 4)),
                  verbose=True) -> dict:
    """First- against second-order Trotter inside the filter, at equal gate count.

    For each order, the energy error at every n and after Richardson
    extrapolation, against the measured two-qubit gates of the circuits used.
    """
    out = {'J': J, 'studies': []}
    for (W, H) in lattices:
        rq = RQSVTGroundState(LatticeGrid(W, H), J=J, plates=default_plates(W),
                              filter_kind="gaussian")
        study = {'lattice': f"{W}x{H}", 'E0': rq.E0, 'orders': []}
        for order, schedule in zip((1, 2), schedules):
            gates1 = rq.resources("trotter", n_steps=1, order=order)['two_qubit_gates']
            est = rq.estimate_energy("trotter", schedule, order=order)
            runs = [{'n': r['n'], 'error': r['energy'] - rq.E0,
                     'two_qubit_gates': gates1 * r['n']} for r in est['runs']]
            study['orders'].append({'order': order, 'runs': runs,
                                    'richardson_error': est['error'],
                                    'richardson_gates': gates1 * sum(schedule)})
            if verbose:
                print(f" {W}x{H} order {order}: " + "  ".join(
                    f"n={r['n']}: {r['error']:+.1e} ({r['two_qubit_gates']:,} 2q)" for r in runs)
                    + f"  | Richardson {est['error']:+.1e} ({gates1 * sum(schedule):,} 2q total)")
        out['studies'].append(study)
    _save("trotter_order", out)

    plt = _plt()
    fig, axes = plt.subplots(1, len(out['studies']), figsize=(5.5 * len(out['studies']), 4),
                             squeeze=False)
    for ax, study in zip(axes[0], out['studies']):
        for o, colour in zip(study['orders'], ("#1f4e79", "#c0392b")):
            ax.loglog([r['two_qubit_gates'] for r in o['runs']], [abs(r['error']) for r in o['runs']],
                      'o-', color=colour, label=f"order {o['order']}")
            ax.loglog([o['richardson_gates']], [abs(o['richardson_error'])], '*', ms=12,
                      color=colour, label=f"order {o['order']}, Richardson")
        ax.set(xlabel="two-qubit gates", ylabel="$|E - E_0|$", title=study['lattice'])
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    _finish(fig, "trotter_order")
    return out


def noise_study(lattices=((2, 2), (2, 3)), J=2.0,
                error_rates=(0.0, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3), trajectories=400,
                verbose=True) -> dict:
    """Energy of the filtered state under two-qubit depolarising noise."""
    from noise import noisy_filter_study
    out = {'studies': []}
    for (W, H) in lattices:
        rq = RQSVTGroundState(LatticeGrid(W, H), J=J, plates="fixed", filter_kind="gaussian")
        study = noisy_filter_study(rq, error_rates, trajectories=trajectories)
        out['studies'].append(study)
        if verbose:
            print(f" {W}x{H}: {study['qubits']} qubits, {study['cx_gates']} CX, "
                  f"E0={study['E0']:.4f}, {trajectories} trajectories per point")
            for r in study['rows']:
                print(f"   p={r['p']:.0e}  errors/circuit={r['expected_errors']:5.2f}  "
                      f"dE raw={r['energy_raw'] - study['E0']:+.4f} +- {r['energy_raw_err']:.4f}  "
                      f"dE gauge-selected={r['energy_gauge'] - study['E0']:+.4f} "
                      f"+- {r['energy_gauge_err']:.4f}  accepted: ancilla {r['accept_raw']:.3f}, "
                      f"+ Gauss {r['accept_gauge']:.3f}")
    _save("noise_study", out)

    plt = _plt()
    fig, axes = plt.subplots(1, len(out['studies']), figsize=(5.5 * len(out['studies']), 4),
                             squeeze=False)
    for ax, study in zip(axes[0], out['studies']):
        rows = [r for r in study['rows'] if r['p'] > 0]
        ps = [r['p'] for r in rows]
        E0 = abs(study['E0'])
        ax.errorbar(ps, [abs(r['energy_raw'] - study['E0']) / E0 for r in rows],
                    yerr=[r['energy_raw_err'] / E0 for r in rows], fmt='o-', color="#c0392b",
                    capsize=3, label="ancilla post-selection only")
        ax.errorbar(ps, [abs(r['energy_gauge'] - study['E0']) / E0 for r in rows],
                    yerr=[r['energy_gauge_err'] / E0 for r in rows], fmt='s-', color="#1f4e79",
                    capsize=3, label="+ Gauss-law post-selection")
        floor = abs(study['rows'][0]['energy_raw'] - study['E0']) / E0
        ax.axhline(floor, color="k", ls=":", label="noiseless (Trotter) floor")
        ax.set(xscale="log", yscale="log", xlabel="two-qubit error rate p",
               ylabel="relative energy error",
               title=f"{study['lattice']}: {study['qubits']} qubits, {study['cx_gates']} CX")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    _finish(fig, "noise_study")
    return out


# ---------------------------------------------------------------------------
# Real-time dynamics
# ---------------------------------------------------------------------------

def quench_study(width=4, separations=(1, 2, 3, 4, 5), J=1.0, rk=0.0, t_max=20.0,
                 n_times=201, trotter=((0.25, 2),), verbose=True) -> dict:
    """Quench from the most-flippable product state at every plate separation."""
    from dynamics import quench_vs_separation
    out = quench_vs_separation(width, separations, J=J, rk=rk, t_max=t_max, n_times=n_times,
                               trotter=trotter, verbose=verbose)
    _save(f"quench_W{width}", out)

    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    cmap = plt.get_cmap("viridis")
    for k, run in enumerate(out['runs']):
        colour = cmap(k / max(1, len(out['runs']) - 1))
        axes[0].plot(run['times'], run['return_probability'], color=colour, label=f"d={run['d']}")
        if run['entropy_midplane']:
            axes[1].plot(run['times'], run['entropy_midplane'], color=colour, label=f"d={run['d']}")
        rows = np.arange(len(run['energy_density_long_time'])) + 0.5
        axes[2].plot(rows / run['d'], np.array(run['energy_density_long_time'])
                     - np.array(run['ground_state_profiles']['energy_density']),
                     'o-', color=colour, label=f"d={run['d']}")
    axes[0].set(xlabel="time", ylabel="return probability", title="(a) Loschmidt echo")
    axes[1].set(xlabel="time", ylabel="entanglement entropy across the midplane",
                title="(b) entanglement growth")
    axes[2].set(xlabel="position across the gap y/d",
                ylabel="late-time minus ground-state energy density",
                title="(c) energy deposited by the quench, row by row")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"Quench from the most-flippable state, W={width}, J={J}, "
                 f"$\\lambda_{{RK}}$={rk}")
    _finish(fig, f"quench_W{width}")
    return out


def trotter_budget_study(width=3, d=2, J=2.0, dts=(0.4, 0.2, 0.1, 0.05), t_max=5.0,
                         fidelity_target=0.99, verbose=True) -> dict:
    """How long a Trotterised quench stays accurate, and what that costs in gates."""
    from dynamics import trotter_budget
    out = trotter_budget(width, d, J=J, dts=dts, t_max=t_max, fidelity_target=fidelity_target)
    if verbose:
        for r in out['rows']:
            print(f" order {r['order']}  dt={r['dt']:.3f}: fidelity >= {fidelity_target} up to "
                  f"t={r['time_within_target']:.2f} ({r['steps_within_target']} steps, "
                  f"{r['two_qubit_gates_to_target']:,} two-qubit gates at "
                  f"{r['two_qubit_gates_per_step']} per step)")
    _save(f"trotter_budget_{width}x{d + 1}", out)

    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for r in out['rows']:
        ax.semilogy(r['times'], np.maximum(1 - np.array(r['fidelity']), 1e-16),
                    '-' if r['order'] == 1 else '--', label=f"order {r['order']}, dt={r['dt']}")
    ax.axhline(1 - fidelity_target, color="k", ls=":")
    ax.set(xlabel="time", ylabel="1 - fidelity with exact evolution",
           title=f"Trotterised quench, {width}x{d + 1}, J={J}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2)
    _finish(fig, f"trotter_budget_{width}x{d + 1}")
    return out


def joshi(verbose=True, big_lattice=(5, 4)) -> dict:
    """Benchmark of the matter-coupled circuits and dynamics against Joshi et al."""
    from joshi_benchmark import run_all
    return run_all(verbose=verbose, big_lattice=big_lattice)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _floats(text):
    return tuple(float(x) for x in text.split(","))


def _ints(text):
    return tuple(int(x) for x in text.split(","))


def _range(d_max):
    return tuple(range(1, d_max + 1))


COMMANDS = {
    'rk-scan': (rk_scan, [("--width", int, 4), ("--rks", _floats, None), ("--dmax", int, 7)]),
    'electric': (electric_scan, [("--width", int, 4), ("--gs", _floats, None),
                                 ("--dmax", int, 7)]),
    'width': (width_extrapolation, [("--widths", _ints, None), ("--dmax", int, 7),
                                    ("--rk", float, 0.0), ("--max-dim", int, SECTOR_MAX_DIM)]),
    'filter-scaling': (filter_scaling, [("--width", int, 4), ("--dmax", int, 4),
                                        ("--filter", str, "gaussian"),
                                        ("--initial", str, "max_flippable"),
                                        ("--order", int, 1)]),
    'filter-design': (filter_design, [("--width", int, 3), ("--height", int, 3)]),
    'trotter-order': (trotter_order, []),
    'noise': (noise_study, [("--trajectories", int, 400), ("--rates", _floats, None)]),
    'quench': (quench_study, [("--width", int, 4), ("--dmax", int, 5), ("--rk", float, 0.0),
                              ("--tmax", float, 20.0)]),
    'trotter-budget': (trotter_budget_study, [("--width", int, 3), ("--d", int, 2)]),
    'joshi': (joshi, []),
}

ARG_TO_KWARG = {'rks': 'rks', 'gs': 'gs', 'widths': 'widths', 'rk': 'rk', 'max_dim': 'max_dim',
                'width': 'width', 'height': 'height', 'filter': 'filter_kind',
                'initial': 'initial_state', 'order': 'order', 'trajectories': 'trajectories',
                'rates': 'error_rates', 'tmax': 't_max', 'd': 'd'}

QUICK = {
    'rk-scan': dict(width=2, rks=(0.0, 0.5), separations=_range(6)),
    'electric': dict(width=2, gs=(0.0, 1.0), separations=_range(6)),
    'width': dict(widths=(2, 4), d_max=4),
    'filter-scaling': dict(width=2, separations=(1, 2, 3)),
    'filter-design': dict(width=2, height=3, sharpness=(3.0,), truncation=(1e-2, 1e-3)),
    'trotter-order': dict(lattices=((2, 3),)),
    'noise': dict(lattices=((2, 2),), trajectories=60, error_rates=(0.0, 1e-3)),
    'quench': dict(width=2, separations=(1, 2, 3), t_max=5.0, n_times=26),
    'trotter-budget': dict(width=2, d=2, dts=(0.2, 0.1), t_max=2.0),
    'joshi': dict(big_lattice=(4, 3)),
}


def _kwargs_from(name, args) -> dict:
    """Keyword arguments of the experiment function from the parsed options."""
    kwargs = {}
    for flag, _, _ in COMMANDS[name][1]:
        key = flag.lstrip("-").replace("-", "_")
        value = getattr(args, key)
        if value is None:
            continue
        if key == "dmax":
            if name == "width":
                kwargs['d_max'] = value
            else:
                kwargs['separations'] = _range(value)
        else:
            kwargs[ARG_TO_KWARG[key]] = value
    return kwargs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="experiment", required=True)
    for name, (_, options) in COMMANDS.items():
        p = sub.add_parser(name, help=COMMANDS[name][0].__doc__.splitlines()[0])
        for flag, typ, default in options:
            p.add_argument(flag, type=typ, default=default)
    p_all = sub.add_parser("all", help="every experiment with its default parameters")
    p_all.add_argument("--quick", action="store_true", help="small lattices, a few minutes")
    args = parser.parse_args(argv)

    if args.experiment == "all":
        jobs = [(name, QUICK[name] if args.quick else {}) for name in COMMANDS]
    else:
        jobs = [(args.experiment, _kwargs_from(args.experiment, args))]
    for name, kwargs in jobs:
        print("=" * 72 + f"\n{name}\n" + "=" * 72)
        start = time.time()
        COMMANDS[name][0](**kwargs)
        print(f" [{time.time() - start:.0f} s]  -> results/")


if __name__ == "__main__":
    main()
