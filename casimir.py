"""Casimir-energy extraction for the 2+1D U(1) quantum link model.

Pipeline
    1. ground-state energy E0(d) against plate separation d = H - 1 at fixed W,
    2. bulk subtraction  E0(d) = a + b d + E_Cas(d),
    3. force by central difference  F(d) = -[E_Cas(d+1) - E_Cas(d-1)] / 2,
    4. model comparison for the decay of E_Cas.

E0(d) comes from exact diagonalisation in the gauge-invariant sector (the
reference), optionally cross-checked by the penalised full-space Hamiltonian,
by DMRG (tensor_network.py) and by the single-ancilla filter (rqsvt.py).

What the fit can and cannot show. The continuum prediction F/L ~ -d^-3 assumes a
massless photon and d much larger than the lattice spacing. Compact U(1) gauge
theory in 2+1D is confining, so at large d the residual must decay exponentially;
a power law can only appear as an intermediate regime (the spin-1/2 QLM at
lam_RK = 0 has an unusually light pseudo-Goldstone mode, Banerjee et al. 2013,
which makes that regime plausible but not guaranteed). With d <= 7 the two forms
cannot always be separated, so both fits are reported with their residuals,
together with a model-free second difference, and the exponent is treated as a
measured quantity rather than a verified one.

Second difference. D2(d) = E0(d+1) - 2 E0(d) + E0(d-1) removes a + b d exactly
(proofs/Casimir.lean), so it needs no bulk fit. For E_Cas = -C d^-p it behaves as
-C p (p+1) d^-(p+2): negative D2 means an attractive, convex-decaying residual.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.sparse.linalg import eigsh

from lattice import LatticeGrid
from QuantumFields.gauge_field import QuantumLinkModel, default_plates
from sector import GaugeSector

FULL_SPACE_MAX_QUBITS = 16


# ---------------------------------------------------------------------------
# Ground-state energy by each method
# ---------------------------------------------------------------------------

def ground_energy_sector(width, height, g=0.0, J=1.0, rk=0.0,
                         field_pattern="uniform", plates=None) -> dict:
    """Ground-state energy and in-sector gap from sector ED (the reference)."""
    lattice = LatticeGrid(width, height)
    plates = plates or default_plates(width)
    sector = GaugeSector(lattice, plates=plates)
    Hm = sector.hamiltonian(g, J, rk, field_pattern)
    if plates == "fixed":
        w, _ = sector.lowest(Hm, 2)
        gap = float(w[1] - w[0]) if len(w) > 1 else float('nan')
        energy = float(w[0])
    else:
        gs = sector.ground_state(g, J, rk, field_pattern)
        energy, gap = gs['energy'], gs['gap']
    return {'energy': energy, 'gap': gap, 'dim': sector.dim,
            'qubits': lattice.n_qubits, 'plates': plates}


def ground_energy_full(width, height, g=0.0, J=1.0, rk=0.0, field_pattern="uniform",
                       plates=None, penalty=200.0) -> float:
    """Penalised qubit Hamiltonian in the full 2^N_q space (small lattices only)."""
    lattice = LatticeGrid(width, height)
    if lattice.n_qubits > FULL_SPACE_MAX_QUBITS:
        raise MemoryError(f"{lattice.n_qubits} qubits is beyond full-space ED here")
    H = QuantumLinkModel(lattice, g=g, J=J, rk=rk, gauss_penalty=penalty,
                         field_pattern=field_pattern,
                         plates=plates or default_plates(width)).build_hamiltonian()
    return float(eigsh(H.to_matrix(sparse=True), k=1, which='SA',
                       return_eigenvectors=False)[0])


def ground_energy_rqsvt(width, height, g=0.0, J=1.0, rk=0.0, field_pattern="uniform",
                        plates=None, method="trotter", schedule=(1, 2, 4),
                        filter_kind="gaussian", n_circuits=1) -> dict:
    """Ground-state energy from the emulated single-ancilla filter (Richardson-extrapolated)."""
    from rqsvt import RQSVTGroundState
    rq = RQSVTGroundState(LatticeGrid(width, height), g=g, J=J, rk=rk,
                          field_pattern=field_pattern,
                          plates=plates or default_plates(width),
                          filter_kind=filter_kind)
    est = rq.estimate_energy(method, schedule, n_circuits=n_circuits)
    return {'energy': est['energy_richardson'], 'error': est['error'],
            'runs': est['runs'], 'summary': rq.summary()}


# ---------------------------------------------------------------------------
# Casimir extraction
# ---------------------------------------------------------------------------

def _scan_fit(ds, E0s, basis_fn, grid):
    best = None
    for s in grid:
        A = np.vstack([np.ones_like(ds), ds, basis_fn(ds, s)]).T
        coef, *_ = np.linalg.lstsq(A, E0s, rcond=None)
        sse = float(np.sum((E0s - A @ coef) ** 2))
        if best is None or sse < best[0]:
            best = (sse, s, coef)
    return best


def fit_powerlaw(ds, E0s) -> dict:
    """E0(d) = a + b d + C d^-(alpha - 1); the force then scales as d^-alpha."""
    ds, E0s = np.asarray(ds, float), np.asarray(E0s, float)
    sse, q, (a, b, C) = _scan_fit(ds, E0s, lambda d, q: d ** (-q),
                                  np.arange(0.2, 8.0001, 0.01))
    return {'alpha': q + 1.0, 'a': a, 'b': b, 'C': C, 'sse': sse}


def fit_exponential(ds, E0s) -> dict:
    """E0(d) = a + b d + C exp(-d / xi): the form expected for a massive photon."""
    ds, E0s = np.asarray(ds, float), np.asarray(E0s, float)
    sse, xi, (a, b, C) = _scan_fit(ds, E0s, lambda d, xi: np.exp(-d / xi),
                                   np.arange(0.1, 12.0001, 0.01))
    return {'xi': xi, 'a': a, 'b': b, 'C': C, 'sse': sse}


def second_difference(ds, E0s):
    """D2(d) = E0(d+1) - 2 E0(d) + E0(d-1) on interior separations (unit spacing)."""
    ds, E0s = np.asarray(ds, float), np.asarray(E0s, float)
    return ds[1:-1], E0s[2:] - 2.0 * E0s[1:-1] + E0s[:-2]


def casimir_force(ds, E_cas):
    """F(d) = -(E_Cas(d+1) - E_Cas(d-1)) / 2 at interior separations."""
    ds, E_cas = np.asarray(ds, float), np.asarray(E_cas, float)
    return ds[1:-1], -(E_cas[2:] - E_cas[:-2]) / 2.0


def loglog_slope(x, y):
    """Slope of log|y| against log x, with the number of points used."""
    x, y = np.asarray(x, float), np.abs(np.asarray(y, float))
    ok = y > 1e-14
    if ok.sum() < 2:
        return float('nan'), int(ok.sum())
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0]), int(ok.sum())


def analyse(ds, E0s, d_min: int = 2) -> dict:
    """Bulk subtraction, force and both decay models on separations d >= d_min.

    d = 1 has no interior sites at all (no Gauss constraint between the plates),
    so it is a lattice artefact and is excluded from the fits by default.
    """
    ds, E0s = np.asarray(ds, float), np.asarray(E0s, float)
    use = ds >= d_min
    power = fit_powerlaw(ds[use], E0s[use])
    expo = fit_exponential(ds[use], E0s[use])
    E_cas = E0s - (power['a'] + power['b'] * ds)
    dF, F = casimir_force(ds[use], E_cas[use])
    d2_d, d2 = second_difference(ds, E0s)
    slope_F, nF = loglog_slope(dF, F)
    slope_D2, nD = loglog_slope(d2_d[d2_d >= d_min], d2[d2_d >= d_min])
    return {'d': ds.tolist(), 'E0': E0s.tolist(), 'd_min': d_min,
            'power_law': power, 'exponential': expo,
            'E_cas': E_cas.tolist(),
            'force_d': dF.tolist(), 'force': F.tolist(),
            'alpha_from_force_slope': -slope_F, 'force_points': nF,
            'second_difference_d': d2_d.tolist(),
            'second_difference': d2.tolist(),
            'alpha_from_second_difference': -slope_D2 - 1.0,
            'fraction_attractive': float(np.mean(d2[d2_d >= d_min] < 0))
            if np.any(d2_d >= d_min) else float('nan')}


# ---------------------------------------------------------------------------
# Scan driver
# ---------------------------------------------------------------------------

def run_casimir_scan(width=4, separations=(1, 2, 3, 4, 5, 6), g=0.0, J=1.0, rk=0.0,
                     field_pattern="uniform", plates=None, cross_check=True,
                     verbose=True) -> dict:
    """E0(d) over the separations at fixed W, with full-space cross-checks where small."""
    plates = plates or default_plates(width)
    if verbose:
        print("=" * 72)
        print(f"CASIMIR SCAN  W={width}  g={g} J={J} lam_RK={rk}  plates={plates}")
        print("=" * 72)
    rows = []
    for d in separations:
        H = d + 1
        row = ground_energy_sector(width, H, g, J, rk, field_pattern, plates)
        row['d'] = d
        if cross_check and row['qubits'] <= FULL_SPACE_MAX_QUBITS:
            row['full_space'] = ground_energy_full(width, H, g, J, rk,
                                                   field_pattern, plates)
        rows.append(row)
        if verbose:
            extra = (f"   full-space diff {abs(row['full_space'] - row['energy']):.1e}"
                     if 'full_space' in row else "")
            print(f" d={d}  N_q={row['qubits']:2d}  sector dim={row['dim']:7d}  "
                  f"E0={row['energy']:+.8f}  gap={row['gap']:.4f}{extra}")

    result = {'width': width, 'g': g, 'J': J, 'rk': rk, 'plates': plates,
              'field_pattern': field_pattern, 'rows': rows}
    ds = [r['d'] for r in rows]
    if len(ds) >= 5:
        result['analysis'] = analyse(ds, [r['energy'] for r in rows])
        if verbose:
            _print_analysis(result['analysis'])
    elif verbose:
        print("\nNeed at least five separations for the three-parameter fits.")
    return result


def _print_analysis(an):
    p, x = an['power_law'], an['exponential']
    print(f"\nFits on d >= {an['d_min']}:")
    print(f"  power law    E0 = {p['a']:+.4f} {p['b']:+.4f} d {p['C']:+.4f} d^-{p['alpha'] - 1:.2f}"
          f"   -> alpha = {p['alpha']:.2f}   SSE = {p['sse']:.2e}")
    print(f"  exponential  E0 = {x['a']:+.4f} {x['b']:+.4f} d {x['C']:+.4f} exp(-d/{x['xi']:.2f})"
          f"   SSE = {x['sse']:.2e}")
    print("  second difference D2(d): " +
          "  ".join(f"d={int(d)}: {v:+.5f}" for d, v in
                    zip(an['second_difference_d'], an['second_difference'])))
    print(f"  alpha from slope of |F|: {an['alpha_from_force_slope']:.2f} "
          f"({an['force_points']} points);  from |D2|: "
          f"{an['alpha_from_second_difference']:.2f};  continuum value 3")


def plot_scan(result, path="results/casimir_force.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    an = result['analysis']
    ds, E0 = np.array(an['d']), np.array(an['E0'])
    p, x = an['power_law'], an['exponential']
    dd = np.linspace(ds.min(), ds.max(), 200)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].plot(ds, E0, 'o-', color='#1f4e79', label='$E_0(d)$ (sector ED)')
    axes[0].plot(dd, p['a'] + p['b'] * dd, 'k--', alpha=0.6, label='bulk $a + b\\,d$')
    axes[0].set(xlabel='plate separation $d$', ylabel='$E_0$', title='Ground-state energy')

    axes[1].axhline(0, color='k', lw=0.6)
    axes[1].plot(ds, an['E_cas'], 's', color='#7b2d8e', label='residual $E_{Cas}$')
    axes[1].plot(dd, p['C'] * dd ** (-(p['alpha'] - 1)), '-', color='#7b2d8e', alpha=0.7,
                 label=f"power law, $\\alpha$={p['alpha']:.2f}")
    axes[1].plot(dd, (x['a'] - p['a']) + (x['b'] - p['b']) * dd + x['C'] * np.exp(-dd / x['xi']),
                 ':', color='#c0392b', label=f"exponential, $\\xi$={x['xi']:.2f}")
    axes[1].set(xlabel='plate separation $d$', ylabel='$E_{Cas}(d)$',
                title='Casimir energy after bulk subtraction')

    d2d, d2 = np.array(an['second_difference_d']), np.array(an['second_difference'])
    neg = d2 < 0
    axes[2].loglog(d2d[neg], -d2[neg], 'o', color='#c0392b', label='$-D_2(d)$ (attractive)')
    if (~neg).any():
        axes[2].loglog(d2d[~neg], d2[~neg], 'x', color='gray', label='$+D_2(d)$')
    ref = d2d[d2d >= 2]
    if len(ref):
        anchor = abs(d2[d2d >= 2][0])
        axes[2].loglog(ref, anchor * (ref / ref[0]) ** -4.0, 'k:', label='$d^{-4}$ (continuum)')
    axes[2].set(xlabel='plate separation $d$', ylabel='$|D_2(d)|$',
                title='Model-free second difference')
    for ax in axes:
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=8)
    fig.suptitle(f"W={result['width']}, J={result['J']}, g={result['g']}, "
                 f"$\\lambda_{{RK}}$={result['rk']}, plates={result['plates']}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    out = run_casimir_scan(width=4, separations=(1, 2, 3, 4, 5, 6, 7))
    plot_scan(out)
    with open("results/casimir_scan_W4.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("\n[saved results/casimir_force.png, results/casimir_scan_W4.json]")
