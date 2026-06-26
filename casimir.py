"""Casimir-force extraction for the 2+1D U(1) quantum link model (thesis §2.3.5).

Pipeline: ground-state energy E0(d) vs plate separation d = H-1 (fixed width W),
bulk subtraction E0(d) = a + b*d + E_Cas(d), Casimir force F(d) = -dE_Cas/dd by
central difference, and a power-law fit F ∝ -d^{-alpha} (continuum: alpha = 3).

E0(d) is computed by exact diagonalisation (reference), by RQSVT (the quantum
method, as a cross-check at small separations), and by DMRG for the larger
separations that exact diagonalisation cannot reach.

Important caveats (be explicit about these in the thesis):
  1. The d^{-alpha} power law requires a LIGHT / near-gapless gauge field. Deep in
     the confined phase the force decays exponentially (~e^{-m d}), not as a power
     law -- so scan g/J to locate the regime where alpha ~ 3. This is the opposite
     of Joshi's deep-confined string-breaking regime.
  2. Few separations + the spin-1/2 truncation give large finite-size and
     truncation corrections (§2.4.2); alpha can deviate from 3 on small lattices.
  3. RQSVT's |0...0> guess only overlaps the ground state away from the magnetic
     regime. In the light regime needed for (1), use exact diag / DMRG, or supply
     a better guess state to RQSVT.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

from lattice import LatticeGrid
from QuantumFields.gauge_field import QuantumLinkModel

EXACT_MAX_QUBITS = 20          # sparse ED is comfortable up to ~20 link qubits


# ---------------------------------------------------------------------------
# Ground-state energy by each method
# ---------------------------------------------------------------------------

def ground_energy_exact(width, height, g, J, penalty=200.0):
    lattice = LatticeGrid(width, height)
    H = QuantumLinkModel(lattice, g=g, J=J, gauss_penalty=penalty).build_hamiltonian()
    E0 = float(eigsh(H.to_matrix(sparse=True), k=1, which='SA',
                     return_eigenvectors=False)[0])
    return E0, lattice.n_qubits


def ground_energy_dmrg(width, height, g, J):
    """DMRG energy via tensor_compression (needs quimb); None if unavailable."""
    try:
        from tensor_compression import get_ground_state_dmrg
        lattice = LatticeGrid(width, height)
        energy, _ = get_ground_state_dmrg(lattice, g=g)
        return float(energy)
    except Exception as exc:                      # quimb missing or DMRG failed
        print(f"    [DMRG unavailable: {exc}]")
        return None


def ground_energy_rqsvt(width, height, g, J, max_qubits=20):
    """RQSVT (structured Trotter on Aer) energy + guess overlap; None if too big."""
    from rqsvt import RQSVTGroundState
    from qiskit_aer import AerSimulator
    lattice = LatticeGrid(width, height)
    rq = RQSVTGroundState(lattice, g=g, J=J)
    if lattice.n_qubits + rq.n_anc > max_qubits:
        return None
    est = rq.estimate_energy(AerSimulator(method='statevector'),
                             backend='trotter', n_trotter_list=(1, 2))
    _, overlap = rq.guess_energy()
    return {'energy': est['energy_richardson'], 'guess_overlap': overlap}


# ---------------------------------------------------------------------------
# Casimir extraction
# ---------------------------------------------------------------------------

def fit_powerlaw(ds, E0s):
    """Direct fit E0(d) = a + b*d + C*d^{-(alpha-1)}, returning (alpha, a, b, C).

    For a fixed exponent q = alpha-1 the model is linear in (a, b, C), so we scan q
    and keep the least-squares best. This avoids the bias of subtract-then-difference,
    where the linear bulk fit absorbs part of the Casimir tail and steepens alpha.
    """
    ds = np.asarray(ds, float)
    E0s = np.asarray(E0s, float)
    best = None
    for q in np.arange(0.5, 6.0001, 0.01):
        A = np.vstack([np.ones_like(ds), ds, ds ** (-q)]).T
        (a, b, C), *_ = np.linalg.lstsq(A, E0s, rcond=None)
        sse = float(np.sum((E0s - (a + b * ds + C * ds ** (-q))) ** 2))
        if best is None or sse < best[0]:
            best = (sse, q, a, b, C)
    _, q, a, b, C = best
    return q + 1.0, a, b, C


def casimir_force(ds, E_cas):
    """Measured force F(d) = -(E_Cas(d+1) - E_Cas(d-1)) / 2 at interior separations."""
    ds = np.asarray(ds, float)
    dF, F = [], []
    for k in range(1, len(ds) - 1):
        dF.append(ds[k])
        F.append(-(E_cas[k + 1] - E_cas[k - 1]) / 2.0)
    return np.array(dF), np.array(F)


# ---------------------------------------------------------------------------
# Scan + plot
# ---------------------------------------------------------------------------

def run_casimir_scan(width=2, heights=(3, 4, 5, 6, 7), g=1.0, J=2.0,
                     use_rqsvt=True):
    print("=" * 64)
    print(f"CASIMIR SCAN  W={width},  g={g}, J={J}")
    print("=" * 64)

    ds, E0s, methods = [], [], []
    for H in heights:
        d = H - 1
        E_exact, nq = ground_energy_exact(width, H, g, J) \
            if width * H + width * (H - 1) <= EXACT_MAX_QUBITS else (None, None)
        if E_exact is not None:
            E0, method = E_exact, "exact"
        else:
            E0, method = ground_energy_dmrg(width, H, g, J), "dmrg"
            if E0 is None:
                print(f" d={d} (H={H}): skipped (too big for ED, DMRG unavailable)")
                continue
        ds.append(d)
        E0s.append(E0)
        methods.append(method)
        print(f" d={d} (H={H}): E0 = {E0:.6f}   [{method}]")

        if use_rqsvt:
            rq = ground_energy_rqsvt(width, H, g, J)
            if rq is not None:
                print(f"           RQSVT E0 = {rq['energy']:.6f}   "
                      f"(guess overlap {rq['guess_overlap']:.3f})")

    if len(ds) < 4:
        print("\nNeed >=4 separations for the power-law fit; extend `heights` "
              "(DMRG covers the larger ones).")
        return

    alpha, a, b, C = fit_powerlaw(ds, E0s)
    E_cas = np.asarray(E0s, float) - (a + b * np.asarray(ds, float))   # residual, fitted bulk
    dF, F = casimir_force(ds, E_cas)
    print(f"\nFit:  E0(d) ≈ {a:.4f} + {b:.4f} d + ({C:.4f}) d^(-{alpha - 1:.2f})")
    for d, f in zip(dF, F):
        print(f"  measured F(d={int(d)}) = {f:+.5f}")
    print(f"\nExponent:  F ∝ -d^(-{alpha:.2f})    (continuum target alpha = 3)")
    if not 2.0 < alpha < 4.5:
        print("  [!] far from 3 -- likely the wrong regime (too confined -> "
              "exponential, or finite-size dominated). Scan g/J and add separations.")

    _plot(ds, E0s, a, b, C, alpha, E_cas, dF, F, width, g, J)


def _plot(ds, E0s, a, b, C, alpha, E_cas, dF, F, width, g, J):
    ds = np.asarray(ds, float)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(ds, E0s, 'o-', label='E0(d)')
    axes[0].plot(ds, a + b * ds, 'k--', alpha=0.6, label=f'bulk {a:.2f}+{b:.2f}d')
    axes[0].set_xlabel('separation d'); axes[0].set_ylabel('ground energy E0')
    axes[0].set_title('Ground-state energy'); axes[0].legend(); axes[0].grid(alpha=0.3)

    dd = np.linspace(ds.min(), ds.max(), 100)
    axes[1].plot(ds, E_cas, 's', color='purple', label='residual')
    axes[1].plot(dd, C * dd ** (-(alpha - 1)), '-', color='purple', alpha=0.6,
                 label=f'C d^-{alpha - 1:.2f}')
    axes[1].set_xlabel('separation d'); axes[1].set_ylabel('E_Cas(d)')
    axes[1].set_title('Casimir energy'); axes[1].legend(); axes[1].grid(alpha=0.3)

    attractive = F < 0
    if attractive.any():
        axes[2].loglog(dF[attractive], -F[attractive], 'o', color='crimson', label='|F| data')
    F_fit = abs(C) * (alpha - 1) * dd ** (-alpha)            # |dE_Cas/dd| of the fit
    axes[2].loglog(dd, F_fit, 'k--', label=f'fit α={alpha:.2f}')
    axes[2].loglog(dd, F_fit[0] * (dd / dd[0]) ** -3.0, ':', color='gray',
                   alpha=0.7, label='d^-3 guide')
    axes[2].set_xlabel('separation d'); axes[2].set_ylabel('|F(d)|')
    axes[2].set_title('Casimir force (log-log)'); axes[2].legend()
    axes[2].grid(alpha=0.3, which='both')

    fig.suptitle(f'Casimir force, W={width}, g={g}, J={J}  (α={alpha:.2f})')
    plt.tight_layout()
    plt.savefig('casimir_force.png', dpi=150, bbox_inches='tight')
    print("\n[Saved: casimir_force.png]")
    plt.close(fig)


if __name__ == "__main__":
    # Default: W=2 strip, separations d = 2..6. Exact ED for the small ones,
    # DMRG for H>=6. g near the confinement crossover so RQSVT's guess is usable;
    # scan g/J to find where the exponent approaches 3.
    run_casimir_scan(width=2, heights=(3, 4, 5, 6, 7), g=1.0, J=2.0, use_rqsvt=True)
