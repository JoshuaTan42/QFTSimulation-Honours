"""Tensor-network reference track for U(1) QLM ground states (thesis Sec. 2.3.2).

  * DMRG (quimb) on the penalised qubit Hamiltonian, with the 2D lattice mapped
    onto a 1D chain in link-qubit order. The Pauli-sum MPO is compressed before
    use (the naive construction has one bond per Pauli term).
  * MPS compression of exactly diagonalised ground states, as an entanglement
    diagnostic for how far the 1D mapping can be pushed.

Sector ED (sector.py) is the primary classical reference. DMRG is the
independent cross-check required by Sec. 2.4.2 and the route to lattices whose
gauge-invariant sector no longer fits in memory.
"""

from __future__ import annotations

import warnings

import numpy as np

from lattice import LatticeGrid
from QuantumFields.gauge_field import QuantumLinkModel
from sector import GaugeSector

_PAULI = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}


def _qtn():
    import quimb.tensor as qtn
    return qtn


def pauli_op_to_mpo(pauli_op, n_qubits, compress_cutoff: float = 1e-12):
    """Qiskit SparsePauliOp -> quimb MPO (site k = qubit k), then compressed."""
    qtn = _qtn()
    labels = [p.to_label()[::-1] for p in pauli_op.paulis]      # label[k] acts on qubit k
    coeffs = np.asarray(pauli_op.coeffs)
    T = len(labels)
    arrays = []
    for site in range(n_qubits):
        if site == 0:
            arr = np.zeros((T, 2, 2), dtype=complex)
            for t in range(T):
                arr[t] = coeffs[t] * _PAULI[labels[t][site]]
        elif site == n_qubits - 1:
            arr = np.zeros((T, 2, 2), dtype=complex)
            for t in range(T):
                arr[t] = _PAULI[labels[t][site]]
        else:
            arr = np.zeros((T, T, 2, 2), dtype=complex)
            for t in range(T):
                arr[t, t] = _PAULI[labels[t][site]]
        arrays.append(arr)
    mpo = qtn.MatrixProductOperator(arrays, shape='lrud')
    if compress_cutoff:
        mpo.compress(cutoff=compress_cutoff)
    return mpo


def get_ground_state_dmrg(lattice, g=0.0, J=1.0, rk=0.0, field_pattern="uniform",
                          plates="free", penalty=20.0, bond_schedule=(16, 32, 64, 128),
                          rel_tol=1e-6, seed=0, verbose=False):
    """DMRG ground state with the bond dimension raised until the energy settles.

    The bond dimension is increased along `bond_schedule` until the relative
    energy change between two consecutive schedule entries is below rel_tol
    (the convergence rule of Sec. 2.3.2).
    """
    qtn = _qtn()
    n = lattice.n_qubits
    H = QuantumLinkModel(lattice, g=g, J=J, rk=rk, gauss_penalty=penalty,
                         field_pattern=field_pattern, plates=plates).build_hamiltonian()
    mpo = pauli_op_to_mpo(H, n)
    state = qtn.MPS_rand_state(n, bond_dim=bond_schedule[0], dtype=complex, seed=seed)
    history = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for chi in bond_schedule:
            dmrg = qtn.DMRG2(mpo, which='SA', bond_dims=[chi], cutoffs=1e-12, p0=state)
            dmrg.solve(tol=1e-11, max_sweeps=30, verbosity=0)
            state = dmrg.state
            history.append((chi, float(np.real(dmrg.energy))))
            if verbose:
                print(f"    chi={chi:4d}  E={history[-1][1]:.10f}  "
                      f"max bond used={max(state.bond_sizes())}")
            if len(history) >= 2:
                e_new, e_old = history[-1][1], history[-2][1]
                if abs(e_new - e_old) <= rel_tol * max(1.0, abs(e_new)):
                    break
    return history[-1][1], state, {'history': history, 'mpo_bond': mpo.max_bond()}


def validate_dmrg(lattice, g=0.0, J=1.0, rk=0.0, plates="free", verbose=True) -> dict:
    """Classical-classical cross-check: sector ED against DMRG (Sec. 2.4.2, item 1)."""
    exact = GaugeSector(lattice, plates=plates).ground_state(g=g, J=J, rk=rk)['energy']
    energy, state, info = get_ground_state_dmrg(lattice, g=g, J=J, rk=rk, plates=plates)
    rel = abs(energy - exact) / max(1.0, abs(exact))
    if verbose:
        print(f"  {lattice.width}x{lattice.height} ({lattice.n_qubits} qubits, {plates} plates): "
              f"ED {exact:.10f}  DMRG {energy:.10f}  rel. diff {rel:.1e}  "
              f"MPO bond {info['mpo_bond']}  MPS bond {max(state.bond_sizes())}")
    return {'exact': exact, 'dmrg': energy, 'relative_difference': rel, **info}


# ---------------------------------------------------------------------------
# MPS compression of exact ground states
# ---------------------------------------------------------------------------

def compress_and_measure(state_vector, n_qubits, chi_values) -> tuple:
    """Infidelity 1 - |<psi|psi_chi>|^2 after truncating to bond dimension chi.

    Qiskit statevectors are little-endian, so the tensor is reshaped with
    qubit N_q - 1 first and the axes reversed to make MPS site k = qubit k.
    """
    qtn = _qtn()
    psi = np.asarray(state_vector, dtype=complex).reshape([2] * n_qubits)
    psi = psi.transpose(*reversed(range(n_qubits))).reshape(-1)
    mps = qtn.MatrixProductState.from_dense(psi, dims=[2] * n_qubits)
    exact_bonds = list(mps.bond_sizes())
    errors = {}
    for chi in chi_values:
        truncated = mps.copy()
        truncated.compress(max_bond=chi)
        truncated.normalize()
        fidelity = abs(mps.H @ truncated) ** 2
        errors[chi] = max(1.0 - float(np.real(fidelity)), 0.0)
    return exact_bonds, errors


def run_compression_scan(lattice_sizes, chi_values=(2, 4, 8, 16, 32), J=1.0, rk=0.0,
                         verbose=True) -> list:
    results = []
    for W, H in lattice_sizes:
        lattice = LatticeGrid(W, H)
        plates = "fixed" if W % 2 == 0 else "free"
        sector = GaugeSector(lattice, plates=plates)
        gs = sector.ground_state(J=J, rk=rk)
        bonds, errors = compress_and_measure(sector.embed(gs['state']),
                                             lattice.n_qubits, chi_values)
        results.append({'lattice': f"{W}x{H}", 'qubits': lattice.n_qubits,
                        'max_exact_bond': max(bonds), 'errors': errors})
        if verbose:
            print(f"  {W}x{H}: exact max bond {max(bonds)};  " +
                  "  ".join(f"chi={c}: {e:.1e}" for c, e in errors.items()))
    return results


if __name__ == "__main__":
    print("DMRG validation against sector ED")
    validate_dmrg(LatticeGrid(2, 2), plates="fixed")
    validate_dmrg(LatticeGrid(2, 3), plates="fixed")
    validate_dmrg(LatticeGrid(3, 3), plates="free")
    print("\nMPS compressibility of exact ground states")
    run_compression_scan([(2, 3), (3, 3), (4, 3)])
