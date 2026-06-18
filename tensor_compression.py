"""Tensor-network experiments for U(1) QLM ground states.

Two classical-reference tracks used to validate the quantum pipeline:
  - MPS compression: how compressible an exactly-diagonalised ground state is
    as a function of coupling and lattice size (entanglement diagnostics).
  - DMRG (quimb): ground-state energies for lattices beyond sparse ED, via a
    Pauli-operator -> MPO conversion and a coupling-ramp warm start.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
import quimb.tensor as qtn

from lattice import LatticeGrid
from QuantumFields.gauge_field import QuantumLinkModel, GaussLaw


# =============================================================================
# Exact diagonalisation + MPS compression
# =============================================================================

def get_ground_state(lattice, g, penalty=20.0):
    qlm = QuantumLinkModel(lattice, g=g, gauss_penalty=penalty)
    H_sparse = qlm.build_hamiltonian().to_matrix(sparse=True)

    # Lowest 2 eigenvalues only — far cheaper in memory than dense eigh.
    eigenvalues, eigenvectors = eigsh(H_sparse, k=2, which='SA')
    gs_energy = eigenvalues[0]
    gs_vector = eigenvectors[:, 0]

    is_valid = GaussLaw(lattice).verify(gs_vector, tol=1e-4, verbose=False)
    return gs_energy, gs_vector, is_valid


def compress_and_measure(state_vector, n_qubits, chi_values):
    """Compress the state to each bond dimension chi and report 1 - fidelity."""
    mps = qtn.MatrixProductState.from_dense(state_vector, dims=[2] * n_qubits)
    exact_bonds = mps.bond_sizes()
    vec_exact = mps.to_dense().flatten()

    errors = {}
    for chi in chi_values:
        mps_compressed = mps.copy()
        mps_compressed.compress(max_bond=chi)
        vec_compressed = mps_compressed.to_dense().flatten()
        fidelity = abs(np.conj(vec_exact) @ vec_compressed) ** 2
        errors[chi] = max(1 - fidelity, 0.0)

    return exact_bonds, errors


def run_coupling_scan(lattice, couplings, chi_values):
    results = []
    for g2 in couplings:
        print(f"\n--- g = {g2} ---")
        energy, gs, is_valid = get_ground_state(lattice, g=g2)
        print(f"Energy: {energy:.6f}  Gauss: {'✓' if is_valid else '✗'}")

        exact_bonds, errors = compress_and_measure(gs, lattice.n_qubits, chi_values)
        print(f"Exact bond dims: {exact_bonds}")
        for chi in chi_values:
            print(f"χ={chi:4d}  error={errors[chi]:.2e}")

        results.append({
            'coupling': g2,
            'energy': energy,
            'gauge_invariant': is_valid,
            'exact_bonds': exact_bonds,
            'max_exact_bond': max(exact_bonds),
            'errors': errors,
        })
    return results


def run_lattice_scan(lattice_sizes, g, chi_values):
    results = []
    for W, H_lat in lattice_sizes:
        lattice = LatticeGrid(width=W, height=H_lat)
        print(f"\n--- {W}x{H_lat} lattice ({lattice.n_qubits} qubits) ---")
        energy, gs, is_valid = get_ground_state(lattice, g=g)
        print(f"  Energy: {energy:.6f}  Gauss: {'✓' if is_valid else '✗'}")

        exact_bonds, errors = compress_and_measure(gs, lattice.n_qubits, chi_values)
        print(f"Exact bond dims: {exact_bonds}")
        print(f"Max bond: {max(exact_bonds)}")
        for chi in chi_values:
            print(f"χ={chi:4d}  error={errors[chi]:.2e}")

        results.append({
            'lattice_size': f"{W}×{H_lat}",
            'n_qubits': lattice.n_qubits,
            'energy': energy,
            'exact_bonds': exact_bonds,
            'max_exact_bond': max(exact_bonds),
            'errors': errors,
        })
    return results


# =============================================================================
# DMRG (quimb)
# =============================================================================

def pauli_op_to_mpo(pauli_op, n_qubits):
    """Convert a Qiskit SparsePauliOp into a quimb MatrixProductOperator."""
    pauli_matrices = {
        'I': np.eye(2, dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex),
    }

    n_terms = len(pauli_op)
    site_arrays = []

    for site in range(n_qubits):
        if site == 0:
            # First site: no left bond -> (bond_right, phys_up, phys_down).
            arr = np.zeros((n_terms, 2, 2), dtype=complex)
            for t, (pauli_str, coeff) in enumerate(
                    zip(pauli_op.paulis, pauli_op.coeffs)):
                label = pauli_str.to_label()[::-1]
                arr[t, :, :] = pauli_matrices[label[site]] * coeff

        elif site == n_qubits - 1:
            # Last site: no right bond -> (bond_left, phys_up, phys_down).
            arr = np.zeros((n_terms, 2, 2), dtype=complex)
            for t, (pauli_str, _) in enumerate(
                    zip(pauli_op.paulis, pauli_op.coeffs)):
                label = pauli_str.to_label()[::-1]
                arr[t, :, :] = pauli_matrices[label[site]]

        else:
            # Middle sites: (bond_left, bond_right, phys_up, phys_down).
            arr = np.zeros((n_terms, n_terms, 2, 2), dtype=complex)
            for t, (pauli_str, _) in enumerate(
                    zip(pauli_op.paulis, pauli_op.coeffs)):
                label = pauli_str.to_label()[::-1]
                arr[t, t, :, :] = pauli_matrices[label[site]]

        site_arrays.append(arr)

    return qtn.MatrixProductOperator(site_arrays, shape='lrud')


def get_ground_state_dmrg(lattice, g, penalty=20.0, max_bond=256):
    """DMRG ground state, warm-started from weak coupling and ramped up."""
    n = lattice.n_qubits

    # Stage 1: solve at weak coupling, where DMRG converges easily.
    start_coupling = 0.05
    print(f"  Stage 1: ground state at g={start_coupling}")
    H_start = QuantumLinkModel(lattice, g=start_coupling,
                               gauss_penalty=penalty).build_hamiltonian()
    psi0 = qtn.MPS_rand_state(n, bond_dim=64, dtype=complex)
    dmrg = qtn.DMRG2(pauli_op_to_mpo(H_start, n), which='SA',
                     bond_dims=[64, 128, max_bond], cutoffs=1e-12, p0=psi0)
    dmrg.solve(tol=1e-10, max_sweeps=30)
    current_state = dmrg.state
    print(f"    Energy: {dmrg.energy}")

    # Stage 2: ramp through intermediate couplings to the target.
    ramp_couplings = np.linspace(start_coupling, g, 5)[1:]
    print(f"  Stage 2: ramping through {[f'{g:.3f}' for g in ramp_couplings]}")
    for g2 in ramp_couplings:
        H_step = QuantumLinkModel(lattice, g=g2,
                                  gauss_penalty=penalty).build_hamiltonian()
        dmrg = qtn.DMRG2(pauli_op_to_mpo(H_step, n), which='SA',
                         bond_dims=[128, max_bond], cutoffs=1e-12, p0=current_state)
        dmrg.solve(tol=1e-10, max_sweeps=20)
        current_state = dmrg.state
        print(f"    g={g2:.3f}  energy={dmrg.energy}")

    return dmrg.energy, dmrg.state


def validate_dmrg(lattice, g):
    """Cross-check exact diagonalisation against DMRG on the same lattice."""
    print(f"\n--- Validating DMRG (g={g}) ---")

    energy_exact, gs_exact, _ = get_ground_state(lattice, g)
    print(f"  Exact energy:  {energy_exact:.8f}")

    energy_dmrg, gs_mps = get_ground_state_dmrg(lattice, g)
    print(f"  DMRG energy:   {energy_dmrg:.8f}")
    print(f"  Difference:    {abs(energy_exact - energy_dmrg):.2e}")
    print(f"  MPS bonds:     {gs_mps.bond_sizes()}")

    vec_dmrg = gs_mps.to_dense().flatten()
    fidelity = abs(np.conj(gs_exact) @ vec_dmrg) ** 2
    print(f"  Fidelity:      {fidelity:.10f}")
    return energy_exact, energy_dmrg, fidelity


def explore_dmrg_lattice(lattice, max_bond):
    """DMRG energies at high/low coupling and the Gauss-penalty contribution."""
    print(f"\n{lattice.width}x{lattice.height} lattice: {lattice.n_qubits} qubits")

    for label, g2 in [("High", 1.0), ("Low", 0.1)]:
        energy, gs_mps = get_ground_state_dmrg(lattice, g=g2, max_bond=max_bond)
        print(f"Energy {label} Coupling: {energy}")
        print(f"MPS bonds {label} Coupling: {gs_mps.bond_sizes()}")
        print(f"Max bond {label} Coupling: {max(gs_mps.bond_sizes())}")

    for g2 in [1.0, 0.1]:
        print(f"\n--- g = {g2} ---")
        energy_pen, _ = get_ground_state_dmrg(lattice, g=g2,
                                              penalty=20.0, max_bond=max_bond)
        energy_nopen, _ = get_ground_state_dmrg(lattice, g=g2,
                                                penalty=0.0, max_bond=max_bond)
        print(f"  With penalty:         {energy_pen}")
        print(f"  Without penalty:      {energy_nopen}")
        print(f"  Penalty contribution: {energy_pen - energy_nopen}")


# =============================================================================
# Plotting
# =============================================================================

def plot_coupling_scan(results, chi_values):
    """Compression error vs χ, one curve per coupling."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        errs = [r['errors'][chi] for chi in chi_values]
        label = f"g={r['coupling']} (max bond={r['max_exact_bond']})"
        ax.semilogy(chi_values, errs, 'o-', label=label, markersize=5)

    ax.set_xlabel('Bond dimension χ', fontsize=12)
    ax.set_ylabel('Compression error (1 − fidelity)', fontsize=12)
    ax.set_title('MPS compression of U(1) QLM ground states', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(chi_values)
    plt.tight_layout()
    plt.show()


def plot_lattice_scan(results, chi_values):
    """Compression error vs χ, one curve per lattice size."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        errs = [r['errors'][chi] for chi in chi_values]
        label = f"{r['lattice_size']} ({r['n_qubits']}q, max bond={r['max_exact_bond']})"
        ax.semilogy(chi_values, errs, 's-', label=label, markersize=5)

    ax.set_xlabel('Bond dimension χ', fontsize=12)
    ax.set_ylabel('Compression error (1 − fidelity)', fontsize=12)
    ax.set_title('MPS compression scaling with lattice size', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(chi_values)
    plt.tight_layout()
    plt.show()


def plot_bond_dimensions(results):
    """Bar chart of exact bond dimensions for each coupling."""
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / len(results)
    for idx, r in enumerate(results):
        bonds = r['exact_bonds']
        positions = np.arange(len(bonds)) + idx * width
        ax.bar(positions, bonds, width, label=f"g={r['coupling']}", alpha=0.8)

    ax.set_xlabel('Bond index (cut position)', fontsize=12)
    ax.set_ylabel('Exact bond dimension', fontsize=12)
    ax.set_title('Entanglement structure — exact MPS bond dimensions', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    chi_values = [2, 4, 8, 16, 32, 64]

    # Experiment 1: compression vs coupling on a 2×3 lattice.
    print("=" * 60)
    print("EXPERIMENT 1: Compression vs coupling (2x3 lattice)")
    print("=" * 60)
    lattice_2x3 = LatticeGrid(width=2, height=3)
    couplings = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    coupling_results = run_coupling_scan(lattice_2x3, couplings, chi_values)
    plot_coupling_scan(coupling_results, chi_values)
    plot_bond_dimensions(coupling_results)

    # Experiment 2: compression vs lattice size at fixed coupling.
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Compression vs lattice size (g=1.0)")
    print("=" * 60)
    lattice_results = run_lattice_scan([(2, 3)], g=1.0, chi_values=chi_values)
    plot_lattice_scan(lattice_results, chi_values)

    # Experiment 3: validate DMRG against exact diagonalisation, then explore
    # larger lattices that are out of reach for sparse ED.
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: DMRG validation")
    print("=" * 60)
    validate_dmrg(lattice_2x3, g=1.0)
    validate_dmrg(lattice_2x3, g=0.1)
    validate_dmrg(LatticeGrid(width=3, height=3), g=1.0)

    explore_dmrg_lattice(LatticeGrid(width=3, height=4), max_bond=128)
    explore_dmrg_lattice(LatticeGrid(width=4, height=4), max_bond=512)
