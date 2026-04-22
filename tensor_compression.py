import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import quimb.tensor as qtn
 
from lattice import LatticeGrid
from QuantumFields.gauge_field import QuantumLinkModel, GaussLaw

def get_ground_state(lattice, coupling, penalty=20.0):
    qlm = QuantumLinkModel(lattice, coupling=coupling, gauss_penalty=penalty)
    H = qlm.build_hamiltonian()
    H_sparse = H.to_matrix(sparse=True)  # scipy sparse matrix, not dense

    # Find only the lowest 2 eigenvalues — uses ~100x less memory than eigh
    eigenvalues, eigenvectors = eigsh(H_sparse, k=2, which='SA')

    gs_energy = eigenvalues[0]
    gs_vector = eigenvectors[:, 0]

    gauss = GaussLaw(lattice)
    is_valid = gauss.verify(gs_vector, tol=1e-4, verbose=False)

    return gs_energy, gs_vector, is_valid
 
    return gs_energy, gs_vector, is_valid

def compress_and_measure(state_vector, n_qubits, chi_values):
    mps = qtn.MatrixProductState.from_dense(state_vector, dims=[2] * n_qubits)
    exact_bonds = mps.bond_sizes()
    vec_exact = mps.to_dense().flatten()

    errors = {}
    for chi in chi_values:
        mps_compressed = mps.copy()
        mps_compressed.compress(max_bond=chi)

        vec_compressed = mps_compressed.to_dense().flatten()
        overlap = np.conj(vec_exact) @ vec_compressed
        fidelity = abs(overlap) ** 2
        errors[chi] = max(1 - fidelity, 0.0)

    return exact_bonds, errors

def run_coupling_scan(lattice, couplings, chi_values):
    results = []
 
    for g2 in couplings:
        print(f"\n--- g² = {g2} ---")
        energy, gs, is_valid = get_ground_state(lattice, coupling=g2)
        print(f"Energy: {energy:.6f}  Gauss: {'✓' if is_valid else '✗'}")
 
        exact_bonds, errors = compress_and_measure(
            gs, lattice.n_qubits, chi_values
        )
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

def run_lattice_scan(lattice_sizes, coupling, chi_values):
    results = []
 
    for W, H_lat in lattice_sizes:
        print(f"\n--- {W}x{H_lat} lattice ({W * H_lat + W * (H_lat-1)} qubits) ---")
        lattice = LatticeGrid(width=W, height=H_lat)
        energy, gs, is_valid = get_ground_state(lattice, coupling=coupling)
        print(f"  Energy: {energy:.6f}  Gauss: {'✓' if is_valid else '✗'}")
 
        exact_bonds, errors = compress_and_measure(
            gs, lattice.n_qubits, chi_values
        )
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


def get_ground_state_dmrg(lattice, coupling, penalty=20.0, max_bond=256):
    n = lattice.n_qubits

    # Stage 1: start at weak coupling where DMRG always works
    start_coupling = 0.05
    print(f"  Stage 1: ground state at g²={start_coupling}")
    qlm_start = QuantumLinkModel(lattice, coupling=start_coupling, gauss_penalty=penalty)
    H_start = qlm_start.build_hamiltonian()
    H_mpo_start = pauli_op_to_mpo(H_start, n)

    psi0 = qtn.MPS_rand_state(n, bond_dim=64, dtype=complex)
    dmrg = qtn.DMRG2(
        H_mpo_start,
        which='SA',
        bond_dims=[64, 128, max_bond],
        cutoffs=1e-12,
        p0=psi0
    )
    dmrg.solve(tol=1e-10, max_sweeps=30)
    current_state = dmrg.state
    print(f"    Energy: {dmrg.energy}")

    # Stage 2: ramp through intermediate couplings to target
    ramp_couplings = np.linspace(start_coupling, coupling, 5)[1:]  # skip the first (already done)
    print(f"  Stage 2: ramping through {[f'{g:.3f}' for g in ramp_couplings]}")

    for g2 in ramp_couplings:
        qlm_step = QuantumLinkModel(lattice, coupling=g2, gauss_penalty=penalty)
        H_step = qlm_step.build_hamiltonian()
        H_mpo_step = pauli_op_to_mpo(H_step, n)

        dmrg = qtn.DMRG2(
            H_mpo_step,
            which='SA',
            bond_dims=[128, max_bond],
            cutoffs=1e-12,
            p0=current_state
        )
        dmrg.solve(tol=1e-10, max_sweeps=20)
        current_state = dmrg.state
        print(f"    g²={g2:.3f}  energy={dmrg.energy}")

    return dmrg.energy, dmrg.state


def run_dmrg_with_mpo(H_mpo, max_bond=64):
    dmrg = qtn.DMRG2(H_mpo, bond_dims=[10, 20, 40, max_bond])
    dmrg.solve(tol=1e-9, max_sweeps=20)
    return dmrg.energy, dmrg.state


def pauli_op_to_mpo(pauli_op, n_qubits):
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
            # First site: no left bond → shape (bond_right, phys_up, phys_down)
            # lrud with no l → rud
            arr = np.zeros((n_terms, 2, 2), dtype=complex)
            for t, (pauli_str, coeff) in enumerate(
                    zip(pauli_op.paulis, pauli_op.coeffs)):
                label = pauli_str.to_label()[::-1]
                arr[t, :, :] = pauli_matrices[label[site]] * coeff

        elif site == n_qubits - 1:
            # Last site: no right bond → shape (bond_left, phys_up, phys_down)
            # lrud with no r → lud
            arr = np.zeros((n_terms, 2, 2), dtype=complex)
            for t, (pauli_str, _) in enumerate(
                    zip(pauli_op.paulis, pauli_op.coeffs)):
                label = pauli_str.to_label()[::-1]
                arr[t, :, :] = pauli_matrices[label[site]]

        else:
            # Middle sites: shape (bond_left, bond_right, phys_up, phys_down)
            arr = np.zeros((n_terms, n_terms, 2, 2), dtype=complex)
            for t, (pauli_str, _) in enumerate(
                    zip(pauli_op.paulis, pauli_op.coeffs)):
                label = pauli_str.to_label()[::-1]
                arr[t, t, :, :] = pauli_matrices[label[site]]

        site_arrays.append(arr)

    return qtn.MatrixProductOperator(site_arrays, shape='lrud')


def validate_dmrg(lattice, coupling):
    """
    Compare exact diag vs DMRG on the same lattice.
    """
    print(f"\n--- Validating DMRG (g²={coupling}) ---")

    # Exact
    energy_exact, gs_exact, valid = get_ground_state(lattice, coupling)
    print(f"  Exact energy:  {energy_exact:.8f}")

    # DMRG
    energy_dmrg, gs_mps = get_ground_state_dmrg(lattice, coupling)
    print(f"  DMRG energy:   {energy_dmrg:.8f}")
    print(f"  Difference:    {abs(energy_exact - energy_dmrg):.2e}")
    print(f"  MPS bonds:     {gs_mps.bond_sizes()}")

    # Fidelity
    vec_dmrg = gs_mps.to_dense().flatten()
    fidelity = abs(np.conj(gs_exact) @ vec_dmrg) ** 2
    print(f"  Fidelity:      {fidelity:.10f}")

    return energy_exact, energy_dmrg, fidelity
 
 
# =============================================================================
# Plotting
# =============================================================================
 
def plot_coupling_scan(results, chi_values):
    """Compression error vs χ, one curve per coupling."""
    fig, ax = plt.subplots(figsize=(8, 6))
 
    for r in results:
        errs = [r['errors'][chi] for chi in chi_values]
        label = f"g²={r['coupling']} (max bond={r['max_exact_bond']})"
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
 
    n_results = len(results)
    width = 0.8 / n_results
 
    for idx, r in enumerate(results):
        bonds = r['exact_bonds']
        positions = np.arange(len(bonds)) + idx * width
        ax.bar(positions, bonds, width, label=f"g²={r['coupling']}", alpha=0.8)
 
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
 
    # Experiment 1: Coupling scan on 2×3 lattice
    print("=" * 60)
    print("EXPERIMENT 1: Compression vs coupling (2x3 lattice)")
    print("=" * 60)
 
    lattice = LatticeGrid(width=2, height=3)
    couplings = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
 
    coupling_results = run_coupling_scan(lattice, couplings, chi_values)
    plot_coupling_scan(coupling_results, chi_values)
    plot_bond_dimensions(coupling_results)
 
    # Experiment 2: Lattice size scan at fixed coupling
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Compression vs lattice size (g²=1.0)")
    print("=" * 60)
 
    lattice_sizes = [(2, 3)]  # add (3, 4) if you have the RAM
    lattice_results = run_lattice_scan(lattice_sizes, coupling=1.0, chi_values=chi_values)
    plot_lattice_scan(lattice_results, chi_values)

    # Experiment 3: Validate DMRG against exact diag
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: DMRG validation")
    print("=" * 60)

    lattice_2x3 = LatticeGrid(width=2, height=3)
    validate_dmrg(lattice_2x3, coupling=1.0)
    validate_dmrg(lattice_2x3, coupling=0.1)

    lattice_3x3 = LatticeGrid(width=3, height=3)
    validate_dmrg(lattice_3x3, coupling=1.0)
    validate_dmrg(lattice_2x3, coupling=0.1)

    lattice_3x4 = LatticeGrid(width=3, height=4)
    print(f"\n3x4 lattice: {lattice_3x4.n_qubits} qubits")
    energy_hc, gs_mps_hc = get_ground_state_dmrg(lattice_3x4, coupling=1.0, max_bond=128)
    energy_lc, gs_mps_lc = get_ground_state_dmrg(lattice_3x4, coupling=0.1, max_bond=128)

    print(f"Energy High Coupling: {energy_hc}")
    print(f"MPS bonds High Coupling: {gs_mps_hc.bond_sizes()}")
    print(f"Max bond High Coupling: {max(gs_mps_hc.bond_sizes())}")

    print(f"Energy Low Coupling: {energy_lc}")
    print(f"MPS bonds Low Coupling: {gs_mps_lc.bond_sizes()}")
    print(f"Max bond Low Coupling: {max(gs_mps_lc.bond_sizes())}")

    for g2 in [1.0, 0.1]:
        print(f"\n--- g² = {g2} ---")
        energy_pen, _ = get_ground_state_dmrg(lattice_3x4, coupling=g2,
                                            penalty=20.0, max_bond=128)
        energy_nopen, _ = get_ground_state_dmrg(lattice_3x4, coupling=g2,
                                                penalty=0.0, max_bond=128)
        print(f"  With penalty:       {energy_pen}")
        print(f"  Without penalty:    {energy_nopen}")
        print(f"  Penalty contribution: {energy_pen - energy_nopen}")

    lattice_4x4 = LatticeGrid(width=4, height=4)
    print(f"\n4x4 lattice: {lattice_4x4.n_qubits} qubits")
    energy_hc, gs_mps_hc = get_ground_state_dmrg(lattice_4x4, coupling=1.0, max_bond=512)
    energy_lc, gs_mps_lc = get_ground_state_dmrg(lattice_4x4, coupling=0.1, max_bond=512)

    print(f"Energy High Coupling: {energy_hc}")
    print(f"MPS bonds High Coupling: {gs_mps_hc.bond_sizes()}")
    print(f"Max bond High Coupling: {max(gs_mps_hc.bond_sizes())}")

    print(f"Energy Low Coupling: {energy_lc}")
    print(f"MPS bonds Low Coupling: {gs_mps_lc.bond_sizes()}")
    print(f"Max bond Low Coupling: {max(gs_mps_lc.bond_sizes())}")

    for g2 in [1.0, 0.1]:
        print(f"\n--- g² = {g2} ---")
        energy_pen, _ = get_ground_state_dmrg(lattice_4x4, coupling=g2,
                                            penalty=20.0, max_bond=512)
        energy_nopen, _ = get_ground_state_dmrg(lattice_4x4, coupling=g2,
                                                penalty=0.0, max_bond=512)
        electric_shift = g2 / 8.0 * lattice_4x4.n_links_total
        print(f"  With penalty:       {energy_pen}")
        print(f"  Without penalty:    {energy_nopen}")
        print(f"  Penalty contribution: {energy_pen - energy_nopen}")
        print(f"  Electric shift:     {electric_shift}")
 