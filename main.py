from lattice import LatticeGrid
from QuantumFields.gauge_field import GaussLaw, QuantumLinkModel
from circuit import DynamicalSimulation, StructuredTrotter, TrotterEvolution
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh
from qiskit.quantum_info import SparsePauliOp
from qiskit import transpile
from qiskit_aer import AerSimulator

# ============================================================
# Setup
# ============================================================
lattice = LatticeGrid(width=2, height=3)
gauss = GaussLaw(lattice)

lattice.debug()
lattice.visualize_ascii()

# ============================================================
# Verify [G, H] = 0
# ============================================================
print("\n=== Gauge invariance check ===")
qlm_check = QuantumLinkModel(lattice, kappa=1.0, mass=0.5, g_squared=1.0, J=2.0)
H_check = qlm_check.build_hamiltonian()
gauss_ops = gauss.build_gauss_operators()
for idx, G_op in enumerate(gauss_ops):
    if G_op is None:
        continue
    i, j = lattice.site_to_coords(idx)
    comm = (G_op @ H_check - H_check @ G_op).simplify()
    comm_norm = sum(abs(c) for c in comm.coeffs)
    print(f"  ||[G({i},{j}), H]|| = {comm_norm:.2e}")

# ============================================================
# Circuit depth comparison
# ============================================================
print("\n=== Circuit depth comparison ===")
structured = StructuredTrotter(lattice, kappa=1.0, mass=0.5, g_squared=1.0, J=1.0, dt=0.05)
step_s = structured.build_step()
step_s = transpile(step_s, basis_gates=['cx','rz','ry','rx','h','x','y','z','s','sdg'], optimization_level=3)
print(f"Structured: depth={step_s.depth()}, CNOTs={step_s.count_ops().get('cx', 0)}")

generic = TrotterEvolution(H_check, lattice.n_qubits, total_time=0.05, n_steps=1)
step_g = generic.build_circuit()
step_g = transpile(step_g, basis_gates=['cx','rz','ry','rx','h','x','y','z','s','sdg'], optimization_level=3)
print(f"Generic:    depth={step_g.depth()}, CNOTs={step_g.count_ops().get('cx', 0)}")

# ============================================================
# Find gauge-invariant ground state
# ============================================================
print("\n=== Finding gauge-invariant ground state ===")
qlm_strong = QuantumLinkModel(lattice, kappa=1.0, mass=0.5, g_squared=1.0, J=2.0,
                               gauss_penalty=200.0)
H_strong = qlm_strong.build_hamiltonian()
H_sparse = H_strong.to_matrix(sparse=True)
eigenvalues, eigenvectors = eigsh(H_sparse, k=2, which='SA')
gs_vector = eigenvectors[:, 0]

print(f"Ground state energy: {eigenvalues[0]:.6f}")
print(f"First excited:       {eigenvalues[1]:.6f}")
print(f"Gap:                 {eigenvalues[1] - eigenvalues[0]:.6f}")
gauss.verify(gs_vector, tol=1e-3)

# ============================================================
# Quench dynamics
# ============================================================
print("\n=== Quench dynamics (J=2.0 -> J=0.0) ===")
evolve_kappa = 0.1
evolve_mass = 2.0
evolve_g2 = 1.0
evolve_J = 2.0

qlm_evolve = QuantumLinkModel(lattice, kappa=evolve_kappa, mass=evolve_mass,
                               g_squared=evolve_g2, J=evolve_J)
H_evolve = qlm_evolve.build_hamiltonian()

sv_check = np.conj(gs_vector) @ H_evolve.to_matrix(sparse=True) @ gs_vector
print(f"Terms: {len(H_evolve)}")
print(f"Quench energy: {sv_check.real:.6f}")

sim = DynamicalSimulation(
    H_evolve, lattice,
    kappa=evolve_kappa,
    mass=evolve_mass,
    g_squared=evolve_g2,
    J=evolve_J,
    total_time=20.0,
    n_steps=400,
    initial_state=gs_vector,
    use_structured_trotter=True
)

results = sim.run()

backend = AerSimulator(method='statevector')
results_aer = sim.run_on_backend(backend, shots=4096,
                                  time_points=[0, 50, 100, 200, 300, 400])

# ============================================================
# Plots
# ============================================================

# Plot 1: Statevector observables (from exact run)
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

axes[0].plot(results['times'], results['energies'], 'b-')
axes[0].set_ylabel('Energy')
axes[0].set_title(f'Full QED quench ({lattice.width}x{lattice.height}, '
                   f'{lattice.n_qubits}q: {lattice.n_matter_qubits}m + {lattice.n_gauge_qubits}g)')

axes[1].plot(results['times'], results['gauss_violations'], 'r-')
axes[1].set_ylabel('Gauss violation')
axes[1].set_yscale('log')

axes[2].plot(results['times'], results['plaquette_values'], 'g-')
axes[2].set_ylabel('Plaquette')
axes[2].set_xlabel('Time t')

plt.tight_layout()
plt.savefig('qed_observables.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot 2: Electric field comparison — statevector vs Aer
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Statevector (exact)
e_sv = np.array(results['e_field_profiles'])
im1 = axes[0].imshow(e_sv.T, aspect='auto', cmap='RdBu_r',
                      extent=[0, results['times'][-1], -0.5, lattice.n_gauge_qubits - 0.5],
                      origin='lower')
axes[0].set_title('Statevector (exact)')
axes[0].set_ylabel('Gauge link index')
axes[0].set_xlabel('Time t')
plt.colorbar(im1, ax=axes[0], label='E')

# Aer (shots)
aer_times = sorted(results_aer.keys())
# Extract only gauge link qubits from Aer results (indices n_matter..n_total-1)
e_aer = np.array([[results_aer[t]['e_field'][q] for q in range(lattice.n_matter_qubits, lattice.n_qubits)]
                   for t in aer_times])
im2 = axes[1].imshow(e_aer.T, aspect='auto', cmap='RdBu_r',
                      extent=[aer_times[0], aer_times[-1], -0.5, lattice.n_gauge_qubits - 0.5],
                      origin='lower')
axes[1].set_title(f'Aer ({4096} shots)')
axes[1].set_ylabel('Gauge link index')
axes[1].set_xlabel('Time t')
plt.colorbar(im2, ax=axes[1], label='E')

plt.tight_layout()
plt.savefig('qed_efield_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot 3: Matter site occupation from Aer
fig, ax = plt.subplots(figsize=(10, 5))
matter_occ = np.array([[results_aer[t]['e_field'][q] for q in range(lattice.n_matter_qubits)]
                        for t in aer_times])
# Convert Z/2 expectation to occupation: n = 0.5 - Z/2
matter_n = 0.5 - matter_occ

for site_idx in range(lattice.n_matter_qubits):
    i, j = lattice.site_to_coords(site_idx)
    ax.plot(aer_times, matter_n[:, site_idx], 'o-', markersize=4,
            label=f's{site_idx} ({i},{j})')
ax.set_xlabel('Time t')
ax.set_ylabel('Occupation n_s')
ax.set_title('Matter site occupation (Aer)')
ax.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.savefig('qed_matter_occupation.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot 4: Circuit depth scaling
fig, ax = plt.subplots(figsize=(8, 5))
depths = [results_aer[t]['depth'] for t in aer_times]
two_q = [results_aer[t]['two_qubit_gates'] for t in aer_times]

ax.plot(aer_times, depths, 'o-', color='coral', label='Circuit depth')
ax2 = ax.twinx()
ax2.plot(aer_times, two_q, 's-', color='steelblue', label='2-qubit gates')
ax.set_xlabel('Simulation time t')
ax.set_ylabel('Circuit depth', color='coral')
ax2.set_ylabel('Two-qubit gates', color='steelblue')
ax.set_title('Circuit resource scaling')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig('qed_circuit_scaling.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Summary
# ============================================================
print("\n=== Summary ===")
print(f"Lattice: {lattice.width}x{lattice.height} "
      f"({lattice.n_matter_qubits}m + {lattice.n_gauge_qubits}g = {lattice.n_qubits}q)")
print(f"Structured depth: {step_s.depth()}, Generic depth: {step_g.depth()}")
print(f"Energy conservation: +/-{(max(results['energies']) - min(results['energies']))/2:.2e}")
print(f"Max Gauss violation: {max(results['gauss_violations']):.2e}")
print(f"Aer time points: {len(results_aer)}")
print(f"Max circuit depth (Aer): {max(depths)}")
print(f"Max 2-qubit gates (Aer): {max(two_q)}")