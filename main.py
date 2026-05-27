from lattice import LatticeGrid
from QuantumFields.gauge_field import GaussLaw, QuantumLinkModel
from circuit import DynamicalSimulation, StructuredTrotter, TrotterEvolution
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh
from qiskit import transpile
from qiskit_aer import AerSimulator

lattice = LatticeGrid(width=3, height=3)
gauss = GaussLaw(lattice)

lattice.debug()
lattice.visualize_ascii()

print("\n=== Gauge invariance check ===")
qlm_check = QuantumLinkModel(lattice, g_squared=6.0, J=2.0)
H_check = qlm_check.build_hamiltonian()
gauss_ops = gauss.build_gauss_operators()
for idx, G_op in enumerate(gauss_ops):
    if G_op is None:
        continue
    i, j = lattice.site_to_coords(idx)
    comm = (G_op @ H_check - H_check @ G_op).simplify()
    comm_norm = sum(abs(c) for c in comm.coeffs)
    print(f"  ||[G({i},{j}), H]|| = {comm_norm:.2e}")

print("\n=== Circuit depth comparison ===")
structured = StructuredTrotter(lattice, g_squared=6.0, J=2.0, dt=0.05)
step_s = structured.build_step()
step_s = transpile(step_s, basis_gates=['cx','rz','ry','rx','h','x','y','z','s','sdg'], optimization_level=3)
print(f"Structured: depth={step_s.depth()}, CNOTs={step_s.count_ops().get('cx', 0)}")

generic = TrotterEvolution(H_check, lattice.n_qubits, total_time=0.05, n_steps=1)
step_g = generic.build_circuit()
step_g = transpile(step_g, basis_gates=['cx','rz','ry','rx','h','x','y','z','s','sdg'], optimization_level=3)
print(f"Generic:    depth={step_g.depth()}, CNOTs={step_g.count_ops().get('cx', 0)}")

print("\n=== Finding gauge-invariant ground state ===")
g_initial = 6.0
J_initial = 2.0
penalty = 200.0

qlm_ground = QuantumLinkModel(lattice, g_squared=g_initial, J=J_initial, gauss_penalty=penalty)
H_ground = qlm_ground.build_hamiltonian()
H_sparse = H_ground.to_matrix(sparse=True)
eigenvalues, eigenvectors = eigsh(H_sparse, k=2, which='SA')
gs_vector = eigenvectors[:, 0]

print(f"Initial parameters: g²={g_initial}, J={J_initial}")
print(f"Ground state energy: {eigenvalues[0]:.6f}")
print(f"First excited:       {eigenvalues[1]:.6f}")
print(f"Gap:                 {eigenvalues[1] - eigenvalues[0]:.6f}")
gauss.verify(gs_vector, tol=1e-3)

print("\n=== Quench dynamics (J: 2.0 → 0.5) ===")
g_evolve = 6.0
J_evolve = 0.5

qlm_evolve = QuantumLinkModel(lattice, g_squared=g_evolve, J=J_evolve, gauss_penalty=penalty)
H_evolve = qlm_evolve.build_hamiltonian()

initial_energy = np.conj(gs_vector) @ H_evolve.to_matrix(sparse=True) @ gs_vector
print(f"Quench parameters: g²={g_evolve}, J={J_evolve}")
print(f"Terms in H_evolve: {len(H_evolve)}")
print(f"Initial energy in quenched H: {initial_energy.real:.6f}")

sim = DynamicalSimulation(
    H_evolve, lattice,
    g_squared=g_evolve,
    J=J_evolve,
    total_time=20.0,
    n_steps=400,
    initial_state=gs_vector,
    use_structured_trotter=True
)

results = sim.run()

backend = AerSimulator(method='statevector')
results_aer = sim.run_on_backend(backend, shots=2046,
                                  time_points=[0, 50, 100, 200, 300, 400])

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

axes[0].plot(results['times'], results['energies'], 'b-', linewidth=2)
axes[0].set_ylabel('Energy ⟨H⟩', fontsize=12)
axes[0].set_title(f'Pure gauge quench dynamics ({lattice.width}×{lattice.height}, {lattice.n_qubits} qubits)\n'
                   f'J: {J_initial} → {J_evolve}', fontsize=13)
axes[0].grid(alpha=0.3)

axes[1].plot(results['times'], results['gauss_violations'], 'r-', linewidth=2)
axes[1].set_ylabel('Max Gauss violation', fontsize=12)
axes[1].set_yscale('log')
axes[1].grid(alpha=0.3)

axes[2].plot(results['times'], results['plaquette_values'], 'g-', linewidth=2)
axes[2].set_ylabel('⟨Plaquette⟩', fontsize=12)
axes[2].set_xlabel('Time t', fontsize=12)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('pure_gauge_observables.png', dpi=150, bbox_inches='tight')
print("\n[Saved: pure_gauge_observables.png]")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

e_sv = np.array(results['e_field_profiles'])
im1 = axes[0].imshow(e_sv.T, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5,
                      extent=[0, results['times'][-1], -0.5, lattice.n_qubits - 0.5],
                      origin='lower')
axes[0].set_title('Electric field (statevector)', fontsize=12)
axes[0].set_ylabel('Link index', fontsize=11)
axes[0].set_xlabel('Time t', fontsize=11)
plt.colorbar(im1, ax=axes[0], label='⟨E⟩')

aer_times = sorted(results_aer.keys())
e_aer = np.array([[results_aer[t]['e_field'][q] for q in range(lattice.n_qubits)]
                   for t in aer_times])
im2 = axes[1].imshow(e_aer.T, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5,
                      extent=[aer_times[0], aer_times[-1], -0.5, lattice.n_qubits - 0.5],
                      origin='lower')
axes[1].set_title(f'Electric field (Aer, {4096} shots)', fontsize=12)
axes[1].set_ylabel('Link index', fontsize=11)
axes[1].set_xlabel('Time t', fontsize=11)
plt.colorbar(im2, ax=axes[1], label='⟨E⟩')

plt.tight_layout()
plt.savefig('pure_gauge_efield.png', dpi=150, bbox_inches='tight')
print("[Saved: pure_gauge_efield.png]")
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
depths = [results_aer[t]['depth'] for t in aer_times]
two_q = [results_aer[t]['two_qubit_gates'] for t in aer_times]

ax.plot(aer_times, depths, 'o-', color='coral', linewidth=2, markersize=8, label='Circuit depth')
ax2 = ax.twinx()
ax2.plot(aer_times, two_q, 's-', color='steelblue', linewidth=2, markersize=8, label='2-qubit gates')
ax.set_xlabel('Simulation time t', fontsize=12)
ax.set_ylabel('Circuit depth', color='coral', fontsize=12)
ax2.set_ylabel('Two-qubit gates', color='steelblue', fontsize=12)
ax.set_title('Circuit resource scaling (structured Trotter)', fontsize=13)
ax.tick_params(axis='y', labelcolor='coral')
ax2.tick_params(axis='y', labelcolor='steelblue')
ax.grid(alpha=0.3)
ax.legend(loc='upper left', fontsize=10)
ax2.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig('pure_gauge_scaling.png', dpi=150, bbox_inches='tight')
print("[Saved: pure_gauge_scaling.png]")
plt.show()

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Lattice size:          {lattice.width}×{lattice.height} ({lattice.n_qubits} gauge qubits)")
print(f"Initial H:             g²={g_initial}, J={J_initial}")
print(f"Quench H:              g²={g_evolve}, J={J_evolve}")
print(f"Ground state energy:   {eigenvalues[0]:.6f}")
print(f"Gap:                   {eigenvalues[1] - eigenvalues[0]:.6f}")
print(f"\nCircuit comparison:")
print(f"  Structured Trotter:  depth={step_s.depth()}, CNOTs={step_s.count_ops().get('cx', 0)}")
print(f"  Generic Trotter:     depth={step_g.depth()}, CNOTs={step_g.count_ops().get('cx', 0)}")
print(f"  Improvement:         {step_g.depth()/step_s.depth():.1f}× shallower")
print(f"\nDynamics:")
print(f"  Energy drift:        ±{(max(results['energies']) - min(results['energies']))/2:.2e}")
print(f"  Max Gauss violation: {max(results['gauss_violations']):.2e}")
print(f"  Aer time points:     {len(results_aer)}")
print(f"  Max circuit depth:   {max(depths)}")
print(f"  Max 2-qubit gates:   {max(two_q)}")
print("="*60)