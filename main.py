from lattice import LatticeGrid
from QuantumFields.gauge_field import GaussLaw, QuantumLinkModel
from circuit import DynamicalSimulation
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

lattice = LatticeGrid(width=3, height=3)
gauss = GaussLaw(lattice)

lattice.debug()
lattice.visualize_ascii()

# ============================================================
# Step 1: Find gauge-invariant ground state at STRONG coupling
# ============================================================
print("\n=== Finding gauge-invariant ground state (g²=1.0) ===")
qlm_strong = QuantumLinkModel(lattice, coupling=0.01, gauss_penalty=20.0)
H_strong = qlm_strong.build_hamiltonian()

eigenvalues, eigenvectors = eigh(H_strong.to_matrix())
gs_vector = eigenvectors[:, 0]

print(f"Ground state energy: {eigenvalues[0]:.6f}")
print(f"First excited:       {eigenvalues[1]:.6f}")
print(f"Gap:                 {eigenvalues[1] - eigenvalues[0]:.6f}")
gauss.verify(gs_vector, tol=1e-4)

# ============================================================
# Step 2: Build evolution Hamiltonian at WEAK coupling (quench)
#          NO penalty — gauge invariance is preserved by [G,H]=0
# ============================================================
print("\n=== Building quench Hamiltonian (g²=0.1, no penalty) ===")
evolution_coupling = 0.1
qlm_evolve = QuantumLinkModel(lattice, coupling=evolution_coupling, gauss_penalty=0.0)
H_evolve = qlm_evolve.build_hamiltonian()

print(f"Number of terms: {len(H_evolve)}")
print(f"Evolution coupling: g²={evolution_coupling}")

# Check: what energy does the strong-coupling ground state have
# under the weak-coupling Hamiltonian?
from qiskit.quantum_info import Statevector
sv_gs = Statevector(gs_vector)
E_quench = sv_gs.expectation_value(H_evolve).real
print(f"⟨ψ_strong|H_weak|ψ_strong⟩ = {E_quench:.6f}")

# ============================================================
# Step 3: Run the quench dynamics
#          Pass gs_vector as initial_state
# ============================================================
print("\n=== Running quench dynamics ===")
sim = DynamicalSimulation(
    H_evolve, lattice,
    coupling=evolution_coupling,
    gauss_penalty=0.0,
    total_time=20.0,
    n_steps=400,
    initial_state=gs_vector  # <-- THIS IS THE KEY LINE
)
results = sim.run()

# ============================================================
# Plots
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axes[0].plot(results['times'], results['energies'], 'b-')
axes[0].set_ylabel('⟨H⟩')
axes[0].set_title('Quantum quench: g²=1.0 → g²=0.1 (2×3 lattice)')

axes[1].plot(results['times'], results['gauss_violations'], 'r-')
axes[1].set_ylabel('Max Gauss violation')
axes[1].set_yscale('log')

axes[2].plot(results['times'], results['plaquette_values'], 'g-')
axes[2].set_ylabel('⟨H_B⟩')
axes[2].set_xlabel('Time t')

plt.tight_layout()
plt.show()

e_data = np.array(results['e_field_profiles'])

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(e_data.T, aspect='auto', cmap='RdBu_r',
               extent=[0, results['times'][-1], -0.5, lattice.n_qubits - 0.5],
               origin='lower')
ax.set_xlabel('Time t')
ax.set_ylabel('Link index')
ax.set_title('Electric field profile ⟨E_ℓ(t)⟩')
plt.colorbar(im, label='⟨Z/2⟩')
plt.tight_layout()
plt.show()