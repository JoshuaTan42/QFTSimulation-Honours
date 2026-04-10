from lattice import LatticeGrid
from QuantumFields.gauge_field import GaussLaw, QuantumLinkModel
from circuit import DynamicalSimulation
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_aer import AerSimulator

QiskitRuntimeService.save_account(
    token="", overwrite=True
)

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

H_sparse = H_strong.to_matrix(sparse=True)  # scipy sparse matrix, not dense
eigenvalues, eigenvectors = eigsh(H_sparse, k=2, which='SA')
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
    total_time=40.0,
    n_steps=800,
    initial_state=gs_vector  # <-- THIS IS THE KEY LINE
)
#results = sim.run()


backend = AerSimulator(method='statevector')
hw_results = sim.run_on_backend(backend, shots=4096,
                             time_points=[0, 50, 100, 200, 400])

"""service = QiskitRuntimeService()
backend = service.least_busy(operational=True)
print(f"Running on: {backend.name}")

hw_results = sim.run_on_backend(backend, shots=4096,
                             time_points=[0, 50, 100, 200, 400])"""

# ============================================================
# Plots
# ============================================================

# Plot hardware results
times = sorted(hw_results.keys())
e_fields = [hw_results[t]['e_field'] for t in times]
depths = [hw_results[t]['depth'] for t in times]

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Electric field heatmap
e_data = np.array(e_fields)
im = axes[0].imshow(e_data.T, aspect='auto', cmap='RdBu_r',
                     extent=[times[0], times[-1], -0.5, lattice.n_qubits - 0.5],
                     origin='lower')
axes[0].set_ylabel('Link index')
axes[0].set_title('Electric field (Aer, 4096 shots)')
plt.colorbar(im, ax=axes[0], label='⟨Z/2⟩')

# Circuit depth at each time point
axes[1].plot(times, depths, 'o-', color='coral')
axes[1].set_xlabel('Time t')
axes[1].set_ylabel('Circuit depth')
axes[1].set_title('Circuit depth vs simulation time')

plt.tight_layout()
plt.show()