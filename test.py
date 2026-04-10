from lattice import LatticeGrid
from QuantumFields.gauge_field import GaussLaw, QuantumLinkModel
from circuit import DynamicalSimulation, AdiabaticStatePrep
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Statevector
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
# Step 1: Quantum state preparation (fully circuit-based)
# ============================================================
print("\n=== Adiabatic state preparation (g²=10 → g²=1.0) ===")
adiabatic = AdiabaticStatePrep(lattice, target_coupling=1.0,
                                n_ramp_steps=5000, dt=0.01)
adiabatic_qc = adiabatic.build_circuit()
sv_adiabatic = Statevector.from_instruction(adiabatic_qc)

print(f"Circuit: depth={adiabatic_qc.depth()}, "
      f"CNOTs={adiabatic_qc.count_ops().get('cx', 0)}")

adiabatic_vector = np.array(sv_adiabatic)

# Measure energy from the quantum state
qlm_measure = QuantumLinkModel(lattice, coupling=1.0, gauss_penalty=0.0)
H_measure = qlm_measure.build_hamiltonian()
quantum_energy = sv_adiabatic.expectation_value(H_measure).real
print(f"Prepared state energy: {quantum_energy:.6f}")

# Check gauge invariance
gauss.verify(adiabatic_vector, tol=1e-2)

# ============================================================
# Step 2: Convergence scan — how many ramp steps are enough?
# ============================================================
print("\n=== Adiabatic convergence scan ===")
ramp_steps_list = [20, 50, 100, 200, 500]
energies_scan = []

for n_ramp in ramp_steps_list:
    adb = AdiabaticStatePrep(lattice, target_coupling=1.0,
                              n_ramp_steps=n_ramp, dt=0.05)
    qc = adb.build_circuit()
    qc = transpile(qc,
        basis_gates=['cx','rz','ry','rx','h','x','y','z','s','sdg'],
        optimization_level=1)
    sv = Statevector.from_instruction(qc)
    eng = sv.expectation_value(H_measure).real
    energies_scan.append(eng)
    print(f"  n_ramp={n_ramp:4d}  energy={eng:.6f}")

# ============================================================
# Step 3: Quantum quench dynamics on Aer
# ============================================================
print("\n=== Quantum quench: g²=1.0 → g²=0.1 ===")
evolution_coupling = 0.1
qlm_evolve = QuantumLinkModel(lattice, coupling=evolution_coupling, gauss_penalty=0.0)
H_evolve = qlm_evolve.build_hamiltonian()

sim = DynamicalSimulation(
    H_evolve, lattice,
    coupling=evolution_coupling,
    gauss_penalty=0.0,
    total_time=20.0,
    n_steps=400,
    initial_state=adiabatic_vector
)

# Statevector run (full observables)
print("\n--- Statevector evolution ---")
results_sv = sim.run(verbose=True)

# Aer with shots (hardware-realistic)
print("\n--- Aer with shots ---")
backend = AerSimulator(method='statevector')
results_aer = sim.run_on_backend(backend, shots=4096,
                                  time_points=[0, 50, 100, 200, 300, 400])

# ============================================================
# Plots
# ============================================================

# Plot 1: Convergence scan
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(ramp_steps_list, energies_scan, 'o-', color='teal', markersize=8)
ax.set_xlabel('Ramp steps')
ax.set_ylabel('Prepared state energy')
ax.set_title('Adiabatic state preparation convergence')
ax.axhline(y=energies_scan[-1], color='gray', linestyle='--', alpha=0.5,
           label=f'Best energy: {energies_scan[-1]:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 2: Full observables from statevector
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

axes[0].plot(results_sv['times'], results_sv['energies'], 'b-')
axes[0].set_ylabel('⟨H⟩')
axes[0].set_title('Quantum quench: g²=1.0 → g²=0.1 (adiabatic prep, 2×3 lattice)')

axes[1].plot(results_sv['times'], results_sv['gauss_violations'], 'r-')
axes[1].set_ylabel('Max Gauss violation')
axes[1].set_yscale('log')

axes[2].plot(results_sv['times'], results_sv['plaquette_values'], 'g-')
axes[2].set_ylabel('⟨H_B⟩')
axes[2].set_xlabel('Time t')

plt.tight_layout()
plt.show()

# Plot 3: Electric field — statevector vs Aer shots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

e_sv = np.array(results_sv['e_field_profiles'])
im1 = axes[0].imshow(e_sv.T, aspect='auto', cmap='RdBu_r',
                      extent=[0, 20, -0.5, lattice.n_qubits-0.5], origin='lower')
axes[0].set_title('Statevector (exact)')
axes[0].set_ylabel('Link index')
axes[0].set_xlabel('Time t')
plt.colorbar(im1, ax=axes[0], label='⟨Z/2⟩')

aer_times = sorted(results_aer.keys())
e_aer = np.array([results_aer[t]['e_field'] for t in aer_times])
im2 = axes[1].imshow(e_aer.T, aspect='auto', cmap='RdBu_r',
                      extent=[aer_times[0], aer_times[-1], -0.5, lattice.n_qubits-0.5],
                      origin='lower')
axes[1].set_title('Aer simulator (4096 shots)')
axes[1].set_ylabel('Link index')
axes[1].set_xlabel('Time t')
plt.colorbar(im2, ax=axes[1], label='⟨Z/2⟩')

plt.tight_layout()
plt.show()

# Plot 4: Circuit depth scaling
aer_times_sorted = sorted(results_aer.keys())
depths = [results_aer[t]['depth'] for t in aer_times_sorted]
two_q = [results_aer[t].get('two_qubit_gates', results_aer[t].get('cnots', 0))
         for t in aer_times_sorted]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(aer_times_sorted, depths, 'o-', color='coral', label='Circuit depth')
ax.set_xlabel('Simulation time t')
ax.set_ylabel('Circuit depth')
ax.set_title('Circuit resource scaling')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# Summary
print("\n=== Summary ===")
print(f"Prepared state energy:    {quantum_energy:.6f}")
print(f"Gauss violation:          {max(results_sv['gauss_violations']):.2e}")
print(f"Energy conservation:      ±{(max(results_sv['energies']) - min(results_sv['energies']))/2:.6f}")
print(f"Aer time points:          {len(results_aer)}")
print(f"Max circuit depth:        {max(depths)}")