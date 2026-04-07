from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from QuantumFields.gauge_field import QuantumLinkModel, GaussLaw
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector



class TrotterEvolution:
    def __init__(self, hamiltonian, n_qubits:int, total_time:float, n_steps:int):
        self.hamiltonian = hamiltonian
        self.n_qubits = n_qubits
        self.total_time = total_time
        self.n_steps = n_steps

    def build_circuit(self) -> QuantumCircuit:
        evolution_gate = PauliEvolutionGate(
            self.hamiltonian,
            time=self.total_time,
            synthesis=SuzukiTrotter(order=2, reps=self.n_steps))

        qc = QuantumCircuit(self.n_qubits)
        qc.append(evolution_gate, range(self.n_qubits))
        return qc
    

class DynamicalSimulation:
    def __init__(self, hamiltonian, lattice, coupling, gauss_penalty,
                 total_time: float, n_steps: int, initial_state=None):
        self.hamiltonian = hamiltonian
        self.lattice = lattice
        self.n_qubits = lattice.n_qubits
        self.total_time = total_time
        self.n_steps = n_steps
        self.dt = total_time / n_steps
        self.initial_state = initial_state

        self._gauss = GaussLaw(self.lattice)
        self._gauss_ops = self._gauss.build_gauss_operators()
        self._H_B = QuantumLinkModel(lattice, coupling=coupling, gauss_penalty=gauss_penalty).build_magnetic_term()

    def run(self, verbose=True):
        # Build one Trotter step: exp(-iH dt)
        single_step = TrotterEvolution(
            self.hamiltonian, self.n_qubits,
            total_time=self.dt, n_steps=1
        )
        step_circuit = single_step.build_circuit()
        step_circuit = transpile(step_circuit, basis_gates=['cx', 'rz', 'ry', 'rx', 'h', 'x', 'y', 'z', 's', 'sdg'], optimization_level=3)
        print(f"Depth: {step_circuit.depth()}, CNOTs: {step_circuit.count_ops().get('cx', 0)}")

        # Initial state: strong coupling vacuum |+⟩^n
        gauss = GaussLaw(self.lattice)
        init_qc = gauss.get_initial_circuit(state_vector=self.initial_state)
        sv = Statevector.from_instruction(init_qc)

        # Storage
        times = [0.0]
        energies = [sv.expectation_value(self.hamiltonian).real]
        gauss_violations = [self._max_gauss_violation(sv)]
        plaquette_values = [self._plaquette_expectation(sv)]
        e_profiles = [self._electric_field_profile(sv)]

        if verbose:
            print(f"{'t':>6s}  {'⟨H⟩':>12s}  {'Gauss viol':>12s}  {'⟨plaq⟩':>10s}")
            print(f"{0.0:6.3f}  {energies[0]:12.6f}  {gauss_violations[0]:12.2e}  {plaquette_values[0]:10.6f}")

        # Evolve step by step
        for step in range(1, self.n_steps + 1):
            sv = sv.evolve(step_circuit)
            t = step * self.dt

            energy = sv.expectation_value(self.hamiltonian).real
            gv = self._max_gauss_violation(sv)
            plaq = self._plaquette_expectation(sv)

            times.append(t)
            energies.append(energy)
            gauss_violations.append(gv)
            plaquette_values.append(plaq)
            e_profiles.append(self._electric_field_profile(sv))

            if verbose and (step % max(1, self.n_steps // 10) == 0 or step == self.n_steps):
                print(f"{t:6.3f}  {energy:12.6f}  {gv:12.2e}  {plaq:10.6f}")

        return {
            'times': np.array(times),
            'energies': np.array(energies),
            'gauss_violations': np.array(gauss_violations),
            'plaquette_values': np.array(plaquette_values),
            'e_field_profiles': np.array(e_profiles)
        }

    def _max_gauss_violation(self, sv: Statevector) -> float:
        sv_arr = np.array(sv)
        max_viol = 0.0
        for G_op in self._gauss_ops:
            if G_op is None:
                continue
            viol = np.linalg.norm(G_op.to_matrix() @ sv_arr)
            max_viol = max(max_viol, viol)
        return max_viol

    def _electric_field_profile(self, sv: Statevector) -> list:
        """⟨E_ℓ⟩ = ⟨Z_ℓ⟩/2 on each link."""
        profile = []
        for q in range(self.lattice.n_qubits):
            label = ['I'] * self.lattice.n_qubits
            label[q] = 'Z'
            op = SparsePauliOp([''.join(label)], coeffs=[0.5])
            profile.append(sv.expectation_value(op).real)
        return profile

    def _plaquette_expectation(self, sv: Statevector) -> float:
        return sv.expectation_value(self._H_B).real
    
