"""Time evolution for the pure-gauge U(1) quantum link model.

Provides a generic second-order Suzuki-Trotter circuit (TrotterEvolution) as a
baseline, the structured constant-depth Trotter step (StructuredTrotter,
following Joshi et al. 2026), and an exact statevector simulator used to track
observables and benchmark circuits (DynamicalSimulation).
"""

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate, StatePreparation
from qiskit.synthesis import SuzukiTrotter
from QuantumFields.gauge_field import QuantumLinkModel, GaussLaw
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.sparse.linalg import expm_multiply


class TrotterEvolution:
    """Generic second-order Suzuki-Trotter evolution of an arbitrary Hamiltonian."""

    def __init__(self, hamiltonian, n_qubits: int, total_time: float, n_steps: int):
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


class StructuredTrotter:
    """One constant-depth Trotter step: half-electric, two plaquette sublayers,
    half-electric (Joshi et al. 2026). The checkerboard sublayer split keeps the
    two-qubit gate depth independent of lattice size.
    """

    def __init__(self, lattice, g_squared=1.0, J=1.0, dt=0.05):
        self.lattice = lattice
        self.n_qubits = lattice.n_qubits
        self.g_squared = g_squared
        self.J = J
        self.dt = dt

        self.plaq_sublayers = lattice.get_plaquette_sublayers()
        self.electric_qubits = lattice.get_electric_qubits()

    def build_step(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)

        self._apply_electric_layer(qc, fraction=0.5)
        qc.barrier()

        self._apply_plaquette_sublayer(qc, self.plaq_sublayers[0])
        qc.barrier()

        self._apply_plaquette_sublayer(qc, self.plaq_sublayers[1])
        qc.barrier()

        self._apply_electric_layer(qc, fraction=0.5)

        return qc

    def _apply_plaquette_sublayer(self, qc, plaquettes):
        if self.J == 0.0:
            return
        for qubits in plaquettes:
            for pauli_str, sign in QuantumLinkModel._PLAQ_PAULIS:
                self._exp_pauli_string(qc, sign * self.dt * self.J / 8.0,
                                       list(qubits), list(pauli_str))

    def _apply_electric_layer(self, qc, fraction=1.0):
        """Parallel single-qubit Rz rotations realising exp(-i dt H_E * fraction)."""
        angle = self.dt * self.g_squared * fraction / 4.0
        for link_q in self.electric_qubits:
            qc.rz(angle, link_q)

    def _exp_pauli_string(self, qc, angle, qubits, paulis):
        """exp(-i*angle*P) via the CNOT-ladder + Rz + reverse-ladder construction."""
        for q, p in zip(qubits, paulis):
            if p == 'X':
                qc.h(q)
            elif p == 'Y':
                qc.sdg(q)
                qc.h(q)

        for i in range(len(qubits) - 1):
            qc.cx(qubits[i], qubits[i + 1])

        qc.rz(2 * angle, qubits[-1])

        for i in range(len(qubits) - 2, -1, -1):
            qc.cx(qubits[i], qubits[i + 1])

        for q, p in zip(qubits, paulis):
            if p == 'X':
                qc.h(q)
            elif p == 'Y':
                qc.h(q)
                qc.s(q)


class DynamicalSimulation:
    """Exact statevector evolution under H, tracking energy, Gauss violation,
    plaquette expectation and the electric-field profile. run_on_backend rebuilds
    the same evolution from structured Trotter steps to estimate circuit resources.
    """

    def __init__(self, hamiltonian, lattice,
                 g_squared=1.0, J=1.0,
                 total_time=20.0, n_steps=400,
                 initial_state=None,
                 use_structured_trotter=True):

        self.hamiltonian = hamiltonian
        self.lattice = lattice
        self.n_qubits = lattice.n_qubits
        self.total_time = total_time
        self.n_steps = n_steps
        self.dt = total_time / n_steps
        self.initial_state = initial_state
        self.use_structured_trotter = use_structured_trotter

        self._g_squared = g_squared
        self._J = J

        self._H_sparse = self.hamiltonian.to_matrix(sparse=True)

        gauss = GaussLaw(self.lattice)
        self._gauss_sparse = []
        for G_op in gauss.build_gauss_operators():
            if G_op is None:
                self._gauss_sparse.append(None)
            else:
                self._gauss_sparse.append(G_op.to_matrix(sparse=True))

        H_B_op = QuantumLinkModel(
            lattice, g_squared=g_squared, J=J
        ).build_magnetic_term()
        self._H_B_sparse = H_B_op.to_matrix(sparse=True)

        self._efield_ops_sparse = []
        for link_q in self.lattice.get_electric_qubits():
            label = ['I'] * self.n_qubits
            label[self.n_qubits - 1 - link_q] = 'Z'
            op = SparsePauliOp([''.join(label)], coeffs=[0.5])
            self._efield_ops_sparse.append(op.to_matrix(sparse=True))

    def run(self, verbose=True):
        if self.initial_state is not None:
            sv = self.initial_state.copy().astype(complex)
        else:
            sv = np.zeros(2**self.n_qubits, dtype=complex)
            sv[0] = 1.0

        E0 = (np.conj(sv) @ self._H_sparse @ sv).real
        if verbose:
            print(f"Initial energy (verified): {E0:.6f}")

        minus_i_dt_H = -1j * self.dt * self._H_sparse

        times = [0.0]
        energies = [E0]
        gauss_violations = [self._max_gauss_violation(sv)]
        plaquette_values = [self._plaquette_expectation(sv)]
        e_profiles = [self._electric_field_profile(sv)]

        if verbose:
            print(f"{'t':>6s}  {'⟨H⟩':>12s}  {'Gauss viol':>12s}  {'⟨plaq⟩':>10s}")
            print(f"{0.0:6.3f}  {energies[0]:12.6f}  "
                  f"{gauss_violations[0]:12.2e}  {plaquette_values[0]:10.6f}")

        for step in range(1, self.n_steps + 1):
            sv = expm_multiply(minus_i_dt_H, sv)
            t = step * self.dt

            energy = (np.conj(sv) @ self._H_sparse @ sv).real
            gv = self._max_gauss_violation(sv)
            plaq = self._plaquette_expectation(sv)

            times.append(t)
            energies.append(energy)
            gauss_violations.append(gv)
            plaquette_values.append(plaq)
            e_profiles.append(self._electric_field_profile(sv))

            if verbose and (step % max(1, self.n_steps // 10) == 0
                            or step == self.n_steps):
                print(f"{t:6.3f}  {energy:12.6f}  {gv:12.2e}  {plaq:10.6f}")

        return {
            'times': np.array(times),
            'energies': np.array(energies),
            'gauss_violations': np.array(gauss_violations),
            'plaquette_values': np.array(plaquette_values),
            'e_field_profiles': np.array(e_profiles)
        }

    def run_on_backend(self, backend, shots=4096, time_points=None):
        from qiskit_ibm_runtime import EstimatorV2 as Estimator

        estimator = Estimator(mode=backend)
        estimator.options.resilience_level = 1
        estimator.options.default_shots = shots

        observables = []
        for q in range(self.n_qubits):
            label = ['I'] * self.n_qubits
            label[q] = 'Z'
            observables.append(SparsePauliOp([''.join(label)], coeffs=[0.5]))

        if time_points is None:
            time_points = [0, 50, 100, 200, self.n_steps]

        if self.use_structured_trotter:
            structured = StructuredTrotter(
                self.lattice,
                g_squared=self._g_squared, J=self._J,
                dt=self.dt
            )
            step_circuit = structured.build_step()
            step_circuit = transpile(
                step_circuit,
                basis_gates=['cx', 'rz', 'ry', 'rx', 'h', 'x', 'y', 'z', 's', 'sdg'],
                optimization_level=3
            )
        else:
            single_step = TrotterEvolution(
                self.hamiltonian, self.n_qubits,
                total_time=self.dt, n_steps=1
            )
            step_circuit = single_step.build_circuit()
            step_circuit = step_circuit.decompose(reps=5)
            step_circuit = transpile(
                step_circuit,
                basis_gates=['cx', 'rz', 'ry', 'rx', 'h', 'x', 'y', 'z', 's', 'sdg'],
                optimization_level=3
            )

        results = {}

        for n_step in time_points:
            qc = QuantumCircuit(self.n_qubits)

            if self.initial_state is not None:
                qc.append(StatePreparation(self.initial_state),
                          range(self.n_qubits))
            else:
                gauss = GaussLaw(self.lattice)
                init_qc = gauss.get_initial_circuit()
                qc.compose(init_qc, qubits=range(self.n_qubits), inplace=True)

            for _ in range(n_step):
                qc.compose(step_circuit, qubits=range(self.n_qubits),
                           inplace=True)

            isa_circuit = transpile(qc, backend, optimization_level=2)

            mapped_obs = [
                obs.apply_layout(isa_circuit.layout) for obs in observables
            ]

            job = estimator.run([(isa_circuit, mapped_obs)])
            result = job.result()

            e_field = [val.item() for val in result[0].data.evs]

            t = n_step * self.dt
            two_qubit_gates = (isa_circuit.count_ops().get('cx', 0)
                               + isa_circuit.count_ops().get('ecr', 0))

            results[t] = {
                'e_field': e_field,
                'depth': isa_circuit.depth(),
                'two_qubit_gates': two_qubit_gates,
            }

            print(f"t={t:.2f}  depth={results[t]['depth']}  "
                  f"2q_gates={two_qubit_gates}  "
                  f"E_field={[f'{v:.4f}' for v in e_field[:3]]}...")

        return results

    def _max_gauss_violation(self, sv: np.ndarray) -> float:
        max_viol = 0.0
        for G_sparse in self._gauss_sparse:
            if G_sparse is None:
                continue
            viol = np.linalg.norm(G_sparse @ sv)
            max_viol = max(max_viol, viol)
        return max_viol

    def _electric_field_profile(self, sv: np.ndarray) -> list:
        return [(np.conj(sv) @ op @ sv).real for op in self._efield_ops_sparse]

    def _plaquette_expectation(self, sv: np.ndarray) -> float:
        return (np.conj(sv) @ self._H_B_sparse @ sv).real