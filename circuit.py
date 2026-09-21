"""Time evolution for the pure-gauge U(1) quantum link model (thesis Sec. 2.3.3).

  StructuredTrotter    half-electric layer, one sublayer per plaquette colour,
                       half-electric layer. Plaquettes inside a sublayer share
                       no link, so they run in parallel and the two-qubit depth
                       of a step does not grow with W or H.
  TrotterEvolution     generic Suzuki-Trotter baseline on the flat Pauli list.
  DynamicalSimulation  exact statevector evolution with observables, plus an
                       Estimator-based hardware runner.

Joshi et al. (2026) quote a two-qubit depth of 28 per step. That figure belongs
to their model (matter hopping, open boundaries, native ZZ gates): four
sublayers of depth <= 7 set by the hopping circuit. It does not transfer to the
pure-gauge cylinder, where the depth is set by the plaquette circuit and, for
odd W, by three sublayers instead of two. The numbers reported by this module
are measured on the circuits it actually builds.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import (PauliEvolutionGate, StatePreparation, UCRXGate,
                                    UCRZGate)
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from qiskit.synthesis import SuzukiTrotter
from scipy.sparse.linalg import expm_multiply

from QuantumFields.gauge_field import (PLAQ_PAULIS, RK_PAULIS, GaussLaw,
                                       QuantumLinkModel, field_signs, pauli_label)

BASIS_GATES = ['cx', 'rz', 'sx', 'x']


def append_pauli_rotation(qc: QuantumCircuit, angle: float, qubits, paulis):
    """Append exp(-i * angle * P), P = tensor product of `paulis` on `qubits`.

    Basis change to Z, CNOT ladder onto the last qubit, Rz(2*angle), undo.
    Identity factors are skipped.
    """
    active = [(q, p) for q, p in zip(qubits, paulis) if p != 'I']
    if not active:
        qc.global_phase -= angle
        return
    for q, p in active:
        if p == 'X':
            qc.h(q)
        elif p == 'Y':
            qc.sdg(q)
            qc.h(q)
    qs = [q for q, _ in active]
    for a, b in zip(qs[:-1], qs[1:]):
        qc.cx(a, b)
    qc.rz(2.0 * angle, qs[-1])
    for a, b in reversed(list(zip(qs[:-1], qs[1:]))):
        qc.cx(a, b)
    for q, p in active:
        if p == 'X':
            qc.h(q)
        elif p == 'Y':
            qc.h(q)
            qc.s(q)


PLAQUETTE_SYNTHESIS = "ucr"      # "ucr" (compact, default) or "pauli" (eight ladders)


def _fan(qc, plaquette, reverse=False):
    """CNOT fan-out from the bottom link. It maps the two flippable
    configurations (b,r,t,l) = (1,1,0,0), (0,0,1,1) onto (b, 0, 1, 1), b = 1, 0,
    so U_p + U_p^dag becomes X_b controlled on (r, t, l) = (0, 1, 1)."""
    b, r, t, l = plaquette
    for x in ((l, t, r) if reverse else (r, t, l)):
        qc.cx(b, x)


_FLIPPABLE_INDEX = 0 + 2 + 4         # control pattern r=0, t=1, l=1 (first control = LSB)


def append_plaquette_rotation(qc: QuantumCircuit, theta: float, plaquette,
                              ancilla: int = None, method: str = None):
    """Append exp(-i theta (U_p + U_p^dag)), exactly.

    method="ucr"    fan-out, one uniformly controlled rotation, fan-in:
                    13 CNOTs, or 16 with the RQSVT ancilla attached.
    method="pauli"  the eight commuting Pauli-string rotations of the draft
                    (CNOT ladders): 17 CNOTs after transpilation, 40 with ancilla.
    Both are exact and gauge invariant (a single Pauli string is not). With
    `ancilla` the generator is Z_anc (x) (U_p + U_p^dag), the RQSVT signal
    operator; no controlled time evolution is needed.
    """
    method = method or PLAQUETTE_SYNTHESIS
    if method == "pauli":
        for label, sign in PLAQ_PAULIS:
            qubits, paulis = list(plaquette), list(label)
            if ancilla is not None:
                qubits, paulis = qubits + [ancilla], paulis + ['Z']
            append_pauli_rotation(qc, sign * theta / 8.0, qubits, paulis)
        return
    if method != "ucr":
        raise ValueError(f"unknown plaquette synthesis {method!r}")
    b, r, t, l = plaquette
    angles = [0.0] * 8
    angles[_FLIPPABLE_INDEX] = 2.0 * theta
    _fan(qc, plaquette)
    if ancilla is None:
        qc.append(UCRXGate(angles), [b, r, t, l])
    else:                                   # Z_anc X_b = H_b CX(anc,b) Z_b CX(anc,b) H_b
        qc.h(b)
        qc.cx(ancilla, b)
        qc.append(UCRZGate(angles), [b, r, t, l])
        qc.cx(ancilla, b)
        qc.h(b)
    _fan(qc, plaquette, reverse=True)


def append_flippability_rotation(qc: QuantumCircuit, theta: float, plaquette,
                                 ancilla: int = None, method: str = None):
    """Append exp(-i theta (U_p + U_p^dag)^2) (diagonal RK term)."""
    method = method or PLAQUETTE_SYNTHESIS
    if method == "pauli":
        for label, sign in RK_PAULIS:
            qubits, paulis = list(plaquette), list(label)
            if ancilla is not None:
                qubits, paulis = qubits + [ancilla], paulis + ['Z']
            append_pauli_rotation(qc, sign * theta / 8.0, qubits, paulis)
        return
    b, r, t, l = plaquette
    angles = [0.0] * 8
    angles[_FLIPPABLE_INDEX] = 2.0 * theta
    _fan(qc, plaquette)
    if ancilla is None:                     # phase exp(-i theta) on (r,t,l) = (0,1,1)
        qc.x(r)
        qc.mcp(-theta, [r, t], l)
        qc.x(r)
    else:                                   # exp(-i theta Z_anc) on that pattern
        qc.append(UCRZGate(angles), [ancilla, r, t, l])
    _fan(qc, plaquette, reverse=True)


def two_qubit_depth(qc: QuantumCircuit) -> int:
    return qc.depth(lambda inst: inst.operation.num_qubits == 2)


def two_qubit_count(qc: QuantumCircuit) -> int:
    return sum(1 for inst in qc.data if inst.operation.num_qubits == 2)


def circuit_resources(qc: QuantumCircuit, basis_gates=None,
                      optimization_level: int = 3) -> dict:
    """Transpile and report the counts used in the resource analysis (Sec. 2.3.6)."""
    tqc = transpile(qc, basis_gates=basis_gates or BASIS_GATES,
                    optimization_level=optimization_level)
    n2 = two_qubit_count(tqc)
    return {'qubits': tqc.num_qubits,
            'two_qubit_gates': n2,
            'one_qubit_gates': tqc.size() - n2,
            'two_qubit_depth': two_qubit_depth(tqc),
            'depth': tqc.depth()}


class TrotterEvolution:
    """Generic Suzuki-Trotter evolution of an arbitrary SparsePauliOp (baseline)."""

    def __init__(self, hamiltonian, n_qubits: int, total_time: float,
                 n_steps: int, order: int = 2):
        self.hamiltonian = hamiltonian
        self.n_qubits = n_qubits
        self.total_time = total_time
        self.n_steps = n_steps
        self.order = order

    def build_circuit(self) -> QuantumCircuit:
        gate = PauliEvolutionGate(
            self.hamiltonian, time=self.total_time,
            synthesis=SuzukiTrotter(order=self.order, reps=self.n_steps))
        qc = QuantumCircuit(self.n_qubits)
        qc.append(gate, range(self.n_qubits))
        return qc


class StructuredTrotter:
    """One structured Trotter step of duration dt.

        exp(-i dt H_diag / 2) . prod_c exp(-i dt H_B^(c)) . exp(-i dt H_diag / 2)

    H_diag = H_E + H_RK is diagonal; H_B^(c) collects the plaquettes of colour c.
    order=1 applies the colours once (error O(dt^2) per step, O(dt) overall);
    order=2 applies them palindromically with half steps (O(dt^2) overall).
    For the uniform field [H_E, H_B] = 0, so the only Trotter error comes from
    plaquette sublayers that share links across colours.
    """

    def __init__(self, lattice, g=0.0, J=1.0, dt=0.05, rk=0.0,
                 field_pattern="uniform", order: int = 1, colouring="auto",
                 synthesis: str = None):
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        self.lattice = lattice
        self.n_qubits = lattice.n_qubits
        self.g, self.J, self.rk, self.dt = g, J, rk, dt
        self.order = order
        self.synthesis = synthesis          # None -> PLAQUETTE_SYNTHESIS
        self.signs = field_signs(lattice, field_pattern)
        self.plaq_sublayers = lattice.get_plaquette_sublayers(colouring)
        self.electric_qubits = lattice.get_electric_qubits()

    def build_step(self, barriers: bool = True) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        self._apply_diagonal_layer(qc, 0.5)
        if barriers:
            qc.barrier()
        if self.order == 1:
            for layer in self.plaq_sublayers:
                self._apply_plaquette_sublayer(qc, layer, 1.0)
                if barriers:
                    qc.barrier()
        else:
            for layer in self.plaq_sublayers[:-1]:
                self._apply_plaquette_sublayer(qc, layer, 0.5)
            self._apply_plaquette_sublayer(qc, self.plaq_sublayers[-1], 1.0)
            for layer in reversed(self.plaq_sublayers[:-1]):
                self._apply_plaquette_sublayer(qc, layer, 0.5)
            if barriers:
                qc.barrier()
        self._apply_diagonal_layer(qc, 0.5)
        return qc

    def build_evolution(self, n_steps: int) -> QuantumCircuit:
        step = self.build_step(barriers=False)
        qc = QuantumCircuit(self.n_qubits)
        for _ in range(n_steps):
            qc.compose(step, inplace=True)
        return qc

    def _apply_plaquette_sublayer(self, qc, plaquettes, fraction=1.0):
        if self.J == 0.0:
            return
        for plaquette in plaquettes:
            append_plaquette_rotation(qc, self.J * self.dt * fraction, plaquette,
                                      method=self.synthesis)

    def _apply_diagonal_layer(self, qc, fraction=1.0):
        """exp(-i dt f H_E) exp(-i dt f H_RK).

        H_E = -(g/2) sum s_l Z_l and Rz(a) = exp(-i a Z / 2), so the rotation
        angle on link l is -g s_l dt f.
        """
        if self.g != 0.0:
            for q in self.electric_qubits:
                qc.rz(-self.g * self.signs[q] * self.dt * fraction, q)
        if self.rk != 0.0:
            for layer in self.plaq_sublayers:
                for plaquette in layer:
                    append_flippability_rotation(
                        qc, self.rk * self.dt * fraction, plaquette, method=self.synthesis)

    # backwards-compatible name
    def _apply_electric_layer(self, qc, fraction=1.0):
        self._apply_diagonal_layer(qc, fraction)


def trotter_error(lattice, g, J, dt, rk=0.0, field_pattern="uniform",
                  order=1, colouring="auto") -> float:
    """Spectral-norm distance between one structured step and exp(-i H dt)."""
    from scipy.linalg import expm
    model = QuantumLinkModel(lattice, g=g, J=J, rk=rk, field_pattern=field_pattern)
    H = model.build_physical_hamiltonian().to_matrix()
    step = StructuredTrotter(lattice, g=g, J=J, dt=dt, rk=rk,
                             field_pattern=field_pattern, order=order,
                             colouring=colouring).build_step(barriers=False)
    return float(np.linalg.norm(Operator(step).data - expm(-1j * dt * H), 2))


class DynamicalSimulation:
    """Exact statevector evolution under H with gauge-theory observables."""

    def __init__(self, hamiltonian, lattice, g=0.0, J=1.0, rk=0.0,
                 field_pattern="uniform", plates="free", plate_charges=None,
                 total_time=20.0, n_steps=400, initial_state=None,
                 use_structured_trotter=True):
        self.hamiltonian = hamiltonian
        self.lattice = lattice
        self.n_qubits = lattice.n_qubits
        self.total_time = total_time
        self.n_steps = n_steps
        self.dt = total_time / n_steps
        self.initial_state = initial_state
        self.use_structured_trotter = use_structured_trotter
        self._g, self._J, self._rk = g, J, rk
        self._field_pattern = field_pattern

        self._H_sparse = hamiltonian.to_matrix(sparse=True)
        gauss = GaussLaw(lattice, plates=plates, plate_charges=plate_charges)
        self._gauss_sparse = [G.to_matrix(sparse=True)
                              for G in gauss.build_gauss_operators() if G is not None]
        self._H_B_sparse = QuantumLinkModel(lattice, J=J).build_magnetic_term() \
            .to_matrix(sparse=True)
        self._efield_ops_sparse = [
            SparsePauliOp([pauli_label(self.n_qubits, [q], 'Z')], coeffs=[0.5])
            .to_matrix(sparse=True) for q in lattice.get_electric_qubits()]

    def run(self, verbose=True):
        if self.initial_state is not None:
            sv = np.asarray(self.initial_state, dtype=complex).copy()
        else:
            sv = np.zeros(2 ** self.n_qubits, dtype=complex)
            sv[0] = 1.0

        def measure(state):
            return ((np.conj(state) @ (self._H_sparse @ state)).real,
                    self._max_gauss_violation(state),
                    self._plaquette_expectation(state),
                    self._electric_field_profile(state))

        records = [measure(sv)]
        if verbose:
            print(f"{'t':>6s}  {'<H>':>12s}  {'Gauss viol':>12s}  {'<H_B>':>10s}")
            print(f"{0.0:6.3f}  {records[0][0]:12.6f}  {records[0][1]:12.2e}  "
                  f"{records[0][2]:10.6f}")

        generator = -1j * self.dt * self._H_sparse
        for step in range(1, self.n_steps + 1):
            sv = expm_multiply(generator, sv)
            records.append(measure(sv))
            if verbose and (step % max(1, self.n_steps // 10) == 0
                            or step == self.n_steps):
                e, gv, pl, _ = records[-1]
                print(f"{step * self.dt:6.3f}  {e:12.6f}  {gv:12.2e}  {pl:10.6f}")

        return {'times': self.dt * np.arange(self.n_steps + 1),
                'energies': np.array([r[0] for r in records]),
                'gauss_violations': np.array([r[1] for r in records]),
                'plaquette_values': np.array([r[2] for r in records]),
                'e_field_profiles': np.array([r[3] for r in records]),
                'final_state': sv}

    def run_trotter(self, n_steps=None) -> np.ndarray:
        """Statevector after n_steps structured Trotter steps (noiseless)."""
        n_steps = self.n_steps if n_steps is None else n_steps
        trotter = StructuredTrotter(self.lattice, g=self._g, J=self._J, rk=self._rk,
                                    dt=self.dt, field_pattern=self._field_pattern)
        if self.initial_state is not None:
            sv = Statevector(np.asarray(self.initial_state, dtype=complex))
        else:
            sv = Statevector.from_label('0' * self.n_qubits)
        return sv.evolve(trotter.build_evolution(n_steps)).data

    def run_on_backend(self, backend, shots=4096, time_points=None):
        """<S^z_l>(t) on hardware through the runtime Estimator.

        Not exercised by the test-suite (it needs IBM Quantum credentials).
        """
        from qiskit_ibm_runtime import EstimatorV2 as Estimator

        estimator = Estimator(mode=backend)
        estimator.options.resilience_level = 1
        estimator.options.default_shots = shots

        # little-endian labels: qubit q is character N_q - 1 - q of the label
        observables = [SparsePauliOp([pauli_label(self.n_qubits, [q], 'Z')],
                                     coeffs=[0.5]) for q in range(self.n_qubits)]
        if time_points is None:
            time_points = sorted({0, self.n_steps // 4, self.n_steps // 2,
                                  self.n_steps})

        if self.use_structured_trotter:
            step_circuit = StructuredTrotter(
                self.lattice, g=self._g, J=self._J, rk=self._rk, dt=self.dt,
                field_pattern=self._field_pattern).build_step(barriers=False)
        else:
            step_circuit = TrotterEvolution(self.hamiltonian, self.n_qubits,
                                            total_time=self.dt, n_steps=1
                                            ).build_circuit().decompose(reps=5)

        results = {}
        for n_step in time_points:
            qc = QuantumCircuit(self.n_qubits)
            if self.initial_state is not None:
                qc.append(StatePreparation(self.initial_state), range(self.n_qubits))
            for _ in range(n_step):
                qc.compose(step_circuit, inplace=True)
            isa = transpile(qc, backend, optimization_level=3)
            mapped = [obs.apply_layout(isa.layout) for obs in observables]
            result = estimator.run([(isa, mapped)]).result()
            t = n_step * self.dt
            results[t] = {'e_field': [float(v) for v in result[0].data.evs],
                          'depth': isa.depth(),
                          'two_qubit_gates': two_qubit_count(isa)}
            print(f"t={t:.2f}  depth={results[t]['depth']}  "
                  f"2q_gates={results[t]['two_qubit_gates']}")
        return results

    def _max_gauss_violation(self, sv) -> float:
        return max((float(np.linalg.norm(G @ sv)) for G in self._gauss_sparse),
                   default=0.0)

    def _electric_field_profile(self, sv) -> list:
        return [float((np.conj(sv) @ (op @ sv)).real)
                for op in self._efield_ops_sparse]

    def _plaquette_expectation(self, sv) -> float:
        return float((np.conj(sv) @ (self._H_B_sparse @ sv)).real)
