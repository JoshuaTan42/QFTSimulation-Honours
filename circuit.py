"""Quantum circuits for time evolution in U(1) quantum link models.

Gate constructions
  append_two_level_rotation   exp(-i theta (|a><b| + |b><a|)) for two bit strings
                              a, b on a set of qubits: CNOT fan-out from a pivot
                              qubit, one uniformly controlled rotation, fan-in.
                              Plaquette flips (U_p + U_p^dag) and gauge-matter
                              hopping terms are both of this form, so both are
                              exact and gauge invariant (a single Pauli string of
                              their expansion is not).
  append_two_level_phase      exp(-i theta (|a><a| + |b><b|)), the diagonal
                              Rokhsar-Kivelson term.
  With `ancilla=` both gates implement the generator Z_anc (x) h instead of h,
  which is the signal operator of the single-ancilla ground-state filter
  (rqsvt.py). No controlled time evolution is needed.

Trotter steps
  StructuredTrotter   pure-gauge cylinder: half diagonal layer, one sublayer per
                      plaquette colour, half diagonal layer. Plaquettes in a
                      sublayer share no link and run in parallel, so the
                      two-qubit depth of a step does not grow with W or H.
  MatterTrotter       matter-coupled model of Joshi et al. on an open lattice:
                      four sublayers of commuting hopping and plaquette terms
                      between half layers of the mass and electric terms.
  TrotterEvolution    generic Suzuki-Trotter on the flat Pauli list (baseline).

All resource numbers in this package are measured on these circuits after
transpilation to {cx, rz, sx, x} (or another basis passed explicitly).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate, UCRXGate, UCRZGate
from qiskit.quantum_info import Operator
from qiskit.synthesis import SuzukiTrotter

from QuantumFields.gauge_field import (FLIP_FROM, FLIP_TO, PLAQ_PAULIS, RK_PAULIS,
                                       QuantumLinkModel, field_signs)

BASIS_GATES = ['cx', 'rz', 'sx', 'x']
ZZ_BASIS_GATES = ['rzz', 'rz', 'sx', 'x']      # native parameterised ZZ, as on trapped ions


# ---------------------------------------------------------------------------
# Elementary rotations
# ---------------------------------------------------------------------------

def append_pauli_rotation(qc: QuantumCircuit, angle: float, qubits, paulis):
    """Append exp(-i * angle * P), P the tensor product of `paulis` on `qubits`.

    Basis change to Z, CNOT ladder onto the last qubit, Rz(2*angle), undo.
    Identity factors are skipped; an all-identity P becomes a global phase.
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


def _two_level_frame(qubits, a, b):
    """Pivot qubit, fan-out targets and control pattern for the pair (a, b).

    CNOTs from the pivot (the first qubit where a and b differ) onto every
    other differing qubit map both configurations to the same bits on all
    qubits except the pivot. The pair then becomes the two pivot states,
    selected by the remaining qubits ("controls") being in `pattern`.
    """
    differ = [k for k in range(len(qubits)) if a[k] != b[k]]
    if not differ:
        raise ValueError("the two configurations must differ")
    p = differ[0]
    targets = [qubits[k] for k in differ[1:]]
    controls = [qubits[k] for k in range(len(qubits)) if k != p]
    pattern = [(a[k] ^ a[p]) if k in differ else a[k]
               for k in range(len(qubits)) if k != p]
    return qubits[p], targets, controls, pattern


def _pattern_index(pattern) -> int:
    """Index of a control pattern for the uniformly controlled gates (first control = LSB)."""
    return sum(bit << k for k, bit in enumerate(pattern))


def append_two_level_rotation(qc: QuantumCircuit, theta: float, qubits, a, b,
                              ancilla: int = None):
    """Append exp(-i theta (|a><b| + |b><a|)) on `qubits` (or with Z_anc (x) ...).

    Cost on k qubits: 2(m - 1) fan CNOTs (m = number of differing bits) plus
    one uniformly controlled rotation with k - 1 controls (2^(k-1) CNOTs before
    optimisation). For a plaquette (k = 4, m = 4) this is 13 CNOTs after
    transpilation; with the ancilla attached, 16.
    """
    pivot, targets, controls, pattern = _two_level_frame(list(qubits), a, b)
    angles = [0.0] * (2 ** len(controls))
    angles[_pattern_index(pattern)] = 2.0 * theta
    for t in targets:
        qc.cx(pivot, t)
    if ancilla is None:
        qc.append(UCRXGate(angles), [pivot] + controls)
    else:                                   # Z_anc X_p = H_p CX(anc, p) Z_p CX(anc, p) H_p
        qc.h(pivot)
        qc.cx(ancilla, pivot)
        qc.append(UCRZGate(angles), [pivot] + controls)
        qc.cx(ancilla, pivot)
        qc.h(pivot)
    for t in reversed(targets):
        qc.cx(pivot, t)


def append_two_level_phase(qc: QuantumCircuit, theta: float, qubits, a, b,
                           ancilla: int = None):
    """Append exp(-i theta (|a><a| + |b><b|)) on `qubits` (or with Z_anc (x) ...)."""
    pivot, targets, controls, pattern = _two_level_frame(list(qubits), a, b)
    for t in targets:
        qc.cx(pivot, t)
    if ancilla is None:                     # phase exp(-i theta) when controls = pattern
        flips = [c for c, bit in zip(controls, pattern) if bit == 0]
        for c in flips:
            qc.x(c)
        if len(controls) == 1:
            qc.p(-theta, controls[0])
        else:
            qc.mcp(-theta, controls[:-1], controls[-1])
        for c in flips:
            qc.x(c)
    else:                                   # exp(-i theta Z_anc) on that pattern
        angles = [0.0] * (2 ** len(controls))
        angles[_pattern_index(pattern)] = 2.0 * theta
        qc.append(UCRZGate(angles), [ancilla] + controls)
    for t in reversed(targets):
        qc.cx(pivot, t)


def append_plaquette_rotation(qc: QuantumCircuit, theta: float, plaquette,
                              ancilla: int = None, method: str = "ucr"):
    """Append exp(-i theta (U_p + U_p^dag)) on the (b, r, t, l) link qubits, exactly.

    method="ucr"    two-level rotation: 13 CNOTs, or 16 with the ancilla.
    method="pauli"  the eight commuting Pauli-string rotations of the expansion
                    (CNOT ladders): 17 CNOTs after transpilation, 40 with the
                    ancilla. Kept as the comparison baseline.
    """
    if method == "pauli":
        for label, sign in PLAQ_PAULIS:
            qubits, paulis = list(plaquette), list(label)
            if ancilla is not None:
                qubits, paulis = qubits + [ancilla], paulis + ['Z']
            append_pauli_rotation(qc, sign * theta / 8.0, qubits, paulis)
        return
    if method != "ucr":
        raise ValueError(f"unknown plaquette synthesis {method!r}")
    append_two_level_rotation(qc, theta, plaquette, FLIP_FROM, FLIP_TO, ancilla)


def append_flippability_rotation(qc: QuantumCircuit, theta: float, plaquette,
                                 ancilla: int = None, method: str = "ucr"):
    """Append exp(-i theta (U_p + U_p^dag)^2), the Rokhsar-Kivelson term."""
    if method == "pauli":
        for label, sign in RK_PAULIS:
            qubits, paulis = list(plaquette), list(label)
            if ancilla is not None:
                qubits, paulis = qubits + [ancilla], paulis + ['Z']
            append_pauli_rotation(qc, sign * theta / 8.0, qubits, paulis)
        return
    append_two_level_phase(qc, theta, plaquette, FLIP_FROM, FLIP_TO, ancilla)


def append_hopping_rotation(qc: QuantumCircuit, theta: float, qubits, ancilla: int = None):
    """Append exp(-i theta (phi^dag_a S^+_l phi_b + h.c.)) on (matter a, link l, matter b)."""
    append_two_level_rotation(qc, theta, qubits, (0, 1, 1), (1, 0, 0), ancilla)


# ---------------------------------------------------------------------------
# Resource counting
# ---------------------------------------------------------------------------

def two_qubit_depth(qc: QuantumCircuit) -> int:
    return qc.depth(lambda inst: inst.operation.num_qubits == 2)


def two_qubit_count(qc: QuantumCircuit) -> int:
    return sum(1 for inst in qc.data if inst.operation.num_qubits == 2)


def circuit_resources(qc: QuantumCircuit, basis_gates=None,
                      optimization_level: int = 3) -> dict:
    """Transpile to `basis_gates` (default {cx, rz, sx, x}) and count gates."""
    tqc = transpile(qc, basis_gates=basis_gates or BASIS_GATES,
                    optimization_level=optimization_level)
    n2 = two_qubit_count(tqc)
    return {'qubits': tqc.num_qubits,
            'two_qubit_gates': n2,
            'one_qubit_gates': tqc.size() - n2,
            'two_qubit_depth': two_qubit_depth(tqc),
            'depth': tqc.depth()}


# ---------------------------------------------------------------------------
# Trotter steps
# ---------------------------------------------------------------------------

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


def palindromic_schedule(n_layers: int, order: int) -> list:
    """(sublayer index, fraction of dt) for one step of the given order.

    order=1:  each sublayer once with the full time step;
    order=2:  symmetric (Strang) sequence, half steps on all but the last
              sublayer, applied forward and then backward.
    """
    if order == 1:
        return [(k, 1.0) for k in range(n_layers)]
    if order != 2:
        raise ValueError("order must be 1 or 2")
    return ([(k, 0.5) for k in range(n_layers - 1)] + [(n_layers - 1, 1.0)]
            + [(k, 0.5) for k in reversed(range(n_layers - 1))])


class StructuredTrotter:
    """One structured Trotter step of duration dt for the pure-gauge cylinder.

        exp(-i dt H_diag / 2) . prod_c exp(-i dt H_B^(c)) . exp(-i dt H_diag / 2)

    H_diag = H_E + H_RK is diagonal and H_B^(c) collects the plaquettes of
    colour c. With order=1 the error per step is O(dt^2) (global O(dt)); with
    order=2 the colour sequence is palindromic and the global error is O(dt^2).
    The diagonal layers are split symmetrically in both cases. For the uniform
    field [H_E, H_B] = 0, so the only Trotter error comes from plaquettes of
    different colours that share links.
    """

    def __init__(self, lattice, g=0.0, J=1.0, dt=0.05, rk=0.0,
                 field_pattern="uniform", order: int = 1, colouring="auto",
                 synthesis: str = "ucr"):
        self.lattice = lattice
        self.n_qubits = lattice.n_qubits
        self.g, self.J, self.rk, self.dt = g, J, rk, dt
        self.order = order
        self.synthesis = synthesis
        self.signs = field_signs(lattice, field_pattern)
        self.plaq_sublayers = lattice.get_plaquette_sublayers(colouring)
        self.schedule = palindromic_schedule(len(self.plaq_sublayers), order)

    def build_step(self, barriers: bool = True) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        self._apply_diagonal_layer(qc, 0.5)
        if barriers:
            qc.barrier()
        for k, fraction in self.schedule:
            self._apply_plaquette_sublayer(qc, self.plaq_sublayers[k], fraction)
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
            for q in range(self.n_qubits):
                qc.rz(-self.g * self.signs[q] * self.dt * fraction, q)
        if self.rk != 0.0:
            for layer in self.plaq_sublayers:
                for plaquette in layer:
                    append_flippability_rotation(
                        qc, self.rk * self.dt * fraction, plaquette, method=self.synthesis)


class MatterTrotter:
    """Structured Trotter step for the matter-coupled model (Joshi et al. 2026).

        exp(-i dt H_diag / 2) . prod_k exp(-i dt H^(k)) . exp(-i dt H_diag / 2)

    H_diag = mass + electric term (single-qubit Z rotations). H^(k), k = 1..4,
    are the four sublayers of OpenLattice.sublayers(): one hopping matching
    plus one class of plaquettes, all acting on disjoint qubits. The circuit
    equals the product of exponentials of the ScalarQEDModel terms exactly,
    global phase included.
    """

    def __init__(self, lattice, kappa=1.0, m=0.0, g=0.0, J=0.0, dt=0.05, order: int = 1):
        self.lattice = lattice
        self.n_qubits = lattice.n_qubits
        self.kappa, self.m, self.g, self.J, self.dt = kappa, m, g, J, dt
        self.order = order
        self.sublayers = [layer for layer in lattice.sublayers()
                          if any(kind == 'hop' and kappa != 0.0 or kind == 'plaq' and J != 0.0
                                 for kind, _ in layer)]
        self.schedule = palindromic_schedule(len(self.sublayers), order)
        self._hops = lattice.hopping_terms()
        self._plaqs = lattice.get_plaquettes()

    def build_step(self, barriers: bool = False) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        self._apply_diagonal_layer(qc, 0.5)
        for k, fraction in self.schedule:
            if barriers:
                qc.barrier()
            self._apply_sublayer(qc, self.sublayers[k], fraction)
        if barriers:
            qc.barrier()
        self._apply_diagonal_layer(qc, 0.5)
        return qc

    def build_evolution(self, n_steps: int, initial_bits: int = None) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        if initial_bits is not None:
            for q in range(self.n_qubits):
                if (initial_bits >> q) & 1:
                    qc.x(q)
        step = self.build_step()
        for _ in range(n_steps):
            qc.compose(step, inplace=True)
        return qc

    def _apply_sublayer(self, qc, layer, fraction):
        lat = self.lattice
        for kind, idx in layer:
            if kind == 'hop' and self.kappa != 0.0:
                link, a, b, sign = self._hops[idx]
                append_hopping_rotation(qc, -self.kappa * sign * self.dt * fraction,
                                        (lat.matter_qubit(*a), link, lat.matter_qubit(*b)))
            elif kind == 'plaq' and self.J != 0.0:
                append_plaquette_rotation(qc, self.J * self.dt * fraction, self._plaqs[idx])

    def _apply_diagonal_layer(self, qc, fraction):
        """Mass m s_j (1 - Z_j)/2 and electric -(g/2) Z_l as Z rotations."""
        lat = self.lattice
        tau = self.dt * fraction
        if self.g != 0.0:
            for link in range(lat.n_links):
                qc.rz(-self.g * tau, link)
        if self.m != 0.0:
            for s in range(lat.n_sites):
                i, j = lat.site_coords(s)
                sign = lat.parity(i, j)
                qc.rz(-self.m * sign * tau, lat.matter_qubit(i, j))
                qc.global_phase -= 0.5 * self.m * sign * tau


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
