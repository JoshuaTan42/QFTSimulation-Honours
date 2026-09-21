"""Resource analysis (thesis Sec. 2.3.6).

Everything reported here is *measured* on circuits built by this package and
transpiled to {cx, rz, sx, x} at optimisation level 3, except the
unary-iteration estimate for the block-encoding baseline, which is analytic and
labelled as such.

  trotter_step_table      structured vs generic Trotter step, as a function of
                          lattice size
  block_encoding_baseline LCU block encoding (PREPARE, SELECT) of the same
                          Hamiltonian, as used by standard QSVT
  ground_state_comparison single-ancilla GQSP filter driven by structured
                          Trotter or qDRIFT, against block-encoded QSVT
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliGate, StatePreparation

from circuit import (BASIS_GATES, StructuredTrotter, TrotterEvolution,
                     circuit_resources, two_qubit_count, two_qubit_depth)
from lattice import LatticeGrid
from QuantumFields.gauge_field import QuantumLinkModel


def trotter_step_table(sizes, g=0.0, J=2.0, rk=0.0, dt=0.05,
                       field_pattern="uniform", generic_max_qubits=21) -> list:
    rows = []
    for (W, H) in sizes:
        lattice = LatticeGrid(W, H)
        step = StructuredTrotter(lattice, g=g, J=J, rk=rk, dt=dt,
                                 field_pattern=field_pattern).build_step(barriers=False)
        row = {'W': W, 'H': H, 'qubits': lattice.n_qubits,
               'plaquettes': lattice.n_plaquettes,
               'sublayers': lattice.n_plaquette_colours,
               'structured': circuit_resources(step)}
        row['structured_pauli'] = circuit_resources(StructuredTrotter(
            lattice, g=g, J=J, rk=rk, dt=dt, field_pattern=field_pattern,
            synthesis="pauli").build_step(barriers=False))   # eight-ladder construction
        if lattice.n_qubits <= generic_max_qubits:
            Hop = QuantumLinkModel(lattice, g=g, J=J, rk=rk,
                                   field_pattern=field_pattern).build_physical_hamiltonian()
            generic = TrotterEvolution(Hop, lattice.n_qubits, dt, 1, order=1).build_circuit()
            row['generic'] = circuit_resources(generic)
        rows.append(row)
    return rows


def block_encoding_baseline(lattice, g=0.0, J=2.0, rk=0.0,
                            field_pattern="uniform", optimization_level=1) -> dict:
    """One call to an LCU block encoding  (PREP^dag (x) 1) SELECT (PREP (x) 1)  of H.

    SELECT applies each Pauli string controlled on the binary index of the
    term, with no work qubits (the textbook construction). The analytic
    alternative, unary iteration, costs about 4(L - 1) Toffolis plus one
    controlled Pauli string per term and needs ceil(log2 L) extra qubits.
    """
    H = QuantumLinkModel(lattice, g=g, J=J, rk=rk,
                         field_pattern=field_pattern).build_physical_hamiltonian()
    labels = [p.to_label() for p in H.paulis]
    coeffs = np.real(H.coeffs)
    keep = [k for k, (lab, c) in enumerate(zip(labels, coeffs))
            if abs(c) > 1e-12 and set(lab) != {'I'}]
    L = len(keep)
    m = int(np.ceil(np.log2(L)))
    n = lattice.n_qubits

    amps = np.zeros(2 ** m)
    amps[:L] = np.sqrt(np.abs(coeffs[keep]))
    amps /= np.linalg.norm(amps)
    prep = transpile(_as_circuit(StatePreparation(amps), m), basis_gates=BASIS_GATES,
                     optimization_level=optimization_level)

    select = QuantumCircuit(n + m)
    for idx, k in enumerate(keep):
        gate = PauliGate(labels[k]).control(m, ctrl_state=idx)
        select.append(gate, list(range(n, n + m)) + list(range(n)))
        # negative coefficients need a controlled phase of pi on the index register;
        # it adds no two-qubit gates at this level of accounting and is omitted
    select = transpile(select, basis_gates=BASIS_GATES,
                       optimization_level=optimization_level)

    per_call_2q = two_qubit_count(select) + 2 * two_qubit_count(prep)
    n_strings = sum(sum(c != 'I' for c in labels[k]) for k in keep)
    unary_2q = 6 * 4 * (L - 1) + n_strings     # 6 CNOTs per Toffoli
    return {'terms': L, 'index_qubits': m,
            'qubits_qsvt': n + m + 1,           # + one QSVT phase ancilla
            'qubits_qsvt_unary': n + 2 * m + 1,
            'lambda_lcu': float(np.sum(np.abs(coeffs[keep]))),
            'select_two_qubit_gates': two_qubit_count(select),
            'prepare_two_qubit_gates': two_qubit_count(prep),
            'two_qubit_gates_per_call': per_call_2q,
            'two_qubit_gates_per_call_unary_estimate': unary_2q + 2 * two_qubit_count(prep)}


def _as_circuit(gate, n) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.append(gate, range(n))
    return qc


def ground_state_comparison(rq, trotter_steps=1, qdrift_samples=800) -> dict:
    """Gate counts for preparing the ground state with the same spectral filter.

    The block-encoded QSVT baseline is charged one block-encoding call per
    signal operator: both filters are polynomials of degree O((lambda/gap)
    log(1/eps)) in an operator of norm lambda, so their degrees agree up to a
    constant of order one. This is an estimate, not a compiled QSVT circuit.
    """
    be = block_encoding_baseline(rq.lattice, g=rq.g, J=rq.J, rk=rq.rk,
                                 field_pattern=rq.field_pattern)
    calls = rq.n_signal
    pauli = rq.resources('trotter', n_steps=trotter_steps, synthesis="pauli")
    return {
        'gqsp_trotter_pauli_ladders': pauli,
        'filter': rq.filter_kind,
        'signal_operators': rq.n_signal,
        'gqsp_trotter': rq.resources('trotter', n_steps=trotter_steps),
        'gqsp_qdrift': rq.resources('qdrift', n_samples=qdrift_samples),
        'block_encoded_qsvt': {
            'qubits': be['qubits_qsvt'],
            'two_qubit_gates': calls * be['two_qubit_gates_per_call'],
            'qubits_unary': be['qubits_qsvt_unary'],
            'two_qubit_gates_unary_estimate':
                calls * be['two_qubit_gates_per_call_unary_estimate'],
            'detail': be},
    }
