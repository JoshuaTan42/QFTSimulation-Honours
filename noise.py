"""Gate-noise study of the ground-state filter by Monte Carlo trajectories.

Noise model: after every CX a uniformly random non-identity two-qubit Pauli is
applied with probability p (two-qubit depolarising noise); single-qubit gates
are ideal. Averaging pure-state trajectories reproduces the noisy channel.

Two estimators are reported for the post-selected energy:
    raw     ancilla measured in |0>, nothing else
    gauge   additionally project the link register onto the physical sector
            (Gauss's law and the plate charges of the initial state)
The second is the error-detection scheme a gauge theory provides for free: a
Pauli error almost always violates Gauss's law somewhere, so discarding those
shots removes most of the damage at the price of a lower acceptance rate.

The simulator handles the transpiled gate set {cx, rz, sx, x} on up to about
14 qubits and is validated against qiskit.quantum_info.Statevector at p = 0.
"""

from __future__ import annotations

import numpy as np
from qiskit import transpile

from circuit import BASIS_GATES
from QuantumFields.gauge_field import QuantumLinkModel

_SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
_PAULI = {'X': np.array([[0, 1], [1, 0]], dtype=complex),
          'Y': np.array([[0, -1j], [1j, 0]]),
          'Z': np.array([[1, 0], [0, -1]], dtype=complex)}


def compile_program(qc) -> tuple:
    """Transpile to {cx, rz, sx, x} and flatten to (name, qubits, angle) tuples."""
    tqc = transpile(qc, basis_gates=BASIS_GATES, optimization_level=1)
    program = []
    for inst in tqc.data:
        name = inst.operation.name
        if name == 'barrier':
            continue
        qubits = tuple(tqc.find_bit(q).index for q in inst.qubits)
        angle = float(inst.operation.params[0]) if name == 'rz' else 0.0
        program.append((name, qubits, angle))
    return program, tqc.num_qubits, float(tqc.global_phase)


class TrajectorySimulator:
    """Statevector stored as a rank-n tensor with axis k = qubit n-1-k, so that
    ravel() is the Qiskit (little-endian) statevector."""

    def __init__(self, n_qubits: int):
        self.n = n_qubits

    def _axis(self, q):
        return self.n - 1 - q

    def apply_1q(self, psi, U, q):
        ax = self._axis(q)
        return np.moveaxis(np.tensordot(U, psi, axes=([1], [ax])), 0, ax)

    def apply_rz(self, psi, angle, q):
        shape = [1] * self.n
        shape[self._axis(q)] = 2
        phases = np.array([np.exp(-0.5j * angle), np.exp(0.5j * angle)]).reshape(shape)
        return psi * phases

    def apply_cx(self, psi, c, t):
        idx = [slice(None)] * self.n
        idx[self._axis(c)] = 1
        idx = tuple(idx)
        t_ax = self._axis(t) - (1 if self._axis(t) > self._axis(c) else 0)
        psi = psi.copy()
        psi[idx] = np.flip(psi[idx], axis=t_ax)
        return psi

    def run(self, program, p: float = 0.0, rng=None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        psi = np.zeros((2,) * self.n, dtype=complex)
        psi[(0,) * self.n] = 1.0
        labels = ['I', 'X', 'Y', 'Z']
        for name, qubits, angle in program:
            if name == 'rz':
                psi = self.apply_rz(psi, angle, qubits[0])
            elif name == 'sx':
                psi = self.apply_1q(psi, _SX, qubits[0])
            elif name == 'x':
                psi = np.flip(psi, axis=self._axis(qubits[0]))
            elif name == 'cx':
                psi = self.apply_cx(psi, *qubits)
                if p > 0.0 and rng.random() < p:
                    k = int(rng.integers(1, 16))          # 15 non-identity Paulis
                    for q, lab in zip(qubits, (labels[k // 4], labels[k % 4])):
                        if lab != 'I':
                            psi = self.apply_1q(psi, _PAULI[lab], q)
            else:
                raise ValueError(f"gate {name!r} is outside the supported basis")
        return psi.reshape(-1)


def noisy_filter_study(rq, error_rates=(0.0, 1e-4, 3e-4, 1e-3, 3e-3), method="trotter",
                       n_steps=1, trajectories=200, seed=0) -> dict:
    """Energy of the filtered state against the two-qubit error rate.

    `rq` is an RQSVTGroundState on a lattice small enough for a statevector
    (N_q + 1 <= ~14). Returns, per error rate, the raw and gauge-post-selected
    energies with their acceptance probabilities.
    """
    n = rq.n
    program, n_qubits, _ = compile_program(rq.build_circuit(method, n_steps=n_steps))
    n_cx = sum(1 for g in program if g[0] == 'cx')
    sim = TrajectorySimulator(n_qubits)
    H_full = QuantumLinkModel(rq.lattice, g=rq.g, J=rq.J, rk=rq.rk,
                              field_pattern=rq.field_pattern
                              ).build_physical_hamiltonian().to_matrix(sparse=True)
    rng = np.random.default_rng(seed)
    rows = []
    for p in error_rates:
        shots = 1 if p == 0.0 else trajectories
        acc = dict(raw_num=0.0, raw_den=0.0, g_num=0.0, g_den=0.0)
        for _ in range(shots):
            sv = sim.run(program, p, rng)
            phi = sv[:2 ** n]                              # ancilla (MSB) = 0
            acc['raw_num'] += np.vdot(phi, H_full @ phi).real
            acc['raw_den'] += np.vdot(phi, phi).real
            phys = phi[rq.basis]                           # physical sector only
            acc['g_num'] += np.vdot(phys, rq._H @ phys).real
            acc['g_den'] += np.vdot(phys, phys).real
        rows.append({'p': p,
                     'expected_errors': p * n_cx,
                     'energy_raw': acc['raw_num'] / acc['raw_den'],
                     'energy_gauge': acc['g_num'] / acc['g_den'],
                     'accept_raw': acc['raw_den'] / shots,
                     'accept_gauge': acc['g_den'] / shots,
                     'gauge_weight': acc['g_den'] / acc['raw_den']})
    return {'lattice': f"{rq.lattice.width}x{rq.lattice.height}", 'qubits': n_qubits,
            'cx_gates': n_cx, 'E0': rq.E0, 'gap': rq.gap, 'method': method,
            'n_steps': n_steps, 'trajectories': trajectories, 'rows': rows}
