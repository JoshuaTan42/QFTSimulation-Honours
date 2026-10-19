"""Quench dynamics, the sector Trotter emulator, entanglement and the hardware runner."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from circuit import MatterTrotter, StructuredTrotter
from dynamics import SectorTrotter, bipartite_entropy, midplane_qubits, quench, trotter_budget
from lattice import LatticeGrid, OpenLattice
from matter_sector import MatterSector
from QuantumFields.matter_field import string_state_bits
from rqsvt import RQSVTGroundState
from sector import GaugeSector


@pytest.mark.parametrize("W,H,plates,order", [(2, 3, "fixed", 1), (2, 3, "fixed", 2),
                                              (3, 3, "free", 2)])
def test_sector_trotter_equals_the_circuit(W, H, plates, order):
    lat = LatticeGrid(W, H)
    sector = GaugeSector(lat, plates=plates)
    gs = sector.ground_state(g=0.7, J=2.0, rk=0.3, field_pattern="staggered")
    idx = gs['support']
    k = sector.most_flippable(idx)
    qc = StructuredTrotter(lat, g=0.7, J=2.0, rk=0.3, dt=0.13, field_pattern="staggered",
                           order=order).build_evolution(3)
    sv = Statevector.from_int(sector.label(int(idx[k])), 2 ** lat.n_qubits).evolve(qc).data
    emu = SectorTrotter.for_gauge(sector, g=0.7, J=2.0, rk=0.3, field_pattern="staggered",
                                  dt=0.13, order=order, support=idx)
    psi0 = np.zeros(len(idx), complex)
    psi0[k] = 1.0
    full = np.zeros(2 ** lat.n_qubits, complex)
    full[sector.basis[idx]] = emu.step(psi0, 3)
    assert np.linalg.norm(sv - full) < 1e-12


@pytest.mark.parametrize("order", [1, 2])
def test_sector_trotter_equals_the_matter_circuit(order):
    lat = OpenLattice(3, 2)
    sector = MatterSector(lat)
    bits = string_state_bits(lat)
    qc = MatterTrotter(lat, 1.0, 3.0, 6.0, 2.0, 0.11, order).build_evolution(4, initial_bits=bits)
    sv = Statevector.from_label('0' * lat.n_qubits).evolve(qc).data
    emu = SectorTrotter.for_matter(sector, 1.0, 3.0, 6.0, 2.0, 0.11, order)
    assert np.linalg.norm(sv - sector.embed(emu.step(sector.product_state(bits), 4))) < 1e-12


def test_bipartite_entropy_matches_dense_svd():
    lat = LatticeGrid(2, 3)
    sector = GaugeSector(lat, plates="fixed")
    vec = sector.ground_state(J=1.0, rk=0.2)['state']
    cut = midplane_qubits(lat)
    full = sector.embed(vec).reshape([2] * lat.n_qubits)        # axis k = qubit n-1-k
    axes_a = [lat.n_qubits - 1 - q for q in cut]
    axes_b = [a for a in range(lat.n_qubits) if a not in axes_a]
    M = np.transpose(full, axes_a + axes_b).reshape(2 ** len(axes_a), -1)
    s = np.linalg.svd(M, compute_uv=False) ** 2
    s = s[s > 1e-15]
    assert abs(bipartite_entropy(sector.basis, vec, cut) + np.sum(s * np.log(s))) < 1e-10


def test_quench_observables():
    out = quench(4, 2, J=1.0, t_max=2.0, n_times=11, trotter=((0.05, 2),))
    assert out['return_probability'][0] == pytest.approx(1.0)
    assert out['entropy_midplane'][0] == pytest.approx(0.0, abs=1e-12)
    assert max(out['entropy_midplane']) > 0.1
    tr = out['trotter'][0]
    assert tr['fidelity'][-1] > 0.999 and tr['two_qubit_gates_per_step'] > 0
    assert len(out['flippability'][0]) == 2                       # one value per plaquette row


def test_second_order_quench_stays_accurate_longer():
    budget = trotter_budget(2, 2, J=2.0, dts=(0.2,), orders=(1, 2), t_max=2.0)
    first, second = budget['rows']
    assert second['time_within_target'] > first['time_within_target']


def test_second_order_filter_circuit_matches_emulator():
    lat = LatticeGrid(2, 3)
    rq = RQSVTGroundState(lat, J=2.0, rk=0.3, plates="fixed", filter_kind="gaussian", degree=6)
    qc = rq.build_circuit("trotter", n_steps=2, order=2)
    sv = Statevector.from_label("0" * qc.num_qubits).evolve(qc).data
    phi = sv[:2 ** lat.n_qubits][rq.basis]
    assert np.linalg.norm(phi - rq.run("trotter", n_steps=2, order=2)["state"]) < 1e-10
    rq_full = RQSVTGroundState(lat, J=2.0, plates="fixed", filter_kind="gaussian")
    e1 = abs(rq_full.run("trotter", n_steps=2, order=1)["energy"] - rq_full.E0)
    e2 = abs(rq_full.run("trotter", n_steps=2, order=2)["energy"] - rq_full.E0)
    assert e2 < 0.1 * e1


def test_plaquette_basis_measurement_reads_the_magnetic_term():
    from hardware import append_plaquette_basis, plaquette_basis_observable
    lat = LatticeGrid(2, 3)
    sector = GaugeSector(lat, plates="fixed")
    vec = sector.ground_state(J=1.0, rk=0.4)['state']
    exact = sector.plaquette_expectations(vec)
    plaqs = lat.get_plaquettes()
    for members in lat.plaquette_sublayer_indices():
        qc = QuantumCircuit(lat.n_qubits)
        qc.initialize(sector.embed(vec).astype(complex))
        append_plaquette_basis(qc, [plaqs[p] for p in members])
        probs = Statevector(qc).probabilities()
        keep = np.nonzero(probs > 1e-14)[0]
        bits = ((keep[:, None] >> np.arange(lat.n_qubits)[None, :]) & 1).astype(np.int8)
        measured = plaquette_basis_observable(bits, probs[keep], [plaqs[p] for p in members],
                                              members)
        for p in members:
            assert measured[p][0] == pytest.approx(exact[p], abs=1e-10)


def test_hardware_quench_on_ideal_simulator():
    pytest.importorskip("qiskit_aer")
    from hardware import quench_experiment
    out = quench_experiment(2, 2, J=2.0, dt=0.3, steps=(0, 1), shots=20000, verbose=False)
    for row, ref in zip(out['measured'], out['reference']):
        assert row['physical_fraction'] == 1.0                     # no noise, no violations
        assert abs(row['gauss']['return_probability'][0] - ref['return_probability']) < 0.02
        assert abs(row['magnetic_energy'][0] - ref['magnetic_energy']) < 0.05
