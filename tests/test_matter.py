"""Matter-coupled model of Joshi et al.: geometry, Hamiltonian, sector, circuits."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.linalg import expm

from circuit import MatterTrotter, append_hopping_rotation, circuit_resources
from lattice import OpenLattice
from matter_sector import MatterSector
from QuantumFields.matter_field import ScalarQEDModel, string_state_bits


def _sublayer_qubits(lat, kind, idx):
    if kind == 'hop':
        link, a, b, _ = lat.hopping_terms()[idx]
        return {link, lat.matter_qubit(*a), lat.matter_qubit(*b)}
    return set(lat.get_plaquettes()[idx])


@pytest.mark.parametrize("W,H,nq", [(4, 3, 29), (5, 4, 51), (3, 2, 13), (3, 3, 21)])
def test_open_lattice_counts_and_sublayers(W, H, nq):
    lat = OpenLattice(W, H)
    assert lat.n_qubits == nq                                  # Joshi et al.: 29 and 51
    layers = lat.sublayers()
    assert len(layers) == 4
    terms = [t for layer in layers for t in layer]
    assert len(terms) == lat.n_links + lat.n_plaquettes        # every term exactly once
    for layer in layers:                                       # disjoint inside a sublayer
        used = set()
        for kind, idx in layer:
            q = _sublayer_qubits(lat, kind, idx)
            assert not used & q
            used |= q


@pytest.mark.parametrize("W,H", [(3, 2), (2, 3)])
def test_matter_sector_matches_full_space(W, H):
    lat = OpenLattice(W, H)
    model = ScalarQEDModel(lat, kappa=1.0, m=3.0, g=6.0, J=2.0)
    H_full = model.build_hamiltonian().to_matrix(sparse=True)
    gauss = model.gauss.build_gauss_operators()
    for G in gauss:                                            # every term is gauge invariant
        assert np.abs(((G @ model.build_hamiltonian()) - (model.build_hamiltonian() @ G))
                      .simplify(atol=1e-12).coeffs).sum() < 1e-12
    physical = np.ones(2 ** lat.n_qubits, bool)
    for G in gauss:
        physical &= np.abs(G.to_matrix(sparse=True).diagonal()) < 1e-12
    sector = MatterSector(lat)
    assert sector.dim == physical.sum()
    assert np.array_equal(sector.basis, np.nonzero(physical)[0])
    block = H_full[sector.basis][:, sector.basis].toarray()
    assert np.abs(block - sector.hamiltonian(1.0, 3.0, 6.0, 2.0).toarray()).max() < 1e-12
    sv = np.zeros(2 ** lat.n_qubits)
    sv[string_state_bits(lat)] = 1.0
    assert model.gauss.max_violation(sv) < 1e-12               # string state is physical


def test_hopping_gate_and_step_are_exact():
    lat = OpenLattice(2, 2)
    model = ScalarQEDModel(lat, kappa=1.0, m=3.0, g=6.0, J=2.0, static_charges={})
    link, a, b, _ = lat.hopping_terms()[0]
    h = model.hopping_operator(lat.hopping_terms()[0]).to_matrix()
    for anc in (None, lat.n_qubits):
        qc = QuantumCircuit(lat.n_qubits + (anc is not None))
        append_hopping_rotation(qc, 0.37, (lat.matter_qubit(*a), link, lat.matter_qubit(*b)),
                                ancilla=anc)
        gen = h if anc is None else np.kron(np.diag([1.0, -1.0]), h)
        assert np.linalg.norm(Operator(qc).data - expm(-0.37j * gen), 2) < 1e-12
    dt = 0.1
    trotter = MatterTrotter(lat, 1.0, 3.0, 6.0, 2.0, dt)
    n = lat.n_qubits
    diag = (model.build_mass_term() + model.build_electric_term()).to_matrix()
    ref = expm(-0.5j * dt * diag)
    assert lat.n_plaquettes == 1
    magnetic = model.build_magnetic_term().to_matrix()
    for k, fraction in trotter.schedule:
        Hk = np.zeros((2 ** n, 2 ** n), complex)
        for kind, idx in trotter.sublayers[k]:
            if kind == 'hop':
                term = lat.hopping_terms()[idx]
                Hk -= term[3] * model.hopping_operator(term).to_matrix()
            else:
                Hk += magnetic
        ref = expm(-1j * dt * fraction * Hk) @ ref
    ref = expm(-0.5j * dt * diag) @ ref
    assert np.linalg.norm(Operator(trotter.build_step()).data - ref, 2) < 1e-12


@pytest.mark.parametrize("order,power", [(1, 2.0), (2, 3.0)])
def test_matter_trotter_order(order, power):
    lat = OpenLattice(2, 2)
    H = ScalarQEDModel(lat, 1.0, 3.0, 6.0, 2.0, static_charges={}).build_hamiltonian().to_matrix()
    dts = (0.04, 0.02, 0.01)
    errs = [np.linalg.norm(Operator(MatterTrotter(lat, 1.0, 3.0, 6.0, 2.0, dt, order)
                                    .build_step()).data - expm(-1j * dt * H), 2) for dt in dts]
    assert abs(np.polyfit(np.log(dts), np.log(errs), 1)[0] - power) < 0.15


def test_two_qubit_counts_against_joshi_table_s1():
    """4x3, J = 0: 119 two-qubit gates per step, as in Table S1; 5x4, J = 2: 373 (< 385)."""
    small = circuit_resources(MatterTrotter(OpenLattice(4, 3), 1.0, 3.0, 6.0, 0.0, 0.1).build_step())
    large = circuit_resources(MatterTrotter(OpenLattice(5, 4), 1.0, 3.0, 6.0, 2.0, 0.1).build_step())
    assert small['two_qubit_gates'] == 119
    assert large['two_qubit_gates'] == 373 and large['qubits'] == 51


def test_joshi_sector_dimensions():
    assert MatterSector(OpenLattice(4, 3)).dim == 1826
