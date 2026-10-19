"""Core framework: lattice, Hamiltonian, sector ED, circuits, filter, Casimir analysis."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh

from casimir import analyse, ground_energy_full, ground_energy_sector, second_difference
from circuit import (StructuredTrotter, append_flippability_rotation,
                     append_plaquette_rotation, circuit_resources, trotter_error)
from lattice import LatticeGrid
from QuantumFields.gauge_field import GaussLaw, QuantumLinkModel
from rqsvt import (RQSVTGroundState, fourier_filter, gqsp_angles, gqsp_response,
                   richardson_extrapolate)
from sector import GaugeSector


def l1(op):
    return float(np.sum(np.abs(op.simplify(atol=1e-12).coeffs)))


# ---- lattice ---------------------------------------------------------------

@pytest.mark.parametrize("W,H,nq", [(2, 2, 6), (3, 3, 15), (4, 4, 28), (3, 6, 33)])
def test_link_count(W, H, nq):
    lat = LatticeGrid(W, H)
    assert lat.n_qubits == nq == W * (2 * H - 1)
    assert sorted(lat.get_link_qubit(*lat.link_info(q)) for q in range(nq)) == list(range(nq))


@pytest.mark.parametrize("W", [2, 3, 4, 5, 6, 7])
def test_plaquette_colouring(W):
    lat = LatticeGrid(W, 4)
    assert lat.sublayers_are_disjoint(lat.get_plaquette_sublayers())
    checkerboard_ok = lat.sublayers_are_disjoint(lat.get_plaquette_sublayers("checkerboard"))
    assert checkerboard_ok == (W % 2 == 0)          # fails on every odd-W cylinder


# ---- Hamiltonian -------------------------------------------------------------

def test_gauge_invariance_all_sites():
    lat = LatticeGrid(3, 3)
    H = QuantumLinkModel(lat, g=1.3, J=2.0, rk=0.7,
                         field_pattern="staggered").build_physical_hamiltonian()
    gauss = GaussLaw(lat)
    for site in range(lat.n_sites):
        G = gauss.divergence_operator(*lat.site_to_coords(site))
        assert l1((G @ H) - (H @ G)) < 1e-10


def test_uniform_field_is_conserved_staggered_is_not():
    lat = LatticeGrid(3, 3)
    HB = QuantumLinkModel(lat, J=1.0).build_magnetic_term()
    HE_u = QuantumLinkModel(lat, g=1.0).build_electric_term()
    HE_s = QuantumLinkModel(lat, g=1.0, field_pattern="staggered").build_electric_term()
    assert l1((HE_u @ HB) - (HB @ HE_u)) < 1e-12
    assert l1((HE_s @ HB) - (HB @ HE_s)) > 1.0


def test_strong_uniform_field_gives_a_product_state():
    """g = 6, J = 2: the uniform field wins, ground state |0...0>, E0 = -(g/2) N_q exactly."""
    lat = LatticeGrid(3, 3)
    H = QuantumLinkModel(lat, g=6.0, J=2.0, gauss_penalty=200.0).build_hamiltonian()
    w, v = eigsh(H.to_matrix(sparse=True), k=1, which="SA")
    assert abs(w[0] + 3.0 * lat.n_qubits) < 1e-8
    assert abs(abs(v[0, 0]) - 1.0) < 1e-8


def test_plate_charges_must_be_half_integer():
    lat = LatticeGrid(2, 3)
    with pytest.raises(ValueError):
        GaussLaw(lat, plates="fixed", plate_charges={(0, 0): 0.0})
    with pytest.raises(ValueError):
        GaussLaw(LatticeGrid(3, 3), plates="fixed")          # staggered needs even W


# ---- sector ED against the penalised qubit Hamiltonian -----------------------------------

@pytest.mark.parametrize("W,H,plates,g,rk,pattern", [
    (2, 2, "fixed", 0.0, 0.0, "uniform"), (2, 3, "free", 0.3, 0.4, "staggered"),
    (3, 3, "free", 0.0, 0.0, "uniform"), (2, 3, "fixed", 0.7, -0.3, "staggered")])
def test_sector_matches_full_space(W, H, plates, g, rk, pattern):
    lat = LatticeGrid(W, H)
    sector = GaugeSector(lat, plates=plates)
    gs = sector.ground_state(g=g, J=2.0, rk=rk, field_pattern=pattern)
    full = ground_energy_full(W, H, g=g, J=2.0, rk=rk, field_pattern=pattern, plates=plates,
                              penalty=50.0)
    assert abs(gs["energy"] - full) < 1e-9
    assert GaussLaw(lat, plates=plates).max_violation(sector.embed(gs["state"])) < 1e-12
    Hq = QuantumLinkModel(lat, g=g, J=2.0, rk=rk, field_pattern=pattern
                          ).build_physical_hamiltonian().to_matrix(sparse=True)
    psi = sector.embed(gs["state"])
    assert abs(np.vdot(psi, Hq @ psi).real - gs["energy"]) < 1e-9


def test_sector_dimensions_3x3():
    sector = GaugeSector(LatticeGrid(3, 3))
    gs = sector.ground_state(J=2.0)
    assert sector.dim == 1792 and len(gs["support"]) == 21
    assert gs["gap"] > 1.0                                  # unique inside its charge sector


# ---- circuits -------------------------------------------------------------------

@pytest.mark.parametrize("method", ["ucr", "pauli"])
@pytest.mark.parametrize("with_ancilla", [False, True])
def test_plaquette_gates_are_exact(method, with_ancilla):
    lat = LatticeGrid(2, 2)
    model = QuantumLinkModel(lat)
    plaq = lat.get_plaquettes()[1]
    Z = np.diag([1.0, -1.0])
    for append, op in ((append_plaquette_rotation, model.plaquette_operator(plaq)),
                       (append_flippability_rotation, model.flippability_operator(plaq))):
        qc = QuantumCircuit(7 if with_ancilla else 6)
        append(qc, 0.41, plaq, ancilla=6 if with_ancilla else None, method=method)
        gen = np.kron(Z, op.to_matrix()) if with_ancilla else op.to_matrix()
        assert np.linalg.norm(Operator(qc).data - expm(-0.41j * gen), 2) < 1e-12


def test_compact_plaquette_gate_costs():
    lat = LatticeGrid(2, 2)
    plaq = lat.get_plaquettes()[0]
    counts = {}
    for method in ("ucr", "pauli"):
        qc = QuantumCircuit(7)
        append_plaquette_rotation(qc, 0.3, plaq, ancilla=6, method=method)
        counts[method] = circuit_resources(qc)["two_qubit_gates"]
    assert counts["ucr"] <= 16 < counts["pauli"]


@pytest.mark.parametrize("order,power", [(1, 2.0), (2, 3.0)])
def test_trotter_error_scaling(order, power):
    lat = LatticeGrid(2, 3)
    dts = (0.1, 0.05, 0.025)
    errs = [trotter_error(lat, 0.4, 2.0, dt, rk=0.3, field_pattern="staggered", order=order)
            for dt in dts]
    assert abs(np.polyfit(np.log(dts), np.log(errs), 1)[0] - power) < 0.15


def test_two_qubit_depth_is_size_independent():
    depths = {W: circuit_resources(StructuredTrotter(LatticeGrid(W, H), J=2.0)
                                   .build_step(False))["two_qubit_depth"]
              for W, H in ((2, 3), (4, 3), (4, 5), (6, 4))}
    assert len(set(depths.values())) == 1


# ---- RQSVT ------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["step", "gaussian"])
def test_gqsp_angles_reproduce_the_filter(kind):
    D = 18
    c = fourier_filter(kind, D, 0.3, 0.2)
    theta, phi, lam = gqsp_angles(c)
    ks = np.arange(-D, D + 1)
    for t in np.linspace(-3.0, 3.0, 13):
        target = np.exp(1j * t * ks) @ c
        assert abs(gqsp_response(theta, phi, lam, np.exp(1j * t)) * np.exp(-1j * D * t)
                   - target) < 1e-6


@pytest.mark.parametrize("method,kw", [("trotter", {"n_steps": 2}),
                                       ("qdrift", {"n_samples": 7, "seed": 3})])
def test_rqsvt_circuit_matches_emulator(method, kw):
    lat = LatticeGrid(2, 3)
    rq = RQSVTGroundState(lat, J=2.0, g=0.5, rk=0.3, field_pattern="staggered",
                          plates="fixed", filter_kind="gaussian", degree=8)
    qc = rq.build_circuit(method, **kw)
    assert qc.num_qubits == lat.n_qubits + 1                 # exactly one ancilla
    sv = Statevector.from_label("0" * qc.num_qubits).evolve(qc).data
    anc0 = sv[:2 ** lat.n_qubits]
    phi = anc0[rq.basis]
    assert np.linalg.norm(anc0) ** 2 - np.linalg.norm(phi) ** 2 < 1e-12   # no gauge leakage
    assert np.linalg.norm(phi - rq.run(method, **kw)["state"]) < 1e-10
    rq.DENSE_DIM = 0                                           # sparse path agrees too
    assert np.linalg.norm(phi - rq.run(method, **kw)["state"]) < 1e-10


def test_rqsvt_prepares_the_3x3_ground_state():
    rq = RQSVTGroundState(LatticeGrid(3, 3), J=2.0, filter_kind="gaussian")
    assert rq.summary()["qubits"] == 16
    assert rq.run("exact")["overlap"] > 0.9999
    est = rq.estimate_energy("trotter", (1, 2, 4))
    assert abs(est["error"]) < 5e-4
    errs = [abs(r["energy"] - rq.E0) for r in est["runs"]]
    assert errs[0] > errs[1] > errs[2]


def test_qdrift_converges_like_one_over_n():
    rq = RQSVTGroundState(LatticeGrid(2, 3), J=2.0, plates="fixed", filter_kind="gaussian")
    est = rq.estimate_energy("qdrift", (50, 200), n_circuits=12, seed=1)
    e50, e200 = (abs(r["energy"] - rq.E0) for r in est["runs"])
    assert e200 < 0.5 * e50


def test_richardson():
    hs = [1.0, 0.5, 0.25]
    assert abs(richardson_extrapolate(hs, [3 + 2 * h - 5 * h * h for h in hs]) - 3.0) < 1e-12


# ---- Casimir pipeline ------------------------------------------------------------------

def test_second_difference_removes_bulk():
    d = np.arange(1, 9, dtype=float)
    _, d2 = second_difference(d, 1.7 - 2.3 * d)
    assert np.max(np.abs(d2)) < 1e-12


def test_power_law_recovered_from_synthetic_data():
    d = np.arange(2, 12, dtype=float)
    out = analyse(d, 0.4 - 1.1 * d - 0.3 / d ** 2, d_min=2)
    assert abs(out["power_law"]["alpha"] - 3.0) < 0.05
    assert out["fraction_attractive"] == 1.0


def test_casimir_energies_regression():
    assert abs(ground_energy_sector(4, 4)["energy"] + 6.49837353) < 1e-7
    assert abs(ground_energy_sector(3, 3)["energy"] + 2.93352199) < 1e-7


# ---- explorer ---------------------------------------------------------------------------

def test_explorer_headless(tmp_path):
    from visualize import CasimirExplorer
    ex = CasimirExplorer(W=2, d=2)
    ex.set_params(W=3, d=2, J=2.0, g=1.0, pattern="staggered", rk=0.2)
    ex._on_view("quench")
    for _ in range(3):
        ex.step()
        ex.refresh(light=True)
    assert len(ex.trace["t"]) >= 3
    assert abs(ex.trace["echo"][0] - 1.0) < 1e-12
    assert ex.save(str(tmp_path / "explorer.png"))


# ---- noise and experiment drivers ---------------------------------------------------------

def test_trajectory_simulator_matches_qiskit():
    from noise import TrajectorySimulator, compile_program
    rq = RQSVTGroundState(LatticeGrid(2, 2), J=2.0, plates="fixed", filter_kind="gaussian")
    qc = rq.build_circuit("trotter", n_steps=1)
    program, n, phase = compile_program(qc)
    sv = TrajectorySimulator(n).run(program) * np.exp(1j * phase)
    ref = Statevector.from_label("0" * n).evolve(qc).data
    assert np.linalg.norm(sv - ref) < 1e-10


def test_gauss_law_postselection_mitigates_noise():
    from noise import noisy_filter_study
    rq = RQSVTGroundState(LatticeGrid(2, 2), J=2.0, plates="fixed", filter_kind="gaussian")
    out = noisy_filter_study(rq, error_rates=(0.0, 3e-3), trajectories=150, seed=7)
    clean, noisy = out["rows"]
    assert abs(clean["gauge_weight"] - 1.0) < 1e-12          # no leakage without noise
    assert noisy["gauge_weight"] < 0.95
    assert abs(noisy["energy_gauge"] - rq.E0) < abs(noisy["energy_raw"] - rq.E0)


def test_experiment_drivers_smoke(tmp_path, monkeypatch):
    import experiments
    monkeypatch.chdir(tmp_path)
    experiments.main(["rk-scan", "--width", "2", "--rks", "0.0,0.5", "--dmax", "6"])
    scan = __import__("json").load(open(tmp_path / "results" / "rk_scan_W2.json"))
    assert len(scan["rows"]) == 2
    scaling = experiments.filter_scaling(width=2, separations=(1, 2, 3), verbose=False)
    overlaps = [r["initial_overlap"] for r in scaling["rows"]]
    assert overlaps[0] > overlaps[1] > overlaps[2]            # product-state overlap decays with d
    assert all(abs(r["richardson_error"]) < 5e-3 for r in scaling["rows"])
    for name in ("width", "electric", "trotter-order", "quench", "trotter-budget",
                 "filter-design"):
        experiments.COMMANDS[name][0](verbose=False, **experiments.QUICK[name])
    produced = {p.name for p in (tmp_path / "results").iterdir()}
    for stem in ("width_extrapolation", "electric_scan_W2", "trotter_order", "quench_W2",
                 "trotter_budget_2x3", "filter_design_2x3"):
        assert f"{stem}.json" in produced and f"{stem}.png" in produced


# ---- lattices beyond one 62-bit label --------------------------------------------------------

def test_count_states_matches_enumeration():
    from sector import count_states
    for W, H, plates in [(3, 3, "free"), (2, 5, "fixed"), (4, 4, "fixed"), (3, 2, "free")]:
        lat = LatticeGrid(W, H)
        assert count_states(lat, plates) == GaugeSector(lat, plates=plates).dim
    assert count_states(LatticeGrid(8, 8), "fixed") == 56241588037      # why 8x8 ED is impossible


def test_multiword_labels_reproduce_single_word_results(monkeypatch):
    import sector
    lat = LatticeGrid(4, 4)
    ref = GaugeSector(lat, plates="fixed")
    ref_gs = ref.ground_state(J=1.0, rk=0.3)
    monkeypatch.setattr(sector, "WORD_BITS", 7)                       # force four words
    multi = GaugeSector(lat, plates="fixed")
    assert multi.n_words == 4
    assert sorted(multi.label(k) for k in range(multi.dim)) == sorted(ref.basis.tolist())
    gs = multi.ground_state(J=1.0, rk=0.3)
    assert abs(gs["energy"] - ref_gs["energy"]) < 1e-10 and abs(gs["gap"] - ref_gs["gap"]) < 1e-8
    assert multi.index_of(multi.label(17)) == 17
    with pytest.raises(ValueError):
        multi.basis


def test_sector_refuses_oversized_lattices():
    with pytest.raises(MemoryError):
        GaugeSector(LatticeGrid(8, 8), plates="fixed")
