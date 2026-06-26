"""Top-level diagnostics for the pure-gauge U(1) quantum link model.

On a single lattice: checks gauge invariance, benchmarks the structured vs
generic Trotter circuit depth (resource analysis, thesis §2.3.6), reports the
ground state by exact diagonalisation, and prepares the ground state with RQSVT
(structured Trotter + LCU filter) on Aer.

The Casimir-force computation itself -- ground-state energy vs plate separation,
bulk subtraction and the F ∝ d^-3 fit -- lives in casimir.py (thesis §2.3.5).
"""

from scipy.sparse.linalg import eigsh
from qiskit import transpile
from qiskit_aer import AerSimulator

from lattice import LatticeGrid
from QuantumFields.gauge_field import GaussLaw, QuantumLinkModel
from circuit import StructuredTrotter, TrotterEvolution
from rqsvt import RQSVTGroundState

# --- Parameters -----------------------------------------------------------
WIDTH, HEIGHT = 3, 3
G = 2.0   # linear electric-field strength
J = 2.0   # plaquette coupling
PENALTY = 200.0
DT = 0.05
BASIS_GATES = ['cx', 'rz', 'ry', 'rx', 'h', 'x', 'y', 'z', 's', 'sdg']

# RQSVT ground-state demo runs on a small lattice by default so the (deep) LCU
# circuit finishes quickly on Aer; statevector caps near ~24 qubits, so 5x4 / 7x6
# need an MPS backend or hardware. RQSVT_G sits near the confinement crossover so
# the |0...0> guess has partial (filterable) overlap with the true ground state.
RQSVT_WIDTH, RQSVT_HEIGHT = 2, 3
RQSVT_G = 1.0


def check_gauge_invariance(lattice):
    """||[G_j, H]|| should be ~0 for every interior site."""
    print("\n=== Gauge invariance check ===")
    H = QuantumLinkModel(lattice, g=G, J=J).build_hamiltonian()
    for idx, G_op in enumerate(GaussLaw(lattice).build_gauss_operators()):
        if G_op is None:
            continue
        i, j = lattice.site_to_coords(idx)
        comm = (G_op @ H - H @ G_op).simplify()
        comm_norm = sum(abs(c) for c in comm.coeffs)
        print(f"  ||[G({i},{j}), H]|| = {comm_norm:.2e}")


def compare_circuit_depth(lattice):
    """Structured (constant-depth) vs generic second-order Trotter step."""
    print("\n=== Circuit depth comparison ===")
    H = QuantumLinkModel(lattice, g=G, J=J).build_hamiltonian()

    structured = StructuredTrotter(lattice, g=G, J=J, dt=DT)
    step_s = transpile(structured.build_step(), basis_gates=BASIS_GATES,
                       optimization_level=3)
    print(f"Structured: depth={step_s.depth()}, CNOTs={step_s.count_ops().get('cx', 0)}")

    generic = TrotterEvolution(H, lattice.n_qubits, total_time=DT, n_steps=1)
    step_g = transpile(generic.build_circuit(), basis_gates=BASIS_GATES,
                       optimization_level=3)
    print(f"Generic:    depth={step_g.depth()}, CNOTs={step_g.count_ops().get('cx', 0)}")
    return step_s, step_g


def find_ground_state(lattice):
    """Lowest two eigenstates of the penalised Hamiltonian via sparse ED."""
    print("\n=== Finding gauge-invariant ground state ===")
    qlm = QuantumLinkModel(lattice, g=G, J=J, gauss_penalty=PENALTY)
    H_sparse = qlm.build_hamiltonian().to_matrix(sparse=True)
    eigenvalues, eigenvectors = eigsh(H_sparse, k=2, which='SA')
    gs_vector = eigenvectors[:, 0]

    print(f"Parameters: g={G}, J={J}")
    print(f"Ground state energy: {eigenvalues[0]:.6f}")
    print(f"First excited:       {eigenvalues[1]:.6f}")
    print(f"Gap:                 {eigenvalues[1] - eigenvalues[0]:.6f}")
    GaussLaw(lattice).verify(gs_vector, tol=1e-3)
    return eigenvalues


def print_summary(lattice, eigenvalues, step_s, step_g):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Lattice size:          {lattice.width}×{lattice.height} "
          f"({lattice.n_qubits} gauge qubits)")
    print(f"Hamiltonian:           g={G}, J={J}")
    print(f"Ground state energy:   {eigenvalues[0]:.6f}")
    print(f"Gap:                   {eigenvalues[1] - eigenvalues[0]:.6f}")
    print("\nCircuit comparison (per Trotter step):")
    print(f"  Structured Trotter:  depth={step_s.depth()}, "
          f"CNOTs={step_s.count_ops().get('cx', 0)}")
    print(f"  Generic Trotter:     depth={step_g.depth()}, "
          f"CNOTs={step_g.count_ops().get('cx', 0)}")
    print(f"  Improvement:         {step_g.depth() / step_s.depth():.1f}× shallower")
    print("=" * 60)


def run_rqsvt_ground_state(width=RQSVT_WIDTH, height=RQSVT_HEIGHT,
                           n_trotter_list=(1, 2)):
    """RQSVT ground-state preparation: structured Trotter + LCU filter on Aer.

    Filters the strong-coupling vacuum |0...0> onto the ground state at (RQSVT_G,
    J). RQSVT only has work to do when the guess has partial overlap: if the
    reported overlap is ~1 the regime is too confined (guess already the ground
    state); if ~0 it is too magnetic (guess orthogonal). Tune RQSVT_G accordingly.
    """
    print("\n" + "=" * 60)
    print(f"RQSVT GROUND-STATE PREPARATION ({width}x{height}, g={RQSVT_G})")
    print("=" * 60)
    lattice = LatticeGrid(width=width, height=height)
    rq = RQSVTGroundState(lattice, g=RQSVT_G, J=J)

    e_guess, ov_guess = rq.guess_energy()
    print(f"Filter degree:        {rq.degree}  "
          f"({rq.n_anc} ancillas, {lattice.n_qubits + rq.n_anc} qubits total)")
    print(f"Exact ground energy:  E0 = {rq.E0:.6f}   gap Delta = {rq.E1 - rq.E0:.4f}")
    print(f"Guess |0...0>:        E  = {e_guess:.6f}   overlap = {ov_guess:.4f}")
    if ov_guess > 0.99:
        print("  [!] guess overlap ~1: lower RQSVT_G for a non-trivial filter demo")
    elif ov_guess < 0.05:
        print("  [!] guess overlap ~0: raise RQSVT_G so the guess overlaps the ground state")

    aer = AerSimulator(method='statevector')

    res = rq.prepare(aer, backend='exact')                 # exact-U cross-check
    print(f"RQSVT (exact U):      E  = {res['energy']:.6f}   "
          f"overlap = {res['overlap']:.4f}   P(success) = {res['success_prob']:.3f}")

    est = rq.estimate_energy(aer, backend='trotter', n_trotter_list=n_trotter_list)
    for n, r in est['runs']:
        print(f"RQSVT (trotter n={n}):  E  = {r['energy']:.6f}   overlap = {r['overlap']:.4f}")
    print(f"RQSVT (Richardson):   E  = {est['energy_richardson']:.6f}   "
          f"(target E0 = {rq.E0:.6f})")
    print("=" * 60)


def main():
    lattice = LatticeGrid(width=WIDTH, height=HEIGHT)
    lattice.debug()
    lattice.visualize_ascii()

    check_gauge_invariance(lattice)
    step_s, step_g = compare_circuit_depth(lattice)
    eigenvalues = find_ground_state(lattice)
    print_summary(lattice, eigenvalues, step_s, step_g)

    run_rqsvt_ground_state()

    print("\n[For the Casimir force F(d) ∝ d^-3, run:  python casimir.py]")


if __name__ == "__main__":
    main()
