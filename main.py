"""Top-level pipeline for the pure-gauge U(1) quantum link model.

Builds the gauge Hamiltonian on a 3x3 cylinder, checks gauge invariance,
compares structured vs generic Trotter circuit depth, reports the ground state
by exact diagonalisation, then evolves a far-from-equilibrium gauge-invariant
"string" state (Joshi et al.) both exactly and via structured-Trotter circuits
on Aer, and finally prepares the ground state with RQSVT on Aer.

Note: the strong-coupling vacuum |0...0> is a universal eigenstate
(H_box|0...0> = 0), so quenching from the confined ground state is inert. Real
dynamics need a far-from-equilibrium gauge-invariant state, found by
find_string_state.
"""

import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive: save figures, never block on a GUI window
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from qiskit import transpile
from qiskit_aer import AerSimulator

from lattice import LatticeGrid
from QuantumFields.gauge_field import GaussLaw, QuantumLinkModel
from circuit import DynamicalSimulation, StructuredTrotter, TrotterEvolution
from rqsvt import RQSVTGroundState

# --- Parameters -----------------------------------------------------------
WIDTH, HEIGHT = 3, 3
G = 2.0   # linear electric-field strength (moderate -> visible string fluctuation)
J = 2.0   # plaquette coupling
PENALTY = 200.0

# RQSVT ground-state demo runs on a small lattice by default so the (deep) LCU
# circuit finishes quickly on Aer; statevector caps near ~24 qubits, so 5x4 / 7x6
# need an MPS backend or hardware. RQSVT_G sits near the confinement crossover so
# the |0...0> guess has partial (filterable) overlap with the true ground state.
RQSVT_WIDTH, RQSVT_HEIGHT = 2, 3
RQSVT_G = 1.0

TOTAL_TIME = 20.0
N_STEPS = 400
DT = 0.05
SHOTS = 2048
TIME_POINTS = [0, 50, 100, 200, 300, 400]

BASIS_GATES = ['cx', 'rz', 'ry', 'rx', 'h', 'x', 'y', 'z', 's', 'sdg']


def check_gauge_invariance(lattice):
    """||[G_j, H]|| should be ~0 for every interior site."""
    print("\n=== Gauge invariance check ===")
    qlm = QuantumLinkModel(lattice, g=G, J=J)
    H = qlm.build_hamiltonian()
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
    qlm = QuantumLinkModel(lattice, g=G, J=J,
                           gauss_penalty=PENALTY)
    H_sparse = qlm.build_hamiltonian().to_matrix(sparse=True)
    eigenvalues, eigenvectors = eigsh(H_sparse, k=2, which='SA')
    gs_vector = eigenvectors[:, 0]

    print(f"Initial parameters: g={G}, J={J}")
    print(f"Ground state energy: {eigenvalues[0]:.6f}")
    print(f"First excited:       {eigenvalues[1]:.6f}")
    print(f"Gap:                 {eigenvalues[1] - eigenvalues[0]:.6f}")
    GaussLaw(lattice).verify(gs_vector, tol=1e-3)
    return eigenvalues, gs_vector


def find_string_state(lattice):
    """A far-from-equilibrium, gauge-invariant, H_box-active product state.

    Brute-forces the lowest-weight computational basis state that (i) satisfies
    Gauss's law at every interior site and (ii) has a plaquette in the
    (up,up,down,down) pattern so H_box acts on it (hence it is NOT an eigenstate
    and will evolve). Returns the statevector and the list of flipped links.
    """
    n = lattice.n_qubits
    if n > 20:
        raise ValueError(f"brute-force string search infeasible for {n} links")
    W, H = lattice.width, lattice.height
    interior = [(i, j) for j in range(1, H - 1) for i in range(W)]
    plaqs = lattice.get_plaquettes()                       # (bottom, right, top, left)

    def gauss_ok(down):
        sz = [(-0.5 if q in down else 0.5) for q in range(n)]
        for i, j in interior:
            g = (sz[lattice.get_horizontal_link_index(i, j)]
                 - sz[lattice.get_horizontal_link_index((i - 1) % W, j)]
                 + sz[lattice.get_vertical_link_index(i, j)]
                 - sz[lattice.get_vertical_link_index(i, j - 1)])
            if abs(g) > 1e-9:
                return False
        return True

    def hbox_active(down):
        for bo, r, t, l in plaqs:
            pat = tuple(int(q in down) for q in (bo, r, t, l))
            if pat in ((1, 1, 0, 0), (0, 0, 1, 1)):
                return True
        return False

    for weight in range(2, n + 1):
        for down in itertools.combinations(range(n), weight):
            down = set(down)
            if gauss_ok(down) and hbox_active(down):
                sv = np.zeros(2 ** n, dtype=complex)
                sv[sum(1 << q for q in down)] = 1.0
                return sv, sorted(down)
    raise RuntimeError("no gauge-invariant H_box-active string state found")


def run_string_dynamics(lattice, string_state, flipped):
    """Evolve a far-from-equilibrium string state under the static H(G, J)."""
    print(f"\n=== String dynamics (flipped links {flipped}) ===")
    H = QuantumLinkModel(lattice, g=G, J=J, gauss_penalty=PENALTY).build_hamiltonian()
    print(f"Parameters: g={G}, J={J}   (H_box-active string, not an eigenstate)")

    sim = DynamicalSimulation(
        H, lattice,
        g=G, J=J,
        total_time=TOTAL_TIME, n_steps=N_STEPS,
        initial_state=string_state,
        use_structured_trotter=True,
    )
    results = sim.run()

    backend = AerSimulator(method='statevector')
    results_aer = sim.run_on_backend(backend, shots=SHOTS, time_points=TIME_POINTS)
    return results, results_aer


def plot_observables(lattice, results):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(results['times'], results['energies'], 'b-', linewidth=2)
    axes[0].set_ylabel('Energy ⟨H⟩', fontsize=12)
    axes[0].set_title(
        f'Pure gauge string dynamics ({lattice.width}×{lattice.height}, '
        f'{lattice.n_qubits} qubits)\ng={G}, J={J}', fontsize=13)
    axes[0].grid(alpha=0.3)

    axes[1].plot(results['times'], results['gauss_violations'], 'r-', linewidth=2)
    axes[1].set_ylabel('Max Gauss violation', fontsize=12)
    axes[1].set_yscale('log')
    axes[1].grid(alpha=0.3)

    axes[2].plot(results['times'], results['plaquette_values'], 'g-', linewidth=2)
    axes[2].set_ylabel('⟨Plaquette⟩', fontsize=12)
    axes[2].set_xlabel('Time t', fontsize=12)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('pure_gauge_observables.png', dpi=150, bbox_inches='tight')
    print("\n[Saved: pure_gauge_observables.png]")
    plt.close(fig)


def plot_efield(lattice, results, results_aer):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    extent_y = [-0.5, lattice.n_qubits - 0.5]

    e_sv = np.array(results['e_field_profiles'])
    im1 = axes[0].imshow(e_sv.T, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5,
                         extent=[0, results['times'][-1], *extent_y], origin='lower')
    axes[0].set_title('Electric field (statevector)', fontsize=12)
    axes[0].set_ylabel('Link index', fontsize=11)
    axes[0].set_xlabel('Time t', fontsize=11)
    plt.colorbar(im1, ax=axes[0], label='⟨E⟩')

    aer_times = sorted(results_aer.keys())
    e_aer = np.array([[results_aer[t]['e_field'][q] for q in range(lattice.n_qubits)]
                      for t in aer_times])
    im2 = axes[1].imshow(e_aer.T, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5,
                         extent=[aer_times[0], aer_times[-1], *extent_y], origin='lower')
    axes[1].set_title(f'Electric field (Aer, {SHOTS} shots)', fontsize=12)
    axes[1].set_ylabel('Link index', fontsize=11)
    axes[1].set_xlabel('Time t', fontsize=11)
    plt.colorbar(im2, ax=axes[1], label='⟨E⟩')

    plt.tight_layout()
    plt.savefig('pure_gauge_efield.png', dpi=150, bbox_inches='tight')
    print("[Saved: pure_gauge_efield.png]")
    plt.close(fig)


def plot_scaling(results_aer):
    aer_times = sorted(results_aer.keys())
    depths = [results_aer[t]['depth'] for t in aer_times]
    two_q = [results_aer[t]['two_qubit_gates'] for t in aer_times]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(aer_times, depths, 'o-', color='coral', linewidth=2, markersize=8,
            label='Circuit depth')
    ax2 = ax.twinx()
    ax2.plot(aer_times, two_q, 's-', color='steelblue', linewidth=2, markersize=8,
             label='2-qubit gates')
    ax.set_xlabel('Simulation time t', fontsize=12)
    ax.set_ylabel('Circuit depth', color='coral', fontsize=12)
    ax2.set_ylabel('Two-qubit gates', color='steelblue', fontsize=12)
    ax.set_title('Circuit resource scaling (structured Trotter)', fontsize=13)
    ax.tick_params(axis='y', labelcolor='coral')
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig('pure_gauge_scaling.png', dpi=150, bbox_inches='tight')
    print("[Saved: pure_gauge_scaling.png]")
    plt.close(fig)
    return depths, two_q


def print_summary(lattice, eigenvalues, step_s, step_g, results, results_aer,
                  depths, two_q):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Lattice size:          {lattice.width}×{lattice.height} "
          f"({lattice.n_qubits} gauge qubits)")
    print(f"Hamiltonian:           g={G}, J={J}")
    print(f"Ground state energy:   {eigenvalues[0]:.6f}")
    print(f"Gap:                   {eigenvalues[1] - eigenvalues[0]:.6f}")
    print("\nCircuit comparison:")
    print(f"  Structured Trotter:  depth={step_s.depth()}, "
          f"CNOTs={step_s.count_ops().get('cx', 0)}")
    print(f"  Generic Trotter:     depth={step_g.depth()}, "
          f"CNOTs={step_g.count_ops().get('cx', 0)}")
    print(f"  Improvement:         {step_g.depth() / step_s.depth():.1f}× shallower")
    print("\nDynamics:")
    drift = (max(results['energies']) - min(results['energies'])) / 2
    print(f"  Energy drift:        ±{drift:.2e}")
    print(f"  Max Gauss violation: {max(results['gauss_violations']):.2e}")
    print(f"  Aer time points:     {len(results_aer)}")
    print(f"  Max circuit depth:   {max(depths)}")
    print(f"  Max 2-qubit gates:   {max(two_q)}")
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
    eigenvalues, gs_vector = find_ground_state(lattice)
    string_state, flipped = find_string_state(lattice)
    results, results_aer = run_string_dynamics(lattice, string_state, flipped)

    plot_observables(lattice, results)
    plot_efield(lattice, results, results_aer)
    depths, two_q = plot_scaling(results_aer)

    print_summary(lattice, eigenvalues, step_s, step_g, results, results_aer,
                  depths, two_q)

    run_rqsvt_ground_state()


if __name__ == "__main__":
    main()
