"""Top-level pipeline for the pure-gauge U(1) quantum link model.

Builds the gauge Hamiltonian on a 3x3 cylinder, checks gauge invariance,
compares structured vs generic Trotter circuit depth, finds the gauge-invariant
ground state by exact diagonalisation, quenches J and tracks the dynamics both
exactly (statevector) and via structured-Trotter circuits on Aer, and finally
prepares the ground state with RQSVT (structured Trotter + LCU filter) on Aer.
"""

import numpy as np
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
G = 6.0   # linear electric-field strength
J_INITIAL = 2.0
J_QUENCH = 0.5
PENALTY = 200.0

# RQSVT ground-state demo runs on a small lattice by default so the (deep) LCU
# circuit finishes quickly on Aer's statevector simulator. Scale up by changing
# these -- note statevector caps near ~24 qubits (system + log2(degree) ancillas),
# so 5x4 / 7x6 need an MPS backend or hardware.
RQSVT_WIDTH, RQSVT_HEIGHT = 2, 3

TOTAL_TIME = 20.0
N_STEPS = 400
DT = 0.05
SHOTS = 2048
TIME_POINTS = [0, 50, 100, 200, 300, 400]

BASIS_GATES = ['cx', 'rz', 'ry', 'rx', 'h', 'x', 'y', 'z', 's', 'sdg']


def check_gauge_invariance(lattice):
    """||[G_j, H]|| should be ~0 for every interior site."""
    print("\n=== Gauge invariance check ===")
    qlm = QuantumLinkModel(lattice, g=G, J=J_INITIAL)
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
    H = QuantumLinkModel(lattice, g=G, J=J_INITIAL).build_hamiltonian()

    structured = StructuredTrotter(lattice, g=G, J=J_INITIAL, dt=DT)
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
    qlm = QuantumLinkModel(lattice, g=G, J=J_INITIAL,
                           gauss_penalty=PENALTY)
    H_sparse = qlm.build_hamiltonian().to_matrix(sparse=True)
    eigenvalues, eigenvectors = eigsh(H_sparse, k=2, which='SA')
    gs_vector = eigenvectors[:, 0]

    print(f"Initial parameters: g={G}, J={J_INITIAL}")
    print(f"Ground state energy: {eigenvalues[0]:.6f}")
    print(f"First excited:       {eigenvalues[1]:.6f}")
    print(f"Gap:                 {eigenvalues[1] - eigenvalues[0]:.6f}")
    GaussLaw(lattice).verify(gs_vector, tol=1e-3)
    return eigenvalues, gs_vector


def run_quench(lattice, gs_vector):
    """Quench J: J_INITIAL -> J_QUENCH and evolve the ground state."""
    print(f"\n=== Quench dynamics (J: {J_INITIAL} → {J_QUENCH}) ===")
    H_evolve = QuantumLinkModel(lattice, g=G, J=J_QUENCH,
                                gauss_penalty=PENALTY).build_hamiltonian()

    initial_energy = np.conj(gs_vector) @ H_evolve.to_matrix(sparse=True) @ gs_vector
    print(f"Quench parameters: g={G}, J={J_QUENCH}")
    print(f"Terms in H_evolve: {len(H_evolve)}")
    print(f"Initial energy in quenched H: {initial_energy.real:.6f}")

    sim = DynamicalSimulation(
        H_evolve, lattice,
        g=G, J=J_QUENCH,
        total_time=TOTAL_TIME, n_steps=N_STEPS,
        initial_state=gs_vector,
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
        f'Pure gauge quench dynamics ({lattice.width}×{lattice.height}, '
        f'{lattice.n_qubits} qubits)\nJ: {J_INITIAL} → {J_QUENCH}', fontsize=13)
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
    plt.show()


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
    plt.show()


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
    plt.show()
    return depths, two_q


def print_summary(lattice, eigenvalues, step_s, step_g, results, results_aer,
                  depths, two_q):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Lattice size:          {lattice.width}×{lattice.height} "
          f"({lattice.n_qubits} gauge qubits)")
    print(f"Initial H:             g={G}, J={J_INITIAL}")
    print(f"Quench H:              g={G}, J={J_QUENCH}")
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

    Filters the strong-coupling vacuum |0...0> onto the ground state of the
    confining Hamiltonian (g=G, J=J_INITIAL). Reports the unfiltered baseline,
    an exact-evolution cross-check, and the structured-Trotter result with
    Richardson extrapolation, all against exact diagonalisation.
    """
    print("\n" + "=" * 60)
    print(f"RQSVT GROUND-STATE PREPARATION ({width}x{height})")
    print("=" * 60)
    lattice = LatticeGrid(width=width, height=height)
    rq = RQSVTGroundState(lattice, g=G, J=J_INITIAL)

    e_guess, ov_guess = rq.guess_energy()
    print(f"Filter degree:        {rq.degree}  "
          f"({rq.n_anc} ancillas, {lattice.n_qubits + rq.n_anc} qubits total)")
    print(f"Exact ground energy:  E0 = {rq.E0:.6f}   gap Delta = {rq.E1 - rq.E0:.4f}")
    print(f"Guess |0...0>:        E  = {e_guess:.6f}   overlap = {ov_guess:.4f}")

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
    results, results_aer = run_quench(lattice, gs_vector)

    plot_observables(lattice, results)
    plot_efield(lattice, results, results_aer)
    depths, two_q = plot_scaling(results_aer)

    print_summary(lattice, eigenvalues, step_s, step_g, results, results_aer,
                  depths, two_q)

    run_rqsvt_ground_state()


if __name__ == "__main__":
    main()
