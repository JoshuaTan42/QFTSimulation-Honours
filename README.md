# Towards Optimized Casimir Simulations

Quantum simulation of the Casimir effect in the 2+1D U(1) quantum link model
(pure gauge, W x H cylinder, plates on the boundary rows). The package combines
a structured Trotter decomposition (Joshi et al. 2026) with a single-ancilla
spectral filter for ground-state preparation (Wang et al. 2025), and provides an
exact gauge-sector reference, real-time quench dynamics between the plates,
hardware-ready experiments, a benchmark on the matter-coupled model of Joshi et
al., an interactive explorer and Lean 4 proofs of the structural results.

Honours thesis, University of Technology Sydney. Supervisor: Dr. Christopher Ferrie.

## Install and run

    pip install -r requirements.txt

    python main.py                 # core pipeline -> results/summary.json  (a few minutes)
    python main.py --quick         # small lattices only
    python make_figures.py         # figures from results/summary.json
    python experiments.py all      # every research study at default size
    python experiments.py all --quick
    python experiments.py <study> -h          # options of one study
    python hardware.py quench --backend fake:kingston --dry-run
    python joshi_benchmark.py      # benchmark against Joshi et al.
    python main.py --gui           # interactive explorer (same as: python visualize.py)
    python -m pytest -q            # test-suite
    LEAN=lean proofs/check.sh      # Lean 4 proofs (plain toolchain, no Mathlib)

`GUIDE.md` explains how the program works step by step and lists the
experiments to run for the thesis, with the command for each.

## Modules

| module | role |
|---|---|
| `lattice.py` | `LatticeGrid` (cylinder: link indexing, plaquettes, proper colouring with 2 sublayers for even W and 3 for odd W) and `OpenLattice` (open matter lattice of Joshi et al.) |
| `QuantumFields/gauge_field.py` | pure-gauge Hamiltonian (uniform or staggered electric term, plaquette, Rokhsar-Kivelson) and Gauss's law with free or fixed plate charges |
| `QuantumFields/matter_field.py` | matter-coupled model of Joshi et al.: hardcore bosons, gauge-matter hopping, staggered mass, static charges, string state |
| `sector.py` | exact diagonalisation in the gauge-invariant sector of the cylinder (no qubit limit; `count_states` gives the dimension in advance) |
| `matter_sector.py` | the same for the matter-coupled model (4x3: 1,826 states, 5x4: 636,487) |
| `circuit.py` | exact two-level rotations (plaquette gate: 13 CNOTs, 16 with the filter ancilla; hopping gate: 7), structured Trotter steps for both models, resource counters |
| `rqsvt.py` | GQSP angles, Fourier filters, single-ancilla filter driven by exact, Trotter (first or second order) or qDRIFT evolution, Richardson extrapolation, the Qiskit circuit |
| `resources.py` | measured gate counts and the LCU block-encoding baseline |
| `casimir.py` | E0(d), bulk subtraction, power-law and exponential fits, model-free second difference |
| `dynamics.py` | quench dynamics: sector emulation of the Trotter circuits, return probability, row profiles, midplane entanglement, Trotter budget |
| `hardware.py` | quench and filter experiments for an ideal simulator, an IBM device snapshot (`fake:kingston`) or IBM hardware, with Gauss-law post-selection |
| `noise.py` | trajectory simulator for two-qubit depolarising noise |
| `tensor_network.py` | DMRG (quimb) cross-check and MPS compressibility |
| `experiments.py` | the research studies, each with command-line options |
| `joshi_benchmark.py` | resources and string dynamics against Joshi et al. (2026) |
| `visualize.py` | interactive explorer with sliders, radio buttons and a real-time quench |
| `config.py` | standard parameters; `environment()` records package versions |
| `proofs/` | Lean 4: link counts, gauge invariance, colouring, Pauli expansions, extraction identities |
| `tests/` | `test_framework.py` (core), `test_matter.py`, `test_dynamics.py` |

## Conventions

* Qubit |0> is S^z = +1/2; for matter qubits |1> is an occupied site.
* Basis-state integers are Qiskit statevector indices (bit q = qubit q), and
  Pauli labels are little-endian (qubit q is character N_q - 1 - q).
* All classical numerics run in the gauge-invariant sector, never in 2^N_q.
* Every gate count is measured on circuits built here, transpiled to
  {cx, rz, sx, x} at optimisation level 3 (or to an explicit basis). Counts
  depend on the Qiskit version, which is stored in every result file; compare
  results only between runs with the same versions.
