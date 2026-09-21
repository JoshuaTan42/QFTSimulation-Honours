# Towards Optimized Casimir Simulations

Pure-gauge U(1) quantum link model on a W x H cylinder: structured Trotter
evolution, single-ancilla randomised QSVT ground-state preparation, and
Casimir-energy extraction, with an exact gauge-sector reference and an
interactive explorer.

    pip install -r requirements.txt
    python main.py              # full pipeline, ~90 s, writes results/summary.json
    python main.py --quick      # small lattices only
    python main.py --gui        # pop-up explorer (same as: python visualize.py)
    python make_figures.py      # thesis figures from results/summary.json
    python -m pytest -q         # 38 tests, ~40 s
    LEAN=lean proofs/check.sh   # Lean 4 proofs (plain toolchain, no Mathlib)

| module | role |
|---|---|
| `lattice.py` | geometry, link indexing, proper plaquette colouring (2 sublayers for even W, 3 for odd W) |
| `QuantumFields/gauge_field.py` | Hamiltonian terms (electric uniform/staggered, plaquette, RK), Gauss law with free or fixed plate charges |
| `sector.py` | exact diagonalisation in the gauge-invariant sector (up to 62 link qubits), observables, dynamics |
| `circuit.py` | exact plaquette gate (13 CNOTs; 16 with the RQSVT ancilla), structured Trotter step, resource counters |
| `rqsvt.py` | GQSP angle finding, Fourier filters, single-ancilla filter driven by exact / Trotter / qDRIFT evolution, Richardson extrapolation, Qiskit circuit |
| `resources.py` | measured gate counts and the LCU block-encoding baseline |
| `casimir.py` | E0(d), bulk subtraction, power-law and exponential fits, model-free second difference |
| `tensor_compression.py` | DMRG (quimb, compressed MPO) and MPS compressibility |
| `visualize.py` | interactive explorer with sliders, radio buttons and a real-time quench |
| `proofs/` | Lean 4: link counts, gauge invariance, colouring, Pauli expansions, extraction identities |

Conventions: qubit |0> is S^z = +1/2; basis-state integers are Qiskit statevector
indices; Pauli labels are little-endian. Performance notes: all classical
numerics run in the gauge sector (3x3: 21 states instead of 32768), operators
are assembled once and cached, the RQSVT emulator uses precomputed dense
rotations in small sectors, and the explorer updates persistent artists rather
than redrawing.
