# Guide: how the program works and what to run for the thesis

## Part A. How the program works, step by step

The whole framework is one pipeline. Each step below names the module, what it
takes in, what it produces, and the one command or snippet that exercises it.

### 1. Geometry: `lattice.py`
`LatticeGrid(W, H)` is a cylinder: periodic in x (along the plates), open in y.
Rows 0 and H-1 are the plates, so d = H - 1. Every link is one qubit:
horizontal link (i, j) is qubit `i + W*j`, vertical link (i, j) is qubit
`W*H + i + W*j`, giving N_q = W(2H - 1) (15 for 3x3).
It also returns the plaquettes as (bottom, right, top, left) qubit tuples and
groups them into sublayers that share no link: 2 sublayers for even W, 3 for
odd W (the checkerboard is invalid on an odd-width cylinder).

    python -c "from lattice import LatticeGrid as L; l=L(3,3); l.debug(); l.visualize_ascii()"

### 2. Hamiltonian and Gauss's law: `QuantumFields/gauge_field.py`
`QuantumLinkModel(lattice, g, J, rk, field_pattern, plates)` builds Qiskit
`SparsePauliOp`s for

    H = -g sum s_l S^z_l + J sum (U_p + U_p^dag) + lam_RK sum (U_p + U_p^dag)^2

* `|0>` is S^z = +1/2; labels are little-endian (qubit q is character N_q-1-q).
* `field_pattern="uniform"` is Joshi's term; it commutes with every plaquette, so
  it does nothing in the pure-gauge model. Use `g = 0`, or `"staggered"`, or `rk`.
* `GaussLaw` gives the generators G_j. Plate sites have three links, so they carry
  a conserved static charge of +-1/2. `plates="fixed"` prescribes staggered
  charges (even W only); `plates="free"` leaves the plate rows unconstrained.
* `gauss_penalty` adds lambda * sum G_j^2. It is only needed for full-space
  cross-checks, never inside a quantum circuit.

### 3. Classical reference: `sector.py`
`GaugeSector(lattice, plates)` enumerates only the gauge-invariant basis states
(row by row: horizontal links free, vertical links fixed by Gauss's law) and
builds H as a sparse matrix in that basis. No penalty, no 2^N_q vectors.
3x3: 32768 -> 1792 -> 21 states. There is no qubit limit (labels are stored in
62-bit words); the limit is the sector dimension. `count_states(lattice, plates)`
returns it exactly and instantly, and `GaugeSector(..., max_dim=...)` refuses to
enumerate anything larger than `max_dim` (default 6 million; raise it on a big
machine). Fixed staggered plates: 6x6 = 7.8e5, 6x7 = 1.1e7, 8x5 = 1.5e6,
8x6 = 5.0e7, 8x7 = 1.7e9, 8x8 = 5.6e10 states.

    from lattice import LatticeGrid; from sector import GaugeSector
    s = GaugeSector(LatticeGrid(4, 4), plates="fixed")
    gs = s.ground_state(J=1.0)            # energy, gap, state, plate_charges
    s.electric_field(gs["state"]); s.plaquette_expectations(gs["state"])
    psi = s.embed(gs["state"])            # full Qiskit statevector (N_q <= 26)

With free plates the ground state is degenerate across plate-charge patterns;
`ground_state` picks the dominant pattern and returns the unique state and the
in-sector gap, which is the gap every gauge-invariant algorithm sees.

### 4. Time evolution circuits: `circuit.py`
* `append_plaquette_rotation(qc, theta, plaquette, ancilla=None)` is the exact
  gate exp(-i theta (U_p + U_p^dag)): CNOT fan-out, one uniformly controlled
  rotation, fan-in. 13 CNOTs (16 with the filter ancilla). `method="pauli"`
  gives the eight-ladder construction of draft 1 (17 / 40 CNOTs).
* `StructuredTrotter(lattice, g, J, dt, rk, order).build_step()` = half diagonal
  layer, one sublayer per plaquette colour, half diagonal layer.
* `trotter_error(...)` compares one step with exp(-iH dt) exactly.
* `circuit_resources(qc)` transpiles to {cx, rz, sx, x} and counts.

### 5. Ground-state preparation: `rqsvt.py`
`RQSVTGroundState(lattice, J, ..., filter_kind="gaussian"|"step")` does, in order:
1. sector ED for E0, the gap and the ground-state sector (the prior knowledge);
2. term decomposition at plaquette level, lambda_H = sum |a_k|;
3. time scale t0 = 0.9 pi / lambda_H, Fourier filter of degree D ~ lambda_H/gap;
4. GQSP angles (`gqsp_angles`) for that filter;
5. initial product state = configuration with the most flippable plaquettes.

Then
* `run("exact"|"trotter"|"qdrift")` emulates the circuit in the gauge sector
  (two ancilla branches stored separately) and returns energy, overlap, P(success);
* `estimate_energy(method, schedule)` adds Richardson extrapolation;
* `build_circuit(method, ...)` emits the real (N_q + 1)-qubit circuit: 2D signal
  operators exp(i (t0/2) Z_anc (x) H), each a sequence of ancilla-attached
  plaquette gates, interleaved with single-qubit rotations on the ancilla;
* `resources(method)` counts gates (4 signal operators transpiled, then scaled).

### 6. Casimir analysis: `casimir.py`
`run_casimir_scan(width, separations, J, rk)` -> E0(d) from sector ED ->
`analyse`: power-law fit, exponential fit, central-difference force, and the
model-free second difference D2(d) = E0(d+1) - 2E0(d) + E0(d-1) (negative means
attractive). `plot_scan` draws the three-panel figure.

### 7. Resources and baseline: `resources.py`
Trotter step table against lattice size, and the LCU block-encoding baseline
(PREPARE + SELECT built and transpiled; QSVT charged one call per signal operator).

### 8. Noise: `noise.py`
`noisy_filter_study(rq, error_rates)` runs Monte Carlo trajectories of the real
circuit with two-qubit depolarising noise after every CX, and reports the energy
with ancilla post-selection only and with additional Gauss-law post-selection.

### 9. Drivers
    python main.py                 # Secs. 2.3.1-2.3.6 end to end -> results/summary.json (~90 s)
    python experiments.py all      # rk-scan, filter-scaling, noise -> results/*.json, *.png (~2 min)
    python make_figures.py         # thesis figures from summary.json
    python main.py --gui           # explorer window
    python -m pytest -q            # 44 tests
    LEAN=lean proofs/check.sh      # Lean 4 proofs
All standard parameters live in `config.py`.

### 10. How correctness is layered
Lean (combinatorics and algebra: counts, gauge invariance, colouring, Pauli
expansions) -> pytest (numerics: sector vs full space, gate exactness, Trotter
orders, GQSP angles, circuit vs emulator, simulator vs Qiskit) -> `main.py`
cross-checks (eigsh spread, penalty scan, DMRG).

---

## Part B. Research protocol

Run these in order. Each has a question, a command, what to record, and what
would count as a surprise. R1 to R3 are already reproduced by `main.py`; R4 to R6
by `experiments.py`; R7 to R10 are open.

**R0. Reproduce on your machine (before anything else).**
`pytest`, `proofs/check.sh`, `python main.py`. Diff `results/summary.json` against
the shipped copy: energies must agree to 1e-9, gate counts exactly.
Then `python main.py --gui`: move a slider, switch to quench, press Play.

**R1. Is the model what the thesis says it is?** (Sec. 2.3.1, 3.1)
`main.py` section 1. Record: commutator norms (0), uniform-field commutator (0),
staggered (20), checkerboard valid = False on 3x3.
Thesis use: the two structural results, with the Lean theorems cited.

**R2. Is the classical reference trustworthy?** (Sec. 2.3.2, 2.4)
`main.py` sections 2 and 6. Record: sector vs full space (< 1e-12), eigsh spread,
DMRG relative difference (< 1e-8 required), penalty scan (E0 flat, 1-norm 36 ->
12012). Surprise: any disagreement above 1e-8.

**R3. What does the structured step cost?** (Sec. 3.2)
`main.py` section 3. Record Table 2. Optional extension: rerun
`trotter_step_table` with `rk=0.5` and with `field_pattern="staggered", g=1` to
show how the diagonal terms change depth.

**R4. Does the residual become longer-ranged towards the RK point?**
`python experiments.py rk-scan`. Shipped result (W = 4, d <= 7): only
lam_RK = 0.5 gives a determined decay length (xi = 3.0, exponential better by
4 orders of magnitude); elsewhere at least one fit sits at its bound.
lam_RK = 0.7 gives alpha = 3.10, but 0.8 gives 5.97, so do not quote 0.7 as a
verification of d^-3: it is not stable. To do: refine lam_RK in [0.3, 0.7] in
steps of 0.05, repeat at W = 2 (d <= 10) and W = 6 (d <= 4), and report xi(lam_RK)
only where neither fit is at a bound.

**R5. How does the quantum algorithm scale with d?**
`python experiments.py filter-scaling`. Shipped result (W = 4, d = 1..4, up to
37 qubits): two-qubit gates 1,856 -> 90,112 (about d^2.8, expected d^3); initial
overlap 0.33 -> 0.024, so repetitions grow about 2.4x per unit of d; Richardson
error stays below 6e-4. To do: add d = 5 (45 qubits, a few minutes), repeat with
`filter_kind="step"`, and try `initial_state="best_basis"` to bound what a better
initial state could gain. The decaying overlap is the main scalability limit
and deserves its own paragraph.

**R6. How much noise can the filter tolerate?**
`python experiments.py noise`. Shipped result: errors per circuit p * n_CX must be
well below 1; at p = 1e-3 the 2x3 circuit (2,048 CX) is unusable, and Gauss-law
post-selection cuts the energy error by up to 5x (typically 2x to 4x). Consequence for the framing:
the 3x3 circuit has 6,528 CX, so this is an early-fault-tolerant algorithm, not
a NISQ one. To do: increase trajectories for the 2x3 points (80 is noisy), add
p = 3e-5, and tabulate acceptance rates.

**R7. Trotter order inside the filter (open).**
`RQSVTGroundState._trotter_sequence` is first order. Compare n = 1, 2, 4 against a
second-order (palindromic) sequence at equal gate count. Expectation: second
order wins once n >= 2.

**R8. Filter design (open).**
Vary `sharpness` (2 to 4) and `truncation` (1e-2 to 1e-4) for both filters on 3x3;
plot overlap against signal operators. Goal: the cheapest filter that reaches
overlap 0.999.

**R9. Finite-width extrapolation (open).**
E0(d)/W at fixed d for W = 2, 4, 6, 8 (W = 8 reaches d = 4 with the default
`max_dim`, d = 5 with `max_dim=6e7` and about 20 GB; 8x8 is out of reach of
any exact method). Plot against 1/W^2.
This is the honest route to a 2+1D statement and also shows directly why W = 3
cannot test it.

**R10. Electric dynamics (open, optional).**
Staggered field g in {0.5, 1, 2} at lam_RK = 0: gap and D2 against d. A mass term
should shorten xi; this is the control experiment for R4.

### What to claim, and what not to
* Claim: exact 13-CNOT plaquette gate; size-independent depth; single-ancilla
  filter with measured 35x gate reduction against block-encoded QSVT for
  Trotter-driven evolution; qDRIFT not competitive for this Hamiltonian; uniform
  field conserved; checkerboard invalid on odd W; attractive residual; exponential
  decay favoured at lam_RK = 0.5.
* Do not claim: verification of d^-3; NISQ feasibility; any hardware timing; the
  block-encoded number as a compiled circuit (it is an estimate at equal degree).
