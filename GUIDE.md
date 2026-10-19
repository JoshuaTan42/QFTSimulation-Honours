# Guide: how the program works and what to run for the thesis

Part A walks through the program module by module, in the order the data
flows. Part B is the data-generation protocol: every study, its command, what
to record and what would count as a surprise. Part C maps modules to thesis
sections and lists what the results do and do not support.

## Part A. How the program works

### 1. Geometry: `lattice.py`
`LatticeGrid(W, H)` is the Casimir geometry, a cylinder: periodic in x (along
the plates), open in y. Rows 0 and H-1 are the plates, so d = H - 1. Every link
is one qubit: horizontal link (i, j) is qubit `i + W*j`, vertical link (i, j) is
qubit `W*H + i + W*j`, so N_q = W(2H - 1) (15 for 3x3). Plaquettes are
(bottom, right, top, left) qubit tuples, grouped into sublayers that share no
link: 2 for even W, 3 for odd W (the checkerboard is invalid on an odd-width
cylinder; proofs/Colouring.lean).

`OpenLattice(W, H)` is the geometry of Joshi et al.: W x H matter sites, open
boundaries, links first and then one matter qubit per site (4x3: 29 qubits,
5x4: 51). Its `sublayers()` are four groups of commuting hopping and plaquette
terms.

    python -c "from lattice import LatticeGrid as L; l=L(3,3); l.debug(); l.visualize_ascii()"

### 2. Hamiltonians: `QuantumFields/`
`gauge_field.QuantumLinkModel(lattice, g, J, rk, field_pattern, plates)` builds
Qiskit `SparsePauliOp`s for the pure-gauge model

    H = -g sum s_l S^z_l + J sum (U_p + U_p^dag) + lam_RK sum (U_p + U_p^dag)^2

* `field_pattern="uniform"` is the term of Joshi et al.; without matter it
  commutes with every plaquette and only selects a flux sector. Use g = 0,
  `"staggered"`, or `rk` for non-trivial electric dynamics.
* `GaussLaw` gives the generators G_j. Plate sites have three links and carry a
  static charge of +-1/2. `plates="fixed"` prescribes staggered charges (even W
  only); `plates="free"` leaves the plate rows unconstrained.
  `default_plates(W)` picks fixed for even and free for odd W.
* `gauss_penalty` adds lambda sum G_j^2, needed only for full-space
  cross-checks, never in a circuit.

`matter_field.ScalarQEDModel(lattice, kappa, m, g, J)` is Eq. 1 of Joshi et al.:
hardcore-boson hopping phi^dag S^+ phi, staggered mass, linear electric term,
plaquette. Missing boundary links are fixed at S^z = +1/2, so the all-|0> link
configuration is the electric vacuum, and `string_state_bits` gives their
initial state (a staircase string between static charges at opposite corners).

### 3. Classical reference: `sector.py`, `matter_sector.py`
`GaugeSector(lattice, plates)` enumerates only the gauge-invariant basis states
(row by row: horizontal links free, vertical links fixed by Gauss's law) and
builds H as a sparse matrix in that basis. 3x3: 32768 -> 1792 -> 21 states.
Labels are stored in 62-bit words, so there is no qubit limit; the limit is the
sector dimension, which `count_states(lattice, plates)` returns exactly and
instantly. `max_dim` (default 6 million) guards against enumerating anything
larger. Fixed staggered plates: 6x6 = 7.8e5, 6x7 = 1.1e7, 8x5 = 1.5e6,
8x6 = 5.0e7, 8x8 = 5.6e10 states.

    from lattice import LatticeGrid; from sector import GaugeSector
    s = GaugeSector(LatticeGrid(4, 4), plates="fixed")
    gs = s.ground_state(J=1.0)            # energy, gap, state, support, plate_charges
    s.electric_field(gs["state"]); s.plaquette_expectations(gs["state"])

With free plates the ground state is degenerate across plate-charge patterns;
`ground_state` returns the unique state of the dominant pattern (`support`) and
the in-sector gap, the gap any gauge-invariant algorithm sees.
`most_flippable(support)` gives the product state used to start the filter and
the quenches.

`MatterSector(OpenLattice(W, H))` does the same for the matter-coupled model.
Gauss's law fixes every occupation once the links are known, so the 51-qubit
5x4 experiment of Joshi et al. lives in a sector of 636,487 states and is
solved exactly in minutes.

### 4. Circuits: `circuit.py`
* `append_two_level_rotation(qc, theta, qubits, a, b)` is the exact gate
  exp(-i theta (|a><b| + |b><a|)): CNOT fan-out, one uniformly controlled
  rotation, fan-in. The plaquette gate (13 CNOTs; 16 with the filter ancilla)
  and the hopping gate (7 CNOTs) are both instances. `method="pauli"` gives the
  eight-ladder plaquette construction (17 / 40 CNOTs) for comparison.
* `StructuredTrotter(lattice, g, J, dt, rk, order)`: half diagonal layer, one
  sublayer per plaquette colour (palindromic for order 2), half diagonal layer.
* `MatterTrotter(open_lattice, kappa, m, g, J, dt, order)`: the same for the
  matter-coupled model with the four sublayers of Joshi et al.
* `trotter_error(...)` compares one step with exp(-iH dt) exactly;
  `circuit_resources(qc)` transpiles to {cx, rz, sx, x} and counts.

### 5. Ground-state filter: `rqsvt.py`
`RQSVTGroundState(lattice, J, ..., filter_kind="gaussian"|"step")`:
1. sector ED for E0, the gap and the ground-state sector (the prior knowledge);
2. term decomposition at plaquette level, lambda_H = sum |a_k|;
3. time scale t0 = 0.9 pi / lambda_H and a Fourier filter of degree D ~ lambda_H/gap;
4. GQSP angles for that filter;
5. initial product state: the most-flippable configuration (or `"best_basis"`).

Then `run(method, n_steps, order)` emulates the circuit in the gauge sector,
`estimate_energy(method, schedule, order)` adds Richardson extrapolation,
`build_circuit(...)` emits the real (N_q + 1)-qubit circuit and `resources(...)`
counts its gates. Signal operators are exp(i (t0/2) Z_anc (x) H) built from
ancilla-attached plaquette gates, interleaved with ancilla rotations.

### 6. Casimir analysis: `casimir.py`
`run_casimir_scan(width, separations, J, rk)` -> E0(d) from sector ED ->
`analyse`: power-law and exponential fits, central-difference force, and the
model-free second difference D2(d) = E0(d+1) - 2E0(d) + E0(d-1), which removes
the bulk a + b d exactly (negative means attractive).

### 7. Real-time dynamics: `dynamics.py`
`SectorTrotter` runs a structured Trotter circuit inside the gauge sector: each
plaquette or hopping term is a set of two-level flips, applied in closed form.
It reproduces the Qiskit circuit output to 1e-15 (tests) at the cost of the
sector dimension instead of 2^N_q. `quench(W, d, ...)` evolves the
most-flippable state exactly and records the return probability, flippability
and energy density per plaquette row, the electric field per row, the
entanglement entropy across the midplane between the plates, and the fidelity
of Trotterised evolution for chosen (dt, order).

### 8. Hardware: `hardware.py`
`quench_experiment` builds the quench circuits (n = 0, 1, 2, ... Trotter steps)
plus one extra circuit per plaquette colour that measures U_p + U_p^dag through
the plaquette fan-out. `filter_experiment` does the same for a low-degree
filter on 2x2. Shots violating Gauss's law are discarded (error detection);
results are reported with and without that post-selection, next to the ideal
values from the sector emulation. Backends: `ideal`, `fake:<device>` (Aer with
the noise model of an IBM snapshot, e.g. `fake:kingston`), `ibm:<device>`.
`--dry-run` transpiles and reports gates, depth and an estimated circuit
fidelity from the device error rates without running anything.

### 9. Resources, noise and tensor networks
`resources.py`: Trotter step table against lattice size and the LCU
block-encoding baseline (PREPARE and SELECT built and transpiled; QSVT charged
one call per signal operator). `noise.py`: Monte Carlo trajectories with
two-qubit depolarising noise after every CX; energies with ancilla
post-selection only and with Gauss-law post-selection, with standard errors.
`tensor_network.py`: DMRG on the penalised Hamiltonian (independent of the
sector code) and MPS compressibility of exact ground states.

### 10. Benchmark against Joshi et al.: `joshi_benchmark.py`
Their model, lattices and parameters (kappa = 1, m = 3, g = 6) with this
package's circuits and references: per-step gates against their Table S1,
exact string-breaking dynamics on 4x3 with emulated Trotter circuits, and pair
creation away from the string on 5x4 with and without the plaquette term.

### 11. Commands
    python main.py                 # core pipeline -> results/summary.json
    python make_figures.py         # figures from summary.json
    python experiments.py <study>  # Part B; add -h for the options of a study
    python hardware.py quench|filter --backend ... [--dry-run]
    python joshi_benchmark.py
    python main.py --gui           # explorer window
    python -m pytest -q            # test-suite
    LEAN=lean proofs/check.sh      # Lean 4 proofs
Every result file stores the Python and package versions (`config.environment()`).

### 12. How correctness is layered
Lean (combinatorics and algebra: counts, gauge invariance, colouring, Pauli
expansions, extraction identities) -> pytest (sector against full space for
both models, gate exactness, Trotter orders, GQSP angles, circuit against
emulator, sector Trotter against Qiskit, simulator against Qiskit,
plaquette-basis measurement, ideal hardware run against emulation) ->
`main.py` cross-checks (eigensolver spread, penalty scan, DMRG).

---

## Part B. Data-generation protocol

Run R0 first. Everything after it writes `results/<study>[_tag].json` and a
`.png`; the tag records the main options, so different runs do not overwrite
each other.

**R0. Reproduce on your machine.** `python -m pytest -q`, `proofs/check.sh`,
`python main.py`, `python make_figures.py`, then `python main.py --gui` (move a
slider, switch to quench, press Play, Save PNG). Note the versions in
`results/summary.json["environment"]`; if your Qiskit differs from the shipped
one, two-qubit counts of the structured steps must agree exactly, while
single-qubit counts and the generic baseline may change.

**R1-R3. Model, reference, Trotter step.** `main.py` sections 1-3 and 6.
Record: commutators (0), uniform-field commutator (0), checkerboard invalid on
3x3, sector against full space (< 1e-12), DMRG (< 1e-8), penalty scan, Trotter
orders (dt^2 and dt^3 per step), the step table.

**R4. Range of the Casimir residual against lam_RK.**
`python experiments.py rk-scan --rks 0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7`,
then `--width 2 --dmax 10` and `--width 6 --dmax 5`. Report xi(lam_RK) only where
neither fit is at its bound. Surprise: a determined power law that is stable
between neighbouring lam_RK.

**R5. Filter cost against d.** `python experiments.py filter-scaling --dmax 5`,
then `--filter step`, `--initial best_basis`, `--order 2`. Record gates,
signal operators, initial overlap and Richardson error per d. The decaying
initial overlap is the main scalability limit.

**R6. Noise tolerance.** `python experiments.py noise --trajectories 1000`.
Record the energy error with and without Gauss-law post-selection and both
acceptance rates. Expected: p * n_CX must be well below 1.

**R7. Trotter order inside the filter.** `python experiments.py trotter-order`.
Compare errors at equal two-qubit gate count. On 2x3, second order with
Richardson reaches about 1e-6 with fewer gates than first order reaches 1e-4.

**R8. Filter design.** `python experiments.py filter-design`. The cheapest
filter that reaches overlap 0.999, and its Trotter error at n = 1.

**R9. Finite-width extrapolation.** `python experiments.py width`
(W = 2, 4, 6, 8; add `--max-dim 60000000` on a large-memory machine to reach
8x6). E0/W and D2/W against 1/W^2; this is the route to a 2+1D statement.

**R10. Electric control experiment.** `python experiments.py electric`. A
staggered field gaps the photon and should shorten the residual's range.

**R11. Quench between the plates.** `python experiments.py quench --dmax 6`
(and `--width 3`, `--rk 0.5`). Record the echo decay time, late-time echo,
maximum midplane entanglement and the late-time energy profile per d. This is
the regime where a quantum simulator has an advantage over classical
sampling methods (sign problem in real time) and where entanglement growth
limits tensor networks.

**R12. Trotter budget.** `python experiments.py trotter-budget` (3x3), and
`--width 4 --d 3`. The longest time with fidelity >= 0.99 and the two-qubit
gates to get there, for each (dt, order). Combine with R6 for the hardware
feasibility statement.

**R13. Benchmark against Joshi et al.** `python experiments.py joshi`. Record
the per-step table against their Table S1 and the J = 0 / J = 2 comparison of
pairs created off the string.

**R14. Hardware.** First
`python hardware.py quench --backend fake:kingston --lattice 3x3 --dry-run`,
then the same without `--dry-run` (noisy simulation), then on hardware with
`--backend ibm:ibm_kingston` after `QiskitRuntimeService.save_account(...)`.
Start with `--steps 0 1 2`; each step adds about 165 to 200 two-qubit gates on the
heavy-hex layout. The filter: `python hardware.py filter --lattice 2x2 --degree 2`.
Record raw and Gauss-selected observables against the ideal column.

---

## Part C. Thesis map and claims

| thesis section (Draft 2 numbering) | module | command |
|---|---|---|
| 2.2 System, Gauss's law, plates | `lattice.py`, `QuantumFields/gauge_field.py` | `main.py` sec. 1 |
| 2.3.2 Classical reference | `sector.py`, `tensor_network.py` | `main.py` sec. 2, 6 |
| 2.3.3 Structured Trotter | `circuit.py` | `main.py` sec. 3 |
| 2.3.4 Ground-state filter | `rqsvt.py` | `main.py` sec. 4 |
| 2.3.5 Casimir extraction | `casimir.py` | `main.py` sec. 5, R4, R9, R10 |
| 2.3.6 Resources | `resources.py` | `main.py` sec. 3-4, R5, R7 |
| 2.5 Formal verification | `proofs/` | `proofs/check.sh` |
| 3.5 Interactive framework | `visualize.py` | `main.py --gui` |
| 3.6 Further experiments | `experiments.py`, `noise.py` | R4-R12 |
| validation against prior work | `joshi_benchmark.py`, `QuantumFields/matter_field.py`, `matter_sector.py` | R13 |
| real-time dynamics, hardware | `dynamics.py`, `hardware.py` | R11, R12, R14 |

Supported by the results: exact 13-CNOT plaquette gate; size-independent depth
of the structured step; single-ancilla filter with a large gate reduction
against block-encoded QSVT at equal filter degree (the baseline is an estimate,
not a compiled QSVT circuit); qDRIFT not competitive for this Hamiltonian;
uniform field conserved; checkerboard invalid on odd W; attractive residual;
exponential decay favoured at lam_RK = 0.5; same two-qubit count per step as
Joshi et al. for their J = 0 lattice and fewer for J = 2, at larger depth.

Not supported: verification of the continuum d^-3 law; NISQ feasibility of the
full 3x3 filter; any hardware timing; a classical-hardness claim for the 5x4
experiment of Joshi et al. (its gauge sector is solved exactly here).
