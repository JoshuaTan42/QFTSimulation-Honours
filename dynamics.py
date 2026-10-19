"""Real-time quench dynamics between the plates, exact and Trotterised.

A quench prepares a simple product state inside the gauge sector and lets it
evolve under the quantum link Hamiltonian. This is the regime where a quantum
simulator has a genuine advantage: Monte Carlo cannot follow real-time
evolution (sign problem), and tensor networks are limited by the growth of
entanglement after the quench. The routines here produce the classical
reference for such an experiment and the error budget of its circuits.

SectorTrotter
    Emulates a structured Trotter circuit inside the gauge-invariant sector.
    Every off-diagonal term (plaquette flip, gauge-matter hop) is a sum of
    disjoint two-level flips h, so exp(-i theta h) acts in closed form,
        psi[src] -> cos(theta) psi[src] - i sin(theta) psi[tgt],
    and the diagonal terms are phases. The sequence of terms is the one of
    circuit.StructuredTrotter / circuit.MatterTrotter, so the emulated state
    equals the circuit output exactly (tests check this against Qiskit), but
    at the cost of the sector dimension instead of 2^N_q.

quench
    Exact evolution (Krylov expm_multiply) of a product state, with the
    observables of a plate experiment: return probability (Loschmidt echo),
    flippability and energy density per plaquette row, electric field per row,
    entanglement entropy across the midplane between the plates, and the
    fidelity of Trotterised evolution for chosen time steps and orders.

    With lam_RK = 0 and g = 0 the energy density <J (U_p + U_p^dag)> of every
    plaquette stays exactly zero after a quench from a product state: the
    diagonal operator prod Z over the horizontal links of even rows
    anticommutes with every plaquette term (chiral symmetry), and together
    with time reversal (H and the initial state are real) this forces
    <h_p>(t) = -<h_p>(t). The flippability is the observable that evolves.

quench_vs_separation, trotter_budget
    Drivers used by experiments.py.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import expm_multiply

from circuit import StructuredTrotter, circuit_resources, palindromic_schedule
from lattice import LatticeGrid
from QuantumFields.gauge_field import default_plates
from sector import GaugeSector

ENTROPY_MAX_SIDE = 4096         # largest reduced dimension used for the entanglement SVD


# ---------------------------------------------------------------------------
# Trotter circuits emulated in the sector
# ---------------------------------------------------------------------------

class SectorTrotter:
    """Structured Trotter evolution emulated in a sector basis.

    layers   list of sublayers; each sublayer is a list of (src, tgt, coef)
             two-level terms acting on disjoint degrees of freedom
    diag     diagonal part of H in the same basis
    """

    def __init__(self, layers, diag, dt: float, order: int = 1):
        self.layers = [[(np.asarray(s), np.asarray(t), float(c)) for (s, t, c) in layer
                        if c != 0.0 and len(s)] for layer in layers]
        self.layers = [layer for layer in self.layers if layer]
        self.diag = np.asarray(diag, dtype=float)
        self.dt = float(dt)
        self.order = order
        self.schedule = palindromic_schedule(len(self.layers), order) if self.layers else []
        self._half_phase = np.exp(-0.5j * self.dt * self.diag)

    @classmethod
    def for_gauge(cls, sector: GaugeSector, g=0.0, J=1.0, rk=0.0, field_pattern="uniform",
                  dt=0.05, order=1, support=None):
        """Emulator of circuit.StructuredTrotter on a GaugeSector (optionally a charge block)."""
        lat = sector.lattice
        local = None
        if support is not None:
            local = -np.ones(sector.dim, dtype=np.int64)
            local[support] = np.arange(len(support))
        layers = []
        for colour in lat.plaquette_sublayer_indices():
            layer = []
            for p in colour:
                src, tgt = sector.plaquette_flips(p)
                if local is not None:
                    keep = local[src] >= 0
                    src, tgt = local[src[keep]], local[tgt[keep]]
                layer.append((src, tgt, J))
            layers.append(layer)
        diag = sector.diagonal(g, rk, field_pattern)
        if support is not None:
            diag = diag[support]
        return cls(layers, diag, dt, order)

    @classmethod
    def for_matter(cls, sector, kappa=1.0, m=0.0, g=0.0, J=0.0, dt=0.05, order=1):
        """Emulator of circuit.MatterTrotter on a MatterSector."""
        flips = sector.flip_terms()
        coef = sector.term_coefficients(kappa, J)
        layers = [[(*flips[key], coef[key]) for key in layer]
                  for layer in sector.lattice.sublayers()]
        return cls(layers, sector.diagonal(m, g), dt, order)

    def _apply_layer(self, psi, layer, fraction):
        """Apply every term of a sublayer in turn (in place on psi)."""
        for src, tgt, c in layer:
            theta = c * self.dt * fraction
            psi[src] = np.cos(theta) * psi[src] - 1j * np.sin(theta) * psi[tgt]
        return psi

    def step(self, psi, n_steps: int = 1) -> np.ndarray:
        psi = np.array(psi, dtype=complex)
        for _ in range(n_steps):
            psi = self._half_phase * psi
            for k, fraction in self.schedule:
                psi = self._apply_layer(psi, self.layers[k], fraction)
            psi = self._half_phase * psi
        return psi


# ---------------------------------------------------------------------------
# Entanglement in a sector basis
# ---------------------------------------------------------------------------

def bipartite_entropy(labels: np.ndarray, vec: np.ndarray, qubits_a) -> float:
    """Von Neumann entropy (natural log) of the reduced state on qubits A.

    `labels` are the integer basis labels of the sector (bit q = qubit q). The
    coefficient matrix M[a, b] over the distinct A and B configurations is
    built from the sector and its singular values give the Schmidt spectrum.
    Returns nan if the smaller side exceeds ENTROPY_MAX_SIDE.
    """
    mask_a = np.int64(sum(1 << q for q in qubits_a))
    a_bits, b_bits = labels & mask_a, labels & ~mask_a
    ua, ia = np.unique(a_bits, return_inverse=True)
    ub, ib = np.unique(b_bits, return_inverse=True)
    if min(len(ua), len(ub)) > ENTROPY_MAX_SIDE:
        return float('nan')
    M = np.zeros((len(ua), len(ub)), dtype=complex)
    M[ia.ravel(), ib.ravel()] = vec
    s = np.linalg.svd(M, compute_uv=False) ** 2
    s = s[s > 1e-15]
    s = s / s.sum()
    return float(-np.sum(s * np.log(s)))


def midplane_qubits(lattice: LatticeGrid) -> list:
    """Link qubits in the lower half of the gap (rows j < H/2), the A side of the cut."""
    cut = lattice.height // 2
    return [q for q in range(lattice.n_qubits) if lattice.link_info(q)[1] < cut]


# ---------------------------------------------------------------------------
# Quench in the pure-gauge model
# ---------------------------------------------------------------------------

def initial_state(sector: GaugeSector, gs: dict, initial="max_flippable", pre_quench=None):
    """(state on the ground-state charge block, its bit string or None).

    initial="max_flippable"  product state with the most flippable plaquettes
    initial=<int>            any configuration in the block, given as bits
    initial="ground_state"   ground state of the pre-quench Hamiltonian, with
                             the couplings in `pre_quench` replacing the
                             post-quench ones (an interaction quench)
    """
    idx = gs['support']
    Hs = gs['hamiltonian'][idx][:, idx]
    psi = np.zeros(len(idx), dtype=complex)
    if isinstance(initial, (int, np.integer)):
        hit = np.nonzero(idx == sector.index_of(int(initial)))[0]
        if not len(hit):
            raise ValueError("initial configuration lies outside the ground-state charge sector")
        psi[hit[0]] = 1.0
        return psi, int(initial)
    if initial == "max_flippable":
        k = sector.most_flippable(idx, Hs)
        psi[k] = 1.0
        return psi, sector.label(int(idx[k]))
    if initial == "ground_state":
        params = dict(pre_quench or {})
        Hpre = sector.hamiltonian(params.get('g', 0.0), params.get('J', 1.0),
                                  params.get('rk', 0.0),
                                  params.get('field_pattern', 'uniform'))[idx][:, idx]
        _, v = GaugeSector.lowest(Hpre, 1)
        return v[:, 0].astype(complex), None
    raise ValueError(f"unknown initial state {initial!r}")


def row_profiles(sector: GaugeSector, vec, idx, J, rk) -> dict:
    """Per plaquette row: flippability, energy density, horizontal-link field."""
    full = np.zeros(sector.dim, dtype=complex)
    full[idx] = vec
    lat = sector.lattice
    W, H = lat.width, lat.height
    flip = sector.flippability(full).reshape(H - 1, W).mean(axis=1)
    plaq = sector.plaquette_expectations(full).reshape(H - 1, W).mean(axis=1)
    ez = sector.electric_field(full)
    e_rows = ez[:lat.n_horizontal_links].reshape(H, W).mean(axis=1)
    return {'flippability': flip, 'energy_density': J * plaq + rk * flip,
            'electric_field_rows': e_rows}


def quench(W=4, d=3, J=1.0, g=0.0, rk=0.0, field_pattern="uniform", plates=None,
           initial="max_flippable", pre_quench=None, t_max=10.0, n_times=101,
           trotter=((0.25, 1), (0.25, 2), (0.1, 1)), entanglement=True,
           max_dim=6_000_000) -> dict:
    """Exact quench on the W x (d+1) cylinder with optional Trotter comparison.

    trotter: (dt, order) pairs. For each, the structured Trotter circuit is
    emulated in the sector and its fidelity with the exact state is recorded
    at the multiples of dt that fall on the output time grid.
    """
    lattice = LatticeGrid(W, d + 1)
    plates = plates or default_plates(W)
    sector = GaugeSector(lattice, plates=plates, max_dim=max_dim)
    gs = sector.ground_state(g=g, J=J, rk=rk, field_pattern=field_pattern)
    idx = gs['support']
    Hs = gs['hamiltonian'][idx][:, idx].tocsr()
    psi0, bits = initial_state(sector, gs, initial, pre_quench)
    times = np.linspace(0.0, t_max, n_times)
    dt_out = times[1] - times[0] if n_times > 1 else 0.0

    labels = sector.basis[idx] if sector.n_words == 1 else None
    cut = midplane_qubits(lattice)
    rows = {'flippability': [], 'energy_density': [], 'electric_field_rows': []}
    echo, entropy = [], []
    psi = psi0.copy()
    for k in range(n_times):
        if k:
            psi = expm_multiply(-1j * dt_out * Hs, psi)
        echo.append(float(abs(np.vdot(psi0, psi)) ** 2))
        prof = row_profiles(sector, psi, idx, J, rk)
        for key in rows:
            rows[key].append(prof[key].tolist())
        if entanglement and labels is not None:
            entropy.append(bipartite_entropy(labels, psi, cut))

    out = {'W': W, 'd': d, 'qubits': lattice.n_qubits, 'J': J, 'g': g, 'rk': rk,
           'field_pattern': field_pattern, 'plates': plates, 'initial': str(initial),
           'initial_bits': bits, 'sector_dim': sector.dim, 'block_dim': len(idx),
           'E0': gs['energy'], 'gap': gs['gap'],
           'quench_energy': float(np.vdot(psi0, Hs @ psi0).real),
           'times': times.tolist(), 'return_probability': echo,
           'entropy_midplane': entropy, **{k: v for k, v in rows.items()},
           'ground_state_profiles': {k: v.tolist() for k, v in
                                     row_profiles(sector, gs['state'][idx], idx, J, rk).items()},
           'trotter': []}

    for dt, order in trotter:
        n_total = int(round(t_max / dt))
        emu = SectorTrotter.for_gauge(sector, g=g, J=J, rk=rk, field_pattern=field_pattern,
                                      dt=dt, order=order, support=idx)
        psi_t, psi_e = psi0.copy(), psi0.copy()
        fidelity, echo_trot = [1.0], [1.0]
        for _ in range(n_total):
            psi_t = emu.step(psi_t)
            psi_e = expm_multiply(-1j * dt * Hs, psi_e)
            fidelity.append(float(abs(np.vdot(psi_e, psi_t)) ** 2))
            echo_trot.append(float(abs(np.vdot(psi0, psi_t)) ** 2))
        t_trot = dt * np.arange(n_total + 1)
        step_res = trotter_step_cost(lattice, g, J, rk, field_pattern, dt, order)
        out['trotter'].append({'dt': dt, 'order': order, 'times': t_trot.tolist(),
                               'fidelity': fidelity, 'return_probability': echo_trot,
                               **step_res})
    return out


def trotter_step_cost(lattice, g, J, rk, field_pattern, dt, order) -> dict:
    """Measured two-qubit cost of one structured step (transpiled to {cx, rz, sx, x})."""
    step = StructuredTrotter(lattice, g=g, J=J, dt=dt, rk=rk, field_pattern=field_pattern,
                             order=order).build_step(barriers=False)
    res = circuit_resources(step)
    return {'two_qubit_gates_per_step': res['two_qubit_gates'],
            'two_qubit_depth_per_step': res['two_qubit_depth']}


def time_average(series, times, t_from) -> np.ndarray:
    """Mean of a time series over t >= t_from."""
    series, times = np.asarray(series, float), np.asarray(times, float)
    return series[times >= t_from].mean(axis=0)


def quench_vs_separation(W=4, separations=(1, 2, 3, 4, 5), J=1.0, g=0.0, rk=0.0,
                         field_pattern="uniform", initial="max_flippable", t_max=20.0,
                         n_times=201, trotter=((0.25, 2),), verbose=True) -> dict:
    """The same quench for every plate separation: relaxation and the plate imprint.

    Recorded per d: the return probability and entanglement curves, their
    long-time averages (second half of the window), the time-averaged energy
    density per row compared with the ground state, the time at which the
    return probability first falls below 1/e, and the Trotter fidelity at
    t_max. These quantify how the plates shape the post-quench state and how
    deep a circuit has to be to see it.
    """
    runs = []
    for d in separations:
        run = quench(W, d, J=J, g=g, rk=rk, field_pattern=field_pattern, initial=initial,
                     t_max=t_max, n_times=n_times, trotter=trotter)
        t = np.array(run['times'])
        echo = np.array(run['return_probability'])
        below = np.nonzero(echo < np.exp(-1))[0]
        run['echo_decay_time'] = float(t[below[0]]) if len(below) else float('nan')
        run['echo_long_time'] = float(time_average(echo, t, t_max / 2))
        run['energy_density_long_time'] = time_average(run['energy_density'], t,
                                                       t_max / 2).tolist()
        run['flippability_long_time'] = time_average(run['flippability'], t,
                                                     t_max / 2).tolist()
        if run['entropy_midplane']:
            ent = np.array(run['entropy_midplane'])
            run['entropy_long_time'] = float(time_average(ent, t, t_max / 2))
            run['entropy_max'] = float(np.nanmax(ent))
        runs.append(run)
        if verbose:
            tr = run['trotter'][0] if run['trotter'] else None
            msg = (f" d={d}  {run['qubits']:2d} qubits  block dim {run['block_dim']:6d}  "
                   f"echo 1/e time {run['echo_decay_time']:.2f}  "
                   f"late echo {run['echo_long_time']:.3f}")
            if run['entropy_midplane']:
                msg += f"  S_max {run['entropy_max']:.3f}"
            if tr:
                msg += (f"  Trotter(dt={tr['dt']}, order {tr['order']}) F(t_max)="
                        f"{tr['fidelity'][-1]:.4f}, {tr['two_qubit_gates_per_step']} CX/step")
            print(msg)
    return {'W': W, 'J': J, 'g': g, 'rk': rk, 'field_pattern': field_pattern,
            'initial': str(initial), 't_max': t_max, 'runs': runs}


def trotter_budget(W=3, d=2, J=2.0, g=0.0, rk=0.0, field_pattern="uniform",
                   dts=(0.4, 0.2, 0.1, 0.05), orders=(1, 2), t_max=5.0,
                   fidelity_target=0.99) -> dict:
    """Infidelity of Trotterised quenches and the two-qubit gates they cost.

    For every (dt, order): fidelity with the exact state up to t_max, the
    longest time with fidelity >= fidelity_target, and the two-qubit gates
    needed to reach it. This is the input to the hardware feasibility
    estimate (gates within the error budget of a device).
    """
    out = {'W': W, 'd': d, 'J': J, 'g': g, 'rk': rk, 'fidelity_target': fidelity_target,
           'rows': []}
    for order in orders:
        for dt in dts:
            run = quench(W, d, J=J, g=g, rk=rk, field_pattern=field_pattern, t_max=t_max,
                         n_times=2, trotter=((dt, order),), entanglement=False)
            tr = run['trotter'][0]
            fid = np.array(tr['fidelity'])
            ok = np.nonzero(fid < fidelity_target)[0]
            n_ok = int(ok[0]) - 1 if len(ok) else len(fid) - 1
            out['rows'].append({'dt': dt, 'order': order,
                                'times': tr['times'], 'fidelity': tr['fidelity'],
                                'steps_within_target': n_ok,
                                'time_within_target': n_ok * dt,
                                'two_qubit_gates_per_step': tr['two_qubit_gates_per_step'],
                                'two_qubit_gates_to_target':
                                    n_ok * tr['two_qubit_gates_per_step']})
    return out
