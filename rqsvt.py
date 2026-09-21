"""Randomised QSVT ground-state preparation (thesis Sec. 2.3.4).

Single-ancilla implementation in the spirit of Wang et al. (2025): a Fourier
(Laurent-polynomial) spectral filter F(H) is realised by generalised quantum
signal processing (GQSP, Motlagh & Wiebe 2024) with the signal operator

        W = exp(i (t0/2) Z_anc (x) H),

so the ancilla-|0> branch evolves forward and the ancilla-|1> branch backward in
time. No block encoding and no controlled time evolution are needed: for a term
h_k of H, exp(i theta Z (x) h_k) is an ordinary Pauli rotation on one more
qubit. 2D signal operators interleaved with 2D+1 single-qubit rotations give

        <0|_anc  circuit  |0>_anc  =  F(H) = sum_{k=-D}^{D} c_k exp(i k t0 H).

Three realisations of W are provided:
    "exact"    exp(+-i t0 H / 2) applied exactly (reference for the filter),
    "trotter"  n structured Trotter steps per signal operator,
    "qdrift"   N randomly sampled term rotations per signal operator
               (importance sampling with probability |a_k| / lambda_H).
Energies obtained at several n (or N) are Richardson-extrapolated to 1/n -> 0.

Gauge invariance. H is decomposed at *plaquette* granularity,
H = sum_k a_k h_k with h_k in {U_p + U_p^dag, (U_p + U_p^dag)^2, Z_l}. Every
h_k commutes with every Gauss generator, so each sampled rotation is exactly
gauge invariant and the filter never leaves the sector of the initial state. A
single Pauli string of the plaquette expansion does not have this property.
For the same reason the penalty term is not needed inside the filter, which
keeps lambda_H (and hence the circuit length) small.

Classical emulation works in the gauge-invariant sector with the two ancilla
branches stored separately, so it is cheap even for 30+ link qubits.
build_circuit() emits the actual (N_q + 1)-qubit Qiskit circuit; the test-suite
checks that both agree.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.special import erfc

from qiskit import QuantumCircuit

from circuit import (append_flippability_rotation, append_pauli_rotation,
                     append_plaquette_rotation, circuit_resources)
from QuantumFields.gauge_field import QuantumLinkModel, field_signs
from sector import GaugeSector


# ---------------------------------------------------------------------------
# Generalised quantum signal processing
# ---------------------------------------------------------------------------

def gqsp_rotation(theta: float, phi: float, lam: float = 0.0) -> np.ndarray:
    """R(theta, phi, lambda) of Motlagh & Wiebe, a generic SU(2) element up to phase."""
    return np.array(
        [[np.exp(1j * (lam + phi)) * np.cos(theta), np.exp(1j * phi) * np.sin(theta)],
         [np.exp(1j * lam) * np.sin(theta), -np.cos(theta)]])


def complementary_polynomial(P: np.ndarray) -> np.ndarray:
    """Q with |P(z)|^2 + |Q(z)|^2 = 1 on |z| = 1 and deg Q = deg P.

    FFT construction of Berntson & Suenderhauf (2025): Q = exp(Pi[log(1-|P|^2)]),
    where Pi keeps the analytic half of the Fourier series. Requires |P| < 1.
    """
    d = len(P) - 1
    N = 2 ** int(np.ceil(np.log2(128 * (d + 1))))
    z = np.exp(2j * np.pi * np.arange(N) / N)
    Pz = np.polyval(P[::-1], z)
    if np.max(np.abs(Pz)) >= 1.0:
        raise ValueError("GQSP needs |P(z)| < 1 on the unit circle")
    c = np.fft.fft(np.log(1.0 - np.abs(Pz) ** 2)) / N
    analytic = np.zeros(N, dtype=complex)
    analytic[0] = c[0] / 2.0
    analytic[1:N // 2] = c[1:N // 2]
    Qz = np.exp(np.fft.ifft(analytic) * N)
    outer = (np.fft.fft(Qz) / N)[:d + 1]
    # The outer function has a vanishing leading coefficient, which makes layer
    # stripping divide by ~1e-10 and lose all accuracy for step-like filters.
    # Its reciprocal polynomial z^d conj(Q(1/conj z)) has the same modulus on the
    # circle and an O(1) leading coefficient; angles then come out to 1e-14.
    return np.conj(outer[::-1])


def gqsp_angles(P: np.ndarray, Q: np.ndarray = None):
    """Rotation angles (theta_k, phi_k, lambda) implementing P by layer stripping."""
    P = np.asarray(P, dtype=complex)
    Q = complementary_polynomial(P) if Q is None else np.asarray(Q, dtype=complex)
    S = np.array([P, Q])
    d = len(P) - 1
    theta, phi, lam = np.zeros(d + 1), np.zeros(d + 1), 0.0
    for k in range(d, -1, -1):
        a, b = S[0, k], S[1, k]
        theta[k] = np.arctan2(abs(b), abs(a))
        if abs(a) > 1e-300 and abs(b) > 1e-300:
            phi[k] = np.angle(a) - np.angle(b)
        elif abs(b) <= 1e-300:
            phi[k] = np.angle(a)
        else:
            phi[k] = -np.angle(b)
        if k == 0:
            lam = np.angle(b) if abs(b) > 1e-300 else 0.0
        else:
            S = gqsp_rotation(theta[k], phi[k]).conj().T @ S
            S = np.array([S[0, 1:k + 1], S[1, 0:k]])
    return theta, phi, lam


def gqsp_response(theta, phi, lam, z: complex) -> complex:
    """<0| prod_k R_k diag(z, 1) ... R_0 |0>: the polynomial the angles realise."""
    M = gqsp_rotation(theta[0], phi[0], lam)
    A = np.diag([z, 1.0])
    for k in range(1, len(theta)):
        M = gqsp_rotation(theta[k], phi[k]) @ A @ M
    return M[0, 0]


# ---------------------------------------------------------------------------
# Spectral filters as Fourier series in theta = t0 * (E - E_c)
# ---------------------------------------------------------------------------

def fourier_filter(kind: str, D: int, centre: float, width: float,
                   headroom: float = 0.02) -> np.ndarray:
    """Coefficients c_{-D..D} of a degree-D Fourier filter on theta in (-pi, pi].

    kind="step"      smooth indicator of theta < centre (shifted-sign filter of
                     the draft); `width` is the erf smoothing scale. The second,
                     unavoidable edge of the periodic function sits at theta = pi,
                     outside the rescaled spectrum.
    kind="gaussian"  exp(-(theta - centre)^2 / (2 width^2)), needs E0 itself.
    The result is rescaled so that max |F| = 1 - headroom.
    """
    N = 2 ** int(np.ceil(np.log2(64 * (D + 1))))
    theta = -np.pi + 2 * np.pi * np.arange(N) / N
    if kind == "step":
        lower = 0.5 * erfc(-(theta + np.pi - 3 * width) / (np.sqrt(2) * width))
        upper = 0.5 * erfc((theta - centre) / (np.sqrt(2) * width))
        target = lower * upper
    elif kind == "gaussian":
        delta = (theta - centre + np.pi) % (2 * np.pi) - np.pi
        target = np.exp(-delta ** 2 / (2 * width ** 2))
    else:
        raise ValueError(f"unknown filter kind {kind!r}")
    ks = np.arange(-D, D + 1)
    # c_k = (1/N) sum_n f(theta_n) exp(-i k theta_n), theta_n = -pi + 2 pi n / N
    spectrum = np.fft.fft(target) / N
    coeffs = spectrum[ks % N] * np.exp(1j * np.pi * ks)
    dense = np.exp(1j * np.outer(theta, ks)) @ coeffs
    return coeffs * (1.0 - headroom) / np.max(np.abs(dense))


def filter_degree(width: float, truncation: float = 1e-3) -> int:
    """Smallest D with Gaussian-tail truncation error below `truncation`."""
    return int(np.ceil(np.sqrt(2.0 * np.log(1.0 / truncation)) / width))


def richardson_extrapolate(hs, ys) -> float:
    """Lagrange extrapolation of y(h) to h = 0."""
    total = 0.0
    for i in range(len(hs)):
        weight = 1.0
        for j in range(len(hs)):
            if j != i:
                weight *= (0.0 - hs[j]) / (hs[i] - hs[j])
        total += ys[i] * weight
    return float(total)


# ---------------------------------------------------------------------------
# RQSVT ground-state preparation
# ---------------------------------------------------------------------------

class RQSVTGroundState:
    """Ground-state filter for the pure-gauge QLM with one ancilla qubit.

    The reference spectrum (E0 and the in-sector gap) comes from sector ED; this
    is the prior knowledge assumed by Wang et al. and would be replaced by a
    binary search over the filter threshold on hardware.
    """

    def __init__(self, lattice, g=0.0, J=1.0, rk=0.0, field_pattern="uniform",
                 plates="free", plate_charges=None, filter_kind="step",
                 sharpness: float = 3.0, truncation: float = 1e-3,
                 degree: int = None, spectrum_margin: float = 0.9,
                 initial_state="max_flippable"):
        self.lattice = lattice
        self.n = lattice.n_qubits
        self.g, self.J, self.rk = float(g), float(J), float(rk)
        self.field_pattern = field_pattern
        self.filter_kind = filter_kind
        self.model = QuantumLinkModel(lattice, g=g, J=J, rk=rk,
                                      field_pattern=field_pattern, plates=plates,
                                      plate_charges=plate_charges)

        # --- classical reference in the gauge sector ---------------------
        full_sector = GaugeSector(lattice, plates=plates, plate_charges=plate_charges)
        ref = full_sector.ground_state(g=g, J=J, rk=rk, field_pattern=field_pattern)
        self.support = ref['support']              # plate-charge sector of the GS
        self._sector = full_sector
        self.basis = (full_sector.basis[self.support] if full_sector.n_words == 1 else None)
        self.dim = len(self.support)
        self.E0, self.gap = ref['energy'], ref['gap']
        self.gs_exact = ref['state'][self.support]
        self.plate_charges = ref['plate_charges']
        self._H = ref['hamiltonian'][self.support][:, self.support].tocsr()

        # --- term decomposition H = sum_k a_k h_k (plaquette granularity) --
        self.terms = []                            # (kind, index, coefficient)
        self._term_ops = {}
        for p in range(lattice.n_plaquettes):
            if self.J != 0.0:
                self.terms.append(('plaq', p, self.J))
            if self.rk != 0.0:
                self.terms.append(('flip', p, self.rk))
            hp = full_sector.plaquette_matrix(p)[self.support][:, self.support].tocsr()
            self._term_ops[('plaq', p)] = hp
            self._term_ops[('flip', p)] = full_sector.flippable(p)[self.support]
        if self.g != 0.0:
            signs = field_signs(lattice, field_pattern)
            for q in range(self.n):
                self.terms.append(('z', q, -0.5 * self.g * signs[q]))
                self._term_ops[('z', q)] = 2.0 * full_sector.sz_diagonal(q)[self.support]
        self.lambda_H = float(sum(abs(a) for _, _, a in self.terms))
        self._dense_cache = {}

        # --- spectral rescaling and filter ----------------------------------
        self.t0 = spectrum_margin * np.pi / self.lambda_H
        gap_theta = self.t0 * self.gap
        if filter_kind == "step":
            self.width = gap_theta / (2.0 * sharpness)
            centre = self.t0 * (self.E0 + 0.5 * self.gap)
        else:
            self.width = gap_theta / sharpness
            centre = self.t0 * self.E0
        self.D = int(degree) if degree is not None else filter_degree(self.width, truncation)
        self.coeffs = fourier_filter(filter_kind, self.D, centre, self.width)
        self.theta, self.phi, self.lam = gqsp_angles(self.coeffs)
        self.n_signal = 2 * self.D                 # signal operators in the circuit

        self.psi0 = self._initial_state(initial_state)
        self.initial_overlap = float(abs(np.vdot(self.gs_exact, self.psi0)) ** 2)

    # ---- initial state -------------------------------------------------------

    def _initial_state(self, spec) -> np.ndarray:
        """A computational-basis (product) state inside the ground-state sector."""
        if isinstance(spec, (int, np.integer)):
            hit = np.nonzero(self.support == self._sector.index_of(int(spec)))[0]
            if not len(hit):
                raise ValueError("initial configuration is outside the ground-state sector")
            k = int(hit[0])
        elif spec == "best_basis":                 # oracle choice, for comparison
            k = int(np.argmax(np.abs(self.gs_exact)))
        elif spec == "max_flippable":              # needs no knowledge of the GS
            count = sum(self._term_ops[('flip', p)]
                        for p in range(self.lattice.n_plaquettes))
            best = np.nonzero(count == count.max())[0]
            energy = self._H.diagonal()[best]
            k = int(best[np.argmin(energy)])
        else:
            raise ValueError(f"unknown initial state {spec!r}")
        self.initial_bits = self._sector.label(int(self.support[k]))
        psi = np.zeros(self.dim, dtype=complex)
        psi[k] = 1.0
        return psi

    def filter_response(self, energies) -> np.ndarray:
        """F(E) realised by the GQSP angles (evaluated as a scalar)."""
        ks = np.arange(-self.D, self.D + 1)
        return np.exp(1j * self.t0 * np.outer(np.atleast_1d(energies), ks)) @ self.coeffs

    # ---- classical emulation --------------------------------------------------

    def _apply_term(self, branches, key, angle):
        """exp(i angle Z (x) h_k): +angle on the |0> branch, -angle on |1>."""
        kind = key[0]
        op = self._term_ops[key]
        out = []
        for psi, a in zip(branches, (angle, -angle)):
            if kind == 'plaq':                     # h^3 = h
                h_psi = op @ psi
                out.append(psi + 1j * np.sin(a) * h_psi
                           + (np.cos(a) - 1.0) * (op @ h_psi))
            else:                                  # diagonal term
                out.append(np.exp(1j * a * op) * psi)
        return out

    def _signal_exact(self):
        w, v = np.linalg.eigh(self._H.toarray())
        phases = np.exp(0.5j * self.t0 * w)
        return (v * phases) @ v.conj().T, (v * phases.conj()) @ v.conj().T

    def _trotter_sequence(self, n_steps):
        """Term order of one structured first-order step, scaled by 1/n_steps."""
        seq = []
        diag = [(k, i, a) for (k, i, a) in self.terms if k != 'plaq']
        for kind, idx, a in diag:
            seq.append(((kind, idx), 0.5 * a / n_steps))
        anchors = self.lattice.plaquette_anchors()
        for colour in range(self.lattice.n_plaquette_colours):
            for p, (i, j) in enumerate(anchors):
                if self.lattice.plaquette_colour(i, j) == colour and self.J != 0.0:
                    seq.append((('plaq', p), self.J / n_steps))
        for kind, idx, a in diag:
            seq.append(((kind, idx), 0.5 * a / n_steps))
        return seq * n_steps

    def _qdrift_sequence(self, n_samples, rng):
        probs = np.array([abs(a) for _, _, a in self.terms]) / self.lambda_H
        picks = rng.choice(len(self.terms), size=n_samples, p=probs)
        return [((self.terms[k][0], self.terms[k][1]),
                 np.sign(self.terms[k][2]) * self.lambda_H / n_samples) for k in picks]

    # Dense fast path: in small sectors every term rotation is a precomputed
    # dim x dim matrix, a whole Trotter step collapses to one matrix per branch,
    # and a qDRIFT sample costs two small mat-vecs.
    DENSE_DIM = 400
    EXACT_DENSE_DIM = 3000

    def _dense_term(self, key, angle):
        """(exp(+i angle h_k), exp(-i angle h_k)) as dense matrices."""
        op = self._term_ops[key]
        if key[0] == 'plaq':
            h = op.toarray()
            h2 = h @ h
            eye = np.eye(self.dim)
            return (eye + 1j * np.sin(angle) * h + (np.cos(angle) - 1.0) * h2,
                    eye - 1j * np.sin(angle) * h + (np.cos(angle) - 1.0) * h2)
        phase = np.exp(1j * angle * op)
        return np.diag(phase), np.diag(phase.conj())

    def _dense_trotter_signal(self, n_steps):
        key = ('trotter', n_steps)
        if key not in self._dense_cache:
            plus = minus = np.eye(self.dim, dtype=complex)
            for term, weight in self._trotter_sequence(n_steps)[:len(self._trotter_sequence(1))]:
                up, um = self._dense_term(term, 0.5 * self.t0 * weight)
                plus, minus = up @ plus, um @ minus
            self._dense_cache[key] = (np.linalg.matrix_power(plus, n_steps),
                                      np.linalg.matrix_power(minus, n_steps))
        return self._dense_cache[key]

    def _dense_qdrift_terms(self, n_samples):
        key = ('qdrift', n_samples)
        if key not in self._dense_cache:
            angle = 0.5 * self.t0 * self.lambda_H / n_samples
            mats = [self._dense_term((k, i), np.sign(a) * angle) for k, i, a in self.terms]
            self._dense_cache[key] = (np.array([m[0] for m in mats]),
                                      np.array([m[1] for m in mats]))
        return self._dense_cache[key]

    def run(self, method="exact", n_steps=1, n_samples=20, seed=0) -> dict:
        """Emulate the circuit once; returns the unnormalised filtered state."""
        rng = np.random.default_rng(seed)
        dense = self.dim <= self.DENSE_DIM
        probs = np.array([abs(a) for _, _, a in self.terms]) / self.lambda_H
        big_exact = method == "exact" and self.dim > self.EXACT_DENSE_DIM
        if big_exact:
            from scipy.sparse.linalg import expm_multiply
            gen = 0.5j * self.t0 * self._H
        elif method == "exact":
            Wp, Wm = self._signal_exact()
        elif method == "trotter" and dense:
            Wp, Wm = self._dense_trotter_signal(n_steps)
        elif method == "qdrift" and dense:
            Up, Um = self._dense_qdrift_terms(n_samples)
        b0 = gqsp_rotation(self.theta[0], self.phi[0], self.lam)[:, 0]
        x0, x1 = b0[0] * self.psi0, b0[1] * self.psi0
        n_rotations = 0
        per_step = len(self._trotter_sequence(1))
        for k in range(1, self.n_signal + 1):
            if big_exact:
                x0, x1 = expm_multiply(gen, x0), expm_multiply(-gen, x1)
            elif method == "exact" or (method == "trotter" and dense):
                x0, x1 = Wp @ x0, Wm @ x1
                n_rotations += 0 if method == "exact" else per_step * n_steps
            elif method == "qdrift" and dense:
                # same draw as _qdrift_sequence, so circuit and emulator agree
                for pick in rng.choice(len(self.terms), size=n_samples, p=probs):
                    x0, x1 = Up[pick] @ x0, Um[pick] @ x1
                n_rotations += n_samples
            else:
                seq = (self._trotter_sequence(n_steps) if method == "trotter"
                       else self._qdrift_sequence(n_samples, rng))
                branches = [x0, x1]
                for key, weight in seq:
                    branches = self._apply_term(branches, key, 0.5 * self.t0 * weight)
                x0, x1 = branches
                n_rotations += len(seq)
            R = gqsp_rotation(self.theta[k], self.phi[k])
            x0, x1 = R[0, 0] * x0 + R[0, 1] * x1, R[1, 0] * x0 + R[1, 1] * x1
        phi = x0                                   # ancilla post-selected on |0>
        norm2 = float(np.vdot(phi, phi).real)
        numerator = float(np.vdot(phi, self._H @ phi).real)
        return {'state': phi, 'success_prob': norm2, 'numerator': numerator,
                'energy': numerator / norm2,
                'overlap': float(abs(np.vdot(self.gs_exact, phi)) ** 2) / norm2,
                'term_rotations': n_rotations}

    def estimate_energy(self, method="trotter", schedule=(1, 2, 4),
                        n_circuits=1, seed=0) -> dict:
        """<H> at each step count in `schedule`, Richardson-extrapolated to 1/n -> 0.

        For qDRIFT the numerator <phi|H|phi> and denominator <phi|phi> are
        averaged separately over `n_circuits` random circuits, which is what the
        post-selected measurement estimates on hardware.
        """
        hs, energies, runs = [], [], []
        for n in schedule:
            num = den = ov = 0.0
            reps = n_circuits if method == "qdrift" else 1
            for r in range(reps):
                res = self.run(method, n_steps=n, n_samples=n,
                               seed=seed + 1000 * n + r)
                num += res['numerator']
                den += res['success_prob']
                ov += res['overlap'] * res['success_prob']
            hs.append(1.0 / n)
            energies.append(num / den)
            runs.append({'n': n, 'energy': num / den, 'overlap': ov / den,
                         'success_prob': den / reps})
        extrapolated = (richardson_extrapolate(hs, energies)
                        if len(hs) > 1 else energies[0])
        return {'energy_richardson': extrapolated, 'runs': runs,
                'E0_exact': self.E0, 'error': extrapolated - self.E0}

    # ---- Qiskit circuit --------------------------------------------------------

    def _append_term(self, qc, key, angle, ancilla, synthesis=None):
        """exp(i angle Z_anc (x) h_k) as gates."""
        kind, idx = key
        if kind == 'plaq':
            append_plaquette_rotation(qc, -angle, self.lattice.get_plaquettes()[idx],
                                      ancilla=ancilla, method=synthesis)
        elif kind == 'flip':
            append_flippability_rotation(qc, -angle, self.lattice.get_plaquettes()[idx],
                                         ancilla=ancilla, method=synthesis)
        else:
            append_pauli_rotation(qc, -angle, [idx, ancilla], ['Z', 'Z'])

    def build_circuit(self, method="qdrift", n_steps=1, n_samples=20, seed=0,
                      max_signal: int = None, synthesis: str = None) -> QuantumCircuit:
        """(N_q + 1)-qubit circuit; the ancilla is qubit N_q and starts in |0>.

        `max_signal` truncates the circuit after that many signal operators, for
        resource counting on large lattices.
        """
        if method == "exact":
            raise ValueError("the exact signal operator has no gate decomposition")
        rng = np.random.default_rng(seed)
        anc = self.n
        qc = QuantumCircuit(self.n + 1)
        for q in range(self.n):                    # product initial state
            if (self.initial_bits >> q) & 1:
                qc.x(q)
        qc.unitary(gqsp_rotation(self.theta[0], self.phi[0], self.lam), [anc], label="R0")
        n_signal = self.n_signal if max_signal is None else min(max_signal, self.n_signal)
        for k in range(1, n_signal + 1):
            seq = (self._trotter_sequence(n_steps) if method == "trotter"
                   else self._qdrift_sequence(n_samples, rng))
            for key, weight in seq:
                self._append_term(qc, key, 0.5 * self.t0 * weight, anc, synthesis)
            qc.unitary(gqsp_rotation(self.theta[k], self.phi[k]), [anc], label=f"R{k}")
        return qc

    def resources(self, method="qdrift", n_steps=1, n_samples=20, seed=0,
                  probe_signal: int = 4, synthesis: str = None) -> dict:
        """Gate counts of the full circuit, extrapolated from `probe_signal`
        transpiled signal operators (the circuit is periodic in structure)."""
        probe = min(probe_signal, self.n_signal)
        res = circuit_resources(self.build_circuit(method, n_steps, n_samples, seed,
                                                   max_signal=probe, synthesis=synthesis))
        scale = self.n_signal / probe
        return {'qubits': self.n + 1,
                'signal_operators': self.n_signal,
                'two_qubit_gates': int(round(res['two_qubit_gates'] * scale)),
                'one_qubit_gates': int(round(res['one_qubit_gates'] * scale)),
                'two_qubit_depth': int(round(res['two_qubit_depth'] * scale)),
                'evolution_time': self.D * self.t0,
                'lambda_H': self.lambda_H}

    def summary(self) -> dict:
        return {'qubits': self.n + 1, 'sector_dim': self.dim, 'E0': self.E0,
                'gap': self.gap, 'lambda_H': self.lambda_H, 't0': self.t0,
                'filter': self.filter_kind, 'fourier_degree': self.D,
                'signal_operators': self.n_signal,
                'initial_overlap': self.initial_overlap,
                'initial_bits': self.initial_bits}
