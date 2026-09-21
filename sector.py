"""Exact diagonalisation inside the gauge-invariant sector.

The penalised qubit Hamiltonian lives in a 2^N_q dimensional space, but Gauss's
law removes almost all of it: the 3x3 cylinder has 2^15 = 32768 qubit states,
1792 of which satisfy Gauss's law on the interior sites, and only 21 of those
share the plate-charge pattern of the ground state. Working directly in that
subspace

  * gives the classical reference to machine precision with no penalty term
    (so no penalty-induced conditioning problems),
  * reaches separations d = H - 1 far beyond full-space sparse ED
    (N_q up to 62 link qubits; the limit is the int64 basis label),
  * makes the conserved plate charges explicit, which resolves the ground-state
    degeneracy seen by the penalised Hamiltonian with free plates.

Basis states are integers whose bit q is the state of link qubit q
(bit 1 = |1> = S^z = -1/2), i.e. exactly the Qiskit statevector index.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, expm_multiply

from QuantumFields.gauge_field import GaussLaw, field_signs

MAX_QUBITS = 62
DENSE_LIMIT = 600


class GaugeSector:
    """Gauge-invariant subspace of the U(1) QLM on a cylinder.

    plates="free"   Gauss's law on interior sites only (all plate-charge sectors)
    plates="fixed"  Gauss's law on every site with the given plate charges
    """

    def __init__(self, lattice, plates: str = "free", plate_charges: dict = None,
                 max_dim: int = 6_000_000):
        if lattice.n_qubits > MAX_QUBITS:
            raise ValueError(f"{lattice.n_qubits} link qubits exceed the int64 "
                             f"basis label limit ({MAX_QUBITS})")
        self.lattice = lattice
        self.gauss = GaussLaw(lattice, plates=plates, plate_charges=plate_charges)
        self.plates = plates
        self.max_dim = max_dim
        self.basis = self._enumerate()
        self.dim = len(self.basis)
        self._plaq_cache = None
        self._plaq_mats = {}
        self._HB = None
        self._flip_count = None

    # ------------------------------------------------------------------
    # Basis enumeration, row by row
    # ------------------------------------------------------------------

    def _sz(self, states, qubit):
        return 0.5 - ((states >> qubit) & 1)

    def _enumerate(self) -> np.ndarray:
        lat = self.lattice
        W, H = lat.width, lat.height
        states = np.zeros(1, dtype=np.int64)
        row_choices = np.arange(2 ** W, dtype=np.int64)

        for j in range(H):
            # all configurations of the horizontal links in row j
            shift = lat.get_horizontal_link_index(0, j)
            states = (states[:, None] | (row_choices[None, :] << shift)).ravel()

            constrained = self.gauss.is_constrained(0, j)
            if j == H - 1:
                if constrained:                       # top plate, fixed charges
                    ok = np.ones(len(states), dtype=bool)
                    for i in range(W):
                        div = (self._sz(states, lat.get_horizontal_link_index(i, j))
                               - self._sz(states, lat.get_horizontal_link_index(i - 1, j))
                               - self._sz(states, lat.get_vertical_link_index(i, j - 1)))
                        ok &= (div == self.gauss.charge(i, j))
                    states = states[ok]
                break

            if not constrained:                       # free bottom plate
                shift = lat.get_vertical_link_index(0, j)
                states = (states[:, None] | (row_choices[None, :] << shift)).ravel()
            else:                                     # Gauss's law fixes the up-links
                ok = np.ones(len(states), dtype=bool)
                add = np.zeros(len(states), dtype=np.int64)
                for i in range(W):
                    sz_up = (self.gauss.charge(i, j)
                             - self._sz(states, lat.get_horizontal_link_index(i, j))
                             + self._sz(states, lat.get_horizontal_link_index(i - 1, j)))
                    if j > 0:
                        sz_up = sz_up + self._sz(states, lat.get_vertical_link_index(i, j - 1))
                    ok &= (np.abs(sz_up) == 0.5)
                    add |= (sz_up < 0).astype(np.int64) << lat.get_vertical_link_index(i, j)
                states = (states | add)[ok]
            if len(states) > self.max_dim:
                raise MemoryError(f"sector larger than max_dim={self.max_dim}")
        return np.sort(states)

    # ------------------------------------------------------------------
    # Operators in the sector basis
    # ------------------------------------------------------------------

    def _plaquette_data(self):
        """Per plaquette: (flippable mask, source indices, target indices)."""
        if self._plaq_cache is None:
            B = self.basis
            data = []
            for (b, r, t, l) in self.lattice.get_plaquettes():
                bb, rr = (B >> b) & 1, (B >> r) & 1
                tt, ll = (B >> t) & 1, (B >> l) & 1
                flippable = ((bb == 1) & (rr == 1) & (tt == 0) & (ll == 0)) | \
                            ((bb == 0) & (rr == 0) & (tt == 1) & (ll == 1))
                mask = (1 << b) | (1 << r) | (1 << t) | (1 << l)
                src = np.nonzero(flippable)[0]
                tgt = np.searchsorted(B, B[src] ^ mask)
                if len(src) and not np.array_equal(B[tgt], B[src] ^ mask):
                    raise RuntimeError("plaquette flip left the gauge sector")
                data.append((flippable, src, tgt))
            self._plaq_cache = data
        return self._plaq_cache

    def plaquette_matrix(self, p: int) -> sp.csr_matrix:
        """U_p + U_p^dag restricted to the sector (cached)."""
        if p not in self._plaq_mats:
            _, src, tgt = self._plaquette_data()[p]
            self._plaq_mats[p] = sp.coo_matrix(
                (np.ones(len(src)), (tgt, src)), shape=(self.dim, self.dim)).tocsr()
        return self._plaq_mats[p]

    def magnetic_matrix(self) -> sp.csr_matrix:
        """sum_p (U_p + U_p^dag), assembled in one pass and cached."""
        if self._HB is None:
            data = self._plaquette_data()
            src = np.concatenate([d[1] for d in data])
            tgt = np.concatenate([d[2] for d in data])
            self._HB = sp.coo_matrix((np.ones(len(src)), (tgt, src)),
                                     shape=(self.dim, self.dim)).tocsr()
        return self._HB

    def flippable_count(self) -> np.ndarray:
        if self._flip_count is None:
            self._flip_count = np.sum([d[0] for d in self._plaquette_data()],
                                      axis=0).astype(float)
        return self._flip_count

    def flippable(self, p: int) -> np.ndarray:
        return self._plaquette_data()[p][0].astype(float)

    def sz_diagonal(self, qubit: int) -> np.ndarray:
        return self._sz(self.basis, qubit)

    def hamiltonian(self, g: float = 0.0, J: float = 1.0, rk: float = 0.0,
                    field_pattern: str = "uniform") -> sp.csr_matrix:
        """H_E + H_B + H_RK in the sector basis (no penalty needed)."""
        diag = np.zeros(self.dim)
        if rk != 0.0:
            diag += rk * self.flippable_count()
        if g != 0.0:
            signs = field_signs(self.lattice, field_pattern)
            for q in range(self.lattice.n_qubits):
                diag += -g * signs[q] * self.sz_diagonal(q)
        H = J * self.magnetic_matrix() if J != 0.0 else sp.csr_matrix((self.dim, self.dim))
        return (H + sp.diags(diag)).tocsr()

    # ------------------------------------------------------------------
    # Conserved plate charges
    # ------------------------------------------------------------------

    def plate_charge_table(self) -> np.ndarray:
        """(dim, 2W) array: divergence of E at every plate site, bottom row first."""
        lat = self.lattice
        cols = []
        for j in (0, lat.height - 1):
            for i in range(lat.width):
                div = np.zeros(self.dim)
                for qubit, sign in lat.site_links(i, j):
                    div += sign * self.sz_diagonal(qubit)
                cols.append(div)
        return np.array(cols).T

    def charge_sectors(self) -> list:
        """[(charge pattern, basis indices)] for every static plate-charge sector."""
        table = self.plate_charge_table()
        patterns, inverse = np.unique(table, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).ravel()
        return [(patterns[k], np.nonzero(inverse == k)[0]) for k in range(len(patterns))]

    # ------------------------------------------------------------------
    # Spectra
    # ------------------------------------------------------------------

    @staticmethod
    def lowest(Hm, k: int = 2, seed: int = 0):
        """Lowest k eigenpairs; dense for small blocks, Lanczos otherwise."""
        n = Hm.shape[0]
        k = max(1, min(k, n))
        if n <= DENSE_LIMIT:
            w, v = np.linalg.eigh(Hm.toarray() if sp.issparse(Hm) else Hm)
            return w[:k], v[:, :k]
        rng = np.random.default_rng(seed)
        w, v = eigsh(Hm, k=k, which='SA', v0=rng.standard_normal(n))
        order = np.argsort(w)
        return w[order], v[:, order]

    def ground_state(self, g=0.0, J=1.0, rk=0.0, field_pattern="uniform",
                     resolve_charges: bool = True, seed: int = 0) -> dict:
        """Ground state of the sector.

        With free plates the ground state is degenerate across symmetry-related
        plate-charge patterns. resolve_charges=True diagonalises each pattern
        separately and returns the unique ground state of the best one, with the
        in-sector gap (the gap relevant to any gauge-invariant algorithm).
        """
        Hm = self.hamiltonian(g, J, rk, field_pattern)
        if self.plates == "fixed" or not resolve_charges:
            w, v = self.lowest(Hm, 2, seed)
            idx = np.arange(self.dim)
            pattern = None
            vec = v[:, 0]
        else:
            # One Lanczos run in the whole sector. The (possibly degenerate)
            # ground vector is a mixture of symmetry-related plate-charge
            # patterns; keep the dominant pattern and diagonalise only its block.
            _, v_all = self.lowest(Hm, 1, seed)
            table = self.plate_charge_table()
            patterns, inverse = np.unique(table, axis=0, return_inverse=True)
            inverse = np.asarray(inverse).ravel()
            weights = np.bincount(inverse, weights=np.abs(v_all[:, 0]) ** 2,
                                  minlength=len(patterns))
            k = int(np.argmax(weights))
            pattern, idx = patterns[k], np.nonzero(inverse == k)[0]
            w, v_k = self.lowest(Hm[idx][:, idx], 2, seed)
            vec = np.zeros(self.dim)
            vec[idx] = v_k[:, 0]
        if vec[np.argmax(np.abs(vec))] < 0:
            vec = -vec
        gap = float(w[1] - w[0]) if len(w) > 1 else float('nan')
        return {'energy': float(w[0]), 'gap': gap, 'state': vec,
                'support': idx, 'plate_charges': pattern, 'hamiltonian': Hm}

    # ------------------------------------------------------------------
    # Observables and dynamics
    # ------------------------------------------------------------------

    def electric_field(self, vec) -> np.ndarray:
        prob = np.abs(vec) ** 2
        return np.array([prob @ self.sz_diagonal(q)
                         for q in range(self.lattice.n_qubits)])

    def plaquette_expectations(self, vec) -> np.ndarray:
        """<U_p + U_p^dag> for every plaquette (magnetic energy density / J)."""
        vec = np.asarray(vec)
        return np.array([np.real(np.vdot(vec[tgt], vec[src]))
                         for _, src, tgt in self._plaquette_data()])

    def flippability(self, vec) -> np.ndarray:
        prob = np.abs(vec) ** 2
        return np.array([prob @ d[0] for d in self._plaquette_data()])

    def evolve(self, vec, Hm, times) -> np.ndarray:
        """|psi(t)> = exp(-i H t)|psi(0)> at the requested (uniform) times."""
        times = np.asarray(times, dtype=float)
        if len(times) == 1:
            return expm_multiply(-1j * times[0] * Hm, vec.astype(complex))[None, :]
        return expm_multiply(-1j * Hm, vec.astype(complex), start=times[0],
                             stop=times[-1], num=len(times), endpoint=True)

    def product_state(self, bits: int) -> np.ndarray:
        k = np.searchsorted(self.basis, bits)
        if k >= self.dim or self.basis[k] != bits:
            raise ValueError("configuration is not in the gauge-invariant sector")
        vec = np.zeros(self.dim)
        vec[k] = 1.0
        return vec

    def embed(self, vec) -> np.ndarray:
        """Sector vector -> full 2^N_q Qiskit statevector."""
        if self.lattice.n_qubits > 26:
            raise MemoryError("full statevector too large to embed")
        full = np.zeros(2 ** self.lattice.n_qubits, dtype=np.asarray(vec).dtype)
        full[self.basis] = vec
        return full

    def project(self, full) -> np.ndarray:
        """Full statevector -> sector components (weight outside is dropped)."""
        return np.asarray(full)[self.basis]
