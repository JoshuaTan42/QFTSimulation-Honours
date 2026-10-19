"""Exact diagonalisation inside the gauge-invariant sector of the pure-gauge model.

The qubit Hamiltonian lives in a 2^N_q dimensional space, but Gauss's law
removes almost all of it: the 3x3 cylinder has 2^15 = 32768 qubit states, 1792
of which satisfy Gauss's law on the interior sites, and only 21 of those share
the plate-charge pattern of the ground state. Working directly in that subspace

  * gives the classical reference to machine precision with no penalty term,
    so there is no penalty-induced ill-conditioning,
  * reaches plate separations far beyond full-space sparse ED. Basis labels
    are stored in 62-bit words, as many as needed, so the number of link
    qubits is not limited; the sector dimension is, and `count_states`
    returns it exactly (transfer-matrix count) before anything is enumerated,
  * makes the conserved plate charges explicit, which resolves the
    ground-state degeneracy seen by the penalised Hamiltonian with free plates.

Basis states are bit strings whose bit q is the state of link qubit q
(bit 1 = |1> = S^z = -1/2), i.e. exactly the Qiskit statevector index. For
N_q <= 62 they are exposed as the int64 array `basis`; beyond that use
`label(k)` / `index_of(bits)`, which work with Python integers.

The basis is enumerated row by row: the horizontal links of a row are free,
and Gauss's law at each site of the row then fixes the vertical link above it.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, expm_multiply

from QuantumFields.gauge_field import GaussLaw, field_signs

WORD_BITS = 62                 # link qubits per int64 word of a basis label
DENSE_LIMIT = 600              # dense eigensolver below this dimension, Lanczos above
CHUNK = 1 << 22                # states expanded at once during enumeration


def count_states(lattice, plates: str = "free", plate_charges: dict = None) -> int:
    """Exact dimension of the gauge sector by a transfer-matrix count.

    The state passed from row to row is the configuration of the W vertical
    links below the row; for each of the 2^W horizontal configurations Gauss's
    law fixes (or forbids) the vertical links above. Cost O(H 4^W), no
    enumeration, so it answers "does this lattice fit?" instantly.
    """
    gauss = GaussLaw(lattice, plates=plates, plate_charges=plate_charges)
    W, H = lattice.width, lattice.height
    cfg = np.arange(2 ** W)
    sz = 0.5 - ((cfg[:, None] >> np.arange(W)[None, :]) & 1)          # (2^W, W)
    div_h = sz - np.roll(sz, 1, axis=1)                                # S^z_right - S^z_left
    counts = None                                                      # over v_down configs
    for j in range(H):
        q = np.array([gauss.charge(i, j) for i in range(W)])
        constrained = gauss.is_constrained(0, j)
        n_cfg = 2 ** W
        last = j == H - 1
        new = np.zeros(n_cfg)
        total = 0.0
        block = max(1, (1 << 21) // n_cfg)                   # v_down configs per block
        downs = [None] if j == 0 else range(0, n_cfg, block)
        for lo in downs:
            if lo is None:
                sz_up = q[None, :] - div_h                                     # (h, W)
                weight = np.ones(n_cfg)
            else:
                hi = min(lo + block, n_cfg)
                if not counts[lo:hi].any():
                    continue
                sz_up = (q[None, None, :] - div_h[None, :, :]
                         + sz[lo:hi, None, :]).reshape(-1, W)
                weight = np.repeat(counts[lo:hi], n_cfg)
            if last:
                ok = np.all(sz_up == 0.0, axis=1) if constrained else slice(None)
                total += float(weight[ok].sum())
            elif constrained:
                ok = np.all(np.abs(sz_up) == 0.5, axis=1)
                v_up = ((sz_up[ok] < 0).astype(np.int64) << np.arange(W)[None, :]).sum(axis=1)
                new += np.bincount(v_up, weights=weight[ok], minlength=n_cfg)
            else:                                                              # free plate
                new += float(weight.sum())
        if last:
            return int(round(total))
        counts = new
    raise AssertionError("unreachable")


class GaugeSector:
    """Gauge-invariant subspace of the U(1) QLM on a cylinder.

    plates="free"   Gauss's law on interior sites only (all plate-charge sectors)
    plates="fixed"  Gauss's law on every site with the given plate charges
    """

    def __init__(self, lattice, plates: str = "free", plate_charges: dict = None,
                 max_dim: int = 6_000_000):
        self.lattice = lattice
        self.gauss = GaussLaw(lattice, plates=plates, plate_charges=plate_charges)
        self.plates = plates
        self.max_dim = max_dim
        self.n_words = (lattice.n_qubits + WORD_BITS - 1) // WORD_BITS
        expected = count_states(lattice, plates, plate_charges)
        if expected > max_dim:
            raise MemoryError(f"gauge sector has {expected:,} states, above max_dim="
                              f"{max_dim:,}; pass a larger max_dim if the machine allows "
                              f"(roughly {(8 * lattice.n_plaquettes + 60) * expected / 1e9:.1f} GB "
                              f"for the basis, plaquette tables and Hamiltonian)")
        self._words, self._key, self._u_words = self._enumerate()
        self.dim = len(self._key)
        assert self.dim == expected, "enumeration disagrees with the transfer-matrix count"
        self._plaq_cache = None
        self._plaq_mats = {}
        self._HB = None
        self._flip_count = None

    @property
    def basis(self) -> np.ndarray:
        """int64 labels (= Qiskit statevector indices); only for N_q <= 62."""
        if self.n_words > 1:
            raise ValueError(f"{self.lattice.n_qubits} link qubits do not fit one int64 label; "
                             "use label(k) and index_of(bits)")
        return self._words[0]

    def label(self, k: int) -> int:
        """Basis state k as a Python integer bit string (any number of qubits)."""
        return sum(int(w[k]) << (WORD_BITS * n) for n, w in enumerate(self._words))

    def index_of(self, bits: int) -> int:
        """Index of a configuration given as an integer; ValueError if not physical."""
        mask = (1 << WORD_BITS) - 1
        words = [np.array([(bits >> (WORD_BITS * n)) & mask], dtype=np.int64)
                 for n in range(self.n_words)]
        k = self._lookup(words)[0]
        if k < 0:
            raise ValueError("configuration is not in the gauge-invariant sector")
        return int(k)

    # ------------------------------------------------------------------
    # Basis enumeration, row by row
    # ------------------------------------------------------------------

    @staticmethod
    def _bit(words, qubit):
        return (words[qubit // WORD_BITS] >> (qubit % WORD_BITS)) & 1

    def _sz(self, words, qubit):
        return 0.5 - self._bit(words, qubit)

    def _row_patterns(self, first_qubit):
        """Per-word bit patterns of all 2^W settings of W consecutive link qubits."""
        W = self.lattice.width
        choices = np.arange(2 ** W, dtype=np.int64)
        out = [np.zeros(2 ** W, dtype=np.int64) for _ in range(self.n_words)]
        for i in range(W):
            q = first_qubit + i
            out[q // WORD_BITS] |= ((choices >> i) & 1) << (q % WORD_BITS)
        return out

    @staticmethod
    def _expand(words, patterns):
        return [(w[:, None] | p[None, :]).ravel() for w, p in zip(words, patterns)]

    def _enumerate(self):
        """Row by row; each row is expanded and filtered in chunks to bound memory."""
        lat = self.lattice
        W, H = lat.width, lat.height
        words = [np.zeros(1, dtype=np.int64) for _ in range(self.n_words)]

        for j in range(H):
            h_pat = self._row_patterns(lat.get_horizontal_link_index(0, j))
            constrained = self.gauss.is_constrained(0, j)
            free_up = (j < H - 1) and not constrained
            v_pat = self._row_patterns(lat.get_vertical_link_index(0, j)) if free_up else None
            step = max(1, CHUNK >> (W * (2 if free_up else 1)))
            pieces = [[] for _ in range(self.n_words)]
            for lo in range(0, len(words[0]), step):
                chunk = self._expand([w[lo:lo + step] for w in words], h_pat)
                if free_up:                                   # free bottom plate
                    chunk = self._expand(chunk, v_pat)
                elif constrained:
                    ok = np.ones(len(chunk[0]), dtype=bool)
                    for i in range(W):
                        up = (self.gauss.charge(i, j)
                              - self._sz(chunk, lat.get_horizontal_link_index(i, j))
                              + self._sz(chunk, lat.get_horizontal_link_index(i - 1, j)))
                        if j > 0:
                            up = up + self._sz(chunk, lat.get_vertical_link_index(i, j - 1))
                        if j == H - 1:                        # top plate: nothing above
                            ok &= (up == 0.0)
                        else:                                 # Gauss's law fixes the up-link
                            ok &= (np.abs(up) == 0.5)
                            q = lat.get_vertical_link_index(i, j)
                            chunk[q // WORD_BITS] |= (up < 0).astype(np.int64) << (q % WORD_BITS)
                    chunk = [w[ok] for w in chunk]
                for n in range(self.n_words):
                    pieces[n].append(chunk[n])
            words = [np.concatenate(p) for p in pieces]

        # sort key: the label itself (one word), else a composite rank of the words
        if self.n_words == 1:
            key, uniques = words[0], None
        else:
            uniques = [np.unique(w) for w in words]
            key = np.zeros(len(words[0]), dtype=np.int64)
            for n in reversed(range(self.n_words)):
                key = key * len(uniques[n]) + np.searchsorted(uniques[n], words[n])
        order = np.argsort(key, kind="stable")
        return [w[order] for w in words], key[order], uniques

    def _lookup(self, words) -> np.ndarray:
        """Indices of the given labels in the basis; -1 where absent."""
        if self.n_words == 1:
            key = words[0]
            valid = np.ones(len(key), dtype=bool)
        else:
            key = np.zeros(len(words[0]), dtype=np.int64)
            valid = np.ones(len(words[0]), dtype=bool)
            for n in reversed(range(self.n_words)):
                u = self._u_words[n]
                r = np.minimum(np.searchsorted(u, words[n]), len(u) - 1)
                valid &= (u[r] == words[n])
                key = key * len(u) + r
        pos = np.minimum(np.searchsorted(self._key, key), self.dim - 1)
        found = valid & (self._key[pos] == key)
        return np.where(found, pos, -1)

    # ------------------------------------------------------------------
    # Operators in the sector basis
    # ------------------------------------------------------------------

    def _plaquette_data(self):
        """Per plaquette: (flippable mask, source indices, target indices), cached.

        A plaquette flips exactly two configurations of its links into each
        other, so U_p + U_p^dag restricted to the sector is the permutation
        src[k] <-> tgt[k] on the flippable states.
        """
        if self._plaq_cache is None:
            idx_t = np.int32 if self.dim < 2 ** 31 else np.int64
            data = []
            for (b, r, t, l) in self.lattice.get_plaquettes():
                bb, rr = self._bit(self._words, b), self._bit(self._words, r)
                tt, ll = self._bit(self._words, t), self._bit(self._words, l)
                flippable = ((bb == 1) & (rr == 1) & (tt == 0) & (ll == 0)) | \
                            ((bb == 0) & (rr == 0) & (tt == 1) & (ll == 1))
                src = np.nonzero(flippable)[0]
                flipped = [w[src].copy() for w in self._words]
                for q in (b, r, t, l):
                    flipped[q // WORD_BITS] ^= np.int64(1) << (q % WORD_BITS)
                tgt = self._lookup(flipped)
                if len(src) and tgt.min() < 0:
                    raise RuntimeError("plaquette flip left the gauge sector")
                data.append((flippable, src.astype(idx_t), tgt.astype(idx_t)))
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
        """Number of flippable plaquettes in every basis state."""
        if self._flip_count is None:
            self._flip_count = np.sum([d[0] for d in self._plaquette_data()],
                                      axis=0).astype(float)
        return self._flip_count

    def flippable(self, p: int) -> np.ndarray:
        """1 where plaquette p is flippable, else 0, for every basis state."""
        return self._plaquette_data()[p][0].astype(float)

    def plaquette_flips(self, p: int) -> tuple:
        """(src, tgt): U_p + U_p^dag maps basis state src[k] to tgt[k] (and back)."""
        _, src, tgt = self._plaquette_data()[p]
        return src, tgt

    def sz_diagonal(self, qubit: int) -> np.ndarray:
        """S^z of link `qubit` in every basis state."""
        return self._sz(self._words, qubit)

    def diagonal(self, g: float = 0.0, rk: float = 0.0,
                 field_pattern: str = "uniform") -> np.ndarray:
        """Diagonal part H_E + H_RK of the Hamiltonian in the sector basis."""
        diag = np.zeros(self.dim)
        if rk != 0.0:
            diag += rk * self.flippable_count()
        if g != 0.0:
            signs = field_signs(self.lattice, field_pattern)
            for q in range(self.lattice.n_qubits):
                diag += -g * signs[q] * self.sz_diagonal(q)
        return diag

    def hamiltonian(self, g: float = 0.0, J: float = 1.0, rk: float = 0.0,
                    field_pattern: str = "uniform") -> sp.csr_matrix:
        """H_E + H_B + H_RK in the sector basis (no penalty needed)."""
        H = J * self.magnetic_matrix() if J != 0.0 else sp.csr_matrix((self.dim, self.dim))
        return (H + sp.diags(self.diagonal(g, rk, field_pattern))).tocsr()

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

    def most_flippable(self, support, H_support=None) -> int:
        """Local index (within `support`) of the basis state with most flippable plaquettes.

        This product state needs no knowledge of the ground state; it is the
        initial state of the ground-state filter and of the quench studies.
        Ties are broken by the lowest diagonal energy of `H_support`.
        """
        count = self.flippable_count()[support]
        best = np.nonzero(count == count.max())[0]
        if H_support is None:
            return int(best[0])
        return int(best[np.argmin(H_support.diagonal()[best])])

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
        """<S^z> on every link qubit."""
        prob = np.abs(vec) ** 2
        return np.array([prob @ self.sz_diagonal(q)
                         for q in range(self.lattice.n_qubits)])

    def plaquette_expectations(self, vec) -> np.ndarray:
        """<U_p + U_p^dag> for every plaquette (magnetic energy density / J)."""
        vec = np.asarray(vec)
        return np.array([np.real(np.vdot(vec[tgt], vec[src]))
                         for _, src, tgt in self._plaquette_data()])

    def flippability(self, vec) -> np.ndarray:
        """<(U_p + U_p^dag)^2> for every plaquette."""
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
        """Sector vector of the computational-basis state with the given bits."""
        vec = np.zeros(self.dim)
        vec[self.index_of(int(bits))] = 1.0
        return vec

    def embed(self, vec) -> np.ndarray:
        """Sector vector -> full 2^N_q Qiskit statevector."""
        if self.lattice.n_qubits > 26:
            raise MemoryError("full statevector too large to embed")
        full = np.zeros(2 ** self.lattice.n_qubits, dtype=np.asarray(vec).dtype)
        full[self.basis] = vec
        return full
