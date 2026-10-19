"""Exact diagonalisation and dynamics of the matter-coupled model in its gauge sector.

Gauss's law fixes the matter occupation of every site once the links are
known (n_j = div E_j - Q_j + vacuum occupation, which must be 0 or 1), so a
physical state is labelled by its link configuration alone. The basis is
enumerated site by site: at each site the outgoing horizontal and vertical
links are branched, and configurations whose implied occupation is not 0 or 1
are discarded immediately. Every basis label is the full qubit bit string
(links and matter), i.e. the Qiskit statevector index, so sector vectors embed
directly into circuit statevectors.

The 4x3 lattice of Joshi et al. (29 qubits) has a sector of a few thousand
states and is solved exactly in milliseconds; the 5x4 lattice (51 qubits) has
order 10^7 states with the default corner charges and is treated with the
dynamics routines only when the machine allows.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from QuantumFields.gauge_field import FLIP_FROM, FLIP_TO
from QuantumFields.matter_field import MatterGaussLaw, vacuum_occupation
from sector import GaugeSector

CHUNK = 1 << 22                 # labels expanded at once during enumeration
HOP_FROM = (0, 1, 1)            # (n_start, link, n_end) bits mapped to HOP_TO by the hopping
HOP_TO = (1, 0, 0)


class MatterSector:
    """Gauge-invariant subspace of ScalarQEDModel on an OpenLattice."""

    def __init__(self, lattice, static_charges: dict = None, max_dim: int = 20_000_000):
        if lattice.n_qubits > 62:
            raise ValueError("MatterSector stores labels in one int64 word (<= 62 qubits)")
        self.lattice = lattice
        self.gauss = MatterGaussLaw(lattice, static_charges)
        self.max_dim = max_dim
        self.basis = self._enumerate()
        self.dim = len(self.basis)
        self._flips = None

    # ------------------------------------------------------------------
    # Enumeration
    # ------------------------------------------------------------------

    def _bit(self, labels, qubit):
        return (labels >> np.int64(qubit)) & 1

    def _enumerate(self) -> np.ndarray:
        lat = self.lattice
        labels = np.zeros(1, dtype=np.int64)
        for j in range(lat.height):
            for i in range(lat.width):
                new_links = []
                if i < lat.width - 1:
                    new_links.append(lat.horizontal_link(i, j))
                if j < lat.height - 1:
                    new_links.append(lat.vertical_link(i, j))
                pieces = []
                step = max(1, CHUNK >> len(new_links))
                for lo in range(0, len(labels), step):
                    chunk = labels[lo:lo + step]
                    for q in new_links:                      # branch the new links
                        chunk = np.concatenate([chunk, chunk | (np.int64(1) << np.int64(q))])
                    div = np.full(len(chunk), lat.boundary_divergence(i, j))
                    for qubit, sign in lat.site_links(i, j):
                        div += sign * (0.5 - self._bit(chunk, qubit))
                    n = div - self.gauss.charge(i, j) + vacuum_occupation(i, j)
                    ok = (n == 0.0) | (n == 1.0)
                    chunk = chunk[ok]
                    chunk |= (n[ok] == 1.0).astype(np.int64) << np.int64(lat.matter_qubit(i, j))
                    pieces.append(chunk)
                labels = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int64)
                if len(labels) > self.max_dim:
                    raise MemoryError(f"matter sector exceeds max_dim={self.max_dim:,} "
                                      f"after site ({i}, {j})")
        return np.sort(labels)

    def index_of(self, bits: int) -> int:
        k = int(np.searchsorted(self.basis, np.int64(bits)))
        if k >= self.dim or self.basis[k] != bits:
            raise ValueError("configuration is not in the gauge-invariant sector")
        return k

    def _lookup(self, labels) -> np.ndarray:
        pos = np.minimum(np.searchsorted(self.basis, labels), self.dim - 1)
        return np.where(self.basis[pos] == labels, pos, -1)

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------

    def _pattern_flips(self, qubits, a, b):
        """Basis states in configuration a or b on `qubits`, and their partners."""
        bits = [self._bit(self.basis, q) for q in qubits]
        in_a = np.all([bq == v for bq, v in zip(bits, a)], axis=0)
        in_b = np.all([bq == v for bq, v in zip(bits, b)], axis=0)
        src = np.nonzero(in_a | in_b)[0]
        mask = np.int64(sum(1 << q for q, x, y in zip(qubits, a, b) if x != y))
        tgt = self._lookup(self.basis[src] ^ mask)
        if len(src) and tgt.min() < 0:
            raise RuntimeError("a gauge-invariant term left the sector")
        return src, tgt

    def flip_terms(self) -> dict:
        """Off-diagonal terms as {('hop', link) | ('plaq', p): (src, tgt)} (cached).

        Each term h is a sum of disjoint two-level flips, h|src_k> = |tgt_k>, so
        h^3 = h and exp(-i theta h) acts in closed form (dynamics.SectorTrotter).
        """
        if self._flips is None:
            lat = self.lattice
            flips = {}
            for (link, a, b, _) in lat.hopping_terms():
                qubits = (lat.matter_qubit(*a), link, lat.matter_qubit(*b))
                flips[('hop', link)] = self._pattern_flips(qubits, HOP_FROM, HOP_TO)
            for p, plaq in enumerate(lat.get_plaquettes()):
                flips[('plaq', p)] = self._pattern_flips(plaq, FLIP_FROM, FLIP_TO)
            self._flips = flips
        return self._flips

    def term_coefficients(self, kappa, J) -> dict:
        coef = {('hop', link): -kappa * s for (link, _, _, s) in self.lattice.hopping_terms()}
        coef.update({('plaq', p): J for p in range(self.lattice.n_plaquettes)})
        return coef

    def occupations(self) -> np.ndarray:
        """(dim, n_sites) matter occupations n_j of every basis state."""
        lat = self.lattice
        return np.array([self._bit(self.basis, lat.matter_qubit(*lat.site_coords(s)))
                         for s in range(lat.n_sites)]).T.astype(float)

    def link_sz(self) -> np.ndarray:
        """(dim, n_links) S^z of every link in every basis state."""
        return np.array([0.5 - self._bit(self.basis, l)
                         for l in range(self.lattice.n_links)]).T.astype(float)

    def diagonal(self, m=0.0, g=0.0) -> np.ndarray:
        lat = self.lattice
        signs = np.array([lat.parity(*lat.site_coords(s)) for s in range(lat.n_sites)])
        return m * (self.occupations() @ signs) - g * self.link_sz().sum(axis=1)

    def hamiltonian(self, kappa=1.0, m=0.0, g=0.0, J=0.0) -> sp.csr_matrix:
        rows, cols, vals = [], [], []
        coefficients = self.term_coefficients(kappa, J)
        for key, (src, tgt) in self.flip_terms().items():
            c = coefficients[key]
            if c != 0.0 and len(src):
                rows.append(tgt), cols.append(src), vals.append(np.full(len(src), c))
        H = sp.diags(self.diagonal(m, g)).tocsr()
        if rows:
            H = H + sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows),
                                   np.concatenate(cols))), shape=(self.dim, self.dim)).tocsr()
        return H.tocsr()

    # ------------------------------------------------------------------
    # States and observables
    # ------------------------------------------------------------------

    def product_state(self, bits: int) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=complex)
        vec[self.index_of(bits)] = 1.0
        return vec

    def ground_state(self, kappa=1.0, m=0.0, g=0.0, J=0.0, seed=0) -> dict:
        Hm = self.hamiltonian(kappa, m, g, J)
        w, v = GaugeSector.lowest(Hm, 2, seed)
        return {'energy': float(w[0]), 'gap': float(w[1] - w[0]) if len(w) > 1 else np.nan,
                'state': v[:, 0], 'hamiltonian': Hm}

    def particle_density(self, vec) -> np.ndarray:
        """<|rho_j|> per site: probability of a particle (even) or antiparticle (odd)."""
        lat = self.lattice
        vac = np.array([vacuum_occupation(*lat.site_coords(s)) for s in range(lat.n_sites)])
        return (np.abs(vec) ** 2) @ np.abs(self.occupations() - vac[None, :])

    def electric_field(self, vec) -> np.ndarray:
        return (np.abs(vec) ** 2) @ self.link_sz()

    def evolve(self, vec, Hm, times) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        if len(times) == 1:
            return expm_multiply(-1j * times[0] * Hm, vec.astype(complex))[None, :]
        return expm_multiply(-1j * Hm, vec.astype(complex), start=times[0],
                             stop=times[-1], num=len(times), endpoint=True)

    def embed(self, vec) -> np.ndarray:
        if self.lattice.n_qubits > 26:
            raise MemoryError("full statevector too large to embed")
        full = np.zeros(2 ** self.lattice.n_qubits, dtype=complex)
        full[self.basis] = vec
        return full
