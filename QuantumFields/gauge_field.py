"""Pure-gauge U(1) quantum link model: Hamiltonian and Gauss's law.

Hamiltonian (Joshi et al. 2026, Eq. 1 with the matter sector removed, plus an
optional Rokhsar-Kivelson term):

    H = -g sum_l s_l S^z_l  +  J sum_p (U_p + U_p^dag)  +  lam_RK sum_p (U_p + U_p^dag)^2
        \\______ H_E _____/     \\________ H_B ________/     \\____________ H_RK ___________/

Encoding (spin-1/2 truncation, one qubit per link, |0> = S^z = +1/2):
    electric field      S^z = Z/2
    gauge connection    S^+ = (X + iY)/2 = |0><1|,  S^- = (X - iY)/2
    plaquette           U_p = S^+_b S^+_r S^-_t S^-_l   (bottom, right, top, left)

Electric term. The quadratic energy (g^2/2) sum (S^z)^2 is a constant for
spin 1/2, so Joshi et al. add a *linear* term. Two sign patterns s_l:

  "uniform"    s_l = +1. This is Joshi's term. With matter it produces string
               tension. In the PURE-GAUGE model it commutes with every plaquette
               operator, [sum_l S^z_l, U_p] = (1 + 1 - 1 - 1) U_p = 0
               (proofs/GaugeInvariance.lean), so it is a conserved chemical
               potential for total flux and generates no dynamics of its own.
  "staggered"  s_l = -(-1)^(i+j) on horizontal and +(-1)^(i+j) on vertical
               links. [H_E, U_p] = -4 g (-1)^(i+j) U_p is non-zero on every
               plaquette, giving genuine electric dynamics. Translation
               invariant along x only for even W.

Every term is diagonal in S^z or built from U_p, hence commutes with every
Gauss generator G_j, including the generators on the plate rows.

Gauss's law and the plates.
    G_j = sum_{links at j} (+-) S^z  -  q_j,       physical states: G_j |psi> = 0.
Interior sites have four links and q_j = 0. Plate sites have three links, so
sum(+-)S^z is half-integer there and q_j must be +-1/2 or +-3/2: a spin-1/2
plate site necessarily carries a static charge. Two plate models:

  plates="free"   no constraint on plate rows. The plate charges are conserved
                  but unconstrained, so the Hilbert space is the direct sum of
                  all static plate-charge sectors and the ground state picks the
                  energetically best induced-charge pattern (ideal conductor).
  plates="fixed"  G_j = 0 enforced on plate rows with prescribed charges. The
                  default pattern is staggered, q_(i,j) = (1/2) (-1)^(i+j),
                  which needs even W; pass `plate_charges` for anything else.

Pauli labels use the index transform q -> N_q - 1 - q so that qubit q lands in
the right position under Qiskit's little-endian ordering.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


# Eight-Pauli-string expansion  U_p + U_p^dag = (1/8) sum_k sign_k P_k
# (proofs/PlaquettePauli.lean checks this identity entry by entry).
PLAQ_PAULIS = [
    ("XXXX", +1.0),
    ("XXYY", -1.0),
    ("XYXY", +1.0),
    ("XYYX", +1.0),
    ("YXXY", +1.0),
    ("YXYX", +1.0),
    ("YYXX", -1.0),
    ("YYYY", +1.0),
]

# (U_p + U_p^dag)^2 = projector onto flippable plaquettes
#                   = (1/8) [1 + Z_b Z_r + Z_t Z_l - Z_b Z_t - Z_b Z_l - Z_r Z_t - Z_r Z_l + Z_b Z_r Z_t Z_l]
RK_PAULIS = [
    ("IIII", +1.0),
    ("ZZII", +1.0),
    ("IIZZ", +1.0),
    ("ZIZI", -1.0),
    ("ZIIZ", -1.0),
    ("IZZI", -1.0),
    ("IZIZ", -1.0),
    ("ZZZZ", +1.0),
]

FIELD_PATTERNS = ("uniform", "staggered")
PLATE_MODES = ("free", "fixed")


def pauli_label(n_qubits: int, qubits, paulis) -> str:
    """Full-register Pauli label with `paulis[k]` acting on `qubits[k]`."""
    full = ['I'] * n_qubits
    for q, p in zip(qubits, paulis):
        full[n_qubits - 1 - q] = p
    return ''.join(full)


def _zero_op(n_qubits: int) -> SparsePauliOp:
    return SparsePauliOp(['I' * n_qubits], coeffs=[0.0])


def field_signs(lattice, pattern: str = "uniform") -> np.ndarray:
    """Sign s_l of the linear electric term on each link qubit."""
    if pattern not in FIELD_PATTERNS:
        raise ValueError(f"field pattern must be one of {FIELD_PATTERNS}")
    signs = np.ones(lattice.n_qubits)
    if pattern == "staggered":
        for q in range(lattice.n_qubits):
            i, j, direction = lattice.link_info(q)
            parity = (-1) ** (i + j)
            signs[q] = -parity if direction == 'x' else parity
    return signs


def staggered_plate_charges(lattice) -> dict:
    """q_(i,j) = (1/2)(-1)^(i+j) on both plate rows (requires even W)."""
    if lattice.width % 2:
        raise ValueError("staggered plate charges need even W; on odd-W "
                         "cylinders use plates='free' or pass plate_charges")
    return {(i, j): 0.5 * (-1) ** (i + j)
            for j in (0, lattice.height - 1) for i in range(lattice.width)}


class GaussLaw:
    """Gauss-law generators G_j and the associated penalty Hamiltonian."""

    def __init__(self, lattice, plates: str = "free", plate_charges: dict = None):
        if plates not in PLATE_MODES:
            raise ValueError(f"plates must be one of {PLATE_MODES}")
        self.lattice = lattice
        self.n_qubits = lattice.n_qubits
        self.n_sites = lattice.n_sites
        self.plates = plates
        self.charges = {}
        if plates == "fixed":
            self.charges = dict(plate_charges) if plate_charges is not None \
                else staggered_plate_charges(lattice)
            for (i, j), q in self.charges.items():
                n_links = len(lattice.site_links(i, j))
                if (2 * q - n_links) % 2:
                    raise ValueError(
                        f"charge {q} at site ({i},{j}) is incompatible with "
                        f"{n_links} spin-1/2 links")

    def charge(self, i: int, j: int) -> float:
        return float(self.charges.get((i % self.lattice.width, j), 0.0))

    def is_constrained(self, i: int, j: int) -> bool:
        return self.plates == "fixed" or not self.lattice.is_plate_site(i, j)

    def divergence_operator(self, i: int, j: int) -> SparsePauliOp:
        """Lattice divergence of E at site (i, j), without the charge."""
        labels, coeffs = [], []
        for qubit, sign in self.lattice.site_links(i, j):
            labels.append(pauli_label(self.n_qubits, [qubit], 'Z'))
            coeffs.append(0.5 * sign)
        return SparsePauliOp(labels, coeffs=coeffs).simplify()

    def generator(self, i: int, j: int) -> SparsePauliOp:
        """G_j = div E - q_j for any site, constrained or not."""
        op = self.divergence_operator(i, j)
        q = self.charge(i, j)
        if q != 0.0:
            op = (op + SparsePauliOp(['I' * self.n_qubits], coeffs=[-q])).simplify()
        return op

    def build_gauss_operators(self) -> list:
        """One SparsePauliOp per site (site order); None where unconstrained."""
        ops = []
        for site_idx in range(self.n_sites):
            i, j = self.lattice.site_to_coords(site_idx)
            ops.append(self.generator(i, j) if self.is_constrained(i, j) else None)
        return ops

    def build_penalty_hamiltonian(self, lam: float) -> SparsePauliOp:
        """lambda * sum_j G_j^2 over all constrained sites."""
        terms = [(G @ G).simplify() for G in self.build_gauss_operators()
                 if G is not None]
        if not terms or lam == 0.0:
            return _zero_op(self.n_qubits)
        return (lam * sum(terms[1:], terms[0])).simplify()

    def get_initial_circuit(self) -> QuantumCircuit:
        """Equal superposition over link qubits (Hadamard on every link)."""
        qc = QuantumCircuit(self.n_qubits)
        for link_q in self.lattice.get_electric_qubits():
            qc.h(link_q)
        return qc

    def max_violation(self, sv: np.ndarray) -> float:
        """max_j || G_j |psi> || over constrained sites."""
        worst = 0.0
        for G in self.build_gauss_operators():
            if G is not None:
                worst = max(worst, float(np.linalg.norm(G.to_matrix(sparse=True) @ sv)))
        return worst

    def verify(self, sv: np.ndarray, tol: float = 1e-6, verbose: bool = True) -> bool:
        """Check max_j ||G_j |psi>|| <= tol, i.e. the state is gauge invariant."""
        violations = []
        worst = 0.0
        for idx, G in enumerate(self.build_gauss_operators()):
            if G is None:
                continue
            viol = float(np.linalg.norm(G.to_matrix(sparse=True) @ sv))
            worst = max(worst, viol)
            if viol > tol:
                violations.append((*self.lattice.site_to_coords(idx), viol))
        if verbose:
            if not violations:
                print(f"Gauss law satisfied (max violation = {worst:.2e})")
            else:
                print(f"[!] Gauss law violated at {len(violations)} sites "
                      f"(max = {worst:.2e})")
                for i, j, v in violations:
                    print(f"   Site ({i},{j}): {v:.2e}")
        return not violations


class QuantumLinkModel:
    """Qubit Hamiltonian of the pure-gauge spin-1/2 U(1) quantum link model."""

    _PLAQ_PAULIS = PLAQ_PAULIS          # kept for backwards compatibility

    def __init__(self, lattice, g: float = 0.0, J: float = 1.0, rk: float = 0.0,
                 gauss_penalty: float = 0.0, field_pattern: str = "uniform",
                 plates: str = "free", plate_charges: dict = None):
        self.lattice = lattice
        self.g = float(g)
        self.J = float(J)
        self.rk = float(rk)
        self.gauss_penalty = float(gauss_penalty)
        self.field_pattern = field_pattern
        self.n_qubits = lattice.n_qubits
        self.gauss = GaussLaw(lattice, plates=plates, plate_charges=plate_charges)
        self.field_signs = field_signs(lattice, field_pattern)

    # ---- individual terms ------------------------------------------------

    def build_electric_term(self) -> SparsePauliOp:
        """H_E = -g sum_l s_l S^z_l = -(g/2) sum_l s_l Z_l."""
        if self.g == 0.0:
            return _zero_op(self.n_qubits)
        labels = [pauli_label(self.n_qubits, [q], 'Z')
                  for q in self.lattice.get_electric_qubits()]
        coeffs = [-0.5 * self.g * self.field_signs[q]
                  for q in self.lattice.get_electric_qubits()]
        return SparsePauliOp(labels, coeffs=coeffs)

    def plaquette_operator(self, plaquette) -> SparsePauliOp:
        """U_p + U_p^dag on one plaquette (norm 1, eigenvalues 0, +-1)."""
        labels = [pauli_label(self.n_qubits, plaquette, p) for p, _ in PLAQ_PAULIS]
        coeffs = [s / 8.0 for _, s in PLAQ_PAULIS]
        return SparsePauliOp(labels, coeffs=coeffs)

    def flippability_operator(self, plaquette) -> SparsePauliOp:
        """(U_p + U_p^dag)^2: projector onto the two flippable configurations."""
        labels = [pauli_label(self.n_qubits, plaquette, p) for p, _ in RK_PAULIS]
        coeffs = [s / 8.0 for _, s in RK_PAULIS]
        return SparsePauliOp(labels, coeffs=coeffs)

    def plaquette_operators(self) -> list:
        return [self.plaquette_operator(p) for p in self.lattice.get_plaquettes()]

    def build_magnetic_term(self) -> SparsePauliOp:
        """H_B = J sum_p (U_p + U_p^dag): eight Pauli strings per plaquette."""
        if self.J == 0.0:
            return _zero_op(self.n_qubits)
        ops = self.plaquette_operators()
        return (self.J * sum(ops[1:], ops[0])).simplify()

    def build_rk_term(self) -> SparsePauliOp:
        """H_RK = lam_RK sum_p (U_p + U_p^dag)^2 (diagonal, counts flippable plaquettes)."""
        if self.rk == 0.0:
            return _zero_op(self.n_qubits)
        ops = [self.flippability_operator(p) for p in self.lattice.get_plaquettes()]
        return (self.rk * sum(ops[1:], ops[0])).simplify()

    def build_gauss_penalty(self) -> SparsePauliOp:
        return self.gauss.build_penalty_hamiltonian(self.gauss_penalty)

    # ---- assembled Hamiltonians -----------------------------------------------

    def build_physical_hamiltonian(self) -> SparsePauliOp:
        """H_phys = H_E + H_B + H_RK (no penalty)."""
        return (self.build_electric_term() + self.build_magnetic_term()
                + self.build_rk_term()).simplify()

    def build_hamiltonian(self) -> SparsePauliOp:
        """H_phys plus the Gauss-law penalty when gauss_penalty > 0."""
        H = self.build_physical_hamiltonian()
        if self.gauss_penalty > 0:
            H = (H + self.build_gauss_penalty()).simplify()
        return H

    def one_norm(self) -> float:
        """lambda_H = sum of term norms at plaquette granularity (qDRIFT weight)."""
        n_p = self.lattice.n_plaquettes
        return (abs(self.J) * n_p + abs(self.rk) * n_p
                + 0.5 * abs(self.g) * self.n_qubits)

    # backwards-compatible helper
    def _pauli_string_on_qubits(self, qubits, pauli_chars) -> str:
        return pauli_label(self.n_qubits, qubits, pauli_chars)
