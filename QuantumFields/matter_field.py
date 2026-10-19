"""U(1) quantum link model coupled to hardcore-boson matter (Joshi et al. 2026, Eq. 1).

This is the model behind the structured Trotter decomposition that the thesis
adapts to the pure-gauge cylinder. It is implemented here so that the circuits
of this package can be benchmarked on exactly the problem Joshi et al. ran on
hardware (OpenLattice geometry, open boundaries).

    H = -kappa sum_<j,k> s_jk (phi^dag_j S^+_jk phi_k + h.c.)       hopping (gauge-matter)
        + m sum_j (-1)^(i+j) n_j                                     staggered mass
        - g sum_l S^z_l                                              electric (string tension)
        + J sum_p (U_p + U_p^dag)                                    plaquette

Encoding. Link qubits as in the pure-gauge model (|0> = S^z = +1/2). Matter is
a hardcore boson per site with |1> = occupied: n_j = (1 - Z_j)/2,
phi^dag = |1><0| = (X - iY)/2. Hardcore bosons need no Jordan-Wigner string, so
every hopping term acts on three qubits (start site, link, end site).

Staggered vacuum. The mass term favours n = 0 on even sites and n = 1 on odd
sites, so the vacuum has the odd sites filled. The dynamical charge is
    rho_j = n_j - (1 - (-1)^(i+j))/2,
which is 0 in the vacuum, +1 for a particle on an even site and -1 for a hole on
an odd site. A particle-antiparticle pair costs 2m.

Gauss's law.
    G_j = div E_j - rho_j - Q_j = 0,
where div E_j sums S^z over the links at j (+1 leaving, -1 entering) plus the
boundary contribution of the missing links, which are fixed at S^z = +1/2
(OpenLattice.boundary_divergence). Q_j are static charges. Every hopping term
commutes with every G_j: phi^dag_j S^+_jk phi_k raises rho_j and div E_j by one
and lowers rho_k and div E_k by one.

String state. With static charges Q = -1 at the corner (0, 0) and Q = +1 at
(W-1, H-1), the configuration with all matter in the vacuum, S^z = -1/2 on a
staircase path of links between the corners and S^z = +1/2 elsewhere is
physical. This is the initial state of Joshi et al. The string of length ell
costs ell * g above the vacuum, so it can break into a pair at the resonance
2m = ell g; at m = 3, g = 6 single-link segments are resonant.
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from QuantumFields.gauge_field import PLAQ_PAULIS, pauli_label, zero_operator


def vacuum_occupation(i: int, j: int) -> int:
    """Occupation of site (i, j) in the staggered vacuum: 0 on even, 1 on odd sites."""
    return (i + j) % 2


def corner_charges(lattice) -> dict:
    """Static charges -1 at (0, 0) and +1 at (W-1, H-1), as used by Joshi et al."""
    end = (lattice.width - 1, lattice.height - 1)
    if (end[0] + end[1]) % 2 == 0:
        raise ValueError("the far corner must be an odd site (W + H must be odd); "
                         "pass static_charges explicitly for this lattice")
    return {(0, 0): -1, end: +1}


def staircase_string(lattice) -> list:
    """Links of the diagonal staircase from (0, 0) to (W-1, H-1).

    Alternates +x and +y steps while both are possible, then finishes along
    the remaining direction.
    """
    i = j = 0
    links = []
    step_x = True
    while (i, j) != (lattice.width - 1, lattice.height - 1):
        can_x, can_y = i < lattice.width - 1, j < lattice.height - 1
        if can_x and (step_x or not can_y):
            links.append(lattice.horizontal_link(i, j))
            i += 1
        else:
            links.append(lattice.vertical_link(i, j))
            j += 1
        step_x = not step_x
    return links


def string_state_bits(lattice, string_links=None) -> int:
    """Qubit bit string of the string initial state (Qiskit statevector index)."""
    bits = 0
    for link in (staircase_string(lattice) if string_links is None else string_links):
        bits |= 1 << link
    for site in range(lattice.n_sites):
        i, j = lattice.site_coords(site)
        if vacuum_occupation(i, j):
            bits |= 1 << lattice.matter_qubit(i, j)
    return bits


def _single(n_qubits, qubit, matrix_name):
    """One-qubit operator on `qubit` as a SparsePauliOp."""
    ops = {
        "Z": [("Z", 1.0)],
        "raise_link": [("X", 0.5), ("Y", 0.5j)],         # S^+   = |0><1| = (X + iY)/2
        "create": [("X", 0.5), ("Y", -0.5j)],            # phi^d = |1><0| = (X - iY)/2
        "annihilate": [("X", 0.5), ("Y", 0.5j)],         # phi   = |0><1| = (X + iY)/2
    }[matrix_name]
    return SparsePauliOp([pauli_label(n_qubits, [qubit], p) for p, _ in ops],
                         coeffs=[c for _, c in ops])


class MatterGaussLaw:
    """Gauss generators G_j = div E_j - rho_j - Q_j of the matter-coupled model."""

    def __init__(self, lattice, static_charges: dict = None):
        self.lattice = lattice
        self.charges = corner_charges(lattice) if static_charges is None \
            else dict(static_charges)

    def charge(self, i: int, j: int) -> float:
        return float(self.charges.get((i, j), 0.0))

    def generator(self, i: int, j: int) -> SparsePauliOp:
        lat = self.lattice
        n = lat.n_qubits
        labels, coeffs = [], []
        for qubit, sign in lat.site_links(i, j):              # div E: S^z = Z/2
            labels.append(pauli_label(n, [qubit], 'Z'))
            coeffs.append(0.5 * sign)
        # -rho_j = -(1 - Z_j)/2 + vacuum occupation
        labels.append(pauli_label(n, [lat.matter_qubit(i, j)], 'Z'))
        coeffs.append(0.5)
        const = (lat.boundary_divergence(i, j) - 0.5 + vacuum_occupation(i, j)
                 - self.charge(i, j))
        labels.append('I' * n)
        coeffs.append(const)
        return SparsePauliOp(labels, coeffs=coeffs).simplify()

    def build_gauss_operators(self) -> list:
        lat = self.lattice
        return [self.generator(*lat.site_coords(s)) for s in range(lat.n_sites)]

    def max_violation(self, sv: np.ndarray) -> float:
        return max(float(np.linalg.norm(G.to_matrix(sparse=True) @ sv))
                   for G in self.build_gauss_operators())


class ScalarQEDModel:
    """Qubit Hamiltonian of the matter-coupled quantum link model on an OpenLattice."""

    def __init__(self, lattice, kappa: float = 1.0, m: float = 0.0, g: float = 0.0,
                 J: float = 0.0, static_charges: dict = None):
        self.lattice = lattice
        self.kappa, self.m, self.g, self.J = float(kappa), float(m), float(g), float(J)
        self.n_qubits = lattice.n_qubits
        self.gauss = MatterGaussLaw(lattice, static_charges)

    def hopping_operator(self, term) -> SparsePauliOp:
        """phi^dag_a S^+_l phi_b + h.c. for one hopping term (without kappa, s_l)."""
        link, a, b, _ = term
        n, lat = self.n_qubits, self.lattice
        op = (_single(n, lat.matter_qubit(*a), "create")
              @ _single(n, link, "raise_link")
              @ _single(n, lat.matter_qubit(*b), "annihilate"))
        return (op + op.adjoint()).simplify()

    def build_hopping_term(self) -> SparsePauliOp:
        if self.kappa == 0.0:
            return zero_operator(self.n_qubits)
        ops = [-self.kappa * term[3] * self.hopping_operator(term)
               for term in self.lattice.hopping_terms()]
        return sum(ops[1:], ops[0]).simplify()

    def build_mass_term(self) -> SparsePauliOp:
        """m sum_j (-1)^(i+j) (1 - Z_j)/2."""
        if self.m == 0.0:
            return zero_operator(self.n_qubits)
        lat, n = self.lattice, self.n_qubits
        labels, coeffs = ['I' * n], [0.0]
        for s in range(lat.n_sites):
            i, j = lat.site_coords(s)
            sign = lat.parity(i, j)
            coeffs[0] += 0.5 * self.m * sign
            labels.append(pauli_label(n, [lat.matter_qubit(i, j)], 'Z'))
            coeffs.append(-0.5 * self.m * sign)
        return SparsePauliOp(labels, coeffs=coeffs).simplify()

    def build_electric_term(self) -> SparsePauliOp:
        """-g sum_l S^z_l = -(g/2) sum_l Z_l."""
        if self.g == 0.0:
            return zero_operator(self.n_qubits)
        n = self.n_qubits
        return SparsePauliOp([pauli_label(n, [l], 'Z') for l in range(self.lattice.n_links)],
                             coeffs=[-0.5 * self.g] * self.lattice.n_links)

    def build_magnetic_term(self) -> SparsePauliOp:
        if self.J == 0.0 or self.lattice.n_plaquettes == 0:
            return zero_operator(self.n_qubits)
        labels, coeffs = [], []
        for plaq in self.lattice.get_plaquettes():
            for p, s in PLAQ_PAULIS:
                labels.append(pauli_label(self.n_qubits, plaq, p))
                coeffs.append(self.J * s / 8.0)
        return SparsePauliOp(labels, coeffs=coeffs).simplify()

    def build_hamiltonian(self) -> SparsePauliOp:
        return (self.build_hopping_term() + self.build_mass_term()
                + self.build_electric_term() + self.build_magnetic_term()).simplify()
