"""Pure-gauge U(1) quantum link model: Hamiltonian and Gauss's law.

Encoding (spin-1/2 truncation):
    electric field      S^z = (1/2) Z
    gauge connection    S^+ = (X + iY)/2,  S^- = (X - iY)/2

Pauli labels are built with the index transform q -> N_q - 1 - q so that
qubit q lands in the correct position under Qiskit's little-endian ordering.
Without this, [G_j, H] is spuriously non-zero even when the physics is right.
"""

from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
import numpy as np


class QuantumLinkModel:
    # Eight-Pauli-string expansion of the plaquette operator U_p + U_p^dagger.
    _PLAQ_PAULIS = [
        ("XXXX", +1.0),
        ("XXYY", -1.0),
        ("XYXY", +1.0),
        ("XYYX", +1.0),
        ("YXXY", +1.0),
        ("YXYX", +1.0),
        ("YYXX", -1.0),
        ("YYYY", +1.0),
    ]

    def __init__(self, lattice, g_squared: float = 1.0, J: float = 1.0,
                 gauss_penalty: float = 0.0):
        self.lattice = lattice
        self.g_squared = g_squared
        self.J = J
        self.gauss_penalty = gauss_penalty
        self.n_qubits = lattice.n_qubits

    def build_electric_term(self) -> SparsePauliOp:
        """H_E = (g^2/2) sum (S^z)^2.

        Under the spin-1/2 truncation (S^z)^2 = (1/4) I, so this collapses to a
        constant shift g^2/8 * N_links. (A linear S^z 'string-tension' term would
        be added for non-trivial electric dynamics; not included in pure gauge.)
        """
        shift = self.g_squared / 8.0 * self.lattice.n_links_total
        return SparsePauliOp(['I' * self.n_qubits], coeffs=[shift])

    def build_magnetic_term(self) -> SparsePauliOp:
        """H_B = (J/8) sum_plaquettes (U_p + U_p^dagger), eight Paulis per plaquette."""
        if self.J == 0.0:
            return SparsePauliOp(['I' * self.n_qubits], coeffs=[0.0])

        plaq_coeff = self.J / 8.0
        pauli_strings = []
        coefficients = []
        for plaquette in self.lattice.get_plaquettes():
            for pauli_str, sign in self._PLAQ_PAULIS:
                pauli_strings.append(self._pauli_string_on_qubits(plaquette, pauli_str))
                coefficients.append(sign * plaq_coeff)
        return SparsePauliOp(pauli_strings, coeffs=coefficients)

    def build_gauss_penalty(self) -> SparsePauliOp:
        """lambda * sum_j G_j^2, enforcing Gauss's law via a quadratic penalty."""
        if self.gauss_penalty == 0.0:
            return SparsePauliOp(['I' * self.n_qubits], coeffs=[0.0])
        return GaussLaw(self.lattice).build_penalty_hamiltonian(self.gauss_penalty)

    def build_hamiltonian(self) -> SparsePauliOp:
        H = self.build_electric_term() + self.build_magnetic_term()
        if self.gauss_penalty > 0:
            H = H + self.build_gauss_penalty()
        return H.simplify()

    def _pauli_string_on_qubits(self, qubits: list, pauli_chars: str) -> str:
        full = ['I'] * self.n_qubits
        for q, p in zip(qubits, pauli_chars):
            full[self.n_qubits - 1 - q] = p
        return ''.join(full)


class GaussLaw:
    """Gauss-law generators G_j and the associated penalty Hamiltonian.

    Interior site (i, j): G_j = sum of signed S^z on the four incident links
    (right +, left -, up +, down -). Boundary rows (j = 0, H-1) are the plates
    and carry no generator.
    """

    def __init__(self, lattice):
        self.lattice = lattice
        self.n_qubits = lattice.n_qubits
        self.n_sites = lattice.n_sites

    def build_gauss_operators(self) -> list:
        """One SparsePauliOp per site; None for the boundary (plate) rows."""
        gauss_ops = []
        for site_idx in range(self.n_sites):
            i, j = self.lattice.site_to_coords(site_idx)

            if j == 0 or j == self.lattice.height - 1:
                gauss_ops.append(None)
                continue

            neighbours = self.lattice.get_neighbours(i, j)
            pauli_strings = []
            coefficients = []

            if "right" in neighbours:
                link_idx = self.lattice.get_horizontal_link_index(i, j)
                pauli_strings.append(self._pauli_z_on_qubit(link_idx))
                coefficients.append(+0.5)

            if "left" in neighbours:
                i_left, j_left = neighbours["left"]
                link_idx = self.lattice.get_horizontal_link_index(i_left, j_left)
                pauli_strings.append(self._pauli_z_on_qubit(link_idx))
                coefficients.append(-0.5)

            if "up" in neighbours:
                link_idx = self.lattice.get_vertical_link_index(i, j)
                pauli_strings.append(self._pauli_z_on_qubit(link_idx))
                coefficients.append(+0.5)

            if "down" in neighbours:
                _, j_down = neighbours["down"]
                link_idx = self.lattice.get_vertical_link_index(i, j_down)
                pauli_strings.append(self._pauli_z_on_qubit(link_idx))
                coefficients.append(-0.5)

            G_site = SparsePauliOp(pauli_strings, coeffs=coefficients).simplify()
            gauss_ops.append(G_site)

        return gauss_ops

    def build_penalty_hamiltonian(self, lam: float) -> SparsePauliOp:
        """lambda * sum_j G_j^2 over all interior sites."""
        penalty_terms = [
            (G_op @ G_op).simplify()
            for G_op in self.build_gauss_operators() if G_op is not None
        ]
        if not penalty_terms:
            return SparsePauliOp(['I' * self.n_qubits], coeffs=[0.0])

        H_penalty = lam * sum(penalty_terms[1:], penalty_terms[0])
        return H_penalty.simplify()

    def get_initial_circuit(self) -> QuantumCircuit:
        """Equal superposition over link qubits (Hadamard on every link)."""
        qc = QuantumCircuit(self.n_qubits)
        for link_q in self.lattice.get_electric_qubits():
            qc.h(link_q)
        return qc

    def verify(self, sv: np.ndarray, tol: float = 1e-6, verbose: bool = True) -> bool:
        """Check max_j ||G_j |psi>|| <= tol, i.e. the state is gauge invariant."""
        max_viol = 0.0
        violations = []

        for idx, G_op in enumerate(self.build_gauss_operators()):
            if G_op is None:
                continue
            viol = np.linalg.norm(G_op.to_matrix(sparse=True) @ sv)
            max_viol = max(max_viol, viol)
            if viol > tol:
                i, j = self.lattice.site_to_coords(idx)
                violations.append((i, j, viol))

        if verbose:
            if not violations:
                print(f"Gauss law satisfied (max violation = {max_viol:.2e})")
            else:
                print(f"[!] Gauss law violated at {len(violations)} sites "
                      f"(max = {max_viol:.2e})")
                for i, j, v in violations:
                    print(f"   Site ({i},{j}): {v:.2e}")
        return len(violations) == 0

    def _pauli_z_on_qubit(self, qubit_index: int) -> str:
        parts = ['I'] * self.n_qubits
        parts[self.n_qubits - 1 - qubit_index] = 'Z'
        return ''.join(parts)
