from qiskit.quantum_info import SparsePauliOp
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation

class QuantumLinkModel:
    _PLAQ_PAULIS = [
        ("XXXX", +1.0),
        ("XXYY", -1.0),
        ("XYXY", +1.0),   # FIXED: was -1.0
        ("XYYX", +1.0),
        ("YXXY", +1.0),
        ("YXYX", +1.0),   # FIXED: was -1.0
        ("YYXX", -1.0),
        ("YYYY", +1.0),
    ]

    def __init__(self, lattice, coupling: float, gauss_penalty: float = 0.0):
        self.lattice = lattice
        self.g_squared = coupling
        self.gauss_penalty = gauss_penalty
        self.n_qubits = lattice.n_qubits

    def build_electric_term(self) -> SparsePauliOp:
        # Constant shift: (g²/8) per link
        shift = self.g_squared / 8.0 * self.lattice.n_links_total
        identity_str = 'I' * self.n_qubits
        return SparsePauliOp([identity_str], coeffs=[shift])

    def build_magnetic_term(self) -> SparsePauliOp:
        coeff = -1.0 / (2.0 * self.g_squared)
        plaq_coeff = coeff / 8.0
        pauli_strings = []
        coefficients = []

        for (link1, link2, link3, link4) in self.lattice.get_plaquettes():
            qubits = [link1, link2, link3, link4]

            for pauli_str, sign in self._PLAQ_PAULIS:
                pauli_strings.append(
                    self._pauli_string_on_qubits(qubits, pauli_str)
                )
                coefficients.append(sign * plaq_coeff)

        return SparsePauliOp(pauli_strings, coeffs=coefficients)

    def build_gauss_penalty(self) -> SparsePauliOp:
        if self.gauss_penalty == 0.0:
            identity_str = 'I' * self.n_qubits
            return SparsePauliOp([identity_str], coeffs=[0.0])

        gauss = GaussLaw(self.lattice)
        gauss_ops = gauss.build_gauss_operators()

        # Build Σ G_s² as a SparsePauliOp
        penalty_terms = []

        for G_op in gauss_ops:
            if G_op is None:
                continue
            # G² = G† @ G (G is Hermitian so G² = G @ G)
            G_squared = (G_op @ G_op).simplify()
            penalty_terms.append(G_squared)

        if not penalty_terms:
            identity_str = 'I' * self.n_qubits
            return SparsePauliOp([identity_str], coeffs=[0.0])

        H_penalty = self.gauss_penalty * sum(penalty_terms[1:], penalty_terms[0])
        return H_penalty.simplify()

    def build_hamiltonian(self) -> SparsePauliOp:
        """Full Hamiltonian: H = H_E + H_B + λΣG²"""
        H = self.build_electric_term() + self.build_magnetic_term()

        if self.gauss_penalty > 0:
            H = H + self.build_gauss_penalty()

        return H.simplify()

    def _pauli_string_on_qubits(self, qubits: list, pauli_chars: str) -> str:
        """Place Pauli characters on specified qubits, I elsewhere."""
        full = ['I'] * self.n_qubits
        for q, p in zip(qubits, pauli_chars):
            full[q] = p
        return ''.join(full)
    

class GaussLaw:
    def __init__(self, lattice):
        self.lattice = lattice
        self.n_qubits = lattice.n_qubits
        self.n_sites = lattice.n_sites

    def build_gauss_operators(self) -> list:
        gauss_ops = []

        for site_idx in range(self.n_sites):
            i, j = self.lattice.site_to_coords(site_idx)

            # Skip boundary sites (Casimir plates at top and bottom)
            is_boundary = (j == 0 or j == self.lattice.height - 1)
            if is_boundary:
                gauss_ops.append(None)
                continue

            neighbours = self.lattice.get_neighbours(i, j)
            pauli_strings = []
            coefficients = []

            # Horizontal links: convention is left→right
            # Right link from (i,j) to (i+1,j): OUTGOING → +Z/2
            if "right" in neighbours:
                link_idx = self.lattice.get_horizontal_link_index(i, j)
                pauli_strings.append(self._pauli_z_on_qubit(link_idx))
                coefficients.append(+0.5)

            # Left link from (i-1,j) to (i,j): INCOMING → -Z/2
            if "left" in neighbours:
                i_left, j_left = neighbours["left"]
                link_idx = self.lattice.get_horizontal_link_index(i_left, j_left)
                pauli_strings.append(self._pauli_z_on_qubit(link_idx))
                coefficients.append(-0.5)

            # Vertical links: convention is down→up
            # Up link from (i,j) to (i,j+1): OUTGOING → +Z/2
            if "up" in neighbours:
                link_idx = self.lattice.get_vertical_link_index(i, j)
                pauli_strings.append(self._pauli_z_on_qubit(link_idx))
                coefficients.append(+0.5)

            # Down link from (i,j-1) to (i,j): INCOMING → -Z/2
            if "down" in neighbours:
                i_down, j_down = neighbours["down"]
                link_idx = self.lattice.get_vertical_link_index(i, j_down)
                pauli_strings.append(self._pauli_z_on_qubit(link_idx))
                coefficients.append(-0.5)

            if len(pauli_strings) > 0:
                G_site = SparsePauliOp(pauli_strings, coeffs=coefficients)
                gauss_ops.append(G_site)
            else:
                gauss_ops.append(None)

        return gauss_ops

    def get_initial_circuit(self, state_vector=None) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        if state_vector is not None:
            qc.append(StatePreparation(state_vector), range(self.n_qubits))
        else:
            for i in range(self.n_qubits):
                qc.h(i)
            for i in range(self.n_qubits // 2):
                qc.ry(0.8, i)
        return qc

    def build_penalty_hamiltonian(self, lam: float) -> SparsePauliOp:
        gauss_ops = self.build_gauss_operators()

        penalty_terms = []
        for G_op in gauss_ops:
            if G_op is None:
                continue
            G_squared = (G_op @ G_op).simplify()
            penalty_terms.append(G_squared)

        if not penalty_terms:
            identity_str = 'I' * self.n_qubits
            return SparsePauliOp([identity_str], coeffs=[0.0])

        H_penalty = lam * sum(penalty_terms[1:], penalty_terms[0])
        return H_penalty.simplify()

    def verify(self, sv: np.ndarray, tol: float = 1e-6, verbose: bool = True) -> bool:
        max_viol = 0.0
        violations = []

        for idx, G_op in enumerate(self.build_gauss_operators()):
            if G_op is None:
                continue

            # Check ||G_s |ψ⟩|| — should be 0 for physical states
            G_psi = G_op.to_matrix() @ sv
            viol = np.linalg.norm(G_psi)
            max_viol = max(max_viol, viol)

            if viol > tol:
                i, j = self.lattice.site_to_coords(idx)
                violations.append((i, j, viol))

        if verbose:
            if not violations:
                print(f"✓ Gauss law satisfied (max violation = {max_viol:.2e})")
            else:
                print(f"✗ Gauss law violated at {len(violations)} sites (max = {max_viol:.2e})")
                for i, j, v in violations:
                    print(f"   Site ({i},{j}): {v:.2e}")
        return len(violations) == 0

    def _pauli_z_on_qubit(self, qubit_index: int) -> str:
        """Single Z on the specified qubit, I elsewhere."""
        parts = ['I'] * self.n_qubits
        parts[qubit_index] = 'Z'
        return ''.join(parts)