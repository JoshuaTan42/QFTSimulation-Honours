from qiskit.quantum_info import SparsePauliOp
import numpy as np
from qiskit import QuantumCircuit

class QuantumLinkModel:
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

    def __init__(self, lattice,
                 g_squared: float = 1.0,
                 J: float = 1.0,
                 gauss_penalty: float = 0.0):
        self.lattice = lattice
        self.g_squared = g_squared
        self.J = J
        self.gauss_penalty = gauss_penalty
        self.n_qubits = lattice.n_qubits

    def build_electric_term(self) -> SparsePauliOp:
        shift = self.g_squared / 8.0 * self.lattice.n_links_total
        identity_str = 'I' * self.n_qubits
        return SparsePauliOp([identity_str], coeffs=[shift])

    def build_magnetic_term(self) -> SparsePauliOp:
        if self.J == 0.0:
            return SparsePauliOp(['I' * self.n_qubits], coeffs=[0.0])

        plaq_coeff = self.J / 8.0
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
            return SparsePauliOp(['I' * self.n_qubits], coeffs=[0.0])

        gauss = GaussLaw(self.lattice)
        gauss_ops = gauss.build_gauss_operators()

        penalty_terms = []
        for G_op in gauss_ops:
            if G_op is None:
                continue
            G_squared = (G_op @ G_op).simplify()
            penalty_terms.append(G_squared)

        if not penalty_terms:
            return SparsePauliOp(['I' * self.n_qubits], coeffs=[0.0])

        H_penalty = self.gauss_penalty * sum(penalty_terms[1:], penalty_terms[0])
        return H_penalty.simplify()

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
    def __init__(self, lattice):
        self.lattice = lattice
        self.n_qubits = lattice.n_qubits
        self.n_sites = lattice.n_sites

    def build_gauss_operators(self) -> list:
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
                i_down, j_down = neighbours["down"]
                link_idx = self.lattice.get_vertical_link_index(i, j_down)
                pauli_strings.append(self._pauli_z_on_qubit(link_idx))
                coefficients.append(-0.5)

            if len(pauli_strings) > 0:
                G_site = SparsePauliOp(pauli_strings, coeffs=coefficients).simplify()
            else:
                label_i = 'I' * self.n_qubits
                G_site = SparsePauliOp([label_i], coeffs=[0.0])

            gauss_ops.append(G_site)

        return gauss_ops

    def get_initial_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)

        for link_q in self.lattice.get_electric_qubits():
            qc.h(link_q)

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
            return SparsePauliOp(['I' * self.n_qubits], coeffs=[0.0])

        H_penalty = lam * sum(penalty_terms[1:], penalty_terms[0])
        return H_penalty.simplify()

    def verify(self, sv: np.ndarray, tol: float = 1e-6,
               verbose: bool = True) -> bool:
        max_viol = 0.0
        violations = []

        for idx, G_op in enumerate(self.build_gauss_operators()):
            if G_op is None:
                continue

            G_sparse = G_op.to_matrix(sparse=True)
            G_psi = G_sparse @ sv
            viol = np.linalg.norm(G_psi)
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