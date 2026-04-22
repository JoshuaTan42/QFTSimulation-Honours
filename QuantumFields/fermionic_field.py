import numpy as np
from qiskit.quantum_info import SparsePauliOp


class FermionicField:
    def __init__(self, lattice):
        self.lattice = lattice
        self.n_qubits = lattice.n_qubits

    def build_mass_term(self, mass: float) -> SparsePauliOp:
        pauli_strings = []
        coefficients = []

        for qubit, sign in self.lattice.get_mass_qubits():
            label_i = ['I'] * self.n_qubits
            pauli_strings.append(''.join(reversed(label_i)))
            coefficients.append(mass * sign * 0.5)

            label_z = ['I'] * self.n_qubits
            label_z[qubit] = 'Z'
            pauli_strings.append(''.join(reversed(label_z)))
            coefficients.append(-mass * sign * 0.5)

        return SparsePauliOp(pauli_strings, coeffs=coefficients).simplify()

    def build_hopping_term(self, kappa: float) -> SparsePauliOp:
        pauli_strings = []
        coefficients = []

        hopping_paulis = [
            ('X', 'X', 'X', +1.0),
            ('X', 'Y', 'Y', -1.0),
            ('Y', 'X', 'Y', +1.0),
            ('Y', 'Y', 'X', +1.0),
        ]

        for site_a, link, site_b, direction, stagger in self.lattice.get_hopping_terms():
            coeff = -kappa * stagger / 4.0

            # Find qubits between site_a and site_b that need JW Z-string
            jw_qubits = self._jw_string_qubits(site_a, site_b)

            for pa, pl, pb, sign in hopping_paulis:
                label = ['I'] * self.n_qubits
                label[site_a] = pa
                label[link] = pl
                label[site_b] = pb

                # Apply Z on all qubits in the JW string
                for jw_q in jw_qubits:
                    if label[jw_q] == 'I':
                        label[jw_q] = 'Z'
                    else:
                        # Multiply Pauli: X*Z = -iY, Y*Z = iX, Z*Z = I
                        label[jw_q] = self._pauli_multiply(label[jw_q], 'Z')

                pauli_strings.append(''.join(reversed(label)))
                coefficients.append(sign * coeff)

        return SparsePauliOp(pauli_strings, coeffs=coefficients).simplify()

    def _jw_string_qubits(self, site_a: int, site_b: int) -> list:
        low = min(site_a, site_b)
        high = max(site_a, site_b)

        # Only matter qubits (0 to n_sites-1) between low and high
        jw = [q for q in range(low + 1, high)
              if q < self.lattice.n_matter_qubits]
        return jw

    def _pauli_multiply(self, p1: str, p2: str) -> str:
        multiply_table = {
            ('X', 'Z'): 'Y',
            ('Y', 'Z'): 'X',
            ('Z', 'Z'): 'I',
            ('Z', 'X'): 'Y',
            ('Z', 'Y'): 'X',
            ('X', 'Y'): 'Z',
            ('Y', 'X'): 'Z',
        }
        return multiply_table.get((p1, p2), 'I')

    def build_number_operator(self, site_idx: int) -> SparsePauliOp:
        qubit = self.lattice.get_matter_qubit_by_site(site_idx)
        label_i = ['I'] * self.n_qubits
        label_z = ['I'] * self.n_qubits
        label_z[qubit] = 'Z'
        return SparsePauliOp(
            [''.join(reversed(label_i)), ''.join(reversed(label_z))],
            coeffs=[0.5, -0.5]
        )

    def build_staggered_charge_operator(self, site_idx: int) -> SparsePauliOp:
        i, j = self.lattice.site_to_coords(site_idx)
        sign = self.lattice.stagger_sign(i, j)
        vacuum_charge = (1 - sign) / 2

        qubit = self.lattice.get_matter_qubit_by_site(site_idx)
        label_i = ['I'] * self.n_qubits
        label_z = ['I'] * self.n_qubits
        label_z[qubit] = 'Z'

        return SparsePauliOp(
            [''.join(reversed(label_i)), ''.join(reversed(label_z))],
            coeffs=[(sign * 0.5 - vacuum_charge), -sign * 0.5]
        )