from lattice import LatticeGrid
from QuantumFields.gauge_field import QuantumLinkModel, GaussLaw
import matplotlib.pyplot as plt

lattice = LatticeGrid(width=3, height=3)
lattice.debug()
lattice.visualize_ascii()

# Full QED Hamiltonian
qlm = QuantumLinkModel(lattice, kappa=1.0, mass=0.5, g_squared=1.0, J=1.0)
H = qlm.build_hamiltonian()
print(f"Full QED Hamiltonian: {len(H)} terms")

# Pure gauge for comparison
H_pure = qlm.build_hamiltonian_pure_gauge()
print(f"Pure gauge Hamiltonian: {len(H_pure)} terms")

# Check individual terms
print(f"Electric: {len(qlm.build_electric_term())} terms")
print(f"Magnetic: {len(qlm.build_magnetic_term())} terms")
print(f"Hopping:  {len(qlm.build_hopping_term())} terms")
print(f"Mass:     {len(qlm.build_mass_term())} terms")

# Gauss law now includes fermion charge
gauss = GaussLaw(lattice)
ops = gauss.build_gauss_operators()
for idx, op in enumerate(ops):
    if op is None:
        continue
    i, j = lattice.site_to_coords(idx)
    print(f"G({i},{j}): {len(op)} terms")

# Test the staggered vacuum initial state
qc = gauss.get_initial_circuit()
print(f"\nInitial circuit depth: {qc.depth()}")
print(f"Initial circuit ops: {dict(qc.count_ops())}")

qc = gauss.get_initial_circuit()
qc.draw(output='mpl', fold=-1)
plt.savefig('initial_state_circuit.png', dpi=150, bbox_inches='tight')
plt.show()