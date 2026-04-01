from lattice import LatticeGrid

lattice = LatticeGrid(width=3, height=2, spacing=0.5, boundary_condition="Casimir")
lattice.debug()
lattice.visualize_ascii()