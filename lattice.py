"""Lattice geometry for the 2+1D U(1) quantum link model.

The lattice is a W x H square grid with cylinder topology: periodic in x
(along the plates) and open in y (perpendicular to them). The two boundary
rows (j = 0 and j = H-1) represent the conducting plates, so the plate
separation is d = H - 1 in lattice units.

Each link carries a spin-1/2 degree of freedom encoded on one qubit:
    - horizontal links: one per site,            N_h = W * H
    - vertical links:   one per interior y-bond, N_v = W * (H - 1)
    - total link qubits:                         N_q = N_h + N_v

Link-qubit indexing (used everywhere as the qubit register order):
    horizontal link (i, j) -> i + W*j                  in [0, N_h)
    vertical   link (i, j) -> N_h + i + W*j            in [N_h, N_q)
"""


class LatticeGrid:
    def __init__(self, width: int, height: int, boundary: str = "cylinder"):
        self.width = width
        self.height = height
        self.boundary = boundary  # descriptive only; geometry is always cylinder

        self.n_sites = width * height
        self.n_horizontal_links = self.n_sites
        self.n_vertical_links = width * (height - 1)
        self.n_links_total = self.n_horizontal_links + self.n_vertical_links
        self.n_qubits = self.n_links_total

    # ------------------------------------------------------------------
    # Site and link indexing
    # ------------------------------------------------------------------

    def site_to_coords(self, site_idx: int) -> tuple:
        if not (0 <= site_idx < self.n_sites):
            raise ValueError(f"site_idx={site_idx} out of bounds [0, {self.n_sites})")
        return (site_idx % self.width, site_idx // self.width)

    def coords_to_site(self, i: int, j: int) -> int:
        i = i % self.width
        if not (0 <= j < self.height):
            raise ValueError(f"j={j} out of bounds [0, {self.height})")
        return i + self.width * j

    def get_horizontal_link_index(self, i: int, j: int) -> int:
        return (i % self.width) + self.width * j

    def get_vertical_link_index(self, i: int, j: int) -> int:
        if j >= self.height - 1:
            raise ValueError(f"No vertical link at j={j}")
        return self.n_horizontal_links + (i % self.width) + self.width * j

    def get_link_qubit(self, i: int, j: int, direction: str) -> int:
        if direction == 'x':
            return self.get_horizontal_link_index(i, j)
        if direction == 'y':
            return self.get_vertical_link_index(i, j)
        raise ValueError(f"Direction must be 'x' or 'y', got '{direction}'")

    def get_neighbours(self, i: int, j: int) -> dict:
        """Neighbouring sites, wrapping in x and clipping in y (open boundary)."""
        neighbours = {
            "right": ((i + 1) % self.width, j),
            "left": ((i - 1) % self.width, j),
        }
        if j < self.height - 1:
            neighbours["up"] = (i, j + 1)
        if j > 0:
            neighbours["down"] = (i, j - 1)
        return neighbours

    # ------------------------------------------------------------------
    # Plaquettes
    # ------------------------------------------------------------------

    def get_plaquettes(self) -> list:
        """All plaquettes as (bottom, right, top, left) link-qubit tuples."""
        plaquettes = []
        for j in range(self.height - 1):
            for i in range(self.width):
                plaquettes.append(self._plaquette_links(i, j))
        return plaquettes

    def get_plaquette_sublayers(self) -> list:
        """Plaquettes split into two checkerboard sublayers by (i+j) parity.

        Within a sublayer no two plaquettes share a link, so their evolutions
        commute and run in parallel (structured Trotter, Joshi et al. 2026).
        """
        sublayers = [[], []]
        for j in range(self.height - 1):
            for i in range(self.width):
                colour = (i + j) % 2
                sublayers[colour].append(self._plaquette_links(i, j))
        return sublayers

    def _plaquette_links(self, i: int, j: int) -> tuple:
        link_bottom = self.get_horizontal_link_index(i, j)
        link_right = self.get_vertical_link_index((i + 1) % self.width, j)
        link_top = self.get_horizontal_link_index(i, j + 1)
        link_left = self.get_vertical_link_index(i, j)
        return (link_bottom, link_right, link_top, link_left)

    def get_electric_qubits(self) -> list:
        """Every link carries an electric (S^z) operator under spin-1/2."""
        return list(range(self.n_qubits))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def visualize_ascii(self):
        """Print ASCII art of the lattice with link-qubit indices."""
        print("_" * (self.width * 14))
        print("\n")

        for j in range(self.height - 1, -1, -1):
            print("".join(f"─g{self.get_horizontal_link_index(i, j)}─"
                           for i in range(self.width)))

            if j > 0:
                vline = "".join(f"g{self.get_vertical_link_index(i, j - 1)}       "
                                for i in range(self.width))
                rung = f"  |{'         |' * (self.width - 1)}"
                print(rung)
                print(vline)
                print(rung)
            else:
                print("▔" * (self.width * 14))

    def debug(self):
        print("Printing Statistics:")
        print("+=" * 20)
        print(f"Total Qubits          | {self.n_qubits}")
        print(f"Lattice Width         | {self.width}")
        print(f"Lattice Height        | {self.height}")
        print(f"Lattice Sites         | {self.n_sites}")
        print(f"Lattice X_Links       | {self.n_horizontal_links}")
        print(f"Lattice Y_Links       | {self.n_vertical_links}")
        print(f"Lattice Total Links   | {self.n_links_total}")
        print(f"Boundary              | {self.boundary}")
        print()
