"""Lattice geometry for the 2+1D U(1) quantum link model (thesis Sec. 2.2.1).

The lattice is a W x H square grid with cylinder topology: periodic in x
(along the plates) and open in y (perpendicular to them). The two boundary
rows (j = 0 and j = H-1) represent the conducting plates, so the plate
separation is d = H - 1 in lattice units.

Each link carries a spin-1/2 degree of freedom encoded on one qubit:
    horizontal links: one per site,            N_h = W * H
    vertical links:   one per interior y-bond, N_v = W * (H - 1)
    total link qubits:                         N_q = N_h + N_v = W * (2H - 1)

Link-qubit indexing (the qubit register order used everywhere):
    horizontal link (i, j) -> i + W*j                  in [0, N_h)
    vertical   link (i, j) -> N_h + i + W*j            in [N_h, N_q)

The 3x3 cylinder therefore has 15 link qubits. Ground-state preparation with
RQSVT adds exactly one ancilla, for 16 qubits in total.
"""

from __future__ import annotations


class LatticeGrid:
    def __init__(self, width: int, height: int, boundary: str = "cylinder"):
        if width < 2:
            raise ValueError("width must be >= 2 (periodic direction)")
        if height < 2:
            raise ValueError("height must be >= 2 (two plates)")
        if boundary != "cylinder":
            raise ValueError("only cylinder topology is implemented")
        self.width = width
        self.height = height
        self.boundary = boundary

        self.n_sites = width * height
        self.n_horizontal_links = self.n_sites
        self.n_vertical_links = width * (height - 1)
        self.n_links_total = self.n_horizontal_links + self.n_vertical_links
        self.n_qubits = self.n_links_total
        self.n_plaquettes = width * (height - 1)

    @property
    def separation(self) -> int:
        """Plate separation d = H - 1 in lattice units."""
        return self.height - 1

    # ------------------------------------------------------------------
    # Site and link indexing
    # ------------------------------------------------------------------

    def site_to_coords(self, site_idx: int) -> tuple:
        if not (0 <= site_idx < self.n_sites):
            raise ValueError(f"site_idx={site_idx} out of bounds [0, {self.n_sites})")
        return (site_idx % self.width, site_idx // self.width)

    def coords_to_site(self, i: int, j: int) -> int:
        if not (0 <= j < self.height):
            raise ValueError(f"j={j} out of bounds [0, {self.height})")
        return (i % self.width) + self.width * j

    def is_plate_site(self, i: int, j: int) -> bool:
        return j == 0 or j == self.height - 1

    def get_horizontal_link_index(self, i: int, j: int) -> int:
        if not (0 <= j < self.height):
            raise ValueError(f"No horizontal link at j={j}")
        return (i % self.width) + self.width * j

    def get_vertical_link_index(self, i: int, j: int) -> int:
        if not (0 <= j < self.height - 1):
            raise ValueError(f"No vertical link at j={j}")
        return self.n_horizontal_links + (i % self.width) + self.width * j

    def get_link_qubit(self, i: int, j: int, direction: str) -> int:
        if direction == 'x':
            return self.get_horizontal_link_index(i, j)
        if direction == 'y':
            return self.get_vertical_link_index(i, j)
        raise ValueError(f"Direction must be 'x' or 'y', got '{direction}'")

    def link_info(self, qubit: int) -> tuple:
        """Inverse of get_link_qubit: (i, j, direction) of a link qubit."""
        if not (0 <= qubit < self.n_qubits):
            raise ValueError(f"qubit={qubit} out of bounds")
        if qubit < self.n_horizontal_links:
            return (qubit % self.width, qubit // self.width, 'x')
        q = qubit - self.n_horizontal_links
        return (q % self.width, q // self.width, 'y')

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

    def site_links(self, i: int, j: int) -> list:
        """Signed links meeting site (i, j): [(qubit, sign)], sign = +1 for
        links leaving the site (right, up) and -1 for links entering it
        (left, down). This is the lattice divergence used in Gauss's law."""
        links = [(self.get_horizontal_link_index(i, j), +1),
                 (self.get_horizontal_link_index(i - 1, j), -1)]
        if j < self.height - 1:
            links.append((self.get_vertical_link_index(i, j), +1))
        if j > 0:
            links.append((self.get_vertical_link_index(i, j - 1), -1))
        return links

    # ------------------------------------------------------------------
    # Plaquettes
    # ------------------------------------------------------------------

    def plaquette_anchors(self) -> list:
        return [(i, j) for j in range(self.height - 1) for i in range(self.width)]

    def _plaquette_links(self, i: int, j: int) -> tuple:
        link_bottom = self.get_horizontal_link_index(i, j)
        link_right = self.get_vertical_link_index(i + 1, j)
        link_top = self.get_horizontal_link_index(i, j + 1)
        link_left = self.get_vertical_link_index(i, j)
        return (link_bottom, link_right, link_top, link_left)

    def get_plaquettes(self) -> list:
        """All plaquettes as (bottom, right, top, left) link-qubit tuples."""
        return [self._plaquette_links(i, j) for (i, j) in self.plaquette_anchors()]

    def plaquette_colour(self, i: int, j: int) -> int:
        """Colour class of plaquette (i, j) for parallel execution.

        Two plaquettes conflict when they share a link. Vertical neighbours
        (i, j), (i, j+1) share a horizontal link; horizontal neighbours
        (i, j), (i+1, j) share a vertical link, *including the periodic pair*
        (W-1, j), (0, j).

        Even W: the checkerboard (i + j) mod 2 is a proper 2-colouring.
        Odd  W: every row of plaquettes is an odd cycle, so two colours cannot
        work (for W = 3 plaquettes (0, j) and (2, j) have equal parity but
        share the vertical link at i = 0). Three colours are necessary and
        sufficient: c(i) = i mod 2 for i < W-1, c(W-1) = 2, shifted by j.
        Both statements are machine-checked in proofs/Colouring.lean.
        """
        if self.width % 2 == 0:
            return (i + j) % 2
        ring = 2 if (i % self.width) == self.width - 1 else (i % self.width) % 2
        return (ring + j) % 3

    @property
    def n_plaquette_colours(self) -> int:
        return 2 if self.width % 2 == 0 else 3

    def get_plaquette_sublayers(self, scheme: str = "auto") -> list:
        """Plaquettes grouped into sublayers of mutually link-disjoint plaquettes.

        scheme="auto"          proper colouring (2 sublayers, or 3 for odd W)
        scheme="checkerboard"  the (i + j) mod 2 split of Joshi et al. (2026),
                               valid for open boundaries or even W only; kept so
                               the failure on odd-W cylinders can be demonstrated.
        """
        if scheme == "checkerboard":
            sublayers = [[], []]
            for (i, j) in self.plaquette_anchors():
                sublayers[(i + j) % 2].append(self._plaquette_links(i, j))
            return sublayers
        if scheme != "auto":
            raise ValueError(f"unknown colouring scheme {scheme!r}")
        sublayers = [[] for _ in range(self.n_plaquette_colours)]
        for (i, j) in self.plaquette_anchors():
            sublayers[self.plaquette_colour(i, j)].append(self._plaquette_links(i, j))
        return [s for s in sublayers if s]

    @staticmethod
    def sublayers_are_disjoint(sublayers: list) -> bool:
        """True when no two plaquettes in the same sublayer share a link."""
        for layer in sublayers:
            seen = set()
            for plaquette in layer:
                if seen.intersection(plaquette) or len(set(plaquette)) < 4:
                    return False
                seen.update(plaquette)
        return True

    def get_electric_qubits(self) -> list:
        """Every link carries an electric (S^z) operator under spin-1/2."""
        return list(range(self.n_qubits))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def visualize_ascii(self):
        """Print ASCII art of the lattice with link-qubit indices."""
        cell = 10
        print("plate (j = H-1)".center(self.width * cell))
        for j in range(self.height - 1, -1, -1):
            print("".join(f"o--q{self.get_horizontal_link_index(i, j):<3d}---"
                          for i in range(self.width)) + "o")
            if j > 0:
                print("".join("|".ljust(cell) for _ in range(self.width)) + "|")
                print("".join(f"q{self.get_vertical_link_index(i, j - 1)}".ljust(cell)
                              for i in range(self.width)))
                print("".join("|".ljust(cell) for _ in range(self.width)) + "|")
        print("plate (j = 0)".center(self.width * cell))

    def debug(self):
        print("Lattice statistics")
        print("-" * 40)
        print(f"Width x Height        | {self.width} x {self.height}")
        print(f"Plate separation d    | {self.separation}")
        print(f"Sites                 | {self.n_sites}")
        print(f"Horizontal links      | {self.n_horizontal_links}")
        print(f"Vertical links        | {self.n_vertical_links}")
        print(f"Link qubits N_q       | {self.n_qubits}")
        print(f"Plaquettes            | {self.n_plaquettes}")
        print(f"Plaquette sublayers   | {self.n_plaquette_colours}")
        print(f"Boundary              | {self.boundary}")
        print()
