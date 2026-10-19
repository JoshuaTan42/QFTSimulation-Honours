"""Lattice geometries for the 2+1D U(1) quantum link model.

LatticeGrid   W x H cylinder used for the Casimir problem: periodic in x (along
              the plates), open in y (across the gap). Rows j = 0 and j = H-1
              are the conducting plates, so the plate separation is d = H - 1.
              Links only, no matter.

OpenLattice   W x H matter sites with open boundaries in both directions, the
              geometry of Joshi et al. (2026). Used for the benchmark of the
              matter-coupled model (QuantumFields/matter_field.py).

Every link carries a spin-1/2 quantum link, encoded on one qubit with
|0> = S^z = +1/2 and |1> = S^z = -1/2.

LatticeGrid qubit register
    horizontal link (i, j) -> i + W*j                  in [0, N_h),  N_h = W*H
    vertical   link (i, j) -> N_h + i + W*j            in [N_h, N_q), N_v = W*(H-1)
    N_q = W*(2H - 1); the 3x3 cylinder has 15 link qubits, and the single-ancilla
    ground-state filter adds one more, for 16 in total.

Links are oriented along +x and +y. Link (i, j, 'x') joins sites (i, j) and
(i+1, j); link (i, j, 'y') joins (i, j) and (i, j+1). A plaquette is labelled by
its lower-left site and listed as (bottom, right, top, left) link qubits.
"""

from __future__ import annotations

from itertools import permutations


class LatticeGrid:
    """W x H cylinder: periodic in x, open in y, plates on the boundary rows."""

    def __init__(self, width: int, height: int):
        if width < 2:
            raise ValueError("width must be >= 2 (periodic direction)")
        if height < 2:
            raise ValueError("height must be >= 2 (two plates)")
        self.width = width
        self.height = height
        self.n_sites = width * height
        self.n_horizontal_links = self.n_sites
        self.n_vertical_links = width * (height - 1)
        self.n_qubits = self.n_horizontal_links + self.n_vertical_links
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

    def site_links(self, i: int, j: int) -> list:
        """Signed links meeting site (i, j) as [(qubit, sign)].

        sign = +1 for links leaving the site (right, up) and -1 for links
        entering it (left, down), so sum(sign * S^z) is the lattice divergence
        of the electric field that enters Gauss's law. Plate sites have three
        links instead of four.
        """
        links = [(self.get_horizontal_link_index(i, j), +1),
                 (self.get_horizontal_link_index(i - 1, j), -1)]
        if j < self.height - 1:
            links.append((self.get_vertical_link_index(i, j), +1))
        if j > 0:
            links.append((self.get_vertical_link_index(i, j - 1), -1))
        return links

    # ------------------------------------------------------------------
    # Plaquettes and their parallel schedule
    # ------------------------------------------------------------------

    def plaquette_anchors(self) -> list:
        """Lower-left sites of all plaquettes, in plaquette-index order."""
        return [(i, j) for j in range(self.height - 1) for i in range(self.width)]

    def plaquette_links(self, i: int, j: int) -> tuple:
        """(bottom, right, top, left) link qubits of the plaquette at (i, j)."""
        return (self.get_horizontal_link_index(i, j),
                self.get_vertical_link_index(i + 1, j),
                self.get_horizontal_link_index(i, j + 1),
                self.get_vertical_link_index(i, j))

    def get_plaquettes(self) -> list:
        """All plaquettes as (bottom, right, top, left) link-qubit tuples."""
        return [self.plaquette_links(i, j) for (i, j) in self.plaquette_anchors()]

    def plaquette_colour(self, i: int, j: int) -> int:
        """Colour class of plaquette (i, j) for parallel execution.

        Two plaquettes conflict when they share a link. Vertical neighbours
        (i, j), (i, j+1) share a horizontal link; horizontal neighbours
        (i, j), (i+1, j) share a vertical link, including the periodic pair
        (W-1, j), (0, j).

        Even W: the checkerboard (i + j) mod 2 is a proper 2-colouring.
        Odd W: every row of plaquettes is a cycle of odd length, so two colours
        cannot work (for W = 3, plaquettes (0, j) and (2, j) have equal parity
        but share the vertical link at i = 0). Three colours suffice:
        c(i) = i mod 2 for i < W-1, c(W-1) = 2, shifted by j modulo 3.
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

        scheme="auto"          the proper colouring above (2 sublayers, 3 for odd W)
        scheme="checkerboard"  the (i + j) mod 2 split used by Joshi et al. for
                               open boundaries. On an odd-W cylinder it puts
                               link-sharing plaquettes in the same sublayer;
                               it is provided to demonstrate that failure.
        """
        if scheme == "checkerboard":
            sublayers = [[], []]
            for (i, j) in self.plaquette_anchors():
                sublayers[(i + j) % 2].append(self.plaquette_links(i, j))
            return sublayers
        if scheme != "auto":
            raise ValueError(f"unknown colouring scheme {scheme!r}")
        sublayers = [[] for _ in range(self.n_plaquette_colours)]
        for (i, j) in self.plaquette_anchors():
            sublayers[self.plaquette_colour(i, j)].append(self.plaquette_links(i, j))
        return [s for s in sublayers if s]

    def plaquette_sublayer_indices(self) -> list:
        """The proper colouring as lists of plaquette indices (one list per colour)."""
        layers = [[] for _ in range(self.n_plaquette_colours)]
        for p, (i, j) in enumerate(self.plaquette_anchors()):
            layers[self.plaquette_colour(i, j)].append(p)
        return [layer for layer in layers if layer]

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

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def visualize_ascii(self):
        """Print the lattice with the qubit index of every link."""
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
        print("Boundary              | cylinder (periodic x, open y)")
        print()


class OpenLattice:
    """W x H matter sites with open boundaries (geometry of Joshi et al. 2026).

    Qubit register: link qubits first, then one matter qubit per site.
        horizontal link (i, j), i < W-1  -> i + (W-1)*j
        vertical   link (i, j), j < H-1  -> N_h + i + W*j,     N_h = (W-1)*H
        matter site  (i, j)              -> N_links + i + W*j
    The 4x3 lattice has 17 links + 12 sites = 29 qubits and the 5x4 lattice
    31 + 20 = 51, matching the qubit counts quoted by Joshi et al.
    """

    def __init__(self, width: int, height: int):
        if width < 2 or height < 2:
            raise ValueError("an open lattice needs at least 2 x 2 sites")
        self.width = width
        self.height = height
        self.n_sites = width * height
        self.n_horizontal_links = (width - 1) * height
        self.n_vertical_links = width * (height - 1)
        self.n_links = self.n_horizontal_links + self.n_vertical_links
        self.n_qubits = self.n_links + self.n_sites
        self.n_plaquettes = (width - 1) * (height - 1)

    # ---- indexing ----------------------------------------------------------

    def horizontal_link(self, i: int, j: int) -> int:
        if not (0 <= i < self.width - 1 and 0 <= j < self.height):
            raise ValueError(f"no horizontal link at ({i}, {j})")
        return i + (self.width - 1) * j

    def vertical_link(self, i: int, j: int) -> int:
        if not (0 <= i < self.width and 0 <= j < self.height - 1):
            raise ValueError(f"no vertical link at ({i}, {j})")
        return self.n_horizontal_links + i + self.width * j

    def matter_qubit(self, i: int, j: int) -> int:
        return self.n_links + i + self.width * j

    def site_coords(self, site: int) -> tuple:
        return (site % self.width, site // self.width)

    def link_info(self, link: int) -> tuple:
        """(i, j, direction) of a link qubit."""
        if not (0 <= link < self.n_links):
            raise ValueError(f"{link} is not a link qubit")
        if link < self.n_horizontal_links:
            return (link % (self.width - 1), link // (self.width - 1), 'x')
        q = link - self.n_horizontal_links
        return (q % self.width, q // self.width, 'y')

    def link_endpoints(self, link: int) -> tuple:
        """(start site, end site) of a link, as (i, j) pairs; links point along +x, +y."""
        i, j, direction = self.link_info(link)
        return ((i, j), (i + 1, j)) if direction == 'x' else ((i, j), (i, j + 1))

    @staticmethod
    def parity(i: int, j: int) -> int:
        """(-1)^(i+j): +1 on even sites, -1 on odd sites."""
        return 1 - 2 * ((i + j) % 2)

    def site_links(self, i: int, j: int) -> list:
        """Existing links at site (i, j) as [(qubit, sign)], +1 leaving, -1 entering."""
        links = []
        if i < self.width - 1:
            links.append((self.horizontal_link(i, j), +1))
        if i > 0:
            links.append((self.horizontal_link(i - 1, j), -1))
        if j < self.height - 1:
            links.append((self.vertical_link(i, j), +1))
        if j > 0:
            links.append((self.vertical_link(i, j - 1), -1))
        return links

    def boundary_divergence(self, i: int, j: int) -> float:
        """Divergence contributed by the links missing at the boundary.

        The open boundary is closed by fixing the missing links at S^z = +1/2
        (a uniform background flux along +x and +y). With this convention the
        all-|0> link configuration has zero divergence at every site, so it is
        the electric vacuum, and an electric string between two static charges
        is a path of |1> links on that background.
        """
        out_missing = (i == self.width - 1) + (j == self.height - 1)
        in_missing = (i == 0) + (j == 0)
        return 0.5 * (out_missing - in_missing)

    # ---- terms -------------------------------------------------------------

    def hopping_terms(self) -> list:
        """One gauge-matter hopping term per link: (link, start site, end site, s_l).

        s_l = +1 on horizontal links and (-1)^i on vertical links, the sign
        convention of Joshi et al. (it mimics staggered fermions in 2D).
        """
        terms = []
        for link in range(self.n_links):
            (a, b) = self.link_endpoints(link)
            i, _, direction = self.link_info(link)
            sign = 1 if direction == 'x' else (-1) ** i
            terms.append((link, a, b, sign))
        return terms

    def plaquette_anchors(self) -> list:
        return [(i, j) for j in range(self.height - 1) for i in range(self.width - 1)]

    def plaquette_links(self, i: int, j: int) -> tuple:
        """(bottom, right, top, left) link qubits of the plaquette at (i, j)."""
        return (self.horizontal_link(i, j), self.vertical_link(i + 1, j),
                self.horizontal_link(i, j + 1), self.vertical_link(i, j))

    def get_plaquettes(self) -> list:
        return [self.plaquette_links(i, j) for (i, j) in self.plaquette_anchors()]

    def hopping_classes(self) -> list:
        """The four matchings of the link graph: links that share no site.

        horizontal links with even i, horizontal with odd i, vertical with even
        j, vertical with odd j. Hopping terms within one class commute.
        """
        classes = [[], [], [], []]
        for link in range(self.n_links):
            i, j, direction = self.link_info(link)
            k = (i % 2) if direction == 'x' else 2 + (j % 2)
            classes[k].append(link)
        return classes

    def sublayers(self) -> list:
        """Four sublayers of mutually commuting hopping and plaquette terms.

        Each sublayer holds one hopping matching plus one class of plaquettes
        (by (i mod 2, j mod 2)) that touches none of the matching's links. This
        is the four-sublayer schedule of Joshi et al.; the assignment of
        plaquette classes to matchings is found by exhaustive search (4! cases)
        and checked for disjointness. Returns [[('hop', link) | ('plaq', p)]].
        """
        hop = self.hopping_classes()
        anchors = self.plaquette_anchors()
        plaq_classes = [[p for p, (i, j) in enumerate(anchors) if (i % 2, j % 2) == key]
                        for key in ((0, 0), (1, 0), (0, 1), (1, 1))]
        plaquettes = self.get_plaquettes()

        def compatible(links, plaqs):
            used = set(links)
            for p in plaqs:
                if used.intersection(plaquettes[p]):
                    return False
                used.update(plaquettes[p])
            return True

        for perm in permutations(range(4)):
            if all(compatible(hop[k], plaq_classes[perm[k]]) for k in range(4)):
                return [[('hop', link) for link in hop[k]] +
                        [('plaq', p) for p in plaq_classes[perm[k]]] for k in range(4)]
        raise RuntimeError("no four-sublayer schedule found")

    def debug(self):
        print(f"Open lattice {self.width} x {self.height}: {self.n_sites} matter sites, "
              f"{self.n_links} links, {self.n_qubits} qubits, {self.n_plaquettes} plaquettes")
