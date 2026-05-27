class LatticeGrid:
    def __init__(self,
                 width: int,
                 height: int,
                 boundary: str = "closed"):
        self.width = width
        self.height = height
        self.boundary = boundary
        self.n_sites = self.width * self.height
        self.n_horizontal_links = self.n_sites
        self.n_vertical_links = self.width * (self.height - 1)
        self.n_links_total = self.n_horizontal_links + self.n_vertical_links
        self.n_qubits = self.n_links_total
        self._gauge_offset = 0

    def site_to_coords(self, site_idx: int) -> tuple:
        if not (0 <= site_idx < self.n_sites):
            raise ValueError(f"site_idx={site_idx} out of bounds [0, {self.n_sites})")
        i = site_idx % self.width
        j = site_idx // self.width
        return (i, j)

    def coords_to_site(self, i: int, j: int) -> int:
        i = i % self.width
        if not (0 <= j < self.height):
            raise ValueError(f"j={j} out of bounds [0, {self.height})")
        return i + self.width * j

    def get_horizontal_link_index(self, i: int, j: int) -> int:
        i = i % self.width
        return i + self.width * j

    def get_vertical_link_index(self, i: int, j: int) -> int:
        if j >= self.height - 1:
            raise ValueError(f"No vertical link at j={j}")
        return self.n_horizontal_links + i + self.width * j

    def get_link_qubit(self, i: int, j: int, direction: str) -> int:
        if direction == 'x':
            return self.get_horizontal_link_index(i, j)
        elif direction == 'y':
            return self.get_vertical_link_index(i, j)
        else:
            raise ValueError(f"Direction must be 'x' or 'y', got '{direction}'")

    def get_neighbours(self, i: int, j: int) -> dict:
        neighbours = {}
        neighbours["right"] = ((i + 1) % self.width, j)
        neighbours["left"] = ((i - 1) % self.width, j)
        if j < self.height - 1:
            neighbours["up"] = (i, j + 1)
        if j > 0:
            neighbours["down"] = (i, j - 1)
        return neighbours

    def get_plaquettes(self) -> list:
        plaquettes = []
        for j in range(self.height - 1):
            for i in range(self.width):
                link_bottom = self.get_horizontal_link_index(i, j)
                link_right = self.get_vertical_link_index((i + 1) % self.width, j)
                link_top = self.get_horizontal_link_index(i, j + 1)
                link_left = self.get_vertical_link_index(i, j)
                plaquettes.append((link_bottom, link_right, link_top, link_left))
        return plaquettes

    def get_plaquette_sublayers(self) -> list:
        sublayers = [[], []]
        for j in range(self.height - 1):
            for i in range(self.width):
                link_bottom = self.get_horizontal_link_index(i, j)
                link_right = self.get_vertical_link_index((i + 1) % self.width, j)
                link_top = self.get_horizontal_link_index(i, j + 1)
                link_left = self.get_vertical_link_index(i, j)
                colour = (i + j) % 2
                sublayers[colour].append((link_bottom, link_right, link_top, link_left))
        return sublayers

    def get_electric_qubits(self) -> list:
        return list(range(self.n_qubits))

    # =================================================================
    # Visualization
    # =================================================================

    def visualize_ascii(self):
        """Print ASCII art with both site and link qubit indices."""
        print("_" * (self.width * 14))
        print("\n\n")

        for j in range(self.height - 1, -1, -1):
            line = ""
            for i in range(self.width):
                if i < self.width - 1:
                    h_link = self.get_horizontal_link_index(i, j)
                    line += f"─g{h_link:d}─"
                else:
                    h_link = self.get_horizontal_link_index(i, j)
                    line += f"─g{h_link:d}─"
            print(line)

            if j > 0:
                vline = ""
                for i in range(self.width):
                    v_link = self.get_vertical_link_index(i, j - 1)
                    vline += f"g{v_link:d} "
                    if i < self.width - 1:
                        vline += "      "
                print(f"  |{'         |' * (self.width - 1)}")
                print(vline)
                print(f"  |{'         |' * (self.width - 1)}")
            else:
                print("▔" * (self.width * 14))
                pass

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