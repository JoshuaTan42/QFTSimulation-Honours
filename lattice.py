class LatticeGrid:
    def __init__(self,
                 width:int,
                 height:int):
        self.width = width
        self.height = height

        self.n_sites = self.width * self.height

        self.n_horizontal_links = self.n_sites
        self.n_vertical_links = self.width * (self.height - 1)
        self.n_links_total = self.n_horizontal_links + self.n_vertical_links

        self.n_qubits = self.n_links_total


    def site_to_coords(self, site_idx:int) -> tuple:
        if not (0 <= site_idx < self.n_sites):
            raise ValueError(f"site_idx={site_idx} out of bounds [0, {self.n_sites})")

        i = site_idx % self.width
        j = site_idx // self.width
        return (i, j)

    def coords_to_site(self, i:int, j:int) -> int:
        i = i % self.width
        if not (0 <= j < self.height):
            raise ValueError(f"j={j} out of bounds [0, {self.height})")

        return i + self.width * j

    def get_horizontal_link_index(self, i: int, j: int) -> int:
        i = i % self.width  # Handle periodic boundary
        return i + self.width * j

    def get_vertical_link_index(self, i: int, j: int) -> int:
        if j >= self.height - 1:
            raise ValueError(f"No vertical link at j={j}")

        offset = self.n_horizontal_links
        return offset + i + self.width * j

    def get_neighbours(self, i:int, j:int) -> dict:
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

        # Reminder that j is height and i is width
        for j in range(self.height - 1):
            for i in range(self.width):
                link_bottom = self.get_horizontal_link_index(i, j)
                link_right = self.get_vertical_link_index((i + 1) % self.width, j)
                link_top = self.get_horizontal_link_index(i, j + 1)
                link_left = self.get_vertical_link_index(i, j)

                plaquette = (link_bottom, link_right, link_top, link_left)
                plaquettes.append(plaquette)

        return plaquettes

    """
    Below this are visualization and debug code. This may or may not have been created with the assistace of an AI assitant who assists those who needs assistance
    """

    def visualize_ascii(self):
        """Print ASCII art visualization of lattice"""
        # Top plate markers
        print("____________" * self.width)

        # Print from top to bottom (reverse j order)
        for j in range(self.height - 1, -1, -1):
            # Print sites and horizontal links with numbers
            line = ""
            for i in range(self.width):
                line += f"({i},{j})"
                if i < self.width - 1:
                    h_link = self.get_horizontal_link_index(i, j)
                    line += f" ─{h_link:2d} ─ "
                else:
                    # Show wrap link
                    h_link = self.get_horizontal_link_index(i, j)
                    line += f" ─{h_link:2d} ─"
            print(line)

            # Print vertical links with numbers (if not bottom row)
            if j > 0:
                # Top pipe line
                pipe_line = "  │  "
                for i in range(1, self.width):
                    pipe_line += "         │  "
                print(pipe_line)

                # Link numbers line
                vline = ""
                for i in range(self.width):
                    v_link = self.get_vertical_link_index(i, j - 1)
                    vline += f" {v_link:2d}  "
                    if i < self.width - 1:
                        vline += "       "
                print(vline)

                # Bottom pipe line
                pipe_line = "  │  "
                for i in range(1, self.width):
                    pipe_line += "         │  "
                print(pipe_line)
            else:
                # Bottom plate markers
                print("▔▔▔▔▔▔▔▔▔▔▔▔" * self.width)


    def debug(self):
        print("Printing Statistics:")
        print("+="*18)
        print(f"Qubit Count           | {self.n_qubits}\n" +
              f"Lattice Width         | {self.width}\n" +
              f"Lattice Height        | {self.height}\n" +

              f"Lattice Sites         | {self.n_sites}\n" +
              f"Lattice X_Links       | {self.n_horizontal_links}\n" +
              f"Lattice Y_Links       | {self.n_vertical_links}\n" +
              f"Lattice Total Links   | {self.n_links_total}\n")

        """for site in range(self.n_sites):
            i, j = self.site_to_coords(site)
            back = self.coords_to_site(i, j)
            print(f"Site {site:2d} → ({i},{j}) → Site {back:2d}  {'✓' if back == site else '✗'}")"""