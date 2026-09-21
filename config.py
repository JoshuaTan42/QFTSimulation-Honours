"""Standard parameters and scan ranges (thesis Table 2.2.2)."""

from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    width: int = 3                      # W, scan 2, 3, 4
    height: int = 3                     # H, scan 2..6  (d = H - 1)
    g: float = 0.0                      # linear electric field strength
    J: float = 2.0                      # plaquette coupling
    rk: float = 0.0                     # Rokhsar-Kivelson coupling
    field_pattern: str = "uniform"      # "uniform" (Joshi et al.) or "staggered"
    penalty: float = 200.0              # Gauss-law penalty (full-space checks only)
    penalty_scan: tuple = (2, 10, 50, 100, 150, 500, 1000)
    dt_scan: tuple = (0.1, 0.05, 0.025)
    casimir_widths: dict = field(default_factory=lambda: {
        2: tuple(range(1, 11)), 3: tuple(range(1, 7)), 4: tuple(range(1, 8)),
        6: tuple(range(1, 5))})
    seed: int = 2026

    @property
    def separation(self) -> int:
        return self.height - 1
