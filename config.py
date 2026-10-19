"""Standard parameters and scan ranges of the core pipeline (main.py).

J = 2 with g = 0: the uniform electric term is conserved in the pure-gauge
model, so it only shifts the energy within a flux sector and is switched off
for the Casimir scans (see QuantumFields/gauge_field.py).
"""

import platform
from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    width: int = 3                      # W of the reference lattice (3x3: 15 link qubits)
    height: int = 3                     # H; plate separation d = H - 1
    g: float = 0.0                      # linear electric field strength
    J: float = 2.0                      # plaquette coupling
    rk: float = 0.0                     # Rokhsar-Kivelson coupling
    field_pattern: str = "uniform"      # "uniform" (Joshi et al.) or "staggered"
    penalty: float = 200.0              # Gauss-law penalty (full-space checks only)
    penalty_scan: tuple = (2, 10, 50, 100, 150, 500, 1000)
    dt_scan: tuple = (0.1, 0.05, 0.025)  # time steps for the Trotter error order
    casimir_widths: dict = field(default_factory=lambda: {
        2: tuple(range(1, 11)), 3: tuple(range(1, 7)), 4: tuple(range(1, 8)),
        6: tuple(range(1, 6))})       # separations per width; 6x6 = 66 qubits, 781,043 states
    seed: int = 2026

    @property
    def separation(self) -> int:
        return self.height - 1


def environment() -> dict:
    """Interpreter and package versions, stored with every result file.

    Gate counts depend on the Qiskit transpiler version (single-qubit counts
    and the generic Trotter baseline in particular), so results should be
    compared only between runs with the same versions.
    """
    import numpy
    import qiskit
    import scipy
    return {'python': platform.python_version(), 'qiskit': qiskit.__version__,
            'numpy': numpy.__version__, 'scipy': scipy.__version__}
