import numpy as np


# Grid and simulation parameters
SIZE = 600
GRID_WIDTH = SIZE
GRID_HEIGHT = SIZE
TIME_STEPS = 2000
TIME_DELTA = 2  # Reduced DT for richer time evolution
POTENTIAL_STRENGTH = 1.0  # Coulomb strength
MAX_GATES_PER_CELL = 4  # Quantum gates per cell

# Zoom configuration (visualization decoupled)
ZOOM = 40000.0
BASE_SCALE = 400.0
SCALE = BASE_SCALE / ZOOM # grid-to-physics scale

# Momentum terms
KX = 0.5 * np.pi / SIZE
KY = 0.3 * np.pi / SIZE

# Physical constants
COULOMB_STRENGTH = 1.0
NUCLEAR_CORE_RADIUS = 2.0
NUCLEAR_REPULSION_STRENGTH = 0.5
STRONG_FORCE_STRENGTH = 0.1
STRONG_FORCE_RANGE = 3.0
ELECTRON_REPULSION_STRENGTH = 0.08

# --- Initial world state
X, Y = np.meshgrid(np.arange(GRID_WIDTH), np.arange(GRID_HEIGHT))
center_x, center_y = SIZE // 2, SIZE // 2

### Unused utility variables, kept for reference
BASE_SIGMA = 0.01
SIGMA_AMPLIFIER = BASE_SIGMA / ZOOM  # smaller sigma → envelope decays slower in grid units
