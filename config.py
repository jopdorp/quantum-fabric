import numpy as np


# Grid and simulation parameters
SIZE = 512
GRID_WIDTH = SIZE
GRID_HEIGHT = SIZE
TIME_STEPS = 3000
TIME_DELTA = 8  # Reduced DT for richer time evolution
POTENTIAL_STRENGTH = 0.6 # Coulomb strength * 10 to confine electrons
MAX_GATES_PER_CELL = 10  # Quantum gates per cell

# Zoom out configuration
ZOOM = 8.0
BASE_SCALE = 400.0
SCALE = BASE_SCALE / ZOOM # grid-to-physics scale

# Physical constants
COULOMB_STRENGTH = 2.0
NUCLEAR_CORE_RADIUS = 2.0
NUCLEAR_REPULSION_STRENGTH = 0.5
STRONG_FORCE_STRENGTH = 0.1
STRONG_FORCE_RANGE = 3.0
ELECTRON_REPULSION_STRENGTH = 0.08

# --- Initial world state
X, Y = np.meshgrid(np.arange(GRID_WIDTH), np.arange(GRID_HEIGHT))
center_x, center_y = SIZE // 2, SIZE // 2
