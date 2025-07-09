import numpy as np
import torch


# Simulation mode - can be changed at runtime
SIMULATION_MODE = "3D"  # Can be "2D" or "3D"
ZOOM = 10.0

# Grid and simulation parameters - adaptive for 2D/3D
if SIMULATION_MODE == "3D":
    SIZE = 128  # Reduced from 1024 due to 3D memory requirements (256^3 vs 1024^2)
    SIZE_X = SIZE
    SIZE_Y = SIZE
    SIZE_Z = SIZE
    ZOOM = 200.0
else:  # 2D mode
    SIZE = 1024  # Full resolution for 2D
    SIZE_X = SIZE
    SIZE_Y = SIZE
    SIZE_Z = 1  # Minimal Z dimension for 2D compatibility

GRID_WIDTH = SIZE_X
GRID_HEIGHT = SIZE_Y
GRID_DEPTH = SIZE_Z
TIME_STEPS = 3000
TIME_DELTA = 1
  # Reduced DT for richer time evolution
POTENTIAL_STRENGTH = 0.6 # Coulomb strength * 10 to confine electrons
MAX_GATES_PER_CELL = 10  # Quantum gates per cell

# Performance optimization parameters
USE_BATCHED_PROPAGATION = True  # Always use batched electron propagation
ENABLE_VECTORIZED_REPULSION = True  # Use vectorized electron-electron repulsion calculation

# Zoom out configuration
BASE_SCALE = 400.0
SCALE = BASE_SCALE / ZOOM # grid-to-physics scale

# Physical constants
COULOMB_STRENGTH = 2.0
NUCLEAR_CORE_RADIUS = 2.0
NUCLEAR_REPULSION_STRENGTH = 0.5
STRONG_FORCE_STRENGTH = 0.1
STRONG_FORCE_RANGE = 3.0
ELECTRON_REPULSION_STRENGTH = 0.08

# --- Initial world state - 3D
import torch

# Determine device
if torch.xpu.is_available():
    DEVICE = torch.device("xpu")
    print(f"Using Intel XPU: {torch.xpu.get_device_name()}")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda") 
    print(f"Using CUDA: {torch.cuda.get_device_name()}")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# Set default device for all tensors
torch.set_default_device(DEVICE)

# Create coordinate tensors on the appropriate device - adaptive for 2D/3D
if SIMULATION_MODE == "3D":
    X, Y, Z = torch.meshgrid(
        torch.arange(GRID_WIDTH, dtype=torch.float32, device=DEVICE), 
        torch.arange(GRID_HEIGHT, dtype=torch.float32, device=DEVICE), 
        torch.arange(GRID_DEPTH, dtype=torch.float32, device=DEVICE), 
        indexing='ij'
    )
else:  # 2D mode
    X, Y = torch.meshgrid(
        torch.arange(GRID_WIDTH, dtype=torch.float32, device=DEVICE), 
        torch.arange(GRID_HEIGHT, dtype=torch.float32, device=DEVICE), 
        indexing='ij'
    )
    Z = None  # No Z coordinate in 2D mode

center_x, center_y, center_z = SIZE_X // 2, SIZE_Y // 2, SIZE_Z // 2

# Helper function to set simulation mode
def set_simulation_mode(mode: str):
    """Set simulation mode to 2D or 3D and update global variables"""
    global SIMULATION_MODE, SIZE, SIZE_X, SIZE_Y, SIZE_Z, GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH
    global X, Y, Z, center_x, center_y, center_z
    
    if mode not in ["2D", "3D"]:
        raise ValueError("Mode must be '2D' or '3D'")
    
    SIMULATION_MODE = mode
    
    if mode == "3D":
        SIZE = 256
        SIZE_X = SIZE
        SIZE_Y = SIZE  
        SIZE_Z = SIZE
    else:  # 2D mode
        SIZE = 1024
        SIZE_X = SIZE
        SIZE_Y = SIZE
        SIZE_Z = 1
    
    GRID_WIDTH = SIZE_X
    GRID_HEIGHT = SIZE_Y
    GRID_DEPTH = SIZE_Z
    center_x, center_y, center_z = SIZE_X // 2, SIZE_Y // 2, SIZE_Z // 2
    
    # Recreate coordinate tensors
    if mode == "3D":
        X, Y, Z = torch.meshgrid(
            torch.arange(GRID_WIDTH, dtype=torch.float32, device=DEVICE), 
            torch.arange(GRID_HEIGHT, dtype=torch.float32, device=DEVICE), 
            torch.arange(GRID_DEPTH, dtype=torch.float32, device=DEVICE), 
            indexing='ij'
        )
    else:  # 2D mode
        X, Y = torch.meshgrid(
            torch.arange(GRID_WIDTH, dtype=torch.float32, device=DEVICE), 
            torch.arange(GRID_HEIGHT, dtype=torch.float32, device=DEVICE), 
            indexing='ij'
        )
        Z = None
    
    print(f"Simulation mode set to {mode}")
    print(f"Grid size: {SIZE_X}x{SIZE_Y}" + (f"x{SIZE_Z}" if mode == "3D" else ""))


def get_coordinate_tensors():
    """Get the current coordinate tensors for the simulation mode"""
    return X, Y, Z
