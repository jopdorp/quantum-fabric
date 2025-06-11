import numpy as np
from config import center_x, center_y, X, Y
from particles import create_atom_electron
from simulation import run_simulation, Nucleus, Electron, get_default_repulsion_sigmas


# Carbon nucleus position (pixels)
nucleus1_x, nucleus1_y = center_x, center_y
print("Creating Carbon multi-electron system with all 6 electrons...")

# Create all 6 electrons according to carbon's electron configuration: 1s² 2s² 2p²
# Add slight spatial offsets so electrons don't overlap completely
offset_distance = 15  # pixels

# 1s orbital - 2 electrons (filled shell)
psi_1s_1 = create_atom_electron(X, Y, nucleus1_x + 3, nucleus1_y + 3, (1,0,0), atomic_number=6)
psi_1s_2 = create_atom_electron(X, Y, nucleus1_x - 3, nucleus1_y - 3, (1,0,0), atomic_number=6)

# 2s orbital - 2 electrons (filled shell)
psi_2s_1 = create_atom_electron(X, Y, nucleus1_x + 8, nucleus1_y + 8, (2,0,0), atomic_number=6)
psi_2s_2 = create_atom_electron(X, Y, nucleus1_x - 8, nucleus1_y - 8, (2,0,0), atomic_number=6)

# 2p orbitals - 2 electrons (partially filled, following Hund's rule)
psi_2px = create_atom_electron(X, Y, nucleus1_x + offset_distance, nucleus1_y, (2,1,1), atomic_number=6)
psi_2py = create_atom_electron(X, Y, nucleus1_x - offset_distance, nucleus1_y + offset_distance, (2,1,-1), atomic_number=6)

# Create single carbon nucleus (charge = 6 protons)
carbon_nucleus = Nucleus(nucleus1_x, nucleus1_y, charge=6)
nuclei = [carbon_nucleus]

# Create electrons - all 6 electrons bound to the carbon nucleus
electrons = [
    # 1s orbital - 2 electrons
    Electron(psi_1s_1, "1s_1", nucleus_index=0),
    Electron(psi_1s_2, "1s_2", nucleus_index=0),
    # 2s orbital - 2 electrons  
    Electron(psi_2s_1, "2s_1", nucleus_index=0),
    Electron(psi_2s_2, "2s_2", nucleus_index=0),
    # 2p orbitals - 2 electrons (following Hund's rule)
    Electron(psi_2px, "2p_x", nucleus_index=0),
    Electron(psi_2py, "2p_y", nucleus_index=0)
]

print("Starting multi-electron simulation...")

# Run simulation with multi-electron interactions
simulation = run_simulation(
    nuclei, 
    electrons,
    video_file="carbon_multi_electron.avi",
    fps=24,
    electron_repulsion_strength=0.1,  # Enable electron-electron repulsion (matches original)
    enable_nuclear_motion=False,      # Keep nuclei fixed
    orbital_mixing_strength=0.1,      # Enable orbital mixing for dynamics (matches original)
    mixing_frequency=400,             # Mix orbitals every 400 steps (matches original)
    repulsion_sigmas=get_default_repulsion_sigmas()
)

print("Multi-electron Carbon simulation complete!")
