import numpy as np
from config import center_x, center_y, X, Y
from particles import create_atom_electron
from simulation import run_simulation, Nucleus, Electron, compute_repulsion_sigma_from_orbital_radius


# Carbon nucleus position (pixels)
nucleus1_x, nucleus1_y = center_x, center_y
orb_px = 12

print("Creating Carbon multi-electron system...")

# Create separate electrons that will evolve independently
# We'll focus on the 3 valence p-electrons for visual clarity

# 2p orbitals - 3 separate electrons with slight spatial offsets to avoid overlap
# Add slight spatial offsets so electrons don't overlap completely
offset_distance = 5  # pixels

psi_2px_shifted = create_atom_electron(X, Y, nucleus1_x + offset_distance, nucleus1_y, 12, (2,1,1), atomic_number=6)
psi_2py_shifted = create_atom_electron(X, Y, nucleus1_x - offset_distance, nucleus1_y + offset_distance, 12, (2,1,-1), atomic_number=6)
psi_2pz_shifted = create_atom_electron(X, Y, nucleus1_x, nucleus1_y - offset_distance, 12, (2,1,0), atomic_number=6)

# Create single carbon nucleus (charge = 6 protons)
carbon_nucleus = Nucleus(nucleus1_x, nucleus1_y, charge=6)
nuclei = [carbon_nucleus]

# Create electrons - all bound to the same carbon nucleus
# The electrons are created at offset positions but all feel the same central nucleus
electrons = [
    Electron(psi_2px_shifted, "2p_x", nucleus_index=0),
    Electron(psi_2py_shifted, "2p_y", nucleus_index=0),
    Electron(psi_2pz_shifted, "2p_z", nucleus_index=0)
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
    repulsion_sigmas=compute_repulsion_sigma_from_orbital_radius(orb_px)
)

print("Multi-electron Carbon simulation complete!")
