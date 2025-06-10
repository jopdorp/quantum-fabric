import numpy as np
from video_utils import open_video
from config import center_x, center_y, X, Y
from particles import create_atom_electron
from simulation import run_simulation, create_simple_atom_simulation


# Hydrogen nucleus position (pixels)
nucleus1_x, nucleus1_y = center_x, center_y
orb_px = 12

print("Creating Hydrogen atom simulation...")

# Define 4 different hydrogen eigenstates to test
hydrogen_eigenstates = [
    (1, 0, 0),  # 1s orbital
    (2, 0, 0),  # 2s orbital  
    (2, 1, 0),  # 2p_z orbital
    (3, 2, 1),  # 3d orbital
]

# Create initial wavefunction using first eigenstate (1s)
current_quantum_numbers = hydrogen_eigenstates[0]
print(f"Starting with eigenstate: n={current_quantum_numbers[0]}, l={current_quantum_numbers[1]}, m={current_quantum_numbers[2]}")

# Create initial electron wavefunction
psi1 = create_atom_electron(X, Y, nucleus1_x, nucleus1_y, orb_px, 
                           current_quantum_numbers, atomic_number=1, alpha=1.0)

# Create simple atom simulation
nuclei, electrons = create_simple_atom_simulation(
    nucleus1_x, nucleus1_y, 
    [psi1], 
    ["1s_electron"], 
    atomic_number=1
)

# Custom progress callback to cycle through eigenstates
def hydrogen_progress_callback(step, total_steps):
    if step % 400 == 0 and step > 0:
        # Cycle through the 4 different hydrogen eigenstates
        eigenstate_index = (step // 400) % len(hydrogen_eigenstates)
        quantum_numbers = hydrogen_eigenstates[eigenstate_index]
        
        print(f"Switching to eigenstate: n={quantum_numbers[0]}, l={quantum_numbers[1]}, m={quantum_numbers[2]}")
        
        # Create new wavefunction
        scale_factor = int(orb_px * 1 / (1 + eigenstate_index / 10))
        new_psi = create_atom_electron(X, Y, nucleus1_x, nucleus1_y, 
                                     scale_factor, quantum_numbers, atomic_number=1, alpha=1.0)
        
        # Update the electron wavefunction in the simulation
        electrons[0].wavefunction = new_psi

# Run simulation
simulation = run_simulation(
    nuclei, 
    electrons,
    video_file="hydrogen_atom_sim.avi",
    fps=24,
    progress_callback=hydrogen_progress_callback,
    electron_repulsion_strength=0.0,  # No electron-electron repulsion for single electron
    enable_nuclear_motion=False,      # Nucleus stays fixed
    orbital_mixing_strength=0.0       # No mixing for single electron
)

print("Hydrogen atom simulation complete!")
