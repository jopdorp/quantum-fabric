from config import center_x, center_y, X, Y
import config
from particles import create_atom_electron, apply_wavefunction_dynamics
from simulation import run_simulation, create_simple_atom_simulation, get_default_repulsion_sigmas


# Hydrogen nucleus position (pixels)
nucleus1_x, nucleus1_y = center_x, center_y

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
psi1 = create_atom_electron(X, Y, nucleus1_x, nucleus1_y, current_quantum_numbers, atomic_number=1, alpha=1.0)
psi1 = apply_wavefunction_dynamics(psi1, X, Y, nucleus1_x, nucleus1_y)

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
        # adjust config.ZOOM for higher eigenstates
        config.ZOOM = 10.0 * (1 + eigenstate_index)
        # Recalculate SCALE since it depends on ZOOM
        config.SCALE = config.BASE_SCALE / config.ZOOM
        quantum_numbers = hydrogen_eigenstates[eigenstate_index]
        
        print(f"Switching to eigenstate: n={quantum_numbers[0]}, l={quantum_numbers[1]}, m={quantum_numbers[2]}")
        print(f"New ZOOM: {config.ZOOM}, new SCALE: {config.SCALE}")
        
        # Create new wavefunction
        new_psi = create_atom_electron(X, Y, nucleus1_x, nucleus1_y, quantum_numbers, atomic_number=1, alpha=1.0)
        
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
    orbital_mixing_strength=0.0,      # No mixing for single electron
    repulsion_sigmas=get_default_repulsion_sigmas()
)

print("Hydrogen atom simulation complete!")
