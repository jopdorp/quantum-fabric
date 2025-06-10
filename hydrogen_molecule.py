import numpy as np
from config import center_x, center_y, X, Y
from particles import create_atom_electron
from simulation import run_simulation, Nucleus, Electron, compute_repulsion_sigma_from_orbital_radius


# Hydrogen molecule H2 setup
print("Creating Hydrogen molecule (H2) simulation...")

# Nuclear separation distance (pixels)
bond_length = 40  # pixels - typical H-H bond length in simulation units

# Position nuclei symmetrically around center
nucleus1_x = center_x - bond_length // 2
nucleus1_y = center_y
nucleus2_x = center_x + bond_length // 2
nucleus2_y = center_y

orb_px = 30  # Base orbital radius

# Create hydrogen 1s orbitals for each electron
# Each electron is initially localized around one nucleus
psi1 = create_atom_electron(X, Y, nucleus1_x, nucleus1_y, orb_px, (1,0,0), atomic_number=1, alpha=1.0)
psi2 = create_atom_electron(X, Y, nucleus2_x, nucleus2_y, orb_px, (1,0,0), atomic_number=1, alpha=1.0)

# Create nuclei
nuclei = [
    Nucleus(nucleus1_x, nucleus1_y, charge=1, mass_ratio=1836.0),  # Proton 1
    Nucleus(nucleus2_x, nucleus2_y, charge=1, mass_ratio=1836.0)   # Proton 2
]

# Create electrons
electrons = [
    Electron(psi1, "H1_1s", nucleus_index=0),  # Electron 1 bound to nucleus 1
    Electron(psi2, "H2_1s", nucleus_index=1)   # Electron 2 bound to nucleus 2
]

print(f"Nuclei separation: {bond_length} pixels")
print(f"Nucleus 1 at: ({nucleus1_x}, {nucleus1_y})")
print(f"Nucleus 2 at: ({nucleus2_x}, {nucleus2_y})")

def hydrogen_molecule_progress_callback(step, total_steps):
    """Custom progress callback for molecular dynamics."""
    if step % 200 == 0:
        print(f"Molecular step {step}/{total_steps}...")
        
        # Optionally add some perturbations for dynamics
        if step % 800 == 0 and step > 0:
            print("Adding small molecular vibration...")
            # Could add small nuclear motion perturbations here if desired

# Run simulation with molecular interactions
simulation = run_simulation(
    nuclei, 
    electrons,
    video_file="hydrogen_molecule.avi",
    fps=24,
    progress_callback=hydrogen_molecule_progress_callback,
    electron_repulsion_strength=0.05,  # Moderate electron-electron repulsion
    enable_nuclear_motion=True,        # Allow nuclear motion for molecular dynamics
    orbital_mixing_strength=0.05,      # Weak orbital mixing to preserve molecular structure
    mixing_frequency=600,              # Less frequent mixing for stability
    repulsion_sigmas=compute_repulsion_sigma_from_orbital_radius(orb_px)
)

print("Hydrogen molecule simulation complete!")
print("The simulation includes:")
print("- Two hydrogen nuclei (protons)")
print("- Two electrons with initial 1s character")
print("- Nuclear-nuclear Coulomb repulsion")
print("- Electron-electron repulsion")
print("- Nuclear motion (molecular dynamics)")
print("- Quantum mechanical electron evolution")
