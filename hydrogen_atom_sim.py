#!/usr/bin/env python3
"""
Hydrogen Atom Orbital Progression Simulation

Demonstrates the evolution of a hydrogen atom through different orbitals:
1. 1s orbital (400 steps)
2. 2p orbital (400 steps) 
3. 3d orbital (400 steps)
4. 2p orbital with dynamics (800 steps)

Shows orbital shapes, sizes, and dynamic behavior.
"""

from unified_hybrid_molecular_simulation import (
    AtomConfig, create_atom_simulation, UnifiedHybridMolecularSimulation,
    X, Y, create_atom_electron, ElectronInfo
)
from hybrid_molecular_simulation import MolecularNucleus
from particles import apply_wavefunction_dynamics
from config import center_x, center_y, SCALE
from video_utils import StreamingVideoWriter, open_video
import torch
import numpy as np

def create_hydrogen_orbital_progression() -> None:
    """Create and run a hydrogen atom simulation showing orbital progression."""
    print("=== Hydrogen Atom Orbital Progression ===")
    print("1. 1s orbital (400 steps)")
    print("2. 2p orbital (400 steps)")
    print("3. 3d orbital (400 steps)")
    print("4. 2p orbital with dynamics (800 steps)")
    print()
    
    # Set up video writer for the complete sequence
    video_writer = StreamingVideoWriter(
        output_file="hydrogen_orbital_progression.avi",
        fps=24,
        sample_frames=50,
        keep_first_batch=True,
        first_batch_size=100
    )
    
    # Phase 1: 1s orbital (400 steps)
    print("Phase 1: Creating 1s orbital...")
    simulation = create_hydrogen_simulation((1, 0, 0))
    run_orbital_phase(simulation, video_writer, 400, "1s orbital")
    
    # Phase 2: 2p orbital (400 steps)
    print("\nPhase 2: Transitioning to 2p orbital...")
    simulation = create_hydrogen_simulation((2, 1, 0))
    run_orbital_phase(simulation, video_writer, 400, "2p orbital")
    
    # Phase 3: 3d orbital (400 steps)
    print("\nPhase 3: Transitioning to 3d orbital...")
    simulation = create_hydrogen_simulation((3, 2, 1))
    run_orbital_phase(simulation, video_writer, 400, "3d orbital")
    
    # Phase 4: 2p orbital with dynamics (800 steps)
    print("\nPhase 4: 2p orbital with applied dynamics...")
    simulation = create_hydrogen_simulation_with_dynamics((2, 1, 0))
    run_orbital_phase(simulation, video_writer, 800, "2p with dynamics")
    
    # Finalize video
    video_writer.finalize()
    print(f"\nHydrogen orbital progression simulation complete!")
    print("Video saved as: hydrogen_orbital_progression.avi")
    
    # Open video for viewing
    try:
        open_video("hydrogen_orbital_progression.avi")
    except Exception as e:
        print(f"Could not open video automatically: {e}")


def create_hydrogen_simulation(quantum_numbers) -> UnifiedHybridMolecularSimulation:
    """Create a hydrogen atom simulation with specific quantum numbers."""
    
    # Create nucleus
    nuclei = [MolecularNucleus(center_x, center_y, atomic_number=1, atom_id=0)]
    
    # Create electron wavefunction
    n, l, m = quantum_numbers
    psi = create_atom_electron(X, Y, center_x, center_y, quantum_numbers, 
                              atomic_number=1, scale=SCALE/10)
    
    # Convert to torch tensor
    unified_psi = torch.tensor(psi, dtype=torch.complex64)
    
    # Normalize
    norm = torch.sqrt(torch.sum(torch.abs(unified_psi)**2))
    if norm > 0:
        unified_psi = unified_psi / norm
    
    # Create electron info
    orbital_names = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
    orbital_name = f"{n}{orbital_names.get(l, f'l{l}')}"
    electron_infos = [ElectronInfo(atom_id=0, electron_name=orbital_name)]
    
    return UnifiedHybridMolecularSimulation(
        nuclei=nuclei,
        electron_infos=electron_infos,
        unified_wavefunction=unified_psi,
        electron_repulsion_strength=0.08,
        nuclear_motion_enabled=False,  # Single atom doesn't need nuclear motion
        damping_factor=0.999
    )


def create_hydrogen_simulation_with_dynamics(quantum_numbers) -> UnifiedHybridMolecularSimulation:
    """Create a hydrogen atom simulation with applied dynamics."""
    
    # Create nucleus
    nuclei = [MolecularNucleus(center_x, center_y, atomic_number=1, atom_id=0)]
    
    # Create electron wavefunction
    n, l, m = quantum_numbers
    psi = create_atom_electron(X, Y, center_x, center_y, quantum_numbers, 
                              atomic_number=1, scale=SCALE/10)
    
    # Apply dynamics using particles.py function
    print(f"  Applying dynamics to {n}{['s','p','d','f'][l]} orbital...")
    
    # Convert PyTorch tensors to numpy for particles.py function
    X_numpy = X.cpu().numpy()
    Y_numpy = Y.cpu().numpy()
    psi_numpy = psi if isinstance(psi, np.ndarray) else psi.cpu().numpy()
    
    psi_dynamic = apply_wavefunction_dynamics(
        psi_numpy, X_numpy, Y_numpy, center_x, center_y,
        momentum_x=0.08,      # Moderate momentum in x direction
        momentum_y=0.04,      # Slight momentum in y direction
        orbital_offset_x=2.0, # Offset the orbital center
        orbital_offset_y=1.0
    )
    
    # Convert to torch tensor
    unified_psi = torch.tensor(psi_dynamic, dtype=torch.complex64)
    
    # Normalize
    norm = torch.sqrt(torch.sum(torch.abs(unified_psi)**2))
    if norm > 0:
        unified_psi = unified_psi / norm
    
    # Create electron info
    orbital_names = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
    orbital_name = f"{n}{orbital_names.get(l, f'l{l}')}_dynamic"
    electron_infos = [ElectronInfo(atom_id=0, electron_name=orbital_name)]
    
    return UnifiedHybridMolecularSimulation(
        nuclei=nuclei,
        electron_infos=electron_infos,
        unified_wavefunction=unified_psi,
        electron_repulsion_strength=0.08,
        nuclear_motion_enabled=False,  # Single atom doesn't need nuclear motion
        damping_factor=0.999
    )


def run_orbital_phase(simulation: UnifiedHybridMolecularSimulation, 
                     video_writer: StreamingVideoWriter, 
                     num_steps: int, 
                     phase_name: str) -> None:
    """Run a phase of the orbital progression simulation."""
    from hybrid_molecular_simulation import gaussian_filter_torch
    
    print(f"  Running {num_steps} steps for {phase_name}...")
    
    for step in range(num_steps):
        if step % 100 == 0:
            print(f"    Step {step}/{num_steps}")
        
        # Evolve the system
        simulation.evolve_step(step)
        
        # Get visualization data
        combined_psi = simulation.get_combined_wavefunction()
        
        # Create frames
        frame_real = torch.real(combined_psi).cpu().numpy()
        frame_imag = torch.imag(combined_psi).cpu().numpy()
        frame_phase = torch.angle(gaussian_filter_torch(combined_psi, sigma=1)).cpu().numpy()
        frame_prob = torch.abs(combined_psi)**2
        
        # Normalize probability for visualization
        if torch.max(frame_prob) > 0:
            frame_prob = frame_prob / torch.max(frame_prob)
        
        frame_prob = frame_prob.cpu().numpy()
        
        video_writer.add_frame(frame_real, frame_imag, frame_phase, frame_prob)


if __name__ == "__main__":
    create_hydrogen_orbital_progression()
