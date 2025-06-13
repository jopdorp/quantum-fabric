#!/usr/bin/env python3
"""
Carbon Atom Simulation with Bell State Transition

Demonstrates the simulation of a carbon atom with 6 electrons using the unified
molecular simulation framework. Shows:
1. First 1200 steps: Normal carbon electron configuration (1s² 2s² 2p²)
2. Next 1200 steps: Bell-type state with modified quantum numbers and dynamics
"""

import torch
import numpy as np
from unified_hybrid_molecular_simulation import (
    AtomConfig, create_atom_simulation, UnifiedHybridMolecularSimulation, MolecularNucleus, ElectronInfo
)
from hybrid_molecular_simulation import create_atom_electron, X, Y, SCALE
from particles import apply_wavefunction_dynamics
from video_utils import StreamingVideoWriter, open_video
from config import center_x, center_y
import hybrid_molecular_simulation

def create_carbon_atom_multi_phase() -> None:
    """Create and run a carbon atom simulation with bell state transition."""
    print("=== Carbon Atom Multi-Phase Simulation ===")
    print("Phase 1 (0-1200 steps): Normal carbon configuration 1s² 2s² 2p²")
    print("Phase 2 (1200-2400 steps): Bell-type state with quantum dynamics")
    print()
    
    # Initialize video writer
    video_writer = StreamingVideoWriter("carbon_atom_bell_state.avi", fps=30)
    
    # Phase 1: Normal carbon atom (1200 steps)
    print("Creating normal carbon atom configuration...")
    normal_carbon = create_normal_carbon_simulation()
    run_carbon_phase(normal_carbon, video_writer, 1200, "Normal Carbon 1s² 2s² 2p²")
    
    # Phase 2: Bell-type carbon state (1200 steps)
    print("Creating bell-type carbon state...")
    bell_carbon = create_bell_carbon_simulation()
    run_carbon_phase(bell_carbon, video_writer, 1200, "Bell-type Carbon State")
    
    # Finalize video
    video_writer.finalize()
    print(f"\nCarbon atom bell state simulation complete!")
    print("Video saved as: carbon_atom_bell_state.avi")
    
    # Open video for viewing
    try:
        open_video("carbon_atom_bell_state.avi")
    except Exception as e:
        print(f"Could not open video automatically: {e}")


def create_normal_carbon_simulation() -> UnifiedHybridMolecularSimulation:
    """Create standard carbon atom with 1s² 2s² 2p² configuration."""
    
    # Create nucleus
    nuclei = [MolecularNucleus(center_x, center_y, atomic_number=6, atom_id=0)]
    
    # Create electrons with normal carbon configuration
    electron_configs = [
        # 1s orbital (2 electrons)
        (1, 0, 0), (1, 0, 0),
        # 2s orbital (2 electrons) 
        (2, 0, 0), (2, 0, 0),
        # 2p orbitals (2 electrons)
        (2, 1, 0), (2, 1, 1)
    ]
    
    # Generate individual electron wavefunctions
    electron_psis = []
    electron_infos = []
    
    for i, (n, l, m) in enumerate(electron_configs):
        # Create individual electron wavefunction
        psi = create_atom_electron(X, Y, center_x, center_y, (n, l, m), 
                                  atomic_number=6, scale=SCALE/8)
        electron_psis.append(psi)
        
        # Create electron info
        orbital_names = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
        orbital_name = f"{n}{orbital_names.get(l, f'l{l}')}"
        electron_infos.append(ElectronInfo(atom_id=0, electron_name=f"{orbital_name}_{i+1}"))
    
    # Combine into unified wavefunction
    unified_psi = sum(electron_psis) / len(electron_psis)
    unified_psi = torch.tensor(unified_psi, dtype=torch.complex64)
    
    # Normalize
    norm = torch.sqrt(torch.sum(torch.abs(unified_psi)**2))
    if norm > 0:
        unified_psi = unified_psi / norm
    
    return UnifiedHybridMolecularSimulation(
        nuclei=nuclei,
        electron_infos=electron_infos,
        unified_wavefunction=unified_psi,
        electron_repulsion_strength=0.12,
        nuclear_motion_enabled=False,  # Single atom doesn't need nuclear motion
        damping_factor=0.998
    )


def create_bell_carbon_simulation() -> UnifiedHybridMolecularSimulation:
    """Create carbon atom in bell-type quantum state with entangled-like behavior."""
    
    # Create nucleus
    nuclei = [MolecularNucleus(center_x, center_y, atomic_number=6, atom_id=0)]
    
    # Bell-type state: Use higher quantum numbers and quantum dynamics
    bell_configs = [
        # Core electrons in excited states
        (2, 1, -1), (2, 1, 1),  # 2p excited core
        # Valence electrons in d-like states (carbon transitioning to higher configuration)
        (3, 2, -2), (3, 2, 0),  # 3d-like orbitals
        # Additional excited p states
        (3, 1, -1), (3, 1, 1)   # 3p orbitals
    ]
    
    print("  Creating bell-state electrons with quantum dynamics...")
    
    # Generate bell-state electron wavefunctions with dynamics
    electron_psis = []
    electron_infos = []
    
    for i, (n, l, m) in enumerate(bell_configs):
        # Create base electron wavefunction
        psi = create_atom_electron(X, Y, center_x, center_y, (n, l, m), 
                                  atomic_number=6, scale=SCALE/6)
        
        # Apply quantum dynamics using particles.py
        X_numpy = X.cpu().numpy()
        Y_numpy = Y.cpu().numpy()
        psi_numpy = psi if isinstance(psi, np.ndarray) else psi.cpu().numpy()
        
        # Apply different dynamics for each electron to create bell-like entanglement
        momentum_x = 0.1 * np.cos(i * np.pi / 3)  # Circular momentum pattern
        momentum_y = 0.1 * np.sin(i * np.pi / 3)
        offset_x = 3.0 * np.cos(i * np.pi / 2)    # Offset pattern for spatial separation
        offset_y = 3.0 * np.sin(i * np.pi / 2)
        
        psi_dynamic = apply_wavefunction_dynamics(
            psi_numpy, X_numpy, Y_numpy, center_x, center_y,
            momentum_x=momentum_x,
            momentum_y=momentum_y,
            orbital_offset_x=offset_x,
            orbital_offset_y=offset_y
        )
        
        electron_psis.append(torch.tensor(psi_dynamic, dtype=torch.complex64))
        
        # Create electron info
        orbital_names = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
        orbital_name = f"{n}{orbital_names.get(l, f'l{l}')}_bell"
        electron_infos.append(ElectronInfo(atom_id=0, electron_name=f"{orbital_name}_{i+1}"))
    
    # Create bell-like superposition with phase relationships
    bell_psi = torch.zeros_like(electron_psis[0])
    
    # Bell state: Create quantum superposition with entangled-like phases
    for i, psi in enumerate(electron_psis):
        # Apply different phases to create bell-like interference
        phase_factor = torch.exp(1j * torch.tensor(i * np.pi / 4))
        bell_weight = 1.0 / np.sqrt(len(electron_psis))  # Normalize weights
        
        # Add quantum interference effects
        interference_phase = torch.tensor(np.pi/2 if i % 2 == 0 else -np.pi/2)
        bell_psi += bell_weight * psi * phase_factor * torch.exp(1j * interference_phase)
    
    # Normalize
    norm = torch.sqrt(torch.sum(torch.abs(bell_psi)**2))
    if norm > 0:
        bell_psi = bell_psi / norm
    
    return UnifiedHybridMolecularSimulation(
        nuclei=nuclei,
        electron_infos=electron_infos,
        unified_wavefunction=bell_psi,
        electron_repulsion_strength=0.08,  # Lower repulsion for more coherent bell state
        nuclear_motion_enabled=False,
        damping_factor=0.9995  # Less damping for bell state dynamics
    )


def run_carbon_phase(simulation: UnifiedHybridMolecularSimulation, 
                    video_writer: StreamingVideoWriter, 
                    num_steps: int, 
                    phase_name: str) -> None:
    """Run a phase of the carbon atom simulation."""
    
    print(f"  Running {num_steps} steps for {phase_name}...")
    
    for step in range(num_steps):
        if step % 200 == 0:
            print(f"    Step {step}/{num_steps}")
        
        # Evolve the system
        simulation.evolve_step(step)
        
        # Get visualization data
        combined_psi = simulation.get_combined_wavefunction()
        
        # Create frames
        frame_real = torch.real(combined_psi).cpu().numpy()
        frame_imag = torch.imag(combined_psi).cpu().numpy()
        frame_phase = torch.angle(hybrid_molecular_simulation.gaussian_filter_torch(combined_psi, sigma=1)).cpu().numpy()
        frame_prob = torch.abs(combined_psi)**2
        
        # Normalize probability for visualization
        if torch.max(frame_prob) > 0:
            frame_prob = frame_prob / torch.max(frame_prob)
        
        frame_prob = frame_prob.cpu().numpy()
        
        video_writer.add_frame(frame_real, frame_imag, frame_phase, frame_prob)


def create_carbon_atom() -> None:
    """Create and run a simple carbon atom simulation (backward compatibility)."""
    print("=== Simple Carbon Atom Simulation ===")
    print("Carbon atom with 6 electrons: 1s² 2s² 2p²")
    print()
    
    # Configure carbon atom with full electron configuration
    carbon_config = AtomConfig(
        atomic_number=6, 
        position=(center_x, center_y),
        electron_configs=[
            # 1s orbital (2 electrons)
            (1, 0, 0), (1, 0, 0),
            # 2s orbital (2 electrons) 
            (2, 0, 0), (2, 0, 0),
            # 2p orbitals (2 electrons in different p orbitals)
            (2, 1, 0), (2, 1, 1)
        ]
    )
    
    print("Creating carbon atom simulation...")
    from unified_hybrid_molecular_simulation import run_simulation
    simulation = create_atom_simulation(carbon_config)
    
    print("Running simulation...")
    run_simulation(simulation, "carbon_atom_unified.avi", time_steps=1500)
    
    print("Carbon atom simulation complete!")


if __name__ == "__main__":   
    # Run multi-phase carbon atom with bell state transition
    create_carbon_atom_multi_phase()
