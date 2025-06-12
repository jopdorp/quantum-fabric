#!/usr/bin/env python3
"""
Hybrid Molecular Simulation - Individual Electrons with Unified Forces

This approach combines the best of both worlds:
1. Individual electron wavefunctions that evolve separately (like original simulation)
2. Unified field approach for computing inter-electron forces
3. Realistic molecular behavior through prop    # Create electrons with initial momentum toward each other (3x bigger for better visualization)
    print("Creating electron 1 (moving toward center)...")
    psi1 = create_atom_electron(X, Y, nucleus1_x, nucleus_y, (1, 0, 0), 
                               atomic_number=1, alpha=0.11)  # alpha=0.11 makes atoms ~3x bigger
    psi1 = apply_wavefunction_dynamics(psi1, X, Y, nucleus1_x, nucleus_y,
                                     momentum_x=0.08, momentum_y=0.02,  # Moving right
                                     orbital_offset_x=1.0, orbital_offset_y=0.0)
    
    print("Creating electron 2 (moving toward center)...")
    psi2 = create_atom_electron(X, Y, nucleus2_x, nucleus_y, (1, 0, 0),
                               atomic_number=1, alpha=0.11)  # alpha=0.11 makes atoms ~3x bigger-electron interactions

Key differences from unified_wave_prototype.py:
- Each electron maintains its own wavefunction and identity
- Electrons interact through computed potential fields (not direct superposition)
- Uses batched FFT evolution for efficiency while preserving individual dynamics
- More physically accurate representation of molecular bonding
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from typing import List, Tuple, Optional
import os

# Import existing components
from config import X, Y, SIZE, TIME_STEPS, SCALE, center_x, center_y
from particles import create_atom_electron, apply_wavefunction_dynamics
from physics import create_nucleus_potential, compute_force_from_density
from torch_physics import propagate_wave_batch_with_potentials
from video_utils import StreamingVideoWriter, open_video
from frame_utils import limit_frame


class MolecularElectron:
    """Enhanced electron class with molecular properties."""
    
    def __init__(self, wavefunction: np.ndarray, atom_id: int, electron_name: str = "electron"):
        self.wavefunction = torch.tensor(wavefunction, dtype=torch.complex64)
        self.atom_id = atom_id  # Which atom this electron belongs to
        self.name = electron_name
        self.normalize()
        
    def get_density(self) -> np.ndarray:
        """Get probability density as numpy array."""
        if isinstance(self.wavefunction, torch.Tensor):
            density = torch.abs(self.wavefunction)**2
            return density.cpu().numpy()
        else:
            return np.abs(self.wavefunction)**2
    
    def normalize(self):
        """Normalize the wavefunction."""
        if isinstance(self.wavefunction, torch.Tensor):
            norm = torch.sqrt(torch.sum(torch.abs(self.wavefunction)**2))
            if norm > 0:
                self.wavefunction = self.wavefunction / norm
        else:
            norm = np.sqrt(np.sum(np.abs(self.wavefunction)**2))
            if norm > 0:
                self.wavefunction = self.wavefunction / norm


class MolecularNucleus:
    """Enhanced nucleus class with molecular properties."""
    
    def __init__(self, x: float, y: float, atomic_number: int = 1, atom_id: int = 0):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.zeros(2, dtype=float)
        self.atomic_number = atomic_number
        self.atom_id = atom_id
        self.mass_ratio = 1836.0  # Proton to electron mass ratio


class HybridMolecularSimulation:
    """
    Hybrid approach: Individual electrons + unified field forces
    
    This simulation maintains individual electron wavefunctions while using
    unified field concepts for computing inter-electron interactions.
    """
    
    def __init__(self, nuclei: List[MolecularNucleus], electrons: List[MolecularElectron],
                 electron_repulsion_strength: float = 0.15,
                 nuclear_motion_enabled: bool = True,
                 damping_factor: float = 0.98):
        self.nuclei = nuclei
        self.electrons = electrons
        self.electron_repulsion_strength = electron_repulsion_strength
        self.nuclear_motion_enabled = nuclear_motion_enabled
        self.damping_factor = damping_factor
        
        # For dual-range electron repulsion
        self.short_range_sigma = 2.0 * SCALE / 100.0
        self.long_range_sigma = 8.0 * SCALE / 100.0
        
        print(f"Initialized molecular simulation:")
        print(f"  - {len(nuclei)} nuclei")
        print(f"  - {len(electrons)} electrons")
        print(f"  - Electron repulsion strength: {electron_repulsion_strength}")
        print(f"  - Nuclear motion: {nuclear_motion_enabled}")
    
    def compute_electron_potential(self, target_electron_idx: int) -> np.ndarray:
        """
        Compute the total potential experienced by one electron.
        Includes nuclear attraction + repulsion from other electrons.
        """
        potential = np.zeros_like(X, dtype=float)
        
        # Nuclear attraction potentials
        for nucleus in self.nuclei:
            V_nuclear = create_nucleus_potential(
                X, Y, nucleus.position[0], nucleus.position[1], nucleus.atomic_number
            )
            potential += V_nuclear
        
        # Electron-electron repulsion (from all OTHER electrons)
        for i, other_electron in enumerate(self.electrons):
            if i != target_electron_idx:
                other_density = other_electron.get_density()
                
                # Dual-range repulsion for more realistic interactions
                short_repulsion = gaussian_filter(other_density, sigma=self.short_range_sigma) * 2.0
                long_repulsion = gaussian_filter(other_density, sigma=self.long_range_sigma) * 0.5
                
                potential += self.electron_repulsion_strength * (short_repulsion + long_repulsion)
        
        return potential
    
    def compute_all_electron_potentials(self) -> List[np.ndarray]:
        """Compute potentials for all electrons efficiently."""
        potentials = []
        
        # Pre-compute nuclear potentials (same for all electrons)
        nuclear_potential = np.zeros_like(X, dtype=float)
        for nucleus in self.nuclei:
            V_nuclear = create_nucleus_potential(
                X, Y, nucleus.position[0], nucleus.position[1], nucleus.atomic_number
            )
            nuclear_potential += V_nuclear
        
        # Pre-compute all electron densities
        electron_densities = [e.get_density() for e in self.electrons]
        
        # Pre-filter all densities for efficiency
        short_filtered = [gaussian_filter(d, sigma=self.short_range_sigma) for d in electron_densities]
        long_filtered = [gaussian_filter(d, sigma=self.long_range_sigma) for d in electron_densities]
        
        # Compute potential for each electron
        for i in range(len(self.electrons)):
            potential = nuclear_potential.copy()
            
            # Add repulsion from all OTHER electrons
            for j in range(len(self.electrons)):
                if i != j:
                    repulsion = (short_filtered[j] * 2.0 + long_filtered[j] * 0.5)
                    potential += self.electron_repulsion_strength * repulsion
            
            potentials.append(potential)
        
        return potentials
    
    def compute_nuclear_forces(self) -> List[np.ndarray]:
        """Compute forces on nuclei from electron densities and other nuclei."""
        forces = [np.zeros(2) for _ in self.nuclei]
        
        # Forces from electrons
        for i, nucleus in enumerate(self.nuclei):
            for electron in self.electrons:
                density = electron.get_density()
                force = compute_force_from_density(density, nucleus.position)
                
                # All electrons exert forces on all nuclei
                # Electrons from same atom: attractive
                # Electrons from other atoms: can create bonding or anti-bonding forces
                if electron.atom_id == nucleus.atom_id:
                    forces[i] += force  # Own electrons are attractive
                else:
                    # Inter-atomic electron forces create molecular bonding
                    forces[i] += force * 0.4  # Weaker but significant bonding force
        
        # Nuclear-nuclear repulsion
        for i, nucleus1 in enumerate(self.nuclei):
            for j, nucleus2 in enumerate(self.nuclei):
                if i != j:
                    dx = nucleus2.position[0] - nucleus1.position[0]
                    dy = nucleus2.position[1] - nucleus1.position[1]
                    r = np.sqrt(dx**2 + dy**2)
                    r = max(r, 2.0)  # Prevent singularity
                    
                    # Coulomb repulsion F = k*q1*q2/r^2
                    force_magnitude = 8.0 * nucleus1.atomic_number * nucleus2.atomic_number / (r**2)
                    
                    # Direction away from other nucleus
                    force_x = -force_magnitude * dx / r
                    force_y = -force_magnitude * dy / r
                    
                    forces[i][0] += force_x
                    forces[i][1] += force_y
        
        return forces
    
    def evolve_step(self, step: int):
        """Evolve all electrons and nuclei by one time step."""
        
        # Compute potentials for all electrons
        potentials = self.compute_all_electron_potentials()
        
        # Prepare batch data for electron evolution
        psi_list = [e.wavefunction for e in self.electrons]
        
        # Convert torch tensors to numpy for the physics engine
        psi_numpy_list = []
        for psi in psi_list:
            if isinstance(psi, torch.Tensor):
                psi_numpy_list.append(psi.cpu().numpy())
            else:
                psi_numpy_list.append(psi)
        
        # Use batched wave propagation for efficiency
        evolved_psi_list = propagate_wave_batch_with_potentials(psi_numpy_list, potentials)
        
        # Update electron wavefunctions
        for i, electron in enumerate(self.electrons):
            electron.wavefunction = torch.tensor(evolved_psi_list[i], dtype=torch.complex64)
            # Apply frame limits to prevent boundary issues
            if isinstance(electron.wavefunction, torch.Tensor):
                wf_numpy = electron.wavefunction.cpu().numpy()
                wf_limited = limit_frame(wf_numpy)
                electron.wavefunction = torch.tensor(wf_limited, dtype=torch.complex64)
            electron.normalize()
        
        # Update nuclear positions if enabled
        if self.nuclear_motion_enabled and len(self.nuclei) > 1:
            forces = self.compute_nuclear_forces()
            dt = 3.0  # Time step for nuclear motion
            
            for i, nucleus in enumerate(self.nuclei):
                # Update velocity and position
                nucleus.velocity += forces[i] * dt / nucleus.mass_ratio
                nucleus.position += nucleus.velocity * dt
                
                # Apply damping to prevent runaway oscillations
                nucleus.velocity *= self.damping_factor
                
                # Keep nuclei within simulation bounds
                nucleus.position[0] = np.clip(nucleus.position[0], SIZE//8, 7*SIZE//8)
                nucleus.position[1] = np.clip(nucleus.position[1], SIZE//8, 7*SIZE//8)
    
    def get_combined_density(self) -> np.ndarray:
        """Get combined electron density for visualization."""
        total_density = np.zeros_like(X, dtype=float)
        
        for electron in self.electrons:
            density = electron.get_density()
            total_density += density
        
        return total_density
    
    def get_combined_wavefunction(self) -> np.ndarray:
        """Get combined wavefunction for visualization."""
        if not self.electrons:
            return np.zeros_like(X, dtype=complex)
        
        combined = np.zeros_like(X, dtype=complex)
        for electron in self.electrons:
            if isinstance(electron.wavefunction, torch.Tensor):
                wf = electron.wavefunction.cpu().numpy()
            else:
                wf = electron.wavefunction
            combined += wf
        
        # Normalize
        norm = np.sqrt(np.sum(np.abs(combined)**2))
        if norm > 0:
            combined = combined / norm
        
        return combined


def create_hydrogen_molecule_simulation() -> HybridMolecularSimulation:
    """Create a realistic H2 molecule simulation."""
    # Molecular parameters
    bond_length = 2 * SCALE  # Increased H-H bond length for better visualization
    nucleus1_x = center_x - bond_length/2
    nucleus2_x = center_x + bond_length/2
    nucleus_y = center_y
    
    # Create nuclei
    nuclei = [
        MolecularNucleus(nucleus1_x, nucleus_y, atomic_number=1, atom_id=0),
        MolecularNucleus(nucleus2_x, nucleus_y, atomic_number=1, atom_id=1)
    ]
    
    # Create electrons with realistic hydrogen 1s orbitals (3x bigger for better visualization)
    print("Creating electron 1 (atom 0)...")
    psi1 = create_atom_electron(X, Y, nucleus1_x, nucleus_y, (2, 1, 0), 
                               atomic_number=1, scale=SCALE /10)
    
    print("Creating electron 2 (atom 1)...")
    psi2 = create_atom_electron(X, Y, nucleus2_x, nucleus_y, (3, 2, 1), 
                               atomic_number=1, scale=SCALE /10)
    
    electrons = [
        MolecularElectron(psi1, atom_id=0, electron_name="electron_1"),
        MolecularElectron(psi2, atom_id=1, electron_name="electron_2")
    ]
    
    # Create simulation
    simulation = HybridMolecularSimulation(
        nuclei=nuclei,
        electrons=electrons,
        electron_repulsion_strength=0.12,  # Moderate repulsion for realistic behavior
        nuclear_motion_enabled=True,       # Allow nuclei to move
        damping_factor=0.999               # Slight damping to prevent oscillations
    )
    
    return simulation


def run_molecular_simulation(simulation: HybridMolecularSimulation, 
                           video_filename: str = "hybrid_h2_molecule.avi",
                           fps: int = 24) -> None:
    """Run the molecular simulation and save video."""
    
    print(f"Starting hybrid molecular simulation...")
    print(f"Output video: {video_filename}")
    
    # Set up video writer
    video_writer = StreamingVideoWriter(
        output_file=video_filename,
        fps=fps,
        sample_frames=50,
        keep_first_batch=True,
        first_batch_size=100
    )
    
    # Run simulation
    for step in range(TIME_STEPS):
        if step % 100 == 0:
            print(f"Step {step}/{TIME_STEPS}")
            
            # Report nuclear positions for H2 molecules
            if len(simulation.nuclei) == 2:
                dx = simulation.nuclei[1].position[0] - simulation.nuclei[0].position[0]
                dy = simulation.nuclei[1].position[1] - simulation.nuclei[0].position[1]
                bond_length = np.sqrt(dx**2 + dy**2)
                print(f"  Bond length: {bond_length:.2f} pixels")
        
        # Evolve the system
        simulation.evolve_step(step)
        
        # Get visualization data
        combined_psi = simulation.get_combined_wavefunction()
        
        # Create frames
        frame_real = np.real(combined_psi)
        frame_imag = np.imag(combined_psi)
        frame_phase = np.angle(gaussian_filter(combined_psi, sigma=1))
        frame_prob = np.abs(combined_psi)**2
        
        # Normalize probability for visualization
        if np.max(frame_prob) > 0:
            frame_prob = frame_prob / np.max(frame_prob)
        
        video_writer.add_frame(frame_real, frame_imag, frame_phase, frame_prob)
    
    # Finalize video
    video_writer.finalize()
    print(f"Simulation complete! Video saved as {video_filename}")
    
    # Open video for viewing
    try:
        open_video(video_filename)
    except Exception as e:
        print(f"Could not open video automatically: {e}")


if __name__ == "__main__":
    print("=== Hybrid Molecular Simulation ===")
    print("Individual electrons + unified field forces")
    print()
    
    # Test 1: Stable H2 molecule
    print("1. Creating stable H2 molecule simulation...")
    sim1 = create_hydrogen_molecule_simulation()
    run_molecular_simulation(sim1, "hybrid_stable_h2.avi")
    print()
    
    # Test 2: Dynamic H2 formation
    print("2. Creating dynamic H2 formation simulation...")
    sim2 = create_dynamic_formation_simulation()
    run_molecular_simulation(sim2, "hybrid_dynamic_h2_formation.avi")
    print()
    
    print("All simulations complete!")
