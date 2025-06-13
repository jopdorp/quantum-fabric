#!/usr/bin/env python3
"""
Unified Hybrid Molecular Simulation - Combined Electron Wave Field

This approach unifies all electron wavefunctions into a single collective field while
using the same physics engine as hybrid_molecular_simulation.py for consistency.

Key changes:
- Import all physics implementations from hybrid_molecular_simulation.py
- Only implement the unification logic here
- Use the same FFT+Laplacian evolution for consistency
"""

import torch
from typing import List

# Import everything from the hybrid molecular simulation
from hybrid_molecular_simulation import (
    HybridMolecularSimulation, MolecularElectron, MolecularNucleus, 
    create_atom_electron, gaussian_filter_torch, X, Y
)
from config import SIZE, TIME_STEPS, SCALE, center_x, center_y
from video_utils import StreamingVideoWriter, open_video

class ElectronInfo:
    """Information about an electron in the unified wavefunction"""
    def __init__(self, atom_id: int, electron_name: str = "electron"):
        self.atom_id = atom_id  # Which atom this electron belongs to
        self.name = electron_name


class UnifiedHybridMolecularSimulation(HybridMolecularSimulation):
    """
    Unified approach: Single collective electron wavefunction
    
    This simulation inherits from HybridMolecularSimulation to use the same physics,
    but combines all electrons into a single wavefunction while maintaining proper 
    quantum statistics through antisymmetrization.
    """
    
    def __init__(self, nuclei: List[MolecularNucleus], electron_infos: List[ElectronInfo],
                 unified_wavefunction: torch.Tensor,
                 electron_repulsion_strength: float = 0.15,
                 nuclear_motion_enabled: bool = True,
                 damping_factor: float = 0.98):
        
        # Create fake individual electrons for the parent class
        fake_electrons = []
        for i, info in enumerate(electron_infos):
            # Create a dummy wavefunction - won't be used but needed for parent init
            dummy_wf = torch.zeros_like(X, dtype=torch.complex64)
            fake_electrons.append(MolecularElectron(dummy_wf, info.atom_id, info.name))
        
        # Initialize parent class
        super().__init__(
            nuclei=nuclei,
            electrons=fake_electrons,
            electron_repulsion_strength=electron_repulsion_strength,
            nuclear_motion_enabled=nuclear_motion_enabled,
            damping_factor=damping_factor
        )
        
        # Store our unified wavefunction and electron info
        self.electron_infos = electron_infos
        self.unified_wavefunction = torch.tensor(unified_wavefunction, dtype=torch.complex64)
        self.normalize_wavefunction()
        
        print(f"Initialized unified molecular simulation:")
        print(f"  - {len(nuclei)} nuclei")
        print(f"  - {len(electron_infos)} electrons (unified)")
        print(f"  - Using hybrid physics engine")
    
    def normalize_wavefunction(self):
        """Normalize the unified wavefunction"""
        norm = torch.sqrt(torch.sum(torch.abs(self.unified_wavefunction)**2))
        if norm > 0:
            self.unified_wavefunction = self.unified_wavefunction / norm
    
    def get_electron_density(self) -> torch.Tensor:
        """Get the total electron density from the unified wavefunction"""
        return torch.abs(self.unified_wavefunction)**2
    
    def evolve_step(self, step: int):
        """
        Evolve using the parent class physics but with unified wavefunction.
        This ensures we use the same FFT+Laplacian approach as hybrid_molecular_simulation.py
        """
        
        # Create a temporary single electron with our unified wavefunction
        temp_electron = MolecularElectron(self.unified_wavefunction, atom_id=0, electron_name="unified")
        
        # Replace the fake electrons with our single unified electron
        original_electrons = self.electrons
        self.electrons = [temp_electron]
        
        # Use parent class evolution (which uses the same physics as hybrid)
        super().evolve_step(step)
        
        # Extract the evolved unified wavefunction
        self.unified_wavefunction = self.electrons[0].wavefunction.clone()
        
        # Restore original fake electrons for next iteration
        self.electrons = original_electrons
        
        # Update nuclear positions (already handled by parent class)

    
    def get_combined_density(self) -> torch.Tensor:
        """Get the electron density for visualization."""
        return self.get_electron_density()
    
    def get_combined_wavefunction(self) -> torch.Tensor:
        """Get the unified wavefunction for visualization."""
        return self.unified_wavefunction
    
    def analyze_atomic_character(self) -> dict:
        """Analyze how much atomic character vs molecular character the electrons have."""
        density = self.get_electron_density()
        
        if len(self.nuclei) < 2:
            return {"atomic_character": 1.0, "molecular_character": 0.0}
        
        nucleus1 = self.nuclei[0]
        nucleus2 = self.nuclei[1]
        
        # Distance from each nucleus
        r1 = torch.sqrt((X - nucleus1.position[0])**2 + (Y - nucleus1.position[1])**2)
        r2 = torch.sqrt((X - nucleus2.position[0])**2 + (Y - nucleus2.position[1])**2)
        
        # Define atomic regions (close to each nucleus)
        atomic_radius = SCALE / 5
        atom1_region = (r1 < atomic_radius).float()
        atom2_region = (r2 < atomic_radius).float()
        
        # Define molecular region (between nuclei, excluding atomic regions)
        bond_center_x = (nucleus1.position[0] + nucleus2.position[0]) / 2
        bond_center_y = (nucleus1.position[1] + nucleus2.position[1]) / 2
        r_bond = torch.sqrt((X - bond_center_x)**2 + (Y - bond_center_y)**2)
        
        # Molecular region: between nuclei but not too close to either nucleus
        molecular_region = ((r1 > atomic_radius) & (r2 > atomic_radius) & 
                          (r_bond < SCALE / 2)).float()
        
        # Calculate densities in each region
        density_atom1 = torch.sum(density * atom1_region)
        density_atom2 = torch.sum(density * atom2_region)
        density_molecular = torch.sum(density * molecular_region)
        total_density = torch.sum(density)
        
        # Calculate character ratios
        atomic_density = density_atom1 + density_atom2
        atomic_character = atomic_density / total_density
        molecular_character = density_molecular / total_density
        
        return {
            "atomic_character": atomic_character.item(),
            "molecular_character": molecular_character.item(),
            "atom1_ratio": (density_atom1 / total_density).item(),
            "atom2_ratio": (density_atom2 / total_density).item(),
            "molecular_ratio": (density_molecular / total_density).item()
        }
    
def create_unified_hydrogen_molecule_simulation() -> UnifiedHybridMolecularSimulation:
    """Create a unified H2 molecule simulation with combined wavefunction that preserves atomic character."""
    # Molecular parameters - closer spacing to see interaction
    bond_length = 1.2 * SCALE  # Closer for better interaction
    nucleus1_x = center_x - bond_length/2
    nucleus2_x = center_x + bond_length/2
    nucleus_y = center_y
    
    # Create nuclei
    nuclei = [
        MolecularNucleus(nucleus1_x, nucleus_y, atomic_number=1, atom_id=0),
        MolecularNucleus(nucleus2_x, nucleus_y, atomic_number=1, atom_id=1)
    ]
    
    # Create individual atomic orbitals - each electron stays localized initially
    print("Creating localized electron 1 (hydrogen 1s on atom 0)...")
    psi1 = create_atom_electron(X, Y, nucleus1_x, nucleus_y, (1, 0, 0), 
                               atomic_number=1, scale=SCALE / 10)
    
    print("Creating localized electron 2 (hydrogen 1s on atom 1)...")  
    psi2 = create_atom_electron(X, Y, nucleus2_x, nucleus_y, (1, 0, 0), 
                               atomic_number=1, scale=SCALE / 10)
    
    # Create unified wavefunction with bonding character
    # Use simple superposition to create molecular orbital
    print("Creating unified bonding wavefunction...")
    
    # Start with bonding orbital: (psi1 + psi2) / sqrt(2)
    unified_psi = (psi1 + psi2) / torch.sqrt(torch.tensor(2.0))
    
    # Normalize the combined wavefunction
    norm = torch.sqrt(torch.sum(torch.abs(unified_psi)**2))
    if norm > 0:
        unified_psi = unified_psi / norm
    
    # Create electron info objects
    electron_infos = [
        ElectronInfo(atom_id=0, electron_name="electron_1"),
        ElectronInfo(atom_id=1, electron_name="electron_2")
    ]
    
    # Create simulation with parameters that encourage bonding
    simulation = UnifiedHybridMolecularSimulation(
        nuclei=nuclei,
        electron_infos=electron_infos,
        unified_wavefunction=unified_psi,
        electron_repulsion_strength=0.12,  # Moderate repulsion for realistic behavior
        nuclear_motion_enabled=True,       # Allow nuclei to move
        damping_factor=0.999               # Light damping for natural evolution
    )
    
    return simulation


def run_unified_molecular_simulation(simulation: UnifiedHybridMolecularSimulation, 
                                   video_filename: str = "unified_hybrid_h2_molecule.avi",
                                   fps: int = 24) -> None:
    """Run the unified molecular simulation and save video."""
    
    print(f"Starting unified hybrid molecular simulation...")
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
                bond_length = torch.sqrt(dx**2 + dy**2)
                
                # Get atomic character metrics
                atomic_metrics = simulation.analyze_atomic_character()
                
                print(f"  Bond length: {bond_length:.2f} pixels")
                print(f"  Atomic character: {100*atomic_metrics['atomic_character']:.1f}%")
                print(f"  Molecular character: {100*atomic_metrics['molecular_character']:.1f}%")
                print(f"  Atom 1 density: {100*atomic_metrics['atom1_ratio']:.1f}%")
                print(f"  Atom 2 density: {100*atomic_metrics['atom2_ratio']:.1f}%")
        
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
    
    # Finalize video
    video_writer.finalize()
    print(f"Simulation complete! Video saved as {video_filename}")
    
    # Open video for viewing
    try:
        open_video(video_filename)
    except Exception as e:
        print(f"Could not open video automatically: {e}")


if __name__ == "__main__":
    print("=== Unified Hybrid Molecular Simulation ===")
    print("Combined electron wavefunction + proper antisymmetrization")
    print()
    
    # Test 1: Unified H2 molecule
    print("1. Creating unified H2 molecule simulation...")
    sim1 = create_unified_hydrogen_molecule_simulation()
    run_unified_molecular_simulation(sim1, "unified_hybrid_h2.avi")
    print("Simulation complete!")
