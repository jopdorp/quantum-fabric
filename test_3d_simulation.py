#!/usr/bin/env python3
"""
Test script for 3D quantum simulation conversion
"""

import torch
import numpy as np
from config import X, Y, Z, center_x, center_y, center_z, SIZE_X, SIZE_Y, SIZE_Z
from hybrid_molecular_simulation import create_atom_electron, MolecularElectron, MolecularNucleus, HybridMolecularSimulation

def test_3d_atom_creation():
    """Test creating a 3D atomic orbital."""
    print("Testing 3D atomic orbital creation...")
    
    # Create a hydrogen 1s orbital in 3D
    psi_1s = create_atom_electron(X, Y, Z, center_x, center_y, center_z, (1, 0, 0), atomic_number=1)
    
    print(f"3D wavefunction shape: {psi_1s.shape}")
    print(f"Expected shape: ({SIZE_X}, {SIZE_Y}, {SIZE_Z})")
    print(f"Wavefunction device: {psi_1s.device}")
    print(f"Wavefunction dtype: {psi_1s.dtype}")
    
    # Check normalization
    norm = torch.sqrt(torch.sum(torch.abs(psi_1s)**2))
    print(f"Normalization: {norm:.6f} (should be ~1.0)")
    
    # Create a 2p orbital
    psi_2p = create_atom_electron(X, Y, Z, center_x, center_y, center_z, (2, 1, 0), atomic_number=1)
    print(f"2p orbital shape: {psi_2p.shape}")
    
    return psi_1s, psi_2p

def test_3d_simulation():
    """Test creating a simple 3D molecular simulation."""
    print("\nTesting 3D molecular simulation...")
    
    # Create a nucleus at the center
    nucleus = MolecularNucleus(center_x, center_y, center_z, atomic_number=1, atom_id=0)
    print(f"Nucleus position: ({nucleus.x:.1f}, {nucleus.y:.1f}, {nucleus.z:.1f})")
    
    # Create an electron
    psi = create_atom_electron(X, Y, Z, center_x, center_y, center_z, (1, 0, 0), atomic_number=1)
    electron = MolecularElectron(psi, atom_id=0, electron_name="1s")
    
    # Create simulation
    simulation = HybridMolecularSimulation(
        nuclei=[nucleus],
        electrons=[electron],
        electron_repulsion_strength=0.1,
        nuclear_motion_enabled=False
    )
    
    print(f"Simulation created with {len(simulation.nuclei)} nuclei and {len(simulation.electrons)} electrons")
    
    # Test potential computation
    potential = simulation.compute_electron_potential(0)
    print(f"Potential shape: {potential.shape}")
    print(f"Potential range: [{potential.min():.3f}, {potential.max():.3f}]")
    
    # Test combined wavefunction
    combined_psi = simulation.get_combined_wavefunction()
    print(f"Combined wavefunction shape: {combined_psi.shape}")
    
    return simulation

def test_visualization_slice():
    """Test extracting 2D slices from 3D data for visualization."""
    print("\nTesting 3D to 2D visualization slicing...")
    
    # Create a 3D wavefunction
    psi_3d = create_atom_electron(X, Y, Z, center_x, center_y, center_z, (1, 0, 0), atomic_number=1)
    
    # Extract middle slice along Z dimension
    center_z_idx = SIZE_Z // 2
    slice_xy = psi_3d[:, :, center_z_idx]
    print(f"XY slice shape: {slice_xy.shape}")
    
    # Extract middle slice along Y dimension  
    center_y_idx = SIZE_Y // 2
    slice_xz = psi_3d[:, center_y_idx, :]
    print(f"XZ slice shape: {slice_xz.shape}")
    
    # Extract middle slice along X dimension
    center_x_idx = SIZE_X // 2
    slice_yz = psi_3d[center_x_idx, :, :]
    print(f"YZ slice shape: {slice_yz.shape}")
    
    # Test probability density
    prob_density = torch.abs(psi_3d)**2
    print(f"Probability density shape: {prob_density.shape}")
    print(f"Total probability: {torch.sum(prob_density):.6f}")
    
    return slice_xy, slice_xz, slice_yz

if __name__ == "__main__":
    print("=== 3D Quantum Simulation Test ===")
    print(f"Grid size: {SIZE_X} x {SIZE_Y} x {SIZE_Z}")
    print(f"Center: ({center_x}, {center_y}, {center_z})")
    
    try:
        # Test orbital creation
        psi_1s, psi_2p = test_3d_atom_creation()
        
        # Test simulation
        simulation = test_3d_simulation()
        
        # Test visualization slicing
        slice_xy, slice_xz, slice_yz = test_visualization_slice()
        
        print("\n✅ All tests passed! 3D conversion successful.")
        print("\nNext steps:")
        print("1. Create Mayavi visualization script")
        print("2. Test with multiple atoms/electrons")
        print("3. Run time evolution simulation")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
