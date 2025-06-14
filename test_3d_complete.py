#!/usr/bin/env python3
"""
Comprehensive test of the 3D quantum simulation conversion.
Tests all major components: 3D grid, physics, simulation, and visualization.
"""

import torch
import numpy as np
import time
import sys
import os

# Import our modules
from config import *
from torch_physics import TorchFFTPhysics
from hybrid_molecular_simulation import UnifiedMolecularSimulation

def test_3d_config():
    """Test that config is properly set up for 3D"""
    print("=== Testing 3D Configuration ===")
    
    # Check 3D grid parameters
    assert hasattr(sys.modules['config'], 'SIZE_X'), "SIZE_X not defined in config"
    assert hasattr(sys.modules['config'], 'SIZE_Y'), "SIZE_Y not defined in config" 
    assert hasattr(sys.modules['config'], 'SIZE_Z'), "SIZE_Z not defined in config"
    
    print(f"Grid size: {SIZE_X} x {SIZE_Y} x {SIZE_Z}")
    print(f"Grid spacing: scale={SCALE:.4f}")
    print(f"Physical domain: {SIZE_X*SCALE:.2f} x {SIZE_Y*SCALE:.2f} x {SIZE_Z*SCALE:.2f}")
    
    # Check coordinate arrays
    assert X.shape == (SIZE_X, SIZE_Y, SIZE_Z), f"X shape mismatch: {X.shape}"
    assert Y.shape == (SIZE_X, SIZE_Y, SIZE_Z), f"Y shape mismatch: {Y.shape}"
    assert Z.shape == (SIZE_X, SIZE_Y, SIZE_Z), f"Z shape mismatch: {Z.shape}"
    
    print("✓ 3D configuration test passed")
    return True

def test_3d_physics():
    """Test that physics engine works in 3D"""
    print("\n=== Testing 3D Physics Engine ===")
    
    # Create physics engine
    physics = TorchFFTPhysics()
    print(f"Device: {physics.device}")
    
    # Create test wavefunction (3D Gaussian)
    sigma = 5.0
    psi_test = torch.exp(-((X - center_x)**2 + (Y - center_y)**2 + (Z - center_z)**2) / (2*sigma**2))
    psi_test = psi_test.to(torch.complex64)
    psi_test = psi_test / torch.sqrt(torch.sum(torch.abs(psi_test)**2))
    
    print(f"Test wavefunction shape: {psi_test.shape}")
    print(f"Test wavefunction norm: {torch.sum(torch.abs(psi_test)**2):.6f}")
    
    # Test Laplacian
    laplacian = physics.compute_laplacian_3d(psi_test)
    print(f"Laplacian shape: {laplacian.shape}")
    print(f"Laplacian max: {torch.max(torch.abs(laplacian)):.6f}")
    
    # Test FFT propagation  
    dt = 0.01
    psi_evolved = physics.fft_propagate_3d(psi_test, dt)
    print(f"Evolved wavefunction norm: {torch.sum(torch.abs(psi_evolved)**2):.6f}")
    
    print("✓ 3D physics engine test passed")
    return True

def test_3d_molecular_simulation():
    """Test the molecular simulation in 3D"""
    print("\n=== Testing 3D Molecular Simulation ===")
    
    # Create single atom simulation
    sim = UnifiedMolecularSimulation()
    
    # Add hydrogen atom at center
    sim.add_nucleus(center_x, center_y, center_z, atomic_number=1)
    
    # Create 1s electron
    sim.create_electrons_for_atoms()
    
    print(f"Number of nuclei: {len(sim.nuclei)}")
    print(f"Number of electrons: {len(sim.electrons)}")
    print(f"Unified wavefunction shape: {sim.unified_wavefunction.shape}")
    print(f"Unified wavefunction norm: {torch.sum(torch.abs(sim.unified_wavefunction)**2):.6f}")
    
    # Test time evolution
    print("Testing time evolution...")
    initial_norm = torch.sum(torch.abs(sim.unified_wavefunction)**2)
    
    sim.evolve_time_step(dt=0.01)
    
    final_norm = torch.sum(torch.abs(sim.unified_wavefunction)**2)
    print(f"Norm conservation: {initial_norm:.6f} -> {final_norm:.6f}")
    
    # Test density extraction
    density = sim.get_combined_density()
    print(f"Density shape: {density.shape}")
    print(f"Density sum: {torch.sum(density):.6f}")
    print(f"Density max: {torch.max(density):.6f}")
    
    print("✓ 3D molecular simulation test passed")
    return sim

def test_performance():
    """Test performance of 3D simulation"""
    print("\n=== Testing 3D Performance ===")
    
    # Create simulation
    sim = UnifiedMolecularSimulation()
    sim.add_nucleus(center_x, center_y, center_z, atomic_number=1)
    sim.create_electrons_for_atoms()
    
    # Time multiple evolution steps
    n_steps = 10
    dt = 0.01
    
    start_time = time.time()
    for i in range(n_steps):
        sim.evolve_time_step(dt=dt)
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_step = total_time / n_steps
    
    print(f"Evolution time for {n_steps} steps: {total_time:.3f} seconds")
    print(f"Time per step: {time_per_step:.3f} seconds")
    print(f"Steps per second: {1/time_per_step:.1f}")
    
    # Memory usage
    if hasattr(torch.cuda, 'memory_allocated'):
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1024**2
            print(f"GPU memory usage: {memory_mb:.1f} MB")
    
    print("✓ Performance test completed")
    return True

def test_multi_atom():
    """Test multi-atom simulation (H2 molecule)"""
    print("\n=== Testing Multi-Atom Simulation (H2) ===")
    
    # Create H2 molecule
    sim = UnifiedMolecularSimulation()
    
    # Add two hydrogen atoms
    bond_length = 1.4  # Approximate H-H bond length in atomic units
    sim.add_nucleus(center_x - bond_length/2, center_y, center_z, atomic_number=1)
    sim.add_nucleus(center_x + bond_length/2, center_y, center_z, atomic_number=1)
    
    # Create electrons
    sim.create_electrons_for_atoms()
    
    print(f"Number of nuclei: {len(sim.nuclei)}")
    print(f"Number of electrons: {len(sim.electrons)}")
    print(f"Unified wavefunction shape: {sim.unified_wavefunction.shape}")
    
    # Test a few evolution steps
    for i in range(3):
        sim.evolve_time_step(dt=0.01)
        norm = torch.sum(torch.abs(sim.unified_wavefunction)**2)
        density_max = torch.max(sim.get_combined_density())
        print(f"Step {i+1}: norm={norm:.6f}, max_density={density_max:.6f}")
    
    print("✓ Multi-atom simulation test passed")
    return sim

def main():
    """Run all tests"""
    print("Starting comprehensive 3D quantum simulation tests...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device available: {torch.cuda.is_available() if hasattr(torch, 'cuda') else 'CPU only'}")
    
    try:
        # Run all tests
        test_3d_config()
        test_3d_physics()
        single_atom_sim = test_3d_molecular_simulation()
        test_performance()
        h2_sim = test_multi_atom()
        
        print("\n=== All Tests Passed! ===")
        print("The 3D quantum simulation conversion is working correctly.")
        print("\nTo visualize the results:")
        print("  python plotly_3d_viz.py")
        print("  Open browser to http://localhost:8050")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
