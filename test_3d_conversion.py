#!/usr/bin/env python3
"""
Test script for 3D quantum simulation conversion

Simple test to verify that the 3D conversion is working correctly.
"""

import torch
import numpy as np

def test_3d_conversion():
    """Test basic 3D functionality."""
    print("Testing 3D quantum simulation conversion...")
    
    # Test config
    print("1. Testing config.py...")
    try:
        from config import SIZE_X, SIZE_Y, SIZE_Z, X, Y, Z, center_x, center_y, center_z
        print(f"   Grid size: {SIZE_X} x {SIZE_Y} x {SIZE_Z}")
        print(f"   Coordinate tensors: X{X.shape}, Y{Y.shape}, Z{Z.shape}")
        print(f"   Center: ({center_x}, {center_y}, {center_z})")
        print("   ✓ Config working")
    except Exception as e:
        print(f"   ✗ Config error: {e}")
        return False
    
    # Test torch_physics
    print("\\n2. Testing torch_physics.py...")
    try:
        from torch_physics import WavePropagationModel
        shape_3d = (SIZE_Z, SIZE_Y, SIZE_X)
        model = WavePropagationModel(shape_3d)
        print(f"   Created 3D wave propagation model for shape {shape_3d}")
        
        # Test with dummy wavefunction
        dummy_psi = torch.randn(shape_3d, dtype=torch.complex64)
        dummy_potential = torch.randn(shape_3d, dtype=torch.float32)
        result = model.forward(dummy_psi, dummy_potential)
        print(f"   Propagation test: input {dummy_psi.shape} -> output {result.shape}")
        print("   ✓ Physics engine working")
    except Exception as e:
        print(f"   ✗ Physics error: {e}")
        return False
    
    # Test create_atom_electron
    print("\\n3. Testing create_atom_electron...")
    try:
        from hybrid_molecular_simulation import create_atom_electron
        
        # Test 3D orbital creation
        psi = create_atom_electron(X, Y, Z, center_x, center_y, center_z, (1, 0, 0), atomic_number=1)
        print(f"   Created 3D hydrogen 1s orbital: {psi.shape}")
        print(f"   Data type: {psi.dtype}")
        print(f"   Value range: {torch.abs(psi).min().item():.6f} to {torch.abs(psi).max().item():.6f}")
        
        # Test normalization
        norm = torch.sqrt(torch.sum(torch.abs(psi)**2))
        print(f"   Normalization: {norm.item():.6f}")
        print("   ✓ Orbital creation working")
    except Exception as e:
        print(f"   ✗ Orbital creation error: {e}")
        return False
    
    # Test molecular simulation
    print("\\n4. Testing molecular simulation...")
    try:
        from unified_hybrid_molecular_simulation import create_atom_simulation, AtomConfig
        
        # Create hydrogen atom
        hydrogen_config = AtomConfig(
            atomic_number=1, 
            position=(center_x, center_y, center_z)
        )
        simulation = create_atom_simulation(hydrogen_config)
        print(f"   Created simulation with {len(simulation.nuclei)} nuclei")
        print(f"   Nucleus position: {simulation.nuclei[0].position}")
        
        # Test evolution
        initial_psi = simulation.get_combined_wavefunction()
        print(f"   Initial wavefunction: {initial_psi.shape}")
        
        simulation.evolve_step(0)
        evolved_psi = simulation.get_combined_wavefunction()
        print(f"   Evolved wavefunction: {evolved_psi.shape}")
        print("   ✓ Simulation working")
    except Exception as e:
        print(f"   ✗ Simulation error: {e}")
        return False
    
    print("\\n✓ All tests passed! 3D conversion successful.")
    return True

def test_performance():
    """Quick performance test."""
    print("\\n5. Performance test...")
    try:
        import time
        from unified_hybrid_molecular_simulation import create_atom_simulation, AtomConfig
        
        hydrogen_config = AtomConfig(atomic_number=1, position=(128, 128, 128))
        simulation = create_atom_simulation(hydrogen_config)
        
        # Time 10 evolution steps
        start_time = time.time()
        for i in range(10):
            simulation.evolve_step(i)
        elapsed = time.time() - start_time
        
        print(f"   10 evolution steps took {elapsed:.2f} seconds")
        print(f"   Average: {elapsed/10:.3f} seconds per step")
        print(f"   Estimated FPS: {10/elapsed:.1f}")
        
        if elapsed/10 < 0.1:
            print("   ✓ Good performance for real-time visualization")
        else:
            print("   ⚠ May be too slow for smooth real-time visualization")
            
    except Exception as e:
        print(f"   ✗ Performance test error: {e}")

if __name__ == "__main__":
    if test_3d_conversion():
        test_performance()
        print("\\nReady for 3D visualization!")
        print("Run: python mayavi_3d_viz.py")
    else:
        print("\\nPlease fix the errors before proceeding.")
