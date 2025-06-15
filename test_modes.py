#!/usr/bin/env python3
"""
Quick test script to verify that 2D and 3D modes work correctly 
and create proper wavefunction shapes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import set_simulation_mode, SIMULATION_MODE
from unified_hybrid_molecular_simulation import create_molecule_simulation, AtomConfig

def test_mode(mode):
    print(f"\n=== Testing {mode} Mode ===")
    
    # Set the mode
    set_simulation_mode(mode)
    
    # Create simple H2 molecule configuration
    bond_length = 1.2 * 40  # SCALE is 40 by default
    center_x, center_y, center_z = 128, 128, 128  # Will be adjusted based on mode
    
    if mode == "3D":
        hydrogen_configs = [
            AtomConfig(
                atomic_number=1, 
                position=(center_x - bond_length/2, center_y, center_z),
                electron_configs=[(1, 0, 0)]
            ),
            AtomConfig(
                atomic_number=1, 
                position=(center_x + bond_length/2, center_y, center_z),
                electron_configs=[(1, 0, 0)]
            )
        ]
    else:  # 2D mode
        hydrogen_configs = [
            AtomConfig(
                atomic_number=1, 
                position=(center_x - bond_length/2, center_y),
                electron_configs=[(1, 0, 0)]
            ),
            AtomConfig(
                atomic_number=1, 
                position=(center_x + bond_length/2, center_y),
                electron_configs=[(1, 0, 0)]
            )
        ]
    
    # Create simulation
    simulation = create_molecule_simulation(hydrogen_configs, bond_type="bonding")
    
    # Check wavefunction shape
    wavefunction = simulation.get_combined_wavefunction()
    print(f"Wavefunction shape: {wavefunction.shape}")
    
    if mode == "3D":
        expected_dims = 3
        expected_shape_pattern = "256x256x256"
    else:
        expected_dims = 2
        expected_shape_pattern = "1024x1024"
    
    if len(wavefunction.shape) == expected_dims:
        print(f"✅ SUCCESS: Correct {expected_dims}D wavefunction created!")
        print(f"   Expected pattern: {expected_shape_pattern}")
        print(f"   Actual shape: {'x'.join(map(str, wavefunction.shape))}")
    else:
        print(f"❌ FAILURE: Expected {expected_dims}D, got {len(wavefunction.shape)}D")
        print(f"   Shape: {wavefunction.shape}")
    
    return len(wavefunction.shape) == expected_dims

def main():
    print("Testing 2D and 3D Molecular Simulation Modes")
    print("=" * 50)
    
    # Test both modes
    success_2d = test_mode("2D")
    success_3d = test_mode("3D")
    
    print(f"\n=== Summary ===")
    print(f"2D Mode: {'✅ PASS' if success_2d else '❌ FAIL'}")
    print(f"3D Mode: {'✅ PASS' if success_3d else '❌ FAIL'}")
    
    if success_2d and success_3d:
        print("\n🎉 All tests passed! Both 2D and 3D modes work correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
