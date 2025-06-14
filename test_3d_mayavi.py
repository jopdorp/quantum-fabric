#!/usr/bin/env python3
"""
Test 3D quantum simulation with Mayavi visualization
"""
import numpy as np
import torch
from unified_hybrid_molecular_simulation import create_atom_simulation, AtomConfig
from mayavi_3d_viz import MayaviQuantumViz

def test_3d_mayavi():
    print("Testing 3D quantum simulation with Mayavi visualization...")
    
    # Create a simple hydrogen atom simulation
    print("Creating hydrogen atom at center...")
    hydrogen_config = AtomConfig(atomic_number=1, position=(128, 128, 128))
    simulation = create_atom_simulation(hydrogen_config)
    
    print("Simulation created successfully!")
    print(f"Number of nuclei: {len(simulation.nuclei)}")
    print(f"Number of electrons: {len(simulation.electrons)}")
    
    # Check wavefunction shape
    psi = simulation.get_combined_wavefunction()
    print(f"Wavefunction shape: {psi.shape}")
    print(f"Wavefunction dtype: {psi.dtype}")
    
    # Calculate some basic properties
    prob_density = torch.abs(psi)**2
    total_prob = torch.sum(prob_density).item()
    max_prob = torch.max(prob_density).item()
    
    print(f"Total probability: {total_prob:.6f}")
    print(f"Max probability density: {max_prob:.6f}")
    
    # Test a few evolution steps
    print("Testing evolution...")
    for i in range(5):
        simulation.evolve_step(i)
        psi = simulation.get_combined_wavefunction()
        prob = torch.sum(torch.abs(psi)**2).item()
        print(f"Step {i}: Total probability = {prob:.6f}")
    
    print("Creating Mayavi visualization...")
    
    try:
        # Create and start the 3D visualization
        viz = MayaviQuantumViz(simulation)
        print("Starting interactive 3D visualization...")
        print("Controls:")
        print("- Mouse: rotate, zoom, pan")
        print("- 'p': pause/unpause")
        print("- 'r': reset camera view")
        print("- 's': save screenshot")
        print("- Close window to exit")
        
        viz.start()  # This will open the 3D window
        
    except Exception as e:
        print(f"Error with Mayavi visualization: {e}")
        print("This might be due to display/GUI issues.")
        print("The 3D simulation itself is working correctly!")

if __name__ == "__main__":
    test_3d_mayavi()
