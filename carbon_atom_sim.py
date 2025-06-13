#!/usr/bin/env python3
"""
Carbon Atom Simulation using Unified Framework

Demonstrates the simulation of a carbon atom with 6 electrons using the unified
molecular simulation framework. Shows complex multi-electron dynamics.
"""

from unified_hybrid_molecular_simulation import (
    AtomConfig, create_atom_simulation, run_simulation
)
from config import center_x, center_y

def create_carbon_atom() -> None:
    """Create and run a carbon atom simulation with full electron configuration."""
    print("=== Carbon Atom Simulation ===")
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
    simulation = create_atom_simulation(carbon_config)
    
    print("Running simulation...")
    run_simulation(simulation, "carbon_atom_unified.avi", time_steps=1500)
    
    print("Carbon atom simulation complete!")


def create_carbon_ground_state() -> None:
    """Create carbon atom with simplified electron configuration for stability."""
    print("=== Carbon Atom (Simplified) Simulation ===")
    print("Carbon atom with representative electrons")
    print()
    
    # Simplified carbon configuration for better visualization
    carbon_simple_config = AtomConfig(
        atomic_number=6, 
        position=(center_x, center_y),
        electron_configs=[
            (1, 0, 0),  # 1s electron
            (2, 0, 0),  # 2s electron  
            (2, 1, 0),  # 2px electron
            (2, 1, 1)   # 2py electron
        ]
    )
    
    print("Creating simplified carbon simulation...")
    simulation = create_atom_simulation(carbon_simple_config)
    
    print("Running simulation...")
    run_simulation(simulation, "carbon_simple_unified.avi", time_steps=1000)
    
    print("Simplified carbon simulation complete!")


if __name__ == "__main__":
    # Run simplified carbon atom first
    create_carbon_ground_state()
    
    print("\n" + "="*50 + "\n")
    
    # Run full carbon atom
    create_carbon_atom()
