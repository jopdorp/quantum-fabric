#!/usr/bin/env python3
"""
Hydrogen Molecule (H2) Simulation using Unified Framework

Demonstrates the simulation of an H2 molecule using the unified
molecular simulation framework with proper bonding dynamics.
"""

from unified_hybrid_molecular_simulation import (
    AtomConfig, create_molecule_simulation, run_simulation
)
from config import center_x, center_y, SCALE

def create_hydrogen_molecule(bond_type: str = "bonding") -> None:
    """Create and run an H2 molecule simulation."""
    print("=== Hydrogen Molecule (H2) Simulation ===")
    print(f"Two hydrogen atoms forming {bond_type} molecular orbital")
    print()
    
    # Bond length for H2
    bond_length = 1.2 * SCALE
    
    # Configure hydrogen atoms
    hydrogen_configs = [
        AtomConfig(
            atomic_number=1, 
            position=(center_x - bond_length/2, center_y),
            electron_configs=[(1, 0, 0)]  # 1s electron
        ),
        AtomConfig(
            atomic_number=1, 
            position=(center_x + bond_length/2, center_y),
            electron_configs=[(1, 0, 0)]  # 1s electron
        )
    ]
    
    print("Creating H2 molecule simulation...")
    simulation = create_molecule_simulation(hydrogen_configs, bond_type=bond_type)
    
    print("Running simulation...")
    video_name = f"hydrogen_molecule_{bond_type}.avi"
    run_simulation(simulation, video_name, time_steps=2000)
    
    print(f"H2 molecule ({bond_type}) simulation complete!")


def create_h2_comparison() -> None:
    """Create both bonding and antibonding H2 simulations for comparison."""
    print("=== H2 Bonding vs Antibonding Comparison ===")
    print()
    
    # Create bonding H2
    print("1. Creating bonding H2...")
    create_hydrogen_molecule("bonding")
    
    print()
    
    # Create antibonding H2
    print("2. Creating antibonding H2...")
    create_hydrogen_molecule("antibonding")
    
    print("\nComparison complete!")
    print("Check videos: hydrogen_molecule_bonding.avi vs hydrogen_molecule_antibonding.avi")


if __name__ == "__main__":
    # Default: create bonding H2 molecule
    create_hydrogen_molecule("bonding")
    
    # Uncomment to create comparison:
    # create_h2_comparison()
