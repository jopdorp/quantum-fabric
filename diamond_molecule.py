#!/usr/bin/env python3
"""
Diamond Molecule (C2) Simulation using Unified Framework

Demonstrates the simulation of a C2 molecule (carbon dimer) using the unified
molecular simulation framework. This represents a building block of diamond structure.
"""

from unified_hybrid_molecular_simulation import (
    AtomConfig, create_molecule_simulation, run_simulation
)
from config import center_x, center_y, SCALE

def create_carbon_dimer(bond_type: str = "bonding") -> None:
    """Create and run a C2 molecule simulation."""
    print("=== Carbon Dimer (C2) Simulation ===")
    print(f"Two carbon atoms forming {bond_type} molecular orbital")
    print("This represents a basic unit of diamond structure")
    print()
    
    # Bond length for C2 (shorter than H2 due to stronger bonding)
    bond_length = 1.0 * SCALE
    
    # Configure carbon atoms with representative electrons
    carbon_configs = [
        AtomConfig(
            atomic_number=6, 
            position=(center_x - bond_length/2, center_y),
            electron_configs=[
                (1, 0, 0),  # 1s electron
                (2, 0, 0),  # 2s electron
                (2, 1, 0),  # 2px electron
                (2, 1, 1),  # 2py electron
            ]
        ),
        AtomConfig(
            atomic_number=6, 
            position=(center_x + bond_length/2, center_y),
            electron_configs=[
                (1, 0, 0),  # 1s electron
                (2, 0, 0),  # 2s electron
                (2, 1, 0),  # 2px electron
                (2, 1, 1),  # 2py electron
            ]
        )
    ]
    
    print("Creating C2 molecule simulation...")
    simulation = create_molecule_simulation(carbon_configs, bond_type=bond_type)
    
    print("Running simulation...")
    video_name = f"carbon_dimer_{bond_type}.avi"
    run_simulation(simulation, video_name, time_steps=2000)
    
    print(f"C2 molecule ({bond_type}) simulation complete!")


def create_diamond_unit_cell() -> None:
    """Create a simplified diamond unit cell simulation with 4 carbon atoms."""
    print("=== Diamond Unit Cell Simulation ===")
    print("Four carbon atoms in tetrahedral arrangement")
    print()
    
    # Tetrahedral positions for diamond structure
    bond_length = 0.8 * SCALE
    
    carbon_configs = [
        # Central carbon
        AtomConfig(
            atomic_number=6, 
            position=(center_x, center_y),
            electron_configs=[(1, 0, 0), (2, 0, 0), (2, 1, 0), (2, 1, 1)]
        ),
        # Surrounding carbons in tetrahedral positions
        AtomConfig(
            atomic_number=6, 
            position=(center_x + bond_length, center_y + bond_length),
            electron_configs=[(1, 0, 0), (2, 0, 0), (2, 1, 0)]
        ),
        AtomConfig(
            atomic_number=6, 
            position=(center_x - bond_length, center_y + bond_length),
            electron_configs=[(1, 0, 0), (2, 0, 0), (2, 1, 1)]
        ),
        AtomConfig(
            atomic_number=6, 
            position=(center_x, center_y - bond_length),
            electron_configs=[(1, 0, 0), (2, 0, 0), (2, 1, 0)]
        )
    ]
    
    print("Creating diamond unit cell simulation...")
    simulation = create_molecule_simulation(carbon_configs, bond_type="bonding")
    
    print("Running simulation...")
    run_simulation(simulation, "diamond_unit_cell.avi", time_steps=1500)
    
    print("Diamond unit cell simulation complete!")


def create_carbon_chain() -> None:
    """Create a linear carbon chain (carbyne) simulation."""
    print("=== Carbon Chain (Carbyne) Simulation ===")
    print("Linear chain of 3 carbon atoms")
    print()
    
    # Linear chain spacing
    spacing = 0.9 * SCALE
    
    carbon_configs = [
        AtomConfig(
            atomic_number=6, 
            position=(center_x - spacing, center_y),
            electron_configs=[(1, 0, 0), (2, 0, 0), (2, 1, 0)]
        ),
        AtomConfig(
            atomic_number=6, 
            position=(center_x, center_y),
            electron_configs=[(1, 0, 0), (2, 0, 0), (2, 1, 0), (2, 1, 1)]
        ),
        AtomConfig(
            atomic_number=6, 
            position=(center_x + spacing, center_y),
            electron_configs=[(1, 0, 0), (2, 0, 0), (2, 1, 1)]
        )
    ]
    
    print("Creating carbon chain simulation...")
    simulation = create_molecule_simulation(carbon_configs, bond_type="bonding")
    
    print("Running simulation...")
    run_simulation(simulation, "carbon_chain.avi", time_steps=1800)
    
    print("Carbon chain simulation complete!")


if __name__ == "__main__":
    print("Choose simulation type:")
    print("1. Carbon dimer (C2)")
    print("2. Diamond unit cell")  
    print("3. Carbon chain")
    print("4. All simulations")
    
    choice = input("\nEnter choice (1-4) or press Enter for C2: ").strip()
    
    if choice == "2":
        create_diamond_unit_cell()
    elif choice == "3":
        create_carbon_chain()
    elif choice == "4":
        print("Running all carbon simulations...\n")
        create_carbon_dimer("bonding")
        print("\n" + "="*50 + "\n")
        create_diamond_unit_cell()
        print("\n" + "="*50 + "\n")
        create_carbon_chain()
    else:
        # Default: C2 dimer
        create_carbon_dimer("bonding")
