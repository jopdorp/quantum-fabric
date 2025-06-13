#!/usr/bin/env python3
"""
Diamond Crystal Simulation using Unified Framework

Demonstrates the simulation of a diamond crystal structure with multiple carbon atoms
arranged in a tetrahedral lattice. This shows collective electronic behavior in
a solid-state diamond structure.
"""

import numpy as np
from unified_hybrid_molecular_simulation import (
    AtomConfig, create_molecule_simulation, run_simulation
)
from config import center_x, center_y, SCALE, SIZE
from diamond_molecule import (
    create_diamond_crystal_from_dimers_config,
    create_diamond_crystal_2x2x2_config,
    create_diamond_tetrahedral_unit_config,
    create_diamond_hexagonal_ring_config,
    create_diamond_linear_chain_config,
    run_diamond_structure_simulation,  # Import the unified runner
)

def create_diamond_crystal_from_dimers() -> None:
    """Create a diamond crystal from multiple C2 dimer molecules arranged in a lattice."""
    rows, cols = 3, 3
    carbon_configs = create_diamond_crystal_from_dimers_config(rows=rows, cols=cols)
    
    run_diamond_structure_simulation(
        carbon_configs, 
        "Diamond Crystal from C2 Dimers",
        "diamond_crystal_from_dimers.avi",
        3000,
        extra_info=f"Crystal lattice: {rows}x{cols} grid, {len(carbon_configs) // 2} C2 dimers"
    )

def create_diamond_crystal_2x2x2() -> None:
    """Create a 2x2x2 diamond crystal structure with 8 carbon atoms."""
    carbon_configs = create_diamond_crystal_2x2x2_config()
    
    run_diamond_structure_simulation(
        carbon_configs,
        "Diamond Crystal (2x2x2)",
        "diamond_crystal_2x2x2.avi", 
        2400,
        extra_info="Eight carbon atoms in diamond lattice structure"
    )


def create_diamond_tetrahedral_unit() -> None:
    """Create a single tetrahedral unit of diamond structure."""
    carbon_configs = create_diamond_tetrahedral_unit_config()
    
    run_diamond_structure_simulation(
        carbon_configs,
        "Diamond Tetrahedral Unit", 
        "diamond_tetrahedral_unit.avi",
        2000,
        extra_info="Five carbon atoms: one central + four tetrahedral neighbors"
    )

def create_diamond_hexagonal_ring() -> None:
    """Create a hexagonal ring of carbon atoms (graphene-like structure)."""
    carbon_configs = create_diamond_hexagonal_ring_config()
    
    run_diamond_structure_simulation(
        carbon_configs,
        "Diamond Hexagonal Ring",
        "diamond_hexagonal_ring.avi", 
        2200,
        extra_info="Six carbon atoms in hexagonal arrangement (graphene-like)"
    )


def create_diamond_linear_chain() -> None:
    """Create a linear chain of carbon atoms (carbyne structure)."""
    carbon_configs = create_diamond_linear_chain_config(num_carbons=5)
    
    run_diamond_structure_simulation(
        carbon_configs,
        "Diamond Linear Chain",
        "diamond_linear_chain.avi",
        2000,
        extra_info="Five carbon atoms in linear chain (carbyne-like)"
    )


if __name__ == "__main__":
    print("Diamond Crystal Structure Simulations")
    print("=====================================")
    print("Choose simulation type:")
    print("1. Diamond Crystal (2x2x2) - 8 carbon atoms")
    print("2. Tetrahedral Unit - 5 carbon atoms")
    print("3. Hexagonal Ring - 6 carbon atoms")
    print("4. Linear Chain - 5 carbon atoms")
    print("5. Diamond Crystal from C2 Dimers - 18 carbon atoms (NEW)")
    print("6. All simulations")
    
    choice = input("\nEnter choice (1-6) or press Enter for crystal: ").strip()
    
    if choice == "1" or choice == "":
        create_diamond_crystal_2x2x2()
    elif choice == "2":
        create_diamond_tetrahedral_unit()
    elif choice == "3":
        create_diamond_hexagonal_ring()
    elif choice == "4":
        create_diamond_linear_chain()
    elif choice == "5":
        create_diamond_crystal_from_dimers()
    elif choice == "6":
        print("Running all diamond structure simulations...\n")
        
        print("1/5: Diamond Crystal (2x2x2)")
        create_diamond_crystal_2x2x2()
        print("\n" + "="*60 + "\n")
        
        print("2/5: Tetrahedral Unit")
        create_diamond_tetrahedral_unit()
        print("\n" + "="*60 + "\n")
        
        print("3/5: Hexagonal Ring")
        create_diamond_hexagonal_ring()
        print("\n" + "="*60 + "\n")
        
        print("4/5: Linear Chain")
        create_diamond_linear_chain()
        print("\n" + "="*60 + "\n")
        
        print("5/5: Diamond Crystal from Dimers")
        create_diamond_crystal_from_dimers()
        
        print("\nAll diamond structure simulations complete!")
    else:
        print("Invalid choice. Running default diamond crystal simulation...")
        create_diamond_crystal_2x2x2()
