#!/usr/bin/env python3
"""
Carbon Allotropes Simulation

Simulates different forms of carbon with different hybridization:
- Graphene: sp2 hybridization (hexagonal sheets)
- Carbyne: sp hybridization (linear chains) 
- Carbon dimers: basic C2 molecules

These are NOT diamond structures (which uses sp3 hybridization).
For diamond simulations, use diamond_crystal_simulation.py
"""

from carbon_molecules import (
    create_graphene_hexagonal_ring_config,  # Actually graphene
    create_diamond_linear_chain_config,    # Actually carbyne
    create_carbon_dimer,
    run_carbon_structure_simulation,
)

def create_graphene_hexagonal_ring() -> None:
    """Create a hexagonal ring of carbon atoms (graphene structure with sp2 hybridization)."""
    carbon_configs = create_graphene_hexagonal_ring_config()
    
    run_carbon_structure_simulation(
        carbon_configs,
        "Graphene Hexagonal Ring",
        "graphene_hexagonal_ring.avi", 
        2200,
        extra_info="Six carbon atoms in hexagonal arrangement (sp2 hybridization)"
    )

def create_carbyne_linear_chain() -> None:
    """Create a linear chain of carbon atoms (carbyne structure with sp hybridization)."""
    carbon_configs = create_diamond_linear_chain_config(num_carbons=5)
    
    run_carbon_structure_simulation(
        carbon_configs,
        "Carbyne Linear Chain",
        "carbyne_linear_chain.avi",
        2000,
        extra_info="Five carbon atoms in linear chain (sp hybridization)"
    )

def create_carbon_dimer_simulation() -> None:
    """Create a C2 carbon dimer molecule."""
    carbon_configs = create_carbon_dimer("bonding")
    
    run_carbon_structure_simulation(
        carbon_configs,
        "Carbon Dimer (C2)",
        "carbon_dimer_c2.avi",
        1800,
        extra_info="Two carbon atoms forming double bond"
    )

if __name__ == "__main__":
    print("Carbon Allotropes Simulations")
    print("=============================")
    print("Different forms of carbon with different hybridization:")
    print("• Diamond: sp3 (tetrahedral) - use diamond_crystal_simulation.py")
    print("• Graphene: sp2 (planar hexagonal)")  
    print("• Carbyne: sp (linear)")
    print("• C2 dimer: molecular carbon")
    print()
    print("Choose carbon allotrope:")
    print("1. Carbon Dimer (C2)")
    print("2. Graphene Ring (sp2)")
    print("3. Carbyne Chain (sp)")
    print("4. All allotropes")
    
    choice = input("\nEnter choice (1-4) or press Enter for C2: ").strip()
    
    if choice == "1" or choice == "":
        create_carbon_dimer_simulation()
    elif choice == "2":
        create_graphene_hexagonal_ring()
    elif choice == "3":
        create_carbyne_linear_chain()
    elif choice == "4":
        print("Running all carbon allotrope simulations...\n")
        
        print("1/3: Carbon Dimer (C2)")
        create_carbon_dimer_simulation()
        print("\n" + "="*50 + "\n")
        
        print("2/3: Graphene Ring (sp2)")
        create_graphene_hexagonal_ring()
        print("\n" + "="*50 + "\n")
        
        print("3/3: Carbyne Chain (sp)")
        create_carbyne_linear_chain()
        
        print("\nAll carbon allotrope simulations complete!")
    else:
        print("Invalid choice. Running default C2 dimer simulation...")
        create_carbon_dimer_simulation()
