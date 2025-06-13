#!/usr/bin/env python3
"""
Diamond Molecule Configuration Library

This module provides configuration functions for creating various carbon structures:
- Carbon dimers (C2 molecules)
- Diamond unit cells
- Carbon chains (carbyne)
- Tetrahedral carbon clusters
- Graphene-like hexagonal rings
"""

from unified_hybrid_molecular_simulation import (
    AtomConfig, create_molecule_simulation, run_simulation
)
from config import center_x, center_y, SCALE


def run_carbon_structure_simulation(carbon_configs, structure_name, video_filename, time_steps, extra_info=None):
    """Unified function to run any carbon structure simulation."""
    print(f"=== {structure_name} Simulation ===")
    
    print(f"Created {len(carbon_configs)} carbon atoms")
    if extra_info:
        print(extra_info)
    
    # Show basic position info
    for i, config in enumerate(carbon_configs):
        print(f"  Carbon {i+1}: position ({config.position[0]:.1f}, {config.position[1]:.1f})")
    print()
    
    print(f"Creating {structure_name.lower()} simulation...")
    simulation = create_molecule_simulation(carbon_configs, bond_type="bonding")
    
    print(f"Running {structure_name.lower()} simulation...")
    run_simulation(simulation, video_filename, time_steps=time_steps)
    
    print(f"{structure_name} simulation complete!")

def create_carbon_atom_config(position, electron_configs=None):
    """Create a carbon atom configuration with proper 6-electron setup."""
    if electron_configs is None:
        # Default carbon electron configuration: 1s² 2s² 2p²
        electron_configs = [
            (1, 0, 0),  # 1s electron 1
            (1, 0, 0),  # 1s electron 2
            (2, 0, 0),  # 2s electron 1
            (2, 0, 0),  # 2s electron 2
            (2, 1, 0),  # 2px electron
            (2, 1, 1),  # 2py electron
        ]
    
    return AtomConfig(
        atomic_number=6,
        position=position,
        electron_configs=electron_configs
    )


def create_carbon_dimer(center_pos, bond_length=None):
    """Create a C2 molecule (dimer) at the specified center position."""
    if bond_length is None:
        bond_length = 1.0 * SCALE
    
    center_x_pos, center_y_pos = center_pos
    
    # Two carbon atoms with proper 6-electron configuration
    carbon_configs = [
        create_carbon_atom_config((center_x_pos - bond_length/2, center_y_pos)),
        create_carbon_atom_config((center_x_pos + bond_length/2, center_y_pos))
    ]
    
    return carbon_configs


def create_diamond_crystal_2x2x2_config():
    """Create a 2x2x2 diamond crystal structure with 8 carbon atoms.
    
    Returns the carbon configurations without running the simulation.
    """
    from config import SIZE
    
    # Diamond lattice parameters
    # In diamond, each carbon is tetrahedrally coordinated to 4 others
    # The lattice constant is ~3.57 Å, we'll scale appropriately
    lattice_constant = 1.2 * SCALE  # Adjusted for simulation grid
    
    # Base positions for diamond structure (tetrahedral coordination)
    base_positions = [
        (0, 0, 0),      # Corner 1
        (0.5, 0.5, 0),  # Face center 1
        (0.5, 0, 0.5),  # Face center 2
        (0, 0.5, 0.5),  # Face center 3
        (0.25, 0.25, 0.25),  # Tetrahedral site 1
        (0.75, 0.75, 0.25),  # Tetrahedral site 2
        (0.75, 0.25, 0.75),  # Tetrahedral site 3
        (0.25, 0.75, 0.75),  # Tetrahedral site 4
    ]
    
    # Convert to actual grid coordinates, centering the crystal
    offset_x = center_x - lattice_constant
    offset_y = center_y - lattice_constant * 0.6  # Slightly offset for better view
    
    carbon_configs = []
    
    for i, (x_frac, y_frac, z_frac) in enumerate(base_positions):
        # Map 3D positions to 2D grid (project z onto xy plane with offset)
        x = offset_x + x_frac * lattice_constant * 2
        y = offset_y + y_frac * lattice_constant * 2 + z_frac * lattice_constant * 0.5
        
        # Ensure positions are within grid bounds
        x = max(50, min(SIZE - 50, x))
        y = max(50, min(SIZE - 50, y))
        
        # Each carbon has 6 electrons: 1s² 2s² 2p²
        carbon_configs.append(create_carbon_atom_config((x, y)))
    
    return carbon_configs


def create_graphene_hexagonal_ring_config():
    """Create a hexagonal ring of carbon atoms (graphene-like structure).
    
    Returns the carbon configurations without running the simulation.
    """
    import numpy as np
    
    # Hexagonal ring geometry
    ring_radius = 1.1 * SCALE
    num_carbons = 6
    
    carbon_configs = []
    
    for i in range(num_carbons):
        angle = 2 * np.pi * i / num_carbons
        x = center_x + ring_radius * np.cos(angle)
        y = center_y + ring_radius * np.sin(angle)
        
        # All carbons get proper 6-electron configuration: 1s² 2s² 2p²
        carbon_configs.append(create_carbon_atom_config((x, y)))
    
    return carbon_configs


def create_carbyne_chain(num_carbons=3, spacing=None):
    """Create a linear carbon chain (carbyne) configuration.
    
    Returns the carbon configurations without running the simulation.
    """
    if spacing is None:
        spacing = 0.9 * SCALE
    
    carbon_configs = []
    
    # Start position for the chain
    start_x = center_x - (num_carbons - 1) * spacing / 2
    
    for i in range(num_carbons):
        x = start_x + i * spacing
        y = center_y
        carbon_configs.append(create_carbon_atom_config((x, y)))
    
    return carbon_configs


def create_tetrahedral_carbon_cluster_4atom():
    """Create a 4-atom tetrahedral carbon cluster (1 central + 3 neighbors).
    
    Note: This is NOT a true diamond unit cell, but rather a small tetrahedral 
    cluster of 4 carbon atoms arranged in a 2D projection of tetrahedral geometry.
    A real diamond unit cell contains 8 carbon atoms in a specific 3D arrangement.
    
    Returns the carbon configurations without running the simulation.
    """
    # Tetrahedral positions for diamond structure
    bond_length = 0.8 * SCALE
    
    carbon_configs = [
        # Central carbon with proper 6-electron configuration
        create_carbon_atom_config((center_x, center_y)),
        # Surrounding carbons in tetrahedral positions
        create_carbon_atom_config((center_x + bond_length, center_y + bond_length)),
        create_carbon_atom_config((center_x - bond_length, center_y + bond_length)),
        create_carbon_atom_config((center_x, center_y - bond_length))
    ]
    
    return carbon_configs


def create_tetrahedral_carbon_cluster_5atom():
    """Create a 5-atom tetrahedral carbon cluster (1 central + 4 neighbors).
    
    Note: This is NOT a true diamond tetrahedral unit. Real tetrahedral geometry
    uses 109.5° angles, but this uses 90° spacing due to 2D projection constraints.
    
    Returns the carbon configurations without running the simulation.
    """
    import numpy as np
    
    # Tetrahedral geometry for diamond
    bond_length = 1.0 * SCALE
    
    # Central carbon atom
    central_pos = (center_x, center_y)
    
    # Four tetrahedral neighbors at 109.5° angles
    # In 2D projection, we approximate the 3D tetrahedral geometry
    tetrahedral_angles = [0, np.pi/2, np.pi, 3*np.pi/2]  # 90° spacing in 2D
    neighbor_positions = []
    
    for angle in tetrahedral_angles:
        x = center_x + bond_length * np.cos(angle)
        y = center_y + bond_length * np.sin(angle)
        neighbor_positions.append((x, y))
    
    carbon_configs = []
    
    # Central carbon with full 6 electrons: 1s² 2s² 2p²
    carbon_configs.append(create_carbon_atom_config(central_pos))
    
    # Four neighboring carbons with full 6-electron configurations
    for pos in neighbor_positions:
        carbon_configs.append(create_carbon_atom_config(pos))
    
    return carbon_configs


def create_c2_molecular_crystal_config(rows=3, cols=3, lattice_spacing=None):
    """Create a molecular crystal from multiple C2 dimer molecules arranged in a lattice.
    
    This creates a lattice of isolated C2 dimers, NOT a diamond crystal structure.
    Each dimer is essentially a separate C2 molecule arranged in a 2D grid.
    
    Returns the carbon configurations without running the simulation.
    """
    if lattice_spacing is None:
        lattice_spacing = 2.5 * SCALE
    
    # Calculate starting positions to center the crystal
    total_width = (cols - 1) * lattice_spacing
    total_height = (rows - 1) * lattice_spacing
    start_x = center_x - total_width / 2
    start_y = center_y - total_height / 2
    
    all_carbon_configs = []
    
    for row in range(rows):
        for col in range(cols):
            # Calculate center position for this dimer
            dimer_center_x = start_x + col * lattice_spacing
            dimer_center_y = start_y + row * lattice_spacing
            
            # Alternate orientation of dimers for more realistic crystal packing
            if (row + col) % 2 == 0:
                # Horizontal orientation
                dimer_configs = create_carbon_dimer(
                    (dimer_center_x, dimer_center_y), 
                    bond_length=0.8 * SCALE
                )
            else:
                # Vertical orientation (rotate by 90 degrees)
                dimer_configs = [
                    create_carbon_atom_config((dimer_center_x, dimer_center_y - 0.4 * SCALE)),
                    create_carbon_atom_config((dimer_center_x, dimer_center_y + 0.4 * SCALE))
                ]
            
            all_carbon_configs.extend(dimer_configs)
    
    return all_carbon_configs


if __name__ == "__main__":
    print("Diamond Molecule Configurations")
    print("==============================")
    print("This file provides carbon configurations for various structures.")
    print("For crystal simulations, use diamond_crystal_simulation.py")
    print()
    print("Choose basic simulation type:")
    print("1. Carbon dimer (C2)")
    print("2. Tetrahedral carbon cluster (4 atoms)")  
    print("3. Carbon chain (carbyne)")
    print("4. Tetrahedral carbon cluster (5 atoms)")
    print("5. Graphene hexagonal ring (6 atoms)")
    print("6. Diamond crystal 2x2x2 (8 atoms)")
    print("7. C2 molecular crystal lattice")
    print("8. All basic simulations (1-3)")
    
    choice = input("\nEnter choice (1-8) or press Enter for C2: ").strip()
    
    if choice == "2":
        # Tetrahedral carbon cluster simulation
        carbon_configs = create_tetrahedral_carbon_cluster_4atom()
        run_carbon_structure_simulation(
            carbon_configs,
            "Tetrahedral Carbon Cluster",
            "tetrahedral_cluster.avi",
            1500,
            extra_info="Four carbon atoms in tetrahedral arrangement (NOT true diamond unit cell)"
        )
    elif choice == "3":
        # Carbon chain simulation
        carbon_configs = create_carbyne_chain()
        run_carbon_structure_simulation(
            carbon_configs,
            "Carbon Chain (Carbyne)",
            "carbon_chain.avi",
            1800,
            extra_info="Linear chain of 3 carbon atoms"
        )
    elif choice == "4":
        # 5-atom tetrahedral carbon cluster simulation
        carbon_configs = create_tetrahedral_carbon_cluster_5atom()
        run_carbon_structure_simulation(
            carbon_configs,
            "Tetrahedral Carbon Cluster (5 atoms)",
            "tetrahedral_cluster_5atom.avi",
            1500,
            extra_info="Five carbon atoms: 1 central + 4 neighbors (NOT true diamond unit)"
        )
    elif choice == "5":
        # Graphene hexagonal ring simulation
        carbon_configs = create_graphene_hexagonal_ring_config()
        run_carbon_structure_simulation(
            carbon_configs,
            "Graphene Hexagonal Ring",
            "graphene_hexagonal_ring.avi",
            1200,
            extra_info="Six carbon atoms in hexagonal ring (graphene-like structure)"
        )
    elif choice == "6":
        # Diamond crystal 2x2x2 simulation
        carbon_configs = create_diamond_crystal_2x2x2_config()
        run_carbon_structure_simulation(
            carbon_configs,
            "Diamond Crystal 2x2x2",
            "diamond_crystal_2x2x2.avi",
            2000,
            extra_info="Eight carbon atoms in diamond crystal structure (3D projected to 2D)"
        )
    elif choice == "7":
        # C2 molecular crystal lattice simulation
        carbon_configs = create_c2_molecular_crystal_config()
        run_carbon_structure_simulation(
            carbon_configs,
            "C2 Molecular Crystal Lattice",
            "c2_molecular_crystal.avi",
            1800,
            extra_info="3x3 lattice of C2 dimer molecules (18 carbon atoms total)"
        )
    elif choice == "8":
        print("Running all basic carbon simulations...\n")
        
        # C2 dimer
        carbon_configs = create_carbon_dimer((center_x, center_y))
        run_carbon_structure_simulation(
            carbon_configs,
            "Carbon Dimer (C2)",
            "carbon_dimer_bonding.avi",
            2000,
            extra_info="Two carbon atoms forming molecular orbital"
        )
        print("\n" + "="*50 + "\n")
        
        # Tetrahedral carbon cluster
        carbon_configs = create_tetrahedral_carbon_cluster_4atom()
        run_carbon_structure_simulation(
            carbon_configs,
            "Tetrahedral Carbon Cluster",
            "tetrahedral_cluster.avi",
            1500,
            extra_info="Four carbon atoms in tetrahedral arrangement (NOT true diamond unit cell)"
        )
        print("\n" + "="*50 + "\n")
        
        # Carbon chain
        carbon_configs = create_carbyne_chain()
        run_carbon_structure_simulation(
            carbon_configs,
            "Carbon Chain (Carbyne)",
            "carbon_chain.avi",
            1800,
            extra_info="Linear chain of 3 carbon atoms"
        )
    else:
        # Default: C2 dimer
        carbon_configs = create_carbon_dimer((center_x, center_y))
        run_carbon_structure_simulation(
            carbon_configs,
            "Carbon Dimer (C2)",
            "carbon_dimer_bonding.avi",
            2000,
            extra_info="Two carbon atoms forming molecular orbital"
        )