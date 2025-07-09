#!/usr/bin/env python3
"""
Hydrogen Molecule (H2) Simulation using Unified Framework

Demonstrates the simulation of an H2 molecule using the unified
molecular simulation framework with proper bonding dynamics.

Supports both 2D and 3D simulation modes.
"""

import sys
from unified_hybrid_molecular_simulation import (
    AtomConfig, create_molecule_simulation, run_simulation
)
from config import center_x, center_y, center_z, SCALE, set_simulation_mode

def create_hydrogen_molecule(bond_type: str = "bonding", mode: str = "2D") -> None:
    """Create and run an H2 molecule simulation.
    
    Args:
        bond_type: "bonding" or "antibonding" molecular orbital type
        mode: "2D" or "3D" simulation mode
    """
    # Set simulation mode
    set_simulation_mode(mode)
    
    print(f"=== Hydrogen Molecule (H2) Simulation - {mode} Mode ===")
    print(f"Two hydrogen atoms forming {bond_type} molecular orbital")
    print()
    
    # Bond length for H2
    bond_length = 1.2 * SCALE
    
    # Calculate true geometric center for perfect symmetry
    from config import SIZE_X, SIZE_Y, SIZE_Z
    true_center_x = (SIZE_X - 1) / 2.0
    true_center_y = (SIZE_Y - 1) / 2.0
    true_center_z = (SIZE_Z - 1) / 2.0
    
    print(f"Using true geometric center: ({true_center_x:.3f}, {true_center_y:.3f})")
    print(f"Bond length: {bond_length:.3f}, placing nuclei at symmetric positions")
    
    # Configure hydrogen atoms based on mode
    if mode == "3D":
        # 3D positions
        hydrogen_configs = [
            AtomConfig(
                atomic_number=1, 
                position=(true_center_x - bond_length/2, true_center_y, true_center_z),
                electron_configs=[(1, 0, 0)]  # 1s electron
            ),
            AtomConfig(
                atomic_number=1, 
                position=(true_center_x + bond_length/2, true_center_y, true_center_z),
                electron_configs=[(1, 0, 0)]  # 1s electron
            )
        ]
    else:
        # 2D positions
        hydrogen_configs = [
            AtomConfig(
                atomic_number=1, 
                position=(true_center_x - bond_length/2, true_center_y),
                electron_configs=[(1, 0, 0)]  # 1s electron
            ),
            AtomConfig(
                atomic_number=1, 
                position=(true_center_x + bond_length/2, true_center_y),
                electron_configs=[(1, 0, 0)]  # 1s electron
            )
        ]
    
    print(f"Creating H2 molecule simulation in {mode} mode...")
    simulation = create_molecule_simulation(hydrogen_configs, bond_type=bond_type)
    
    print("Running simulation...")
    video_name = f"hydrogen_molecule_{bond_type}_{mode.lower()}.avi"
    run_simulation(simulation, video_name, time_steps=2000)
    
    print(f"H2 molecule ({bond_type}) simulation complete in {mode} mode!")


def create_h2_comparison(mode: str = "2D") -> None:
    """Create both bonding and antibonding H2 simulations for comparison."""
    print(f"=== H2 Bonding vs Antibonding Comparison - {mode} Mode ===")
    print()
    
    # Create bonding H2
    print("1. Creating bonding H2...")
    create_hydrogen_molecule("bonding", mode)
    
    print()
    
    # Create antibonding H2
    print("2. Creating antibonding H2...")
    create_hydrogen_molecule("antibonding", mode)
    
    print(f"\nComparison complete in {mode} mode!")
    print(f"Check videos: hydrogen_molecule_bonding_{mode.lower()}.avi vs hydrogen_molecule_antibonding_{mode.lower()}.avi")


def create_2d_vs_3d_comparison() -> None:
    """Compare 2D vs 3D simulations of the same H2 molecule."""
    print("=== H2 2D vs 3D Mode Comparison ===")
    print()
    
    # Test 2D mode
    print("1. Testing 2D mode...")
    create_hydrogen_molecule("bonding", "2D")
    
    print()
    
    # Test 3D mode  
    print("2. Testing 3D mode...")
    create_hydrogen_molecule("bonding", "3D")
    
    print("\n2D vs 3D comparison complete!")
    print("Check videos: hydrogen_molecule_bonding_2d.avi vs hydrogen_molecule_bonding_3d.avi")


def interactive_mode_selection():
    """Interactive mode selection for testing."""
    print("=== Hydrogen Molecule Simulation ===")
    print("Choose simulation mode:")
    print("1. 2D mode (faster, good for testing)")
    print("2. 3D mode (slower, more realistic)")
    print("3. 2D mode with real-time visualization")
    print("4. 3D mode with real-time visualization") 
    print("5. Both 2D and 3D comparison")
    print("6. Bonding vs Antibonding in 2D")
    print("7. Bonding vs Antibonding in 3D")
    print()  # Add blank line
    
    try:
        # Use input() with explicit flush
        import sys
        sys.stdout.flush()
        choice = input("Enter your choice (1-7): ").strip()
        print(f"Selected option: {choice}")  # Confirm selection
        print()
        
        if choice == "1":
            print("Running 2D mode simulation...")
            create_hydrogen_molecule("bonding", "2D")
        elif choice == "2":
            print("Running 3D mode simulation...")
            create_hydrogen_molecule("bonding", "3D")
        elif choice == "3":
            print("Starting 2D visualization...")
            create_hydrogen_molecule_with_visualization("bonding", "2D")
        elif choice == "4":
            print("Starting 3D visualization...")
            create_hydrogen_molecule_with_visualization("bonding", "3D")
        elif choice == "5":
            print("Running 2D vs 3D comparison...")
            create_2d_vs_3d_comparison()
        elif choice == "6":
            print("Running 2D bonding vs antibonding comparison...")
            create_h2_comparison("2D")
        elif choice == "7":
            print("Running 3D bonding vs antibonding comparison...")
            create_h2_comparison("3D")
        else:
            print("Invalid choice. Running default 2D bonding simulation.")
            create_hydrogen_molecule("bonding", "2D")
            
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted. Running default 2D bonding simulation.")
        create_hydrogen_molecule("bonding", "2D")
    except Exception as e:
        print(f"Error in interactive mode: {e}")
        print("Running default 2D bonding simulation.")
        create_hydrogen_molecule("bonding", "2D")


def create_hydrogen_molecule_with_visualization(bond_type: str = "bonding", mode: str = "2D") -> None:
    """Create H2 molecule simulation with real-time pygame visualization.
    
    Args:
        bond_type: "bonding" or "antibonding" molecular orbital type
        mode: "2D" or "3D" simulation mode
    """
    try:
        # Check if we have a display available
        import os
        if 'DISPLAY' not in os.environ and os.name != 'nt':
            print("No display available. Falling back to video-only simulation.")
            create_hydrogen_molecule(bond_type, mode)
            return
            
        # Import pygame visualization
        from pygame_quantum_viz import PygameQuantumViz
        
        # Set simulation mode
        set_simulation_mode(mode)
        
        print(f"=== H2 Molecule Interactive Visualization - {mode} Mode ===")
        print(f"Two hydrogen atoms forming {bond_type} molecular orbital")
        print("Controls:")
        print("  - Mouse: Rotate view")
        print("  - Mouse wheel: Zoom")
        print("  - SPACE: Pause/resume evolution")
        print("  - S: Manual evolution step")
        print("  - Up/Down arrows: Adjust threshold")
        print("  - ESC: Exit")
        print()
        
        # Bond length for H2
        bond_length = 1.2 * SCALE
        
        # Configure hydrogen atoms based on mode
        if mode == "3D":
            # 3D positions
            hydrogen_configs = [
                AtomConfig(
                    atomic_number=1, 
                    position=(center_x - bond_length/2, center_y, center_z),
                    electron_configs=[(1, 0, 0)]  # 1s electron
                ),
                AtomConfig(
                    atomic_number=1, 
                    position=(center_x + bond_length/2, center_y, center_z),
                    electron_configs=[(1, 0, 0)]  # 1s electron
                )
            ]
        else:
            # 2D positions
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
        
        print(f"Creating H2 molecule simulation in {mode} mode...")
        simulation = create_molecule_simulation(hydrogen_configs, bond_type=bond_type)
        
        print("Starting interactive visualization...")
        print("Note: If the window hangs, close it and run without visualization (options 1-2)")
        
        # Create and configure pygame visualization
        viz = PygameQuantumViz()
        viz.init_pygame()
        
        # Set the simulation directly instead of creating a default one
        viz.simulation = simulation
        
        # Set proper dimensions based on the actual wavefunction and current mode
        from config import SIZE_X, SIZE_Y, SIZE_Z, SIMULATION_MODE
        
        # Use the grid dimensions from the current simulation mode
        viz.SIZE_X = SIZE_X
        viz.SIZE_Y = SIZE_Y  
        viz.SIZE_Z = SIZE_Z
        
        print(f"Visualization mode: {SIMULATION_MODE}")
        print(f"Grid dimensions: {SIZE_X}x{SIZE_Y}" + (f"x{SIZE_Z}" if SIMULATION_MODE == "3D" else ""))
        
        # Update quantum data and run visualization
        viz.update_quantum_data()
        print(f"Visualization initialized with {len(viz.points)} quantum points")
        viz.run()
        
        print(f"H2 molecule ({bond_type}) interactive visualization complete!")
        
    except ImportError as e:
        print(f"Pygame visualization not available: {e}")
        print("Falling back to video-only simulation...")
        create_hydrogen_molecule(bond_type, mode)
    except Exception as e:
        print(f"Visualization error: {e}")
        print("This may be due to display/graphics issues.")
        print("Falling back to video-only simulation...")
        create_hydrogen_molecule(bond_type, mode)


if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].upper()
        if mode in ["2D", "3D"]:
            create_hydrogen_molecule("bonding", mode)
        elif mode == "VIZ2D":
            create_hydrogen_molecule_with_visualization("bonding", "2D")
        elif mode == "VIZ3D":
            create_hydrogen_molecule_with_visualization("bonding", "3D")
        elif mode == "COMPARE":
            create_2d_vs_3d_comparison()
        elif mode == "INTERACTIVE":
            interactive_mode_selection()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python hydrogen_molecule_sim.py [2D|3D|VIZ2D|VIZ3D|COMPARE|INTERACTIVE]")
            create_hydrogen_molecule("bonding", "2D")  # Default to 2D
    else:
        # Interactive mode by default
        interactive_mode_selection()
    
    # Uncomment to create comparison:
    # create_h2_comparison("2D")
    # create_2d_vs_3d_comparison()
