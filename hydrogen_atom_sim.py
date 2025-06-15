#!/usr/bin/env python3
"""
Hydrogen Atom Orbital Progression Simulation

Demonstrates the evolution of a hydrogen atom through different orbitals:
1. 1s orbital (400 steps)
2. 2p orbital (400 steps) 
3. 3d orbital (400 steps)
4. 2p orbital with dynamics (800 steps)

Supports both 2D and 3D simulation modes with interactive selection.
"""

from unified_hybrid_molecular_simulation import (
    AtomConfig, create_atom_simulation, UnifiedHybridMolecularSimulation,
    create_atom_electron, ElectronInfo
)
from hybrid_molecular_simulation import MolecularNucleus
from particles import apply_wavefunction_dynamics
from config import center_x, center_y, center_z, SCALE, set_simulation_mode, get_coordinate_tensors
from video_utils import StreamingVideoWriter, open_video
import torch
import numpy as np
import sys


def interactive_mode_selection():
    """Interactive mode selection for hydrogen atom simulation."""
    print("=== Hydrogen Atom Orbital Progression ===")
    print("Choose simulation mode:")
    print("1. 2D mode (faster, good for testing)")
    print("2. 3D mode (slower, more realistic)")
    print("3. 2D mode with real-time visualization")
    print("4. 3D mode with real-time visualization")
    print("5. Both 2D and 3D comparison")
    print()
    
    try:
        choice = input("Enter your choice (1-5): ").strip()
        print(f"Selected option: {choice}")
        print()
        
        if choice == "1":
            create_hydrogen_orbital_progression("2D")
        elif choice == "2":
            create_hydrogen_orbital_progression("3D")
        elif choice == "3":
            create_hydrogen_orbital_progression_with_visualization("2D")
        elif choice == "4":
            create_hydrogen_orbital_progression_with_visualization("3D")
        elif choice == "5":
            create_hydrogen_orbital_comparison()
        else:
            print("Invalid choice. Using 2D mode.")
            create_hydrogen_orbital_progression("2D")
            
    except (EOFError, KeyboardInterrupt):
        print("\nInterrupted. Using 2D mode.")
        create_hydrogen_orbital_progression("2D")
    except Exception as e:
        print(f"Error in interactive mode: {e}")
        print("Running default 2D progression.")
        create_hydrogen_orbital_progression("2D")


def create_hydrogen_orbital_progression(mode: str = "2D") -> None:
    """Create and run a hydrogen atom simulation showing orbital progression."""
    # Set simulation mode
    set_simulation_mode(mode)
    
    print(f"=== Hydrogen Atom Orbital Progression - {mode} Mode ===")
    print("1. 1s orbital (400 steps)")
    print("2. 2p orbital (400 steps)")
    print("3. 3d orbital (400 steps)")
    print("4. 2p orbital with dynamics (800 steps)")
    print()
    
    # Set up video writer for the complete sequence
    video_filename = f"hydrogen_orbital_progression_{mode.lower()}.avi"
    video_writer = StreamingVideoWriter(
        output_file=video_filename,
        fps=24,
        sample_frames=50,
        keep_first_batch=True,
        first_batch_size=100
    )
    
    # Phase 1: 1s orbital (400 steps)
    print("Phase 1: Creating 1s orbital...")
    simulation = create_hydrogen_simulation((1, 0, 0), mode)
    run_orbital_phase(simulation, video_writer, 400, "1s orbital")
    
    # Phase 2: 2p orbital (400 steps)
    print("\nPhase 2: Transitioning to 2p orbital...")
    simulation = create_hydrogen_simulation((2, 1, 0), mode)
    run_orbital_phase(simulation, video_writer, 400, "2p orbital")
    
    # Phase 3: 3d orbital (400 steps)
    print("\nPhase 3: Transitioning to 3d orbital...")
    simulation = create_hydrogen_simulation((3, 2, 1), mode)
    run_orbital_phase(simulation, video_writer, 400, "3d orbital")
    
    # Phase 4: 2p orbital with dynamics (800 steps)
    print("\nPhase 4: 2p orbital with applied dynamics...")
    simulation = create_hydrogen_simulation_with_dynamics((2, 1, 0), mode)
    run_orbital_phase(simulation, video_writer, 800, "2p with dynamics")
    
    # Finalize video
    video_writer.finalize()
    print(f"\nHydrogen orbital progression simulation complete!")
    print(f"Video saved as: {video_filename}")
    
    # Open video for viewing
    try:
        open_video(video_filename)
    except Exception as e:
        print(f"Could not open video automatically: {e}")


def create_hydrogen_orbital_progression_with_visualization(mode: str):
    """Create hydrogen atom progression with real-time visualization."""
    try:
        # Import pygame visualization
        from pygame_quantum_viz import PygameQuantumViz
        
        # Set simulation mode
        set_simulation_mode(mode)
        
        print(f"=== Hydrogen Atom Interactive Visualization - {mode} Mode ===")
        print("Hydrogen atom orbital progression with real-time visualization")
        print("Controls:")
        print("  - Mouse: Rotate view")
        print("  - Mouse wheel: Zoom")
        print("  - SPACE: Pause/resume evolution")
        print("  - S: Manual evolution step")
        print("  - Up/Down arrows: Adjust threshold")
        print("  - ESC: Exit")
        print()
        
        # Create initial 1s simulation
        print("Starting with 1s orbital...")
        simulation = create_hydrogen_simulation((1, 0, 0), mode)
        
        print("Starting interactive visualization...")
        print("Note: If the window hangs, close it and run without visualization (options 1-2)")
        
        # Create and configure pygame visualization
        viz = PygameQuantumViz()
        viz.init_pygame()
        
        # Set the simulation directly
        viz.simulation = simulation
        
        # Set proper dimensions based on current mode
        from config import SIZE_X, SIZE_Y, SIZE_Z, SIMULATION_MODE
        viz.SIZE_X = SIZE_X
        viz.SIZE_Y = SIZE_Y  
        viz.SIZE_Z = SIZE_Z
        
        print(f"Visualization mode: {SIMULATION_MODE}")
        print(f"Grid dimensions: {SIZE_X}x{SIZE_Y}" + (f"x{SIZE_Z}" if SIMULATION_MODE == "3D" else ""))
        
        # Update quantum data and run visualization
        viz.update_quantum_data()
        print(f"Visualization initialized with {len(viz.points)} quantum points")
        viz.run()
        
        print(f"Hydrogen atom interactive visualization complete!")
        
    except ImportError as e:
        print(f"Pygame visualization not available: {e}")
        print("Falling back to video-only simulation...")
        create_hydrogen_orbital_progression(mode)
    except Exception as e:
        print(f"Visualization error: {e}")
        print("This may be due to display/graphics issues.")
        print("Falling back to video-only simulation...")
        create_hydrogen_orbital_progression(mode)


def create_hydrogen_orbital_comparison():
    """Create both 2D and 3D hydrogen atom simulations for comparison."""
    print("=== Hydrogen Atom 2D vs 3D Comparison ===")
    print("Creating both 2D and 3D orbital progressions...")
    print()
    
    print("Starting 2D simulation...")
    create_hydrogen_orbital_progression("2D")
    
    print("\nStarting 3D simulation...")
    create_hydrogen_orbital_progression("3D")
    
    print("\nComparison complete! Check both video files:")
    print("- hydrogen_orbital_progression_2d.avi")
    print("- hydrogen_orbital_progression_3d.avi")


def create_hydrogen_simulation(quantum_numbers, mode: str) -> UnifiedHybridMolecularSimulation:
    """Create a hydrogen atom simulation with specific quantum numbers."""
    # Get current coordinate tensors
    X, Y, Z = get_coordinate_tensors()
    
    # Create nucleus based on mode
    if mode == "3D":
        nuclei = [MolecularNucleus(center_x, center_y, center_z, atomic_number=1, atom_id=0)]
    else:
        nuclei = [MolecularNucleus(center_x, center_y, atomic_number=1, atom_id=0)]
    
    # Create electron wavefunction
    n, l, m = quantum_numbers
    
    # Scale down larger orbitals to fit in the simulation frame
    # n=1: scale = SCALE/10 (largest)
    # n=2: scale = SCALE/15 (medium) 
    # n=3: scale = SCALE/25 (smallest)
    orbital_scale = SCALE / (10 + 5 * (n - 1))
    print(f"  Using orbital scale: {orbital_scale:.2f} for n={n}")
    
    if mode == "3D":
        # 3D electron wavefunction
        psi = create_atom_electron(X, Y, Z, center_x, center_y, center_z, quantum_numbers, 
                                  atomic_number=1, scale=orbital_scale)
    else:
        # 2D electron wavefunction
        psi = create_atom_electron(X, Y, None, center_x, center_y, None, quantum_numbers, 
                                  atomic_number=1, scale=orbital_scale)
    
    # Convert to torch tensor
    unified_psi = torch.tensor(psi, dtype=torch.complex64)
    
    # Normalize
    norm = torch.sqrt(torch.sum(torch.abs(unified_psi)**2))
    if norm > 0:
        unified_psi = unified_psi / norm
    
    # Create electron info
    orbital_names = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
    orbital_name = f"{n}{orbital_names.get(l, f'l{l}')}"
    electron_infos = [ElectronInfo(atom_id=0, electron_name=orbital_name)]
    
    return UnifiedHybridMolecularSimulation(
        nuclei=nuclei,
        electron_infos=electron_infos,
        unified_wavefunction=unified_psi,
        electron_repulsion_strength=0.08,
        nuclear_motion_enabled=False,  # Single atom doesn't need nuclear motion
        damping_factor=0.999
    )


def create_hydrogen_simulation_with_dynamics(quantum_numbers, mode: str) -> UnifiedHybridMolecularSimulation:
    """Create a hydrogen atom simulation with applied dynamics."""
    # Get current coordinate tensors
    X, Y, Z = get_coordinate_tensors()
    
    # Create nucleus based on mode
    if mode == "3D":
        nuclei = [MolecularNucleus(center_x, center_y, center_z, atomic_number=1, atom_id=0)]
    else:
        nuclei = [MolecularNucleus(center_x, center_y, atomic_number=1, atom_id=0)]
    
    # Create electron wavefunction with improved scaling
    n, l, m = quantum_numbers
    
    # Apply scaling based on principal quantum number n to keep larger orbitals manageable
    if n == 1:
        scale_factor = SCALE / 10      # n=1 (1s): smallest, use base scale
    elif n == 2:
        scale_factor = SCALE / 15      # n=2 (2s,2p): slightly larger  
    elif n == 3:
        scale_factor = SCALE / 25      # n=3 (3s,3p,3d): much larger, scale down more
    elif n == 4:
        scale_factor = SCALE / 40      # n=4: even larger, scale down significantly
    else:
        scale_factor = SCALE / (10 + 15*n)  # n>=5: progressive scaling for higher orbitals
    
    if mode == "3D":
        # 3D electron wavefunction
        psi = create_atom_electron(X, Y, Z, center_x, center_y, center_z, quantum_numbers, 
                                  atomic_number=1, scale=scale_factor)
    else:
        # 2D electron wavefunction
        psi = create_atom_electron(X, Y, None, center_x, center_y, None, quantum_numbers, 
                                  atomic_number=1, scale=scale_factor)
    
    # Apply dynamics using particles.py function (only for 2D mode for now)
    if mode == "2D":
        print(f"  Applying dynamics to {n}{['s','p','d','f'][l]} orbital...")
        
        # Convert PyTorch tensors to numpy for particles.py function
        X_numpy = X.cpu().numpy()
        Y_numpy = Y.cpu().numpy()
        psi_numpy = psi if isinstance(psi, np.ndarray) else psi.cpu().numpy()
        
        psi_dynamic = apply_wavefunction_dynamics(
            psi_numpy, X_numpy, Y_numpy, center_x, center_y,
            momentum_x=0.08,      # Moderate momentum in x direction
            momentum_y=0.04,      # Slight momentum in y direction
            orbital_offset_x=2.0, # Offset the orbital center
            orbital_offset_y=1.0
        )
        psi = psi_dynamic
    else:
        print(f"  Note: Dynamics not yet implemented for 3D mode, using static {n}{['s','p','d','f'][l]} orbital")
    
    # Convert to torch tensor
    unified_psi = torch.tensor(psi, dtype=torch.complex64)
    
    # Normalize
    norm = torch.sqrt(torch.sum(torch.abs(unified_psi)**2))
    if norm > 0:
        unified_psi = unified_psi / norm
    
    # Create electron info
    orbital_names = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
    suffix = "_dynamic" if mode == "2D" else ""
    orbital_name = f"{n}{orbital_names.get(l, f'l{l}')}{suffix}"
    electron_infos = [ElectronInfo(atom_id=0, electron_name=orbital_name)]
    
    return UnifiedHybridMolecularSimulation(
        nuclei=nuclei,
        electron_infos=electron_infos,
        unified_wavefunction=unified_psi,
        electron_repulsion_strength=0.08,
        nuclear_motion_enabled=False,  # Single atom doesn't need nuclear motion
        damping_factor=0.999
    )


def run_orbital_phase(simulation: UnifiedHybridMolecularSimulation, 
                     video_writer: StreamingVideoWriter, 
                     num_steps: int, 
                     phase_name: str) -> None:
    """Run a phase of the orbital progression simulation."""
    from hybrid_molecular_simulation import gaussian_filter_torch
    
    print(f"  Running {num_steps} steps for {phase_name}...")
    
    for step in range(num_steps):
        if step % 100 == 0:
            print(f"    Step {step}/{num_steps}")
        
        # Evolve the system
        simulation.evolve_step(step)
        
        # Get visualization data
        combined_psi = simulation.get_combined_wavefunction()
        
        # Create frames
        frame_real = torch.real(combined_psi).cpu().numpy()
        frame_imag = torch.imag(combined_psi).cpu().numpy()
        frame_phase = torch.angle(gaussian_filter_torch(combined_psi, sigma=1)).cpu().numpy()
        frame_prob = torch.abs(combined_psi)**2
        
        # Normalize probability for visualization
        if torch.max(frame_prob) > 0:
            frame_prob = frame_prob / torch.max(frame_prob)
        
        frame_prob = frame_prob.cpu().numpy()
        
        video_writer.add_frame(frame_real, frame_imag, frame_phase, frame_prob)


def create_single_orbital_simulation(quantum_numbers, mode: str):
    """Create a simulation with a single specified orbital."""
    from video_utils import StreamingVideoWriter
    
    print(f"=== Single Hydrogen Orbital Simulation - {mode} Mode ===")
    
    n, l, m = quantum_numbers
    orbital_names = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
    orbital_name = f"{n}{orbital_names.get(l, f'l{l}')}"
    
    print(f"Simulating {orbital_name} orbital (n={n}, l={l}, m={m})")
    
    # Initialize video writer
    video_filename = f"hydrogen_{orbital_name}_{mode.lower()}.avi"
    print(f"Initializing streaming video writer: {video_filename}")
    video_writer = StreamingVideoWriter(video_filename, sample_frames=50, first_batch_size=100)
    
    try:
        # Create the simulation
        simulation = create_hydrogen_simulation(quantum_numbers, mode)
        
        # Run simulation
        print(f"Running 800 steps for {orbital_name} orbital...")
        run_orbital_phase(simulation, video_writer, 800, f"{orbital_name} orbital")
        
        print(f"Single orbital simulation complete! Video saved as: {video_filename}")
    
    finally:
        video_writer.close()


def create_single_orbital_with_visualization(quantum_numbers, mode: str):
    """Create a single orbital simulation with real-time visualization."""
    print(f"=== Single Hydrogen Orbital Visualization - {mode} Mode ===")
    
    n, l, m = quantum_numbers
    orbital_names = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
    orbital_name = f"{n}{orbital_names.get(l, f'l{l}')}"
    
    print(f"Visualizing {orbital_name} orbital (n={n}, l={l}, m={m}) in real-time")
    print("Press 'q' to quit, 'p' to pause/resume, 'r' to reset")
    
    # Create the simulation  
    simulation = create_hydrogen_simulation(quantum_numbers, mode)
    
    # Run real-time visualization
    from pygame_quantum_viz import run_pygame_visualization
    run_pygame_visualization(simulation, title=f"Hydrogen {orbital_name} Orbital - {mode}")


if __name__ == "__main__":
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Hydrogen Atom Orbital Simulation')
    parser.add_argument('--mode', choices=['2D', '3D', 'VIZ2D', 'VIZ3D', 'COMPARE'], 
                       help='Simulation mode (2D, 3D, VIZ2D, VIZ3D, COMPARE)')
    parser.add_argument('--orbital', type=str, 
                       help='Specific orbital to simulate in format n,l,m (e.g., 3,2,0)')
    parser.add_argument('--viz', action='store_true', 
                       help='Enable real-time visualization')
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        # Check if using new argument format
        if sys.argv[1].startswith('--'):
            args = parser.parse_args()
            
            if args.orbital:
                # Parse orbital quantum numbers
                try:
                    n, l, m = map(int, args.orbital.split(','))
                    quantum_numbers = (n, l, m)
                    
                    # Determine mode
                    if args.mode:
                        mode = args.mode
                    else:
                        mode = "2D"  # default
                    
                    # Add visualization suffix if needed
                    if args.viz and not mode.startswith('VIZ'):
                        mode = f"VIZ{mode}"
                    
                    print(f"Running single orbital simulation: {n}{['s','p','d','f'][l]} in {mode} mode")
                    
                    if mode == "VIZ2D":
                        create_single_orbital_with_visualization(quantum_numbers, "2D")
                    elif mode == "VIZ3D":
                        create_single_orbital_with_visualization(quantum_numbers, "3D")
                    else:
                        create_single_orbital_simulation(quantum_numbers, mode)
                        
                except ValueError:
                    print("Error: Orbital format should be n,l,m (e.g., 3,2,0)")
                    sys.exit(1)
            elif args.mode:
                # Run standard progression
                if args.mode in ["2D", "3D"]:
                    create_hydrogen_orbital_progression(args.mode)
                elif args.mode == "VIZ2D":
                    create_hydrogen_orbital_progression_with_visualization("2D")
                elif args.mode == "VIZ3D":
                    create_hydrogen_orbital_progression_with_visualization("3D")
                elif args.mode == "COMPARE":
                    create_hydrogen_orbital_comparison()
            else:
                print("Error: Must specify either --mode or --orbital")
                parser.print_help()
        else:
            # Legacy single argument format
            mode = sys.argv[1].upper()
            if mode in ["2D", "3D"]:
                create_hydrogen_orbital_progression(mode)
            elif mode == "VIZ2D":
                create_hydrogen_orbital_progression_with_visualization("2D")
            elif mode == "VIZ3D":
                create_hydrogen_orbital_progression_with_visualization("3D")
            elif mode == "COMPARE":
                create_hydrogen_orbital_comparison()
            else:
                print(f"Unknown mode: {mode}")
                print("Valid modes: 2D, 3D, VIZ2D, VIZ3D, COMPARE")
                interactive_mode_selection()
    else:
        # Interactive mode selection
        interactive_mode_selection()
