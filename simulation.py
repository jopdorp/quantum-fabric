"""
Quantum Atom Simulation Module

This module provides a reusable simulation framework for arbitrary atoms and electrons.
It handles the simulation of multi-electron, multi-nuclear systems with proper
electron-electron interactions and nuclear forces.
"""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from typing import List, Tuple, Dict, Optional, Callable, Union

from config import TIME_STEPS, X, Y, SIZE
from frame_utils import limit_frame
from physics import (
    create_nucleus_potential,
    compute_force_from_density,
)
from torch_physics import propagate_wave_with_potential
from video_utils import StreamingVideoWriter, open_video


def get_default_repulsion_sigmas() -> Tuple[float, float]:
    """
    Get default Gaussian sigmas for electron-electron repulsion scaled to simulation size.
    
    Returns:
        Tuple of (short_range_sigma, long_range_sigma) for dual-range repulsion
    """
    from config import SCALE
    
    # Scale sigmas with the simulation scale
    # Base values that work well at standard scale, then scale proportionally
    base_short_range = 2.0
    base_long_range = 8.0
    
    # Scale with SCALE to maintain proper physics at different zoom levels
    short_range = base_short_range * SCALE / 100.0  # Assuming SCALE ~100 is "standard"
    long_range = base_long_range * SCALE / 100.0
    
    # Ensure minimum values for numerical stability
    short_range = max(1.0, short_range)
    long_range = max(4.0, long_range)
    
    return short_range, long_range


class Nucleus:
    """Represents a nucleus with position, charge, and velocity."""
    
    def __init__(self, x: float, y: float, charge: int = 1, mass_ratio: float = 1836.0):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.zeros(2, dtype=float)
        self.charge = charge
        self.mass_ratio = mass_ratio  # Ratio to electron mass
        
    def update_position(self, force: np.ndarray, dt: float):
        """Update nucleus position using classical dynamics."""
        self.velocity += dt * force / self.mass_ratio
        self.position += dt * self.velocity


class Electron:
    """Represents an electron wavefunction with metadata."""
    
    def __init__(self, wavefunction: np.ndarray, name: str = "electron", 
                 nucleus_index: int = 0):
        self.wavefunction = wavefunction
        self.name = name
        self.nucleus_index = nucleus_index  # Which nucleus this electron is bound to
        
    def get_density(self) -> torch.Tensor:
        """Get the probability density of this electron."""
        # Ensure data is a torch tensor
        if not isinstance(self.wavefunction, torch.Tensor):
            self.wavefunction = torch.tensor(self.wavefunction, dtype=torch.complex64)
        
        return torch.abs(self.wavefunction)**2
        
    def normalize(self):
        """Normalize the wavefunction."""
        # Ensure data is a torch tensor
        if not isinstance(self.wavefunction, torch.Tensor):
            self.wavefunction = torch.tensor(self.wavefunction, dtype=torch.complex64)
        
        norm = torch.sqrt(torch.sum(torch.abs(self.wavefunction)**2))
        if norm > 0:
            self.wavefunction = self.wavefunction / norm


class AtomSimulation:
    """Main simulation class for multi-electron, multi-nuclear systems."""
    
    def __init__(self, nuclei: List[Nucleus], electrons: List[Electron],
                 electron_repulsion_strength: float = 0.1,
                 enable_nuclear_motion: bool = False,
                 orbital_mixing_strength: float = 0.1,
                 mixing_frequency: int = 400,
                 repulsion_sigmas: Union[float, Tuple[float, float]] = None,
                 bond_spring_constant: float = 0.0,
                 equilibrium_bond_length: float = None):
        """
        Initialize the atom simulation.
        
        Args:
            nuclei: List of Nucleus objects
            electrons: List of Electron objects
            electron_repulsion_strength: Strength of electron-electron repulsion
            enable_nuclear_motion: Whether nuclei can move (for molecules)
            orbital_mixing_strength: Strength of orbital mixing between electrons
            mixing_frequency: How often to apply orbital mixing (in steps)
            repulsion_sigmas: Either single sigma (backward compatibility) or tuple of (short_range_sigma, long_range_sigma)
            bond_spring_constant: Spring constant for bond length stabilization (0 = disabled)
            equilibrium_bond_length: Target bond length for stabilization (auto-detect if None)
        """
        self.nuclei = nuclei
        self.electrons = electrons
        self.electron_repulsion_strength = electron_repulsion_strength
        self.enable_nuclear_motion = enable_nuclear_motion
        self.orbital_mixing_strength = orbital_mixing_strength
        self.mixing_frequency = mixing_frequency
        self.bond_spring_constant = bond_spring_constant
        
        # Auto-detect equilibrium bond length from initial nuclear positions
        if equilibrium_bond_length is None and len(nuclei) == 2:
            dx = nuclei[1].position[0] - nuclei[0].position[0]
            dy = nuclei[1].position[1] - nuclei[0].position[1]
            self.equilibrium_bond_length = np.sqrt(dx**2 + dy**2)
        else:
            self.equilibrium_bond_length = equilibrium_bond_length
        
        # Auto-compute repulsion_sigma based on typical orbital size if not provided
        if repulsion_sigmas is None:
            # Use default sigmas that work well for atomic simulations
            self.short_range_sigma, self.long_range_sigma = get_default_repulsion_sigmas()
        else:
            # For backward compatibility, if single sigma provided, use it for short range and 4x for long range
            if isinstance(repulsion_sigmas, (int, float)):
                self.short_range_sigma = repulsion_sigmas
                self.long_range_sigma = repulsion_sigmas * 4.0
            else:
                # Assume tuple of (short, long) sigmas
                self.short_range_sigma, self.long_range_sigma = repulsion_sigmas
        
        # Apply initial frame limits
        for electron in self.electrons:
            electron.wavefunction = limit_frame(electron.wavefunction)
    
    def compute_electron_electron_repulsion(self, target_electron_index: int) -> np.ndarray:
        """Compute repulsive potential from other electrons using dual-range interactions."""
        repulsion = np.zeros_like(X, dtype=float)
        
        for i, other_electron in enumerate(self.electrons):
            if i != target_electron_index:
                other_density = other_electron.get_density()
                
                # Convert to numpy if it's a torch tensor
                if hasattr(other_density, 'cpu'):  # Check if it's a torch tensor
                    other_density_np = other_density.cpu().numpy()
                else:
                    other_density_np = other_density
                
                # Dual-range repulsion: short-range (strong) + long-range (weak)
                # This matches the physics.py enhanced_electron_electron_repulsion approach
                short_range_repulsion = gaussian_filter(other_density_np, sigma=self.short_range_sigma) * 2.0
                long_range_repulsion = gaussian_filter(other_density_np, sigma=self.long_range_sigma) * 0.5
                
                repulsion += short_range_repulsion + long_range_repulsion
        
        return self.electron_repulsion_strength * repulsion
    
    def compute_nuclear_forces(self) -> List[np.ndarray]:
        """Compute forces on each nucleus from electrons and other nuclei."""
        forces = [np.zeros(2) for _ in self.nuclei]
        
        # Forces from ALL electrons (not just bound ones)
        # This is crucial for proper molecular bonding forces
        for i, nucleus in enumerate(self.nuclei):
            for electron in self.electrons:
                # All electrons exert forces on all nuclei
                density = electron.get_density()
                force = compute_force_from_density(density, nucleus.position)
                
                # If electron is bound to this nucleus: attractive force (negative charge)
                # If electron is bound to another nucleus: can be attractive (bonding) or repulsive
                if electron.nucleus_index == i:
                    forces[i] += force  # Attractive force from own electrons
                else:
                    # Inter-nuclear electron forces - these create molecular bonding
                    # Increased coupling for stronger molecular bonds
                    forces[i] += force * 0.6  # Increased from 0.3 to 0.6 for stronger bonding
        
        # Nuclear-nuclear forces (for molecules)
        if len(self.nuclei) > 1:
            for i, nucleus1 in enumerate(self.nuclei):
                for j, nucleus2 in enumerate(self.nuclei):
                    if i != j:
                        # Strong Coulomb repulsion between nuclei
                        dx = nucleus2.position[0] - nucleus1.position[0]
                        dy = nucleus2.position[1] - nucleus1.position[1]
                        r = np.sqrt(dx**2 + dy**2)
                        
                        # Prevent complete overlap with minimum separation
                        r = max(r, 3.0)  # Minimum nuclear separation (3 pixels)
                        
                        # FIXED: Proper Coulomb force F = k*q1*q2/r^2 (not r^3!)
                        # Much stronger coefficient to prevent nuclear overlap
                        force_magnitude = 10.0 * nucleus1.charge * nucleus2.charge / (r**2)
                        
                        # Direction: repulsive (away from other nucleus)
                        force_direction_x = dx / r  # Unit vector
                        force_direction_y = dy / r
                        
                        forces[i][0] += force_magnitude * force_direction_x
                        forces[i][1] += force_magnitude * force_direction_y
                        
                        # Add harmonic bond stabilization for diatomic molecules
                        if (self.bond_spring_constant > 0 and 
                            self.equilibrium_bond_length is not None and 
                            len(self.nuclei) == 2):  # Only for diatomic molecules
                            
                            # Harmonic restoring force: F = -k*(r - r0)
                            displacement = r - self.equilibrium_bond_length
                            spring_force_magnitude = -self.bond_spring_constant * displacement / r
                            forces[i][0] += spring_force_magnitude * dx
                            forces[i][1] += spring_force_magnitude * dy
        
        return forces
    
    def apply_orbital_mixing(self, step: int):
        """Apply orbital mixing between electrons for visual dynamics."""
        if len(self.electrons) > 1 and step % self.mixing_frequency == 0 and step > 0:
            mix_strength = self.orbital_mixing_strength * np.sin(step * 0.01)
            
            # Create temporary copies for mixing
            temp_electrons = [np.copy(e.wavefunction) for e in self.electrons]
            
            # Apply circular mixing pattern
            for i in range(len(self.electrons)):
                next_i = (i + 1) % len(self.electrons)
                self.electrons[i].wavefunction += mix_strength * temp_electrons[next_i]
            
            # Renormalize all electrons
            for electron in self.electrons:
                electron.normalize()
    
    def evolve_step(self, step: int):
        """Evolve the system by one time step."""
        
        # Apply orbital mixing for dynamics
        self.apply_orbital_mixing(step)
        
        # Evolve each electron
        for i, electron in enumerate(self.electrons):
            # Get the nucleus this electron is bound to
            nucleus = self.nuclei[electron.nucleus_index]
            
            # Create nuclear potential
            V = create_nucleus_potential(X, Y, *nucleus.position, nucleus.charge)
            
            # Add electron-electron repulsion
            V += self.compute_electron_electron_repulsion(i)
            
            # Propagate wavefunction
            electron.wavefunction = propagate_wave_with_potential(electron.wavefunction, V)
            electron.wavefunction = limit_frame(electron.wavefunction)
        
        # Update nuclear positions if enabled
        if self.enable_nuclear_motion:
            forces = self.compute_nuclear_forces()
            for i, nucleus in enumerate(self.nuclei):
                nucleus.update_position(forces[i], dt=4.0)  # TIME_DELTA from config
                
                # Add velocity damping to prevent runaway oscillations
                damping_factor = 0.995  # Slight velocity damping
                nucleus.velocity *= damping_factor
    
    def get_combined_wavefunction(self, combination_method: str = "superposition") -> np.ndarray:
        """
        Combine all electron wavefunctions for visualization.
        
        Args:
            combination_method: "superposition" or "phase_weighted"
        """
        if not self.electrons:
            return np.zeros_like(X, dtype=complex)
        
        if combination_method == "superposition":
            # Simple superposition
            combined = sum(e.wavefunction for e in self.electrons) / len(self.electrons)
        
        elif combination_method == "phase_weighted":
            # Weighted superposition with different phases
            combined = np.zeros_like(self.electrons[0].wavefunction, dtype=complex)
            for i, electron in enumerate(self.electrons):
                phase_factor = np.exp(1j * i * 0.5)  # Different phase for each electron
                combined += electron.wavefunction * phase_factor
            combined /= len(self.electrons)
        
        else:
            raise ValueError(f"Unknown combination method: {combination_method}")
        
        return combined


def run_simulation(nuclei: List[Nucleus], electrons: List[Electron],
                  video_file: str = "atom_simulation.avi",
                  fps: int = 24,
                  progress_callback: Optional[Callable[[int, int], None]] = None,
                  **simulation_kwargs) -> AtomSimulation:
    """
    Run a complete atom simulation and save to video.
    
    Args:
        nuclei: List of Nucleus objects
        electrons: List of Electron objects
        video_file: Output video filename
        fps: Video frame rate
        progress_callback: Optional callback function for progress updates
        **simulation_kwargs: Additional arguments for AtomSimulation
    
    Returns:
        The simulation object after completion
    """
    
    # Create simulation
    sim = AtomSimulation(nuclei, electrons, **simulation_kwargs)
    
    # Set up video writer
    video_writer = StreamingVideoWriter(
        output_file=video_file,
        fps=fps,
        sample_frames=50,
        keep_first_batch=True,
        first_batch_size=100
    )
    
    # Report memory usage
    memory_info = video_writer.get_memory_usage_estimate((SIZE, SIZE))
    print(f"Memory usage estimate:")
    print(f"  - Frames in memory: {memory_info['frames_in_memory']}")
    print(f"  - Memory per frame: {memory_info['memory_per_frame_mb']:.1f} MB")
    print(f"  - Total memory for frames: {memory_info['total_memory_mb']:.1f} MB")
    
    print(f"Starting simulation with {len(nuclei)} nuclei and {len(electrons)} electrons...")
    
    # Main simulation loop
    for step in range(TIME_STEPS):
        if step % 200 == 0:
            print(f"Step {step}/{TIME_STEPS}...")
            
        if progress_callback:
            progress_callback(step, TIME_STEPS)
        
        # Evolve the system
        sim.evolve_step(step)
        
        # Get combined wavefunction for visualization
        combined_psi = sim.get_combined_wavefunction("superposition")
        
        # Create video frames
        if isinstance(combined_psi, torch.Tensor):
            # Convert to numpy for video processing
            combined_psi_np = combined_psi.cpu().numpy()
            frame_real = np.real(combined_psi_np)
            frame_imag = np.imag(combined_psi_np)
            frame_phase = np.angle(gaussian_filter(combined_psi_np, sigma=1))
            frame_prob = np.abs(combined_psi_np)**2
        else:
            frame_real = np.real(combined_psi)
            frame_imag = np.imag(combined_psi)
            frame_phase = np.angle(gaussian_filter(combined_psi, sigma=1))
            frame_prob = np.abs(combined_psi)**2
        
        frame_prob = frame_prob / np.max(frame_prob) if np.max(frame_prob) > 0 else frame_prob
        
        video_writer.add_frame(frame_real, frame_imag, frame_phase, frame_prob)
    
    # Finalize video
    video_writer.finalize()
    open_video(video_file)
    print(f"Simulation complete! Video saved as {video_file}")
    
    return sim


def create_simple_atom_simulation(nucleus_x: float, nucleus_y: float, 
                                electron_wavefunctions: List[np.ndarray],
                                electron_names: List[str] = None,
                                atomic_number: int = 1) -> Tuple[List[Nucleus], List[Electron]]:
    """
    Helper function to create a simple single-atom simulation.
    
    Args:
        nucleus_x, nucleus_y: Nucleus position
        electron_wavefunctions: List of electron wavefunctions
        electron_names: Optional names for electrons
        atomic_number: Nuclear charge
    
    Returns:
        Tuple of (nuclei, electrons) lists
    """
    # Create nucleus
    nucleus = Nucleus(nucleus_x, nucleus_y, charge=atomic_number)
    nuclei = [nucleus]
    
    # Create electrons
    electrons = []
    for i, psi in enumerate(electron_wavefunctions):
        name = electron_names[i] if electron_names and i < len(electron_names) else f"electron_{i}"
        electron = Electron(psi, name=name, nucleus_index=0)
        electrons.append(electron)
    
    return nuclei, electrons
