#!/usr/bin/env python3
"""
Hybrid Molecular Simulation - Individual Electrons with Unified Forces

This approach combines the best of both worlds:
1. Individual electron wavefunctions that evolve separately (like original simulation)
2. Unified field approach for computing inter-electron forces
3. Realistic molecular behavior through prop    # Create electrons with initial momentum toward each other (3x bigger for better visualization)
    print("Creating electron 1 (moving toward center)...")
    psi1 = create_atom_electron(X, Y, nucleus1_x, nucleus_y, (1, 0, 0), 
                               atomic_number=1, alpha=0.11)  # alpha=0.11 makes atoms ~3x bigger
    psi1 = apply_wavefunction_dynamics(psi1, X, Y, nucleus1_x, nucleus_y,
                                     momentum_x=0.08, momentum_y=0.02,  # Moving right
                                     orbital_offset_x=1.0, orbital_offset_y=0.0)
    
    print("Creating electron 2 (moving toward center)...")
    psi2 = create_atom_electron(X, Y, nucleus2_x, nucleus_y, (1, 0, 0),
                               atomic_number=1, alpha=0.11)  # alpha=0.11 makes atoms ~3x bigger-electron interactions
"""

import torch
import torch.nn.functional as F
from typing import List
import math
from scipy.special import genlaguerre
import numpy as np

# Import existing components
from config import SIZE_X, SIZE_Y, SIZE_Z, TIME_STEPS, SCALE, center_x, center_y, center_z, POTENTIAL_STRENGTH, NUCLEAR_REPULSION_STRENGTH, NUCLEAR_CORE_RADIUS, DEVICE, X, Y, Z
from torch_physics import propagate_wave_batch_with_potentials
from video_utils import StreamingVideoWriter, open_video

def gaussian(M, std, device=None):
    """PyTorch version of Gaussian window function"""
    if M < 1:
        return torch.tensor([], device=device)
    if M == 1:
        return torch.ones(1, dtype=torch.float32, device=device)
    
    n = torch.arange(0, M, dtype=torch.float32, device=device) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w / w.sum()  # Normalize

def gaussian_filter_torch(input_tensor, sigma):
    """PyTorch implementation of Gaussian filter using separable convolution - supports 2D and 3D"""
    device = input_tensor.device
    
    # Handle complex tensors
    if input_tensor.dtype.is_complex:
        real_part = gaussian_filter_torch(input_tensor.real, sigma)
        imag_part = gaussian_filter_torch(input_tensor.imag, sigma)
        return torch.complex(real_part, imag_part)
    
    # Calculate kernel size (should be odd and cover ~3 standard deviations)
    kernel_size = int(2 * torch.ceil(torch.tensor(3 * sigma)).item()) + 1
    
    # Create 1D Gaussian kernel
    kernel_1d = gaussian(kernel_size, sigma, device=device)
    
    if len(input_tensor.shape) == 3:  # 3D case
        # Add batch and channel dimensions to input
        input_5d = input_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        
        # Apply separable 3D convolution
        # First apply along Z dimension
        kernel_z = kernel_1d.view(1, 1, kernel_size, 1, 1)
        padding_z = kernel_size // 2
        filtered_z = F.conv3d(input_5d, kernel_z, padding=(padding_z, 0, 0))
        
        # Then apply along Y dimension
        kernel_y = kernel_1d.view(1, 1, 1, kernel_size, 1)
        padding_y = kernel_size // 2
        filtered_zy = F.conv3d(filtered_z, kernel_y, padding=(0, padding_y, 0))
        
        # Finally apply along X dimension
        kernel_x = kernel_1d.view(1, 1, 1, 1, kernel_size)
        padding_x = kernel_size // 2
        filtered_zyx = F.conv3d(filtered_zy, kernel_x, padding=(0, 0, padding_x))
        
        # Remove batch and channel dimensions
        return filtered_zyx.squeeze(0).squeeze(0)
    
    else:  # 2D case (backward compatibility)
        # Reshape for convolution: [out_channels, in_channels, kernel_size]
        kernel_h = kernel_1d.view(1, 1, kernel_size)  # Horizontal kernel
        kernel_v = kernel_1d.view(1, 1, kernel_size, 1)  # Vertical kernel
        
        # Add batch and channel dimensions to input
        input_4d = input_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Apply separable convolution
        # First apply horizontal filter
        padding_h = kernel_size // 2
        filtered_h = F.conv2d(input_4d, kernel_h.unsqueeze(2), padding=(0, padding_h))
        
        # Then apply vertical filter
        padding_v = kernel_size // 2
        filtered_hv = F.conv2d(filtered_h, kernel_v, padding=(padding_v, 0))
        
        # Remove batch and channel dimensions
        return filtered_hv.squeeze(0).squeeze(0)

def create_atom_electron(x_tensor, y_tensor, z_tensor, cx, cy, cz, quantum_numbers, atomic_number=1, alpha=None, scale=SCALE):
    """Create 3D atomic orbital wavefunctions using pure PyTorch for GPU acceleration.
    
    This is a fully GPU-accelerated 3D atomic orbital generator.
    
    Args:
        x, y, z: coordinate grids (3D torch tensors)
        cx, cy, cz: center position (nucleus location)
        quantum_numbers: (n, l, m) quantum numbers
        atomic_number: nuclear charge Z (default: 1 for hydrogen)
        alpha: screening parameter for Z_eff (default: auto-calculated)
    
    Returns:
        Complex wavefunction representing the 3D atomic orbital (torch tensor)
    """
    n, l, m = quantum_numbers
    device = x_tensor.device
    
    # Calculate screening parameter (alpha) using a simple universal formula
    if alpha is None:
        if atomic_number == 1:
            alpha = 1.0
        else:
            # Simple universal screening for multi-electron atoms
            inner_electrons = max(0, atomic_number - 2)
            if l == 0:
                screening_per_electron = 0.25
            elif l == 1:
                screening_per_electron = 0.35  
            elif l == 2:
                screening_per_electron = 0.45
            else:
                screening_per_electron = 0.50
            
            n_factor = 1.0 + 0.1 * (n - 1)
            total_screening = inner_electrons * screening_per_electron * n_factor
            z_eff = atomic_number - total_screening
            alpha = max(z_eff / atomic_number, 0.1)
    
    # Calculate effective nuclear charge
    z_eff = atomic_number * alpha
    
    # Calculate physics-based Bohr radius using torch
    bohr_radius = scale / torch.sqrt(torch.tensor(POTENTIAL_STRENGTH * z_eff, device=device))
    orbital_radius = bohr_radius * n**2
    
    # Create coordinates relative to center (pure torch)
    dx = x_tensor - cx
    dy = y_tensor - cy
    dz = z_tensor - cz
    r = torch.sqrt(dx**2 + dy**2 + dz**2)
    
    # Spherical coordinates
    theta = torch.acos(torch.clamp(dz / (r + 1e-10), -1.0, 1.0))  # Polar angle (0 to π)
    phi = torch.atan2(dy, dx)  # Azimuthal angle (0 to 2π)
    
    # Create atomic radial wavefunction using effective nuclear charge
    rho = 2 * z_eff * r / (n * orbital_radius)
    
    # Simplified radial function for GPU efficiency (avoids scipy)
    # For s orbitals (l=0): R ~ exp(-rho/2)
    # For p orbitals (l=1): R ~ rho * exp(-rho/2)  
    # For d orbitals (l=2): R ~ rho^2 * exp(-rho/2)
    radial = (rho**l) * torch.exp(-rho/2)
    
    # Add oscillations for higher n quantum numbers
    if n > 1:
        oscillations = 1 + 0.8 * torch.cos(torch.pi * rho * (n-l-1) / n)
        radial = radial * oscillations
    
    # Create angular part using simplified real spherical harmonics
    if l == 0:  # s orbitals - spherically symmetric
        angular = torch.ones_like(theta)
    elif l == 1:  # p orbitals
        if m == -1:
            angular = torch.sin(theta) * torch.sin(phi)  # p_y
        elif m == 0:
            angular = torch.cos(theta)                    # p_z  
        elif m == 1:
            angular = torch.sin(theta) * torch.cos(phi)   # p_x
    elif l == 2:  # d orbitals
        if m == -2:
            angular = torch.sin(theta)**2 * torch.sin(2*phi)  # d_xy
        elif m == -1:
            angular = torch.sin(theta) * torch.cos(theta) * torch.sin(phi)  # d_yz
        elif m == 0:
            angular = 3*torch.cos(theta)**2 - 1  # d_z²
        elif m == 1:
            angular = torch.sin(theta) * torch.cos(theta) * torch.cos(phi)  # d_xz
        elif m == 2:
            angular = torch.sin(theta)**2 * torch.cos(2*phi)  # d_x²-y²
    else:  # Higher l orbitals - simplified patterns
        angular = torch.cos(l * phi) * (torch.sin(theta)**l)
    
    # Combine radial and angular parts
    psi = radial * angular
    
    # Convert to complex tensor
    psi = psi.to(torch.complex64)
    
    # Normalization (GPU accelerated)
    norm_factor = torch.sqrt(torch.sum(torch.abs(psi)**2))
    if norm_factor > 0:
        psi = psi / norm_factor
    
    return psi


class MolecularElectron:
    def __init__(self, wavefunction: torch.Tensor, atom_id: int, electron_name: str = "electron"):
        self.wavefunction = wavefunction.detach().clone().to(torch.complex64)
        self.atom_id = atom_id  # Which atom this electron belongs to
        self.name = electron_name
        self.normalize()
        
    def get_density(self) -> torch.Tensor:
        return torch.abs(self.wavefunction)**2
    
    def normalize(self):
        norm = torch.sqrt(torch.sum(torch.abs(self.wavefunction)**2))
        if norm > 0:
            self.wavefunction = self.wavefunction / norm


class MolecularNucleus:
    """Enhanced nucleus class with molecular properties - supports 3D."""
    
    def __init__(self, x: float, y: float, z: float = None, atomic_number: int = 1, atom_id: int = 0):
        if z is None:  # Backward compatibility for 2D
            self.position = torch.tensor([x, y], dtype=torch.float32, requires_grad=False)
            self.velocity = torch.zeros(2, dtype=torch.float32, requires_grad=False)
        else:  # 3D mode
            self.position = torch.tensor([x, y, z], dtype=torch.float32, requires_grad=False)
            self.velocity = torch.zeros(3, dtype=torch.float32, requires_grad=False)
        self.atomic_number = atomic_number
        self.atom_id = atom_id
        self.mass_ratio = 1836.0  # Proton to electron mass ratio
        
    @property
    def x(self):
        return self.position[0].item()
    
    @property 
    def y(self):
        return self.position[1].item()
    
    @property
    def z(self):
        return self.position[2].item() if len(self.position) > 2 else 0.0


class HybridMolecularSimulation:
    """
    Hybrid approach: Individual electrons + unified field forces
    
    This simulation maintains individual electron wavefunctions while using
    unified field concepts for computing inter-electron interactions.
    """
    
    def __init__(self, nuclei: List[MolecularNucleus], electrons: List[MolecularElectron],
                 electron_repulsion_strength: float = 0.15,
                 nuclear_motion_enabled: bool = True,
                 damping_factor: float = 0.98):
        self.nuclei = nuclei
        self.electrons = electrons
        self.electron_repulsion_strength = electron_repulsion_strength
        self.nuclear_motion_enabled = nuclear_motion_enabled
        self.damping_factor = damping_factor
        
        # For dual-range electron repulsion
        self.short_range_sigma = 2.0 * SCALE / 100.0
        self.long_range_sigma = 8.0 * SCALE / 100.0
        
        print(f"Initialized molecular simulation:")
        print(f"  - {len(nuclei)} nuclei")
        print(f"  - {len(electrons)} electrons")
        print(f"  - Electron repulsion strength: {electron_repulsion_strength}")
        print(f"  - Nuclear motion: {nuclear_motion_enabled}")
    
    def compute_electron_potential(self, target_electron_idx: int) -> torch.Tensor:
        """
        Compute the total potential experienced by one electron.
        Includes nuclear attraction + repulsion from other electrons.
        Supports both 2D and 3D.
        """
        if len(self.electrons[0].wavefunction.shape) == 3:  # 3D case
            potential = torch.zeros_like(X, dtype=torch.float32)
        else:  # 2D case
            potential = torch.zeros_like(X, dtype=torch.float32)
        
        # Nuclear attraction potentials
        for nucleus in self.nuclei:
            if len(nucleus.position) == 3:  # 3D nucleus
                V_nuclear = self.create_nucleus_potential_3d(
                    X, Y, Z, nucleus.position[0], nucleus.position[1], nucleus.position[2], nucleus.atomic_number
                )
            else:  # 2D nucleus
                V_nuclear = self.create_nucleus_potential_2d(
                    X, Y, nucleus.position[0], nucleus.position[1], nucleus.atomic_number
                )
            potential += V_nuclear
        
        # Electron-electron repulsion (from all OTHER electrons)
        for i, other_electron in enumerate(self.electrons):
            if i != target_electron_idx:
                other_density = other_electron.get_density()
                
                # Dual-range repulsion for more realistic interactions
                short_repulsion = gaussian_filter_torch(other_density, sigma=self.short_range_sigma) * 2.0
                long_repulsion = gaussian_filter_torch(other_density, sigma=self.long_range_sigma) * 0.5
                
                potential += self.electron_repulsion_strength * (short_repulsion + long_repulsion)
        
        return potential
    
    def create_nucleus_potential_2d(self, x, y, nucleus_x, nucleus_y, charge=1):
        """Create a 2D nucleus potential with short-range repulsion."""
        r = torch.sqrt((x - nucleus_x)**2 + (y - nucleus_y)**2)
        r = torch.maximum(r, torch.tensor(0.1, dtype=torch.float32))
        
        # Long-range Coulomb attraction: V = -k*Z/r (attractive for electrons)
        coulomb_attraction = -POTENTIAL_STRENGTH * charge / r
        
        # Short-range repulsion to prevent collapse into nucleus (models quantum effects)
        nuclear_repulsion = NUCLEAR_REPULSION_STRENGTH * torch.exp(-r / NUCLEAR_CORE_RADIUS) / (r + 0.1)
        
        return coulomb_attraction + nuclear_repulsion
    
    def create_nucleus_potential_3d(self, x, y, z, nucleus_x, nucleus_y, nucleus_z, charge=1):
        """Create a 3D nucleus potential with short-range repulsion."""
        r = torch.sqrt((x - nucleus_x)**2 + (y - nucleus_y)**2 + (z - nucleus_z)**2)
        r = torch.maximum(r, torch.tensor(0.1, dtype=torch.float32))
        
        # Long-range Coulomb attraction: V = -k*Z/r (attractive for electrons)
        coulomb_attraction = -POTENTIAL_STRENGTH * charge / r
        
        # Short-range repulsion to prevent collapse into nucleus (models quantum effects)
        nuclear_repulsion = NUCLEAR_REPULSION_STRENGTH * torch.exp(-r / NUCLEAR_CORE_RADIUS) / (r + 0.1)
        
        return coulomb_attraction + nuclear_repulsion

    # Backward compatibility
    def create_nucleus_potential(self, x, y, nucleus_x, nucleus_y, charge=1):
        """Backward compatibility wrapper."""
        return self.create_nucleus_potential_2d(x, y, nucleus_x, nucleus_y, charge)

    def compute_all_electron_potentials(self) -> List[torch.Tensor]:
        """Compute potentials for all electrons efficiently."""
        potentials = []
        
        # Pre-compute nuclear potentials (same for all electrons)
        nuclear_potential = torch.zeros_like(X, dtype=torch.float32)
        for nucleus in self.nuclei:
            V_nuclear = self.create_nucleus_potential(
                X, Y, nucleus.position[0], nucleus.position[1], nucleus.atomic_number
            )
            nuclear_potential += V_nuclear
        
        # Pre-compute all electron densities
        electron_densities = [e.get_density() for e in self.electrons]
        
        # Pre-filter all densities for efficiency
        short_filtered = [gaussian_filter_torch(d, sigma=self.short_range_sigma) for d in electron_densities]
        long_filtered = [gaussian_filter_torch(d, sigma=self.long_range_sigma) for d in electron_densities]
        
        # Compute potential for each electron
        for i in range(len(self.electrons)):
            potential = nuclear_potential.clone()
            
            # Add repulsion from all OTHER electrons
            for j in range(len(self.electrons)):
                if i != j:
                    repulsion = (short_filtered[j] * 2.0 + long_filtered[j] * 0.5)
                    potential += self.electron_repulsion_strength * repulsion
            
            potentials.append(potential)
        
        return potentials
    

    def compute_force_from_density(self, charge_density, nucleus_pos):
        """Compute forces on nucleus from electron density - supports torch tensors."""
        dx = X - nucleus_pos[0]
        dy = Y - nucleus_pos[1]
        r = torch.sqrt(dx**2 + dy**2)
        r = torch.maximum(r, torch.tensor(1.0))
        force_x = torch.sum((dx / r**3) * charge_density)
        force_y = torch.sum((dy / r**3) * charge_density)
        return torch.tensor([force_x, force_y], dtype=torch.float32)

    def compute_nuclear_forces(self) -> List[torch.Tensor]:
        """Compute forces on nuclei from electron densities and other nuclei."""
        forces = [torch.zeros(2) for _ in self.nuclei]
        
        # Forces from electrons
        for i, nucleus in enumerate(self.nuclei):
            for electron in self.electrons:
                density = electron.get_density()
                force = self.compute_force_from_density(density, nucleus.position)
                
                # All electrons exert forces on all nuclei
                # Electrons from same atom: attractive
                # Electrons from other atoms: can create bonding or anti-bonding forces
                if electron.atom_id == nucleus.atom_id:
                    forces[i] += force * 10.0  # Much stronger attraction from own electrons
                else:
                    # Inter-atomic electron forces create molecular bonding
                    forces[i] += force * 4.0  # Stronger bonding force
        
        # Nuclear-nuclear repulsion
        for i, nucleus1 in enumerate(self.nuclei):
            for j, nucleus2 in enumerate(self.nuclei):
                if i != j:
                    dx = nucleus2.position[0] - nucleus1.position[0]
                    dy = nucleus2.position[1] - nucleus1.position[1]
                    r = torch.sqrt(dx**2 + dy**2)
                    r = torch.maximum(r, torch.tensor(2.0))  # Prevent singularity
                    
                    # Coulomb repulsion F = k*q1*q2/r^2
                    force_magnitude = 2.0 * nucleus1.atomic_number * nucleus2.atomic_number / (r**2)  # Much weaker repulsion
                    
                    # Direction away from other nucleus
                    force_x = -force_magnitude * dx / r
                    force_y = -force_magnitude * dy / r
                    
                    forces[i][0] += force_x
                    forces[i][1] += force_y
        
        return forces
    
    def apply_absorbing_boundaries(self, wavefunction: torch.Tensor) -> torch.Tensor:
        # Create circular absorbing mask - exponential decay near circular boundary
        # Use global X, Y, Z tensors for grid coordinates
        
        # Calculate distance from center of simulation domain
        if len(wavefunction.shape) == 3:  # 3D case
            center_x_pos = SIZE_X / 2.0
            center_y_pos = SIZE_Y / 2.0
            center_z_pos = SIZE_Z / 2.0
            r = torch.sqrt((X - center_x_pos)**2 + (Y - center_y_pos)**2 + (Z - center_z_pos)**2)
            max_radius = min(SIZE_X, SIZE_Y, SIZE_Z) / 2.0 * 0.9  # Spherical boundary
        else:  # 2D case
            center_x_pos = SIZE_X / 2.0
            center_y_pos = SIZE_Y / 2.0
            r = torch.sqrt((X - center_x_pos)**2 + (Y - center_y_pos)**2)
            max_radius = min(SIZE_X, SIZE_Y) / 2.0 * 0.9  # Circular boundary
        
        # Define boundary parameters
        absorption_width = 20  # Width of absorption region
        absorption_start = max_radius - absorption_width
        
        # Create absorbing mask
        absorption_strength = 1.2  # Stronger absorption for boundaries
        
        # Only apply absorption outside the absorption start radius
        absorption_region = torch.maximum(
            torch.tensor(0.0, device=wavefunction.device), 
            r - absorption_start
        )
        
        # Use cubic falloff for smooth absorption with circular geometry
        # Normalize by absorption width to get 0-1 range
        normalized_depth = torch.clamp(absorption_region / absorption_width, 0, 1)
        mask = torch.exp(-absorption_strength * normalized_depth ** 3)
        
        # Add quantum noise near boundaries for realism
        # Noise strength increases near boundaries (where quantum uncertainty is highest)
        noise_strength = 0.0002 * normalized_depth  # Stronger noise near boundary
        
        # Generate complex quantum noise (both amplitude and phase fluctuations)
        noise_real = torch.randn_like(wavefunction.real) * noise_strength
        noise_imag = torch.randn_like(wavefunction.imag) * noise_strength
        quantum_noise = torch.complex(noise_real, noise_imag)
        
        # Apply absorption with noise
        absorbed_wf = wavefunction * mask
        
        # Add noise only in boundary regions
        noise_mask = (normalized_depth > 0.1).float()  # Only add noise where there's some absorption
        noisy_wf = absorbed_wf + quantum_noise * noise_mask
        
        return noisy_wf

    def evolve_step(self, step: int):
        """Evolve all electrons and nuclei by one time step."""
        
        # Compute potentials for all electrons
        potentials = self.compute_all_electron_potentials()
        
        # Prepare batch data for electron evolution
        psi_list = [e.wavefunction for e in self.electrons]
        
        # Convert torch tensors to numpy for the physics engine
        
        # Use batched wave propagation for efficiency
        evolved_psi_list = propagate_wave_batch_with_potentials(psi_list, potentials, propagation_method="fft_medium_damping")
        
        # Update electron wavefunctions
        for i, electron in enumerate(self.electrons):
            electron.wavefunction = evolved_psi_list[i].detach().clone().to(torch.complex64)
            wf_numpy = electron.wavefunction
            wf_limited = self.apply_absorbing_boundaries(wf_numpy)
            electron.wavefunction = wf_limited.detach().clone().to(torch.complex64)
            electron.normalize()
        
        # Update nuclear positions if enabled
        if self.nuclear_motion_enabled and len(self.nuclei) > 1:
            forces = self.compute_nuclear_forces()
            dt = 3.0  # Time step for nuclear motion
            
            for i, nucleus in enumerate(self.nuclei):
                # Update velocity and position
                nucleus.velocity += forces[i] * dt / nucleus.mass_ratio
                nucleus.position += nucleus.velocity * dt
                
                # Apply damping to prevent runaway oscillations
                nucleus.velocity *= self.damping_factor
                
                # Keep nuclei within simulation bounds
                if len(nucleus.position) == 3:  # 3D case
                    nucleus.position[0] = torch.clip(nucleus.position[0], SIZE_X//8, 7*SIZE_X//8)
                    nucleus.position[1] = torch.clip(nucleus.position[1], SIZE_Y//8, 7*SIZE_Y//8)
                    nucleus.position[2] = torch.clip(nucleus.position[2], SIZE_Z//8, 7*SIZE_Z//8)
                else:  # 2D case
                    nucleus.position[0] = torch.clip(nucleus.position[0], SIZE_X//8, 7*SIZE_X//8)
                    nucleus.position[1] = torch.clip(nucleus.position[1], SIZE_Y//8, 7*SIZE_Y//8)
    
    def get_combined_density(self) -> torch.Tensor:
        """Get combined electron density for visualization."""
        total_density = torch.zeros_like(X, dtype=torch.float32)
        
        for electron in self.electrons:
            density = electron.get_density()
            total_density += density
        
        return total_density
    
    def get_combined_wavefunction(self) -> torch.Tensor:
        """Get combined wavefunction for visualization."""
        if not self.electrons:
            return torch.zeros_like(X, dtype=torch.complex64)
        
        combined = torch.zeros_like(X, dtype=torch.complex64)
        for electron in self.electrons:
            # Keep everything as torch tensors, no need for numpy conversion
            wf = electron.wavefunction
            combined += wf
        
        # Normalize
        norm = torch.sqrt(torch.sum(torch.abs(combined)**2))
        if norm > 0:
            combined = combined / norm
        
        return combined


def create_hydrogen_molecule_simulation() -> HybridMolecularSimulation:
    """Create a realistic H2 molecule simulation."""
    # Molecular parameters
    bond_length = 1.2 * SCALE  # Increased H-H bond length for better visualization
    nucleus1_x = center_x - bond_length/2
    nucleus2_x = center_x + bond_length/2
    nucleus_y = center_y
    
    # Create nuclei
    nuclei = [
        MolecularNucleus(nucleus1_x, nucleus_y, atomic_number=1, atom_id=0),
        MolecularNucleus(nucleus2_x, nucleus_y, atomic_number=1, atom_id=1)
    ]
    
    # Create electrons with realistic hydrogen 1s orbitals (3x bigger for better visualization)
    print("Creating electron 1 (atom 0)...")
    psi1 = create_atom_electron(X, Y, nucleus1_x, nucleus_y, (1, 0, 0), 
                               atomic_number=1, scale=SCALE /10)
    
    print("Creating electron 2 (atom 1)...")
    psi2 = create_atom_electron(X, Y, nucleus2_x, nucleus_y, (1, 0, 0), 
                               atomic_number=1, scale=SCALE /10)
    
    electrons = [
        MolecularElectron(psi1, atom_id=0, electron_name="electron_1"),
        MolecularElectron(psi2, atom_id=1, electron_name="electron_2")
    ]
    
    # Create simulation
    simulation = HybridMolecularSimulation(
        nuclei=nuclei,
        electrons=electrons,
        electron_repulsion_strength=0.12,  # Moderate repulsion for realistic behavior
        nuclear_motion_enabled=True,       # Allow nuclei to move
        damping_factor=0.999               # Slight damping to prevent oscillations
    )
    
    return simulation


def run_molecular_simulation(simulation: HybridMolecularSimulation, 
                           video_filename: str = "hybrid_h2_molecule.avi",
                           fps: int = 24) -> None:
    """Run the molecular simulation and save video."""
    
    print(f"Starting hybrid molecular simulation...")
    print(f"Output video: {video_filename}")
    
    # Set up video writer
    video_writer = StreamingVideoWriter(
        output_file=video_filename,
        fps=fps,
        sample_frames=50,
        keep_first_batch=True,
        first_batch_size=100
    )
    
    # Run simulation
    for step in range(TIME_STEPS):
        if step % 100 == 0:
            print(f"Step {step}/{TIME_STEPS}")
            
            # Report nuclear positions for H2 molecules
            if len(simulation.nuclei) == 2:
                dx = simulation.nuclei[1].position[0] - simulation.nuclei[0].position[0]
                dy = simulation.nuclei[1].position[1] - simulation.nuclei[0].position[1]
                bond_length = torch.sqrt(dx**2 + dy**2)
                print(f"  Bond length: {bond_length:.2f} pixels")
        
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
    
    # Finalize video
    video_writer.finalize()
    print(f"Simulation complete! Video saved as {video_filename}")
    
    # Open video for viewing
    try:
        open_video(video_filename)
    except Exception as e:
        print(f"Could not open video automatically: {e}")


if __name__ == "__main__":
    print("=== Hybrid Molecular Simulation ===")
    print("Individual electrons + unified field forces")
    print()
    
    # Test 1: Stable H2 molecule
    print("1. Creating stable H2 molecule simulation...")
    sim1 = create_hydrogen_molecule_simulation()
    run_molecular_simulation(sim1, "hybrid_stable_h2.avi")
    print("Simulation complete!")
