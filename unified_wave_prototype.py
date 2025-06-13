"""
Unified Wave Simulation Prototype

This prototype explores representing all particles (electrons and nuclei) as waves
in a single unified quantum system, rather than separate interacting particles.

Key concepts:
- Single system wavefunction describes the entire quantum state
- No separate lists of electrons and nuclei
- All forces emerge from the unified field evolution
- More physically accurate quantum mechanical treatment
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any
import torch.nn.functional as F

from config import SIZE, TIME_DELTA, X, Y, center_x, center_y, TIME_STEPS
from torch_physics import get_device, convert_to_device
from video_utils import StreamingVideoWriter, open_video


class UnifiedQuantumField:
    """
    Represents the entire quantum system as a single unified field.
    No separate electrons/nuclei - everything is waves.
    """
    
    def __init__(self, shape: Tuple[int, int], device=None):
        self.shape = shape
        self.device = device or get_device()
        
        # Single unified wavefunction for the entire system
        self.system_wavefunction = None
        
        # Field components (for interpretation and visualization)
        self.electron_field_component = None
        self.nuclear_field_component = None
        
        # System properties
        self.total_charge = 0
        self.system_energy = 0.0
        
        # Pre-computed operators for efficiency
        self._setup_quantum_operators()
        
    def _setup_quantum_operators(self):
        """Pre-compute quantum mechanical operators"""
        # Kinetic energy operator in momentum space
        kx = torch.fft.fftfreq(self.shape[1], d=1.0, device=self.device) * (2 * torch.pi)
        ky = torch.fft.fftfreq(self.shape[0], d=1.0, device=self.device) * (2 * torch.pi)
        KY, KX = torch.meshgrid(ky, kx, indexing='ij')
        
        # Kinetic operator: T = p²/(2m) = -ℏ²∇²/(2m)
        self.kinetic_operator = -(KX**2 + KY**2) * 0.5  # ℏ²/(2m) = 0.5 in our units
        
        # Laplacian operator for real-space calculations
        laplacian_kernel = torch.tensor([
            [0.0,  1.0, 0.0],
            [1.0, -4.0, 1.0], 
            [0.0,  1.0, 0.0]
        ], dtype=torch.complex64, device=self.device).unsqueeze(0).unsqueeze(0)
        self.laplacian_kernel = laplacian_kernel
        
    def create_hydrogen_atom_field(self, nucleus_x: float, nucleus_y: float):
        """
        Create a hydrogen atom as a unified quantum field.
        
        Instead of separate electron + nucleus, we create a single field
        that embodies the bound state solution.
        """
        # Create spatial grids
        x_grid = torch.arange(self.shape[1], dtype=torch.float32, device=self.device)
        y_grid = torch.arange(self.shape[0], dtype=torch.float32, device=self.device)
        Y_grid, X_grid = torch.meshgrid(y_grid, x_grid, indexing='ij')
        
        # Distance from nucleus
        r = torch.sqrt((X_grid - nucleus_x)**2 + (Y_grid - nucleus_y)**2)
        r = torch.clamp(r, min=0.1)  # Avoid singularity
        
        # Hydrogen 1s wavefunction: ψ = exp(-r/a₀) where a₀ is Bohr radius
        bohr_radius = 5.0  # Scaled for our simulation to keep atoms compact
        electron_amplitude = torch.exp(-r / bohr_radius)
        
        # Add nuclear component as highly localized wave
        nuclear_width = 1.0
        nuclear_amplitude = 0.1 * torch.exp(-((X_grid - nucleus_x)**2 + (Y_grid - nucleus_y)**2) / (2 * nuclear_width**2))
        
        # Combine into unified field with phase relationships
        # The phases encode the quantum mechanical correlations
        electron_phase = torch.zeros_like(r)  # 1s is real
        nuclear_phase = torch.zeros_like(r)   # Nuclear field in phase
        
        # Unified wavefunction: superposition of electron and nuclear components
        electron_part = electron_amplitude * torch.exp(1j * electron_phase)
        nuclear_part = nuclear_amplitude * torch.exp(1j * nuclear_phase)
        
        # The unified field is the quantum superposition
        self.system_wavefunction = electron_part + 0.1j * nuclear_part  # Small nuclear contribution
        
        # Store components for analysis
        self.electron_field_component = electron_part
        self.nuclear_field_component = nuclear_part
        
        # Normalize
        self._normalize_wavefunction()
        
        print(f"Created hydrogen atom unified field at ({nucleus_x}, {nucleus_y})")
        
    def create_hydrogen_molecule_field(self, nucleus1_pos: Tuple[float, float], 
                                     nucleus2_pos: Tuple[float, float]):
        """
        Create H₂ molecule as a unified quantum field.
        
        This represents the true molecular wavefunction as a single entity,
        not separate atoms that interact.
        """
        x_grid = torch.arange(self.shape[1], dtype=torch.float32, device=self.device)
        y_grid = torch.arange(self.shape[0], dtype=torch.float32, device=self.device)
        Y_grid, X_grid = torch.meshgrid(y_grid, x_grid, indexing='ij')
        
        # Distances from both nuclei
        r1 = torch.sqrt((X_grid - nucleus1_pos[0])**2 + (Y_grid - nucleus1_pos[1])**2)
        r2 = torch.sqrt((X_grid - nucleus2_pos[0])**2 + (Y_grid - nucleus2_pos[1])**2)
        r1 = torch.clamp(r1, min=0.1)
        r2 = torch.clamp(r2, min=0.1)
        
        # Molecular orbital: bonding combination of atomic orbitals
        bohr_radius = 6.0  # Appropriate size for molecular orbitals
        psi1 = torch.exp(-r1 / bohr_radius)  # Atomic orbital on nucleus 1
        psi2 = torch.exp(-r2 / bohr_radius)  # Atomic orbital on nucleus 2
        
        # Create bonding molecular orbital (σ_g) - electrons shared between nuclei
        # This creates the characteristic "dumbbell" shape with charge concentration between nuclei
        bonding_orbital = (psi1 + psi2) / torch.sqrt(torch.tensor(2.0))
        
        # Add overlap enhancement in the bonding region (between nuclei)
        bond_center_x = (nucleus1_pos[0] + nucleus2_pos[0]) / 2
        bond_center_y = (nucleus1_pos[1] + nucleus2_pos[1]) / 2
        r_bond = torch.sqrt((X_grid - bond_center_x)**2 + (Y_grid - bond_center_y)**2)
        
        # Enhance electron density in bonding region
        bond_enhancement = 1.0 + 0.3 * torch.exp(-r_bond / (bohr_radius * 0.8))
        bonding_orbital = bonding_orbital * bond_enhancement
        
        # Nuclear field components
        nuclear_width = 1.0
        nuclear1 = 0.05 * torch.exp(-((X_grid - nucleus1_pos[0])**2 + (Y_grid - nucleus1_pos[1])**2) / (2 * nuclear_width**2))
        nuclear2 = 0.05 * torch.exp(-((X_grid - nucleus2_pos[0])**2 + (Y_grid - nucleus2_pos[1])**2) / (2 * nuclear_width**2))
        
        # Unified molecular field
        electron_part = bonding_orbital
        nuclear_part = nuclear1 + nuclear2
        
        # Create quantum superposition with proper phases
        self.system_wavefunction = electron_part + 0.1j * nuclear_part
        
        # Store components
        self.electron_field_component = electron_part.to(torch.complex64)
        self.nuclear_field_component = nuclear_part.to(torch.complex64)
        
        self._normalize_wavefunction()
        
        bond_length = np.sqrt((nucleus1_pos[0] - nucleus2_pos[0])**2 + (nucleus1_pos[1] - nucleus2_pos[1])**2)
        print(f"Created H₂ molecule unified field with bond length {bond_length:.1f}")
        
    def _normalize_wavefunction(self):
        """Normalize the system wavefunction"""
        if self.system_wavefunction is not None:
            norm = torch.sqrt(torch.sum(torch.abs(self.system_wavefunction)**2))
            if norm > 0:
                self.system_wavefunction = self.system_wavefunction / norm
                
    def evolve_step(self, dt: float = TIME_DELTA):
        """
        Evolve the entire unified field using quantum dynamics.
        
        This is the heart of the unified approach - single evolution
        for the entire system, not separate particle propagations.
        """
        if self.system_wavefunction is None:
            return
            
        # Compute self-consistent potential from the field itself
        potential = self._compute_self_consistent_potential()
        
        # Split-step evolution: exp(-iHt) ≈ exp(-iVt/2) exp(-iTt) exp(-iVt/2)
        self._apply_potential_evolution(potential, dt/2)
        self._apply_kinetic_evolution(dt)
        self._apply_potential_evolution(potential, dt/2)
        
        # Apply absorbing boundary conditions
        self._apply_absorbing_boundaries()
        
        # Update system properties
        self._update_system_properties()
        
    def _compute_self_consistent_potential(self) -> torch.Tensor:
        """
        Compute the potential that the field creates for itself.
        
        This is where the magic happens - the field generates its own
        dynamics through self-interaction.
        """
        if self.system_wavefunction is None:
            return torch.zeros(self.shape, device=self.device)
            
        # Extract charge density from the wavefunction
        charge_density = torch.abs(self.system_wavefunction)**2
        
        # Electron-electron repulsion (Hartree potential)
        # ∇²φ = -4πρ, solved approximately with Gaussian convolution
        electron_repulsion = self._compute_coulomb_repulsion(charge_density)
        
        # Nuclear attraction potential
        nuclear_attraction = self._compute_nuclear_attraction()
        
        # Self-interaction potential (prevents field collapse)
        self_interaction = self._compute_self_interaction(charge_density)
        
        return nuclear_attraction + electron_repulsion + self_interaction
        
    def _compute_coulomb_repulsion(self, charge_density: torch.Tensor) -> torch.Tensor:
        """Compute electron-electron Coulomb repulsion"""
        # Use Gaussian convolution to approximate Coulomb interaction
        sigma = 3.0
        kernel_size = int(6 * sigma) + 1
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1  # Ensure odd
        
        # Create Gaussian kernel
        x = torch.arange(kernel_size, dtype=torch.float32, device=self.device) - kernel_size // 2
        y = torch.arange(kernel_size, dtype=torch.float32, device=self.device) - kernel_size // 2
        Y_k, X_k = torch.meshgrid(y, x, indexing='ij')
        
        gaussian_kernel = torch.exp(-(X_k**2 + Y_k**2) / (2 * sigma**2))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)
        
        # Apply convolution
        density_4d = charge_density.real.unsqueeze(0).unsqueeze(0)
        repulsion = F.conv2d(density_4d, gaussian_kernel, padding=kernel_size//2)
        
        return 0.15 * repulsion.squeeze(0).squeeze(0)  # Stronger repulsion to see individual electron dynamics
        
    def _compute_nuclear_attraction(self) -> torch.Tensor:
        """Compute nuclear attraction potential"""
        # Extract nuclear positions from the nuclear field component
        if self.nuclear_field_component is None:
            return torch.zeros(self.shape, device=self.device)
            
        x_grid = torch.arange(self.shape[1], dtype=torch.float32, device=self.device)
        y_grid = torch.arange(self.shape[0], dtype=torch.float32, device=self.device)
        Y_grid, X_grid = torch.meshgrid(y_grid, x_grid, indexing='ij')
        
        # Use nuclear field density to create attraction potential
        nuclear_density = torch.abs(self.nuclear_field_component)**2
        
        # Create Coulomb potential from nuclear density
        # Simple model: V(r) = -nuclear_density * strength
        attraction_strength = 2.0
        nuclear_attraction = -attraction_strength * nuclear_density.real
        
        return nuclear_attraction
        
    def _compute_self_interaction(self, charge_density: torch.Tensor) -> torch.Tensor:
        """Compute self-interaction to prevent collapse"""
        # Pauli exclusion-like repulsion
        return 0.01 * charge_density.real
        
    def _apply_potential_evolution(self, potential: torch.Tensor, dt: float):
        """Apply potential evolution: ψ → exp(-iVt)ψ"""
        potential_phase = torch.exp(-1j * dt * potential)
        self.system_wavefunction = self.system_wavefunction * potential_phase
        
    def _apply_kinetic_evolution(self, dt: float):
        """Apply kinetic evolution: ψ → exp(-iTt)ψ"""
        # Transform to momentum space
        psi_k = torch.fft.fft2(self.system_wavefunction)
        
        # Apply kinetic operator
        kinetic_phase = torch.exp(-1j * dt * self.kinetic_operator)
        psi_k = psi_k * kinetic_phase
        
        # Transform back to position space
        self.system_wavefunction = torch.fft.ifft2(psi_k)
    
    def _apply_absorbing_boundaries(self):
        """Apply circular absorbing boundary conditions to prevent reflection"""
        if self.system_wavefunction is None:
            return
            
        # Create circular absorbing mask
        x_grid = torch.arange(self.shape[1], dtype=torch.float32, device=self.device)
        y_grid = torch.arange(self.shape[0], dtype=torch.float32, device=self.device)
        Y_grid, X_grid = torch.meshgrid(y_grid, x_grid, indexing='ij')
        
        # Calculate distance from center of simulation domain
        center_x_pos = self.shape[1] / 2.0
        center_y_pos = self.shape[0] / 2.0
        r = torch.sqrt((X_grid - center_x_pos)**2 + (Y_grid - center_y_pos)**2)
        
        # Define circular boundary parameters
        max_radius = min(center_x_pos, center_y_pos) * 0.85  # Conservative radius
        absorption_width = 50  # Width of absorption region
        absorption_start = max_radius - absorption_width
        
        # Create circular absorbing mask
        absorption_strength = 0.08  # Moderate absorption for unified wave
        
        # Only apply absorption outside the absorption start radius
        absorption_region = torch.maximum(
            torch.tensor(0.0, device=self.device), 
            r - absorption_start
        )
        
        # Use cubic falloff for smooth circular absorption
        normalized_depth = torch.clamp(absorption_region / absorption_width, 0, 1)
        mask = torch.exp(-absorption_strength * normalized_depth ** 3)
        
        # Apply absorption
        self.system_wavefunction = self.system_wavefunction * mask
        
    def _update_system_properties(self):
        """Update computed system properties"""
        if self.system_wavefunction is None:
            return
            
        # Total charge (should be conserved)
        self.total_charge = torch.sum(torch.abs(self.system_wavefunction)**2).item()
        
        # System energy (kinetic + potential)
        kinetic_energy = self._compute_kinetic_energy()
        potential_energy = self._compute_potential_energy()
        self.system_energy = kinetic_energy + potential_energy
        
    def _compute_kinetic_energy(self) -> float:
        """Compute kinetic energy of the system"""
        if self.system_wavefunction is None:
            return 0.0
            
        # <ψ|T|ψ> = <ψ|(-∇²/2)|ψ>
        psi_k = torch.fft.fft2(self.system_wavefunction)
        kinetic_psi_k = self.kinetic_operator * psi_k
        kinetic_psi = torch.fft.ifft2(kinetic_psi_k)
        
        kinetic_energy = torch.real(torch.sum(torch.conj(self.system_wavefunction) * kinetic_psi))
        return kinetic_energy.item()
        
    def _compute_potential_energy(self) -> float:
        """Compute potential energy of the system"""
        if self.system_wavefunction is None:
            return 0.0
            
        potential = self._compute_self_consistent_potential()
        density = torch.abs(self.system_wavefunction)**2
        potential_energy = torch.sum(potential * density)
        return potential_energy.item()
        
    def get_electron_density(self) -> np.ndarray:
        """Extract electron density for visualization"""
        if self.system_wavefunction is None:
            return np.zeros(self.shape)
            
        density = torch.abs(self.system_wavefunction)**2
        return density.cpu().numpy()
        
    def get_nuclear_density(self) -> np.ndarray:
        """Extract nuclear density for visualization"""
        if self.nuclear_field_component is None:
            return np.zeros(self.shape)
            
        density = torch.abs(self.nuclear_field_component)**2
        return density.cpu().numpy()
        
    def get_phase_map(self) -> np.ndarray:
        """Get the phase structure of the wavefunction"""
        if self.system_wavefunction is None:
            return np.zeros(self.shape)
            
        phase = torch.angle(self.system_wavefunction)
        return phase.cpu().numpy()
    
    def _create_video_frame(self) -> Dict[str, np.ndarray]:
        """Create video frame data compatible with StreamingVideoWriter"""
        if self.system_wavefunction is None:
            zeros = np.zeros(self.shape)
            return {'real': zeros, 'imag': zeros, 'phase': zeros, 'prob': zeros}
        
        wf = self.system_wavefunction.cpu().numpy()
        
        return {
            'real': np.real(wf),
            'imag': np.imag(wf), 
            'phase': np.angle(wf),
            'prob': np.abs(wf)**2
        }


def simulate_individual_hydrogen_atom(nucleus_pos: Tuple[float, float], 
                                    shape: Tuple[int, int], 
                                    evolution_steps: int = 300) -> torch.Tensor:
    """
    Simulate a single hydrogen atom in isolation to get its natural wavefunction.
    This gives us the proper individual electron dynamics before interaction.
    """
    print(f"Simulating individual hydrogen atom at {nucleus_pos} for {evolution_steps} steps...")
    
    # Create individual field for this atom
    atom_field = UnifiedQuantumField(shape)
    
    # Create spatial grids
    x_grid = torch.arange(shape[1], dtype=torch.float32, device=atom_field.device)
    y_grid = torch.arange(shape[0], dtype=torch.float32, device=atom_field.device)
    Y_grid, X_grid = torch.meshgrid(y_grid, x_grid, indexing='ij')
    
    # Initialize with hydrogen ground state
    r = torch.sqrt((X_grid - nucleus_pos[0])**2 + (Y_grid - nucleus_pos[1])**2)
    r = torch.clamp(r, min=0.1)
    
    bohr_radius = 8.0
    initial_wavefunction = torch.exp(-r / bohr_radius).to(torch.complex64)
    
    # Normalize
    norm = torch.sqrt(torch.sum(torch.abs(initial_wavefunction)**2))
    if norm > 0:
        initial_wavefunction = initial_wavefunction / norm
    
    atom_field.system_wavefunction = initial_wavefunction
    
    # Create nuclear field for this atom only
    nuclear_width = 1.0
    nuclear_field = 0.05 * torch.exp(-((X_grid - nucleus_pos[0])**2 + 
                                     (Y_grid - nucleus_pos[1])**2) / (2 * nuclear_width**2))
    atom_field.nuclear_field_component = nuclear_field.to(torch.complex64)
    
    # Let it evolve to reach natural bound state
    for step in range(evolution_steps):
        atom_field.evolve_step()
        
        if step % 100 == 0:
            print(f"  Step {step}: Energy = {atom_field.system_energy:.6f}")
    
    print(f"Individual atom simulation complete. Final energy: {atom_field.system_energy:.6f}")
    return atom_field.system_wavefunction.clone()


def run_unified_hydrogen_simulation(create_video: bool = True, video_file: str = "unified_hydrogen.avi"):
    """
    Run a hydrogen atom simulation using the unified field approach.
    NEW APPROACH: First simulate atoms individually, then combine their natural wavefunctions!
    """
    print("Starting Unified Wave Hydrogen Simulation with Individual Pre-Evolution...")
    
    # Define atom positions
    bond_length = 80
    nucleus1_x = center_x - bond_length // 2
    nucleus1_y = center_y
    nucleus2_x = center_x + bond_length // 2  
    nucleus2_y = center_y
    
    print(f"Two hydrogen atoms will be pre-evolved separately, then combined")
    print(f"Initial separation: {bond_length} pixels")
    print(f"Nucleus 1 at ({nucleus1_x}, {nucleus1_y}), Nucleus 2 at ({nucleus2_x}, {nucleus2_y})")
    
    # STAGE 1: Simulate each atom individually to get natural wavefunctions
    print("\n=== STAGE 1: Individual Atom Simulations ===")
    atom1_wavefunction = simulate_individual_hydrogen_atom(
        (nucleus1_x, nucleus1_y), (SIZE, SIZE), evolution_steps=300
    )
    
    atom2_wavefunction = simulate_individual_hydrogen_atom(
        (nucleus2_x, nucleus2_y), (SIZE, SIZE), evolution_steps=300
    )
    
    # STAGE 2: Combine the evolved atoms and let them interact
    print("\n=== STAGE 2: Combined System Evolution ===")
    field = UnifiedQuantumField((SIZE, SIZE))
    
    # Start with the naturally evolved individual wavefunctions
    # Add them with a small random phase to break perfect symmetry
    phase1 = torch.tensor(0.0, device=field.device)
    phase2 = torch.tensor(0.1, device=field.device)  # Small phase difference to break symmetry
    
    field.system_wavefunction = (atom1_wavefunction * torch.exp(1j * phase1) + 
                               atom2_wavefunction * torch.exp(1j * phase2))
    
    # Create combined nuclear field
    x_grid = torch.arange(field.shape[1], dtype=torch.float32, device=field.device)
    y_grid = torch.arange(field.shape[0], dtype=torch.float32, device=field.device)
    Y_grid, X_grid = torch.meshgrid(y_grid, x_grid, indexing='ij')
    
    nuclear_width = 1.0
    nuclear1 = 0.05 * torch.exp(-((X_grid - nucleus1_x)**2 + (Y_grid - nucleus1_y)**2) / (2 * nuclear_width**2))
    nuclear2 = 0.05 * torch.exp(-((X_grid - nucleus2_x)**2 + (Y_grid - nucleus2_y)**2) / (2 * nuclear_width**2))
    field.nuclear_field_component = (nuclear1 + nuclear2).to(torch.complex64)
    
    # Final normalization
    field._normalize_wavefunction()
    
    print("Combined system created from individually evolved atoms!")
    print("Now the natural electron dynamics should interact...")
    
    # Storage for analysis
    energies = []
    charges = []
    
    print(f"Initial combined system energy: {field.system_energy:.6f}")
    print(f"Initial total charge: {field.total_charge:.6f}")
    
    # Set up video writer if requested
    video_writer = None
    if create_video:
        video_writer = StreamingVideoWriter(
            output_file=video_file,
            fps=24,
            sample_frames=50,
            keep_first_batch=True,
            first_batch_size=100
        )
        print(f"Video will be saved as: {video_file}")
    
    # Main evolution loop - watch the individual dynamics interact
    simulation_steps = min(TIME_STEPS, 1500)
    for step in range(simulation_steps):
        # Evolve the combined field
        field.evolve_step()
        
        # Record system properties
        energies.append(field.system_energy)
        charges.append(field.total_charge)
        
        # Create video frame if requested
        if create_video and video_writer is not None:
            frame_data = field._create_video_frame()
            video_writer.add_frame(
                frame_data['real'], 
                frame_data['imag'], 
                frame_data['phase'], 
                frame_data['prob']
            )
        
        if step % 50 == 0:
            print(f"Step {step}: Energy = {field.system_energy:.6f}, Charge = {field.total_charge:.6f}")
    
    # Finalize video
    if create_video and video_writer is not None:
        video_writer.finalize()
        open_video(video_file)
        print(f"Video saved as {video_file}")
            
    # Analysis
    print("\n=== Simulation Complete ===")
    print(f"Final energy: {field.system_energy:.6f}")
    print(f"Final charge: {field.total_charge:.6f}")
    print(f"Energy conservation: {abs(energies[-1] - energies[0]):.6f}")
    print(f"Charge conservation: {abs(charges[-1] - charges[0]):.6f}")
    
    return field, energies, charges


def run_unified_h2_simulation(create_video: bool = True, video_file: str = "unified_h2_molecule.avi"):
    """
    Run a hydrogen molecule simulation using the unified field approach.
    """
    print("Starting Unified Wave H₂ Simulation...")
    
    # Create unified quantum field
    field = UnifiedQuantumField((SIZE, SIZE))
    
    # Create H₂ molecule as unified field - offset from center
    bond_length = 40
    offset_x, offset_y = -80, 0  # Offset from center to see wave propagation
    nucleus1_pos = (center_x + offset_x - bond_length//2, center_y + offset_y)
    nucleus2_pos = (center_x + offset_x + bond_length//2, center_y + offset_y)
    field.create_hydrogen_molecule_field(nucleus1_pos, nucleus2_pos)
    
    print(f"H₂ molecule positioned at offset ({offset_x}, {offset_y}) from center")
    
    # Storage for analysis
    energies = []
    charges = []
    
    print(f"Initial molecular energy: {field.system_energy:.6f}")
    print(f"Initial total charge: {field.total_charge:.6f}")
    
    # Set up video writer if requested
    video_writer = None
    if create_video:
        video_writer = StreamingVideoWriter(
            output_file=video_file,
            fps=24,
            sample_frames=50,
            keep_first_batch=True,
            first_batch_size=100
        )
        print(f"Video will be saved as: {video_file}")
    
    # Main evolution loop
    simulation_steps = min(TIME_STEPS // 5, 500)
    for step in range(simulation_steps):
        # Evolve the unified field
        field.evolve_step()
        
        # Record system properties
        energies.append(field.system_energy)
        charges.append(field.total_charge)
        
        # Create video frame if requested
        if create_video and video_writer is not None:
            frame_data = field._create_video_frame()
            video_writer.add_frame(
                frame_data['real'], 
                frame_data['imag'], 
                frame_data['phase'], 
                frame_data['prob']
            )
        
        if step % 50 == 0:
            print(f"Step {step}: Energy = {field.system_energy:.6f}, Charge = {field.total_charge:.6f}")
    
    # Finalize video
    if create_video and video_writer is not None:
        video_writer.finalize()
        open_video(video_file)
        print(f"Video saved as {video_file}")
            
    print("\n=== H₂ Simulation Complete ===")
    print(f"Final energy: {field.system_energy:.6f}")
    print(f"Final charge: {field.total_charge:.6f}")
    
    return field, energies, charges


def run_two_hydrogen_atoms_simulation(create_video: bool = True, video_file: str = "two_hydrogen_atoms.avi"):
    """
    Run a simulation of two separate hydrogen atoms to see interference patterns.
    This should show double-slit-like behavior, different from a true molecule.
    """
    print("Starting Two Independent Hydrogen Atoms Simulation...")
    
    # Create unified quantum field
    field = UnifiedQuantumField((SIZE, SIZE))
    
    # Create 2 hydrogen atoms at different positions to see wave interactions
    atom_positions = [
        (center_x - 30, center_y),   # Left atom
        (center_x + 30, center_y),   # Right atom
    ]
    
    print(f"Creating {len(atom_positions)} independent hydrogen atoms...")
    
    # Create all atom wavefunctions first, then superpose with equal weights
    atom_wavefunctions = []
    
    # Create spatial grids once
    x_grid = torch.arange(field.shape[1], dtype=torch.float32, device=field.device)
    y_grid = torch.arange(field.shape[0], dtype=torch.float32, device=field.device)
    Y_grid, X_grid = torch.meshgrid(y_grid, x_grid, indexing='ij')
    
    for i, (x, y) in enumerate(atom_positions):
        print(f"Creating hydrogen atom {i+1} at ({x}, {y})")
        
        # Create hydrogen atom wavefunction
        r = torch.sqrt((X_grid - x)**2 + (Y_grid - y)**2)
        r = torch.clamp(r, min=0.1)
        
        bohr_radius = 5.0  # Smaller atoms to keep probability within simulation bounds
        electron_amplitude = torch.exp(-r / bohr_radius)
        
        # Convert to complex and normalize individually
        atom_psi = electron_amplitude.to(torch.complex64)
        norm = torch.sqrt(torch.sum(torch.abs(atom_psi)**2))
        if norm > 0:
            atom_psi = atom_psi / norm
            
        atom_wavefunctions.append(atom_psi)
    
    # Create equal superposition of all atoms
    field.system_wavefunction = torch.zeros_like(atom_wavefunctions[0])
    for atom_psi in atom_wavefunctions:
        field.system_wavefunction = field.system_wavefunction + atom_psi
    
    # Final normalization of the combined system
    field._normalize_wavefunction()
    
    # Storage for analysis
    energies = []
    charges = []
    
    print(f"Initial system energy: {field.system_energy:.6f}")
    print(f"Initial total charge: {field.total_charge:.6f}")
    
    # Set up video writer if requested
    video_writer = None
    if create_video:
        video_writer = StreamingVideoWriter(
            output_file=video_file,
            fps=24,
            sample_frames=50,
            keep_first_batch=True,
            first_batch_size=100
        )
        print(f"Video will be saved as: {video_file}")
    
    # Main evolution loop
    simulation_steps = min(TIME_STEPS, 1000)
    for step in range(simulation_steps):
        # Evolve the unified field
        field.evolve_step()
        
        # Record system properties
        energies.append(field.system_energy)
        charges.append(field.total_charge)
        
        # Create video frame if requested
        if create_video and video_writer is not None:
            frame_data = field._create_video_frame()
            video_writer.add_frame(
                frame_data['real'], 
                frame_data['imag'], 
                frame_data['phase'], 
                frame_data['prob']
            )
        
        if step % 50 == 0:
            print(f"Step {step}: Energy = {field.system_energy:.6f}, Charge = {field.total_charge:.6f}")
    
    # Finalize video
    if create_video and video_writer is not None:
        video_writer.finalize()
        open_video(video_file)
        print(f"Video saved as {video_file}")
            
    print("\n=== Two Atoms Simulation Complete ===")
    print(f"Final energy: {field.system_energy:.6f}")
    print(f"Final charge: {field.total_charge:.6f}")
    
    return field, energies, charges


def visualize_unified_field(field: UnifiedQuantumField, title: str = "Unified Quantum Field"):
    """
    Visualize the unified quantum field.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title)
    
    # Electron density
    electron_density = field.get_electron_density()
    im1 = axes[0,0].imshow(electron_density, cmap='viridis', origin='lower')
    axes[0,0].set_title('Electron Density |ψ|²')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Nuclear density
    nuclear_density = field.get_nuclear_density()
    im2 = axes[0,1].imshow(nuclear_density, cmap='Reds', origin='lower')
    axes[0,1].set_title('Nuclear Field Component')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Phase map
    phase_map = field.get_phase_map()
    im3 = axes[1,0].imshow(phase_map, cmap='hsv', origin='lower', vmin=-np.pi, vmax=np.pi)
    axes[1,0].set_title('Phase Structure')
    plt.colorbar(im3, ax=axes[1,0])
    
    # Combined field intensity
    if field.system_wavefunction is not None:
        total_intensity = torch.abs(field.system_wavefunction).cpu().numpy()
        im4 = axes[1,1].imshow(total_intensity, cmap='plasma', origin='lower')
        axes[1,1].set_title('Total Field Intensity')
        plt.colorbar(im4, ax=axes[1,1])
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=== Unified Wave Simulation Prototype ===")
    print("This prototype demonstrates representing atoms/molecules as unified quantum fields")
    print("rather than separate interacting particles.\n")
    
    # Test dynamic H₂ formation from separate atoms
    print("1. Testing Dynamic H₂ Formation (separate atoms → molecular bond):")
    h_field, h_energies, h_charges = run_unified_hydrogen_simulation(
        create_video=True, 
        video_file="dynamic_h2_formation.avi"
    )
    
    print("\n" + "="*50 + "\n")
    
    # Test static H₂ molecule (pre-formed bonding orbital)
    print("2. Testing Static H₂ Molecule (pre-formed bonding orbital):")
    h2_field, h2_energies, h2_charges = run_unified_h2_simulation(
        create_video=True,
        video_file="static_h2_molecule.avi"
    )
    
    print("\n" + "="*50 + "\n")
    
    # Test two independent hydrogen atoms with video
    print("3. Testing Two Independent Hydrogen Atoms (with video):")
    two_h_field, two_h_energies, two_h_charges = run_two_hydrogen_atoms_simulation(
        create_video=True,
        video_file="two_hydrogen_atoms.avi"
    )
    
    print("\n=== Prototype Results ===")
    print("The unified approach successfully:")
    print("- Represents entire systems as single wavefunctions")
    print("- Evolves dynamics through self-consistent field equations")
    print("- Eliminates separate particle lists")
    print("- Provides more fundamental quantum description")
    print("- Generates videos showing unified field evolution")
    
    # Visualize final states (static plots)
    try:
        print("\nGenerating final state visualizations...")
        visualize_unified_field(h_field, "Hydrogen Atom - Final Unified Field")
        visualize_unified_field(h2_field, "H₂ Molecule - Final Unified Field")
        visualize_unified_field(two_h_field, "Two Independent Hydrogen Atoms - Final Unified Field")
    except Exception as e:
        print(f"Static visualization skipped: {e}")
        
    print("\nVideos generated:")
    print("- dynamic_h2_formation.avi: Two separate atoms evolving into H₂ molecule")
    print("- static_h2_molecule.avi: Pre-formed H₂ molecule with bonding orbital") 
    print("- two_hydrogen_atoms.avi: Two independent atoms showing interference")
