import torch
import torch.nn as nn
import numpy as np
import torch.fft
import torch.xpu  # Import Intel XPU support if available
from config import TIME_DELTA

# Check if Intel XPU is available
def get_device():
    """Get the best available device (Intel XPU, CUDA, or CPU)"""
    if torch.xpu.is_available():
        print(f"Using Intel XPU: {torch.xpu.get_device_name()}")
        return torch.device("xpu")
    elif torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
        return torch.device("cuda")
    else:
        print("Using CPU with PyTorch")
        return torch.device("cpu")

# Global device
DEVICE = get_device()


def convert_to_device(tensor, device=DEVICE):
    # Convert inputs to torch tensors on device
    if isinstance(tensor, np.ndarray):
        converted_tensor = torch.from_numpy(tensor).to(device)
    else:
        converted_tensor = tensor.to(DEVICE)
    
    converted_tensor = converted_tensor.to(torch.complex64)
    return converted_tensor

class WavePropagationModel(nn.Module):
    def __init__(self, shape, dt=TIME_DELTA, device=DEVICE):
        super().__init__()
        self.shape = shape
        self.dt = dt
        
        # Pre-compute and register kinetic phase as buffer (not trainable)
        kx = torch.fft.fftfreq(shape[1], d=1.0, device=device) * (2 * torch.pi)
        ky = torch.fft.fftfreq(shape[0], d=1.0, device=device) * (2 * torch.pi)
        KY, KX = torch.meshgrid(ky, kx, indexing='ij')
        
        kinetic_phase = torch.exp(-1j * dt * (KX**2 + KY**2) * 0.5)
        self.register_buffer('kinetic_phase', kinetic_phase)
        
    def forward(self, psi, potential):
        """Split-step propagation: V/2 -> T -> V/2"""
        potential_phase = torch.exp(-1j * self.dt * potential * 0.5)
        
        # Split-step propagation
        psi = psi * potential_phase
        psi_hat = torch.fft.fft2(psi)
        psi_hat = psi_hat * self.kinetic_phase
        psi = torch.fft.ifft2(psi_hat)
        psi = psi * potential_phase
        
        return psi
        
    def forward_batch(self, psi_batch, potential_batch):
        """Batched split-step propagation for multiple electrons simultaneously"""
        # psi_batch shape: [N_electrons, H, W]
        # potential_batch shape: [N_electrons, H, W]
        
        potential_phase = torch.exp(-1j * self.dt * potential_batch * 0.5)
        
        # Split-step propagation with batching
        psi_batch = psi_batch * potential_phase
        psi_hat_batch = torch.fft.fft2(psi_batch, dim=(-2, -1))  # FFT over spatial dims
        psi_hat_batch = psi_hat_batch * self.kinetic_phase.unsqueeze(0)  # Broadcast kinetic phase
        psi_batch = torch.fft.ifft2(psi_hat_batch, dim=(-2, -1))
        psi_batch = psi_batch * potential_phase
        
        return psi_batch

class LaplacianWaveModel(nn.Module):
    def __init__(self, shape, dt=TIME_DELTA, device=DEVICE):
        super().__init__()
        self.shape = shape
        self.dt = dt
        
        # Pre-compute Laplacian finite difference kernels
        # Second derivative approximation: f''(x) ≈ (f(x+h) - 2f(x) + f(x-h))/h²
        # For 2D Laplacian: ∇²ψ = ∂²ψ/∂x² + ∂²ψ/∂y²
        
        # 5-point stencil Laplacian kernel
        laplacian_kernel = torch.tensor([
            [0.0,  1.0, 0.0],
            [1.0, -4.0, 1.0], 
            [0.0,  1.0, 0.0]
        ], dtype=torch.complex64, device=device).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('laplacian_kernel', laplacian_kernel)
        
        # For kinetic energy operator: T = -ℏ²/(2m) * ∇²
        # We'll use h=1 and ℏ²/(2m) = 0.5 for simplicity
        self.kinetic_factor = 0.5

    def forward(self, psi, potential):
        """Single instance propagation using finite differences."""
        # Unsqueeze to create a batch of 1
        psi_batch = psi.unsqueeze(0)
        potential_batch = potential.unsqueeze(0)
        
        # Propagate using the batch method
        result_batch = self.forward_batch(psi_batch, potential_batch)
        
        # Squeeze to remove the batch dimension
        return result_batch.squeeze(0)
        
    def forward_batch(self, psi_batch, potential_batch):
        """
        Batched time evolution using operator splitting with finite differences:
        ψ(t+dt) ≈ exp(-i*dt*V/2) * exp(-i*dt*T) * exp(-i*dt*V/2) * ψ(t)
        """
        # Split-step: V/2 -> T -> V/2
        potential_phase_half = torch.exp(-1j * self.dt * potential_batch * 0.5)
        
        # Step 1: Apply potential for half time step
        psi_batch = psi_batch * potential_phase_half
        
        # Step 2: Apply kinetic operator using finite differences
        # Add batch and channel dimensions for conv2d if needed
        if len(psi_batch.shape) == 3:  # [N, H, W]
            psi_batched = psi_batch.unsqueeze(1)  # Shape: [N, 1, H, W]
        else:
            psi_batched = psi_batch
        
        # Apply Laplacian via convolution with padding
        # Fix: Don't use grouped convolution, just apply the same kernel to all batch elements
        laplacian_psi = torch.nn.functional.conv2d(
            psi_batched, 
            self.laplacian_kernel,  # Same kernel for all
            padding=1
        )
        
        if len(psi_batch.shape) == 3:  # Remove channel dim if we added it
            laplacian_psi = laplacian_psi.squeeze(1)
        
        # Kinetic evolution: exp(-i*dt*T) where T = -∇²/2
        kinetic_phase = torch.exp(1j * self.dt * self.kinetic_factor * laplacian_psi)
        psi_batch = psi_batch * kinetic_phase
        
        # Step 3: Apply potential for remaining half time step  
        psi_batch = psi_batch * potential_phase_half
        
        return psi_batch

# Update the model selection
def get_wave_model(shape, propagation_method: str, dt=TIME_DELTA, device=DEVICE):
    """Get or create the wave propagation model based on the chosen method."""
    global _wave_models
    
    model = _wave_models.get(propagation_method)
    
    if propagation_method == "fft":
        if model is None or model.shape != shape or \
           not hasattr(model, 'kinetic_phase') or \
           device != model.kinetic_phase.device or model.dt != dt:
            model = WavePropagationModel(shape, dt, device)
            _wave_models["fft"] = model
    elif propagation_method == "fft_heavy_damping":
        if model is None or model.shape != shape or model.dt != dt:
            model = DetailPreservingFFTModel(shape, dt, device, mode="adaptive_precision")
            _wave_models["fft_heavy_damping"] = model
    elif propagation_method == "fft_medium_damping":
        if model is None or model.shape != shape or model.dt != dt:
            model = DetailPreservingFFTModel(shape, dt, device, mode="detail_preserving")
            _wave_models["fft_medium_damping"] = model
    elif propagation_method == "fft_light_damping":
        if model is None or model.shape != shape or model.dt != dt:
            model = DetailPreservingFFTModel(shape, dt, device, mode="selective_damping")
            _wave_models["fft_light_damping"] = model
    elif propagation_method == "laplacian":
        if model is None or model.shape != shape or model.dt != dt:
            model = LaplacianWaveModel(shape, dt, device)
            _wave_models["laplacian"] = model
    else:
        available = ["fft", "fft_heavy_damping", "fft_medium_damping", "fft_light_damping", "laplacian"]
        raise ValueError(f"Unknown propagation_method: {propagation_method}. Choose from {available}")
        
    return model

def propagate_wave_with_potential(psi, potential, propagation_method="fft", dt=TIME_DELTA, device=DEVICE):
    """
    PyTorch model-based wave propagation using split-step Fourier method.
    Much faster due to pre-computed tensors and optimized computation graph.
    """
    is_numpy = isinstance(psi, np.ndarray) or isinstance(potential, np.ndarray)
    if is_numpy:
        # Convert numpy arrays to torch tensors
        psi = convert_to_device(psi, device)
        potential = convert_to_device(potential, device)

    psi = convert_to_device(psi, device)
    potential = convert_to_device(potential, device)
    model = get_wave_model(psi.shape, propagation_method, dt, device)
    # Convert to torch tensors
    
    # Propagate
    result = model(psi, potential)  
    if is_numpy:
        # Convert back to numpy array
        return result.cpu().numpy()  
    return result

def propagate_wave_batch_with_potentials(psi_list, potential_list, propagation_method="fft", dt=TIME_DELTA, device=DEVICE):
    """
    Batched wave propagation for multiple electrons simultaneously.
    This is much faster than calling propagate_wave_with_potential in a loop.
    
    Args:
        psi_list: List of wave functions [psi1, psi2, ...]
        potential_list: List of potentials [V1, V2, ...]
        dt: Time step
        device: Device to run on
        
    Returns:
        List of propagated wave functions
    """
    if len(psi_list) == 0:
        return []
    
    # Check if inputs are numpy arrays to determine output format
    is_numpy = isinstance(psi_list[0], np.ndarray)
    
    # Convert all to torch tensors and stack into batches
    psi_tensors = []
    potential_tensors = []
    
    for psi, potential in zip(psi_list, potential_list):
        psi_tensor = convert_to_device(psi, device)
        potential_tensor = convert_to_device(potential, device)
        psi_tensors.append(psi_tensor)
        potential_tensors.append(potential_tensor)
    
    # Stack into batch tensors: [N_electrons, H, W]
    psi_batch = torch.stack(psi_tensors, dim=0)
    potential_batch = torch.stack(potential_tensors, dim=0)
    
    # Get model and propagate batch
    # Shape for the model is the shape of individual wavefunctions, not the batch shape
    model = get_wave_model(psi_batch.shape[1:], propagation_method, dt, device)
    result_batch = model.forward_batch(psi_batch, potential_batch)
    
    # Convert back to list
    result_list = []
    for i in range(result_batch.shape[0]):
        result = result_batch[i]
        if is_numpy:
            result = result.cpu().numpy()
        result_list.append(result)
    
    return result_list

# Monitoring function for debugging

def check_wavefunction_health(psi, step=0, name="wavefunction"):
    """Monitor wavefunction for numerical issues"""
    norm = torch.sqrt(torch.sum(torch.abs(psi)**2))
    max_val = torch.max(torch.abs(psi))
    min_val = torch.min(torch.abs(psi))
    
    # Check for problematic conditions
    issues = []
    if norm == 0:
        issues.append("zero norm")
    elif norm > 10.0:
        issues.append(f"large norm ({norm:.3f})")
    elif norm < 0.1:
        issues.append(f"small norm ({norm:.3f})")
    
    if max_val > 100.0:
        issues.append(f"large amplitude ({max_val:.3f})")
    
    if torch.isnan(psi).any():
        issues.append("NaN values")
    
    if torch.isinf(psi).any():
        issues.append("infinite values")
    
    if issues and step % 100 == 0:  # Report occasionally to avoid spam
        print(f"WARNING: {name} at step {step} has issues: {', '.join(issues)}")
    
    return len(issues) == 0

class DetailPreservingFFTModel(nn.Module):
    """
    Enhanced FFT model that preserves maximum detail while ensuring stability.
    Uses selective filtering instead of blunt stabilization.
    """
    
    def __init__(self, shape, dt=TIME_DELTA, device=DEVICE, mode="detail_preserving"):
        super().__init__()
        self.shape = shape
        self.dt = dt
        self.mode = mode
        
        # Pre-compute k-space coordinates
        kx = torch.fft.fftfreq(shape[1], d=1.0, device=device) * (2 * torch.pi)
        ky = torch.fft.fftfreq(shape[0], d=1.0, device=device) * (2 * torch.pi)
        KY, KX = torch.meshgrid(ky, kx, indexing='ij')
        k_squared = KX**2 + KY**2
        
        # For detail preservation, we rely on selective filtering instead of substeps
        # This preserves the speed benefit of larger time steps
        self.n_substeps = 1  # No substeps - use filtering for stability
        self.effective_dt = dt
        
        # Create selective stabilization filters
        k_magnitude = torch.sqrt(k_squared)
        k_nyquist = torch.pi
        
        if mode == "detail_preserving":
            # Only filter the most unstable high frequencies (top 2%) - more aggressive preservation
            k_cutoff = k_nyquist * 0.98
            stability_filter = torch.where(
                k_magnitude <= k_cutoff,
                torch.ones_like(k_magnitude),
                torch.exp(-((k_magnitude - k_cutoff) / (0.01 * k_nyquist))**6)  # Sharp cutoff for detail preservation
            )
        elif mode == "selective_damping":
            # Apply very mild damping to high frequencies while preserving detail
            # Focus only on preventing the most extreme instabilities
            k_critical = k_nyquist * 0.85  # Keep 85% of frequencies untouched
            damping_strength = k_squared * self.effective_dt / (torch.pi * 0.5)
            stability_filter = torch.where(
                k_magnitude <= k_critical,
                torch.ones_like(k_magnitude),  # No damping for most frequencies
                torch.exp(-0.05 * torch.maximum(torch.tensor(0.0, device=device), damping_strength - 1.0))  # Very mild damping
            )
        elif mode == "adaptive_precision":
            # Use higher precision for critical frequencies
            # This preserves important physics while stabilizing
            physics_relevant_k = k_nyquist * 0.3  # Keep low-mid frequencies intact
            stability_filter = torch.where(
                k_magnitude <= physics_relevant_k,
                torch.ones_like(k_magnitude),  # Perfect preservation
                torch.exp(-((k_magnitude - physics_relevant_k) / (0.3 * k_nyquist))**4)
            )
        
        kinetic_phase = torch.exp(-1j * self.effective_dt * k_squared * 0.5)
        self.register_buffer('kinetic_phase', kinetic_phase)
        self.register_buffer('stability_filter', stability_filter)
        
        # Calculate preserved frequency range
        preserved_fraction = torch.sum(stability_filter > 0.99) / torch.numel(stability_filter)
        print(f"Detail-Preserving FFT Model ({mode}):")
        print(f"  Single step dt: {self.effective_dt:.6f} (no substeps)")
        print(f"  Frequency preservation: {preserved_fraction:.1%}")
        print(f"  Detail loss: Minimal (only top {100*(1-preserved_fraction):.1f}% frequencies)")
        
    def forward(self, psi, potential):
        """Single instance propagation."""
        return self.forward_batch(psi.unsqueeze(0), potential.unsqueeze(0)).squeeze(0)
        
    def forward_batch(self, psi_batch, potential_batch):
        """Enhanced FFT with minimal detail loss - single step with selective filtering"""
        potential_phase = torch.exp(-1j * self.effective_dt * potential_batch * 0.5)
        
        # Split-step: V/2 -> T -> V/2
        psi_batch = psi_batch * potential_phase
        psi_hat_batch = torch.fft.fft2(psi_batch, dim=(-2, -1))
        
        # Apply selective stabilization (preserves most detail)
        psi_hat_batch = psi_hat_batch * self.stability_filter.unsqueeze(0)
        
        # Kinetic evolution
        psi_hat_batch = psi_hat_batch * self.kinetic_phase.unsqueeze(0)
        psi_batch = torch.fft.ifft2(psi_hat_batch, dim=(-2, -1))
        psi_batch = psi_batch * potential_phase
        
        return psi_batch

# Extend global wave models dictionary
_wave_models = {
    "fft": None, 
    "fft_heavy_damping": None,
    "fft_medium_damping": None,
    "fft_light_damping": None,
    "laplacian": None
}  # Global variable to hold wave model instances
