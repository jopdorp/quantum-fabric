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

_wave_models = {"fft": None, "laplacian": None}  # Global variable to hold wave model instances

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
    elif propagation_method == "laplacian":
        if model is None or model.shape != shape or \
           not hasattr(model, 'laplacian_kernel') or \
           device != model.laplacian_kernel.device or model.dt != dt:
            model = LaplacianWaveModel(shape, dt, device)
            _wave_models["laplacian"] = model
    else:
        raise ValueError(f"Unknown propagation_method: {propagation_method}. Choose 'fft' or 'laplacian'.")
        
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
