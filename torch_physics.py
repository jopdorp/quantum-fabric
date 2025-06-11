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
        """
        Time evolution using operator splitting with finite differences:
        ψ(t+dt) ≈ exp(-i*dt*V/2) * exp(-i*dt*T) * exp(-i*dt*V/2) * ψ(t)
        """
        # Split-step: V/2 -> T -> V/2
        potential_phase_half = torch.exp(-1j * self.dt * potential * 0.5)
        
        # Step 1: Apply potential for half time step
        psi = psi * potential_phase_half
        
        # Step 2: Apply kinetic operator using finite differences
        # Add batch and channel dimensions for conv2d
        psi_batched = psi.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
        
        # Apply Laplacian via convolution with padding
        laplacian_psi = torch.nn.functional.conv2d(
            psi_batched, 
            self.laplacian_kernel, 
            padding=1
        ).squeeze(0).squeeze(0)  # Remove batch/channel dims
        
        # Kinetic evolution: exp(-i*dt*T) where T = -∇²/2
        kinetic_phase = torch.exp(1j * self.dt * self.kinetic_factor * laplacian_psi)
        psi = psi * kinetic_phase
        
        # Step 3: Apply potential for remaining half time step  
        psi = psi * potential_phase_half
        
        return psi

_wave_model = None  # Global variable to hold the wave model instance
# Update the model selection
def get_wave_model(shape, dt=TIME_DELTA, device=DEVICE):
    """Get or create the wave propagation model."""
    global _wave_model
    if _wave_model is None or _wave_model.shape != shape or device != _wave_model.laplacian_kernel.device or dt != _wave_model.dt:
        _wave_model = LaplacianWaveModel(shape, dt, device)  # Use Laplacian model
    return _wave_model

def propagate_wave_with_potential(psi, potential, dt=TIME_DELTA, device=DEVICE):
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
    model = get_wave_model(psi.shape, dt, device)
    # Convert to torch tensors
    
    # Propagate
    result = model(psi, potential)  
    if is_numpy:
        # Convert back to numpy array
        return result.cpu().numpy()  
    return result
