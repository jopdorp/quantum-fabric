import numpy as np
import torch
from config import SIZE, X, Y, center_x, center_y

def normalize_wavefunction(psi):
    """Normalize psi so that sum(|psi|^2)==1 - torch version"""
    # Handle both numpy arrays and torch tensors
    was_numpy = isinstance(psi, np.ndarray)
    
    # Ensure data is a torch tensor
    if not isinstance(psi, torch.Tensor):
        psi = torch.tensor(psi, dtype=torch.complex64)
    
    norm = torch.sqrt(torch.sum(torch.abs(psi)**2))
    result = psi / norm if norm > 0 else psi
    
    # Convert back to numpy if input was numpy
    if was_numpy:
        result = result.cpu().numpy()
    
    return result

def apply_low_pass_filter(psi, cutoff, device=None):
    """Torch-optimized low-pass filter with cached frequency grids"""
    # Handle both numpy arrays and torch tensors
    was_numpy = isinstance(psi, np.ndarray)
    
    # Ensure data is a torch tensor
    if not isinstance(psi, torch.Tensor):
        psi = torch.tensor(psi, dtype=torch.complex64)
    
    if device is None:
        device = psi.device
    psi = psi.to(device)
    
    shape = psi.shape[-2:]  # Get last 2 dimensions for 2D FFT

    # Pre-compute and cache frequency grid on the correct device
    kx = torch.fft.fftfreq(shape[1], device=device) * (2 * torch.pi)
    ky = torch.fft.fftfreq(shape[0], device=device) * (2 * torch.pi)
    KX, KY = torch.meshgrid(kx, ky, indexing='xy')
    k_squared = KX**2 + KY**2

    
    # Apply filter using torch FFT
    psi_hat = torch.fft.fft2(psi, dim=(-2, -1))  # FFT over last 2 dimensions
    mask = k_squared <= cutoff**2
    result = torch.fft.ifft2(psi_hat * mask, dim=(-2, -1))
    
    # Convert back to numpy if input was numpy
    if was_numpy:
        result = result.cpu().numpy()
    
    return result

def apply_absorbing_edge(psi, strength=1):
    """Apply absorbing edge boundary - torch version"""
    # Handle both numpy arrays and torch tensors
    was_numpy = isinstance(psi, np.ndarray)
    
    # Ensure data is a torch tensor
    if not isinstance(psi, torch.Tensor):
        psi = torch.tensor(psi, dtype=torch.complex64)
    
    device = psi.device
    
    # Convert coordinate grids to torch tensors on the same device
    x_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(Y, dtype=torch.float32, device=device)
    
    r = torch.sqrt((x_tensor - center_x)**2 + (y_tensor - center_y)**2)
    
    # Start absorption at 80% of max possible radius, total absorption at frame edge
    max_radius = min(center_x, center_y, SIZE - center_x, SIZE - center_y)
    absorption_start = max_radius * 0.7
    absorption_end = max_radius * 0.96  # Ensure total absorption before frame edge
    
    # Create mask with the same shape as psi
    mask = torch.ones_like(psi, dtype=torch.float32, device=device)
    
    # Handle different tensor shapes (2D vs 3D with batch dimension)
    if len(psi.shape) == 2:
        # Standard 2D wavefunction
        outer_region = r > absorption_start
        if torch.any(outer_region):
            absorption_depth = (r[outer_region] - absorption_start) / (absorption_end - absorption_start)
            absorption_depth = torch.clamp(absorption_depth, 0, 1)
            mask[outer_region] = torch.exp(-strength * absorption_depth**3)
    else:
        # Handle higher dimensional tensors (e.g., batched)
        # Broadcast the outer_region mask to match psi dimensions
        outer_region = r > absorption_start
        if torch.any(outer_region):
            absorption_depth = (r[outer_region] - absorption_start) / (absorption_end - absorption_start)
            absorption_depth = torch.clamp(absorption_depth, 0, 1)
            absorption_values = torch.exp(-strength * absorption_depth**3)
            
            # Apply mask across all dimensions
            for batch_idx in range(psi.shape[0]) if len(psi.shape) > 2 else [0]:
                if len(psi.shape) > 2:
                    mask[batch_idx][outer_region] = absorption_values
                else:
                    mask[outer_region] = absorption_values
    
    result = psi * mask
    
    # Convert back to numpy if input was numpy
    if was_numpy:
        result = result.cpu().numpy()
    
    return result


smoothing_factor = 200000
smooth_cy, smooth_cx = SIZE // 2, SIZE // 2

def center_wave(psi):
    global smooth_cy, smooth_cx
    prob = np.abs(psi)**2
    total = np.sum(prob)
    if total == 0:
        return psi
    y_idx, x_idx = np.indices(prob.shape)
    cy = np.sum(y_idx * prob) / total
    cx = np.sum(x_idx * prob) / total
    smooth_cy = smooth_cy + (cy - smooth_cy) / smoothing_factor
    smooth_cx = smooth_cx + (cx - smooth_cx) / smoothing_factor
    shift_y = int(np.round((SIZE // 2) - smooth_cy))
    shift_x = int(np.round((SIZE // 2) - smooth_cx))
    return np.roll(np.roll(psi, shift_y, axis=0), shift_x, axis=1)

def limit_frame(psi1):
    psi1 = apply_absorbing_edge(psi1)
    psi1 = apply_low_pass_filter(psi1, cutoff=1.2)

    # psi1 = center_wave(psi1)
    psi1 = normalize_wavefunction(psi1)
    return psi1