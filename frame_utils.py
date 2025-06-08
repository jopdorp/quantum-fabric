import numpy as np
from config import SIZE, X, Y, center_x, center_y

def normalize_wavefunction(psi):
    """Normalize psi so that sum(|psi|^2)==1"""
    norm = np.sqrt(np.sum(np.abs(psi)**2))
    return psi / norm if norm > 0 else psi

def apply_low_pass_filter(psi, cutoff):
    psi_hat = np.fft.fft2(psi)
    kx = np.fft.fftfreq(psi.shape[1])*2*np.pi
    ky = np.fft.fftfreq(psi.shape[0])*2*np.pi
    KX, KY = np.meshgrid(kx, ky)
    mask = (KX**2 + KY**2) <= cutoff**2
    return np.fft.ifft2(psi_hat * mask)

def apply_absorbing_edge(psi, strength=1):
    # Circular absorbing mask - starts gentle, becomes total at frame edges
    r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Start absorption at 80% of max possible radius, total absorption at frame edge
    max_radius = min(center_x, center_y, SIZE - center_x, SIZE - center_y)
    absorption_start = max_radius * 0.7
    absorption_end = max_radius * 0.96  # Ensure total absorption before frame edge
    
    mask = np.ones(psi.shape, dtype=np.float32)
    
    # Only apply absorption in outer region
    outer_region = r > absorption_start
    if np.any(outer_region):
        # Smooth exponential absorption that ramps up toward edges
        absorption_depth = (r[outer_region] - absorption_start) / (absorption_end - absorption_start)
        absorption_depth = np.clip(absorption_depth, 0, 1)
        # Exponential curve for strong absorption near edges
        mask[outer_region] = np.exp(-strength * absorption_depth**3)
    
    return psi * mask


smoothing_factor = 20000
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
    psi1 = apply_low_pass_filter(psi1, cutoff=1)

    psi1 = center_wave(psi1)
    psi1 = normalize_wavefunction(psi1)
    return psi1