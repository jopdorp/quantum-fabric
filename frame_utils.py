import numpy as np
from scipy.ndimage import gaussian_filter
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

def apply_absorbing_edge(psi, strength=5):
    # Circular absorbing mask - starts gentle, becomes total at frame edges
    r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Start absorption at 80% of max possible radius, total absorption at frame edge
    max_radius = min(center_x, center_y, SIZE - center_x, SIZE - center_y)
    absorption_start = max_radius * 0.8
    absorption_end = max_radius * 0.98  # Ensure total absorption before frame edge
    
    mask = np.ones(psi.shape, dtype=np.float32)
    
    # Only apply absorption in outer region
    outer_region = r > absorption_start
    if np.any(outer_region):
        # Smooth exponential absorption that ramps up toward edges
        absorption_depth = (r[outer_region] - absorption_start) / (absorption_end - absorption_start)
        absorption_depth = np.clip(absorption_depth, 0, 1)
        # Exponential curve for strong absorption near edges
        mask[outer_region] = np.exp(-strength * absorption_depth**3 * 10)
    
    return psi * mask


radius=SIZE * 0.35
r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
transition_start = radius * 0.7
transition_end = radius * 1.5
transition_width = transition_end - transition_start
normalized_distance = (r - transition_start) / transition_width
transition_mask = np.clip(normalized_distance, 0, 1)
transition_mask = transition_mask * transition_mask * (3 - 2 * transition_mask)
blur_region = r > transition_start

def blur_edges(psi, blur_strength=100.0):
    """Ultra-fast blur using precomputed masks and separable filtering."""
    # Use separable filtering for speed
    from scipy.ndimage import gaussian_filter1d
    sigma = blur_strength / 3  # Compensate for separable filtering
    blurred_psi = gaussian_filter1d(gaussian_filter1d(psi, sigma, axis=0), sigma, axis=1)
    
    # Apply precomputed transition mask
    return psi * (1 - transition_mask) + blurred_psi * transition_mask

def center_wave(psi):
    global smooth_cy, smooth_cx, smoothing_factor
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