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


# TODO: Phase damping is not working
def apply_absorbing_edge(psi, strength=1):
    r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    # Not sure why we need 0.35, but it seems to work well
    # This is the radius at which the wavefunction starts to be absorbed
    absorb_r = min(SIZE * 0.35, center_x - 30)
    mask = np.ones_like(r)
    outside = r > absorb_r
    mask[outside] = np.exp(-strength * (r[outside] - absorb_r))
    mask 
    return psi * mask

### Some unused utility functions below, kept for reference
def add_noise(psi, noise_level=0.001):
    noise = (np.random.rand(*psi.shape) - 0.5) * noise_level
    return psi + noise.astype(np.complex128)

def apply_edge_blur(psi, blur_factor=SIZE // 2):
    """Apply a large Gaussian blur to the edges of the world."""
    r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    edge_blur = np.exp(-r**2 / (2 * blur_factor**2))
    return psi * edge_blur

def reverse_gaussian_blur(psi, center_x, center_y):
    """Apply an optimized reverse Gaussian blur with stronger blur for 50% of the image."""
    r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_sigma = SIZE // 2  # Maximum blur for 50% of the image
    min_sigma = 1          # Minimum blur at the center
    sigma_map = min_sigma + (max_sigma - min_sigma) * (r / np.max(r))

    # Precompute a spatially varying kernel
    blurred_psi = gaussian_filter(psi, sigma=max_sigma)

    # Blend the original and blurred wavefunction based on sigma_map
    blend_factor = (sigma_map - min_sigma) / (max_sigma - min_sigma)
    optimized_psi = psi * (1 - blend_factor) + blurred_psi * blend_factor

    return optimized_psi

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