import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.ndimage import gaussian_filter
from config import (
    POTENTIAL_STRENGTH,
    NUCLEAR_REPULSION_STRENGTH,
    NUCLEAR_CORE_RADIUS,
    COULOMB_STRENGTH,
    STRONG_FORCE_STRENGTH,
    STRONG_FORCE_RANGE,
    TIME_DELTA,
    X,Y
)


def create_nucleus_potential(x, y, nucleus_x, nucleus_y, charge=1):
    """Create a more realistic nucleus potential with short-range repulsion."""
    r = np.sqrt((x - nucleus_x)**2 + (y - nucleus_y)**2)
    r = np.maximum(r, 0.1)  # Smaller minimum to allow closer approach
    
    # Long-range Coulomb attraction: V = -k*Z/r (attractive for electrons)
    coulomb_attraction = -POTENTIAL_STRENGTH * charge / r
    
    # Short-range repulsion to prevent collapse into nucleus (models quantum effects)
    # This represents the Pauli exclusion principle and quantum uncertainty
    nuclear_repulsion = NUCLEAR_REPULSION_STRENGTH * np.exp(-r / NUCLEAR_CORE_RADIUS) / (r + 0.1)
    
    return coulomb_attraction + nuclear_repulsion

def add_mean_field_coulomb_repulsion(source_psi, strength=0.05, sigma=5):
    """Create a repulsive potential based on the source electron density."""
    charge_density = np.abs(source_psi)**2
    repulsion = gaussian_filter(charge_density, sigma=sigma)
    return strength * repulsion


def compute_force_from_density(density, pos):
    dx = X - pos[0]; dy = Y - pos[1]
    r = np.sqrt(dx**2 + dy**2); r = np.maximum(r, 1.0)
    fx = np.sum((dx / r**3) * density)
    fy = np.sum((dy / r**3) * density)
    return np.array([fx, fy])

def compute_force_from_density(charge_density, nucleus_pos):
    """Compute forces on nucleus from electron density."""
    dx = X - nucleus_pos[0]
    dy = Y - nucleus_pos[1]
    r = np.sqrt(dx**2 + dy**2)
    r = np.maximum(r, 1.0)
    force_x = np.sum((dx / r**3) * charge_density)
    force_y = np.sum((dy / r**3) * charge_density)
    return np.array([force_x, force_y])

def compute_nuclear_force(nucleus1_pos, nucleus2_pos):
    """Compute forces between two nuclei including Coulomb repulsion and nuclear binding."""
    diff = nucleus2_pos - nucleus1_pos
    r = np.linalg.norm(diff)
    r = max(r, 0.5)  # Prevent singularity
    
    # Coulomb repulsion between protons: F = k*q1*q2/r^2
    coulomb_force = COULOMB_STRENGTH * diff / r**3
    
    # Strong nuclear force (attractive at very short range, models neutron-proton binding)
    if r < STRONG_FORCE_RANGE:
        # Exponentially decaying attractive force
        nuclear_attraction = -STRONG_FORCE_STRENGTH * np.exp(-r / STRONG_FORCE_RANGE) * diff / r
    else:
        nuclear_attraction = np.zeros(2)
    
    return coulomb_force + nuclear_attraction

def propagate_wave_with_potential(psi, potential, dt=TIME_DELTA):
    potential_phase = np.exp(-1j * dt * potential / 2)
    psi = psi * potential_phase
    psi_hat = np.fft.fft2(psi)
    kx = np.fft.fftfreq(psi.shape[1]) * 2 * np.pi
    ky = np.fft.fftfreq(psi.shape[0]) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    kinetic_phase = np.exp(-1j * dt * (KX**2 + KY**2) / 2)
    psi_hat *= kinetic_phase
    psi = np.fft.ifft2(psi_hat)
    psi = psi * potential_phase
    return psi

### Some unused utility functions below, kept for reference

def enhanced_electron_electron_repulsion(psi1, psi2, electron_repulsion_strength=0.1):
    """Enhanced electron-electron repulsion with better physical modeling."""
    density1 = np.abs(psi1)**2
    density2 = np.abs(psi2)**2
    
    # Create repulsion potential based on both electron densities
    # Use different sigma values for short and long range interactions
    short_range_repulsion = gaussian_filter(density2, sigma=2) * electron_repulsion_strength * 2
    long_range_repulsion = gaussian_filter(density2, sigma=8) * electron_repulsion_strength * 0.5
    
    return short_range_repulsion + long_range_repulsion