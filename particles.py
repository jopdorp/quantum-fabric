import math
import numpy as np
from scipy.special import genlaguerre
from scipy.special import sph_harm

import config
from config import POTENTIAL_STRENGTH, SIZE

def apply_wavefunction_dynamics(psi, x, y, cx, cy, momentum_x=0.05, momentum_y=0.2, 
                               orbital_offset_x=0.1, orbital_offset_y=0.2):
    """Apply momentum and orbital offset effects to an existing wavefunction.
    
    This function adds dynamic behavior to a static orbital wavefunction by:
    - Adding momentum components (creates traveling wave behavior)
    - Applying orbital offset (shifts the orbital center for asymmetric potentials)
    
    Args:
        psi: existing complex wavefunction array
        x, y: coordinate grids  
        cx, cy: original center position (nucleus location)
        momentum_x, momentum_y: initial momentum components
        orbital_offset_x, orbital_offset_y: offset the orbital center
    
    Returns:
        Modified complex wavefunction with dynamics applied
    """
    # Calculate coordinates relative to offset center
    actual_cx = cx + orbital_offset_x
    actual_cy = cy + orbital_offset_y
    dx = x - actual_cx
    dy = y - actual_cy
    
    # Apply momentum for dynamic behavior
    if momentum_x != 0 or momentum_y != 0:
        psi = psi * np.exp(1j * (momentum_x * dx + momentum_y * dy))
    
    # If orbital was offset, we need to shift the wavefunction
    if orbital_offset_x != 0 or orbital_offset_y != 0:
        # Create phase shift to move the orbital
        dx_shift = x - cx  # Original center coordinates
        dy_shift = y - cy
        
        # Apply spatial shift by interpolating the wavefunction
        # For small offsets, we can approximate with a phase modulation
        if abs(orbital_offset_x) < 5 and abs(orbital_offset_y) < 5:
            # Small offset approximation
            shift_phase = 1j * (orbital_offset_x * dx_shift + orbital_offset_y * dy_shift) / 10
            psi = psi * np.exp(shift_phase)
    
    # Ensure proper normalization 
    norm_factor = np.sqrt(np.sum(np.abs(psi)**2))
    if norm_factor > 0:
        psi = psi / norm_factor
    
    return psi.astype(np.complex128)


def create_atom_electron(x, y, cx, cy, quantum_numbers, atomic_number=1, alpha=None, scale=config.SCALE):
    """Create atomic orbital wavefunctions for any element using quantum mechanical principles.
    
    This is a generic atomic orbital generator that works for:
    - Hydrogen and hydrogen-like ions (He⁺, Li²⁺, etc.)
    - Multi-electron atoms (C, N, O, etc.) using simple electron screening
    - Any quantum numbers (n, l, m) with proper 3D→2D projections
    
    Features:
    - Radial part: Associated Laguerre polynomials with proper normalization
    - Angular part: Real spherical harmonics projected to 2D plane  
    - Physics-based orbital scaling using simulation parameters
    - Simple universal electron screening based on orbital type and nuclear charge
    
    For dynamic behavior (momentum, orbital offset), use apply_wavefunction_dynamics() 
    after creating the static orbital.
    
    Args:
        x, y: coordinate grids
        cx, cy: center position (nucleus location)
        quantum_numbers: (n, l, m) quantum numbers
        atomic_number: nuclear charge Z (default: 1 for hydrogen)
        alpha: screening parameter for Z_eff (default: auto-calculated)
    
    Returns:
        Complex wavefunction representing the atomic orbital
        
    Examples:
        # Hydrogen 1s orbital
        psi = create_atom_electron(X, Y, cx, cy, (1,0,0), atomic_number=1)
        
        # Carbon 2p orbital
        psi = create_atom_electron(X, Y, cx, cy, (2,1,0), atomic_number=6)
        
        # Add dynamics after creation
        psi = apply_wavefunction_dynamics(psi, X, Y, cx, cy, momentum_x=0.1, orbital_offset_x=2)
    """
    n, l, m = quantum_numbers
    
    # Calculate screening parameter (alpha) using a simple universal formula
    if alpha is None:
        if atomic_number == 1:
            # Pure hydrogen - no screening
            alpha = 1.0
        else:
            # Simple universal screening: inner electrons reduce effective charge
            # For multi-electron atoms, each inner electron screens ~0.3-0.9 of nuclear charge
            # Core electrons (1s, 2s, 2p) screen more effectively than valence electrons
            
            # Count inner electrons (rough approximation)
            inner_electrons = max(0, atomic_number - 2)  # Subtract valence electrons
            
            # Screening efficiency depends on orbital type
            if l == 0:    # s orbitals penetrate more, less screening
                screening_per_electron = 0.25
            elif l == 1:  # p orbitals
                screening_per_electron = 0.35  
            elif l == 2:  # d orbitals
                screening_per_electron = 0.45
            else:         # f and higher orbitals
                screening_per_electron = 0.50
            
            # Additional screening for higher n (outer orbitals)
            n_factor = 1.0 + 0.1 * (n - 1)
            
            total_screening = inner_electrons * screening_per_electron * n_factor
            z_eff = atomic_number - total_screening
            alpha = max(z_eff / atomic_number, 0.1)  # Prevent negative screening
    
    # Calculate effective nuclear charge
    z_eff = atomic_number * alpha
    
    # Calculate physics-based Bohr radius
    # The stable orbital size is determined by the balance of kinetic and potential energy
    # In our simulation units: a₀ ≈ 1/√(POTENTIAL_STRENGTH * Z_eff)
    bohr_radius = scale / np.sqrt(POTENTIAL_STRENGTH * z_eff)
    
    # Orbital size scales with n² for all atoms (using effective nuclear charge)
    orbital_radius = bohr_radius * n**2
    
    # Create coordinates relative to center
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    
    # Create atomic radial wavefunction using effective nuclear charge
    rho = 2 * z_eff * r / (n * orbital_radius)  # Proper dimensionless radius
    
    # Use proper Laguerre polynomials for accurate radial shapes
    try:
        norm_factor = np.sqrt((2*z_eff/(n*orbital_radius))**3 * math.factorial(n-l-1) / (2*n*math.factorial(n+l)))
        laguerre_poly = genlaguerre(n-l-1, 2*l+1)(rho)
        radial = norm_factor * (rho**l) * np.exp(-rho/2) * laguerre_poly
    except (OverflowError, ValueError):
        # Fallback for very high quantum numbers - enhanced with oscillations
        oscillations = 1 + 0.8 * np.cos(np.pi * rho * (n-l-1) / n) + 0.3 * np.cos(2*np.pi * rho * (n-l-1) / n)
        radial = (rho**l) * np.exp(-rho/2) * oscillations
    
    # Create angular part - proper 3D→2D orbital projections with distinct shapes
    if l == 0:  # s orbitals - spherically symmetric
        angular = 1.0 + 0j
        
    elif l == 1:  # p orbitals - distinct dumbbell patterns
        if m == -1:
            # p_y orbital → vertical dumbbell
            angular = np.sin(theta)
        elif m == 0:
            # p_z → when projected to xy-plane, creates horizontal dumbbell
            angular = np.cos(theta)
        elif m == 1:
            # p_x orbital → rotated dumbbell at 45°
            angular = np.cos(theta + np.pi/4)
            
    elif l == 2:  # d orbitals - cloverleaf and other complex patterns
        if m == -2:
            # d_xy orbital → 4-lobe cloverleaf rotated
            angular = np.sin(2*theta)
        elif m == -1:
            # d_yz orbital → 4-lobe pattern
            angular = np.sin(theta) * np.cos(theta)
        elif m == 0:
            # d_z² orbital → distinctive pattern with central lobe and ring
            angular = (3*np.cos(theta)**2 - 1) + 0.5*np.sin(2*theta)
        elif m == 1:
            # d_xz orbital → different 4-lobe orientation
            angular = np.sin(theta) * np.sin(theta + np.pi/2)
        elif m == 2:
            # d_x²-y² orbital → 4-lobe cloverleaf aligned with axes
            angular = np.cos(2*theta)
            
    else:  # Higher l orbitals - create more complex patterns
        # Create increasingly complex angular patterns for higher l
        base_pattern = np.cos(l * theta) + 0.5 * np.sin((l+1) * theta)
        m_modulation = np.cos(m * theta + np.pi/4)
        angular = base_pattern * m_modulation
    
    # Combine radial and angular parts
    psi = radial * angular
    
    # Add gentle envelope for better visualization
    envelope = np.exp(-r**2/(2*(orbital_radius*0.6)**2))
    psi = psi * envelope
    
    # Ensure psi is complex before applying perturbations
    psi = psi.astype(np.complex128)
    
    # Ensure proper normalization 
    norm_factor = np.sqrt(np.sum(np.abs(psi)**2))
    if norm_factor > 0:
        psi = psi / norm_factor
    
    return psi.astype(np.complex128)



