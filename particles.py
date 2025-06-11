import numpy as np
import config
from config import POTENTIAL_STRENGTH, SIZE


def apply_wavefunction_dynamics(psi, x, y, cx, cy, momentum_x=0.05, momentum_y=0.2, 
                               orbital_offset_x=0.-1, orbital_offset_y=0.2):
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


def create_atom_electron(x, y, cx, cy, quantum_numbers, **kwargs):
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
    atomic_number = kwargs.get('atomic_number', 1)
    alpha = kwargs.get('alpha', None)
    
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
    bohr_radius = config.SCALE / np.sqrt(POTENTIAL_STRENGTH * z_eff)
    
    # Orbital size scales with n² for all atoms (using effective nuclear charge)
    orbital_radius = bohr_radius * n**2
    
    # # Additional size correction to ensure stability in the simulation
    # # The orbital should be roughly 1/4 of the potential well size
    # max_stable_radius = SIZE / 8  # pixels
    # if orbital_radius > max_stable_radius:
    #     orbital_radius = max_stable_radius
    #     print(f"Warning: Orbital radius clamped to {max_stable_radius:.1f} pixels for stability")
    
    # Create coordinates relative to center
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    
    # Create atomic radial wavefunction using effective nuclear charge
    rho = 2 * z_eff * r / (n * orbital_radius)  # Proper dimensionless radius
    
    # General atomic radial function with Laguerre polynomials
    # This works for any atom using the effective nuclear charge Z_eff
    from scipy.special import genlaguerre
    import math
    
    # Prevent overflow for very large factorials by using safe computation
    try:
        norm_factor = np.sqrt((2*z_eff/(n*orbital_radius))**3 * math.factorial(n-l-1) / (2*n*math.factorial(n+l)))
        laguerre_poly = genlaguerre(n-l-1, 2*l+1)(rho)
        radial = norm_factor * (rho**l) * np.exp(-rho/2) * laguerre_poly
    except (OverflowError, ValueError):
        # Fallback for very high quantum numbers - use simpler approximation
        radial = (rho**l) * np.exp(-rho/2) * (1 + 0.5 * np.cos(np.pi * rho * (n-l-1)))
    
    # Create angular part - proper 3D→2D orbital projections
    if l == 0:  # s orbitals
        angular = 1.0 + 0j  # Spherically symmetric
        
    elif l == 1:  # p orbitals - proper projections
        if m == -1:
            # p_x - i*p_y → creates lobe at 45° angle
            angular = np.cos(theta - np.pi/4) * np.exp(1j * theta)
        elif m == 0:
            # p_z → when projected to xy-plane, creates dipole pattern
            angular = np.cos(theta)  # Creates two-lobe pattern along x-axis
        elif m == 1:
            # p_x + i*p_y → creates lobe at -45° angle  
            angular = np.cos(theta + np.pi/4) * np.exp(-1j * theta)
            
    elif l == 2:  # d orbitals - copy approach from hydrogen_utils.py
        if m == -2:
            angular = np.sin(2*theta) * np.exp(2j * theta)  # d_xy
        elif m == -1:
            angular = np.sin(theta) * np.cos(theta) * np.exp(1j * theta)  # d_xz/d_yz
        elif m == 0:
            angular = (3*np.cos(theta)**2 - 1)  # d_z² → cloverleaf in 2D
        elif m == 1:
            angular = np.sin(theta) * np.cos(theta) * np.exp(-1j * theta)  # d_xz/d_yz
        elif m == 2:
            angular = np.cos(2*theta) * np.exp(-2j * theta)  # d_x²-y²
            
    else:  # Higher l orbitals - use spherical harmonics
        from scipy.special import sph_harm
        phi = theta
        # For 2D projection, use theta=π/2 (xy-plane)
        angular = sph_harm(m, l, phi, np.pi/2)
    
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



