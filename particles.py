import numpy as np
from config import POTENTIAL_STRENGTH, SCALE

def create_electron_wavepacket(x, y, cx, cy, sigma, momentum_x=0, momentum_y=0):
    """Create a simple electron wavepacket with optional momentum.
    
    Args:
        x, y: coordinate grids
        cx, cy: center position
        sigma: width of the wavepacket
        momentum_x, momentum_y: initial momentum
    """
    dx = x - cx
    dy = y - cy
    r_squared = dx*dx + dy*dy
    
    # Gaussian wavepacket
    psi = np.exp(-r_squared / (2 * sigma**2))
    
    # Add momentum via phase
    if momentum_x != 0 or momentum_y != 0:
        psi *= np.exp(1j * (momentum_x * dx + momentum_y * dy))
    
    return psi.astype(np.complex128)

def create_electron_with_angular_momentum(x, y, cx, cy, sigma, angular_momentum=0):
    """Create an electron with angular momentum (rotation).
    
    Args:
        x, y: coordinate grids  
        cx, cy: center position
        sigma: width of the wavepacket
        angular_momentum: integer angular momentum quantum number
    """
    dx = x - cx
    dy = y - cy
    r_squared = dx*dx + dy*dy
    theta = np.arctan2(dy, dx)
    
    # Gaussian radial part
    radial = np.exp(-r_squared / (2 * sigma**2))
    
    # Angular momentum creates rotation
    angular = np.exp(1j * angular_momentum * theta)
    
    return (radial * angular).astype(np.complex128)

def create_orbital_electron(x, y, cx, cy, radius_px, quantum_numbers, **kwargs):
    """Create an electron wavepacket with initial angular momentum and energy.
    
    The physics simulation will then evolve this according to the potential.
    """
    n, l, m = quantum_numbers
    
    # Start with a localized wavepacket at the appropriate scale
    # The radius scales roughly with n^2 for quantum states
    effective_radius = radius_px * (n**1.5)  # Larger for higher n
    
    # Create base wavepacket
    psi = create_electron_wavepacket(x, y, cx, cy, effective_radius)
    
    # Add angular momentum if requested
    if l > 0 and m != 0:
        dx = x - cx
        dy = y - cy
        theta = np.arctan2(dy, dx)
        psi *= np.exp(1j * m * theta)
    
    # Add some radial structure for higher n (nodes)
    if n > 1:
        dx = x - cx
        dy = y - cy
        r = np.sqrt(dx*dx + dy*dy)
        
        # Simple radial modulation to encourage node formation
        # This will evolve naturally under the physics
        radial_modulation = np.cos(np.pi * r / effective_radius * (n-1))
        psi *= (1 + 0.3 * radial_modulation)
    
    return psi.astype(np.complex128)

def create_atom_electron(x, y, cx, cy, radius_px, quantum_numbers, **kwargs):
    """Create atomic orbital wavefunctions for any element using quantum mechanical principles.
    
    This is a generic atomic orbital generator that works for:
    - Hydrogen and hydrogen-like ions (He⁺, Li²⁺, etc.)
    - Multi-electron atoms (C, N, O, etc.) using electron screening
    - Any quantum numbers (n, l, m) with proper 3D→2D projections
    
    Features:
    - Radial part: Associated Laguerre polynomials with proper normalization
    - Angular part: Real spherical harmonics projected to 2D plane  
    - Physics-based orbital scaling using simulation parameters
    - Automatic electron screening calculation using Slater's rules
    - Support for custom electron configurations
    
    Args:
        x, y: coordinate grids
        cx, cy: center position (nucleus location)
        radius_px: base orbital radius in pixels (auto-scaled by physics)
        quantum_numbers: (n, l, m) quantum numbers
        atomic_number: nuclear charge Z (default: 1 for hydrogen)
        alpha: screening parameter for Z_eff (default: auto-calculated)
        electron_config: list of (n,l,occupancy) for Slater screening (optional)
    
    Returns:
        Complex wavefunction representing the atomic orbital
        
    Examples:
        # Hydrogen 1s orbital
        psi = create_atom_electron(X, Y, cx, cy, 12, (1,0,0), atomic_number=1)
        
        # Carbon 2p orbital with screening
        psi = create_atom_electron(X, Y, cx, cy, 12, (2,1,0), atomic_number=6)
        
        # Custom screening for lithium
        psi = create_atom_electron(X, Y, cx, cy, 12, (2,0,0), 
                                  atomic_number=3, alpha=0.31)
    """
    n, l, m = quantum_numbers
    atomic_number = kwargs.get('atomic_number', 1)
    alpha = kwargs.get('alpha', None)
    electron_config = kwargs.get('electron_config', None)
    
    # Calculate screening parameter (alpha) for multi-electron atoms
    if alpha is None:
        if atomic_number == 1:
            # Pure hydrogen - no screening
            alpha = 1.0
        elif electron_config is not None:
            # Calculate screening from electron configuration using Slater's rules
            alpha = calculate_slater_screening(atomic_number, n, l, electron_config)
        else:
            # Use orbital-dependent default screening for common atoms
            screening_rules = {
                # Format: atomic_number: {(n,l): alpha_value}
                3: {(1,0): 0.31, (2,0): 0.31},  # Lithium
                4: {(1,0): 0.31, (2,0): 0.31},  # Beryllium  
                6: {(1,0): 0.31, (2,0): 0.64, (2,1): 0.64},  # Carbon
                7: {(1,0): 0.31, (2,0): 0.64, (2,1): 0.64},  # Nitrogen
                8: {(1,0): 0.31, (2,0): 0.64, (2,1): 0.64},  # Oxygen
                9: {(1,0): 0.31, (2,0): 0.64, (2,1): 0.64},  # Fluorine
                10: {(1,0): 0.31, (2,0): 0.64, (2,1): 0.64}, # Neon
            }
            
            if atomic_number in screening_rules and (n,l) in screening_rules[atomic_number]:
                alpha = screening_rules[atomic_number][(n,l)]
            else:
                # Fallback: orbital-dependent screening
                if l == 0:    # s orbitals
                    alpha = 0.35
                elif l == 1:  # p orbitals  
                    alpha = 0.75
                elif l == 2:  # d orbitals
                    alpha = 0.90
                else:
                    alpha = 0.80  # fallback for higher orbitals
    
    # Calculate effective nuclear charge
    z_eff = atomic_number * alpha
    
    # Calculate physics-based Bohr radius
    # The stable orbital size is determined by the balance of kinetic and potential energy
    # In our simulation units: a₀ ≈ 1/√(POTENTIAL_STRENGTH * Z_eff)
    bohr_radius = SCALE / np.sqrt(POTENTIAL_STRENGTH * z_eff)
    
    # Orbital size scales with n² for hydrogen-like atoms
    orbital_radius = bohr_radius * n**2
    
    # Create coordinates relative to center
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    
    # Create hydrogen-like radial wavefunction using general formula
    rho = 2 * z_eff * r / (n * orbital_radius)  # Proper dimensionless radius
    
    # General hydrogen-like radial function with Laguerre polynomials
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
    
    # Create angular part - proper 3D→2D projections like hydrogen_utils.py
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
    
    # Add gentle envelope like in hydrogen_utils.py for better visualization
    envelope = np.exp(-r**2/(2*(orbital_radius*0.6)**2))
    psi = psi * envelope
    
    # Ensure proper normalization 
    norm_factor = np.sqrt(np.sum(np.abs(psi)**2))
    if norm_factor > 0:
        psi = psi / norm_factor
    
    return psi.astype(np.complex128)

def calculate_slater_screening(atomic_number, n, l, electron_config):
    """Calculate Slater screening parameter using Slater's rules.
    
    Args:
        atomic_number: Nuclear charge Z
        n, l: Principal and angular quantum numbers of the orbital
        electron_config: List of (n, l, occupancy) tuples for filled orbitals
    
    Returns:
        alpha: Screening parameter (Z_eff = Z * alpha)
    """
    # Slater's rules for screening
    sigma = 0.0  # Total screening constant
    
    for config_n, config_l, occupancy in electron_config:
        if occupancy == 0:
            continue
            
        # Same shell (n,l)
        if config_n == n and config_l == l:
            sigma += 0.35 * (occupancy - 1)  # Other electrons in same subshell
            
        # Lower shells
        elif config_n < n:
            if n == 1:
                continue  # No screening for 1s
            elif n == 2:
                sigma += 0.85 * occupancy  # 1s screens 2s/2p by 0.85
            elif n >= 3:
                if config_n == n - 1:
                    sigma += 0.85 * occupancy  # (n-1) shell screens by 0.85
                else:
                    sigma += 1.0 * occupancy   # Inner shells screen completely
                    
        # Same n, lower l (for p, d, f orbitals)
        elif config_n == n and config_l < l:
            sigma += 0.35 * occupancy
    
    # Effective nuclear charge
    z_eff = atomic_number - sigma
    alpha = z_eff / atomic_number
    
    return max(alpha, 0.1)  # Prevent negative or zero screening

