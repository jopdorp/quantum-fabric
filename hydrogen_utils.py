from scipy.special import factorial, genlaguerre, sph_harm
import numpy as np

# Test cases:
# 1. Hydrogen 2p orbital (should show dumbbell)
# psi1 = create_atom_electron(X, Y, nucleus1_x, nucleus1_y, orb_px, (2,1,1), atomic_number=1)
eigen_states = [
    (1, 0, 0),  # 1s
    (2, 0, 0),  # 2s
    (2, 1, -1), # 2p_x
    (2, 1, 0),  # 2p_y
    (2, 1, 1),  # 2p_z
    (3, 2, -2), # 3d_xy
    (3, 2, -1), # 3d_xz
    (3, 2, 0),  # 3d_yz
    (3, 2, 1),  # 3d_x²-y²
    (3, 2, 2),   # 3d_z²
]

def hydrogen_eigenstate_3d_projected(n, l, m, x, y, cx, cy, scale=12, z_scale=0.3):
    """
    3D hydrogen eigenstate projected onto 2D plane with proper orbital orientations.
    """
    # Convert to atomic units
    xs, ys = (x - cx)/scale, (y - cy)/scale
    r = np.sqrt(xs**2 + ys**2)
    theta = np.arctan2(ys, xs)  # angle in xy-plane
    
    # Radial part: R_nl(r) - use 2D radius for better visualization
    rho = 2 * r / n
    
    try:
        norm_factor = np.sqrt((2/n)**3 * factorial(n-l-1) / (2*n*factorial(n+l)))
    except (ValueError, OverflowError):
        norm_factor = 1.0
    
    # Associated Laguerre polynomial
    laguerre_part = genlaguerre(n-l-1, 2*l+1)(rho)
    
    # Complete radial wavefunction
    R_nl = norm_factor * np.exp(-rho/2) * (rho**l) * laguerre_part
    
    # Angular part - create proper 2D projections of 3D orbitals
    if l == 0:  # s orbitals
        angular_part = 1.0 + 0j
        
    elif l == 1:  # p orbitals - proper projections
        if m == -1:
            # p_x - i*p_y → creates lobe at 45° angle
            angular_part = np.cos(theta - np.pi/4) * np.exp(1j * theta)
        elif m == 0:
            # p_z → when projected to xy-plane, creates dipole pattern
            angular_part = np.cos(theta)  # Creates two-lobe pattern along x-axis
        elif m == 1:
            # p_x + i*p_y → creates lobe at -45° angle  
            angular_part = np.cos(theta + np.pi/4) * np.exp(-1j * theta)
            
    elif l == 2:  # d orbitals
        if m == -2:
            angular_part = np.sin(2*theta) * np.exp(2j * theta)  # d_xy
        elif m == -1:
            angular_part = np.sin(theta) * np.cos(theta) * np.exp(1j * theta)  # d_xz/d_yz
        elif m == 0:
            angular_part = (3*np.cos(theta)**2 - 1)  # d_z² → cloverleaf in 2D
        elif m == 1:
            angular_part = np.sin(theta) * np.cos(theta) * np.exp(-1j * theta)  # d_xz/d_yz
        elif m == 2:
            angular_part = np.cos(2*theta) * np.exp(-2j * theta)  # d_x²-y²
    else:
        # Fallback to spherical harmonics for higher l
        phi = theta
        # For 2D projection, use theta=π/2 (xy-plane)
        Y_lm = sph_harm(m, l, phi, np.pi/2)
        angular_part = Y_lm
    
    # Complete wavefunction
    psi = R_nl * angular_part
    
    # Gentle envelope
    envelope = np.exp(-r**2/(2*(scale*0.6)**2))
    
    return (psi * envelope).astype(np.complex128)

def create_hydrogen_orbital(quantum_numbers, x, y, cx, cy, scale=12):
    """
    Create hydrogen orbital with proper quantum numbers.
    
    Examples:
    - (1,0,0): 1s orbital (spherical)
    - (2,0,0): 2s orbital (spherical with node)
    - (2,1,-1), (2,1,0), (2,1,1): 2p orbitals (dumbbell shapes)
    - (3,2,-2), (3,2,-1), (3,2,0), (3,2,1), (3,2,2): 3d orbitals
    """
    n, l, m = quantum_numbers
    if l >= n:
        raise ValueError(f"l ({l}) must be less than n ({n})")
    if abs(m) > l:
        raise ValueError(f"|m| ({abs(m)}) must be <= l ({l})")
    
    return hydrogen_eigenstate_3d_projected(n, l, m, x, y, cx, cy, scale)