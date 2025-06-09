import numpy as np

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
    """Create an electron for atomic simulation.
    
    Creates an initial state that will evolve under the nuclear potential.
    The quantum numbers guide the initial conditions but don't predetermine the final shape.
    """
    n, l, m = quantum_numbers
    atomic_number = kwargs.get('atomic_number', 1)
    
    # Scale size based on atomic number (higher Z = smaller orbitals)
    z_eff = atomic_number * 0.8  # Rough effective nuclear charge
    scaled_radius = radius_px / z_eff**0.5
    
    # Create the initial wavepacket
    psi = create_orbital_electron(x, y, cx, cy, scaled_radius, quantum_numbers)
    
    # Add some initial energy/momentum if the electron is in an excited state
    if n > 1:
        # Add small random momentum components to break symmetry
        # This encourages more realistic orbital evolution
        momentum_scale = 0.1 / n  # Less momentum for higher n
        momentum_x = np.random.normal(0, momentum_scale)
        momentum_y = np.random.normal(0, momentum_scale)
        
        dx = x - cx
        dy = y - cy
        psi *= np.exp(1j * (momentum_x * dx + momentum_y * dy))
    
    return psi.astype(np.complex128)

