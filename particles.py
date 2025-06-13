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
