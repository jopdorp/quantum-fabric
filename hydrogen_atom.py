import numpy as np
from video_utils import StreamingVideoWriter, open_video
from scipy.ndimage import gaussian_filter
from scipy.special import factorial, genlaguerre, sph_harm
from config import TIME_STEPS, X, Y, center_x, center_y, SIZE, SCALE, SIGMA_AMPLIFIER
from particles import create_orbital_electron, create_atom_electron
from frame_utils import limit_frame
from physics import (
    create_nucleus_potential,
    compute_force_from_density,
    propagate_wave_with_potential
)


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

# Hydrogen nuclei positions (pixels)
nucleus1_x, nucleus1_y = center_x, center_y

# Initialize electrons - test different atoms and orbitals
orb_px = 20

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
    (4, 3, -3), # 4f_xyz
    (4, 3, -2), # 4f_xz²
    (4, 3, -1), # 4f_yz²
    (4, 3, 0),  # 4f_x²-y²
    (4, 3, 1),  # 4f_x³
    (4, 3, 2),  # 4f_y³
    (4, 3, 3)   # 4f_z³
]
psi1 = create_hydrogen_orbital(eigen_states[0], X, Y, nucleus1_x, nucleus1_y, scale=orb_px)   # p_x + i*p_y

# Uncomment to test other atoms:
# 2. Carbon 2p orbital (more contracted due to higher Z_eff)
# psi1 = create_atom_electron(X, Y, nucleus1_x, nucleus1_y, orb_px, (2,1,0), 
#                           atomic_number=6, electron_config=[(1,0,2), (2,0,2), (2,1,1)])

# 3. Oxygen 2p orbital (even more contracted)
# psi1 = create_atom_electron(X, Y, nucleus1_x, nucleus1_y, orb_px, (2,1,0),
#                           atomic_number=8, electron_config=[(1,0,2), (2,0,2), (2,1,3)])

psi1 = limit_frame(psi1)
# add momentum to the wavefunction via phase
# MOMENTUM_X = 0.5  # momentum in x direction
# MOMENTUM_Y = 0.8  # momentum in y direction
# psi1 = psi1 * np.exp(1j * (X * MOMENTUM_X * np.pi / orb_px + Y * MOMENTUM_Y * np.pi / orb_px))


# Simulation
video_file = "hydrogen_atom_sim.avi"
# Configure streaming writer: keep first 100 frames in memory for potential overwrites
# This allows streaming of the remaining 400 frames while keeping a small batch in memory
video_writer = StreamingVideoWriter(
    output_file=video_file, 
    fps=24, 
    sample_frames=50,  # Sample fewer frames to start writing sooner
    keep_first_batch=True,
    first_batch_size=100  # Keep only first 100 frames in memory
)

# Report estimated memory usage
memory_info = video_writer.get_memory_usage_estimate((SIZE, SIZE))
print(f"Memory usage estimate:")
print(f"  - Frames in memory: {memory_info['frames_in_memory']}")
print(f"  - Memory per frame: {memory_info['memory_per_frame_mb']:.1f} MB")
print(f"  - Total memory for frames: {memory_info['total_memory_mb']:.1f} MB")
print(f"  - Sample memory: {memory_info['sample_memory_mb']:.1f} MB")

pos1, v1 = np.array([nucleus1_x,nucleus1_y],float), np.zeros(2)
print("Starting sim…")

eigen_state_index = 0
for step in range(TIME_STEPS):
    if step % 400 == 0:
        print(f"Step {step}/{TIME_STEPS}…")
        eigen_state_index += 1
        if eigen_state_index >= len(eigen_states):
            break
        psi1 = create_hydrogen_orbital(eigen_states[eigen_state_index], X, Y, nucleus1_x, nucleus1_y, scale=int(orb_px * 1 / (1 + eigen_state_index / 10)))
    d1 = np.abs(psi1)**2
    f1 = compute_force_from_density(d1, pos1)
    V1 = create_nucleus_potential(X,Y,*pos1)
    psi1 = propagate_wave_with_potential(psi1, V1)
    psi1 = limit_frame(psi1)
    region = psi1
    
    # Stream frames directly to video instead of storing in memory
    frame_real = np.real(region)
    frame_imag = np.imag(region)
    frame_phase = np.angle(gaussian_filter(region,sigma=1))
    frame_prob = np.abs(region)**2
    frame_prob = frame_prob / np.max(frame_prob) if np.max(frame_prob) > 0 else frame_prob
    
    video_writer.add_frame(frame_real, frame_imag, frame_phase, frame_prob)

# Finalize video
video_writer.finalize()
open_video(video_file)
