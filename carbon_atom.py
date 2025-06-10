import numpy as np
import numpy as np
from video_utils import StreamingVideoWriter, open_video
from scipy.ndimage import gaussian_filter
from config import TIME_STEPS, X, Y, center_x, center_y, SIZE, SCALE, SIGMA_AMPLIFIER
from frame_utils import limit_frame
from physics import (
    create_nucleus_potential,
    compute_force_from_density,
    propagate_wave_with_potential
)
from particles import create_atom_electron


# Hydrogen nucleus position (pixels)
nucleus1_x, nucleus1_y = center_x, center_y

# Initialize electrons - test different atoms and orbitals
orb_px = 12  # Match the scale used by hydrogen_eigenstate_3d_projected

# Comment out hydrogen-specific initialization
# eigen_states = eigen_states[::-1]
# psi1 = create_hydrogen_orbital(eigen_states[0], X, Y, nucleus1_x, nucleus1_y, scale=orb_px)   # p_x + i*p_y

# Test 4 different hydrogen eigenstates using generic atom electron creation
# All using atomic_number=1 (hydrogen) with different quantum numbers

# Define 4 different hydrogen eigenstates to test
# hydrogen_eigenstates = [
#     (1, 0, 0),  # 1s orbital
#     (2, 0, 0),  # 2s orbital  
#     (2, 1, 0),  # 2p_z orbital
#     (3, 2, 1),  # 3d orbital
# ]

# Start with the first eigenstate (1s)
# current_eigenstate_index = 0
# current_quantum_numbers = hydrogen_eigenstates[current_eigenstate_index]
# print(f"Starting with eigenstate: n={current_quantum_numbers[0]}, l={current_quantum_numbers[1]}, m={current_quantum_numbers[2]}")

# Create initial wavefunction using generic atom electron creation
# For hydrogen, use alpha=1.0 since there's no electron screening
# psi1 = create_atom_electron(X, Y, nucleus1_x, nucleus1_y, orb_px, 
#                            current_quantum_numbers, atomic_number=1, alpha=1.0)

# Create all 6 electrons for Carbon atom (1s² 2s² 2p²)
# Carbon electron configuration: 1s² 2s² 2p²

# 1s orbital - 2 electrons (spin up and down)
psi_1s_up = create_atom_electron(X, Y, nucleus1_x, nucleus1_y, 12, (1,0,0), atomic_number=6)
psi_1s_down = create_atom_electron(X, Y, nucleus1_x, nucleus1_y, 12, (1,0,0), atomic_number=6)

# 2s orbital - 2 electrons (spin up and down) 
psi_2s_up = create_atom_electron(X, Y, nucleus1_x, nucleus1_y, 12, (2,0,0), atomic_number=6)
psi_2s_down = create_atom_electron(X, Y, nucleus1_x, nucleus1_y, 12, (2,0,0), atomic_number=6)

# 2p orbitals - 2 electrons (following Hund's rule: one in each orbital first)
psi_2px = create_atom_electron(X, Y, nucleus1_x, nucleus1_y, 12, (2,1,1), atomic_number=6)   # 2p_x
psi_2py = create_atom_electron(X, Y, nucleus1_x, nucleus1_y, 12, (2,1,-1), atomic_number=6)  # 2p_y

# Combine all electrons into a single wavefunction
# In reality, this is an approximation - true multi-electron wavefunctions are much more complex
electrons = [psi_1s_up, psi_1s_down, psi_2s_up, psi_2s_down, psi_2px, psi_2py]

# For simulation, we'll use the superposition of all electrons
# (This is a simplified approach - real multi-electron atoms require Hartree-Fock or DFT)
psi1 = sum(electrons) / len(electrons)  # Average all electron orbitals

# Alternative: Focus on just the valence electrons (2p orbitals) for visualization
# psi1 = (psi_2px + psi_2py) / 2

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

for step in range(TIME_STEPS):
    if step % 400 == 0:
        print(f"Step {step}/{TIME_STEPS}…")
        
        # Cycle through the 4 different hydrogen eigenstates
        # current_eigenstate_index = (step // 400) % len(hydrogen_eigenstates)
        # current_quantum_numbers = hydrogen_eigenstates[current_eigenstate_index]
        
        # print(f"Switching to eigenstate: n={current_quantum_numbers[0]}, l={current_quantum_numbers[1]}, m={current_quantum_numbers[2]}")
        
        # # Create new wavefunction using generic atom electron creation
        # # For hydrogen, use alpha=1.0 since there's no electron screening
        # psi1 = create_atom_electron(X, Y, nucleus1_x, nucleus1_y, 
        #                            int(orb_px * 1 / (1 + current_eigenstate_index / 10)), 
        #                            current_quantum_numbers, atomic_number=1, alpha=1.0)
        
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
