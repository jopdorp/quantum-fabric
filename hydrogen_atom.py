import numpy as np
from video_utils import StreamingVideoWriter, open_video
from scipy.ndimage import gaussian_filter
from config import TIME_STEPS, X, Y, center_x, center_y, SIZE
from particles import create_orbital_electron, create_atom_electron
from frame_utils import limit_frame
from physics import (
    create_nucleus_potential,
    compute_force_from_density,
    propagate_wave_with_potential
)



# Hydrogen nuclei positions (pixels)
nucleus1_x, nucleus1_y = center_x, center_y

# Initialize electrons - test different atoms and orbitals
orb_px = 30

# Test cases:
# 1. Hydrogen 2p orbital (should show dumbbell)
psi1 = create_atom_electron(X, Y, nucleus1_x, nucleus1_y, orb_px, (2,1,1), atomic_number=1)

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
video_file = "hydrogen_atom_sim.mkv"
# Configure streaming writer: keep first 100 frames in memory for potential overwrites
# This allows streaming of the remaining 400 frames while keeping a small batch in memory
video_writer = StreamingVideoWriter(
    output_file=video_file, 
    fps=12, 
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
    if step % 100 == 0:
        print(f"Step {step}/{TIME_STEPS}…")
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
