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


# Carbon nucleus position (pixels)
nucleus1_x, nucleus1_y = center_x, center_y
orb_px = 12

print("Creating Carbon multi-electron system...")

# Create separate electrons that will evolve independently
# We'll focus on the 3 valence p-electrons for visual clarity

# 2p orbitals - 3 separate electrons with slight spatial offsets to avoid overlap
psi_2px = create_atom_electron(X, Y, nucleus1_x, nucleus1_y, 12, (2,1,1), atomic_number=6)   # 2p_x
psi_2py = create_atom_electron(X, Y, nucleus1_x, nucleus1_y, 12, (2,1,-1), atomic_number=6)  # 2p_y
psi_2pz = create_atom_electron(X, Y, nucleus1_x, nucleus1_y, 12, (2,1,0), atomic_number=6)   # 2p_z

# Add slight spatial offsets so electrons don't overlap completely
offset_distance = 5  # pixels
psi_2px_shifted = create_atom_electron(X, Y, nucleus1_x + offset_distance, nucleus1_y, 12, (2,1,1), atomic_number=6)
psi_2py_shifted = create_atom_electron(X, Y, nucleus1_x - offset_distance, nucleus1_y + offset_distance, 12, (2,1,-1), atomic_number=6)
psi_2pz_shifted = create_atom_electron(X, Y, nucleus1_x, nucleus1_y - offset_distance, 12, (2,1,0), atomic_number=6)

# Store all electrons
electrons = [psi_2px_shifted, psi_2py_shifted, psi_2pz_shifted]
electron_names = ["2p_x", "2p_y", "2p_z"]

# Apply frame limits
for i in range(len(electrons)):
    electrons[i] = limit_frame(electrons[i])

# Simulation
video_file = "carbon_multi_electron.avi"
video_writer = StreamingVideoWriter(
    output_file=video_file, 
    fps=24, 
    sample_frames=50,
    keep_first_batch=True,
    first_batch_size=100
)

# Report estimated memory usage
memory_info = video_writer.get_memory_usage_estimate((SIZE, SIZE))
print(f"Memory usage estimate:")
print(f"  - Frames in memory: {memory_info['frames_in_memory']}")
print(f"  - Memory per frame: {memory_info['memory_per_frame_mb']:.1f} MB")
print(f"  - Total memory for frames: {memory_info['total_memory_mb']:.1f} MB")

# Nuclear positions for each electron (slightly offset to prevent collapse)
positions = [
    np.array([nucleus1_x + offset_distance, nucleus1_y], float),
    np.array([nucleus1_x - offset_distance, nucleus1_y + offset_distance], float),
    np.array([nucleus1_x, nucleus1_y - offset_distance], float)
]

print("Starting multi-electron simulation...")

for step in range(TIME_STEPS):
    if step % 200 == 0:
        print(f"Step {step}/{TIME_STEPS}...")
        
        # Add small perturbations to break symmetry and create interesting dynamics
        if step % 400 == 0 and step > 0:
            # Add orbital mixing between electrons
            mix_strength = 0.1 * np.sin(step * 0.01)
            electrons[0] = electrons[0] + mix_strength * electrons[1]  # px mixes with py
            electrons[1] = electrons[1] + mix_strength * electrons[2]  # py mixes with pz
            electrons[2] = electrons[2] + mix_strength * electrons[0]  # pz mixes with px
            
            # Renormalize
            for i in range(3):
                norm = np.sqrt(np.sum(np.abs(electrons[i])**2))
                if norm > 0:
                    electrons[i] = electrons[i] / norm
    
    # Evolve each electron independently
    for i in range(3):
        # Compute density and forces for this electron
        density = np.abs(electrons[i])**2
        force = compute_force_from_density(density, positions[i])
        
        # Create potential (each electron feels the nucleus at its position)
        V = create_nucleus_potential(X, Y, *positions[i])
        
        # Add electron-electron repulsion (simplified)
        for j in range(3):
            if i != j:
                other_density = np.abs(electrons[j])**2
                # Simple repulsion: reduce potential where other electrons are
                V = V + 0.1 * other_density  # Weak electron-electron repulsion
        
        # Propagate this electron's wavefunction
        electrons[i] = propagate_wave_with_potential(electrons[i], V)
        electrons[i] = limit_frame(electrons[i])
    
    # Combine all electrons for visualization
    # Method 1: Simple superposition
    combined_psi = sum(electrons) / len(electrons)
    
    # Method 2: Weighted superposition with different phases for visual interest
    # phase_factors = [1.0, np.exp(1j * step * 0.01), np.exp(1j * step * 0.02)]
    # combined_psi = sum(electrons[i] * phase_factors[i] for i in range(3)) / 3
    
    region = combined_psi
    
    # Create frames for video
    frame_real = np.real(region)
    frame_imag = np.imag(region)
    frame_phase = np.angle(gaussian_filter(region, sigma=1))
    frame_prob = np.abs(region)**2
    frame_prob = frame_prob / np.max(frame_prob) if np.max(frame_prob) > 0 else frame_prob
    
    video_writer.add_frame(frame_real, frame_imag, frame_phase, frame_prob)

# Finalize video
video_writer.finalize()
open_video(video_file)
print("Multi-electron Carbon simulation complete!")
