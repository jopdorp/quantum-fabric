import numpy as np
import matplotlib.pyplot as plt
from math import pi
from matplotlib.widgets import Button
from plot_utils import create_and_show_plot
from video_utils import create_video, open_video

# Grid and simulation parameters
SIZE = 256
GRID_WIDTH = SIZE
GRID_HEIGHT = SIZE
TIME_STEPS = SIZE * 3
DT = 2
SIGMA_AMPLIFIER = 0.4
POTENTIAL_STRENTGH = 0.2  # Adjusted for stability
MAX_GATES_PER_CELL = 4

# --- Initial world state
X, Y = np.meshgrid(np.arange(GRID_WIDTH), np.arange(GRID_HEIGHT))
center_x, center_y = SIZE // 2, SIZE // 2

def create_orbital_electron(x, y, center_x, center_y, orbital_radius, quantum_numbers):
    n, l, m = quantum_numbers
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    theta = np.arctan2(y - center_y, x - center_x)
    bohr_radius = orbital_radius
    radial_part = np.exp(-r / (n * bohr_radius)) * (r / bohr_radius)**l
    angular_part = np.exp(1j * m * theta)
    # Calculate the width of the Gaussian envelope
    # The width is related to the principal quantum number n and Bohr radius
    # The factor 0.8 is an empirical adjustment to control the spread of the wavefunction
    n = max(n, 1)  # Ensure n is at least 1
    sigma = n * bohr_radius * SIGMA_AMPLIFIER

    envelope = np.exp(-((r - n * bohr_radius)**2) / (2 * sigma**2))
    psi = radial_part * angular_part * envelope
    return psi.astype(np.complex128)

def create_nucleus_potential(x, y, nucleus_x, nucleus_y, charge=1):
    r = np.sqrt((x - nucleus_x)**2 + (y - nucleus_y)**2)
    r = np.maximum(r, 1.0)
    # Coulomb potential: V = -k*Z/r (attractive for electrons)
    # Adjust potential strength for stability
    # This value can be tuned to control the strength of the potential
    potential_strength = POTENTIAL_STRENTGH
    return -potential_strength * charge / r

def add_noise(psi, noise_level=0.001):
    noise = (np.random.rand(*psi.shape) - 0.5) * noise_level
    return psi + noise.astype(np.complex128)

def normalize_wavefunction(psi):
    norm = np.sqrt(np.sum(np.abs(psi)**2))
    if norm > 0:
        return psi / norm
    return psi

def propagate_wave_with_potential(psi, potential, dt=DT):
    potential_phase = np.exp(-1j * dt * potential / 2)
    psi = psi * potential_phase
    psi_hat = np.fft.fft2(psi)
    kx = np.fft.fftfreq(psi.shape[1]) * 2 * np.pi
    ky = np.fft.fftfreq(psi.shape[0]) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    kinetic_phase = np.exp(-1j * dt * (KX**2 + KY**2) / 2)
    psi_hat *= kinetic_phase
    psi = np.fft.ifft2(psi_hat)
    psi = psi * potential_phase
    return psi

def apply_spatial_gates_from_patch(psi, gate_patch_map):
    updated = psi.copy()
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if abs(psi[y, x]) > 1e-3:
                for i in range(MAX_GATES_PER_CELL):
                    gate_fn = gate_patch_map[y, x, i]
                    if gate_fn is not None:
                        updated[y, x] = gate_fn(updated[y, x])
    return updated

def center_wave(psi):
    global smooth_cy, smooth_cx, smoothing_factor
    prob = np.abs(psi)**2
    total = np.sum(prob)
    if total == 0:
        return psi
    y_idx, x_idx = np.indices(prob.shape)
    cy = np.sum(y_idx * prob) / total
    cx = np.sum(x_idx * prob) / total
    smooth_cy = smooth_cy + (cy - smooth_cy) / smoothing_factor
    smooth_cx = smooth_cx + (cx - smooth_cx) / smoothing_factor
    shift_y = int(np.round((SIZE // 2) - smooth_cy))
    shift_x = int(np.round((SIZE // 2) - smooth_cx))
    return np.roll(np.roll(psi, shift_y, axis=0), shift_x, axis=1)

# Adjust reverse Gaussian blur to increase the blur effect
# Ensure 50% of the image is entirely blurred out

def reverse_gaussian_blur(psi, center_x, center_y):
    """Apply an optimized reverse Gaussian blur with stronger blur for 50% of the image."""
    r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_sigma = SIZE // 2  # Maximum blur for 50% of the image
    min_sigma = 1          # Minimum blur at the center
    sigma_map = min_sigma + (max_sigma - min_sigma) * (r / np.max(r))

    # Precompute a spatially varying kernel
    from scipy.ndimage import gaussian_filter
    blurred_psi = gaussian_filter(psi, sigma=max_sigma)

    # Blend the original and blurred wavefunction based on sigma_map
    blend_factor = (sigma_map - min_sigma) / (max_sigma - min_sigma)
    optimized_psi = psi * (1 - blend_factor) + blurred_psi * blend_factor

    return optimized_psi

# Function to apply a low-pass filter
def apply_low_pass_filter(psi, cutoff_frequency):
    """Apply a low-pass filter to the wavefunction to remove high-frequency components."""
    psi_hat = np.fft.fft2(psi)
    kx = np.fft.fftfreq(psi.shape[1]) * 2 * np.pi
    ky = np.fft.fftfreq(psi.shape[0]) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    k_squared = KX**2 + KY**2

    # Create a low-pass filter mask
    low_pass_mask = k_squared <= (cutoff_frequency**2)
    psi_hat_filtered = psi_hat * low_pass_mask

    # Transform back to spatial domain
    psi_filtered = np.fft.ifft2(psi_hat_filtered)
    return psi_filtered

# Initial conditions
nucleus_x, nucleus_y = center_x, center_y
nuclear_potential = create_nucleus_potential(X, Y, nucleus_x, nucleus_y, charge=1)
center_x_offset = int(center_x * 1.1)
center_y_offset = int(center_y * 1.1)
electron1 = create_orbital_electron(X, Y, center_x_offset, center_y_offset, orbital_radius=25, quantum_numbers=(1, 0, 0))
psi_t = add_noise(electron1, noise_level=0.001)
psi_t = normalize_wavefunction(psi_t)
momentum_x, momentum_y = 0.03, 0.01
momentum_phase = np.exp(1j * (momentum_x * X + momentum_y * Y))
psi_t *= momentum_phase

# Apply the updated reverse Gaussian blur to the initial wavefunction
psi_t = reverse_gaussian_blur(psi_t, center_x, center_y)

# Apply a low-pass filter to the initial wavefunction
cutoff_frequency = 0.1  # Adjust cutoff frequency as needed
psi_t = apply_low_pass_filter(psi_t, cutoff_frequency)

# Quantum gate infrastructure (optional but present)
def hadamard(amp): return amp * (1 + 1j) / np.sqrt(2)
def pauli_x(amp): return -amp
def t_gate(amp): return amp * np.exp(1j * pi / 4)
gate_patch_map = np.full((GRID_HEIGHT, GRID_WIDTH, MAX_GATES_PER_CELL), None, dtype=object)

def add_gate_patch(gate_patch_map, center_y, center_x, gate_fn):
    radius = 2
    for y in range(center_y - radius, center_y + radius + 1):
        for x in range(center_x - radius, center_x + radius + 1):
            if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH:
                if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                    for i in range(MAX_GATES_PER_CELL):
                        if gate_patch_map[y, x, i] is None:
                            gate_patch_map[y, x, i] = gate_fn
                            break

# Centering state
smooth_cy = SIZE // 2
smooth_cx = SIZE // 2
smoothing_factor = 2000

# Define stable states for hydrogen (quantum numbers and orbital radii)
stable_states = [
    (1, 0, 0, 25),  # Ground state
    (2, 1, 1, 50),  # Excited state 1
    (3, 2, 2, 75),  # Excited state 2
    (4, 3, 3, 100)  # Excited state 3
]

# Function to interpolate between two wavefunctions
def tween_wavefunctions(psi_start, psi_end, alpha):
    return (1 - alpha) * psi_start + alpha * psi_end

# Initialize variables for tweening
current_state_index = 0
next_state_index = 1
tween_duration = 30  # Number of time steps for a 1-second tween (assuming 30 FPS)
tween_step = 0

# Simulation loop
frames_real, frames_imag, frames_phase, frames_prob = [], [], [], []
cur = psi_t.copy()
print("Simulating hydrogen atom with single electron...")
print(f"Nuclear position: ({nucleus_x}, {nucleus_y})")
print(f"Time steps: {TIME_STEPS}, dt: {DT}")

for step in range(TIME_STEPS):
    if step % 50 == 0:
        print(f"Step {step}/{TIME_STEPS}")

    # Tweening logic
    if tween_step < tween_duration:
        # Get the current and next stable states
        current_state = stable_states[current_state_index]
        next_state = stable_states[next_state_index]

        # Generate wavefunctions for the current and next states
        psi_current = create_orbital_electron(X, Y, center_x, center_y, current_state[3], current_state[:3])
        psi_next = create_orbital_electron(X, Y, center_x, center_y, next_state[3], next_state[:3])

        # Interpolate between the two states
        alpha = tween_step / tween_duration
        cur = tween_wavefunctions(psi_current, psi_next, alpha)

        # Normalize the wavefunction
        cur = normalize_wavefunction(cur)

        tween_step += 1
    else:
        # Move to the next stable state
        current_state_index = next_state_index
        next_state_index = (next_state_index + 1) % len(stable_states)
        tween_step = 0

    # Propagate the wavefunction
    cur = propagate_wave_with_potential(cur, nuclear_potential)
    cur = center_wave(cur)
    cur = apply_spatial_gates_from_patch(cur, gate_patch_map)

    # Record frames for visualization
    region = cur[SIZE//4:3*SIZE//4, SIZE//4:3*SIZE//4]
    frames_real.append(np.real(region))
    frames_imag.append(np.imag(region))
    frames_phase.append(np.angle(region))

    prob_density = np.abs(region)**2
    if np.max(prob_density) > 0:
        prob_density = prob_density / np.max(prob_density)
    frames_prob.append(prob_density)

# Video output
video_file = "hydrogen_atom_simulation.mkv"
print(f"Creating video: {video_file}")
create_video(
    TIME_STEPS,
    frames_real,
    frames_imag,
    frames_phase,
    frames_prob,
    fps=30,
    output_file=video_file
)

print("Opening video...")
open_video(video_file)
