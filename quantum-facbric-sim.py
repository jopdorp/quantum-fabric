import numpy as np
import matplotlib.pyplot as plt
from math import pi, factorial # Import factorial directly
from matplotlib.widgets import Button
from plot_utils import create_and_show_plot
from video_utils import create_video, open_video
from scipy.special import genlaguerre
from scipy.ndimage import gaussian_filter

# Grid and simulation parameters
SIZE = 256
GRID_WIDTH = SIZE
GRID_HEIGHT = SIZE
TIME_STEPS = SIZE * 6
DT = 0.5  # Reduced DT for richer time evolution
SIGMA_AMPLIFIER = 0.4
POTENTIAL_STRENGTH = 1.0  # Increased potential strength and corrected typo from POTENTIAL_STRENTGH
MAX_GATES_PER_CELL = 4

# Add global variables for configuration
SCALE = 9000.0 # Previously 400.0
# INITIAL_MOMENTUM_X = 3000.3 # Replaced by KX
# INITIAL_MOMENTUM_Y = 1000    # Replaced by KY
KX = 2 * np.pi / SIZE * 1  # Wavevector component for x-direction (reduced from 5 to 1)
KY = 2 * np.pi / SIZE * 1  # Wavevector component for y-direction (reduced from 2 to 1)

# --- Initial world state
X, Y = np.meshgrid(np.arange(GRID_WIDTH), np.arange(GRID_HEIGHT))
center_x, center_y = SIZE // 2, SIZE // 2

def hydrogen_eigenstate_2d(n, m, x, y, center_x, center_y, scale=400.0):
    """2D hydrogen-like eigenstate using Laguerre polynomials (approximation)."""
    assert n > abs(m) >= 0
    x_scaled = (x - center_x) / scale
    y_scaled = (y - center_y) / scale
    r = np.sqrt(x_scaled**2 + y_scaled**2)
    theta = np.arctan2(y_scaled, x_scaled)

    # Define radial quantum number
    p = n - abs(m) - 1
    rho = 2 * r / n

    # Normalization constant (approximate for 2D hydrogen-like system)
    norm = np.sqrt((2 / n)**2 * factorial(p) / (pi * factorial(p + abs(m))))

    # Radial part with generalized Laguerre polynomial
    R = rho**abs(m) * np.exp(-rho / 2) * genlaguerre(p, 2 * abs(m))(rho)

    # Angular part
    angular = np.exp(1j * m * theta)

    return (norm * R * angular).astype(np.complex128)

def create_orbital_electron(x, y, center_x, center_y, orbital_radius, quantum_numbers, scale=None):
    if scale is None:
        scale = SCALE # Use current global SCALE if no specific scale is provided
    n, l, m = quantum_numbers
    # Scale the spatial coordinates
    x_scaled = x / scale
    y_scaled = y / scale
    center_x_scaled = center_x / scale
    center_y_scaled = center_y / scale

    r = np.sqrt((x_scaled - center_x_scaled)**2 + (y_scaled - center_y_scaled)**2)
    theta = np.arctan2(y_scaled - center_y_scaled, x_scaled - center_x_scaled)
    bohr_radius = orbital_radius
    radial_part = np.exp(-r / (n * bohr_radius)) * (r / bohr_radius)**l
    angular_part = np.exp(1j * m * theta)
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
    potential_strength = POTENTIAL_STRENGTH
    return -potential_strength * charge / r

def add_noise(psi, noise_level=0.001):
    noise = (np.random.rand(*psi.shape) - 0.5) * noise_level
    return psi + noise.astype(np.complex128)

def normalize_wavefunction(psi):
    norm = np.sqrt(np.sum(np.abs(psi)**2))
    if norm > 0:
        return psi / norm
    return psi

def add_mean_field_coulomb_repulsion(source_psi, strength=0.05, sigma=5):
    """Create a repulsive potential based on the source electron density."""
    charge_density = np.abs(source_psi)**2
    repulsion = gaussian_filter(charge_density, sigma=sigma)
    return strength * repulsion

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

# Function to apply a low-pass filter to the absorption edge
def apply_absorption_edge_low_pass(psi, cutoff_frequency, blur_factor=SIZE // 2):
    """Apply a low-pass filter specifically to the absorption edge."""
    # Compute the edge mask
    r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    edge_mask = np.exp(-r**2 / (2 * blur_factor**2))

    # Apply the low-pass filter
    psi_hat = np.fft.fft2(psi * edge_mask)
    kx = np.fft.fftfreq(psi.shape[1]) * 2 * np.pi
    ky = np.fft.fftfreq(psi.shape[0]) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    k_squared = KX**2 + KY**2

    # Create a low-pass filter mask
    low_pass_mask = k_squared <= (cutoff_frequency**2)
    psi_hat_filtered = psi_hat * low_pass_mask

    # Transform back to spatial domain
    psi_filtered = np.fft.ifft2(psi_hat_filtered)
    return psi_filtered.real

# Function to apply a large blur to the edges of the world
def apply_edge_blur(psi, blur_factor=SIZE // 2):
    """Apply a large Gaussian blur to the edges of the world."""
    r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    edge_blur = np.exp(-r**2 / (2 * blur_factor**2))
    return psi * edge_blur

# Initial conditions
# Define two hydrogen nuclei
nucleus1_x, nucleus1_y = center_x - SIZE // 6, center_y
nucleus2_x, nucleus2_y = center_x + SIZE // 6, center_y

# Two nuclear potentials
nuclear_potential1 = create_nucleus_potential(X, Y, nucleus1_x, nucleus1_y, charge=1)
nuclear_potential2 = create_nucleus_potential(X, Y, nucleus2_x, nucleus2_y, charge=1)

# Two electrons with opposite initial momentum
psi1 = hydrogen_eigenstate_2d(1, 0, X, Y, nucleus1_x, nucleus1_y, scale=SCALE)
psi2 = hydrogen_eigenstate_2d(1, 0, X, Y, nucleus2_x, nucleus2_y, scale=SCALE)

# Apply asymmetric momentum to reduce perfect phase cancellation
momentum1 = np.exp(1j * (KX * X + KY * Y))
momentum2 = np.exp(-1j * (0.8 * KX * X - 0.6 * KY * Y))
psi1 *= momentum1
psi2 *= momentum2

psi1 = normalize_wavefunction(psi1)
psi2 = normalize_wavefunction(psi2)

# Apply the updated reverse Gaussian blur to the initial wavefunction
psi1 = reverse_gaussian_blur(psi1, center_x, center_y)
psi2 = reverse_gaussian_blur(psi2, center_x, center_y)

# Apply a low-pass filter to the initial wavefunction
cutoff_frequency = 0.1  # Adjust cutoff frequency as needed
psi1 = apply_low_pass_filter(psi1, cutoff_frequency)
psi2 = apply_low_pass_filter(psi2, cutoff_frequency)

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

# Define stable states for hydrogen (quantum numbers: n, m)
stable_states = [
    (1, 0),   # Ground state
    (2, 1),   # Excited state 1
    (3, 2),   # Excited state 2
    (4, 3)    # Excited state 3
]

# Function to interpolate between two wavefunctions
def tween_wavefunctions(psi_start, psi_end, alpha):
    return (1 - alpha) * psi_start + alpha * psi_end

# Timing parameters
stay_duration = 60  # Number of time steps to stay in a state (2 seconds at 30 FPS)
tween_duration = 15  # Number of time steps for a 0.5-second tween (at 30 FPS)
tween_step = 0
stay_step = 0

# Initialize variables for tweening
current_state_index = 0
next_state_index = 1

# Simulation loop
frames_real, frames_imag, frames_phase, frames_prob = [], [], [], []
cur = psi1.copy()
print("Simulating hydrogen atom with single electron...")
print(f"Time steps: {TIME_STEPS}, dt: {DT}")

for step in range(TIME_STEPS):
    if step % 50 == 0:
        print(f"Step {step}/{TIME_STEPS}")

    if stay_step < stay_duration:
        # Stay in the current state
        if stay_step == 0:  # Only create the wavefunction once per state
            current_state = stable_states[current_state_index]
            cur = hydrogen_eigenstate_2d(current_state[0], current_state[1], X, Y, center_x, center_y, scale=SCALE)
            cur = normalize_wavefunction(cur)
        stay_step += 1
    elif tween_step < tween_duration:
        # Tweening logic
        current_state = stable_states[current_state_index]
        next_state = stable_states[next_state_index]

        # Generate wavefunctions for the current and next states
        psi_current = hydrogen_eigenstate_2d(current_state[0], current_state[1], X, Y, center_x, center_y, scale=SCALE)
        psi_next = hydrogen_eigenstate_2d(next_state[0], next_state[1], X, Y, center_x, center_y, scale=SCALE)

        # Interpolate between the two states
        alpha = tween_step / tween_duration
        cur = tween_wavefunctions(psi_current, psi_next, alpha)

        # Add momentum during tweening to create imaginary components
        # Using KX and KY for tweening momentum, scaled by alpha
        momentum_x = KX * 0.1 * alpha  # Using global KX
        momentum_y = KY * 0.1 * alpha  # Using global KY
        # X and Y are the global meshgrid variables
        momentum_phase = np.exp(1j * (momentum_x * X + momentum_y * Y))
        cur *= momentum_phase

        # Normalize the wavefunction
        cur = normalize_wavefunction(cur)

        tween_step += 1
    else:
        # Move to the next stable state
        current_state_index = next_state_index
        next_state_index = (next_state_index + 1) % len(stable_states)
        stay_step = 0
        tween_step = 0

    # Add mean-field repulsion dynamically
    # Combined nuclear potential for both electrons - each electron is attracted to both nuclei
    combined_nuclear_potential = nuclear_potential1 + nuclear_potential2
    
    # Each electron experiences both nuclei attraction and the other electron's repulsion
    V1 = combined_nuclear_potential + add_mean_field_coulomb_repulsion(psi2)
    V2 = combined_nuclear_potential + add_mean_field_coulomb_repulsion(psi1)

    psi1 = propagate_wave_with_potential(psi1, V1)
    psi2 = propagate_wave_with_potential(psi2, V2)
    
    # The following lines involving 'cur' will be updated or removed by subsequent edits
    # cur = center_wave(cur)
    # cur = apply_spatial_gates_from_patch(cur, gate_patch_map)
    # cur = apply_edge_blur(cur)
    # cur = apply_absorption_edge_low_pass(cur, cutoff_frequency=0.1)

    # Record frames for visualization
    # region = cur  # Visualize the entire grid # This line will be replaced
    region = psi1 + psi2 # Display both electrons' combined wavefunction
    frames_real.append(np.real(region))
    frames_imag.append(np.imag(region))

    # Record phase with smooth blurring to prevent sharp transitions
    blurred_region = gaussian_filter(region, sigma=1)
    frames_phase.append(np.angle(blurred_region))

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
