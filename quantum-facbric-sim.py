import numpy as np
import matplotlib.pyplot as plt
from math import pi, factorial  # Import factorial directly
from matplotlib.widgets import Button
from plot_utils import create_and_show_plot
from video_utils import create_video, open_video
from scipy.special import genlaguerre
from scipy.ndimage import gaussian_filter

# Grid and simulation parameters
SIZE = 350
GRID_WIDTH = SIZE
GRID_HEIGHT = SIZE
TIME_STEPS = SIZE
TIME_DELTA = 8  # Reduced DT for richer time evolution
POTENTIAL_STRENGTH = 1.0  # Coulomb strength
MAX_GATES_PER_CELL = 4  # Quantum gates per cell


ZOOM = 10.0       # >1 zooms out
BASE_SCALE = 400.0
SCALE = BASE_SCALE / ZOOM            # smaller SCALE → larger r values → zoom out
BASE_SIGMA = 0.01
SIGMA_AMPLIFIER = BASE_SIGMA / ZOOM  # smaller sigma → envelope decays slower in grid units

KX = 0.5 * np.pi / SIZE                  # momentum terms unaffected by zoom
KY = 0.3 * np.pi / SIZE

# Physical constants and parameters
COULOMB_STRENGTH = 1.0
NUCLEAR_CORE_RADIUS = 2.0
NUCLEAR_REPULSION_STRENGTH = 0.5
STRONG_FORCE_STRENGTH = 0.1
STRONG_FORCE_RANGE = 3.0
ELECTRON_REPULSION_STRENGTH = 0.08

# --- Initial world state
X, Y = np.meshgrid(np.arange(GRID_WIDTH), np.arange(GRID_HEIGHT))
center_x, center_y = SIZE//2, SIZE//2

def hydrogen_eigenstate_2d(n, m, x, y, cx, cy, scale=SCALE, sigma_amp=SIGMA_AMPLIFIER):
    """2D hydrogen eigenstate with Gaussian envelope."""
    xs, ys = (x - cx)/scale, (y - cy)/scale
    r = np.sqrt(xs**2 + ys**2)
    theta = np.arctan2(ys, xs)
    rho = 2*r/n
    p = n - abs(m) - 1
    norm = np.sqrt((2/n)**2 * factorial(p) / (pi*factorial(p+abs(m))))
    R = rho**abs(m) * np.exp(-rho/2) * genlaguerre(p,2*abs(m))(rho)
    envelope = np.exp(-r**2/(2*sigma_amp**2))
    return (norm*R*np.exp(1j*m*theta)*envelope).astype(np.complex128)

def create_orbital_electron(x, y, center_x, center_y, orbital_radius, quantum_numbers, scale=None):
    if scale is None:
        scale = SCALE  # Use current global SCALE if no specific scale is provided
    n, l, m = quantum_numbers
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
    """Create a more realistic nucleus potential with short-range repulsion."""
    r = np.sqrt((x - nucleus_x)**2 + (y - nucleus_y)**2)
    r = np.maximum(r, 0.1)  # Smaller minimum to allow closer approach
    
    # Long-range Coulomb attraction: V = -k*Z/r (attractive for electrons)
    coulomb_attraction = -POTENTIAL_STRENGTH * charge / r
    
    # Short-range repulsion to prevent collapse into nucleus (models quantum effects)
    # This represents the Pauli exclusion principle and quantum uncertainty
    nuclear_repulsion = NUCLEAR_REPULSION_STRENGTH * np.exp(-r / NUCLEAR_CORE_RADIUS) / (r + 0.1)
    
    return coulomb_attraction + nuclear_repulsion

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

def compute_nuclear_force(nucleus1_pos, nucleus2_pos):
    """Compute forces between two nuclei including Coulomb repulsion and nuclear binding."""
    diff = nucleus2_pos - nucleus1_pos
    r = np.linalg.norm(diff)
    r = max(r, 0.5)  # Prevent singularity
    
    # Coulomb repulsion between protons: F = k*q1*q2/r^2
    coulomb_force = COULOMB_STRENGTH * diff / r**3
    
    # Strong nuclear force (attractive at very short range, models neutron-proton binding)
    if r < STRONG_FORCE_RANGE:
        # Exponentially decaying attractive force
        nuclear_attraction = -STRONG_FORCE_STRENGTH * np.exp(-r / STRONG_FORCE_RANGE) * diff / r
    else:
        nuclear_attraction = np.zeros(2)
    
    return coulomb_force + nuclear_attraction

def enhanced_electron_electron_repulsion(psi1, psi2):
    """Enhanced electron-electron repulsion with better physical modeling."""
    density1 = np.abs(psi1)**2
    density2 = np.abs(psi2)**2
    
    # Create repulsion potential based on both electron densities
    # Use different sigma values for short and long range interactions
    short_range_repulsion = gaussian_filter(density2, sigma=2) * ELECTRON_REPULSION_STRENGTH * 2
    long_range_repulsion = gaussian_filter(density2, sigma=8) * ELECTRON_REPULSION_STRENGTH * 0.5
    
    return short_range_repulsion + long_range_repulsion

def propagate_wave_with_potential(psi, potential, dt=TIME_DELTA):
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

def apply_absorption_edge_low_pass(psi, cutoff_frequency, blur_factor=SIZE // 2):
    """Apply a low-pass filter specifically to the absorption edge."""
    r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    edge_mask = np.exp(-r**2 / (2 * blur_factor**2))

    psi_hat = np.fft.fft2(psi * edge_mask)
    kx = np.fft.fftfreq(psi.shape[1]) * 2 * np.pi
    ky = np.fft.fftfreq(psi.shape[0]) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    k_squared = KX**2 + KY**2

    low_pass_mask = k_squared <= (cutoff_frequency**2)
    psi_hat_filtered = psi_hat * low_pass_mask

    psi_filtered = np.fft.ifft2(psi_hat_filtered)
    return psi_filtered.real

def apply_edge_blur(psi, blur_factor=SIZE // 2):
    """Apply a large Gaussian blur to the edges of the world."""
    r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    edge_blur = np.exp(-r**2 / (2 * blur_factor**2))
    return psi * edge_blur

# Initial conditions
# Define two hydrogen nuclei (fixed starting positions)
nucleus1_x, nucleus1_y = center_x - SIZE // (ZOOM * 2), center_y
nucleus2_x, nucleus2_y = center_x + SIZE // (ZOOM * 2), center_y

# Two nuclear potentials (will be overwritten dynamically each step)
nuclear_potential1 = create_nucleus_potential(X, Y, nucleus1_x, nucleus1_y, charge=1)
nuclear_potential2 = create_nucleus_potential(X, Y, nucleus2_x, nucleus2_y, charge=1)

# Two electrons with excited states having angular momentum (swirling orbitals)
psi1 = hydrogen_eigenstate_2d(3, +2, X, Y, nucleus1_x, nucleus1_y, scale=SCALE)
psi2 = hydrogen_eigenstate_2d(3, -2, X, Y, nucleus2_x, nucleus2_y, scale=SCALE)

# Apply phase vortex to enhance rotation
theta1 = np.arctan2(Y - nucleus1_y, X - nucleus1_x)
theta2 = np.arctan2(Y - nucleus2_y, X - nucleus2_x)
# psi1 *= np.exp(1j * 2 * theta1)  # Add additional angular momentum
# psi2 *= np.exp(-1j * 2 * theta2)

# Also apply some linear momentum for interesting dynamics
momentum1 = np.exp(1j * (2 * KX * X + 1 * KY * Y))
momentum2 = np.exp(-1j * (1.5 * KX * X - 1.2 * KY * Y))
# psi1 *= momentum1
# psi2 *= momentum2

# Initial dynamic positions and velocities for protons
nucleus1_pos = np.array([nucleus1_x, nucleus1_y], dtype=np.float64)
nucleus2_pos = np.array([nucleus2_x, nucleus2_y], dtype=np.float64)
nucleus1_vel = np.zeros(2)
nucleus2_vel = np.zeros(2)
proton_mass = 1836.0  # Proton is ~1836 times heavier than electron

def compute_force_from_density(charge_density, nucleus_pos):
    """Compute forces on nucleus from electron density."""
    dx = X - nucleus_pos[0]
    dy = Y - nucleus_pos[1]
    r = np.sqrt(dx**2 + dy**2)
    r = np.maximum(r, 1.0)
    force_x = np.sum((dx / r**3) * charge_density)
    force_y = np.sum((dy / r**3) * charge_density)
    return np.array([force_x, force_y])

# Normalize and pre-process initial wavefunctions
psi1 = normalize_wavefunction(psi1)
psi2 = normalize_wavefunction(psi2)
psi1 = reverse_gaussian_blur(psi1, center_x, center_y)
psi2 = reverse_gaussian_blur(psi2, center_x, center_y)

cutoff_frequency = 0.1  # Adjust as needed
psi1 = apply_low_pass_filter(psi1, cutoff_frequency)
psi2 = apply_low_pass_filter(psi2, cutoff_frequency)

# Quantum gate infrastructure (optional)
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

# Centering state (not currently used)
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

def tween_wavefunctions(psi_start, psi_end, alpha):
    return (1 - alpha) * psi_start + alpha * psi_end

# Timing parameters for tweening
stay_duration = 600000000  # Number of time steps to stay in a state
tween_duration = 15  # Number of time steps for a tween
tween_step = 0
stay_step = 0
current_state_index = 0
next_state_index = 1

# Simulation loop
frames_real, frames_imag, frames_phase, frames_prob = [], [], [], []
cur = psi1.copy()
print("Simulating hydrogen atom with dynamic nuclei and two electrons...")
print(f"Time steps: {TIME_STEPS}, dt: {TIME_DELTA}")

for step in range(TIME_STEPS):
    if step % 50 == 0:
        print(f"Step {step}/{TIME_STEPS}")

    # Hydrogen orbital tweening (optional visualization step)
    if stay_step < stay_duration:
        if stay_step == 0:
            current_state = stable_states[current_state_index]
            cur = hydrogen_eigenstate_2d(
                current_state[0], current_state[1], X, Y,
                int(nucleus1_pos[0]), int(nucleus1_pos[1]), scale=SCALE
            )
            cur = normalize_wavefunction(cur)
        stay_step += 1
    elif tween_step < tween_duration:
        current_state = stable_states[current_state_index]
        next_state = stable_states[next_state_index]
        psi_current = hydrogen_eigenstate_2d(
            current_state[0], current_state[1], X, Y,
            int(nucleus1_pos[0]), int(nucleus1_pos[1]), scale=SCALE
        )
        psi_next = hydrogen_eigenstate_2d(
            next_state[0], next_state[1], X, Y,
            int(nucleus1_pos[0]), int(nucleus1_pos[1]), scale=SCALE
        )
        alpha = tween_step / tween_duration
        cur = tween_wavefunctions(psi_current, psi_next, alpha)
        momentum_x = KX * 0.1 * alpha
        momentum_y = KY * 0.1 * alpha
        momentum_phase = np.exp(1j * (momentum_x * X + momentum_y * Y))
        cur *= momentum_phase
        cur = normalize_wavefunction(cur)
        tween_step += 1
    else:
        current_state_index = next_state_index
        next_state_index = (next_state_index + 1) % len(stable_states)
        stay_step = 0
        tween_step = 0

    # --- Dynamic Proton Updates ---
    density1 = np.abs(psi1)**2
    density2 = np.abs(psi2)**2
    force1 = compute_force_from_density(density1, nucleus1_pos)
    force2 = compute_force_from_density(density2, nucleus2_pos)

    # Enhanced proton-proton nuclear forces (Coulomb + strong nuclear force)
    nuclear_force_1_to_2 = compute_nuclear_force(nucleus1_pos, nucleus2_pos)
    nuclear_force_2_to_1 = -nuclear_force_1_to_2  # Newton's third law

    # Total forces on each proton (electron attraction + nuclear forces)
    force_on_1 = force1 + nuclear_force_2_to_1
    force_on_2 = force2 + nuclear_force_1_to_2

    # Update velocities and positions
    nucleus1_vel += TIME_DELTA * force_on_1 / proton_mass
    nucleus2_vel += TIME_DELTA * force_on_2 / proton_mass
    nucleus1_pos += TIME_DELTA * nucleus1_vel
    nucleus2_pos += TIME_DELTA * nucleus2_vel

    # Recreate potentials based on updated positions
    nuclear_potential1 = create_nucleus_potential(
        X, Y, nucleus1_pos[0], nucleus1_pos[1], charge=1
    )
    nuclear_potential2 = create_nucleus_potential(
        X, Y, nucleus2_pos[0], nucleus2_pos[1], charge=1
    )
    combined_nuclear_potential = nuclear_potential1 + nuclear_potential2

    # Each electron experiences both nuclei attraction and the other electron’s repulsion
    V1 = combined_nuclear_potential + add_mean_field_coulomb_repulsion(psi2)
    V2 = combined_nuclear_potential + add_mean_field_coulomb_repulsion(psi1)

    # Propagate electrons
    psi1 = propagate_wave_with_potential(psi1, V1)
    psi2 = propagate_wave_with_potential(psi2, V2)
    
    # Apply a small low-pass filter at each step to smooth the wavefunctions
    # This helps reduce numerical artifacts and stabilize the simulation
    slight_cutoff_frequency = 0.9  # Higher than initial cutoff, just for light smoothing
    psi1 = apply_low_pass_filter(psi1, slight_cutoff_frequency)
    psi2 = apply_low_pass_filter(psi2, slight_cutoff_frequency)

    # Optional spatial gates or blurs (commented out)
    # cur = center_wave(cur)
    # cur = apply_spatial_gates_from_patch(cur, gate_patch_map)
    # cur = apply_edge_blur(cur)
    cur = apply_absorption_edge_low_pass(cur, cutoff_frequency=0.1)

    # Record frames for visualization
    region = psi1 + psi2  # Display combined electron wavefunctions
    frames_real.append(np.real(region))
    frames_imag.append(np.imag(region))

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
