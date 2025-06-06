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
TIME_STEPS = SIZE * 3  # Longer simulation for orbital dynamics
MAX_GATES_PER_CELL = 4
DT = 2  # Smaller time step for stability with potential

# --- Initial wave packet: Gaussian with momentum ---
X, Y = np.meshgrid(np.arange(GRID_WIDTH), np.arange(GRID_HEIGHT))
center_x, center_y = SIZE // 2, SIZE // 2

def create_electron(x, y, center_x, center_y, wave_length=96):
    """Create an electron wavefunction with Gaussian profile and orbital momentum."""
    r2 = (x - center_x)**2 + (y - center_y)**2
    sigma = 6  # Larger orbital size for electrons
    # Add orbital angular momentum
    momentum = np.exp(1j * 2 * pi * x / wave_length)
    psi = np.exp(-r2 / (2 * sigma**2)) * momentum
    return psi.astype(np.complex128)

# Create atomic nucleus potential (attractive Coulomb potential)
def create_nucleus_potential(x, y, nucleus_x, nucleus_y, charge=1):
    """Create Coulomb potential for atomic nucleus."""
    r = np.sqrt((x - nucleus_x)**2 + (y - nucleus_y)**2)
    # Avoid singularity at nucleus
    r = np.maximum(r, 1.0)
    # Coulomb potential: V = -k*Z/r (attractive for electrons)
    potential_strength = 0.2  # Scaled for simulation stability
    return -potential_strength * charge / r

# Nuclear position (center of atom) - nucleus itself is not visualized
nucleus_x, nucleus_y = center_x, center_y
nuclear_potential = create_nucleus_potential(X, Y, nucleus_x, nucleus_y, charge=1)

# Add electron-electron repulsion potential
def create_electron_repulsion(psi, repulsion_strength=0.1):
    """Add electron-electron repulsion based on current electron density."""
    electron_density = np.abs(psi)**2
    # Smooth the density to avoid numerical instabilities
    from scipy.ndimage import gaussian_filter
    try:
        smoothed_density = gaussian_filter(electron_density, sigma=2.0)
    except:
        # Fallback if scipy not available
        smoothed_density = electron_density
    
    # Repulsion potential proportional to electron density
    repulsion_potential = repulsion_strength * smoothed_density
    return repulsion_potential

# Create electrons in different atomic orbitals with stable distances
def create_orbital_electron(x, y, center_x, center_y, orbital_radius, quantum_numbers):
    """Create electron in specific atomic orbital with quantum numbers (n, l, m)."""
    n, l, m = quantum_numbers  # principal, angular momentum, magnetic quantum numbers
    
    # Convert to polar coordinates relative to nucleus
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    theta = np.arctan2(y - center_y, x - center_x)
    
    # Radial wavefunction (simplified hydrogen-like)
    bohr_radius = orbital_radius
    radial_part = np.exp(-r / (n * bohr_radius)) * (r / bohr_radius)**(l)
    
    # Angular momentum part (orbital shape)
    angular_part = np.exp(1j * m * theta)
    
    # Gaussian envelope for stability
    sigma = n * bohr_radius * 0.8
    envelope = np.exp(-((r - n * bohr_radius)**2) / (2 * sigma**2))
    
    psi = radial_part * angular_part * envelope
    return psi.astype(np.complex128)

# Create single electron in hydrogen atom (1s ground state)
# Hydrogen atom has only one electron
electron1 = create_orbital_electron(X, Y, center_x, center_y, 
                                  orbital_radius=25, quantum_numbers=(1, 0, 0))  # 1s orbital

# Single electron wavefunction for hydrogen atom
psi_t = electron1 

# --- Quantum gate functions ---
def hadamard(amp): return amp * (1 + 1j) / np.sqrt(2)
def pauli_x(amp): return -amp
def t_gate(amp): return amp * np.exp(1j * pi / 4)

# --- Patch-based gate storage ---
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

# --- Split-Step Fourier propagation with nuclear potential ---
def propagate_wave_with_potential(psi, potential, dt=DT):
    """Propagate wave using split-step method with potential energy."""
    # Apply potential energy operator for half time step
    potential_phase = np.exp(-1j * dt * potential / 2)
    psi = psi * potential_phase
    
    # Apply kinetic energy operator (Fourier space)
    psi_hat = np.fft.fft2(psi)
    kx = np.fft.fftfreq(psi.shape[1]) * 2 * np.pi
    ky = np.fft.fftfreq(psi.shape[0]) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    kinetic_phase = np.exp(-1j * dt * (KX**2 + KY**2) / 2)  # Reduced kinetic energy scale
    psi_hat *= kinetic_phase
    psi = np.fft.ifft2(psi_hat)
    
    # Apply potential energy operator for second half time step
    psi = psi * potential_phase
    
    return psi

def propagate_wave(psi, dt=DT):
    """Legacy function for backward compatibility."""
    return propagate_wave_with_potential(psi, nuclear_potential, dt)

# --- Center-of-mass tracker with smoothing ---
# Global variables for smoothing
smooth_cy = SIZE // 2
smooth_cx = SIZE // 2
smoothing_factor = 2000

def center_wave(psi):
    global smooth_cy, smooth_cx, smoothing_factor
    prob = np.abs(psi)**2  # Use the last frame's probability density
    total = np.sum(prob)
    if total == 0:
        return psi  # nothing to center
    
    y_idx, x_idx = np.indices(prob.shape)
    cy = np.sum(y_idx * prob) / total
    cx = np.sum(x_idx * prob) / total
    
    # Apply exponential smoothing
    smooth_cy = smooth_cy + (cy - smooth_cy) / smoothing_factor
    smooth_cx = smooth_cx + (cx - smooth_cx) / smoothing_factor
    
    # Calculate shifts using smoothed center
    shift_y = int(np.round((SIZE // 2) - smooth_cy))
    shift_x = int(np.round((SIZE // 2) - smooth_cx))
    
    return np.roll(np.roll(psi, shift_y, axis=0), shift_x, axis=1)

# --- Run atomic simulation ---
frames_real, frames_imag, frames_phase, frames_prob = [], [], [], []
cur = psi_t.copy()

print("Simulating hydrogen atom with single electron...")
print(f"Nuclear position: ({nucleus_x}, {nucleus_y})")
print(f"Time steps: {TIME_STEPS}, dt: {DT}")

# Function to transition quantum numbers over time
def transition_quantum_numbers(step, total_steps, initial_quantum_numbers, final_quantum_numbers):
    """Smoothly transition quantum numbers from initial to final over time."""
    n1, l1, m1 = initial_quantum_numbers
    n2, l2, m2 = final_quantum_numbers

    # Linear interpolation for simplicity
    n = n1 + (n2 - n1) * (step / total_steps)
    l = l1 + (l2 - l1) * (step / total_steps)
    m = m1 + (m2 - m1) * (step / total_steps)

    return int(round(n)), int(round(l)), int(round(m))

# Initial and final quantum numbers for the transition
initial_quantum_numbers = (1, 0, 0)  # 1s orbital
final_quantum_numbers = (2, 1, 1)    # 2p orbital

for step in range(TIME_STEPS):
    if step % 50 == 0:
        print(f"Step {step}/{TIME_STEPS}")

    # Transition quantum numbers over the simulation
    quantum_numbers = transition_quantum_numbers(step, TIME_STEPS, initial_quantum_numbers, final_quantum_numbers)

    # Update the electron wavefunction with new quantum numbers
    cur = create_orbital_electron(X, Y, center_x, center_y, orbital_radius=25, quantum_numbers=quantum_numbers)

    # For hydrogen, only nuclear attraction (no electron-electron repulsion)
    total_potential = nuclear_potential

    cur = propagate_wave_with_potential(cur, total_potential)
    cur = center_wave(cur)  # center using COM
    cur = apply_spatial_gates_from_patch(cur, gate_patch_map)

    region = cur[SIZE//4:3*SIZE//4, SIZE//4:3*SIZE//4]
    frames_real.append(np.real(region))
    frames_imag.append(np.imag(region))
    frames_phase.append(np.angle(region))

    # Enhanced probability density for better visualization
    prob_density = np.abs(region)**2
    # Normalize for consistent visualization
    if np.max(prob_density) > 0:
        prob_density = prob_density / np.max(prob_density)
    frames_prob.append(prob_density)

# --- Create atomic simulation video ---
video_file = "hydrogen_atom_simulation.mkv"
print(f"Creating video: {video_file}")
create_video(
    TIME_STEPS,
    frames_real,
    frames_imag,
    frames_phase,
    frames_prob,
    fps=30,  # Slightly higher fps for smoother orbital motion
    output_file=video_file
)

print("Opening video...")
open_video(video_file)
