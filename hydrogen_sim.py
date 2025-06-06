import numpy as np
import matplotlib.pyplot as plt
from math import pi
from matplotlib.widgets import Button
from plot_utils import create_and_show_plot
from video_utils import create_video, open_video

# Grid and simulation parameters
SIZE = 512
GRID_WIDTH = SIZE
GRID_HEIGHT = SIZE
TIME_STEPS = SIZE
MAX_GATES_PER_CELL = 4
DT = 0.5

# --- Initial wave packet: Gaussian with momentum ---
X, Y = np.meshgrid(np.arange(GRID_WIDTH), np.arange(GRID_HEIGHT))
center_x, center_y = SIZE // 2, SIZE // 2

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

# Add electron-electron repulsion potential (unused in hydrogen - single electron atom)
def create_electron_repulsion(psi, repulsion_strength=0.1):
    """Add electron-electron repulsion based on current electron density.
    Note: This is unused for hydrogen atom simulation since hydrogen has only one electron.
    This function would be useful for multi-electron atom simulations like helium.
    """
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
    sigma = n * bohr_radius * 1.2  # Increased for smoother decay
    envelope = np.exp(-((r - n * bohr_radius)**2) / (2 * sigma**2))
    
    psi = radial_part * angular_part * envelope
    return psi.astype(np.complex128)

# Create single electron in hydrogen atom (1s ground state)
# Hydrogen atom has only one electron
# Note: orbital_radius=25 is scaled for visualization; for atomic units use bohr_radius=1
electron1 = create_orbital_electron(X, Y, center_x, center_y, 
                                  orbital_radius=25, quantum_numbers=(1, 0, 0))  # 1s orbital

# Single electron wavefunction for hydrogen atom
psi_t = electron1 

# Normalize initial wavefunction
norm = np.linalg.norm(psi_t)
if norm > 0:
    psi_t /= norm 

# --- Add a small random perturbation to break symmetry ---
np.random.seed(42)  # For reproducibility
perturbation = np.random.normal(scale=1e-6, size=psi_t.shape)
psi_t += perturbation.astype(np.complex128)

# --- Create absorbing boundary mask ---
def create_absorbing_mask(size, edge_width=20):
    """Create absorbing boundary mask to prevent reflections."""
    mask = np.ones((size, size))
    for i in range(edge_width):
        fade = np.cos((i / edge_width) * pi / 2) ** 2
        mask[i, :] *= fade
        mask[-(i + 1), :] *= fade
        mask[:, i] *= fade
        mask[:, -(i + 1)] *= fade
    return mask

# Create absorbing boundary mask
absorbing_mask = create_absorbing_mask(SIZE, edge_width=50)

# Apply absorbing boundary mask to the initial wavefunction
psi_t *= absorbing_mask

# --- Split-Step Fourier propagation with nuclear potential ---
def propagate_wave_with_potential(psi, potential, dt=DT):
    """Propagate wave using split-step method with potential energy."""
    # Apply potential energy operator for half time step
    potential_phase = np.exp(-1j * dt * potential / 2)
    psi = psi * potential_phase
    
    # Apply kinetic energy operator (Fourier space) with corrected frequency scaling
    psi_hat = np.fft.fft2(psi)
    # Fix FFT frequency scaling
    kx = np.fft.fftfreq(psi.shape[1], d=1.0 / psi.shape[1]) * 2 * np.pi
    ky = np.fft.fftfreq(psi.shape[0], d=1.0 / psi.shape[0]) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    kinetic_phase = np.exp(-1j * dt * (KX**2 + KY**2) / 2)  # Reduced kinetic energy scale
    psi_hat *= kinetic_phase
    psi = np.fft.ifft2(psi_hat)
    
    # Apply potential energy operator for second half time step
    psi = psi * potential_phase
    
    # Apply absorbing boundary to prevent reflections
    psi *= absorbing_mask
    
    # Normalize wavefunction to preserve probability
    norm = np.linalg.norm(psi)
    if norm > 0:
        psi /= norm
    
    return psi

def calculate_energy(psi, potential, kx_grid, ky_grid):
    """Calculate total energy (kinetic + potential) of the wavefunction."""
    # Potential energy
    V_energy = np.sum(np.abs(psi)**2 * potential)
    
    # Kinetic energy in Fourier space
    psi_hat = np.fft.fft2(psi)
    KE_density = (kx_grid**2 + ky_grid**2) * np.abs(psi_hat)**2
    K_energy = np.sum(KE_density)
    
    total_energy = K_energy + V_energy
    return total_energy.real, K_energy.real, V_energy.real

def propagate_wave(psi, dt=DT):
    """Legacy function for backward compatibility."""
    return propagate_wave_with_potential(psi, nuclear_potential, dt)

# Pre-compute frequency grids for energy calculation
kx_freq = np.fft.fftfreq(SIZE, d=1.0 / SIZE) * 2 * np.pi
ky_freq = np.fft.fftfreq(SIZE, d=1.0 / SIZE) * 2 * np.pi
KX_ENERGY, KY_ENERGY = np.meshgrid(kx_freq, ky_freq)

# --- Center-of-mass tracker with smoothing ---
# Global variables for smoothing
smooth_cy = SIZE // 2
smooth_cx = SIZE // 2
smoothing_factor = 400

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

# Track simulation diagnostics and optional settings
total_probs = []
energies = []
enable_centering = True  # Set to False to see true orbital motion/rotation
track_energy = True  # Track total energy conservation

print("Simulating hydrogen atom with single electron...")
print(f"Nuclear position: ({nucleus_x}, {nucleus_y})")
print(f"Time steps: {TIME_STEPS}, dt: {DT}")
print(f"Initial wavefunction norm: {np.linalg.norm(psi_t):.6f}")
print(f"Initial probability: {np.sum(np.abs(psi_t)**2):.6f}")
print(f"Centering enabled: {enable_centering}")
print(f"Energy tracking enabled: {track_energy}")

# Calculate initial energy
if track_energy:
    initial_E, initial_K, initial_V = calculate_energy(psi_t, nuclear_potential, KX_ENERGY, KY_ENERGY)
    print(f"Initial total energy: {initial_E:.6f} (K: {initial_K:.6f}, V: {initial_V:.6f})")
    energies.append(initial_E)

for step in range(TIME_STEPS):
    if step % 50 == 0:
        total_prob = np.sum(np.abs(cur)**2)
        total_probs.append(total_prob)
        
        # Optional energy tracking
        if track_energy:
            current_E, current_K, current_V = calculate_energy(cur, nuclear_potential, KX_ENERGY, KY_ENERGY)
            energies.append(current_E)
            print(f"Step {step}/{TIME_STEPS}, Prob: {total_prob:.6f}, Energy: {current_E:.6f}")
        else:
            print(f"Step {step}/{TIME_STEPS}, Total probability: {total_prob:.6f}")
    
    # For hydrogen, only nuclear attraction (no electron-electron repulsion)
    total_potential = nuclear_potential
    
    cur = propagate_wave_with_potential(cur, total_potential)
    
    # Optional centering
    if enable_centering:
        cur = center_wave(cur)  # center using COM

    # region = cur[SIZE//4:3*SIZE//4, SIZE//4:3*SIZE//4]
    region = cur
    frames_real.append(np.real(region))
    frames_imag.append(np.imag(region))
    frames_phase.append(np.angle(region))
    
    # Enhanced probability density for better visualization
    prob_density = np.abs(region)**2
    # Normalize for consistent visualization
    if np.max(prob_density) > 0:
        prob_density = prob_density / np.max(prob_density)
    frames_prob.append(prob_density)

# Print final diagnostics
print(f"Final total probability: {np.sum(np.abs(cur)**2):.6f}")
print(f"Probability conservation: {np.std(total_probs):.6f} (lower is better)")

if track_energy and len(energies) > 1:
    energy_drift = np.std(energies)
    energy_change = abs(energies[-1] - energies[0])
    print(f"Energy conservation: drift = {energy_drift:.6f}, change = {energy_change:.6f}")
    print(f"Final energy: {energies[-1]:.6f} (initial: {energies[0]:.6f})")

print(f"Simulation completed with centering={'ON' if enable_centering else 'OFF'}")

# --- Create atomic simulation video ---
video_file = "improved_hydrogen_atom_simulation_v2.mkv"
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
