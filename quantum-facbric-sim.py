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
TIME_STEPS = SIZE * 2
MAX_GATES_PER_CELL = 4
DT = 0.5

# --- Initial wave packet: Gaussian with momentum ---
X, Y = np.meshgrid(np.arange(GRID_WIDTH), np.arange(GRID_HEIGHT))
center_x, center_y = SIZE // 2, SIZE // 2

def create_bit(x, y, center_x, center_y, wave_length=96):
    """Create a quantum bit with Gaussian profile and momentum."""
    r2 = (x - center_x)**2 + (y - center_y)**2
    sigma = 6
    momentum = np.exp(1j * 2 * pi * x / wave_length)
    psi = np.exp(-r2 / (2 * sigma**2)) * momentum
    return psi.astype(np.complex128)

# create first bit
psi_t = create_bit(X, Y, center_x, center_y, 19)

# create second bit with different momentum and position
psi_t = create_bit(X, Y, center_x + 24, center_y + 16, 48) + psi_t

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

# --- Split-Step Fourier propagation ---
def propagate_wave(psi, dt=DT):
    psi_hat = np.fft.fft2(psi)
    kx = np.fft.fftfreq(psi.shape[1]) * 2 * np.pi
    ky = np.fft.fftfreq(psi.shape[0]) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    kinetic_phase = np.exp(-1j * dt * (KX**2 + KY**2))
    psi_hat *= kinetic_phase
    return np.fft.ifft2(psi_hat)

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

# --- Run simulation ---
frames_real, frames_imag, frames_phase, frames_prob = [], [], [], []
cur = psi_t.copy()

for _ in range(TIME_STEPS):
    cur = propagate_wave(cur)
    cur = center_wave(cur)  # center using COM
    cur = apply_spatial_gates_from_patch(cur, gate_patch_map)

    region = cur[SIZE//4:3*SIZE//4, SIZE//4:3*SIZE//4]
    frames_real.append(np.real(region))
    frames_imag.append(np.imag(region))
    frames_phase.append(np.angle(region))
    frames_prob.append(np.abs(region)**2)

# --- Create video ---
video_file = "quantum_wave_centered.mkv"
create_video(
    TIME_STEPS,
    frames_real,
    frames_imag,
    frames_phase,
    frames_prob,
    fps=24,
    output_file=video_file
)

open_video(video_file)
