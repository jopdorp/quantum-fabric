import numpy as np
import matplotlib.pyplot as plt
from math import pi
from matplotlib.widgets import Button
from plot_utils import create_and_show_plot
from video_utils import create_video, open_video

# Grid and simulation parameters
SIZE = 128
GRID_WIDTH = SIZE
GRID_HEIGHT = SIZE
TIME_STEPS = SIZE
MAX_GATES_PER_CELL = 4
DT = 0.1

# --- Improved Gaussian + directional motion ---
X, Y = np.meshgrid(np.arange(GRID_WIDTH), np.arange(GRID_HEIGHT))
center_x, center_y = SIZE // 2 - 30, SIZE // 2  # Start left of center
r2 = (X - center_x)**2 + (Y - center_y)**2
sigma = 6
momentum = np.exp(1j * 2 * pi * X / 48)  # Rightward motion
psi_t = np.exp(-r2 / (2 * sigma**2)) * momentum
psi_t = psi_t.astype(np.complex128)

# --- Quantum gate functions ---
def hadamard(amp):
    return amp * (1 + 1j) / np.sqrt(2)

def pauli_x(amp):
    return -amp

def t_gate(amp):
    return amp * np.exp(1j * pi / 4)

# --- Patch-based gate storage ---
gate_patch_map = np.full((GRID_HEIGHT, GRID_WIDTH, MAX_GATES_PER_CELL), None, dtype=object)

def add_gate_patch(gate_patch_map, center_y, center_x, gate_fn):
    radius = 3
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

# --- Schrödinger-like wave propagation ---
def propagate_wave(psi, dt=DT):
    laplacian = (
        np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) +
        np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) - 4 * psi
    )
    return psi + 1j * dt * laplacian

# --- Add spatial gate regions ---
add_gate_patch(gate_patch_map, center_y + 10, center_x + 20, hadamard)
add_gate_patch(gate_patch_map, center_y - 8, center_x + 4, pauli_x)
add_gate_patch(gate_patch_map, center_y - 3, center_x + 15, t_gate)

# --- Run simulation ---
frames_real, frames_imag, frames_phase, frames_prob = [], [], [], []
cur = psi_t.copy()

for _ in range(TIME_STEPS):
    cur = propagate_wave(cur)
    cur = apply_spatial_gates_from_patch(cur, gate_patch_map)

    frames_real.append(np.real(cur))
    frames_imag.append(np.imag(cur))
    frames_phase.append(np.angle(cur))
    frames_prob.append(np.abs(cur)**2)

# --- Create video with all 4 views (real, imag, phase, prob) ---
video_file = "quantum_wave_simulation.mkv"
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
create_and_show_plot(
    9,
    TIME_STEPS,
    frames_real, frames_imag, frames_phase, frames_prob
)