# Re-import needed libraries after kernel reset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from plot_utils import create_and_show_plot
from video_utils import create_video, open_video

# Grid and simulation parameters
GRID_WIDTH = 256
GRID_HEIGHT = 256
TIME_STEPS = 256

# Initialize state
psi_t = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.complex128)
psi_t1 = np.zeros_like(psi_t)
psi_t[128, 128] = 1.0 + 0j
psi_t1[128, 128] = 1.0 + 0j

# Define gates
gate_map = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int32)
gate_map[108:116, 108:110] = 1  # Hadamard
gate_map[90:94, 90:94] = 2  # Pauli-X

# Gate logic
def hadamard(amp):
    return amp * (1 + 1j) / np.sqrt(2)

def pauli_x(amp):
    return -amp

def apply_spatial_gates_once(psi, gate_map, gate_applied):
    updated = psi.copy()
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            gate = gate_map[y, x]
            if gate != 0 and not gate_applied[y, x] and abs(psi[y, x]) > 0.01:
                if gate == 1:
                    updated[y, x] = hadamard(psi[y, x])
                elif gate == 2:
                    updated[y, x] = pauli_x(psi[y, x])
                gate_applied[y, x] = True
    return updated, gate_applied

def propagate_wave(cur, prev):
    laplacian = (
        np.roll(cur, 1, axis=0) + np.roll(cur, -1, axis=0) +
        np.roll(cur, 1, axis=1) + np.roll(cur, -1, axis=1) - 4 * cur
    )
    return 2 * cur - prev + 0.1 * laplacian

# Run simulation
frames_real, frames_imag, frames_phase = [], [], []
cur = psi_t.copy()
prev = psi_t1.copy()
applied = np.zeros_like(gate_map, dtype=bool)

for _ in range(TIME_STEPS):
    next_ = propagate_wave(cur, prev)
    next_, applied = apply_spatial_gates_once(next_, gate_map, applied)

    frames_real.append(np.real(next_))
    frames_imag.append(np.imag(next_))
    frames_phase.append(np.angle(next_))

    prev, cur = cur, next_

# Configuration for display
# VISIBLE_FRAMES = 9  # Number of frames to show at once

# Create video
video_file = "quantum_wave_simulation.mkv"
create_video(TIME_STEPS, frames_real, frames_imag, frames_phase, fps=24, output_file=video_file)
open_video(video_file)
# Create and show the plot
# create_and_show_plot(VISIBLE_FRAMES, TIME_STEPS, frames_real, frames_imag, frames_phase)
