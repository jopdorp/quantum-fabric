import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
import numpy as np

def create_plot(VISIBLE_FRAMES):
    """Create the plot layout for displaying frames."""
    fig = plt.figure(figsize=(18, 9))

    # Create grid layout: 3 rows (real, imag, phase) x VISIBLE_FRAMES columns + navigation
    gs = gridspec.GridSpec(4, VISIBLE_FRAMES, figure=fig, height_ratios=[1, 1, 1, 0.1])

    # Create subplots for the visible frames
    axes = []
    for row in range(3):
        row_axes = []
        for col in range(VISIBLE_FRAMES):
            ax = fig.add_subplot(gs[row, col])
            row_axes.append(ax)
        axes.append(row_axes)

    # Navigation buttons
    ax_prev = fig.add_subplot(gs[3, :VISIBLE_FRAMES // 2])
    ax_next = fig.add_subplot(gs[3, VISIBLE_FRAMES // 2:])

    return fig, axes, ax_prev, ax_next

def update_plot(start_idx, current_start, TIME_STEPS, VISIBLE_FRAMES, frames_real, frames_imag, frames_phase, axes, fig):
    """Update the plot with the current range of frames."""
    current_start = max(0, min(start_idx, TIME_STEPS - VISIBLE_FRAMES))

    for col in range(VISIBLE_FRAMES):
        frame_idx = current_start + col
        if frame_idx < TIME_STEPS:
            # Real component
            axes[0][col].clear()
            axes[0][col].imshow(frames_real[frame_idx], cmap='coolwarm')
            axes[0][col].axis('off')
            axes[0][col].set_title(f"t={frame_idx}", fontsize=10)

            # Imaginary component
            axes[1][col].clear()
            axes[1][col].imshow(frames_imag[frame_idx], cmap='coolwarm')
            axes[1][col].axis('off')

            # Phase component
            axes[2][col].clear()
            axes[2][col].imshow(frames_phase[frame_idx], cmap='twilight', vmin=-np.pi, vmax=np.pi)
            axes[2][col].axis('off')
        else:
            # Clear axes if no more frames
            for row in range(3):
                axes[row][col].clear()
                axes[row][col].axis('off')

    # Add row labels
    axes[0][0].set_ylabel("Real(ψ)", fontsize=12)
    axes[1][0].set_ylabel("Imag(ψ)", fontsize=12)
    axes[2][0].set_ylabel("Phase(ψ)", fontsize=12)

    # Update title with current range
    end_idx = min(current_start + VISIBLE_FRAMES - 1, TIME_STEPS - 1)
    fig.suptitle(f"Wave Components Over Time — Frames {current_start} to {end_idx}", fontsize=16)

    plt.draw()

    return current_start

def setup_navigation(fig, axes, ax_prev, ax_next, VISIBLE_FRAMES, TIME_STEPS, frames_real, frames_imag, frames_phase):
    """Set up navigation buttons and callbacks for the plot."""
    current_start = 0

    def prev_frames(event):
        nonlocal current_start
        current_start = update_plot(current_start - VISIBLE_FRAMES, current_start, TIME_STEPS, VISIBLE_FRAMES, frames_real, frames_imag, frames_phase, axes, fig)

    def next_frames(event):
        nonlocal current_start
        current_start = update_plot(current_start + VISIBLE_FRAMES, current_start, TIME_STEPS, VISIBLE_FRAMES, frames_real, frames_imag, frames_phase, axes, fig)

    btn_prev = Button(ax_prev, '◀ Previous')
    btn_next = Button(ax_next, 'Next ▶')
    btn_prev.on_clicked(prev_frames)
    btn_next.on_clicked(next_frames)

    # Initial plot
    current_start = update_plot(0, current_start, TIME_STEPS, VISIBLE_FRAMES, frames_real, frames_imag, frames_phase, axes, fig)

    return btn_prev, btn_next

def create_and_show_plot(VISIBLE_FRAMES, TIME_STEPS, frames_real, frames_imag, frames_phase):
    """Create the plot, set up navigation, and show it."""
    # Create the plot
    fig, axes, ax_prev, ax_next = create_plot(VISIBLE_FRAMES)
    
    # Set up navigation
    setup_navigation(fig, axes, ax_prev, ax_next, VISIBLE_FRAMES, TIME_STEPS, frames_real, frames_imag, frames_phase)
    
    plt.tight_layout()
    plt.show()
