import cv2
import numpy as np
import subprocess
from matplotlib import cm

def normalize_to_uint8(data, vmin, vmax):
    """Normalize data to 0-255 uint8 range"""
    # Handle edge case where vmin == vmax
    if np.abs(vmax - vmin) < 1e-10:
        return np.full_like(data, 128, dtype=np.uint8)  # Return mid-gray
    
    normalized = (data - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)
    return (normalized * 255).astype(np.uint8)

def apply_colormap(data, colormap_name, vmin, vmax):
    """Apply matplotlib colormap to data and convert to BGR"""
    normalized = normalize_to_uint8(data, vmin, vmax)
    colormap = cm.get_cmap(colormap_name)
    colored = colormap(normalized / 255.0)  # colormap expects 0-1 range
    # Convert RGBA to BGR and scale to 0-255
    bgr = (colored[:, :, [2, 1, 0]] * 255).astype(np.uint8)  # BGR order for OpenCV
    return bgr

def create_video(TIME_STEPS, frames_real, frames_imag, frames_phase, frames_prob, fps=8, output_file="quantumsim.mkv"):
    """Create a video with horizontally stitched real, imaginary, and phase components"""
    print("Creating video...")
    
    # Pre-calculate global ranges to avoid per-frame percentile calculations
    print("Pre-calculating global ranges...")
    all_real = np.concatenate([f.flatten() for f in frames_real])
    all_prob = np.concatenate([f.flatten() for f in frames_prob])
    all_imag_abs = np.concatenate([np.abs(f).flatten() for f in frames_imag])
    
    # Use global ranges for consistent scaling across all frames
    real_global_min, real_global_max = np.percentile(all_real, [1, 99])
    prob_global_min, prob_global_max = np.percentile(all_prob, [5, 95])
    
    # Handle imaginary component
    if np.max(all_imag_abs) > 1e-10:
        imag_global_min, imag_global_max = np.percentile(all_imag_abs, [0.1, 99.9])
        if imag_global_max - imag_global_min < 1e-10:
            imag_global_min = 0
            imag_global_max = np.max(all_imag_abs) if np.max(all_imag_abs) > 0 else 1e-6
    else:
        imag_global_min, imag_global_max = 0, 1e-6
    
    # Create the first frame to get video dimensions
    frame_real = frames_real[0]
    frame_imag = frames_imag[0]
    frame_prob = frames_prob[0]
    
    real_colored = apply_colormap(frame_real, 'coolwarm', real_global_min, real_global_max)
    abs_imag = np.abs(frame_imag)
    imag_colored = apply_colormap(abs_imag, 'plasma', imag_global_min, imag_global_max)
    phase_colored = apply_colormap(frames_phase[0], 'twilight', -np.pi, np.pi)
    prob_colored = apply_colormap(frame_prob, 'hot', prob_global_min, prob_global_max)
    
    # Stitch frames horizontally
    stitched_frame = np.hstack([real_colored, imag_colored, phase_colored, prob_colored])
    video_height, video_width = stitched_frame.shape[:2]
    
    # Use faster H.264 codec instead of FFV1 for speed
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # FFV1 is truly lossless
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (video_width, video_height))
    
    # Generate and write frames directly with pre-calculated ranges
    print("Generating frames...")
    for t in range(TIME_STEPS):
        frame_real = frames_real[t]
        frame_imag = frames_imag[t]
        frame_prob = frames_prob[t]
        frame_phase = frames_phase[t]
        
        # Use pre-calculated global ranges - much faster than per-frame percentiles
        abs_imag = np.abs(frame_imag)
        
        # Apply colormaps with global scaling
        real_colored = apply_colormap(frame_real, 'coolwarm', real_global_min, real_global_max)
        imag_colored = apply_colormap(abs_imag, 'plasma', imag_global_min, imag_global_max)
        phase_colored = apply_colormap(frame_phase, 'twilight', -np.pi, np.pi)
        prob_colored = apply_colormap(frame_prob, 'hot', prob_global_min, prob_global_max)

        # Stitch frames horizontally
        stitched_frame = np.hstack([real_colored, imag_colored, phase_colored, prob_colored])
        
        # Write frame to video
        video_writer.write(stitched_frame)
        
        if t % 50 == 0:  # Reduced logging frequency
            print(f"Generated frame {t}/{TIME_STEPS}")
    
    video_writer.release()
    
    print(f"Video saved as {output_file}")
    print(f"Video specs: {video_width}x{video_height}, {fps} fps, {TIME_STEPS} frames")

def open_video(video_path):
    """Open video with the default system video player"""
    try:
        # Linux
        subprocess.run(['xdg-open', video_path], check=True)
        print(f"Opening video with default player: {video_path}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # Alternative for Linux
            subprocess.run(['vlc', video_path], check=True)
            print(f"Opening video with VLC: {video_path}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"Could not open video automatically. Please open manually: {video_path}")
