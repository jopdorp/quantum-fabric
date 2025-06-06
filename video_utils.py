import cv2
import numpy as np
import subprocess
from matplotlib import cm

def normalize_to_uint8(data, vmin, vmax):
    """Normalize data to 0-255 uint8 range"""
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

def create_video(TIME_STEPS, frames_real, frames_imag, frames_phase, fps=8, output_file="quantumsim.mkv"):
    """Create a video with horizontally stitched real, imaginary, and phase components"""
    print("Creating video...")
    
    # Calculate global ranges across ALL frames for consistent scaling
    all_real = np.concatenate([frame.flatten() for frame in frames_real])
    all_imag = np.concatenate([frame.flatten() for frame in frames_imag])
    
    real_min, real_max = np.min(all_real), np.max(all_real)
    imag_min, imag_max = np.min(all_imag), np.max(all_imag)
    
    # Add small padding to avoid edge cases
    real_range = real_max - real_min
    imag_range = imag_max - imag_min
    
    real_min -= real_range * 0.05
    real_max += real_range * 0.05
    imag_min -= imag_range * 0.05
    imag_max += imag_range * 0.05
    
    print(f"Real range: {real_min:.6f} to {real_max:.6f}")
    print(f"Imag range: {imag_min:.6f} to {imag_max:.6f}")
    
    # Create the first frame to get video dimensions
    real_colored = apply_colormap(frames_real[0], 'bwr', real_min, real_max)  # Blue-White-Red
    imag_colored = apply_colormap(frames_imag[0], 'coolwarm', imag_min, imag_max)  # Cool-Warm
    phase_colored = apply_colormap(frames_phase[0], 'twilight', -np.pi, np.pi)  # Twilight for phase
    
    # Stitch frames horizontally
    stitched_frame = np.hstack([real_colored, imag_colored, phase_colored])
    video_height, video_width = stitched_frame.shape[:2]
    
    # Create video writer with truly lossless codec
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # FFV1 is truly lossless
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (video_width, video_height))
    
    # Generate and write frames directly
    for t in range(TIME_STEPS):
        # Apply colormaps using the global ranges
        real_colored = apply_colormap(frames_real[t], 'bwr', real_min, real_max)
        imag_colored = apply_colormap(frames_imag[t], 'coolwarm', imag_min, imag_max)
        phase_colored = apply_colormap(frames_phase[t], 'twilight', -np.pi, np.pi)
        
        # Stitch frames horizontally
        stitched_frame = np.hstack([real_colored, imag_colored, phase_colored])
        
        # Write frame to video
        video_writer.write(stitched_frame)
        
        if t % 10 == 0:
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
