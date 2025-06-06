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
    
    # Create the first frame to get video dimensions
    frame_real = frames_real[0]
    frame_imag = frames_imag[0]
    frame_prob = frames_prob[0]
    
    # Calculate per-frame ranges for first frame
    real_frame_min, real_frame_max = np.percentile(frame_real, [5, 95])
    imag_frame_min, imag_frame_max = np.percentile(frame_imag, [5, 95])
    prob_frame_min, prob_frame_max = np.percentile(frame_prob, [5, 95])
    
    real_colored = apply_colormap(frame_real, 'coolwarm', real_frame_min, real_frame_max)
    imag_colored = apply_colormap(frame_imag, 'coolwarm', imag_frame_min, imag_frame_max)
    phase_colored = apply_colormap(frames_phase[0], 'twilight', -np.pi, np.pi)
    prob_colored = apply_colormap(frame_prob, 'hot', prob_frame_min, prob_frame_max)
    
    # Stitch frames horizontally
    stitched_frame = np.hstack([real_colored, imag_colored, phase_colored, prob_colored])
    video_height, video_width = stitched_frame.shape[:2]
    
    # Create video writer with truly lossless codec
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # FFV1 is truly lossless
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (video_width, video_height))
    
    # Generate and write frames directly
    for t in range(TIME_STEPS):
        # Use frame-by-frame normalization for better contrast (like the plot does)
        frame_real = frames_real[t]
        frame_imag = frames_imag[t]
        frame_prob = frames_prob[t]
        
        # Calculate per-frame ranges using percentiles
        real_frame_min, real_frame_max = np.percentile(frame_real, [5, 95])
        imag_frame_min, imag_frame_max = np.percentile(frame_imag, [5, 95])
        prob_frame_min, prob_frame_max = np.percentile(frame_prob, [5, 95])
        
        # Ensure we don't have zero ranges
        if np.abs(real_frame_max - real_frame_min) < 1e-10:
            real_frame_min = np.min(frame_real)
            real_frame_max = np.max(frame_real)
        if np.abs(imag_frame_max - imag_frame_min) < 1e-10:
            imag_frame_min = np.min(frame_imag)
            imag_frame_max = np.max(frame_imag)
        if np.abs(prob_frame_max - prob_frame_min) < 1e-10:
            prob_frame_min = np.min(frame_prob)
            prob_frame_max = np.max(frame_prob)
        
        # Apply colormaps using per-frame ranges
        real_colored = apply_colormap(frame_real, 'coolwarm', real_frame_min, real_frame_max)
        imag_colored = apply_colormap(frame_imag, 'coolwarm', imag_frame_min, imag_frame_max)
        phase_colored = apply_colormap(frames_phase[t], 'twilight', -np.pi, np.pi)
        prob_colored = apply_colormap(frame_prob, 'hot', prob_frame_min, prob_frame_max)
       
        # Stitch frames horizontally
        stitched_frame = np.hstack([real_colored, imag_colored, phase_colored, prob_colored])
        
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
