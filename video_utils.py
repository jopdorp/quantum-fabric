import cv2
import numpy as np
import subprocess
import torch
from matplotlib import cm

def normalize_to_uint8(data, vmin, vmax, device=None):
    """Normalize data to 0-255 uint8 range - torch optimized version"""
    # Ensure data is a torch tensor
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    
    if device is None:
        device = data.device
    data = data.to(device)
    
    # Handle edge case where vmin == vmax
    range_val = vmax - vmin
    if abs(range_val) < 1e-10:
        return torch.full_like(data, 128, dtype=torch.uint8)
    
    # Vectorized operations with direct uint8 conversion
    inv_range = 255.0 / range_val
    offset = -vmin * inv_range
    
    # Single operation: scale, offset, and clip in one step
    result = data * inv_range + offset
    
    # Clip and convert to uint8 in one operation
    return torch.clamp(result, 0, 255).to(torch.uint8)

def apply_colormap(data, colormap_name, vmin, vmax, device=None):
    """Apply matplotlib colormap to torch tensor and convert to BGR - GPU optimized version"""
    # Ensure data is a torch tensor
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    
    if device is None:
        device = data.device
    data = data.to(device)
    
    # Normalize to 0-255 range with torch operations
    range_val = vmax - vmin
    if abs(range_val) < 1e-10:
        normalized = torch.full_like(data, 128, dtype=torch.uint8)
    else:
        inv_range = 255.0 / range_val
        offset = -vmin * inv_range
        result = data * inv_range + offset
        normalized = torch.clamp(result, 0, 255).to(torch.uint8)
    
    # Create colormap LUT on the same device
    colormap = cm.get_cmap(colormap_name)
    lut_indices = np.arange(256, dtype=np.float32) / 255.0
    lut_rgba = colormap(lut_indices)
    lut_bgr_np = (lut_rgba[:, [2, 1, 0]] * 255).astype(np.uint8)
    lut_bgr = torch.tensor(lut_bgr_np, dtype=torch.uint8, device=device)
    
    # Store original shape for reshaping
    original_shape = normalized.shape
    
    # Flatten for indexing, then reshape back
    normalized_flat = normalized.flatten().long()  # Convert to long for proper indexing
    
    # Index into the lookup table (normalized_flat contains indices 0-255)
    # lut_bgr has shape [256, 3], normalized_flat has shape [H*W]
    # Result should have shape [H*W, 3]
    bgr_flat = lut_bgr[normalized_flat]  # Shape: [H*W, 3]
    
    # Reshape back to original spatial dimensions plus color channel
    bgr = bgr_flat.view(*original_shape, 3)  # Shape: [H, W, 3]
    
    return bgr

def apply_probability_colormap(log_prob_data, vmin, vmax, power=2.5, device=None):
    """Apply custom probability colormap using torch - GPU optimized version"""
    # Ensure data is a torch tensor
    if not isinstance(log_prob_data, torch.Tensor):
        log_prob_data = torch.tensor(log_prob_data, dtype=torch.float32)
    
    if device is None:
        device = log_prob_data.device
    log_prob_data = log_prob_data.to(device)
    
    # Normalize to 0-255 range with torch operations
    if abs(vmax - vmin) < 1e-10:
        normalized_uint8 = torch.full_like(log_prob_data, 128, dtype=torch.uint8)
    else:
        # Normalize to 0-1 range first
        normalized = (log_prob_data - vmin) / (vmax - vmin)
        normalized = torch.clamp(normalized, 0, 1)
        
        # Apply power transformation to darken low values
        enhanced = torch.pow(normalized, power)
        
        # Convert to uint8 for LUT indexing
        normalized_uint8 = (enhanced * 255).to(torch.uint8)
    
    # Create colormap LUT on the same device
    colormap = cm.get_cmap('hot')
    lut_indices = np.arange(256, dtype=np.float32) / 255.0
    lut_rgba = colormap(lut_indices)
    lut_bgr_np = (lut_rgba[:, [2, 1, 0]] * 255).astype(np.uint8)
    lut_bgr = torch.tensor(lut_bgr_np, dtype=torch.uint8, device=device)
    
    # Store original shape for reshaping
    original_shape = normalized_uint8.shape
    
    # Flatten for indexing, then reshape back
    normalized_flat = normalized_uint8.flatten().long()  # Convert to long for proper indexing
    
    # Index into the lookup table (normalized_flat contains indices 0-255)
    # lut_bgr has shape [256, 3], normalized_flat has shape [H*W]
    # Result should have shape [H*W, 3]
    bgr_flat = lut_bgr[normalized_flat]  # Shape: [H*W, 3]
    
    # Reshape back to original spatial dimensions plus color channel
    bgr = bgr_flat.view(*original_shape, 3)  # Shape: [H, W, 3]
    
    return bgr

class StreamingVideoWriter:
    """Memory-efficient streaming video writer for quantum simulations"""
    
    def __init__(self, output_file="quantumsim.mkv", fps=12, sample_frames=100, keep_first_batch=True, first_batch_size=500):
        self.output_file = output_file
        self.fps = fps
        self.sample_frames = sample_frames
        self.keep_first_batch = keep_first_batch
        self.first_batch_size = first_batch_size
        self.video_writer = None
        self.frame_count = 0
        
        # Buffers for calculating global ranges
        self.real_samples = []
        self.imag_samples = []
        self.prob_samples = []
        self.phase_samples = []
        
        # Optional frame storage for first batch
        self.stored_frames = [] if keep_first_batch else None
        self.first_batch_written = False
        
        # Global ranges (calculated after sampling)
        self.real_global_min = None
        self.real_global_max = None
        self.imag_global_min = None
        self.imag_global_max = None
        self.log_prob_min = None
        self.log_prob_max = None
        
        print(f"Initializing streaming video writer: {output_file}")
        print(f"Will sample {sample_frames} frames to calculate global ranges...")
        if keep_first_batch:
            print(f"Will keep first {first_batch_size} frames in memory for potential overwrites")
    
    def sample_frame(self, frame_real, frame_imag, frame_phase, frame_prob):
        """Sample frames to calculate global ranges before starting video writer"""
        if len(self.real_samples) < self.sample_frames:
            # Subsample to reduce memory during range calculation
            subsample = 4  # Take every 4th pixel
            self.real_samples.append(frame_real[::subsample, ::subsample].flatten())
            self.imag_samples.append(np.abs(frame_imag[::subsample, ::subsample]).flatten())
            self.prob_samples.append(frame_prob[::subsample, ::subsample].flatten())
            self.phase_samples.append(frame_phase[::subsample, ::subsample].flatten())
            
            if len(self.real_samples) == self.sample_frames:
                self._calculate_global_ranges()
                self._initialize_video_writer(frame_real.shape)
                print("Global ranges calculated, video writer initialized")
    
    def _calculate_global_ranges(self):
        """Calculate global min/max from sampled frames"""
        print("Calculating global ranges from samples...")
        
        # Real component
        all_real = np.concatenate(self.real_samples)
        self.real_global_min, self.real_global_max = np.percentile(all_real, [1, 99])
        
        # Imaginary component  
        all_imag_abs = np.concatenate(self.imag_samples)
        if np.max(all_imag_abs) > 1e-10:
            self.imag_global_min, self.imag_global_max = np.percentile(all_imag_abs, [0.1, 99.9])
            if self.imag_global_max - self.imag_global_min < 1e-10:
                self.imag_global_min = 0
                self.imag_global_max = np.max(all_imag_abs) if np.max(all_imag_abs) > 0 else 1e-6
        else:
            self.imag_global_min, self.imag_global_max = 0, 1e-6
        
        # Probability (log scale)
        all_prob = np.concatenate(self.prob_samples)
        all_prob_nonzero = all_prob[all_prob > 0]
        if len(all_prob_nonzero) > 0:
            self.log_prob_min = np.log10(np.percentile(all_prob_nonzero, 1))
            self.log_prob_max = np.log10(np.percentile(all_prob_nonzero, 99.9))
        else:
            self.log_prob_min, self.log_prob_max = -6, 0
        
        # Clear sample buffers to free memory
        self.real_samples.clear()
        self.imag_samples.clear()
        self.prob_samples.clear()
        self.phase_samples.clear()
    
    def _initialize_video_writer(self, frame_shape):
        """Initialize OpenCV video writer after determining frame dimensions"""
        # Create a sample stitched frame to get dimensions
        sample_real = np.zeros(frame_shape)
        sample_imag = np.zeros(frame_shape)
        sample_phase = np.zeros(frame_shape)
        sample_prob = np.zeros(frame_shape)
        
        real_colored = apply_colormap(sample_real, 'coolwarm', self.real_global_min, self.real_global_max).cpu().numpy()
        imag_colored = apply_colormap(sample_imag, 'plasma', self.imag_global_min, self.imag_global_max).cpu().numpy()
        phase_colored = apply_colormap(sample_phase, 'twilight', -np.pi, np.pi).cpu().numpy()
        prob_colored = apply_probability_colormap(sample_prob, self.log_prob_min, self.log_prob_max).cpu().numpy()
        
        stitched_frame = np.hstack([real_colored, imag_colored, phase_colored, prob_colored])
        video_height, video_width = stitched_frame.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'IYUV')  # Lossless codec
        self.video_writer = cv2.VideoWriter(self.output_file, fourcc, self.fps, (video_width, video_height))
        
        print(f"Video writer initialized: {video_width}x{video_height}, {self.fps} fps")
    
    def add_frame(self, frame_real, frame_imag, frame_phase, frame_prob):
        """Add a frame to the video (either sampling, storing, or writing)"""
        if self.video_writer is None:
            # Still in sampling phase
            self.sample_frame(frame_real, frame_imag, frame_phase, frame_prob)
            return
        
        # If we have stored frames and haven't written them yet, write them all first
        if self.keep_first_batch and self.stored_frames and not self.first_batch_written:
            print(f"Writing stored batch of {len(self.stored_frames)} frames to video...")
            for stored_frame in self.stored_frames:
                self._write_frame_to_video(stored_frame['real'], stored_frame['imag'], 
                                         stored_frame['phase'], stored_frame['prob'])
            
            # Clear stored frames to free memory
            self.stored_frames.clear()
            self.first_batch_written = True
            print("First batch written and cleared from memory")
        
        # Store frames for the first batch if enabled
        if self.keep_first_batch and not self.first_batch_written and len(self.stored_frames) < self.first_batch_size:
            self.stored_frames.append({
                'real': frame_real.copy(),
                'imag': frame_imag.copy(), 
                'phase': frame_phase.copy(),
                'prob': frame_prob.copy()
            })
            print(f"Stored frame {len(self.stored_frames)}/{self.first_batch_size} in memory")
            return
        
        # Write current frame normally (streaming mode after batch is written)
        self._write_frame_to_video(frame_real, frame_imag, frame_phase, frame_prob)
    
    def _write_frame_to_video(self, frame_real, frame_imag, frame_phase, frame_prob):
        """Internal method to write a single frame to the video"""
        # Apply colormaps with pre-calculated global ranges
        abs_imag = np.abs(frame_imag)
        real_colored = apply_colormap(frame_real, 'coolwarm', self.real_global_min, self.real_global_max).cpu().numpy()
        imag_colored = apply_colormap(abs_imag, 'plasma', self.imag_global_min, self.imag_global_max).cpu().numpy()
        phase_colored = apply_colormap(frame_phase, 'twilight', -np.pi, np.pi).cpu().numpy()
        
        # Apply log scale to probability
        log_prob = np.log10(np.maximum(frame_prob, 1e-10))
        prob_colored = apply_probability_colormap(log_prob, self.log_prob_min, self.log_prob_max, power=10).cpu().numpy()
        
        # Stitch frames horizontally
        stitched_frame = np.hstack([real_colored, imag_colored, phase_colored, prob_colored])
        
        # Write frame to video
        self.video_writer.write(stitched_frame)
        self.frame_count += 1
        
        if self.frame_count % 100 == 0:
            print(f"Written frame {self.frame_count}")
    
    def overwrite_first_batch(self, new_frames):
        """Overwrite the first batch of stored frames with new frames"""
        if not self.keep_first_batch:
            print("Warning: First batch storage is disabled, cannot overwrite")
            return False
            
        if self.first_batch_written:
            print("Warning: First batch already written to video, cannot overwrite")
            return False
            
        if len(new_frames) != len(self.stored_frames):
            print(f"Warning: New batch size ({len(new_frames)}) doesn't match stored batch size ({len(self.stored_frames)})")
            return False
        
        print(f"Overwriting first batch with {len(new_frames)} new frames")
        self.stored_frames = new_frames
        return True
    
    def finalize(self):
        """Close video writer and finish the video"""
        # Write any remaining stored frames before finalizing
        if self.keep_first_batch and self.stored_frames and not self.first_batch_written:
            print(f"Writing final batch of {len(self.stored_frames)} stored frames...")
            for stored_frame in self.stored_frames:
                self._write_frame_to_video(stored_frame['real'], stored_frame['imag'], 
                                         stored_frame['phase'], stored_frame['prob'])
            self.stored_frames.clear()
            
        if self.video_writer:
            self.video_writer.release()
            print(f"Video saved as {self.output_file}")
            print(f"Total frames written: {self.frame_count}")
        else:
            print("Warning: Video writer was never initialized (not enough frames sampled)")
    
    def get_memory_usage_estimate(self, frame_shape):
        """Estimate memory usage for the current configuration"""
        frame_size_bytes = np.prod(frame_shape) * 8  # float64
        frames_in_memory = self.first_batch_size if self.keep_first_batch else 0
        sample_frames_memory = self.sample_frames * np.prod(frame_shape) * 8 / 16  # subsampled
        
        total_memory_mb = (frames_in_memory * frame_size_bytes * 4 + sample_frames_memory) / (1024 * 1024)
        
        return {
            'frames_in_memory': frames_in_memory,
            'memory_per_frame_mb': frame_size_bytes * 4 / (1024 * 1024),  # 4 arrays per frame
            'total_memory_mb': total_memory_mb,
            'sample_memory_mb': sample_frames_memory / (1024 * 1024)
        }

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
