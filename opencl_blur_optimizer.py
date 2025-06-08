#!/usr/bin/env python3
"""OpenCL-accelerated in-place Gaussian blur for edge smoothing."""

import numpy as np
import pyopencl as cl

class OpenCLBlurOptimizer:
    def __init__(self):
        # Initialize OpenCL
        self.platforms = cl.get_platforms()
        self.context = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.context)
        self.program = None
        self.transition_mask_gpu = None
        
    def compile_kernels(self, width, height):
        """Compile OpenCL kernels for the given dimensions."""
        kernel_code = f"""
        __kernel void gaussian_blur_horizontal(
            __global float2* input,
            __global float2* output,
            __constant float* gaussian_kernel,
            const int kernel_size,
            const int width,
            const int height
        ) {{
            int gid_x = get_global_id(0);
            int gid_y = get_global_id(1);
            
            if (gid_x >= width || gid_y >= height) return;
            
            float2 sum = (float2)(0.0f, 0.0f);
            int half_kernel = kernel_size / 2;
            
            for (int i = 0; i < kernel_size; i++) {{
                int x = gid_x - half_kernel + i;
                // Handle boundary conditions with clamping
                x = clamp(x, 0, width - 1);
                
                int idx = gid_y * width + x;
                float2 pixel = input[idx];
                float weight = gaussian_kernel[i];
                
                sum.x += pixel.x * weight;
                sum.y += pixel.y * weight;
            }}
            
            int out_idx = gid_y * width + gid_x;
            output[out_idx] = sum;
        }}
        
        __kernel void gaussian_blur_vertical(
            __global float2* input,
            __global float2* output,
            __constant float* gaussian_kernel,
            const int kernel_size,
            const int width,
            const int height
        ) {{
            int gid_x = get_global_id(0);
            int gid_y = get_global_id(1);
            
            if (gid_x >= width || gid_y >= height) return;
            
            float2 sum = (float2)(0.0f, 0.0f);
            int half_kernel = kernel_size / 2;
            
            for (int i = 0; i < kernel_size; i++) {{
                int y = gid_y - half_kernel + i;
                // Handle boundary conditions with clamping
                y = clamp(y, 0, height - 1);
                
                int idx = y * width + gid_x;
                float2 pixel = input[idx];
                float weight = gaussian_kernel[i];
                
                sum.x += pixel.x * weight;
                sum.y += pixel.y * weight;
            }}
            
            int out_idx = gid_y * width + gid_x;
            output[out_idx] = sum;
        }}
        
        __kernel void apply_transition_mask(
            __global float2* original,
            __global float2* blurred,
            __global float* mask,
            __global float2* output,
            const int width,
            const int height
        ) {{
            int gid_x = get_global_id(0);
            int gid_y = get_global_id(1);
            
            if (gid_x >= width || gid_y >= height) return;
            
            int idx = gid_y * width + gid_x;
            float mask_val = mask[idx];
            float inv_mask = 1.0f - mask_val;
            
            float2 orig = original[idx];
            float2 blur = blurred[idx];
            
            output[idx].x = orig.x * inv_mask + blur.x * mask_val;
            output[idx].y = orig.y * inv_mask + blur.y * mask_val;
        }}
        """
        
        self.program = cl.Program(self.context, kernel_code).build()
        
        # Allocate GPU buffers
        self.width = width
        self.height = height
        self.size = width * height
        
        # Create buffers for complex numbers (as float2)
        mf = cl.mem_flags
        self.input_buf = cl.Buffer(self.context, mf.READ_WRITE, size=self.size * 8)  # 2 floats per complex
        self.temp_buf = cl.Buffer(self.context, mf.READ_WRITE, size=self.size * 8)
        self.output_buf = cl.Buffer(self.context, mf.READ_WRITE, size=self.size * 8)
        
    def create_gaussian_kernel(self, sigma, truncate=4.0):
        """Create 1D Gaussian kernel."""
        radius = int(truncate * sigma + 0.5)
        kernel_size = 2 * radius + 1
        
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / np.sum(kernel)  # Normalize
        
        return kernel, kernel_size
    
    def upload_transition_mask(self, transition_mask):
        """Upload the transition mask to GPU."""
        mask_flat = transition_mask.astype(np.float32).flatten()
        mf = cl.mem_flags
        self.transition_mask_gpu = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mask_flat)
    
    def blur_edges_opencl(self, psi, blur_strength=100.0, transition_mask=None):
        """GPU-accelerated blur_edges function."""
        if self.program is None:
            self.compile_kernels(psi.shape[1], psi.shape[0])
        
        if self.transition_mask_gpu is None and transition_mask is not None:
            self.upload_transition_mask(transition_mask)
        
        # Create Gaussian kernel
        sigma = blur_strength / 3
        kernel, kernel_size = self.create_gaussian_kernel(sigma)
        
        # Upload kernel to GPU
        mf = cl.mem_flags
        kernel_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=kernel)
        
        # Convert complex array to interleaved float array
        psi_flat = np.zeros((self.size, 2), dtype=np.float32)
        psi_flat[:, 0] = psi.real.flatten()
        psi_flat[:, 1] = psi.imag.flatten()
        
        # Upload input data
        cl.enqueue_copy(self.queue, self.input_buf, psi_flat)
        
        # Horizontal blur
        self.program.gaussian_blur_horizontal(
            self.queue, (self.width, self.height), None,
            self.input_buf, self.temp_buf, kernel_buf,
            np.int32(kernel_size), np.int32(self.width), np.int32(self.height)
        )
        
        # Vertical blur
        self.program.gaussian_blur_vertical(
            self.queue, (self.width, self.height), None,
            self.temp_buf, self.output_buf, kernel_buf,
            np.int32(kernel_size), np.int32(self.width), np.int32(self.height)
        )
        
        # Apply transition mask if provided
        if self.transition_mask_gpu is not None:
            self.program.apply_transition_mask(
                self.queue, (self.width, self.height), None,
                self.input_buf, self.output_buf, self.transition_mask_gpu, self.temp_buf,
                np.int32(self.width), np.int32(self.height)
            )
            result_buf = self.temp_buf
        else:
            result_buf = self.output_buf
        
        # Download result
        result_flat = np.zeros((self.size, 2), dtype=np.float32)
        cl.enqueue_copy(self.queue, result_flat, result_buf)
        
        # Convert back to complex array
        result = np.zeros(psi.shape, dtype=np.complex64)
        result.real = result_flat[:, 0].reshape(psi.shape)
        result.imag = result_flat[:, 1].reshape(psi.shape)
        
        return result
