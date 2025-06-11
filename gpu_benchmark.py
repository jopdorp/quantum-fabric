#!/usr/bin/env python3
"""
Benchmark Intel XPU vs CPU performance for wave propagation
"""
import numpy as np
import torch
import time
from torch_physics import propagate_wave_with_potential
from physics import propagate_wave_with_potential as original_propagate_wave_with_potential

# Configuration
TIME_DELTA = 0.01  # Time step for wave propagation

# Check if Intel XPU is available
def get_device():
    """Get the best available device (Intel XPU, CUDA, or CPU)"""
    if torch.xpu.is_available():
        print(f"Using Intel XPU: {torch.xpu.get_device_name()}")
        return torch.device("xpu")
    elif torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
        return torch.device("cuda")
    else:
        print("Using CPU with PyTorch")
        return torch.device("cpu")

# Global device
DEVICE = get_device()


def convert_to_device(tensor, device=DEVICE):
    # Convert inputs to torch tensors on device
    if isinstance(tensor, np.ndarray):
        converted_tensor = torch.from_numpy(tensor).to(device)
    else:
        converted_tensor = tensor.to(DEVICE)
    
    converted_tensor = converted_tensor.to(torch.complex64)
    return converted_tensor

def benchmark_wave_propagation():
    """Benchmark wave propagation on different devices and sizes."""
    print("=== Intel Arc GPU vs CPU Benchmark ===\n")
    
    # Test different problem sizes
    sizes = [64, 128, 256, 512, 1024, 2048]
    iterations = 30
    
    for size in sizes:
        print(f"Testing {size}x{size} wave propagation ({iterations} iterations)...")
        
        # Create test data
        x = np.linspace(-10, 10, size)
        y = np.linspace(-10, 10, size)
        X, Y = np.meshgrid(x, y)
        
        # Initial wave function (Gaussian wave packet)
        psi = np.exp(-0.5 * (X**2 + Y**2) + 1j * X)
        
        # Simple potential
        potential = 0.1 * (X**2 + Y**2)
        
        start_time = time.time()
        psi_result = convert_to_device(psi, device=torch.device("cpu"))
        potential_result = convert_to_device(potential, device=torch.device("cpu"))
        for i in range(iterations):
            psi_result = propagate_wave_with_potential(psi_result, potential_result, device=torch.device("cpu"))
        cpu_time = time.time() - start_time
        
        # Test on Intel XPU
        start_time = time.time()
        psi_result = convert_to_device(psi, device=torch.device("xpu"))
        potential_result = convert_to_device(potential, device=torch.device("xpu"))
        for i in range(iterations):
            psi_result = propagate_wave_with_potential(psi_result, potential_result, device=torch.device("xpu"))
        gpu_time = time.time() - start_time

        # Test on original implementation (CPU)
        start_time = time.time()
        for i in range(iterations):
            psi_result = original_propagate_wave_with_potential(psi, potential, dt=TIME_DELTA)
        
        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        
        print(f"  Original CPU time: {time.time() - start_time:.4f}s")
        print(f"  CPU time: {cpu_time:.4f}s")
        print(f"  XPU time: {gpu_time:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print()

if __name__ == "__main__":
    benchmark_wave_propagation()
