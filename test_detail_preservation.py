#!/usr/bin/env python3
"""
Test script to compare detail preservation in different FFT stabilization methods.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the current directory to Python path
sys.path.append('/home/jopdorp/development/waverfont-computer')

from torch_physics import (
    WavePropagationModel, 
    LaplacianWaveModel, 
    HybridWaveModel,
    DetailPreservingFFTModel,
    get_device
)

def create_test_wavefunction(size=256):
    """Create a wavefunction with rich frequency content"""
    device = get_device()
    
    # Create coordinate grids
    x = torch.linspace(-10, 10, size, device=device)
    y = torch.linspace(-10, 10, size, device=device)
    X, Y = torch.meshgrid(y, x, indexing='ij')
    
    # Create a multi-scale wavefunction with:
    # 1. Low frequency Gaussian envelope
    # 2. Medium frequency oscillation
    # 3. High frequency details
    
    # Main envelope (low frequency)
    main_envelope = torch.exp(-0.2 * (X**2 + Y**2))
    
    # Medium frequency wave packet
    medium_freq = torch.exp(1j * (3*X + 2*Y))
    
    # High frequency details (interference pattern)
    high_freq_detail = 0.3 * torch.exp(1j * (12*X + 8*Y)) * torch.exp(-2 * ((X-1)**2 + (Y+1)**2))
    
    # Fine-scale oscillations
    fine_detail = 0.1 * torch.cos(20*X) * torch.cos(15*Y) * torch.exp(-0.5 * (X**2 + Y**2))
    
    # Combine all components
    psi = main_envelope * medium_freq + high_freq_detail + 1j * fine_detail
    
    # Normalize
    norm = torch.sqrt(torch.sum(torch.abs(psi)**2))
    psi = psi / norm
    
    return psi, X, Y

def create_test_potential(X, Y):
    """Create a simple harmonic potential"""
    return 0.02 * (X**2 + Y**2)

def analyze_frequency_content(psi, name=""):
    """Analyze the frequency content of a wavefunction"""
    psi_k = torch.fft.fft2(psi)
    power_spectrum = torch.abs(psi_k)**2
    
    # Get k-space coordinates
    kx = torch.fft.fftfreq(psi.shape[1], device=psi.device) * (2 * torch.pi)
    ky = torch.fft.fftfreq(psi.shape[0], device=psi.device) * (2 * torch.pi)
    KY, KX = torch.meshgrid(ky, kx, indexing='ij')
    k_magnitude = torch.sqrt(KX**2 + KY**2)
    
    # Frequency bins
    k_max = torch.pi
    low_k = k_max * 0.3
    mid_k = k_max * 0.7
    
    # Calculate power in different frequency ranges
    total_power = torch.sum(power_spectrum)
    low_freq_power = torch.sum(power_spectrum[k_magnitude <= low_k])
    mid_freq_power = torch.sum(power_spectrum[(k_magnitude > low_k) & (k_magnitude <= mid_k)])
    high_freq_power = torch.sum(power_spectrum[k_magnitude > mid_k])
    
    print(f"{name} Frequency Analysis:")
    print(f"  Low freq (0-30% k_max):  {100*low_freq_power/total_power:.1f}%")
    print(f"  Mid freq (30-70% k_max): {100*mid_freq_power/total_power:.1f}%")
    print(f"  High freq (70-100% k_max): {100*high_freq_power/total_power:.1f}%")
    print(f"  Total power: {total_power.item():.6f}")
    
    return {
        'low_freq_frac': (low_freq_power/total_power).item(),
        'mid_freq_frac': (mid_freq_power/total_power).item(),
        'high_freq_frac': (high_freq_power/total_power).item(),
        'total_power': total_power.item()
    }

def compare_methods(psi_initial, potential, time_steps=50, dt=0.5):
    """Compare different propagation methods"""
    device = psi_initial.device
    shape = psi_initial.shape
    
    print(f"\n=== Comparing Propagation Methods ===")
    print(f"Grid size: {shape}")
    print(f"Time steps: {time_steps}")
    print(f"Time delta: {dt}")
    print(f"Total evolution time: {time_steps * dt}")
    
    # Initialize all models
    models = {
        'Pure FFT': WavePropagationModel(shape, dt, device),
        'Detail Preserving': DetailPreservingFFTModel(shape, dt, device, "detail_preserving"),
        'Selective Damping': DetailPreservingFFTModel(shape, dt, device, "selective_damping"),
        'Adaptive Hybrid': HybridWaveModel(shape, dt, device, "adaptive"),
        'Conservative Hybrid': HybridWaveModel(shape, dt, device, "conservative"),
        'Laplacian': LaplacianWaveModel(shape, dt, device)
    }
    
    # Initialize wavefunctions
    psi_dict = {name: psi_initial.clone() for name in models.keys()}
    
    # Store initial frequency content
    print(f"\n=== Initial State ===")
    initial_analysis = analyze_frequency_content(psi_initial, "Initial")
    
    # Evolve each method
    print(f"\n=== Evolution Results ===")
    for step in range(time_steps):
        for name, model in models.items():
            psi_dict[name] = model(psi_dict[name], potential)
            
        # Check for stability issues
        if step % 10 == 0:
            for name, psi in psi_dict.items():
                norm = torch.sqrt(torch.sum(torch.abs(psi)**2))
                max_val = torch.max(torch.abs(psi))
                if torch.isnan(psi).any() or torch.isinf(psi).any() or norm > 2.0:
                    print(f"WARNING: {name} became unstable at step {step} (norm={norm:.3f}, max={max_val:.3f})")
    
    # Final analysis
    print(f"\n=== Final Analysis ===")
    results = {}
    reference_psi = psi_dict['Pure FFT']  # Use pure FFT as reference
    
    for name, psi_final in psi_dict.items():
        print(f"\n--- {name} ---")
        
        # Check if stable
        norm = torch.sqrt(torch.sum(torch.abs(psi_final)**2))
        if torch.isnan(psi_final).any() or torch.isinf(psi_final).any():
            print("  UNSTABLE: Contains NaN or Inf values")
            continue
        elif norm > 3.0 or norm < 0.1:
            print(f"  UNSTABLE: Norm = {norm:.3f}")
            continue
        
        # Frequency content analysis
        freq_analysis = analyze_frequency_content(psi_final, name)
        
        # Compare to pure FFT (if stable)
        if not (torch.isnan(reference_psi).any() or torch.isinf(reference_psi).any()):
            # Calculate fidelity
            overlap = torch.abs(torch.sum(torch.conj(reference_psi) * psi_final))
            ref_norm = torch.sqrt(torch.sum(torch.abs(reference_psi)**2))
            psi_norm = torch.sqrt(torch.sum(torch.abs(psi_final)**2))
            fidelity = overlap / (ref_norm * psi_norm)
            
            # Calculate detail preservation
            amplitude_diff = torch.mean(torch.abs(torch.abs(psi_final) - torch.abs(reference_psi)))
            phase_diff = torch.mean(torch.abs(torch.angle(psi_final) - torch.angle(reference_psi)))
            
            print(f"  Fidelity vs Pure FFT: {fidelity:.4f}")
            print(f"  Amplitude difference: {amplitude_diff:.6f}")
            print(f"  Phase difference: {phase_diff:.4f}")
        
        results[name] = {
            'psi': psi_final,
            'freq_analysis': freq_analysis,
            'norm': norm.item(),
            'stable': True
        }
    
    return results

def main():
    print("=== FFT Detail Preservation Test ===")
    
    # Create test case
    psi, X, Y = create_test_wavefunction(size=128)  # Smaller for faster testing
    potential = create_test_potential(X, Y)
    
    # Test with moderate time step first
    print("\n--- Testing with moderate time step (dt=0.2) ---")
    results_moderate = compare_methods(psi, potential, time_steps=25, dt=0.2)
    
    # Test with larger time step
    print("\n\n--- Testing with large time step (dt=1.0) ---")
    results_large = compare_methods(psi, potential, time_steps=25, dt=1.0)
    
    # Test with very large time step
    print("\n\n--- Testing with very large time step (dt=4.0) ---")
    results_huge = compare_methods(psi, potential, time_steps=10, dt=4.0)
    
    print("\n=== Summary ===")
    print("The detail-preserving FFT models should maintain more high-frequency")
    print("content compared to the hybrid and Laplacian methods, especially")
    print("with larger time steps where numerical instability becomes an issue.")

if __name__ == "__main__":
    main()
