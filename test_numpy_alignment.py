#!/usr/bin/env python3
"""
Test script to debug the numpy alignment score issue
"""

import numpy as np
from numpy_wave_interference_factorization import detect_interference_matrix, vectorized_phase_alignment

def test_alignment_scoring():
    """Test the alignment scoring function with various scenarios."""
    print("🔍 Testing NumPy Alignment Score Calculation")
    print("=" * 50)
    
    N = 77  # Test with N = 7 × 11
    
    # Test case 1: Identical values (should score 1.0)
    print("\n1. Testing identical values:")
    wave_buffer = [42, 35, 49, 42, 21, 35]  # Contains repeated values
    found, pos1, pos2, score = detect_interference_matrix(wave_buffer, len(wave_buffer), N, threshold=0.5)
    print(f"   Buffer: {wave_buffer}")
    print(f"   Result: found={found}, pos1={pos1}, pos2={pos2}, score={score:.6f}")
    
    # Test case 2: Close values (should score high but not 1.0)
    print("\n2. Testing close values:")
    wave_buffer = [42, 35, 49, 41, 21, 36]  # Similar but not identical
    found, pos1, pos2, score = detect_interference_matrix(wave_buffer, len(wave_buffer), N, threshold=0.5)
    print(f"   Buffer: {wave_buffer}")
    print(f"   Result: found={found}, pos1={pos1}, pos2={pos2}, score={score:.6f}")
    
    # Test case 3: Very different values (should score low)
    print("\n3. Testing different values:")
    wave_buffer = [42, 35, 49, 5, 21, 70]  # Very different values
    found, pos1, pos2, score = detect_interference_matrix(wave_buffer, len(wave_buffer), N, threshold=0.5)
    print(f"   Buffer: {wave_buffer}")
    print(f"   Result: found={found}, pos1={pos1}, pos2={pos2}, score={score:.6f}")
    
    # Test case 4: Check the vectorized function directly
    print("\n4. Testing vectorized_phase_alignment directly:")
    values = np.array([42, 35, 49, 42, 21, 35], dtype=np.float64)
    i_indices, j_indices, scores = vectorized_phase_alignment(values, N)
    
    print(f"   Values: {values}")
    print(f"   Pairwise scores:")
    for i, j, score in zip(i_indices, j_indices, scores):
        print(f"     {values[i]:4.0f} vs {values[j]:4.0f} -> score = {score:.6f}")
    
    # Test case 5: Edge case with same values at different scales
    print("\n5. Testing phase scoring math:")
    test_pairs = [
        (0, 0, "identical values"),
        (1, 1, "identical values"),
        (42, 42, "identical values"),
        (42, 43, "diff=1, N=77"),
        (42, 35, "diff=7, N=77"),
        (42, 5, "diff=37, N=77"),
        (42, 70, "diff=28, N=77"),
    ]
    
    for val1, val2, desc in test_pairs:
        diff = abs(val1 - val2)
        norm_diff = diff / N
        if norm_diff <= 0.5:
            score = 1.0 - (2.0 * norm_diff)
        else:
            score = 2.0 * (1.0 - norm_diff)
        score = max(0.0, min(1.0, score))
        print(f"   {val1:2d} vs {val2:2d}: diff={diff:2d}, norm={norm_diff:.4f}, score={score:.6f} ({desc})")

def test_wave_sequence_realistic():
    """Test with a realistic wave sequence to see scoring patterns."""
    print("\n" + "="*50)
    print("🌊 Testing with Realistic Wave Sequence")
    print("="*50)
    
    N = 77
    a = 2
    
    # Generate a small wave sequence
    wave_buffer = []
    x = a
    for k in range(20):
        x = (x * a) % N
        wave_buffer.append(x)
    
    print(f"Wave sequence for a={a}, N={N}:")
    print(f"Buffer: {wave_buffer}")
    
    # Test detection at various thresholds
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
    for threshold in thresholds:
        found, pos1, pos2, score = detect_interference_matrix(wave_buffer, len(wave_buffer), N, threshold)
        print(f"Threshold {threshold}: found={found}, score={score:.6f}")

if __name__ == "__main__":
    test_alignment_scoring()
    test_wave_sequence_realistic()
