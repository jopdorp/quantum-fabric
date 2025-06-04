#!/usr/bin/env python3
"""
NumPy-Optimized Wave Interference Factorization with FFT Sweep

Key Features:
• FFT-based harmonic detection
• Automated parameter sweep
• Dynamic depth based on signal entropy with capped depth for large N
"""

import numpy as np
import time
import random
from math import gcd, log2
from numba import jit
from typing import List
from itertools import product
from sympy import nextprime, randprime

# Generate random prime pairs for testing
def generate_test_case(bit_size):
    """Generate a random RSA-like number with two primes of approximately bit_size/2 bits each"""
    # Calculate approximate range for each prime
    half_bits = bit_size // 2
    min_prime = 2 ** (half_bits - 1)
    max_prime = 2 ** half_bits - 1
    
    # Generate two random primes
    p = randprime(min_prime, max_prime)
    q = randprime(min_prime, max_prime)
    
    # Ensure they're different
    while q == p:
        q = randprime(min_prime, max_prime)
    
    N = p * q
    return N, p, q

# Fast GCD
@jit(nopython=True)
def fast_gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Modular exponentiation sequence
def fast_modmul_sequence(a, N, length):
    result = []
    x = a % N
    for _ in range(length):
        result.append(x)
        x = (x * a) % N
    return result

# Estimate max depth based on signal hash resolution with cap
def estimate_max_depth(N, cap=2**16):
    # D(n) = N^{1/4}, capped to avoid excessive computation
    return min(max(int(N ** 0.25), 2**15), cap)

# FFT resonance detection
def fft_resonance(sequence: List[int], N: int, window: int, step: int, top_k: int):
    results = []
    for start in range(0, len(sequence) - window, step):
        windowed = sequence[start:start+window]
        windowed_array = np.array(windowed, dtype=np.float64)
        fft = np.abs(np.fft.fft(windowed_array - np.mean(windowed_array)))
        fft[0] = 0
        top_indices = np.argpartition(fft, -top_k)[-top_k:]
        for idx in top_indices:
            period = round(len(windowed) / idx) if idx != 0 else 0
            results.append((idx, period, fft[idx]))
    results.sort(key=lambda x: -x[2])
    return results

# FFT-based factoring attempt
def fft_wave_factor(N, a, max_depth, window=512, step=128, top_k=8):
    sequence = fast_modmul_sequence(a, N, max_depth)
    harmonics = fft_resonance(sequence, N, window, step, top_k)
    for idx, period, amplitude in harmonics:
        if period <= 0: continue
        y = pow(a, period // 2, N)
        if y != 1 and y != N - 1:
            for delta in [-1, 1]:
                f = gcd(y + delta, N)
                if 1 < f < N:
                    print(f"[🎯] Factor via FFT harmonic resonance: {f} (period ~{period})")
                    return f
    return None

# Sweep multiple FFT configs
def fft_sweep_factor(N):
    max_depth = estimate_max_depth(N)
    print(f"[📐] Using max_depth = 2^{{{int(log2(max_depth))}}} = {max_depth} for N = {N}")
    windows = [512, 1024, 2048, 4096, 8192]
    steps = [64, 128, 256, 512, 1024]
    top_ks = [8, 16, 32, 64, 128]
    bases = range(2, 32)

    for window, step, top_k, base in product(windows, steps, top_ks, bases):
        f = fft_wave_factor(N, base, max_depth=max_depth, window=window, step=step, top_k=top_k)
        if f:
            print(f"✅ SUCCESS: {N} = {f} × {N // f}")
            return f
    print("❌ FAILED: No factor found")
    return None

# Test harness
def test_fft_sweep():
    print("🌊 NumPy-Optimized Wave Interference Factorization")
    print("=" * 60)
    print("Key Features:")
    print("• FFT-based harmonic detection")
    print("• Automated parameter sweep")
    print("• Dynamic depth based on hash resolution")
    print("• Dynamically generated prime test cases")

    # Test cases with increasing bit sizes
    bit_sizes = [16, 20, 24, 32, 40, 48, 56, 64]
    
    for bit_size in bit_sizes:
        print(f"\n🎯 Generating RSA-{bit_size} test case...")
        N, p, q = generate_test_case(bit_size)
        print(f"N = {N:,}")
        print(f"Expected factors: {p} × {q}")
        
        start = time.time()
        result = fft_sweep_factor(N)
        elapsed = time.time() - start
        print(f"⏱️  Time: {elapsed:.3f}s")
        
        # Verify if we found the correct factors
        if result:
            other_factor = N // result
            if (result == p and other_factor == q) or (result == q and other_factor == p):
                print(f"🎉 CORRECT FACTORS FOUND!")
            else:
                print(f"⚠️  Found different factors: {result} × {other_factor}")
        print("-" * 50)

if __name__ == "__main__":
    test_fft_sweep()
