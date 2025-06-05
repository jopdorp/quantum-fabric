#!/usr/bin/env python3
"""
NumPy-Optimized Wave Interference Factorization with FFT Sweep

Key Features:
• FFT-based harmonic detection
• Automated parameter sweep
• Dynamic depth based on signal entropy with capped depth for large N
• Complex-valued modular signal modeling for interference patterns
"""

import numpy as np
import time
from math import gcd, log2
from numba import jit
from typing import List
from itertools import product
from sympy import randprime

# Generate random prime pairs for testing
def generate_test_case(bit_size):
    half_bits = bit_size // 2
    min_prime = 2 ** (half_bits - 1)
    max_prime = 2 ** half_bits - 1
    p = randprime(min_prime, max_prime)
    q = randprime(min_prime, max_prime)
    while q == p:
        q = randprime(min_prime, max_prime)
    return p * q, p, q

# Fast GCD
@jit(nopython=True)
def fast_gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Modular exponentiation sequence
@jit(nopython=True)
def fast_modmul_sequence(a, N, length):
    result = [0] * length
    x = a % N
    for i in range(length):
        result[i] = x
        x = (x * a) % N
    return result

# Estimate max depth conservatively for performance
def estimate_max_depth(N, cap=2**14):
    return min(max(int(N ** 0.25), 1024), cap)

# FFT resonance detection using complex-valued modular signals
def fft_resonance(sequence: List[int], N: int, window: int, step: int, top_k: int):
    best_results = []
    for start in range(0, len(sequence) - window, step):
        windowed = sequence[start:start + window]
        phased_signal = np.exp(2j * np.pi * np.array(windowed) / N)
        spectrum = np.abs(np.fft.fft(phased_signal))
        spectrum[0] = 0  # Remove DC
        top_indices = np.argpartition(spectrum, -top_k)[-top_k:]
        for idx in top_indices:
            if idx == 0:
                continue
            period = round(window / idx)
            best_results.append((period, spectrum[idx]))
    best_results.sort(key=lambda x: -x[1])
    return best_results[:4]

# Cache modexp sequences per base
sequence_cache = {}

def fft_wave_factor(N, a, max_depth, window=512, step=64, top_k=8):
    if max_depth < window + step:
        return None
    if a not in sequence_cache:
        sequence_cache[a] = fast_modmul_sequence(a, N, max_depth)
    sequence = sequence_cache[a]
    harmonics = fft_resonance(sequence, N, window, step, top_k)
    for period, amplitude in harmonics:
        if period <= 0:
            continue
        y = pow(a, period // 2, N)
        if y != 1 and y != N - 1:
            for delta in [-1, 1]:
                f = gcd(y + delta, N)
                if 1 < f < N:
                    print(f"[🎯] Factor via FFT harmonic resonance: {f} (period ~{period})")
                    return f
    return None

# Sweep optimized FFT configs with retry logic
def fft_sweep_factor(N):
    initial_depth = estimate_max_depth(N)
    print(f"[📐] Using max_depth = 2^{{{int(log2(initial_depth))}}} = {initial_depth} for N = {N}")
    windows = [256, 512, 1024] if initial_depth >= 2048 else [256, 512]
    steps = [32, 64]
    top_ks = [8, 16]
    bases = range(2, 40)

    for window, step, top_k, base in product(windows, steps, top_ks, bases):
        f = fft_wave_factor(N, base, max_depth=initial_depth, window=window, step=step, top_k=top_k)
        if f:
            print(f"✅ SUCCESS: {N} = {f} × {N // f}")
            return f

    if initial_depth < 2**14:
        retry_depth = min(initial_depth * 2, 2**14)
        print(f"🔁 Retrying with extended depth = {retry_depth}")
        for window, step, top_k, base in product(windows, steps, top_ks, bases):
            f = fft_wave_factor(N, base, max_depth=retry_depth, window=window, step=step, top_k=top_k)
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

    bit_sizes = [16, 20, 24, 28, 32, 36]

    for bit_size in bit_sizes:
        print(f"\n🎯 Generating RSA-{bit_size} test case...")
        N, p, q = generate_test_case(bit_size)
        print(f"N = {N:,}")
        print(f"Expected factors: {p} × {q}")

        start = time.time()
        result = fft_sweep_factor(N)
        elapsed = time.time() - start
        print(f"⏱️  Time: {elapsed:.3f}s")

        if result:
            other_factor = N // result
            if (result == p and other_factor == q) or (result == q and other_factor == p):
                print(f"🎉 CORRECT FACTORS FOUND!")
            else:
                print(f"⚠️  Found different factors: {result} × {other_factor}")
        print("-" * 50)

if __name__ == "__main__":
    test_fft_sweep()
