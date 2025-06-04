#!/usr/bin/env python3
"""
Wave Interference Factorization (v6)
Autocorrelation-Based Period Detection using Complex Modular Signals

Improvements:
• Switched to autocorrelation of complex modular exponentiation signals
• More robust to noise than FFT
• Dynamic max depth estimation with fallback
• Expanded sweep and adjusted max_depth for scaling
• Increased sweep range and minimum depth to handle RSA-28+
• Increased max_depth cap for RSA-32 and RSA-36 support
"""

import numpy as np
import time
from math import gcd, log2
from numba import jit
from sympy import randprime
from typing import List

# Generate random RSA test case of bit size
def generate_test_case(bit_size):
    half_bits = bit_size // 2
    min_prime = 2 ** (half_bits - 1)
    max_prime = 2 ** half_bits - 1
    p = randprime(min_prime, max_prime)
    q = randprime(min_prime, max_prime)
    while q == p:
        q = randprime(min_prime, max_prime)
    return p * q, p, q

# Generate modular exponentiation sequence
@jit(nopython=True)
def fast_modmul_sequence(a, N, length):
    result = [0] * length
    x = a % N
    for i in range(length):
        result[i] = x
        x = (x * a) % N
    return result

# Convert to complex-valued phase signal
def complex_modular_signal(a, N, length):
    sequence = fast_modmul_sequence(a, N, length)
    return np.exp(2j * np.pi * np.array(sequence) / N)

# Compute autocorrelation score by shifted dot product
def autocorrelation_phase(signal: np.ndarray, max_shift: int, top_k: int = 5):
    scores = []
    for d in range(1, max_shift):
        shifted = np.roll(signal, d)
        score = np.abs(np.sum(signal * np.conj(shifted)))
        scores.append((d, score))
    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]

# Heuristically estimate how deep to go
def estimate_max_depth(N, cap=2**17):
    return min(max(int(N ** 0.25), 4096), cap)

# Core algorithm using autocorrelation to find period r, then compute factors
def wave_autocorr_factor(N, a, max_depth):
    signal = complex_modular_signal(a, N, max_depth)
    peaks = autocorrelation_phase(signal, max_shift=max_depth // 2)
    for r, score in peaks:
        if r <= 0:
            continue
        y = pow(a, r // 2, N)
        if y != 1 and y != N - 1:
            for delta in [-1, 1]:
                f = gcd(y + delta, N)
                if 1 < f < N:
                    print(f"[🎯] Factor via wave autocorrelation: {f} (period ~{r})")
                    return f
    return None

# Try various a values up to 100
def sweep_wave_autocorr(N):
    max_depth = estimate_max_depth(N)
    print(f"[📐] Using max_depth = 2^{{{int(log2(max_depth))}}} = {max_depth} for N = {N}")
    for a in range(2, 100):
        f = wave_autocorr_factor(N, a, max_depth)
        if f:
            print(f"✅ SUCCESS: {N} = {f} × {N // f}")
            return f
    print("❌ FAILED: No factor found")
    return None

# Test for RSA-16 through RSA-36

def test_wave_autocorr():
    print("🌊 Wave Interference Factorization via Autocorrelation")
    print("=" * 64)
    bit_sizes = [16, 20, 24, 28, 32, 36]
    for bit_size in bit_sizes:
        print(f"\n🎯 Generating RSA-{bit_size} test case...")
        N, p, q = generate_test_case(bit_size)
        print(f"N = {N:,}")
        print(f"Expected factors: {p} × {q}")

        start = time.time()
        result = sweep_wave_autocorr(N)
        elapsed = time.time() - start
        print(f"⏱️  Time: {elapsed:.3f}s")

        if result:
            other = N // result
            if (result == p and other == q) or (result == q and other == p):
                print("🎉 CORRECT FACTORS FOUND!")
            else:
                print(f"⚠️  Incorrect factors: {result} × {other}")
        print("-" * 50)

if __name__ == "__main__":
    test_wave_autocorr()
