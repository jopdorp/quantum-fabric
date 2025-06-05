#!/usr/bin/env python3
"""
Wave Interference Factorization (v9.0)
Hybrid FFT Bump Detection + Phase Autocorrelation + Adaptive Depth Scaling
"""

import numpy as np
import time
from math import gcd, log2
from numba import jit
from sympy import randprime
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError
from random import randint

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

# Hann window for smoothing
def hann_window(length):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(length) / length)

# Convert to complex-valued phase signal
def complex_modular_signal(a, N, length):
    sequence = fast_modmul_sequence(a, N, length)
    signal = np.exp(2j * np.pi * np.array(sequence) / N)
    return signal * hann_window(length)

# FFT bump detector
def fft_bump_candidates(signal: np.ndarray, top_k: int = 8):
    spectrum = np.fft.fft(signal)
    magnitudes = np.abs(spectrum[:len(spectrum)//2])
    phase_angles = np.angle(spectrum[:len(spectrum)//2])
    indices = np.argpartition(-magnitudes, top_k)[:top_k]
    return sorted([(i, magnitudes[i], phase_angles[i]) for i in indices], key=lambda x: -x[1])

# Sliding autocorrelation to refine bumps
def sliding_autocorrelation(signal: np.ndarray, max_lag: int, min_peak_ratio=2.5, top_k=5):
    L = len(signal)
    energy = np.sum(np.abs(signal) ** 2)
    scores = []
    ref = signal
    for d in range(1, max_lag):
        shifted = np.empty(L, dtype=np.complex128)
        shifted[:L-d] = signal[d:]
        shifted[L-d:] = 0
        score = np.abs(np.vdot(ref, shifted))
        scores.append((d, score))
    baseline = np.median([s for _, s in scores])
    top = [(d, s) for d, s in scores if s > baseline * min_peak_ratio]
    return sorted(top, key=lambda x: -x[1])[:top_k]

# Estimate how deep to go
def estimate_max_depth(N, cap=2**18):
    return min(max(int(N ** 0.3), 8192), cap)

# Factor using hybrid FFT bump + autocorrelation
def wave_autocorr_factor(N, a, max_depth):
    for depth in [max_depth, max_depth * 2]:
        signal = complex_modular_signal(a, N, depth)
        bump_peaks = fft_bump_candidates(signal, top_k=10)
        refined_peaks = sliding_autocorrelation(signal, max_lag=depth // 2)
        for r, score in refined_peaks:
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

# Pick random base coprime with N
def random_coprime(N):
    while True:
        a = randint(2, N - 2)
        if gcd(a, N) == 1:
            return a

# Try various a values with timeout and fallback
def sweep_wave_autocorr(N, timeout_sec=120, max_trials=90):
    def try_a(a, max_depth):
        return a, wave_autocorr_factor(N, a, max_depth)

    max_depth = estimate_max_depth(N)
    print(f"[📐] Using max_depth = 2^{{{int(log2(max_depth))}}} = {max_depth} for N = {N}")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(try_a, random_coprime(N), max_depth) for _ in range(max_trials)]
        try:
            for future in as_completed(futures):
                if time.time() - start_time > timeout_sec:
                    print("🕒 Timeout reached")
                    break
                a, result = future.result()
                if result:
                    print(f"✅ SUCCESS: {N} = {result} × {N // result} (base {a})")
                    for f in futures:
                        f.cancel()
                    return result
        except CancelledError:
            pass
    
    print("⚠️ First sweep failed – expanding depth …")
    max_depth = min(max_depth * 2, 2**20)
    
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(try_a, random_coprime(N), max_depth) for _ in range(max_trials)]
        try:
            for future in as_completed(futures):
                if time.time() - start_time > timeout_sec * 2:
                    print("🕒 Timeout reached on retry")
                    break
                a, result = future.result()
                if result:
                    print(f"✅ SUCCESS: {N} = {result} × {N // result} (base {a})")
                    for f in futures:
                        f.cancel()
                    return result
        except CancelledError:
            pass
    
    print("❌ FAILED: No factor found")
    return None

# Test for RSA-16 through RSA-128
def test_wave_autocorr():
    print("🌊 Wave Interference Factorization via Hybrid Phase Detection")
    print("=" * 64)
    bit_sizes = list(range(16, 132, 4))
    
    for bit_size in bit_sizes:
        print(f"\n🎯 Generating RSA-{bit_size} test case…")
        N, p, q = generate_test_case(bit_size)
        print(f"N = {N:,}")
        print(f"Expected factors: {p} × {q}")

        start = time.time()
        factor = sweep_wave_autocorr(N)
        elapsed = time.time() - start
        print(f"⏱️ Time: {elapsed:.2f}s")
        
        if factor:
            other = N // factor
            if (factor == p and other == q) or (factor == q and other == p):
                print("🎉 CORRECT FACTORS FOUND!")
            else:
                print(f"⚠️ Incorrect factors: {factor} × {other}")
        print("-" * 50)

if __name__ == "__main__":
    test_wave_autocorr()
