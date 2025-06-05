#!/usr/bin/env python3
"""
Wave Interference Factorization (v9.0 Hybrid)
Hybrid: Deterministic Wavefront + Accumulated Interference + FFT Bump Smoothing
"""

import numpy as np
import time
from math import gcd, log2
from numba import jit
from sympy import randprime
from random import randint
from concurrent.futures import ThreadPoolExecutor, as_completed

# Generate random RSA test case
def generate_test_case(bit_size):
    half_bits = bit_size // 2
    min_prime = 2 ** (half_bits - 1)
    max_prime = 2 ** half_bits - 1
    p = randprime(min_prime, max_prime)
    q = randprime(min_prime, max_prime)
    while q == p:
        q = randprime(min_prime, max_prime)
    return p * q, p, q

@jit(nopython=True)
def fast_modmul_sequence(a, N, length):
    result = [0] * length
    x = a % N
    for i in range(length):
        result[i] = x
        x = (x * a) % N
    return result

def hann_window(length):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(length) / length)

def complex_modular_signal(a, N, length):
    sequence = fast_modmul_sequence(a, N, length)
    signal = np.exp(2j * np.pi * np.array(sequence) / N)
    return signal * hann_window(length)

def sliding_autocorrelation(signal: np.ndarray, max_lag: int, min_peak_ratio=2.5, top_k=5):
    L = len(signal)
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

def fft_bump(signal: np.ndarray):
    fft = np.fft.fft(signal)
    fft = fft * np.exp(-np.abs(np.fft.fftfreq(len(signal))) * 10)
    return np.fft.ifft(fft)

def estimate_max_depth(N, cap=2**20):
    return min(max(int(N ** 0.3), 8192), cap)

def wave_autocorr_factor(N, a, max_depth):
    for depth in [max_depth, max_depth * 2]:
        signal = complex_modular_signal(a, N, depth)
        signal = fft_bump(signal)
        peaks = sliding_autocorrelation(signal, max_lag=depth // 2)
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

def random_coprime(N):
    while True:
        a = randint(2, N - 2)
        if gcd(a, N) == 1:
            return a

def sweep_wave_autocorr(N, timeout_sec=240, max_trials=120):
    def try_a(a, max_depth):
        return a, wave_autocorr_factor(N, a, max_depth)

    max_depth = estimate_max_depth(N)
    print(f"[📐] Using max_depth = 2^{{{int(log2(max_depth))}}} = {max_depth} for N = {N}")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(try_a, random_coprime(N), max_depth) for _ in range(max_trials)]
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
    print("❌ FAILED: No factor found")
    return None

def test_wave_autocorr():
    print("🌊 Hybrid Wave Interference Factorization")
    print("=" * 60)
    bit_sizes = list(range(12, 304, 8))
    for bit_size in bit_sizes:
        print(f"\n🎯 Generating RSA-{bit_size} test case…")
        N, p, q = generate_test_case(bit_size)
        print(f"N = {N:,}")
        print(f"Actual factors: {p:,} × {q:,}")
        
        start_time = time.time()
        result = sweep_wave_autocorr(N)
        elapsed = time.time() - start_time
        
        if result:
            print(f"⏱️  Time: {elapsed:.1f}s")
            print(f"✅ PASS: Found factor {result:,}")
        else:
            print(f"⏱️  Time: {elapsed:.1f}s")
            print(f"❌ FAIL: No factor found")
        print("-" * 60)

if __name__ == "__main__":
    test_wave_autocorr()
