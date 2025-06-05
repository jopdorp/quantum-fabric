#!/usr/bin/env python3
"""
Wave Interference Factorization (v10.0)
Fixed-Rule Fabric + Persistent Cell State + Feedback + Spatial Memory
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

# Cell structure with persistent state
def initialize_cells(length):
    return [{'amplitude': 0.0, 'phase': 0.0, 'hits': 0} for _ in range(length)]

def update_cells(cells, signal):
    for i in range(len(cells)):
        amp = np.abs(signal[i])
        phase = np.angle(signal[i])
        cells[i]['amplitude'] += amp
        cells[i]['phase'] += phase
        cells[i]['hits'] += 1

def compute_feedback_signal(cells):
    signal = np.zeros(len(cells), dtype=np.complex128)
    for i, cell in enumerate(cells):
        if cell['hits'] > 0:
            avg_amp = cell['amplitude'] / cell['hits']
            avg_phase = cell['phase'] / cell['hits']
            signal[i] = avg_amp * np.exp(1j * avg_phase)
    return signal

# Convert to complex-valued signal with phase bump
def complex_modular_signal(a, N, length):
    sequence = fast_modmul_sequence(a, N, length)
    return np.exp(2j * np.pi * np.array(sequence) / N)

# Interference-based peak detection
def detect_peaks(signal, max_lag, min_peak_ratio=2.5, top_k=5):
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

# Heuristically estimate how deep to go
def estimate_max_depth(N, cap=2**18):
    return min(max(int(N ** 0.3), 8192), cap)

# Core algorithm
def wave_fabric_factor(N, a, max_depth):
    cells = initialize_cells(max_depth)
    for round in range(3):
        signal = complex_modular_signal(a, N, max_depth)
        update_cells(cells, signal)
        signal = compute_feedback_signal(cells)
        peaks = detect_peaks(signal, max_lag=max_depth // 2)
        for r, score in peaks:
            if r <= 0:
                continue
            y = pow(a, r // 2, N)
            if y != 1 and y != N - 1:
                for delta in [-1, 1]:
                    f = gcd(y + delta, N)
                    if 1 < f < N:
                        print(f"[🎯] Factor via wave fabric: {f} (period ~{r})")
                        return f
    return None

# Pick random base coprime with N
def random_coprime(N):
    while True:
        a = randint(2, N - 2)
        if gcd(a, N) == 1:
            return a

# Try various a values
def sweep_wave_fabric(N, timeout_sec=180, max_trials=90):
    def try_a(a, max_depth):
        return a, wave_fabric_factor(N, a, max_depth)

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

    print("❌ FAILED: No factor found")
    return None

# Test for RSA-64 and RSA-128
def test_wave_fabric():
    print("🌊 Wave Fabric Factorization with Persistent State")
    print("=" * 64)
    for bit_size in range(16, 129, 4):
        print(f"\n🎯 Generating RSA-{bit_size} test case…")
        N, p, q = generate_test_case(bit_size)
        print(f"N = {N:,}")
        print(f"Expected factors: {p} × {q}")

        start = time.time()
        factor = sweep_wave_fabric(N)
        elapsed = time.time() - start
        print(f"⏱️  Time: {elapsed:.2f}s")

        if factor:
            other = N // factor
            if (factor == p and other == q) or (factor == q and other == p):
                print("🎉 CORRECT FACTORS FOUND!")
            else:
                print(f"⚠️  Incorrect factors: {factor} × {other}")
        print("-" * 50)

if __name__ == "__main__":
    test_wave_fabric()
