#!/usr/bin/env python3 """ Wavefront Interference Factorization (Wavefront v1.0) Simulates a wave-propagation-like cellular computation for factoring without FFT. """

import numpy as np import time from math import gcd from sympy import randprime from typing import List from random import randint

--- Signal and Factorization Logic ---

def generate_test_case(bit_size): half_bits = bit_size // 2 min_prime = 2 ** (half_bits - 1) max_prime = 2 ** half_bits - 1 p = randprime(min_prime, max_prime) q = randprime(min_prime, max_prime) while q == p: q = randprime(min_prime, max_prime) return p * q, p, q

def wavefront_sequence(a: int, N: int, length: int) -> np.ndarray: buffer = np.zeros(length, dtype=complex) x = a % N for i in range(length): phase = 2 * np.pi * x / N buffer[i] = np.exp(1j * phase) x = (x * a) % N return buffer

def detect_wavefront_interference(signal: np.ndarray, threshold=5.0): scores = [] for d in range(1, len(signal) // 2): s = np.sum(signal[:len(signal)-d] * np.conj(signal[d:])) scores.append((d, abs(s))) baseline = np.median([s for _, s in scores]) peaks = [(d, s) for d, s in scores if s > threshold * baseline] return sorted(peaks, key=lambda x: -x[1])

def wavefront_factor(N: int, a: int, length: int): signal = wavefront_sequence(a, N, length) peaks = detect_wavefront_interference(signal) for r, _ in peaks: y = pow(a, r // 2, N) if y != 1 and y != N - 1: for delta in [-1, 1]: f = gcd(y + delta, N) if 1 < f < N: print(f"[🎯] Factor via wavefront: {f} (period ~{r})") return f return None

def test_wavefront(): print("🌊 Wavefront-Based Factorization via Phase Interference") print("=" * 64) bit_sizes = [16, 20, 24, 28] for bit_size in bit_sizes: print(f"\n🎯 Generating RSA-{bit_size} test case…") N, p, q = generate_test_case(bit_size) print(f"N = {N:,}") print(f"Expected factors: {p} × {q}")

a = randint(2, N - 2)
    print(f"[🔁] Using base a = {a}")
    start = time.time()
    factor = wavefront_factor(N, a, length=8192)
    elapsed = time.time() - start
    print(f"⏱️  Time: {elapsed:.2f}s")

    if factor:
        other = N // factor
        if (factor == p and other == q) or (factor == q and other == p):
            print("🎉 CORRECT FACTORS FOUND!")
        else:
            print(f"⚠️  Incorrect factors: {factor} × {other}")
    else:
        print("❌ FAILED: No factor found")
    print("-" * 50)

if name == "main": test_wavefront()
