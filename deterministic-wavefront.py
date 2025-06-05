#!/usr/bin/env python3
"""
Wave Interference Factorization (v9.0)
Hybrid: Accumulating Interference Map + Deterministic Wavefront + Random Base Parallel Sweep
"""

import numpy as np
import time
from math import gcd, log2
from numba import jit
from sympy import randprime
from concurrent.futures import ThreadPoolExecutor, as_completed
from random import randint
import matplotlib.pyplot as plt

# Generate RSA test case

def generate_test_case(bit_size):
    half_bits = bit_size // 2
    min_prime = 2 ** (half_bits - 1)
    max_prime = 2 ** half_bits - 1
    p = randprime(min_prime, max_prime)
    q = randprime(min_prime, max_prime)
    while q == p:
        q = randprime(min_prime, max_prime)
    return p * q, p, q

# Modular exponentiation sequence
@jit(nopython=True)
def fast_modmul_sequence(a, N, length):
    result = [0] * length
    x = a % N
    for i in range(length):
        result[i] = x
        x = (x * a) % N
    return result

# Hann window

def hann_window(length):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(length) / length)

# Convert to complex wave

def complex_modular_signal(a, N, length):
    sequence = fast_modmul_sequence(a, N, length)
    signal = np.exp(2j * np.pi * np.array(sequence) / N)
    return signal * hann_window(length)

# Accumulating interference map with phase overlap sum

def accumulating_interference(signal: np.ndarray, max_lag: int):
    acc_map = np.zeros(max_lag, dtype=np.float64)
    for d in range(1, max_lag):
        shifted = np.roll(signal, -d)
        acc_map[d] += np.abs(np.sum(signal * np.conj(shifted)))
    return acc_map

# Factor via interference detection

def interference_factor(N, a, depth):
    signal = complex_modular_signal(a, N, depth)
    acc = accumulating_interference(signal, depth // 2)
    r = int(np.argmax(acc))
    if r <= 1:
        return None
    y = pow(a, r // 2, N)
    if y != 1 and y != N - 1:
        for delta in [-1, 1]:
            f = gcd(y + delta, N)
            if 1 < f < N:
                print(f"[🌊] Factor: {f} via interference map (r ~ {r})")
                return f
    return None

# Estimate depth heuristically

def estimate_max_depth(N, cap=2**18):
    return min(max(int(N ** 0.3), 8192), cap)

# Random base coprime to N

def random_coprime(N):
    while True:
        a = randint(2, N - 2)
        if gcd(a, N) == 1:
            return a

# Sweep over multiple bases with parallelism

def sweep_interference(N, timeout_sec=120, max_trials=90):
    def try_a(a, max_depth):
        return a, interference_factor(N, a, max_depth)

    max_depth = estimate_max_depth(N)
    print(f"[📐] max_depth = 2^{{{int(log2(max_depth))}}} = {max_depth} for N = {N}")

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(try_a, random_coprime(N), max_depth) for _ in range(max_trials)]
        for future in as_completed(futures):
            if time.time() - start_time > timeout_sec:
                print("🕒 Timeout")
                break
            a, result = future.result()
            if result:
                print(f"✅ SUCCESS: {N} = {result} × {N // result} (base {a})")
                for f in futures:
                    f.cancel()
                return result
    print("❌ No factor found")
    return None

# Test for RSA bit sizes

def test_interference():
    bit_sizes = list(range(16, 132, 4))
    times = []
    success = []

    for bit_size in bit_sizes:
        print(f"\n🔎 RSA-{bit_size} test case")
        N, p, q = generate_test_case(bit_size)
        print(f"N = {N:,}\nExpected: {p} × {q}")

        start = time.time()
        factor = sweep_interference(N)
        elapsed = time.time() - start

        times.append(elapsed)
        if factor in (p, q):
            print("🎉 Correct factor found")
            success.append(True)
        else:
            print("⚠️  Incorrect or no factor")
            success.append(False)

    # Graph times
    plt.figure()
    plt.semilogy(bit_sizes, times, marker='o')
    plt.xlabel('RSA Bit Size')
    plt.ylabel('Time (log seconds)')
    plt.title('Wave Interference Factorization (Accumulating Interference)')
    plt.grid(True)
    plt.show()

    for b, s in zip(bit_sizes, success):
        print(f"RSA-{b}: {'✅' if s else '❌'}")

if __name__ == "__main__":
    test_interference()
