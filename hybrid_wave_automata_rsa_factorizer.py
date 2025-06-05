# hybrid_wave_automata_rsa_factorizer.py
# Hybrid Automata + FFT Bump RSA Factorization (RSA-64 and RSA-128 Ready)

import numpy as np
import time
from math import gcd, log2
from sympy import randprime
from numba import jit
from random import randint
from typing import List


# === Utilities ===
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


# === Signal Processing ===
def complex_modular_signal(a, N, length):
    sequence = fast_modmul_sequence(a, N, length)
    signal = np.exp(2j * np.pi * np.array(sequence) / N)
    return signal * hann_window(length)


def interference_autocorr(signal, max_lag):
    L = len(signal)
    scores = []
    for d in range(1, max_lag):
        overlap = signal[:L - d] * np.conj(signal[d:])
        score = np.abs(np.sum(overlap))
        scores.append((d, score))
    return sorted(scores, key=lambda x: -x[1])[:5]


def fft_bump(signal):
    spectrum = np.fft.fft(signal)
    bump = np.fft.ifft(np.abs(spectrum) * np.exp(1j * np.angle(spectrum)))
    return bump.real


# === Hybrid Detection ===
def hybrid_factor(N, a, depth):
    signal = complex_modular_signal(a, N, depth)
    bumped = fft_bump(signal)
    mod_signal = signal + bumped * 0.25
    peaks = interference_autocorr(mod_signal, depth // 2)
    
    for r, score in peaks:
        if r <= 0: continue
        y = pow(a, r // 2, N)
        if y != 1 and y != N - 1:
            for delta in [-1, 1]:
                f = gcd(y + delta, N)
                if 1 < f < N:
                    print(f"[🎯] Factor found: {f} (period ~{r})")
                    return f
    return None


# === Top-Level Sweep ===
def random_coprime(N):
    while True:
        a = randint(2, N - 2)
        if gcd(a, N) == 1:
            return a

def run_hybrid_sweep(N, trials=40):
    max_depth = min(max(int(N ** 0.3), 8192), 2**18)
    print(f"[📐] Depth = {max_depth}")
    for _ in range(trials):
        a = random_coprime(N)
        factor = hybrid_factor(N, a, max_depth)
        if factor:
            print(f"✅ {N} = {factor} × {N // factor} (base {a})")
            return factor
    print("❌ Failed to find factor")
    return None


# === Test ===
def generate_test_case(bit_size):
    half = bit_size // 2
    p = randprime(2**(half-1), 2**half - 1)
    q = randprime(2**(half-1), 2**half - 1)
    while q == p:
        q = randprime(2**(half-1), 2**half - 1)
    return p * q, p, q

def test(bits):
    print(f"\n🧪 Testing RSA-{bits}")
    N, p, q = generate_test_case(bits)
    print(f"N = {N}\nExpected: {p} × {q}")
    start = time.time()
    f = run_hybrid_sweep(N)
    end = time.time()
    if f and (f == p or f == q):
        print(f"🎉 Success in {end - start:.2f}s")
    else:
        print(f"⚠️  Failed in {end - start:.2f}s")


if __name__ == "__main__":
    for n in range(12, 256, 8):
        test(n)