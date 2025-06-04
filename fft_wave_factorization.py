
#!/usr/bin/env python3
"""
NumPy-Optimized Wave Interference Factorization with FFT Fallback

Includes:
- Vectorized modular exponentiation
- Rolling window pattern detection
- FFT-based period detection as fallback
- Efficient memory management
- Optional fallback to FFT for difficult numbers
"""

import numpy as np
import time
from math import gcd, log2, sqrt
from numba import jit
from scipy.fft import rfft, rfftfreq

@jit(nopython=True)
def fast_modmul_sequence(a, N, length):
    result = np.empty(length, dtype=np.int64)
    x = a % N
    for i in range(length):
        result[i] = x
        x = (x * a) % N
    return result

@jit(nopython=True)
def fast_gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def fft_detect_period(wave_sequence, N):
    sequence = np.array(wave_sequence, dtype=np.float64)
    mean_centered = sequence - np.mean(sequence)
    spectrum = np.abs(rfft(mean_centered))
    freqs = rfftfreq(len(sequence), d=1)
    peak_idx = np.argmax(spectrum[1:]) + 1
    if peak_idx == 0 or peak_idx >= len(freqs):
        return None
    freq = freqs[peak_idx]
    if freq == 0:
        return None
    period_estimate = round(1 / freq)
    return period_estimate if 2 <= period_estimate <= len(sequence) // 2 else None

def extract_factor_fft(a, period_estimate, N):
    if period_estimate and period_estimate % 2 == 0:
        y = pow(a, period_estimate // 2, N)
        if y != 1 and y != N - 1:
            for delta in [-1, 1]:
                f = gcd(y + delta, N)
                if 1 < f < N:
                    return f
    return None

def numpy_wave_factor_fft(N, max_bases=64, max_depth=4096, window_size=512):
    print(f"Wave+FFT Factorization of N={N}")
    for a in range(2, 2 + max_bases):
        if gcd(a, N) != 1:
            factor = gcd(a, N)
            print(f"[!] Trivial factor: gcd({a}, {N}) = {factor}")
            return factor
        print(f"[🔍] Testing base a={a}")
        wave_sequence = fast_modmul_sequence(a, N, max_depth)
        for k in range(max_depth):
            x = wave_sequence[k]
            if x == 1 and k > 1 and (k + 1) % 2 == 0:
                y = pow(a, (k + 1) // 2, N)
                if y != 1 and y != N - 1:
                    for delta in [-1, 1]:
                        f = fast_gcd(y + delta, N)
                        if 1 < f < N:
                            print(f"[🎯] Factor via natural period: {f}")
                            return f
        # FFT fallback
        print(f"[⚡] Trying FFT periodicity detection for base {a}")
        period_estimate = fft_detect_period(wave_sequence[:window_size], N)
        if period_estimate:
            factor = extract_factor_fft(a, period_estimate, N)
            if factor:
                print(f"[🎯] Factor via FFT period estimate ({period_estimate}): {factor}")
                return factor
    print("[×] No factor found")
    return None

def test_wave_fft():
    print("🌊 Wave Interference + FFT Factorization Test")
    print("=" * 55)
    test_cases = [
        ("RSA-16", 176399, "419 × 421"),
        ("RSA-20", 1048573, "1021 × 1027"),
        ("RSA-24", 16777181, "4093 × 4099"),
        ("RSA-28", 268435399, "16381 × 16387"),
        ("RSA-32", 4294967291, "65521 × 65537"),
    ]
    success_count = 0
    total_time = 0
    for name, N, expected in test_cases:
        print(f"🎯 Testing {name}: N = {N:,}")
        print(f"Expected: {expected}")
        start_time = time.time()
        factor = numpy_wave_factor_fft(N)
        elapsed = time.time() - start_time
        total_time += elapsed
        if factor and N % factor == 0 and 1 < factor < N:
            other = N // factor
            print(f"🎉 SUCCESS! {name} FACTORED!")
            print(f"✅ {N:,} = {factor:,} × {other:,}")
            print(f"⏱️  Time: {elapsed:.3f}s")
            success_count += 1
        else:
            print(f"❌ {name} resisted factorization")
            print(f"⏱️  Time: {elapsed:.3f}s")
    print(f"📊 Success rate: {success_count}/{len(test_cases)}")
    if success_count > 0:
        print(f"Average time: {total_time/success_count:.3f}s")
    print(f"Total time: {total_time:.3f}s")

if __name__ == "__main__":
    test_wave_fft()