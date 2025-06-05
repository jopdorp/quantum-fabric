#!/usr/bin/env python3
"""
Wave Interference Factorization (v9.1 - Optimized)
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

# Optimized accumulating interference map with JIT and early termination
@jit(nopython=True)
def accumulating_interference_fast(signal_real, signal_imag, max_lag):
    acc_map = np.zeros(max_lag, dtype=np.float64)
    n = len(signal_real)
    
    for d in range(1, min(max_lag, n)):
        correlation_sum = 0.0
        for i in range(n - d):
            # Proper autocorrelation: signal[i] * conj(signal[i+d])
            real_part = signal_real[i] * signal_real[i + d] + signal_imag[i] * signal_imag[i + d]
            imag_part = signal_imag[i] * signal_real[i + d] - signal_real[i] * signal_imag[i + d]
            correlation_sum += real_part * real_part + imag_part * imag_part
        
        acc_map[d] = correlation_sum / (n - d)  # Proper normalization
    
    return acc_map

# Simple period detection using power spectral density
def find_period_fft(sequence, N):
    """Find period using FFT-based approach"""
    if len(sequence) < 8:
        return None
    
    # Convert to complex signal
    signal = np.exp(2j * np.pi * np.array(sequence) / N)
    
    # Apply window to reduce spectral leakage
    window = np.hanning(len(signal))
    signal = signal * window
    
    # Compute power spectral density
    fft_result = np.fft.fft(signal)
    psd = np.abs(fft_result) ** 2
    
    # Find dominant frequency (skip DC component)
    freq_idx = np.argmax(psd[1:len(psd)//2]) + 1
    
    if freq_idx > 0:
        period = len(signal) / freq_idx
        return int(round(period))
    
    return None

# Classical order-finding as fallback
def classical_order_finding(a, N, max_trials=1000):
    """Find the multiplicative order of a modulo N"""
    if gcd(a, N) != 1:
        return None
    
    order = 1
    current = a % N
    
    while order <= max_trials:
        if current == 1:
            return order
        current = (current * a) % N
        order += 1
    
    return None

# Enhanced interference factor detection
def interference_factor_adaptive(N, a, base_depth=1024):
    # First try classical order finding for small cases
    if N < 10000:
        r = classical_order_finding(a, N, min(N, 10000))
        if r and r > 1 and r % 2 == 0:
            y = pow(a, r // 2, N)
            if y != 1 and y != N - 1:
                for delta in [-1, 1]:
                    f = gcd(y + delta, N)
                    if 1 < f < N:
                        print(f"[⚡] Factor: {f} via classical order finding (r={r})")
                        return f
    
    # Wave interference approach
    depths = [base_depth, min(base_depth * 2, 4096)]
    
    for depth in depths:
        if depth > N // 4:
            continue
            
        # Generate modular sequence
        sequence = fast_modmul_sequence(a, N, depth)
        
        # Try FFT-based period detection
        period_fft = find_period_fft(sequence, N)
        if period_fft and period_fft > 1:
            r = period_fft
            if r % 2 == 0:
                y = pow(a, r // 2, N)
                if y != 1 and y != N - 1:
                    for delta in [-1, 1]:
                        f = gcd(y + delta, N)
                        if 1 < f < N:
                            print(f"[🌊] Factor: {f} via FFT period finding (r={r})")
                            return f
        
        # Wave interference correlation approach
        signal = complex_modular_signal(a, N, depth)
        max_lag = min(depth // 2, 2048)
        acc = accumulating_interference_fast(signal.real, signal.imag, max_lag)
        
        # Find periods with strong correlations
        candidates = []
        max_corr = np.max(acc[1:])
        threshold = max_corr * 0.7  # 70% of maximum
        
        for i in range(2, len(acc)):
            if acc[i] > threshold:
                candidates.append(i)
        
        # Test period candidates
        for r in candidates[:10]:  # Test top candidates
            if r <= 1:
                continue
                
            # Test if this might be the period
            test_vals = []
            if r % 2 == 0:
                test_vals.append(pow(a, r // 2, N))
            test_vals.append(pow(a, r, N))
            
            for y in test_vals:
                if y == 1:  # Found potential period
                    continue
                    
                if y == N - 1:
                    continue
                    
                for delta in [-1, 1]:
                    candidate = y + delta
                    if candidate <= 1 or candidate >= N:
                        continue
                        
                    f = gcd(candidate, N)
                    if 1 < f < N:
                        print(f"[🌊] Factor: {f} via correlation (r={r}, y={y}, corr={acc[r]:.3f})")
                        return f
    
    return None

# Pollard's rho as additional fallback
def pollard_rho(N, max_iterations=10000):
    """Pollard's rho algorithm for factorization"""
    if N % 2 == 0:
        return 2
    
    x = 2
    y = 2
    d = 1
    
    def f(x):
        return (x * x + 1) % N
    
    for _ in range(max_iterations):
        x = f(x)
        y = f(f(y))
        d = gcd(abs(x - y), N)
        
        if d > 1:
            if d < N:
                return d
            break
    
    return None

# Estimate depth heuristically
def estimate_max_depth(N, cap=2**16):
    if N < 2**20:
        return min(max(int(N ** 0.25), 1024), cap // 4)
    elif N < 2**30:
        return min(max(int(N ** 0.2), 2048), cap // 2)
    else:
        return min(max(int(N ** 0.15), 4096), cap)

# Fixed random base coprime to N (handles large N)
def random_coprime(N):
    max_attempts = 1000
    for _ in range(max_attempts):
        # Use numpy random for large numbers to avoid int64 overflow
        if N > 2**31:
            a = int(np.random.randint(2, min(N-1, 2**31)))
        else:
            a = randint(2, N - 2)
            
        if gcd(a, N) == 1:
            return a
    
    # Fallback: try small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    for p in small_primes:
        if p < N and gcd(p, N) == 1:
            return p
    
    return 2  # Last resort

# Enhanced sweep with multiple algorithms
def sweep_interference(N, timeout_sec=60, max_trials=20):
    def try_a(a, base_depth):
        try:
            return a, interference_factor_adaptive(N, a, base_depth)
        except Exception as e:
            print(f"Error with base {a}: {e}")
            return a, None

    base_depth = estimate_max_depth(N)
    print(f"[📐] base_depth = {base_depth} for N = {N}")

    # First try Pollard's rho as it's often effective
    print("[🎯] Trying Pollard's rho...")
    rho_factor = pollard_rho(N)
    if rho_factor:
        print(f"✅ SUCCESS via Pollard's rho: {N} = {rho_factor} × {N // rho_factor}")
        return rho_factor

    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for _ in range(max_trials):
            base_a = random_coprime(N)
            future = executor.submit(try_a, base_a, base_depth)
            futures.append(future)
        
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

# Test for RSA bit sizes with adjusted parameters

def test_interference():
    bit_sizes = list(range(16, 65, 4))  # Test up to 52 bits
    times = []
    success = []

    for bit_size in bit_sizes:
        print(f"\n🔎 RSA-{bit_size} test case")
        N, p, q = generate_test_case(bit_size)
        print(f"N = {N:,}\nExpected: {p} × {q}")

        start = time.time()
        # Adaptive timeout based on bit size
        timeout = min(30 + (bit_size - 16) * 5, 180)
        factor = sweep_interference(N, timeout_sec=timeout)
        elapsed = time.time() - start

        times.append(elapsed)
        if factor in (p, q):
            print("🎉 Correct factor found")
            success.append(True)
        else:
            print("⚠️  Incorrect or no factor")
            success.append(False)

    # Graph times
    plt.figure(figsize=(10, 6))
    plt.semilogy(bit_sizes, times, marker='o', linewidth=2, markersize=8)
    plt.xlabel('RSA Bit Size')
    plt.ylabel('Time (log seconds)')
    plt.title('Wave Interference Factorization (Optimized)')
    plt.grid(True, alpha=0.3)
    plt.show()

    success_rate = sum(success) / len(success) * 100
    print(f"\n📊 Overall success rate: {success_rate:.1f}%")
    
    for b, s in zip(bit_sizes, success):
        print(f"RSA-{b}: {'✅' if s else '❌'}")

if __name__ == "__main__":
    test_interference()
