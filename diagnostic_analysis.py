#!/usr/bin/env python3
"""
Diagnostic Analysis of FFT Wave Factorization
Let's understand what's actually happening mathematically
"""

import numpy as np
from math import gcd, log2
from numba import jit
from sympy import randprime

# Fast modular multiplication sequence
@jit(nopython=True)
def fast_modmul_sequence(a, N, length):
    result = [0] * length
    x = a % N
    for i in range(length):
        result[i] = x
        x = (x * a) % N
    return result

def analyze_sequence_properties(N, p, q, base=2, max_length=2048):
    """Analyze the mathematical properties of the modular sequence"""
    print(f"\n🔍 Analyzing N = {N} = {p} × {q} with base {base}")
    
    # Generate sequence
    sequence = fast_modmul_sequence(base, N, max_length)
    
    # Find actual multiplicative order
    actual_order = None
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[0]:  # Found period when we return to start
            actual_order = i
            break
    
    print(f"Actual multiplicative order of {base} mod {N}: {actual_order}")
    
    # Theoretical orders
    phi_N = (p-1) * (q-1)
    print(f"φ(N) = {phi_N}")
    
    # Check what happens at order/2
    if actual_order:
        half_order = actual_order // 2
        y = pow(base, half_order, N)
        print(f"{base}^({actual_order}//2) mod {N} = {base}^{half_order} mod {N} = {y}")
        
        # Check if this gives us factors
        for delta in [-1, 1]:
            factor_candidate = gcd(y + delta, N)
            if 1 < factor_candidate < N:
                print(f"✅ gcd({y} + {delta}, {N}) = {factor_candidate}")
                return True
            else:
                print(f"❌ gcd({y} + {delta}, {N}) = {factor_candidate}")
    
    # Now let's see what FFT is actually detecting
    print(f"\n🌊 FFT Analysis:")
    window = 512
    windowed = sequence[:window]
    phased_signal = np.exp(2j * np.pi * np.array(windowed) / N)
    spectrum = np.abs(np.fft.fft(phased_signal))
    spectrum[0] = 0  # Remove DC
    
    # Find top frequencies
    top_indices = np.argpartition(spectrum, -5)[-5:]
    top_indices = sorted(top_indices, key=lambda x: spectrum[x], reverse=True)
    
    print("Top FFT frequencies (index -> period -> amplitude):")
    for idx in top_indices:
        if idx == 0:
            continue
        period = round(window / idx)
        print(f"  Index {idx} -> Period ~{period} -> Amplitude {spectrum[idx]:.3f}")
        
        # Test what happens if we use this period
        if period > 0:
            test_y = pow(base, period // 2, N)
            for delta in [-1, 1]:
                test_factor = gcd(test_y + delta, N)
                if 1 < test_factor < N:
                    print(f"    🎯 This period gives factor: {test_factor}")
    
    return False

def test_diagnostic():
    print("🔬 Diagnostic Analysis of Wave Factorization Algorithm")
    print("=" * 70)
    
    # Test the cases that worked and failed
    test_cases = [
        (16, "WORKED"),
        (20, "FAILED"), 
        (24, "WORKED"),
        (28, "FAILED")
    ]
    
    for bit_size, expected in test_cases:
        print(f"\n{'='*50}")
        print(f"RSA-{bit_size} ({expected})")
        print(f"{'='*50}")
        
        # Generate same type of test case
        half_bits = bit_size // 2
        min_prime = 2 ** (half_bits - 1)
        max_prime = 2 ** half_bits - 1
        p = randprime(min_prime, max_prime)
        q = randprime(min_prime, max_prime)
        while q == p:
            q = randprime(min_prime, max_prime)
        N = p * q
        
        success = analyze_sequence_properties(N, p, q)
        print(f"Analysis result: {'SUCCESS' if success else 'FAILED'}")

if __name__ == "__main__":
    test_diagnostic()
