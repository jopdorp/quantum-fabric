#!/usr/bin/env python3
"""
Mathematically Sound Factorization Based on Pollard's rho with FFT Enhancement

The original approach had fundamental flaws. This implements a proven method
with FFT used only for optimization, not as the core mathematical principle.
"""

import numpy as np
import time
from math import gcd
from sympy import randprime
from numba import jit

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
def pollard_rho_function(x, c, N):
    """Standard Pollard's rho function: f(x) = x^2 + c mod N"""
    return (x * x + c) % N

@jit(nopython=True)
def pollard_rho_basic(N, c=1, max_iterations=1000000):
    """Basic Pollard's rho algorithm"""
    x = 2
    y = 2
    
    for _ in range(max_iterations):
        x = pollard_rho_function(x, c, N)
        y = pollard_rho_function(pollard_rho_function(y, c, N), c, N)
        
        d = gcd(abs(x - y), N)
        if d > 1 and d < N:
            return d
    return None

def fft_enhanced_pollard_rho(N, max_sequence_length=10000):
    """
    Use FFT to detect potential cycle patterns in Pollard's rho sequence
    to optimize the choice of parameters
    """
    print(f"[🔍] FFT-enhanced Pollard's rho for N = {N}")
    
    # Try different c values, use FFT to detect promising patterns
    best_c = 1
    best_score = 0
    
    for c in range(1, 20):
        # Generate a sequence of the rho function
        sequence = []
        x = 2
        for i in range(min(1000, max_sequence_length)):
            x = pollard_rho_function(x, c, N)
            sequence.append(x)
        
        # Use FFT to analyze the sequence for patterns
        if len(sequence) >= 256:
            # Convert to phase representation
            phases = np.array(sequence[:256]) / N * 2 * np.pi
            phased_signal = np.exp(1j * phases)
            
            # Get spectrum
            spectrum = np.abs(np.fft.fft(phased_signal))
            spectrum[0] = 0  # Remove DC
            
            # Score based on spectral concentration (more peaks = more structure)
            score = np.std(spectrum)
            
            if score > best_score:
                best_score = score
                best_c = c
    
    print(f"[📊] Best c value from FFT analysis: {best_c} (score: {best_score:.3f})")
    
    # Now run Pollard's rho with the optimized parameter
    factor = pollard_rho_basic(N, c=best_c, max_iterations=1000000)
    return factor

def quadratic_sieve_lite(N, max_factor_base=1000):
    """
    Simplified quadratic sieve - only for educational purposes
    Real QS is much more complex
    """
    print(f"[⚡] Quadratic sieve lite for N = {N}")
    
    # Find small primes for factor base
    factor_base = []
    for p in range(2, max_factor_base):
        if all(p % q != 0 for q in factor_base if q * q <= p):
            # Check if N is quadratic residue mod p
            if pow(N, (p-1)//2, p) == 1:
                factor_base.append(p)
        if len(factor_base) >= 100:  # Limit size for performance
            break
    
    print(f"[📋] Factor base size: {len(factor_base)}")
    
    # This is a very simplified version - real QS needs much more sophistication
    # For now, fall back to trial division for small factors
    for p in factor_base[:20]:  # Only try smallest primes
        if N % p == 0:
            return p
    
    return None

def multi_method_factorization(N):
    """
    Try multiple factorization methods in order of efficiency
    """
    print(f"[🎯] Multi-method factorization of N = {N}")
    
    # Method 1: Trial division for small factors
    print("[1️⃣] Trying trial division...")
    for p in range(2, min(10000, int(N**0.5) + 1)):
        if N % p == 0:
            print(f"[✅] Trial division found: {p}")
            return p
    
    # Method 2: FFT-enhanced Pollard's rho
    print("[2️⃣] Trying FFT-enhanced Pollard's rho...")
    factor = fft_enhanced_pollard_rho(N)
    if factor:
        print(f"[✅] Pollard's rho found: {factor}")
        return factor
    
    # Method 3: Quadratic sieve lite (for educational purposes)
    print("[3️⃣] Trying quadratic sieve lite...")
    factor = quadratic_sieve_lite(N)
    if factor:
        print(f"[✅] Quadratic sieve found: {factor}")
        return factor
    
    print("[❌] No factor found with available methods")
    return None

def test_corrected_approach():
    print("🔧 Mathematically Sound Factorization Approach")
    print("=" * 60)
    print("Methods:")
    print("• Trial division for small factors")
    print("• FFT-enhanced Pollard's rho")
    print("• Simplified quadratic sieve")
    print()

    bit_sizes = [16, 20, 24, 28, 32, 36]

    for bit_size in bit_sizes:
        print(f"\n🎯 Testing RSA-{bit_size}...")
        N, p, q = generate_test_case(bit_size)
        print(f"N = {N:,}")
        print(f"Expected factors: {p} × {q}")

        start = time.time()
        result = multi_method_factorization(N)
        elapsed = time.time() - start
        print(f"⏱️  Time: {elapsed:.3f}s")

        if result:
            other_factor = N // result
            if (result == p and other_factor == q) or (result == q and other_factor == p):
                print(f"🎉 CORRECT FACTORS FOUND!")
            else:
                print(f"⚠️  Found different factors: {result} × {other_factor}")
        print("-" * 50)

if __name__ == "__main__":
    test_corrected_approach()
