#!/usr/bin/env python3
"""
Simple test of wave factorization concept
"""

import hashlib
import random
from math import gcd, log2

def simple_wave_factor(N, verbose=True):
    """Simplified wave-based factorization for testing."""
    if verbose:
        print(f"Attempting to factor N = {N}")
    
    # Try a few bases
    for base in range(2, min(20, N)):
        if gcd(base, N) > 1:
            factor = gcd(base, N)
            if verbose:
                print(f"Found trivial factor: {factor}")
            return factor
        
        # Simple period detection
        seen = {}
        x = 1
        for k in range(1, min(N, 1000)):
            x = (x * base) % N
            
            # Use simple hash
            h = hash(x) % 10000
            
            if h in seen:
                period = k - seen[h]
                if verbose:
                    print(f"Potential period {period} found for base {base}")
                
                # Try to extract factor
                if period % 2 == 0:
                    half = period // 2
                    val = pow(base, half, N)
                    for candidate in [val - 1, val + 1]:
                        factor = gcd(candidate, N)
                        if 1 < factor < N:
                            if verbose:
                                print(f"Found factor: {factor}")
                            return factor
                break
            
            seen[h] = k
            
            if x == 1 and k > 1:
                if verbose:
                    print(f"Natural period {k} for base {base}")
                break
    
    return None

# Test with a simple composite
test_numbers = [15, 21, 35, 77, 91, 143, 187, 209, 323, 391]

print("=== Simple Wave Factorization Test ===")
for N in test_numbers:
    print(f"\nTesting N = {N}")
    factor = simple_wave_factor(N, verbose=False)
    if factor:
        other = N // factor
        print(f"Success: {N} = {factor} × {other}")
    else:
        print(f"Failed to factor {N}")

print("\n=== Detailed Example ===")
N = 77  # 7 × 11
factor = simple_wave_factor(N, verbose=True)
if factor:
    print(f"Final result: {N} = {factor} × {N//factor}")
