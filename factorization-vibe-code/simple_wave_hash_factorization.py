#!/usr/bin/env python3
"""
Simple Wave-Based Hash Factorization

This implements ChatGPT's suggested wave-based factorization approach:
- Uses hash collisions to simulate wave interference patterns
- Much simpler than complex spatial computing simulation
- Focuses on the core mathematical insight of collision detection
- Lightweight and fast implementation

The key insight: Hash collisions in the sequence a^k mod N can reveal
periods that lead to factorization, simulating wave interference.
"""

import hashlib
from math import gcd, log2
import time

def modmul(a, b, mod):
    """Efficient modular multiplication."""
    return (a * b) % mod

def wave_hash(x, bits=64):
    """Simulated wave-interference-style hash (fixed bit width)."""
    h = hashlib.sha256(str(x).encode()).digest()
    return int.from_bytes(h, 'big') % (2 ** bits)

def wave_factor(N, max_bases=32, max_depth=2048, hash_bits=64):
    """
    Simulates wave-based factorization by:
    - Performing repeated a^k mod N
    - Hashing each step
    - Using hash collisions to detect periods (wave interference)
    """
    
    print(f"Attempting to factor N={N} with up to {max_bases} bases and {max_depth} depth per base")
    print(f"Using {hash_bits}-bit hash space for collision detection")

    for a in range(2, 2 + max_bases):
        if gcd(a, N) != 1:
            print(f"[!] Trivial factor found early: gcd({a}, {N}) = {gcd(a, N)}")
            return gcd(a, N)

        # Store both hash values and their positions for accurate period detection
        seen_hashes = {}  # hash -> step
        seen_values = {}  # actual value -> step
        x = 1
        
        for k in range(1, max_depth):
            x = modmul(x, a, N)
            hx = wave_hash(x, bits=hash_bits)

            # Check for direct value collision (most reliable)
            if x in seen_values:
                period = k - seen_values[x]
                print(f"[✓] Direct value collision at base={a}, step={k}, period={period}")
                if period > 0:
                    # Verify it's a real period
                    if pow(a, period, N) == 1:
                        print(f"[✓] Verified multiplicative order: {period}")
                        if period % 2 == 0:
                            y = pow(a, period // 2, N)
                            if y != 1 and y != N - 1:  # Avoid trivial cases
                                factor1 = gcd(y - 1, N)
                                factor2 = gcd(y + 1, N)
                                for f in [factor1, factor2]:
                                    if 1 < f < N:
                                        print(f"[✓] Factor found via verified period: {f}")
                                        return f
                    
                    # Also try partial periods
                    for divisor in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
                        if period % divisor == 0:
                            partial_period = period // divisor
                            if partial_period > 0 and partial_period % 2 == 0:
                                y = pow(a, partial_period // 2, N)
                                if y != 1 and y != N - 1:
                                    factor1 = gcd(y - 1, N)
                                    factor2 = gcd(y + 1, N)
                                    for f in [factor1, factor2]:
                                        if 1 < f < N:
                                            print(f"[✓] Factor found via partial period {partial_period}: {f}")
                                            return f

            # Check for hash collision (wave interference simulation)
            if hx in seen_hashes:
                prev_step = seen_hashes[hx]
                period = k - prev_step
                print(f"[✓] Hash collision at base={a}, step={k}, period={period}")
                
                # Try multiple factor extraction strategies
                strategies = [
                    k // 2,           # Half current step
                    period // 2,      # Half detected period
                    k // 4,           # Quarter current step
                    period // 4,      # Quarter detected period
                    (k + prev_step) // 4,  # Average position
                ]
                
                for exp in strategies:
                    if exp > 0:
                        y = pow(a, exp, N)
                        if y != 1 and y != N - 1:
                            factor1 = gcd(y - 1, N)
                            factor2 = gcd(y + 1, N)
                            for f in [factor1, factor2]:
                                if 1 < f < N:
                                    print(f"[✓] Factor found via hash collision strategy: {f}")
                                    return f

            # Natural period detection (a^k ≡ 1 mod N)
            if x == 1 and k > 1:
                print(f"[✓] Natural period found at base={a}, step={k}")
                if k % 2 == 0:
                    y = pow(a, k // 2, N)
                    if y != 1 and y != N - 1:
                        factor1 = gcd(y - 1, N)
                        factor2 = gcd(y + 1, N)
                        for f in [factor1, factor2]:
                            if 1 < f < N:
                                print(f"[✓] Factor found via natural period: {f}")
                                return f

            seen_hashes[hx] = k
            seen_values[x] = k

    print("[×] No factor found — try more bases or deeper steps")
    return None

def enhanced_wave_factor(N, max_bases=128, max_depth=8192, hash_bits=32):
    """
    Enhanced version with multiple strategies and better parameters.
    """
    print(f"Enhanced wave factorization of N={N}")
    
    # Try different hash bit sizes (smaller = more collisions, but higher false positive rate)
    bit_sizes = [12, 16, 20, 24, 28, 32]
    
    for bits in bit_sizes:
        print(f"\n--- Trying {bits}-bit hash space ---")
        
        # Increase search parameters for smaller hash spaces
        if bits <= 16:
            bases = max_bases * 2
            depth = max_depth * 2
        elif bits <= 24:
            bases = max_bases
            depth = max_depth
        else:
            bases = max_bases // 2
            depth = max_depth // 2
            
        factor = wave_factor(N, bases, depth, bits)
        if factor:
            return factor
    
    return None

def test_wave_hash_factorization():
    """Test the simple wave-based hash factorization."""
    print("🌊 Simple Wave-Based Hash Factorization")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        ("RSA-16", 176399, "419 × 421"),
        ("RSA-20", 1048573, "1021 × 1027"),  
        ("RSA-24", 16777181, "4093 × 4099"),
        ("RSA-28", 268435399, "16381 × 16387"),
        ("RSA-32", 4294836223, "65521 × 65537"),
    ]
    
    for name, N, expected in test_cases:
        print(f"\n🎯 Testing {name}: N = {N:,}")
        print(f"Expected: {expected}")
        
        start_time = time.time()
        
        # Try simple approach first
        factor = wave_factor(N, max_bases=32, max_depth=2048, hash_bits=32)
        
        if not factor:
            # Try enhanced approach
            print("\nTrying enhanced approach...")
            factor = enhanced_wave_factor(N, max_bases=64, max_depth=4096)
        
        elapsed = time.time() - start_time
        
        if factor and N % factor == 0 and 1 < factor < N:
            other = N // factor
            print(f"🎉 SUCCESS! {name} FACTORED!")
            print(f"✅ {N:,} = {factor:,} × {other:,}")
            print(f"⏱️  Time: {elapsed:.3f}s")
        else:
            print(f"❌ {name} resisted factorization")
            print(f"⏱️  Time: {elapsed:.3f}s")
        
        # Don't continue if we fail early cases
        if not factor and int(log2(N)) <= 24:
            print("⚠️  Stopping - early case failed")
            break
    
    print("\n🔬 Algorithm Analysis:")
    print("• Uses hash collisions to simulate wave interference")
    print("• O(bases × depth) time complexity") 
    print("• O(depth) space for collision detection")
    print("• Leverages birthday paradox for collision probability")

if __name__ == "__main__":
    test_wave_hash_factorization()
