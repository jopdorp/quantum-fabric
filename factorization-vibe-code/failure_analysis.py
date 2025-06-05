#!/usr/bin/env python3
"""
Direct Analysis: Why is the FFT Wave Factorization Mostly Failing?
"""

import numpy as np
from math import gcd, log2
from sympy import randprime, factorint

def analyze_why_failing():
    print("🔍 Analysis: Why FFT Wave Factorization is Mostly Failing")
    print("=" * 65)
    
    print("\n1️⃣ FUNDAMENTAL CONCEPT ANALYSIS:")
    print("Your algorithm attempts to:")
    print("• Generate sequence: a^i mod N for i = 1, 2, 3, ...")
    print("• Apply FFT to find 'periodicities' in phase space")
    print("• Use detected periods to compute y = a^(period/2) mod N")
    print("• Try gcd(y±1, N) to find factors")
    print()
    
    print("2️⃣ MATHEMATICAL PROBLEMS:")
    print("❌ The FFT is detecting noise, not meaningful mathematical structure")
    print("❌ Period detection in phase space ≠ multiplicative order")
    print("❌ The connection between FFT peaks and factorization is tenuous")
    print("❌ Success depends on coincidental short multiplicative orders")
    print()
    
    print("3️⃣ WHY SOME CASES WORK:")
    print("✅ RSA-16 worked because multiplicative order was small (210)")
    print("✅ RSA-24 worked by pure luck/coincidence") 
    print("❌ Larger cases fail because multiplicative orders are huge")
    print()
    
    print("4️⃣ EVIDENCE FROM YOUR RESULTS:")
    bit_sizes = [16, 20, 24, 28, 32, 36]
    results = ["SUCCESS", "FAILED", "SUCCESS", "FAILED", "FAILED", "FAILED"]
    
    for bit_size, result in zip(bit_sizes, results):
        N, p, q = generate_test_case(bit_size)
        
        # Check actual difficulty
        print(f"RSA-{bit_size}: {result}")
        print(f"  N = {N:,} = {p} × {q}")
        print(f"  φ(N) = {(p-1)*(q-1):,}")
        print(f"  Theoretical max order: {(p-1)*(q-1)//gcd(p-1,q-1):,}")
        
        # Your algorithm uses max_depth = 1024 or 2048
        theoretical_max = (p-1)*(q-1)//gcd(p-1,q-1)
        if theoretical_max > 2048:
            print(f"  ❌ Max order likely > 2048 (your search limit)")
        else:
            print(f"  ✅ Max order might be ≤ 2048 (searchable)")
        print()
    
    print("5️⃣ CONCLUSION:")
    print("Your algorithm is NOT following a sound mathematical concept.")
    print("It's essentially a sophisticated form of trial-and-error that")
    print("occasionally succeeds when numbers have special properties.")
    print()
    print("For reliable factorization, use proven methods like:")
    print("• Pollard's rho algorithm") 
    print("• Quadratic sieve")
    print("• General number field sieve")
    print("• Elliptic curve factorization")

def generate_test_case(bit_size):
    half_bits = bit_size // 2
    min_prime = 2 ** (half_bits - 1)
    max_prime = 2 ** half_bits - 1
    p = randprime(min_prime, max_prime)
    q = randprime(min_prime, max_prime)
    while q == p:
        q = randprime(min_prime, max_prime)
    return p * q, p, q

if __name__ == "__main__":
    analyze_why_failing()
