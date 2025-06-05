#!/usr/bin/env python3
"""
Pollard's Rho Factorization Algorithm
Optimized implementation with multiple polynomial functions and parallel execution
"""

import time
from math import gcd
from random import randint
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sympy import randprime, isprime

def pollard_rho_basic(N, max_iterations=100000):
    """Basic Pollard's rho with f(x) = x^2 + 1"""
    if N % 2 == 0:
        return 2
    if N == 1:
        return None
    
    x = 2
    y = 2
    d = 1
    
    def f(x):
        return (x * x + 1) % N
    
    iterations = 0
    while d == 1 and iterations < max_iterations:
        x = f(x)
        y = f(f(y))
        d = gcd(abs(x - y), N)
        iterations += 1
    
    return d if 1 < d < N else None

def pollard_rho_brent(N, max_iterations=100000):
    """Brent's improvement to Pollard's rho"""
    if N % 2 == 0:
        return 2
    if N == 1:
        return None
    
    y = randint(1, N - 1)
    c = randint(1, N - 1)
    m = randint(1, N - 1)
    
    def f(x):
        return (x * x + c) % N
    
    r = 1
    q = 1
    g = 1  # Initialize g to avoid UnboundLocalError
    
    iterations = 0
    while iterations < max_iterations:
        x = y
        for _ in range(r):
            y = f(y)
        
        k = 0
        while k < r and q > 1:
            ys = y
            for _ in range(min(m, r - k)):
                y = f(y)
                q = (q * abs(x - y)) % N
            
            g = gcd(q, N)
            k += m
            
            # Check if we found a factor
            if g > 1:
                break
        
        r *= 2
        iterations += r
        
        if g > 1:
            if g == N:
                # Backtrack to find the actual factor
                ys = x
                while True:
                    ys = f(ys)
                    g = gcd(abs(x - ys), N)
                    if g > 1:
                        break
            
            return g if 1 < g < N else None
    
    return None

def pollard_rho_parallel(N, num_workers=4, max_iterations=50000):
    """Parallel Pollard's rho with different starting points"""
    if N % 2 == 0:
        return 2
    if N == 1:
        return None
    
    def worker(seed):
        try:
            np.random.seed(seed)
            # Use safer random generation for large N
            if N > 2**31:
                x = int(np.random.randint(2, min(N, 2**31)))
                c = int(np.random.randint(1, min(N, 2**31)))
            else:
                x = np.random.randint(2, N)
                c = np.random.randint(1, N)
            
            y = x
            
            def f(x):
                return (x * x + c) % N
            
            for _ in range(max_iterations):
                x = f(x)
                y = f(f(y))
                d = gcd(abs(x - y), N)
                
                if 1 < d < N:
                    return d
                if d == N:
                    break
                    
        except Exception as e:
            print(f"Worker {seed} error: {e}")
            return None
        
        return None
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker, i) for i in range(num_workers * 2)]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    # Cancel remaining tasks
                    for f in futures:
                        f.cancel()
                    return result
            except Exception as e:
                print(f"Future error: {e}")
                continue
    
    return None

def pollard_rho_multi_poly(N, max_iterations=100000):
    """Try multiple polynomial functions"""
    if N % 2 == 0:
        return 2
    if N == 1:
        return None
    
    polynomials = [
        lambda x: (x * x + 1) % N,
        lambda x: (x * x + 2) % N,
        lambda x: (x * x - 1) % N,
        lambda x: (x * x + x + 1) % N,
    ]
    
    for poly_idx, f in enumerate(polynomials):
        x = 2 + poly_idx
        y = x
        d = 1
        
        iterations = 0
        while d == 1 and iterations < max_iterations // len(polynomials):
            x = f(x)
            y = f(f(y))
            d = gcd(abs(x - y), N)
            iterations += 1
        
        if 1 < d < N:
            return d
    
    return None

def factorize_pollard(N, timeout_sec=60):
    """Main factorization function using multiple Pollard's rho variants"""
    if N <= 1:
        return []
    if N == 2:
        return [2]
    
    start_time = time.time()
    factors = []
    remaining = N
    
    # Remove small factors first
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    for p in small_primes:
        while remaining % p == 0:
            factors.append(p)
            remaining //= p
        if remaining == 1:
            return factors
    
    # Apply Pollard's rho variants with error handling
    methods = [
        pollard_rho_basic,
        pollard_rho_brent,
        pollard_rho_multi_poly,
        lambda n: pollard_rho_parallel(n, num_workers=4)
    ]
    
    method_names = ["Basic", "Brent", "Multi-poly", "Parallel"]
    
    while remaining > 1 and time.time() - start_time < timeout_sec:
        if remaining in small_primes or is_prime_simple(remaining):
            factors.append(remaining)
            break
        
        found_factor = False
        for method, name in zip(methods, method_names):
            if time.time() - start_time > timeout_sec:
                break
            
            print(f"[🎯] Trying {name} on {remaining}")
            
            try:
                factor = method(remaining)
                
                if factor and 1 < factor < remaining:
                    print(f"[✅] Found factor {factor} using {name}")
                    factors.append(factor)
                    remaining //= factor
                    found_factor = True
                    break
            except Exception as e:
                print(f"[⚠️] Error in {name}: {e}")
                continue
        
        if not found_factor:
            print(f"[❌] No factor found for {remaining}")
            factors.append(remaining)  # Assume it's prime
            break
    
    return sorted(factors)

def is_prime_simple(n, trials=20):
    """Simple Miller-Rabin primality test"""
    if n < 2:
        return False
    if n in [2, 3]:
        return True
    if n % 2 == 0:
        return False
    
    # Write n-1 as d * 2^r
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    
    for _ in range(trials):
        a = randint(2, n - 2)
        x = pow(a, d, n)
        
        if x == 1 or x == n - 1:
            continue
        
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    
    return True

def generate_test_case(bit_size):
    half_bits = bit_size // 2
    min_prime = 2 ** (half_bits - 1)
    max_prime = 2 ** half_bits - 1
    p = randprime(min_prime, max_prime)
    q = randprime(min_prime, max_prime)
    while q == p:
        q = randprime(min_prime, max_prime)
    return p * q, p, q


def test_pollard():
    """Test Pollard's rho on various numbers"""
    # More realistic test range - Pollard's rho is practical up to ~80-90 bits
    rsa_bits = range(16, 81, 4)  
    success_count = 0
    total_count = 0
    
    print("🔬 Pollard's Rho Factorization Performance Test")
    print("=" * 60)
    
    for bits in rsa_bits:
        N, p, q = generate_test_case(bits)
        print(f"\n🔢 RSA-{bits} test case")
        print(f"🔎 Testing N = {N:,}")
        print(f"📋 Expected: {p} × {q}")
        start = time.time()
        
        # More aggressive timeout scaling for larger numbers
        if bits <= 32:
            timeout = 10
        elif bits <= 48:
            timeout = 30
        elif bits <= 64:
            timeout = 60
        else:
            timeout = 120
            
        factors = factorize_pollard(N, timeout_sec=timeout)
        elapsed = time.time() - start
        
        product = 1
        for f in factors:
            product *= f
        
        total_count += 1
        if product == N and len(factors) > 1:
            success_count += 1
            print(f"✅ Success: {N} = {' × '.join(map(str, factors))} ({elapsed:.3f}s)")
        else:
            print(f"❌ Failed: got {factors}, product = {product} ({elapsed:.3f}s)")
    
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    print(f"\n📊 Overall Performance Summary:")
    print(f"   Success Rate: {success_count}/{total_count} ({success_rate:.1f}%)")
    print(f"   Practical Range: RSA numbers up to ~80 bits")
    print(f"   Note: Pollard's rho has exponential complexity - larger numbers require specialized algorithms")

if __name__ == "__main__":
    test_pollard()
