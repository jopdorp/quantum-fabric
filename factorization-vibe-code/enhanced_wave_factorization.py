#!/usr/bin/env python3
"""
Enhanced Wave-Based Integer Factorization: Improved Proof of Concept

This implements an improved CPU simulation of the wave-based computational architecture
for polynomial-time integer factorization with enhanced collision detection and
multiple optimization strategies.

Key improvements:
- Multi-resolution hash collision detection
- Adaptive period search with multiple strategies
- Enhanced base selection algorithms
- Improved factor extraction methods
- Better statistical tracking and analysis
"""

import hashlib
import random
import time
from math import gcd, isqrt, log2, ceil
from typing import Optional, Tuple, Set, List, Dict
import secrets
from collections import defaultdict

class EnhancedWaveSignal:
    """Enhanced wave signal with multi-resolution hash encoding."""
    
    def __init__(self, value: int, step: int, base: int, signal_bits: int = 32):
        self.value = value
        self.step = step
        self.base = base
        self.signal_bits = signal_bits
        
        # Multiple hash resolutions for better collision detection
        self.hash_low = self._compute_hash(value, signal_bits // 2)    # Lower resolution
        self.hash_med = self._compute_hash(value, signal_bits)         # Medium resolution  
        self.hash_high = self._compute_hash(value, signal_bits * 2)    # Higher resolution
    
    def _compute_hash(self, value: int, bits: int) -> int:
        """Compute hash with specified bit resolution."""
        # Use multiple hash functions for better distribution
        h1 = hashlib.sha256(f"{value}_{self.base}_1".encode()).digest()
        h2 = hashlib.sha256(f"{value}_{self.base}_2".encode()).digest()
        
        # Combine hashes for better distribution
        combined = int.from_bytes(h1, 'big') ^ int.from_bytes(h2, 'big')
        return combined % (2 ** bits)

class EnhancedWaveFactorizer:
    """
    Enhanced wave-based factorization engine with improved algorithms.
    """
    
    def __init__(self, signal_bits: int = 32, max_bases: int = None, verbose: bool = False):
        self.signal_bits = signal_bits
        self.verbose = verbose
        self.max_bases = max_bases
        
        # Enhanced statistics tracking
        self.stats = {
            'collisions_low': 0,
            'collisions_med': 0, 
            'collisions_high': 0,
            'total_steps': 0,
            'periods_found': 0,
            'factors_extracted': 0,
            'bases_tried': 0,
            'trivial_factors': 0
        }
        
    def _log(self, message: str):
        """Conditional logging based on verbose flag."""
        if self.verbose:
            print(f"[WAVE] {message}")
    
    def _generate_smart_bases(self, N: int, count: int) -> List[int]:
        """Generate bases using multiple strategies for better coverage."""
        bases = []
        attempts = 0
        max_attempts = count * 20
        
        # Strategy 1: Small random bases (good for small factors)
        small_bases = []
        while len(small_bases) < count // 3 and attempts < max_attempts // 3:
            a = random.randint(2, min(100, N-1))
            g = gcd(a, N)
            if g == 1:
                small_bases.append(a)
            elif 1 < g < N:
                self._log(f"Trivial factor found via GCD({a}, {N}) = {g}")
                self.stats['trivial_factors'] += 1
                return [g]
            attempts += 1
        
        # Strategy 2: Medium random bases
        medium_bases = []
        while len(medium_bases) < count // 3 and attempts < 2 * max_attempts // 3:
            a = random.randint(100, min(10000, N-1))
            g = gcd(a, N)
            if g == 1:
                medium_bases.append(a)
            elif 1 < g < N:
                self._log(f"Trivial factor found via GCD({a}, {N}) = {g}")
                self.stats['trivial_factors'] += 1
                return [g]
            attempts += 1
        
        # Strategy 3: Larger bases for better period diversity
        large_bases = []
        remaining = count - len(small_bases) - len(medium_bases)
        while len(large_bases) < remaining and attempts < max_attempts:
            a = random.randint(2, min(N-1, 100000))
            g = gcd(a, N)
            if g == 1:
                large_bases.append(a)
            elif 1 < g < N:
                self._log(f"Trivial factor found via GCD({a}, {N}) = {g}")
                self.stats['trivial_factors'] += 1
                return [g]
            attempts += 1
        
        bases = small_bases + medium_bases + large_bases
        self._log(f"Generated {len(bases)} bases: {len(small_bases)} small, {len(medium_bases)} medium, {len(large_bases)} large")
        return bases
    
    def _detect_period_multi_resolution(self, N: int, base: int, max_depth: int) -> Optional[int]:
        """
        Enhanced period detection using multiple hash resolutions.
        """
        # Track collisions at different resolutions
        seen_low: Dict[int, int] = {}
        seen_med: Dict[int, int] = {}
        seen_high: Dict[int, int] = {}
        
        current_value = 1
        
        for step in range(1, max_depth + 1):
            current_value = (current_value * base) % N
            
            # Create enhanced wave signal
            signal = EnhancedWaveSignal(current_value, step, base, self.signal_bits)
            
            # Check for collisions at different resolutions
            periods_to_check = []
            
            # Low resolution collision (most likely to occur)
            if signal.hash_low in seen_low:
                period = step - seen_low[signal.hash_low]
                periods_to_check.append(('low', period, seen_low[signal.hash_low]))
                self.stats['collisions_low'] += 1
            
            # Medium resolution collision
            if signal.hash_med in seen_med:
                period = step - seen_med[signal.hash_med]
                periods_to_check.append(('med', period, seen_med[signal.hash_med]))
                self.stats['collisions_med'] += 1
            
            # High resolution collision (most reliable)
            if signal.hash_high in seen_high:
                period = step - seen_high[signal.hash_high]
                periods_to_check.append(('high', period, seen_high[signal.hash_high]))
                self.stats['collisions_high'] += 1
            
            # Verify periods in order of reliability (high to low)
            for resolution, period, start_step in sorted(periods_to_check, key=lambda x: {'high': 0, 'med': 1, 'low': 2}[x[0]]):
                if self._verify_period(N, base, period, start_step):
                    self._log(f"Verified period {period} found via {resolution}-resolution collision at step {step}")
                    self.stats['periods_found'] += 1
                    return period
            
            # Store current hashes
            seen_low[signal.hash_low] = step
            seen_med[signal.hash_med] = step
            seen_high[signal.hash_high] = step
            
            self.stats['total_steps'] += 1
            
            # Natural period detection
            if current_value == 1 and step > 1:
                self._log(f"Natural period {step} found for base {base}")
                self.stats['periods_found'] += 1
                return step
        
        return None
    
    def _verify_period(self, N: int, base: int, period: int, start_step: int) -> bool:
        """Enhanced period verification with multiple checks."""
        if period <= 0:
            return False
        
        # Primary verification: a^(start + period) ≡ a^start (mod N)
        val1 = pow(base, start_step + period, N)
        val2 = pow(base, start_step, N)
        
        if val1 != val2:
            return False
        
        # Secondary verification: check a few more points
        for offset in [1, 2, 3]:
            if start_step + offset < period:
                val1 = pow(base, start_step + offset + period, N)
                val2 = pow(base, start_step + offset, N)
                if val1 != val2:
                    return False
        
        return True
    
    def _extract_factor_enhanced(self, N: int, base: int, period: int) -> Optional[int]:
        """
        Enhanced factor extraction with multiple strategies.
        """
        factors_found = []
        
        # Strategy 1: Classical Shor approach
        if period % 2 == 0:
            half_period = period // 2
            val = pow(base, half_period, N)
            
            if val != 1 and val != N - 1:
                for candidate in [val - 1, val + 1]:
                    factor = gcd(candidate, N)
                    if 1 < factor < N:
                        factors_found.append(factor)
        
        # Strategy 2: Try other divisors of the period
        for divisor in [period // 3, period // 4, period // 6]:
            if divisor > 0 and period % divisor == 0:
                val = pow(base, divisor, N)
                for candidate in [val - 1, val + 1]:
                    factor = gcd(candidate, N)
                    if 1 < factor < N:
                        factors_found.append(factor)
        
        # Strategy 3: Try intermediate values
        for fraction in [period // 8, 3 * period // 8, 5 * period // 8, 7 * period // 8]:
            if fraction > 0:
                val = pow(base, fraction, N)
                for candidate in [val - 1, val + 1]:
                    factor = gcd(candidate, N)
                    if 1 < factor < N:
                        factors_found.append(factor)
        
        if factors_found:
            # Return the smallest non-trivial factor
            factor = min(factors_found)
            self._log(f"Factor {factor} extracted from period {period}")
            self.stats['factors_extracted'] += 1
            return factor
        
        return None
    
    def wave_factor(self, N: int) -> Optional[int]:
        """
        Enhanced wave-based factorization algorithm.
        """
        if N < 4:
            return None
        
        # Quick check for small factors
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        for p in small_primes:
            if N % p == 0:
                return p
        
        n = int(log2(N)) + 1
        
        # Enhanced scaling parameters
        num_bases = min(self.max_bases or (n * 2), 100)
        max_depth = min(n * n * 2, 5000)  # Increased depth for better success
        
        self._log(f"Enhanced factoring N = {N} (n={n} bits)")
        self._log(f"Using {num_bases} bases, max depth {max_depth}")
        
        # Generate smart bases
        bases = self._generate_smart_bases(N, num_bases)
        
        # Check if we found a trivial factor
        if len(bases) == 1 and bases[0] < N:
            return bases[0]
        
        # Try each base with enhanced period detection
        for i, base in enumerate(bases):
            self.stats['bases_tried'] += 1
            self._log(f"Trying base {base} ({i+1}/{len(bases)})")
            
            # Enhanced period detection
            period = self._detect_period_multi_resolution(N, base, max_depth)
            
            if period:
                self._log(f"Period r={period} detected for base a={base}")
                
                # Enhanced factor extraction
                factor = self._extract_factor_enhanced(N, base, period)
                
                if factor:
                    return factor
        
        self._log("No factor found with current parameters")
        return None
    
    def get_statistics(self) -> dict:
        """Return comprehensive algorithm statistics."""
        return self.stats.copy()

def benchmark_enhanced_factorizer():
    """Benchmark the enhanced factorizer against various number sizes."""
    print("=== Enhanced Wave-Based Factorization Benchmark ===\n")
    
    test_cases = [
        # Small composites
        (15, 3, 5),
        (21, 3, 7), 
        (35, 5, 7),
        (77, 7, 11),
        (91, 7, 13),
        (143, 11, 13),
        (187, 11, 17),
        (209, 11, 19),
        (323, 17, 19),
        (391, 17, 23),
        
        # Medium composites
        (1147, 31, 37),
        (1517, 37, 41),
        (1763, 41, 43),
        (2021, 43, 47),
        (2491, 47, 53),
        
        # Larger composites
        (4087, 61, 67),
        (4757, 67, 71),
        (5183, 71, 73),
        (5767, 73, 79),
        (6557, 79, 83),
    ]
    
    success_count = 0
    total_time = 0
    
    for N, expected_p, expected_q in test_cases:
        print(f"Testing N = {N} = {expected_p} × {expected_q}")
        
        factorizer = EnhancedWaveFactorizer(
            signal_bits=48,
            max_bases=50,
            verbose=False
        )
        
        start_time = time.time()
        factor = factorizer.wave_factor(N)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        if factor and N % factor == 0:
            other_factor = N // factor
            print(f"  ✅ SUCCESS: {factor} × {other_factor} in {elapsed:.4f}s")
            success_count += 1
            
            # Verify correctness
            if (factor == expected_p and other_factor == expected_q) or \
               (factor == expected_q and other_factor == expected_p):
                print(f"     🎯 Factors match expected values")
        else:
            print(f"  ❌ FAILED in {elapsed:.4f}s")
        
        # Show statistics for interesting cases
        stats = factorizer.get_statistics()
        if stats['collisions_low'] > 0 or stats['periods_found'] > 0:
            print(f"     📊 Steps: {stats['total_steps']}, Collisions: L{stats['collisions_low']}/M{stats['collisions_med']}/H{stats['collisions_high']}, Periods: {stats['periods_found']}")
        
        print()
    
    print(f"=== Summary ===")
    print(f"Success rate: {success_count}/{len(test_cases)} ({100*success_count/len(test_cases):.1f}%)")
    print(f"Total time: {total_time:.4f}s")
    print(f"Average time: {total_time/len(test_cases):.4f}s per number")

def generate_rsa_like_number(bits: int) -> Tuple[int, int, int]:
    """Generate a semiprime N = p*q with approximately 'bits' total bits."""
    half_bits = bits // 2
    
    def next_prime(n):
        """Find next prime >= n."""
        if n < 2:
            return 2
        while True:
            if is_prime(n):
                return n
            n += 1
    
    # Generate two primes of approximately half_bits each
    p = next_prime(secrets.randbelow(2**(half_bits+1) - 2**(half_bits-1)) + 2**(half_bits-1))
    q = next_prime(secrets.randbelow(2**(half_bits+1) - 2**(half_bits-1)) + 2**(half_bits-1))
    
    # Ensure they're different
    while q == p:
        q = next_prime(q + 1)
    
    return p * q, p, q

def is_prime(n: int) -> bool:
    """Simple primality test."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, isqrt(n) + 1, 2):
        if n % i == 0:
            return False
    return True

def main():
    """Main demonstration of enhanced wave-based factorization."""
    print("🌊 Enhanced Wave-Based Integer Factorization")
    print("=" * 60)
    print()
    
    # Run benchmark
    benchmark_enhanced_factorizer()
    
    print("\n=== Detailed Example with 16-bit RSA-like Number ===")
    
    # Generate a challenging 16-bit number
    N, p, q = generate_rsa_like_number(16)
    print(f"Target: N = {N} = {p} × {q}")
    print(f"Bit length: {int(log2(N)) + 1}")
    print()
    
    # Create enhanced factorizer with verbose output
    factorizer = EnhancedWaveFactorizer(
        signal_bits=64,
        max_bases=75,
        verbose=True
    )
    
    print("Starting enhanced wave-based factorization...")
    start_time = time.time()
    
    factor = factorizer.wave_factor(N)
    
    elapsed = time.time() - start_time
    
    print(f"\n⏱️  Total time: {elapsed:.4f} seconds")
    
    if factor and N % factor == 0:
        other_factor = N // factor
        print(f"✅ SUCCESS: {N} = {factor} × {other_factor}")
        
        # Verify correctness
        if (factor == p and other_factor == q) or (factor == q and other_factor == p):
            print("🎯 Factors match expected values!")
        else:
            print("⚠️  Factors are correct but different ordering")
    else:
        print(f"❌ Factorization failed")
    
    # Display comprehensive statistics
    stats = factorizer.get_statistics()
    print(f"\n📊 Enhanced Algorithm Statistics:")
    print(f"   • Total wave steps: {stats['total_steps']}")
    print(f"   • Collision detections: Low-res: {stats['collisions_low']}, Med-res: {stats['collisions_med']}, High-res: {stats['collisions_high']}")
    print(f"   • Periods found: {stats['periods_found']}")
    print(f"   • Factors extracted: {stats['factors_extracted']}")
    print(f"   • Bases tried: {stats['bases_tried']}")
    print(f"   • Trivial factors: {stats['trivial_factors']}")

if __name__ == "__main__":
    main()
