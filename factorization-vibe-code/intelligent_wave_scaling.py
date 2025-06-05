#!/usr/bin/env python3
"""
Intelligent Wave Interference Scaling for RSA-64
===============================================

The issue with the previous scaling approach:
- We scaled signal length 512x (4,096 → 2,097,152)
- We scaled search depth 512x (1,024 → 524,288)
- This created NOISE, not better signal detection

The correct scaling approach:
- Keep signal length reasonable (4K-32K range)
- Focus on MORE BASES, not longer signals
- Use ADAPTIVE signal length based on expected period
- Scale SEARCH STRATEGIES, not just brute force parameters

Key insight: The successful periods are typically SHORT. 
Making signals 512x longer doesn't help find short periods - it hurts.
"""

import numpy as np
import time
import math
import gc
import sympy
from typing import List, Tuple, Optional

class IntelligentWaveFactorizer:
    """
    Intelligently scaled wave factorizer that focuses on the RIGHT parameters.
    """
    
    def __init__(self):
        # Keep reasonable signal lengths - don't create noise
        self.signal_lengths = [2048, 4096, 8192, 16384, 32768]
        self.search_depths = [512, 1024, 2048, 4096, 8192]
        
    def fast_modular_sequence(self, base: int, modulus: int, length: int) -> np.ndarray:
        """Generate modular sequence efficiently."""
        sequence = np.zeros(length, dtype=np.int64)
        current = 1
        sequence[0] = current
        
        for i in range(1, length):
            current = (current * base) % modulus
            sequence[i] = current
            
        return sequence
    
    def complex_modular_signal(self, base: int, modulus: int, length: int) -> np.ndarray:
        """Create complex wave signal."""
        mod_sequence = self.fast_modular_sequence(base, modulus, length)
        phases = 2.0 * np.pi * mod_sequence / modulus
        return np.exp(1j * phases)
    
    def autocorrelation_interference(self, signal: np.ndarray, max_shift: int) -> List[Tuple[int, float]]:
        """Detect periods through autocorrelation - SAME ALGORITHM as successful v2."""
        L = len(signal)
        interference_scores = []
        
        for d in range(1, min(max_shift, L // 2)):
            overlap_length = L - d
            correlation = np.sum(signal[:overlap_length] * np.conj(signal[d:d+overlap_length]))
            magnitude = np.abs(correlation)
            normalized_score = magnitude / overlap_length
            interference_scores.append((d, normalized_score))
        
        interference_scores.sort(key=lambda x: -x[1])
        return interference_scores[:20]  # Top 20 candidates
    
    def validate_and_extract_factors(self, base: int, modulus: int, period: int) -> Optional[Tuple[int, int]]:
        """Validate period and extract factors - SAME ALGORITHM as v2."""
        # Validate period
        if pow(base, period, modulus) != 1:
            return None
        
        # Extract factors for even periods
        if period % 2 != 0:
            return None
        
        half_period = period // 2
        y = pow(base, half_period, modulus)
        
        if y == 1 or y == modulus - 1:
            return None
        
        # Try both gcd(y-1, N) and gcd(y+1, N)
        factor1 = math.gcd(y - 1, modulus)
        factor2 = math.gcd(y + 1, modulus)
        
        for factor in [factor1, factor2]:
            if 1 < factor < modulus:
                complement = modulus // factor
                if factor * complement == modulus:
                    return (factor, complement)
        
        return None
    
    def multi_strategy_analysis(self, number: int, max_bases: int = 2000, time_limit_minutes: float = 30) -> Optional[Tuple[int, int]]:
        """
        Use MULTIPLE STRATEGIES instead of just scaling one approach.
        
        Strategy 1: Standard approach (like v2) with more bases
        Strategy 2: Adaptive signal lengths
        Strategy 3: Prime base targeting
        Strategy 4: Composite base exploration
        """
        start_time = time.time()
        time_limit_seconds = time_limit_minutes * 60
        
        print(f"🔍 Intelligent multi-strategy analysis")
        print(f"   Target: {number:,} ({number.bit_length()} bits)")
        print(f"   Max bases: {max_bases:,}")
        print(f"   Time limit: {time_limit_minutes} minutes")
        
        # Strategy 1: Standard approach with reasonable signal lengths
        bases_tested = 0
        for signal_length, search_depth in zip(self.signal_lengths, self.search_depths):
            if time.time() - start_time > time_limit_seconds * 0.7:  # Use 70% of time for standard
                break
                
            print(f"\n📊 Strategy 1: Signal={signal_length:,}, Search={search_depth:,}")
            
            # Generate diverse bases coprime to N
            bases = []
            for attempt in range(max_bases * 2):
                if len(bases) >= max_bases // len(self.signal_lengths):
                    break
                base = np.random.randint(2, min(10000, number))
                if math.gcd(base, number) == 1 and base not in bases:
                    bases.append(base)
            
            for base in bases:
                if time.time() - start_time > time_limit_seconds:
                    break
                    
                bases_tested += 1
                if bases_tested % 100 == 0:
                    elapsed = (time.time() - start_time) / 60
                    print(f"   Tested {bases_tested} bases ({elapsed:.1f}min)")
                
                # Generate signal
                signal = self.complex_modular_signal(base, number, signal_length)
                
                # Find interference peaks
                peaks = self.autocorrelation_interference(signal, search_depth)
                
                # Test top candidates
                for period, strength in peaks[:10]:
                    result = self.validate_and_extract_factors(base, number, period)
                    if result:
                        elapsed = time.time() - start_time
                        print(f"🎯 SUCCESS with Strategy 1!")
                        print(f"   Base: {base}, Period: {period}, Time: {elapsed:.1f}s")
                        return result
                
                # Clean up
                del signal
        
        # Strategy 2: Prime base targeting
        print(f"\n🎲 Strategy 2: Prime base targeting")
        prime_bases = [p for p in sympy.primerange(2, min(1000, number)) if math.gcd(p, number) == 1]
        
        for signal_length in [4096, 8192, 16384]:  # Focus on proven lengths
            if time.time() - start_time > time_limit_seconds:
                break
                
            for base in prime_bases[:min(200, len(prime_bases))]:
                if time.time() - start_time > time_limit_seconds:
                    break
                    
                signal = self.complex_modular_signal(base, number, signal_length)
                peaks = self.autocorrelation_interference(signal, signal_length // 4)
                
                for period, strength in peaks[:5]:
                    result = self.validate_and_extract_factors(base, number, period)
                    if result:
                        elapsed = time.time() - start_time
                        print(f"🎯 SUCCESS with Strategy 2 (prime base)!")
                        return result
                
                del signal
        
        elapsed = time.time() - start_time
        print(f"❌ All strategies failed after {elapsed:.1f}s")
        return None
    
    def factorize(self, number: int, time_limit_minutes: float = 30) -> Optional[Tuple[int, int]]:
        """Main factorization entry point."""
        # Quick checks first
        if number % 2 == 0:
            return (2, number // 2)
        
        # Try small primes
        small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        for p in small_primes:
            if number % p == 0:
                return (p, number // p)
        
        # Main intelligent analysis
        return self.multi_strategy_analysis(number, max_bases=2000, time_limit_minutes=time_limit_minutes)

def generate_rsa_challenge(bits: int) -> int:
    """Generate RSA challenge of specified bit length."""
    p_bits = bits // 2
    q_bits = bits - p_bits
    
    p = sympy.randprime(2**(p_bits-1), 2**p_bits)
    q = sympy.randprime(2**(q_bits-1), 2**q_bits)
    
    return p * q

def intelligent_rsa_test():
    """Test the intelligent scaling approach."""
    print("🧠 INTELLIGENT WAVE INTERFERENCE SCALING TEST")
    print("=" * 60)
    print("Key insight: Scale strategies, not just signal size")
    print("Focus: More bases, adaptive lengths, multiple approaches")
    print()
    
    factorizer = IntelligentWaveFactorizer()
    
    # Test on challenging sizes with reasonable time limits
    test_sizes = [20, 24, 28, 32, 36, 40, 44, 48]
    results = []
    
    for bit_size in test_sizes:
        print(f"\n{'='*50}")
        print(f"TESTING RSA-{bit_size}")
        print(f"{'='*50}")
        
        # Generate challenge
        n = generate_rsa_challenge(bit_size)
        print(f"Challenge: {n:,}")
        
        # Determine time limit based on size
        if bit_size <= 32:
            time_limit = 3
        elif bit_size <= 40:
            time_limit = 8
        else:
            time_limit = 20
        
        # Attempt factorization
        start_time = time.time()
        result = factorizer.factorize(n, time_limit)
        elapsed = time.time() - start_time
        
        if result:
            p, q = result
            print(f"\n🎉 SUCCESS: {n:,} = {p:,} × {q:,}")
            print(f"⏱️ Time: {elapsed:.1f}s")
            results.append((bit_size, "SUCCESS", elapsed))
        else:
            print(f"\n❌ FAILED: No factors found")
            print(f"⏱️ Time: {elapsed:.1f}s")
            results.append((bit_size, "FAILED", elapsed))
        
        # Memory cleanup
        gc.collect()
    
    # Results summary
    print(f"\n{'='*60}")
    print("INTELLIGENT SCALING RESULTS")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in results if r[1] == "SUCCESS")
    total_count = len(results)
    
    print(f"Success rate: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")
    
    for bit_size, status, time_taken in results:
        emoji = "✅" if status == "SUCCESS" else "❌"
        print(f"{emoji} RSA-{bit_size}: {status} ({time_taken:.1f}s)")
    
    if success_count > 0:
        max_success = max(r[0] for r in results if r[1] == "SUCCESS")
        print(f"\n🏆 Maximum RSA size factored: RSA-{max_success}")
        
        if max_success >= 40:
            print("🔥 Excellent results! Ready to scale to RSA-64")
        elif max_success >= 32:
            print("👍 Good progress, continue refinement")
        else:
            print("⚠️ Limited success, need algorithm improvements")
    
    return results

if __name__ == "__main__":
    intelligent_rsa_test()
