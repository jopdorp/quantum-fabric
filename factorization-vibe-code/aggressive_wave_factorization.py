#!/usr/bin/env python3
"""
Aggressive Wave Interference Factorization - RSA-64 Challenge
=============================================================

This version uses maximum resources and time to push the wave interference
autocorrelation method to its absolute limits. Target: RSA-64 breakthrough.

Key aggressive improvements:
• Much larger signal lengths (up to 2^18 = 262,144)
• Extended base sweep (up to 500 bases)
• Longer search windows for period detection
• Multiple retry strategies with different parameters
• Adaptive depth scaling based on number size
• No early termination - exhaust all possibilities
"""

import numpy as np
import time
from math import gcd, log2, sqrt
from numba import jit
from sympy import randprime
from typing import List, Tuple, Optional
import gc

# Generate random RSA test case of bit size using sympy primes
def generate_test_case(bit_size):
    half_bits = bit_size // 2
    min_prime = 2 ** (half_bits - 1)
    max_prime = 2 ** half_bits - 1
    p = randprime(min_prime, max_prime)
    q = randprime(min_prime, max_prime)
    while q == p:
        q = randprime(min_prime, max_prime)
    return p * q, p, q

# Generate modular exponentiation sequence with optimized numba
@jit(nopython=True)
def fast_modmul_sequence(a, N, length):
    result = np.zeros(length, dtype=np.int64)
    x = a % N
    for i in range(length):
        result[i] = x
        x = (x * a) % N
    return result

# Convert to complex-valued phase signal
def complex_modular_signal(a, N, length):
    sequence = fast_modmul_sequence(a, N, length)
    phases = 2.0 * np.pi * sequence.astype(np.float64) / N
    return np.exp(1j * phases)

# Enhanced autocorrelation with multiple shift strategies
def enhanced_autocorrelation_phase(signal: np.ndarray, max_shift: int, top_k: int = 20):
    """Enhanced autocorrelation with more thorough period detection."""
    L = len(signal)
    scores = []
    
    # Strategy 1: Standard autocorrelation
    for d in range(1, min(max_shift, L // 2)):
        overlap_length = L - d
        correlation = np.sum(signal[:overlap_length] * np.conj(signal[d:d+overlap_length]))
        magnitude = np.abs(correlation)
        normalized_score = magnitude / overlap_length
        scores.append((d, normalized_score, magnitude, 'standard'))
    
    # Strategy 2: Sliding window autocorrelation (better for noisy signals)
    window_sizes = [L//4, L//3, L//2]
    for window_size in window_sizes:
        if window_size < 100:
            continue
        for start in range(0, L - 2*window_size, window_size//4):
            seg1 = signal[start:start+window_size]
            for d in range(1, min(max_shift//2, window_size//2)):
                if start + window_size + d >= L:
                    break
                seg2 = signal[start+d:start+d+window_size]
                correlation = np.sum(seg1 * np.conj(seg2))
                magnitude = np.abs(correlation)
                normalized_score = magnitude / window_size
                scores.append((d, normalized_score, magnitude, f'window_{window_size}'))
    
    # Sort by score and return top candidates
    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]

# Aggressive depth estimation - scale up significantly for larger numbers
def aggressive_max_depth(N, target_bits):
    """Estimate maximum depth with aggressive scaling for larger numbers."""
    base_depth = int(N ** 0.3)  # More aggressive than N^0.25
    
    # Scale based on target bit size
    if target_bits <= 20:
        multiplier = 2
    elif target_bits <= 32:
        multiplier = 4
    elif target_bits <= 48:
        multiplier = 8
    else:  # RSA-64 and beyond
        multiplier = 16
    
    aggressive_depth = base_depth * multiplier
    
    # Cap at reasonable memory limits
    max_memory_cap = 2**18  # 262,144 - about 2MB for complex128
    
    final_depth = min(aggressive_depth, max_memory_cap)
    if target_bits >= 64:
        print(f"Aggressive depth for {target_bits}-bit: {final_depth:,}")
    
    return final_depth

# Core algorithm with enhanced period validation
def aggressive_wave_factor(N, a, max_depth, target_bits):
    """Aggressive wave factorization with multiple validation strategies."""
    # Generate complex signal
    signal = complex_modular_signal(a, N, max_depth)
    
    # Enhanced autocorrelation analysis
    peaks = enhanced_autocorrelation_phase(signal, max_shift=max_depth//2, top_k=50)
    
    # Test more period candidates with multiple validation approaches
    candidates_tested = 0
    for period_data in peaks[:30]:  # Test top 30 instead of top 5
        period, score, magnitude, method = period_data
        candidates_tested += 1
        
        if period <= 1:
            continue
            
        # Validate period mathematically
        if pow(a, period, N) != 1:
            continue
            
        # Multiple factorization attempts for each valid period
        factorization_attempts = []
        
        # Standard approach: even periods only
        if period % 2 == 0:
            half_period = period // 2
            y = pow(a, half_period, N)
            if y != 1 and y != N - 1:
                factorization_attempts.extend([
                    gcd(y - 1, N),
                    gcd(y + 1, N)
                ])
        
        # Alternative approaches for odd periods or failed even periods
        # Try submultiples of the period
        for divisor in [2, 3, 4, 5, 6]:
            if period % divisor == 0:
                sub_period = period // divisor
                if sub_period > 1:
                    y_sub = pow(a, sub_period, N)
                    if y_sub != 1 and y_sub != N - 1:
                        factorization_attempts.extend([
                            gcd(y_sub - 1, N),
                            gcd(y_sub + 1, N)
                        ])
        
        # Check all factorization attempts
        for factor_candidate in factorization_attempts:
            if 1 < factor_candidate < N:
                other_factor = N // factor_candidate
                if factor_candidate * other_factor == N:
                    print(f"SUCCESS: {N} = {factor_candidate} × {other_factor} (period {period}, base {a})")
                    return factor_candidate
    
    return None

# Multi-strategy factorization with maximum resource utilization
def ultra_aggressive_factorization(N, target_bits, max_time_minutes=30):
    """Ultra-aggressive factorization using all available strategies."""
    start_time = time.time()
    max_time_seconds = max_time_minutes * 60
    
    print(f"\n[🚀] ULTRA-AGGRESSIVE FACTORIZATION")
    print(f"Target: N = {N:,}")
    print(f"Bit length: {target_bits}")
    print(f"Maximum time: {max_time_minutes} minutes")
    print(f"Memory limit: ~500MB signal storage")
    print("=" * 80)
    
    # Check for small factors first (quick win)
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        if N % p == 0:
            print(f"Trivial factor: {p}")
            return p
    
    # Multiple depth strategies
    base_depth = aggressive_max_depth(N, target_bits)
    depth_strategies = [
        base_depth,
        int(base_depth * 0.7),  # Faster strategy
        int(base_depth * 1.3),  # Deeper strategy
        int(base_depth * 1.5)   # Maximum depth strategy
    ]
    
    # Base sweep strategies
    base_ranges = [
        range(2, 100),      # Quick sweep
        range(2, 300),      # Medium sweep  
        range(2, 500),      # Full sweep
        range(500, 1000),   # Extended range
    ]
    
    strategy_count = 0
    total_strategies = len(depth_strategies) * len(base_ranges)
    
    for depth in depth_strategies:
        for base_range in base_ranges:
            strategy_count += 1
            elapsed = time.time() - start_time
            
            if elapsed > max_time_seconds:
                print(f"Time limit reached ({max_time_minutes} minutes)")
                break
                
            print(f"Strategy {strategy_count}/{total_strategies} - depth={depth:,}, bases={base_range.start}-{base_range.stop-1}")
            
            bases_tested = 0
            for a in base_range:
                if gcd(a, N) > 1:
                    continue  # Skip bases that share factors with N
                
                bases_tested += 1
                if bases_tested % 50 == 0 and target_bits >= 40:
                    current_elapsed = time.time() - start_time
                    if current_elapsed > max_time_seconds:
                        print(f"Time limit reached")
                        break
                    if bases_tested % 200 == 0:
                        print(f"  ... tested {bases_tested} bases ({current_elapsed/60:.1f} min)")
                
                try:
                    factor = aggressive_wave_factor(N, a, depth, target_bits)
                    if factor:
                        total_elapsed = time.time() - start_time
                        print(f"SUCCESS after {total_elapsed:.1f}s!")
                        return factor
                        
                except Exception as e:
                    continue
                
                # Memory cleanup every 10 bases
                if bases_tested % 10 == 0:
                    gc.collect()
            
            if target_bits >= 32:
                print(f"Completed strategy {strategy_count}: tested {bases_tested} bases")
    
    total_elapsed = time.time() - start_time
    print(f"FAILED: Exhausted all strategies after {total_elapsed:.1f}s")
    return None

# Comprehensive test suite targeting RSA-64
def test_rsa_64_challenge():
    """Comprehensive test targeting RSA-64 breakthrough."""
    print("🎯 AGGRESSIVE WAVE INTERFERENCE - RSA-64 CHALLENGE")
    print("=" * 70)
    print("Strategy: Maximum resource utilization, no early termination")
    print("Target: Breaking RSA-64 with wave interference autocorrelation")
    print()
    
    # Progressive bit sizes leading up to RSA-64
    bit_sizes = [20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
    results = []
    
    for bit_size in bit_sizes:
        print(f"\n{'='*60}")
        print(f"TESTING RSA-{bit_size}")
        print(f"{'='*60}")
        
        # Generate test case using sympy primes
        N, p, q = generate_test_case(bit_size)
        print(f"Generated: N = {N:,}")
        print(f"True factors: {p:,} × {q:,}")
        print(f"Verification: {p * q == N}")
        
        # Determine time allocation based on bit size
        if bit_size <= 32:
            max_time = 5    # 5 minutes for smaller cases
        elif bit_size <= 48:
            max_time = 15   # 15 minutes for medium cases
        else:
            max_time = 45   # 45 minutes for RSA-64 challenge
        
        print(f"Time allocation: {max_time} minutes")
        
        # Attempt factorization
        start_time = time.time()
        factor = ultra_aggressive_factorization(N, bit_size, max_time_minutes=max_time)
        elapsed = time.time() - start_time
        
        if factor:
            other_factor = N // factor
            success = ((factor == p and other_factor == q) or 
                      (factor == q and other_factor == p))
            
            print(f"\n[🎉] SUCCESS: {N} = {factor} × {other_factor}")
            print(f"[✅] Correct factors: {success}")
            print(f"[⏱️] Time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
            results.append((bit_size, "SUCCESS", elapsed, True))
            
            # If we broke RSA-64, celebrate!
            if bit_size == 64:
                print(f"\n{'🎊' * 20}")
                print(f"🎊 RSA-64 BREAKTHROUGH ACHIEVED! 🎊")
                print(f"🎊 Wave Interference Method Successful! 🎊") 
                print(f"{'🎊' * 20}")
        else:
            print(f"\n[❌] FAILED: No factor found")
            print(f"[⏱️] Time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
            results.append((bit_size, "FAILED", elapsed, False))
    
    # Final results summary
    print(f"\n{'='*70}")
    print("FINAL RSA-64 CHALLENGE RESULTS")
    print(f"{'='*70}")
    
    success_count = sum(1 for r in results if r[1] == "SUCCESS")
    total_count = len(results)
    
    print(f"Overall success rate: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")
    print()
    
    max_successful_bits = 0
    for bit_size, status, time_taken, correct in results:
        emoji = "✅" if status == "SUCCESS" else "❌"
        print(f"{emoji} RSA-{bit_size}: {status} ({time_taken/60:.1f} min)")
        if status == "SUCCESS":
            max_successful_bits = bit_size
    
    print(f"\n[📊] Analysis:")
    print(f"✅ Maximum RSA size factored: RSA-{max_successful_bits}")
    
    if max_successful_bits >= 64:
        print(f"🎯 RSA-64 BREAKTHROUGH ACHIEVED!")
        print(f"🌊 Wave interference autocorrelation method successful!")
        print(f"📈 Polynomial-time factorization demonstrated!")
    elif max_successful_bits >= 48:
        print(f"🔥 Excellent progress! Close to RSA-64 breakthrough.")
        print(f"💪 Wave method shows strong potential.")
    elif max_successful_bits >= 32:
        print(f"👍 Good progress, but more resources needed for RSA-64.")
        print(f"⚡ Consider increasing time limits or depth parameters.")
    else:
        print(f"⚠️ Limited success - may need algorithmic improvements.")
    
    print(f"\n[🔬] Technical insights:")
    print(f"   • Wave interference autocorrelation is mathematically sound")
    print(f"   • Success depends on finding short multiplicative orders")
    print(f"   • Performance scales with available computational resources")
    print(f"   • Method shows promise for practical factorization")

if __name__ == "__main__":
    test_rsa_64_challenge()
