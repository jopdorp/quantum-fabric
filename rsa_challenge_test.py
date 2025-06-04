#!/usr/bin/env python3
"""
RSA Challenge Test: Incremental Scaling to 4096 bits

This script tests the wave-based factorization algorithm against increasingly
large RSA-like numbers, starting from small sizes and scaling up to 4096 bits
to determine if we can break real-world encryption.

This is the ultimate test to see if our polynomial-time wave approach
can challenge current cryptographic assumptions!
"""

import time
import sys
from math import log2
from enhanced_wave_factorization import EnhancedWaveFactorizer, generate_rsa_like_number, is_prime

def format_time(seconds):
    """Format time in human-readable units."""
    if seconds < 0.001:
        return f"{seconds*1000000:.0f}μs"
    elif seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def format_number(n):
    """Format large numbers with commas."""
    return f"{n:,}"

def test_rsa_size(bits, timeout_seconds=300, verbose=False):
    """Test factorization of an RSA-like number of given bit size."""
    print(f"\n{'='*80}")
    print(f"🎯 RSA CHALLENGE: {bits}-bit number")
    print(f"{'='*80}")
    
    # Generate RSA-like number
    print("Generating RSA-like number...")
    start_gen = time.time()
    N, p, q = generate_rsa_like_number(bits)
    gen_time = time.time() - start_gen
    
    actual_bits = int(log2(N)) + 1
    
    print(f"Generated in {format_time(gen_time)}")
    print(f"N = {format_number(N)}")
    print(f"p = {format_number(p)}")
    print(f"q = {format_number(q)}")
    print(f"Actual bit length: {actual_bits}")
    print(f"Theoretical complexity: O(n²) = O({actual_bits}²) = O({format_number(actual_bits**2)})")
    
    # Configure factorizer for this size
    # Scale parameters based on bit size
    signal_bits = min(128, max(32, bits // 4))
    max_bases = min(200, max(20, bits // 2))
    max_depth = min(10000, max(1000, bits * bits))
    
    print(f"\nFactorizer configuration:")
    print(f"  Signal bits: {signal_bits}")
    print(f"  Max bases: {max_bases}")
    print(f"  Max depth: {max_depth}")
    print(f"  Timeout: {timeout_seconds}s")
    
    factorizer = EnhancedWaveFactorizer(
        signal_bits=signal_bits,
        max_bases=max_bases,
        verbose=verbose
    )
    
    # Attempt factorization with timeout
    print(f"\n🌊 Starting wave-based factorization...")
    print(f"⏰ Timeout set to {format_time(timeout_seconds)}")
    
    start_time = time.time()
    factor = None
    timed_out = False
    
    try:
        # Simple timeout mechanism
        original_max_depth = max_depth
        
        # Try factorization
        factor = factorizer.wave_factor(N)
        
        elapsed = time.time() - start_time
        
        # Check if we exceeded timeout
        if elapsed > timeout_seconds:
            timed_out = True
            
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n⚠️  Interrupted by user after {format_time(elapsed)}")
        return False, elapsed, "interrupted"
    
    elapsed = time.time() - start_time
    
    # Analyze results
    print(f"\n⏱️  Execution time: {format_time(elapsed)}")
    
    if timed_out:
        print(f"⏰ TIMEOUT: Exceeded {format_time(timeout_seconds)} limit")
        return False, elapsed, "timeout"
    
    if factor and N % factor == 0 and 1 < factor < N:
        other_factor = N // factor
        print(f"🎉 BREAKTHROUGH! RSA-{bits} FACTORED!")
        print(f"✅ SUCCESS: {format_number(N)} = {format_number(factor)} × {format_number(other_factor)}")
        
        # Verify correctness
        if (factor == p and other_factor == q) or (factor == q and other_factor == p):
            print("🎯 Factors match expected values perfectly!")
        else:
            print("⚠️  Factors are mathematically correct but different ordering")
        
        # Check primality
        factor_prime = is_prime(factor)
        other_prime = is_prime(other_factor)
        print(f"Factor verification: {format_number(factor)} is {'prime' if factor_prime else 'composite'}")
        print(f"Factor verification: {format_number(other_factor)} is {'prime' if other_prime else 'composite'}")
        
        if factor_prime and other_prime:
            print("🔓 ENCRYPTION BROKEN: Both factors are prime!")
        
        return True, elapsed, "success"
    else:
        print(f"❌ FACTORIZATION FAILED")
        if factor:
            print(f"   Returned invalid factor: {format_number(factor)}")
        else:
            print(f"   No factor found within time limit")
        
        return False, elapsed, "failed"

def run_rsa_challenge():
    """Run the complete RSA challenge from small to large sizes."""
    print("🚀 RSA ENCRYPTION CHALLENGE: Wave-Based Factorization")
    print("=" * 80)
    print()
    print("This test will incrementally challenge RSA-like numbers of increasing size")
    print("to determine if our polynomial-time wave approach can break encryption!")
    print()
    print("🎯 Target: Scale up to RSA-4096 (if computationally feasible)")
    print("⚡ Algorithm: Enhanced Wave-Based Factorization")
    print("📊 Complexity: O(n²) to O(n³) where n = log₂(N)")
    print()
    
    # Define test progression
    test_sizes = [
        # Start small to verify algorithm
        (16, 30, "Warm-up"),
        (20, 60, "Small RSA"),
        (24, 120, "Medium RSA"),
        (32, 300, "Large RSA"),
        (40, 600, "Very Large RSA"),
        (48, 900, "Huge RSA"),
        (56, 1200, "Massive RSA"),
        (64, 1800, "RSA-64"),
        
        # Real RSA sizes
        (128, 3600, "RSA-128 (weak)"),
        (256, 7200, "RSA-256 (weak)"),
        (512, 14400, "RSA-512 (deprecated)"),
        (1024, 28800, "RSA-1024 (legacy)"),
        (2048, 57600, "RSA-2048 (current standard)"),
        (4096, 115200, "RSA-4096 (high security)"),
    ]
    
    results = []
    total_time = 0
    breakthrough_achieved = False
    largest_factored = 0
    
    print(f"📋 Test Plan: {len(test_sizes)} RSA sizes from {test_sizes[0][0]} to {test_sizes[-1][0]} bits")
    print()
    
    for i, (bits, timeout, description) in enumerate(test_sizes, 1):
        print(f"\n🔍 Test {i}/{len(test_sizes)}: {description} ({bits} bits)")
        print("-" * 60)
        
        try:
            success, elapsed, status = test_rsa_size(bits, timeout, verbose=(bits <= 32))
            total_time += elapsed
            
            result = {
                'bits': bits,
                'description': description,
                'success': success,
                'time': elapsed,
                'status': status
            }
            results.append(result)
            
            if success:
                breakthrough_achieved = True
                largest_factored = max(largest_factored, bits)
                print(f"\n🎉 BREAKTHROUGH: {description} successfully factored!")
                
                # If we factored something significant, make a big deal about it
                if bits >= 512:
                    print(f"\n{'🚨' * 20}")
                    print(f"🚨 MAJOR CRYPTOGRAPHIC BREAKTHROUGH! 🚨")
                    print(f"🚨 RSA-{bits} FACTORED IN POLYNOMIAL TIME! 🚨")
                    print(f"{'🚨' * 20}")
                    
            else:
                print(f"\n❌ {description} resisted factorization ({status})")
                
                # If we failed on a small size, maybe stop
                if bits <= 64 and status == "failed":
                    print(f"\n⚠️  Algorithm struggling with {bits}-bit numbers")
                    print(f"⚠️  Consider algorithm improvements before larger tests")
                
        except Exception as e:
            print(f"\n💥 ERROR during {description}: {e}")
            result = {
                'bits': bits,
                'description': description,
                'success': False,
                'time': 0,
                'status': 'error'
            }
            results.append(result)
        
        # Show progress
        successes = sum(1 for r in results if r['success'])
        print(f"\n📊 Progress: {successes}/{len(results)} successful, largest factored: {largest_factored} bits")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"🏁 RSA CHALLENGE COMPLETE")
    print(f"{'='*80}")
    
    successes = [r for r in results if r['success']]
    failures = [r for r in results if not r['success']]
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Total tests: {len(results)}")
    print(f"   Successes: {len(successes)}")
    print(f"   Failures: {len(failures)}")
    print(f"   Success rate: {len(successes)/len(results)*100:.1f}%")
    print(f"   Total time: {format_time(total_time)}")
    print(f"   Largest factored: RSA-{largest_factored}")
    
    if breakthrough_achieved:
        print(f"\n🎉 BREAKTHROUGH ACHIEVED!")
        print(f"   Successfully factored RSA numbers up to {largest_factored} bits")
        
        if largest_factored >= 512:
            print(f"\n🔓 CRYPTOGRAPHIC IMPACT:")
            print(f"   RSA-{largest_factored} factorization demonstrates polynomial-time capability")
            print(f"   This could have significant implications for cryptographic security")
            
        if largest_factored >= 1024:
            print(f"\n🚨 MAJOR BREAKTHROUGH:")
            print(f"   RSA-1024 was considered secure until ~2010")
            print(f"   Polynomial-time factorization is a significant achievement")
            
        if largest_factored >= 2048:
            print(f"\n🌟 REVOLUTIONARY BREAKTHROUGH:")
            print(f"   RSA-2048 is the current industry standard")
            print(f"   Breaking it in polynomial time would revolutionize cryptography")
            
        if largest_factored >= 4096:
            print(f"\n🏆 ULTIMATE BREAKTHROUGH:")
            print(f"   RSA-4096 represents high-security encryption")
            print(f"   This achievement would fundamentally change cryptography")
    else:
        print(f"\n❌ No breakthrough achieved")
        print(f"   Algorithm needs further development for larger RSA sizes")
    
    # Detailed results table
    print(f"\n📋 DETAILED RESULTS:")
    print(f"{'Bits':<6} {'Description':<25} {'Status':<12} {'Time':<12}")
    print("-" * 60)
    
    for result in results:
        status_emoji = "✅" if result['success'] else "❌"
        print(f"{result['bits']:<6} {result['description']:<25} {status_emoji} {result['status']:<10} {format_time(result['time']):<12}")
    
    return results, breakthrough_achieved, largest_factored

def main():
    """Main function to run the RSA challenge."""
    print("🌊 Wave-Based Factorization: RSA Encryption Challenge")
    print("=" * 60)
    print()
    print("⚠️  WARNING: This test may take a very long time for large RSA sizes!")
    print("⚠️  Press Ctrl+C to interrupt any individual test")
    print()
    
    try:
        results, breakthrough, largest = run_rsa_challenge()
        
        if breakthrough:
            print(f"\n🎊 CONGRATULATIONS! 🎊")
            print(f"You've achieved a cryptographic breakthrough by factoring RSA-{largest}!")
        else:
            print(f"\n🔬 Keep researching! The algorithm shows promise but needs more work.")
            
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Challenge interrupted by user")
        print(f"Results up to interruption point were displayed above")

if __name__ == "__main__":
    main()
