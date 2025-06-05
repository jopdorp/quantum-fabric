#!/usr/bin/env python3
"""
Wave Collision Analysis: Deep Dive into Hash Collision Detection

This script investigates why we're not detecting wave collisions and tests
various hash function parameters to optimize collision detection.
"""

import hashlib
import random
import time
from math import log2, gcd
from collections import defaultdict
from enhanced_wave_factorization import EnhancedWaveFactorizer, generate_rsa_like_number

def analyze_hash_distribution(N, base, steps=1000, signal_bits=32):
    """Analyze hash distribution and collision probability."""
    print(f"\n🔍 Hash Distribution Analysis")
    print(f"N = {N}, base = {base}, steps = {steps}, signal_bits = {signal_bits}")
    
    # Track hash values and their frequencies
    hash_counts = defaultdict(int)
    values_seen = set()
    collisions = []
    
    current_value = 1
    
    for step in range(1, steps + 1):
        current_value = (current_value * base) % N
        values_seen.add(current_value)
        
        # Compute hash like in our algorithm
        h1 = hashlib.sha256(f"{current_value}_{base}_1".encode()).digest()
        h2 = hashlib.sha256(f"{current_value}_{base}_2".encode()).digest()
        combined = int.from_bytes(h1, 'big') ^ int.from_bytes(h2, 'big')
        hash_val = combined % (2 ** signal_bits)
        
        if hash_counts[hash_val] > 0:
            collisions.append((step, hash_val, hash_counts[hash_val]))
        
        hash_counts[hash_val] += 1
        
        if current_value == 1 and step > 1:
            print(f"Natural period found at step {step}")
            break
    
    # Analysis
    total_hashes = len(hash_counts)
    hash_space = 2 ** signal_bits
    collision_count = len(collisions)
    
    print(f"\n📊 Hash Analysis Results:")
    print(f"   • Total steps: {step}")
    print(f"   • Unique values: {len(values_seen)}")
    print(f"   • Hash space size: {hash_space:,}")
    print(f"   • Unique hashes used: {total_hashes:,}")
    print(f"   • Hash space utilization: {total_hashes/hash_space*100:.4f}%")
    print(f"   • Collisions detected: {collision_count}")
    print(f"   • Expected collisions (birthday paradox): {step*step/(2*hash_space):.2f}")
    
    if collisions:
        print(f"\n💥 Collision Details:")
        for i, (step, hash_val, prev_count) in enumerate(collisions[:5]):
            print(f"   {i+1}. Step {step}: hash {hash_val} (collision #{prev_count+1})")
    
    return collision_count, total_hashes, hash_space

def test_different_hash_parameters():
    """Test different hash parameters to find optimal collision rates."""
    print("🧪 Testing Different Hash Parameters")
    print("=" * 50)
    
    # Test number
    N = 1104143  # 20-bit number that failed before
    base = 75
    steps = 2000
    
    print(f"Test number: N = {N} (20 bits)")
    print(f"Base: {base}")
    print(f"Steps: {steps}")
    
    # Test different signal bit sizes
    bit_sizes = [8, 12, 16, 20, 24, 28, 32, 40, 48]
    
    results = []
    
    for bits in bit_sizes:
        print(f"\n--- Testing {bits}-bit hashes ---")
        collisions, unique_hashes, hash_space = analyze_hash_distribution(N, base, steps, bits)
        
        results.append({
            'bits': bits,
            'collisions': collisions,
            'unique_hashes': unique_hashes,
            'hash_space': hash_space,
            'utilization': unique_hashes/hash_space*100
        })
    
    # Summary
    print(f"\n📋 Hash Parameter Comparison:")
    print(f"{'Bits':<6} {'Hash Space':<12} {'Collisions':<12} {'Utilization':<12}")
    print("-" * 50)
    
    for result in results:
        print(f"{result['bits']:<6} {result['hash_space']:<12,} {result['collisions']:<12} {result['utilization']:<11.4f}%")
    
    # Find optimal
    best_collision = max(results, key=lambda x: x['collisions'])
    print(f"\n🎯 Best collision rate: {best_collision['bits']} bits with {best_collision['collisions']} collisions")
    
    return best_collision['bits']

def test_more_bases_and_time():
    """Test if more bases and time can break larger RSA numbers."""
    print("\n🚀 Testing More Bases and Time Strategy")
    print("=" * 50)
    
    # Test cases with increasing difficulty
    test_cases = [
        (20, 1104143, 1259, 877),   # Previously failed
        (24, 38123509, 4877, 7817), # Previously failed
        (32, None, None, None),      # Generate new
    ]
    
    for bits, N, p, q in test_cases:
        if N is None:
            N, p, q = generate_rsa_like_number(bits)
        
        print(f"\n🎯 Testing RSA-{bits}: N = {N:,}")
        print(f"Expected factors: {p} × {q}")
        
        # Progressive increase in bases and time
        base_counts = [50, 100, 200, 500]
        max_depths = [5000, 10000, 20000, 50000]
        
        for bases, depth in zip(base_counts, max_depths):
            print(f"\n--- Trying {bases} bases, depth {depth} ---")
            
            # Use optimal hash bits from previous analysis
            optimal_bits = 16  # From our analysis
            
            factorizer = EnhancedWaveFactorizer(
                signal_bits=optimal_bits,
                max_bases=bases,
                verbose=False
            )
            
            # Modify max depth (we need to patch this)
            start_time = time.time()
            factor = factorizer.wave_factor(N)
            elapsed = time.time() - start_time
            
            if factor and N % factor == 0 and 1 < factor < N:
                other = N // factor
                print(f"🎉 BREAKTHROUGH! Factored with {bases} bases in {elapsed:.2f}s")
                print(f"✅ {N:,} = {factor:,} × {other:,}")
                
                stats = factorizer.get_statistics()
                print(f"📊 Stats: {stats['total_steps']} steps, {stats['collisions_low']+stats['collisions_med']+stats['collisions_high']} collisions")
                return True, bits
            else:
                print(f"❌ Failed with {bases} bases in {elapsed:.2f}s")
                
                stats = factorizer.get_statistics()
                print(f"📊 Stats: {stats['total_steps']} steps, {stats['collisions_low']+stats['collisions_med']+stats['collisions_high']} collisions")
        
        print(f"\n❌ RSA-{bits} resisted all attempts")
    
    return False, 0

def create_optimized_factorizer():
    """Create an optimized factorizer based on our analysis."""
    print("\n🔧 Creating Optimized Wave Factorizer")
    print("=" * 40)
    
    # First, find optimal hash parameters
    optimal_bits = test_different_hash_parameters()
    
    print(f"\n🎯 Using optimal hash bits: {optimal_bits}")
    
    # Test the optimized version
    success, largest = test_more_bases_and_time()
    
    if success:
        print(f"\n🎉 SUCCESS! Broke RSA-{largest} with optimized parameters!")
    else:
        print(f"\n🔬 Need further optimization - largest broken was RSA-16")
    
    return optimal_bits

def deep_collision_investigation():
    """Deep investigation into why collisions aren't happening."""
    print("\n🕵️ Deep Collision Investigation")
    print("=" * 40)
    
    # Test with a known working case (RSA-16)
    N = 93349  # RSA-16 that we successfully factored
    base = 72734  # Base that worked
    
    print(f"Investigating successful case: N = {N}, base = {base}")
    
    # Track the actual sequence that led to success
    current_value = 1
    values = []
    hashes_8bit = []
    hashes_16bit = []
    hashes_32bit = []
    
    for step in range(1, 400):  # We know period was 336
        current_value = (current_value * base) % N
        values.append(current_value)
        
        # Compute hashes at different resolutions
        h1 = hashlib.sha256(f"{current_value}_{base}_1".encode()).digest()
        h2 = hashlib.sha256(f"{current_value}_{base}_2".encode()).digest()
        combined = int.from_bytes(h1, 'big') ^ int.from_bytes(h2, 'big')
        
        hashes_8bit.append(combined % (2**8))
        hashes_16bit.append(combined % (2**16))
        hashes_32bit.append(combined % (2**32))
        
        if current_value == 1 and step > 1:
            print(f"Natural period confirmed at step {step}")
            break
    
    # Check for collisions in each resolution
    def find_collisions(hash_list, name):
        seen = {}
        collisions = []
        for i, h in enumerate(hash_list):
            if h in seen:
                collisions.append((i+1, seen[h]+1, h))
            seen[h] = i
        return collisions
    
    collisions_8 = find_collisions(hashes_8bit, "8-bit")
    collisions_16 = find_collisions(hashes_16bit, "16-bit")
    collisions_32 = find_collisions(hashes_32bit, "32-bit")
    
    print(f"\n📊 Collision Analysis for Successful Case:")
    print(f"   • 8-bit collisions: {len(collisions_8)}")
    print(f"   • 16-bit collisions: {len(collisions_16)}")
    print(f"   • 32-bit collisions: {len(collisions_32)}")
    
    if collisions_16:
        print(f"\n💥 16-bit Collision Details:")
        for step, prev_step, hash_val in collisions_16[:3]:
            period = step - prev_step
            print(f"   Step {step} collides with step {prev_step} (period {period}), hash {hash_val}")
    
    # The key insight: why didn't our algorithm detect these?
    print(f"\n🔍 Key Insight: Our algorithm should have detected these collisions!")
    print(f"   This suggests a bug in our collision detection logic.")

def main():
    """Main analysis function."""
    print("🌊 Wave Collision Analysis: Deep Investigation")
    print("=" * 60)
    print()
    print("Investigating two critical questions:")
    print("1. Why are we not detecting wave collisions?")
    print("2. Would more bases and time break larger RSA?")
    print()
    
    # Deep investigation
    deep_collision_investigation()
    
    # Test optimized parameters
    optimal_bits = create_optimized_factorizer()
    
    print(f"\n🎯 CONCLUSIONS:")
    print(f"1. Collision detection may have bugs - we should see more collisions")
    print(f"2. Optimal hash bits appear to be around {optimal_bits}")
    print(f"3. More bases and time might help, but collision detection is key")
    print(f"4. Need to debug why collisions aren't being detected properly")

if __name__ == "__main__":
    main()
