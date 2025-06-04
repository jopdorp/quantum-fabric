#!/usr/bin/env python3
"""
Incremental testing of numpy wave factorization on progressively larger numbers
"""

import time
from math import log2
from numpy_wave_interference_factorization import numpy_wave_factor

def test_incremental_sizes():
    """Test factorization on incrementally larger numbers."""
    print("🌊 Incremental Size Testing: NumPy Wave Factorization")
    print("=" * 60)
    print("Testing progressively larger numbers to find scaling limits\n")
    
    # Test cases arranged by bit length
    test_cases = [
        # Very small (6-8 bits)
        (35, 5, 7, "6-bit: 5 × 7"),
        (77, 7, 11, "7-bit: 7 × 11"),
        (143, 11, 13, "8-bit: 11 × 13"),
        
        # Small (9-11 bits)
        (323, 17, 19, "9-bit: 17 × 19"),
        (667, 23, 29, "10-bit: 23 × 29"),
        (1147, 31, 37, "11-bit: 31 × 37"),
        
        # Medium (12-14 bits)
        (2491, 47, 53, "12-bit: 47 × 53"),
        (4687, 67, 71, "13-bit: 67 × 71"),
        (6557, 79, 83, "13-bit: 79 × 83"),
        
        # Larger (15-17 bits)
        (10403, 101, 103, "14-bit: 101 × 103"),
        (14717, 119, 127, "14-bit: twin primes"),
        (22201, 139, 149, "15-bit: 139 × 149"),
        (26611, 157, 169, "15-bit: 157 × 169"),
        
        # Challenge (18-20 bits)
        (57277, 239, 241, "16-bit: 239 × 241"),
        (108077, 317, 331, "17-bit: 317 × 331"),
        (160969, 389, 409, "18-bit: 389 × 409"),
        (215441, 457, 471, "18-bit: 457 × 471"),
    ]
    
    results = []
    total_time = 0
    
    for i, (N, p, q, description) in enumerate(test_cases, 1):
        bit_length = int(log2(N)) + 1
        print(f"\n{'='*50}")
        print(f"🎯 Test {i:2d}: {description}")
        print(f"    N = {N:,} (actual: {p} × {q})")
        print(f"    Bit length: {bit_length}")
        print(f"{'='*50}")
        
        # Adaptive parameters based on size
        if bit_length <= 10:
            max_bases, max_depth, window_size = 16, 1000, 128
        elif bit_length <= 13:
            max_bases, max_depth, window_size = 32, 2000, 256
        elif bit_length <= 16:
            max_bases, max_depth, window_size = 48, 4000, 512
        else:
            max_bases, max_depth, window_size = 64, 8000, 1024
        
        threshold = 0.6  # Consistent threshold
        
        print(f"Parameters: bases={max_bases}, depth={max_depth}, window={window_size}, threshold={threshold}")
        
        start_time = time.time()
        
        try:
            factor = numpy_wave_factor(N, max_bases, max_depth, window_size, threshold)
            elapsed = time.time() - start_time
            total_time += elapsed
            
            if factor and N % factor == 0 and 1 < factor < N:
                other = N // factor
                success = True
                print(f"\n🎉 SUCCESS!")
                print(f"✅ {N:,} = {factor:,} × {other:,}")
                print(f"⏱️  Time: {elapsed:.3f}s")
                
                # Verify correctness
                if (factor == p and other == q) or (factor == q and other == p):
                    print("🎯 Factors match expected values!")
                else:
                    print("⚠️  Factors are correct but different ordering")
                    
            else:
                success = False
                print(f"\n❌ FAILED")
                print(f"⏱️  Time: {elapsed:.3f}s")
                if factor:
                    print(f"   Returned invalid factor: {factor}")
                
        except Exception as e:
            elapsed = time.time() - start_time
            total_time += elapsed
            success = False
            print(f"\n💥 ERROR: {e}")
            print(f"⏱️  Time: {elapsed:.3f}s")
        
        # Record result
        results.append({
            'N': N,
            'bits': bit_length,
            'description': description,
            'success': success,
            'time': elapsed,
            'expected': f"{p} × {q}"
        })
        
        # Early stopping if we hit consecutive failures on small numbers
        if not success and bit_length <= 12:
            print(f"\n⚠️  Early failure on {bit_length}-bit number - may indicate issues")
        
        # Stop if large numbers are taking too long
        if elapsed > 60:  # 1 minute timeout
            print(f"\n⏰ Stopping due to long execution time ({elapsed:.1f}s)")
            break
    
    # Summary analysis
    print(f"\n" + "="*60)
    print("📊 RESULTS SUMMARY")
    print("="*60)
    
    successes = [r for r in results if r['success']]
    failures = [r for r in results if not r['success']]
    
    print(f"Overall success rate: {len(successes)}/{len(results)} ({100*len(successes)/len(results):.1f}%)")
    print(f"Total time: {total_time:.2f}s")
    if successes:
        print(f"Average time per success: {sum(r['time'] for r in successes)/len(successes):.3f}s")
    
    # Success by bit length
    print(f"\n📈 Success Rate by Bit Length:")
    bit_groups = {}
    for r in results:
        bits = r['bits']
        if bits not in bit_groups:
            bit_groups[bits] = {'total': 0, 'success': 0}
        bit_groups[bits]['total'] += 1
        if r['success']:
            bit_groups[bits]['success'] += 1
    
    for bits in sorted(bit_groups.keys()):
        g = bit_groups[bits]
        rate = 100 * g['success'] / g['total']
        print(f"   {bits:2d} bits: {g['success']}/{g['total']} ({rate:5.1f}%)")
    
    # Detailed results
    print(f"\n📋 Detailed Results:")
    for r in results:
        status = "✅" if r['success'] else "❌"
        print(f"   {status} {r['bits']:2d}-bit {r['N']:>8,}: {r['description']} - {r['time']:.3f}s")
    
    # Performance analysis
    if successes:
        print(f"\n🚀 Performance Analysis:")
        fastest = min(successes, key=lambda r: r['time'])
        slowest = max(successes, key=lambda r: r['time'])
        print(f"   Fastest: {fastest['bits']}-bit in {fastest['time']:.3f}s")
        print(f"   Slowest: {slowest['bits']}-bit in {slowest['time']:.3f}s")
        
        # Check if time scales polynomially
        times_by_bits = {}
        for r in successes:
            if r['bits'] not in times_by_bits:
                times_by_bits[r['bits']] = []
            times_by_bits[r['bits']].append(r['time'])
        
        print(f"\n   Average time by bit length:")
        for bits in sorted(times_by_bits.keys()):
            avg_time = sum(times_by_bits[bits]) / len(times_by_bits[bits])
            theoretical_poly = (bits ** 2) / 1000  # Rough O(n²) estimate
            print(f"     {bits:2d} bits: {avg_time:.3f}s (theory: ~{theoretical_poly:.3f}s)")

if __name__ == "__main__":
    test_incremental_sizes()
