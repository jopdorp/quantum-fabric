#!/usr/bin/env python3
"""
Wave Method Limitation Analysis
=============================

Understanding why the wave interference method hits a wall around RSA-28.

The method works by finding the multiplicative order r where a^r ≡ 1 (mod N).
Then factors are extracted via gcd(a^(r/2) ± 1, N).

This analysis checks: Are the multiplicative orders too large for our method?
"""

import sympy
import math
import numpy as np
from typing import List, Tuple

def multiplicative_order(a: int, n: int, max_order: int = 100000) -> int:
    """
    Find the multiplicative order of a modulo n.
    Returns the smallest positive integer r such that a^r ≡ 1 (mod n).
    Returns -1 if order > max_order.
    """
    if math.gcd(a, n) != 1:
        return -1
    
    current = a % n
    for r in range(1, max_order + 1):
        if current == 1:
            return r
        current = (current * a) % n
    
    return -1  # Order too large

def analyze_rsa_challenge_periods(bits: int, num_samples: int = 20, max_bases: int = 100):
    """Analyze the multiplicative orders for a given RSA bit size."""
    
    print(f"\n🔬 ANALYZING RSA-{bits} MULTIPLICATIVE ORDERS")
    print("=" * 60)
    
    # Generate multiple RSA challenges
    orders_found = []
    
    for sample in range(num_samples):
        # Generate RSA challenge
        p_bits = bits // 2
        q_bits = bits - p_bits
        
        p = sympy.randprime(2**(p_bits-1), 2**p_bits)
        q = sympy.randprime(2**(q_bits-1), 2**q_bits)
        n = p * q
        
        print(f"Sample {sample+1}: N = {n:,} (p={p}, q={q})")
        
        # Try multiple bases
        sample_orders = []
        bases_tried = 0
        
        for attempt in range(max_bases * 3):
            if bases_tried >= max_bases:
                break
                
            base = np.random.randint(2, min(10000, n))
            if math.gcd(base, n) != 1:
                continue
                
            bases_tried += 1
            order = multiplicative_order(base, n, max_order=50000)
            
            if order > 0:
                sample_orders.append(order)
                print(f"  Base {base}: order = {order}")
                
                # Early termination if we find a useful small order
                if order < 10000 and order % 2 == 0:
                    print(f"  ✅ Found useful even order: {order}")
                    break
            else:
                print(f"  Base {base}: order > 50,000 (too large)")
            
            if bases_tried >= 20 and not sample_orders:
                print(f"  ⚠️ No small orders found in first 20 bases")
                break
        
        if sample_orders:
            min_order = min(sample_orders)
            avg_order = sum(sample_orders) / len(sample_orders)
            orders_found.extend(sample_orders)
            print(f"  Summary: {len(sample_orders)} orders found, min={min_order}, avg={avg_order:.0f}")
        else:
            print(f"  ❌ No orders found (all > 50,000)")
    
    # Overall analysis
    print(f"\n📊 OVERALL ANALYSIS FOR RSA-{bits}")
    print("=" * 40)
    
    if orders_found:
        min_order = min(orders_found)
        max_order = max(orders_found)
        avg_order = sum(orders_found) / len(orders_found)
        median_order = sorted(orders_found)[len(orders_found)//2]
        
        even_orders = [o for o in orders_found if o % 2 == 0]
        small_orders = [o for o in orders_found if o < 10000]
        very_small_orders = [o for o in orders_found if o < 5000]
        
        print(f"Total orders found: {len(orders_found)}")
        print(f"Range: {min_order} to {max_order}")
        print(f"Average: {avg_order:.0f}")
        print(f"Median: {median_order}")
        print(f"Even orders: {len(even_orders)}/{len(orders_found)} ({100*len(even_orders)/len(orders_found):.1f}%)")
        print(f"Small orders (<10K): {len(small_orders)}/{len(orders_found)} ({100*len(small_orders)/len(orders_found):.1f}%)")
        print(f"Very small orders (<5K): {len(very_small_orders)}/{len(orders_found)} ({100*len(very_small_orders)/len(orders_found):.1f}%)")
        
        # Wave method viability assessment
        print(f"\n🌊 WAVE METHOD VIABILITY:")
        if avg_order < 5000:
            print(f"✅ EXCELLENT - Average order {avg_order:.0f} is well within detection range")
        elif avg_order < 15000:
            print(f"👍 GOOD - Average order {avg_order:.0f} is challenging but detectable")
        elif avg_order < 50000:
            print(f"⚠️ DIFFICULT - Average order {avg_order:.0f} requires large signal lengths")
        else:
            print(f"❌ IMPRACTICAL - Average order {avg_order:.0f} exceeds detection capabilities")
    else:
        print(f"❌ NO SMALL ORDERS FOUND")
        print(f"All multiplicative orders > 50,000")
        print(f"🌊 WAVE METHOD: INFEASIBLE for RSA-{bits}")
    
    return orders_found

def comprehensive_limitation_analysis():
    """Comprehensive analysis of where the wave method breaks down."""
    
    print("🔬 COMPREHENSIVE WAVE METHOD LIMITATION ANALYSIS")
    print("=" * 70)
    print("Objective: Understand the mathematical limits of wave interference factorization")
    print("Method: Analyze multiplicative order distributions for different RSA sizes")
    print()
    
    bit_sizes = [16, 20, 24, 28, 32, 36, 40, 48, 56, 64]
    results = {}
    
    for bits in bit_sizes:
        orders = analyze_rsa_challenge_periods(bits, num_samples=10, max_bases=50)
        
        if orders:
            avg_order = sum(orders) / len(orders)
            min_order = min(orders)
            small_ratio = len([o for o in orders if o < 10000]) / len(orders)
        else:
            avg_order = float('inf')
            min_order = float('inf')
            small_ratio = 0.0
        
        results[bits] = {
            'avg_order': avg_order,
            'min_order': min_order,
            'small_ratio': small_ratio,
            'orders_found': len(orders)
        }
    
    # Summary analysis
    print(f"\n📈 SCALING BEHAVIOR ANALYSIS")
    print("=" * 50)
    print(f"{'Bits':<6} {'Avg Order':<12} {'Min Order':<12} {'Small %':<10} {'Success Pred.'}")
    print("-" * 60)
    
    for bits in bit_sizes:
        data = results[bits]
        avg = data['avg_order']
        min_val = data['min_order']
        small_pct = data['small_ratio'] * 100
        
        if avg == float('inf'):
            avg_str = ">50,000"
            min_str = ">50,000"
            prediction = "❌ FAIL"
        else:
            avg_str = f"{avg:.0f}"
            min_str = f"{min_val}"
            
            if avg < 5000:
                prediction = "✅ SUCCESS"
            elif avg < 15000:
                prediction = "🤔 MAYBE"
            else:
                prediction = "❌ FAIL"
        
        print(f"{bits:<6} {avg_str:<12} {min_str:<12} {small_pct:<9.1f}% {prediction}")
    
    # Conclusions
    print(f"\n🎯 CONCLUSIONS")
    print("=" * 30)
    
    success_threshold = None
    for bits in sorted(bit_sizes):
        data = results[bits]
        if data['avg_order'] > 15000:
            success_threshold = bits
            break
    
    if success_threshold:
        print(f"🚧 Wave method breakdown point: ~RSA-{success_threshold}")
        print(f"💡 Reason: Multiplicative orders become too large (>15K)")
        print(f"🔧 Technical limit: Autocorrelation detection requires orders < signal length")
        print(f"📊 Signal length vs. order: Need signal ≥ 2×order for reliable detection")
        print(f"💾 Memory constraint: Large signals → exponential memory growth")
    else:
        print(f"✅ Wave method appears viable for all tested sizes")
    
    print(f"\n🚀 RECOMMENDATIONS:")
    print(f"1. Focus optimization on RSA-{success_threshold-4} to RSA-{success_threshold-8}")
    print(f"2. For larger RSA sizes, need fundamentally different approach")
    print(f"3. Consider hybrid methods: wave + classical for different ranges")
    print(f"4. Investigate quantum-inspired alternatives for large orders")

if __name__ == "__main__":
    comprehensive_limitation_analysis()
