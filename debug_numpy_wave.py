#!/usr/bin/env python3
"""
Debug the numpy wave factorization step by step
"""

import numpy as np
from numpy_wave_interference_factorization import (
    fast_modmul_sequence, detect_interference_matrix, 
    matrix_resonance_extraction, fast_gcd
)

def debug_wave_factorization():
    """Debug wave factorization step by step."""
    print("🔍 Step-by-Step Debug: N = 77")
    print("=" * 40)
    
    N = 77
    a = 2  # Start with base 2
    max_depth = 200
    window_size = 50
    threshold = 0.6
    
    print(f"Testing base a = {a}")
    print(f"Parameters: max_depth={max_depth}, window_size={window_size}, threshold={threshold}")
    
    # Check if base is coprime
    g = fast_gcd(a, N)
    if g != 1:
        print(f"🎯 Trivial factor found: gcd({a}, {N}) = {g}")
        return g
    
    # Generate wave sequence
    print(f"\n📊 Generating wave sequence...")
    wave_sequence = fast_modmul_sequence(a, N, max_depth)
    print(f"First 20 values: {wave_sequence[:20].tolist()}")
    
    # Look for natural period
    natural_period = None
    for k in range(max_depth):
        if wave_sequence[k] == 1 and k > 0:
            natural_period = k + 1
            print(f"🔄 Natural period found at step {k+1}: period = {natural_period}")
            break
    
    # Test interference detection at various points
    wave_buffer = []
    detections = []
    
    for k in range(min(100, max_depth)):  # Test first 100 steps
        x = wave_sequence[k]
        wave_buffer.append(x)
        
        # Maintain rolling window
        if len(wave_buffer) > window_size:
            wave_buffer.pop(0)
        
        # Check for interference every 10 steps
        if k >= 10 and k % 10 == 0:
            found, pos1, pos2, score = detect_interference_matrix(
                wave_buffer, len(wave_buffer), N, threshold
            )
            
            if found:
                # Convert buffer positions to sequence positions
                actual_pos1 = k - (len(wave_buffer) - 1 - pos1)
                actual_pos2 = k - (len(wave_buffer) - 1 - pos2)
                
                detection_info = {
                    'step': k,
                    'pos1': actual_pos1,
                    'pos2': actual_pos2,
                    'score': score,
                    'period_est': actual_pos1 - actual_pos2
                }
                detections.append(detection_info)
                
                print(f"\n⚡ Interference at step {k}:")
                print(f"   Positions: {actual_pos2} ↔ {actual_pos1}")
                print(f"   Period estimate: {actual_pos1 - actual_pos2}")
                print(f"   Score: {score:.6f}")
                
                # Try factor extraction
                factor = matrix_resonance_extraction(a, actual_pos1, actual_pos2, N)
                if factor:
                    print(f"   🎯 Factor extracted: {factor}")
                    other = N // factor
                    print(f"   ✅ Verification: {N} = {factor} × {other}")
                    return factor
                else:
                    print(f"   ❌ No factor extracted")
    
    print(f"\n📈 Summary:")
    print(f"   Natural period: {natural_period}")
    print(f"   Interference detections: {len(detections)}")
    
    if detections:
        print(f"   Best detection score: {max(d['score'] for d in detections):.6f}")
        print(f"   Period estimates: {[d['period_est'] for d in detections]}")
    
    return None

if __name__ == "__main__":
    factor = debug_wave_factorization()
    if factor:
        print(f"\n🎉 SUCCESS: Found factor {factor}")
    else:
        print(f"\n❌ No factor found")
