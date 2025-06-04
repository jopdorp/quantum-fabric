#!/usr/bin/env python3
"""
Simple test to check basic functionality
"""

import sys
print("Testing import...")

try:
    from numpy_wave_interference_factorization import fast_gcd, detect_interference_matrix
    print("✅ Import successful")
    
    # Simple test
    result = fast_gcd(77, 14)
    print(f"✅ fast_gcd(77, 14) = {result}")
    
    # Test detection function
    wave_buffer = [2, 4, 8, 16, 32, 64, 51, 25, 50, 23]
    found, pos1, pos2, score = detect_interference_matrix(wave_buffer, len(wave_buffer), 77, 0.5)
    print(f"✅ detect_interference_matrix result: found={found}, score={score:.6f}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

print("All basic tests passed!")
