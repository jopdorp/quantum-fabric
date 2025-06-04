#!/usr/bin/env python3
"""
Focused test of numpy wave factorization on N=77
"""

from numpy_wave_interference_factorization import numpy_wave_factor

def test_n77_detailed():
    """Test N=77 with detailed output."""
    print("🎯 Focused Test: N = 77 = 7 × 11")
    print("=" * 40)
    
    N = 77
    
    # Test with smaller parameters for debugging
    factor = numpy_wave_factor(N, max_bases=16, max_depth=1000, window_size=128, threshold=0.6)
    
    if factor:
        other = N // factor
        print(f"\n✅ SUCCESS: {N} = {factor} × {other}")
    else:
        print(f"\n❌ Failed to factor {N}")

if __name__ == "__main__":
    test_n77_detailed()
