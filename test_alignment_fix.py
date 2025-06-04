#!/usr/bin/env python3
"""
Test script to verify the alignment score fix
"""

from math import gcd, sqrt

def phase_alignment_score(val1, val2, N):
    """
    Calculate phase alignment score between two wave values.
    Higher scores indicate stronger interference patterns.
    """
    # Normalized difference - closer values have higher alignment
    diff = abs(val1 - val2)
    norm_diff = diff / N
    
    # Phase score: 1.0 for identical, approaches 0 for maximally different
    base_score = 1.0 - (2.0 * norm_diff) if norm_diff <= 0.5 else 2.0 * (1.0 - norm_diff)
    
    # Additional harmonic detection
    gcd_factor = gcd(val1, val2)
    if gcd_factor > 1:
        harmonic_bonus = min(0.2, gcd_factor / sqrt(N))
        base_score += harmonic_bonus
    
    # Ensure score doesn't exceed 1.0 to maintain proper threshold behavior
    return max(0.0, min(1.0, base_score))

def main():
    N = 10000
    
    print("Testing alignment score function:")
    print("=" * 50)
    
    # Test identical values
    val1, val2 = 1000, 1000
    score = phase_alignment_score(val1, val2, N)
    print(f"Identical values: {val1}, {val2} -> score: {score:.6f}")
    
    # Test close values
    val1, val2 = 1000, 1001
    score = phase_alignment_score(val1, val2, N)
    print(f"Close values: {val1}, {val2} -> score: {score:.6f}")
    
    # Test far values  
    val1, val2 = 1000, 9000
    score = phase_alignment_score(val1, val2, N)
    print(f"Far values: {val1}, {val2} -> score: {score:.6f}")
    
    # Test with high GCD
    val1, val2 = 1200, 1800  # gcd = 600
    score = phase_alignment_score(val1, val2, N)
    print(f"High GCD values: {val1}, {val2} (gcd={gcd(val1, val2)}) -> score: {score:.6f}")
    
    # Test edge case that was causing issues
    val1, val2 = 5000, 5000
    score = phase_alignment_score(val1, val2, N)
    print(f"Edge case (identical): {val1}, {val2} -> score: {score:.6f}")
    
    # Verify score is never > 1.0
    print("\nVerifying score bounds:")
    max_score = 0.0
    for i in range(100):
        val1 = i * 100
        val2 = i * 100  # Identical values
        score = phase_alignment_score(val1, val2, N)
        if score > max_score:
            max_score = score
    
    print(f"Maximum score found: {max_score:.6f}")
    print(f"Score bounded correctly: {max_score <= 1.0}")

if __name__ == "__main__":
    main()
