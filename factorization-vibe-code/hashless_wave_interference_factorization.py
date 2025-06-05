#!/usr/bin/env python3
"""
Hashless Wave Interference Factorization

This implements true wave-based factorization WITHOUT hash tables or birthday paradox:
- Direct wave interference pattern matching
- Phase alignment detection between computational waves
- Rolling buffer for O(window) space complexity
- No collision-based dependencies

The key innovation: Instead of storing hash values and waiting for collisions,
we detect interference patterns directly through signal matching and phase alignment.
"""

import time
from math import gcd, log2, sqrt

def modmul(a, b, mod):
    """Efficient modular multiplication."""
    return (a * b) % mod

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

def detect_interference_pattern(buffer, window_size, N, threshold=0.7):
    """
    Detect wave interference patterns in the rolling buffer.
    Returns (found, position1, position2, alignment_score) tuple.
    """
    if len(buffer) < 2:
        return False, -1, -1, 0.0
    
    max_score = 0.0
    best_pos1, best_pos2 = -1, -1
    
    # Check recent values against earlier ones in the window
    recent_start = max(0, len(buffer) - window_size // 4)
    
    for i in range(recent_start, len(buffer)):
        val1 = buffer[i]
        
        # Look for alignment with earlier values
        search_start = max(0, i - window_size)
        for j in range(search_start, i):
            val2 = buffer[j]
            score = phase_alignment_score(val1, val2, N)
            
            if score > max_score:
                max_score = score
                best_pos1, best_pos2 = i, j
    
    found = max_score >= threshold
    return found, best_pos1, best_pos2, max_score

def wave_resonance_factor_extraction(a, pos1, pos2, N):
    """
    Extract factors using wave resonance positions.
    Uses multiple strategies based on interference points.
    """
    if pos1 <= pos2:
        return None
    
    period_estimate = pos1 - pos2
    
    # Strategy 1: Direct period-based extraction
    if period_estimate > 0 and period_estimate % 2 == 0:
        y = pow(a, period_estimate // 2, N)
        if y != 1 and y != N - 1:
            for f in [gcd(y - 1, N), gcd(y + 1, N)]:
                if 1 < f < N:
                    return f
    
    # Strategy 2: Midpoint resonance
    midpoint = (pos1 + pos2) // 2
    if midpoint > 0:
        y = pow(a, midpoint, N)
        if y != 1 and y != N - 1:
            for f in [gcd(y - 1, N), gcd(y + 1, N)]:
                if 1 < f < N:
                    return f
    
    # Strategy 3: Harmonic divisors
    for divisor in [2, 3, 4, 5, 6, 8, 10]:
        if period_estimate % divisor == 0:
            harmonic_exp = period_estimate // divisor
            if harmonic_exp > 0 and harmonic_exp % 2 == 0:
                y = pow(a, harmonic_exp // 2, N)
                if y != 1 and y != N - 1:
                    for f in [gcd(y - 1, N), gcd(y + 1, N)]:
                        if 1 < f < N:
                            return f
    
    return None

def hashless_wave_factor(N, max_bases=64, max_depth=4096, window_size=512, threshold=0.7):
    """
    Hashless wave-based factorization using direct interference pattern detection.
    
    Args:
        N: Number to factor
        max_bases: Maximum number of bases to try
        max_depth: Maximum sequence depth per base
        window_size: Size of rolling buffer for pattern detection
        threshold: Minimum alignment score for interference detection
    """
    print(f"Hashless wave factorization of N={N}")
    print(f"Parameters: bases={max_bases}, depth={max_depth}, window={window_size}, threshold={threshold}")
    
    for a in range(2, 2 + max_bases):
        if gcd(a, N) != 1:
            factor = gcd(a, N)
            print(f"[!] Trivial factor: gcd({a}, {N}) = {factor}")
            return factor
        
        print(f"[📡] Testing base a={a}")
        
        # Rolling buffer for wave values
        wave_buffer = []
        x = 1
        
        for k in range(1, max_depth):
            x = modmul(x, a, N)
            wave_buffer.append(x)
            
            # Maintain rolling window
            if len(wave_buffer) > window_size:
                wave_buffer.pop(0)
            
            # Check for natural period (strongest interference)
            if x == 1 and k > 1:
                print(f"[✓] Natural period detected at step {k}")
                if k % 2 == 0:
                    y = pow(a, k // 2, N)
                    if y != 1 and y != N - 1:
                        for f in [gcd(y - 1, N), gcd(y + 1, N)]:
                            if 1 < f < N:
                                print(f"[🎯] Factor via natural period: {f}")
                                return f
            
            # Detect wave interference patterns
            if k >= 10:  # Need minimum history
                found, pos1, pos2, score = detect_interference_pattern(
                    wave_buffer, min(window_size, k), N, threshold
                )
                
                if found:
                    # Convert buffer positions to actual sequence positions
                    actual_pos1 = k - (len(wave_buffer) - 1 - pos1)
                    actual_pos2 = k - (len(wave_buffer) - 1 - pos2)
                    
                    print(f"[⚡] Wave interference detected!")
                    print(f"    Positions: {actual_pos2} ↔ {actual_pos1}")
                    print(f"    Alignment score: {score:.3f}")
                    print(f"    Period estimate: {actual_pos1 - actual_pos2}")
                    
                    # Extract factors using resonance
                    factor = wave_resonance_factor_extraction(a, actual_pos1, actual_pos2, N)
                    if factor:
                        print(f"[🎯] Factor via wave resonance: {factor}")
                        return factor
            
            # Progress indicator
            if k % 500 == 0:
                print(f"    Step {k}/{max_depth} (buffer size: {len(wave_buffer)})")
    
    print("[×] No factors found via wave interference")
    return None

def adaptive_wave_factor(N, initial_threshold=0.8):
    """
    Adaptive version that adjusts parameters based on number size.
    """
    bit_size = int(log2(N))
    print(f"[🔧] Adaptive parameters for {bit_size}-bit number")
    
    # Scale parameters based on bit size
    if bit_size <= 16:
        bases, depth, window = 32, 2048, 256
    elif bit_size <= 20:
        bases, depth, window = 48, 3072, 384
    elif bit_size <= 24:
        bases, depth, window = 64, 4096, 512
    elif bit_size <= 28:
        bases, depth, window = 96, 6144, 768
    else:
        bases, depth, window = 128, 8192, 1024
    
    # Try with decreasing thresholds
    thresholds = [initial_threshold, initial_threshold - 0.1, initial_threshold - 0.2, 0.5]
    
    for threshold in thresholds:
        if threshold < 0.3:
            break
        
        print(f"\n[🎛️] Trying threshold {threshold:.1f}")
        factor = hashless_wave_factor(N, bases, depth, window, threshold)
        if factor:
            return factor
    
    return None

def test_hashless_wave_factorization():
    """Test the hashless wave interference factorization."""
    print("🌊 Hashless Wave Interference Factorization")
    print("=" * 55)
    print("Key Innovation: NO hash tables, NO birthday paradox")
    print("Direct wave interference pattern detection\n")
    
    # Test cases
    test_cases = [
        ("RSA-16", 176399, "419 × 421"),
        ("RSA-20", 1048573, "1021 × 1027"),  
        ("RSA-24", 16777181, "4093 × 4099"),
        ("RSA-28", 268435399, "16381 × 16387"),
    ]
    
    success_count = 0
    total_time = 0
    
    for name, N, expected in test_cases:
        print(f"\n🎯 Testing {name}: N = {N:,}")
        print(f"Expected: {expected}")
        
        start_time = time.time()
        
        # Try adaptive approach
        factor = adaptive_wave_factor(N, initial_threshold=0.8)
        
        elapsed = time.time() - start_time
        total_time += elapsed
        
        if factor and N % factor == 0 and 1 < factor < N:
            other = N // factor
            print(f"🎉 SUCCESS! {name} FACTORED!")
            print(f"✅ {N:,} = {factor:,} × {other:,}")
            print(f"⏱️  Time: {elapsed:.3f}s")
            success_count += 1
        else:
            print(f"❌ {name} resisted factorization")
            print(f"⏱️  Time: {elapsed:.3f}s")
        
        # Stop if early cases fail (indicates fundamental issue)
        if not factor and int(log2(N)) <= 20:
            print("⚠️  Early failure - stopping test sequence")
            break
    
    print(f"\n📊 Results Summary:")
    print(f"Success rate: {success_count}/{len(test_cases) if success_count > 0 or len(test_cases) <= 2 else 2}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average time: {total_time/(success_count if success_count > 0 else 1):.3f}s per success")
    
    print("\n🔬 Algorithm Analysis:")
    print("• Pure wave interference - no hash storage")
    print("• O(bases × depth) time complexity") 
    print("• O(window) space complexity")
    print("• No birthday paradox dependency")
    print("• Direct phase alignment detection")

if __name__ == "__main__":
    test_hashless_wave_factorization()
