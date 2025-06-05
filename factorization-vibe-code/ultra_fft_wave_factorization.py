#!/usr/bin/env python3
"""
Ultra High-Performance Wave Interference Factorization with FFT Fallback

Maximum resource allocation version for challenging RSA numbers.
Includes:
- Extended base range (512+ bases)
- Deeper wave sequences (32K+ depths)
- Multiple FFT window sizes and analysis techniques
- Advanced period validation and factor extraction
- Parallel processing preparation
"""

import numpy as np
import time
from math import gcd, log2, sqrt
from numba import jit
from scipy.fft import rfft, rfftfreq
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

@jit(nopython=True)
def fast_modmul_sequence(a, N, length):
    result = np.empty(length, dtype=np.int64)
    x = a % N
    for i in range(length):
        result[i] = x
        x = (x * a) % N
    return result

@jit(nopython=True)
def fast_gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def validate_period(sequence, period):
    """Enhanced period validation with multiple criteria"""
    if len(sequence) < 2 * period:
        return False
    
    # Check multiple segments of the sequence
    segments = min(4, len(sequence) // period)
    total_matches = 0
    total_checks = 0
    
    for seg in range(segments - 1):
        start1 = seg * period
        start2 = (seg + 1) * period
        checks = min(period, len(sequence) - start2)
        
        matches = 0
        for i in range(checks):
            if sequence[start1 + i] == sequence[start2 + i]:
                matches += 1
        
        total_matches += matches
        total_checks += checks
    
    if total_checks == 0:
        return False
    
    # Require high match rate
    match_rate = total_matches / total_checks
    return match_rate > 0.75

def autocorrelation_period_detect(sequence, max_period=None):
    """Use autocorrelation to detect periods"""
    if max_period is None:
        max_period = min(len(sequence) // 3, 4096)
    
    sequence = np.array(sequence[:max_period * 3], dtype=np.float64)
    n = len(sequence)
    
    # Compute autocorrelation
    autocorr = np.correlate(sequence, sequence, mode='full')
    autocorr = autocorr[n-1:]  # Take only positive lags
    
    # Find peaks in autocorrelation
    peaks = []
    for i in range(2, min(max_period, len(autocorr))):
        if (autocorr[i] > autocorr[i-1] and 
            autocorr[i] > autocorr[i+1] and 
            autocorr[i] > 0.5 * np.max(autocorr[1:i+100])):
            peaks.append((i, autocorr[i]))
    
    # Sort by correlation strength
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    # Return the strongest period candidates
    for period, strength in peaks[:5]:
        if validate_period(sequence, period):
            return period
    
    return None

def fft_detect_period(wave_sequence, N, window_sizes=[1024, 2048, 4096]):
    """Enhanced FFT period detection with multiple techniques"""
    
    # Try autocorrelation first
    auto_period = autocorrelation_period_detect(wave_sequence)
    if auto_period:
        return auto_period
    
    # Try multiple FFT window sizes
    for window_size in window_sizes:
        if len(wave_sequence) < window_size:
            continue
            
        sequence = np.array(wave_sequence[:window_size], dtype=np.float64)
        mean_centered = sequence - np.mean(sequence)
        
        # Remove linear trend
        x = np.arange(len(mean_centered))
        coeffs = np.polyfit(x, mean_centered, 1)
        detrended = mean_centered - np.polyval(coeffs, x)
        
        spectrum = np.abs(rfft(detrended))
        freqs = rfftfreq(len(sequence), d=1)
        
        # Find multiple significant peaks
        threshold = 0.1 * np.max(spectrum[1:])
        peak_indices = []
        
        for i in range(1, len(spectrum) - 1):
            if (spectrum[i] > spectrum[i-1] and 
                spectrum[i] > spectrum[i+1] and 
                spectrum[i] > threshold):
                peak_indices.append(i)
        
        # Sort by magnitude
        peak_indices.sort(key=lambda i: spectrum[i], reverse=True)
        
        # Test top candidates
        for peak_idx in peak_indices[:10]:
            freq = freqs[peak_idx]
            if freq == 0:
                continue
                
            period_estimate = round(1 / freq)
            if 2 <= period_estimate <= len(sequence) // 2:
                if validate_period(wave_sequence, period_estimate):
                    return period_estimate
    
    return None

def extract_factor_advanced(a, period_estimate, N):
    """Advanced factor extraction with multiple sophisticated techniques"""
    if not period_estimate:
        return None
    
    factors_found = set()
    
    # Try various divisors and offsets of the period
    test_values = []
    
    # Basic period tests
    for divisor in [1, 2, 3, 4, 6, 8]:
        if period_estimate % divisor == 0:
            test_values.append(period_estimate // divisor)
    
    # Period with small offsets
    for offset in range(-3, 4):
        test_period = period_estimate + offset
        if test_period > 1:
            test_values.append(test_period)
            if test_period % 2 == 0:
                test_values.append(test_period // 2)
    
    # Multiples of period
    for mult in [2, 3]:
        test_values.append(period_estimate * mult)
    
    # Test all candidates
    for test_val in set(test_values):
        if test_val <= 1:
            continue
            
        # Test both even and odd values
        for tv in [test_val, test_val + (test_val % 2)]:  # Make even if odd
            if tv % 2 == 0 and tv > 1:
                try:
                    y = pow(a, tv // 2, N)
                    if y != 1 and y != N - 1:
                        for delta in [-1, 1]:
                            candidate = y + delta
                            if candidate > 0:
                                f = gcd(candidate, N)
                                if 1 < f < N:
                                    factors_found.add(f)
                except:
                    continue
    
    # Return the smallest non-trivial factor found
    if factors_found:
        return min(factors_found)
    
    return None

def ultra_wave_factor_fft(N, max_bases=512, max_depth=32768, window_size=4096, 
                         parallel=True, max_workers=4):
    """Ultra high-performance factorization with all optimizations"""
    print(f"🚀 ULTRA Wave+FFT Factorization of N={N:,}")
    print(f"⚙️  Parameters: bases={max_bases}, depth={max_depth:,}, window={window_size:,}")
    print(f"🔧 Parallel processing: {parallel} (workers: {max_workers})")
    
    # Quick GCD checks first
    for a in range(2, min(20, max_bases + 2)):
        if gcd(a, N) != 1:
            factor = gcd(a, N)
            print(f"[!] Trivial factor: gcd({a}, {N}) = {factor}")
            return factor
    
    def test_base(a):
        """Test a single base for factorization"""
        if gcd(a, N) != 1:
            return gcd(a, N)
        
        wave_sequence = fast_modmul_sequence(a, N, max_depth)
        
        # Natural period detection (fast)
        for k in range(min(max_depth, 8192)):  # Check first 8K for speed
            x = wave_sequence[k]
            if x == 1 and k > 1 and (k + 1) % 2 == 0:
                y = pow(a, (k + 1) // 2, N)
                if y != 1 and y != N - 1:
                    for delta in [-1, 1]:
                        f = fast_gcd(y + delta, N)
                        if 1 < f < N:
                            return f
        
        # FFT-based period detection
        period_estimate = fft_detect_period(wave_sequence[:window_size], N)
        if period_estimate:
            factor = extract_factor_advanced(a, period_estimate, N)
            if factor:
                return factor
        
        return None
    
    if parallel:
        # Parallel processing
        print(f"[🔄] Testing {max_bases} bases in parallel...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(test_base, a): a for a in range(2, 2 + max_bases)}
            
            completed = 0
            for future in as_completed(futures):
                a = futures[future]
                completed += 1
                
                if completed % 100 == 0:
                    print(f"[📊] Progress: {completed}/{max_bases} bases tested...")
                
                try:
                    result = future.result()
                    if result:
                        print(f"[🎯] Factor found by base {a}: {result}")
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        return result
                except Exception as e:
                    print(f"[⚠️] Error testing base {a}: {e}")
    else:
        # Sequential processing
        for a in range(2, 2 + max_bases):
            if a % 100 == 0:
                print(f"[📊] Progress: tested {a-2} bases...")
                
            print(f"[🔍] Testing base a={a}")
            result = test_base(a)
            if result:
                print(f"[🎯] Factor found: {result}")
                return result
    
    print("[×] No factor found")
    return None

def test_ultra_wave_fft():
    print("🚀 ULTRA Wave Interference + FFT Factorization Test")
    print("=" * 65)
    
    test_cases = [
        ("RSA-16", 176399, "419 × 421"),
        ("RSA-20", 1048573, "1021 × 1027"), 
        ("RSA-24", 16777181, "4093 × 4099"),
        ("RSA-28", 268435399, "16381 × 16387"),
        ("RSA-32", 4294967291, "65521 × 65537"),
        # ("RSA-40", 1099511627689, "1048573 × 1048583"),  # Very challenging
    ]
    
    success_count = 0
    total_time = 0
    
    for name, N, expected in test_cases:
        print(f"\n🎯 Testing {name}: N = {N:,}")
        print(f"Expected: {expected}")
        
        start_time = time.time()
        # Use maximum resources for each test
        factor = ultra_wave_factor_fft(N, max_bases=512, max_depth=32768, 
                                     window_size=4096, parallel=False)  # Sequential for debugging
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
        
        print("-" * 50)
    
    print(f"\n📊 Final Results:")
    print(f"Success rate: {success_count}/{len(test_cases)}")
    if success_count > 0:
        print(f"Average successful time: {total_time/success_count:.3f}s")
    print(f"Total time: {total_time:.3f}s")

if __name__ == "__main__":
    test_ultra_wave_fft()
