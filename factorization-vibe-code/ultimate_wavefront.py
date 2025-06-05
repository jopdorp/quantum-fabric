# Ultimate Wavefront RSA Factorization - Multi-Modal Approach
import numpy as np
from sympy import randprime, gcd
from numpy.fft import fft
import time
from collections import Counter

def generate_rsa_test(bits):
    p = randprime(2**(bits//2 - 1), 2**(bits//2))
    q = randprime(2**(bits//2 - 1), 2**(bits//2))
    while q == p:
        q = randprime(2**(bits//2 - 1), 2**(bits//2))
    return p * q, p, q

def modexp_sequence(a, N, length):
    return np.array([pow(a, i, N) for i in range(length)], dtype=np.int64)

def multi_modal_wavefront_signal(seq, N):
    """Multi-modal wavefront combining different mathematical transforms"""
    
    # Mode 1: Standard phase signal
    mode1 = np.exp(2j * np.pi * seq / N)
    
    # Mode 2: Differential phase (focusing on sequence changes)
    diff_seq = np.diff(seq)
    mode2 = np.zeros_like(mode1)
    mode2[1:] = np.exp(2j * np.pi * diff_seq / N)
    
    # Mode 3: Quadratic residue pattern
    quad_seq = (seq * seq) % N
    mode3 = np.exp(2j * np.pi * quad_seq / N)
    
    # Mode 4: Inverse modular pattern
    inv_seq = []
    for s in seq:
        try:
            inv_seq.append(pow(s, -1, N) if s != 0 else 0)
        except:
            inv_seq.append(0)
    mode4 = np.exp(2j * np.pi * np.array(inv_seq) / N)
    
    # Combine modes with optimized weights
    signal = 0.4 * mode1 + 0.3 * mode2 + 0.2 * mode3 + 0.1 * mode4
    
    # Apply focus window to emphasize early patterns
    window = np.exp(-0.5 * (np.arange(len(seq)) / (len(seq) * 0.3))**2)
    return signal * window

def autocorrelate_fft(signal):
    fft_signal = fft(signal)
    power = fft_signal * np.conj(fft_signal)
    corr = np.real(np.fft.ifft(power))
    return corr / corr[0]

def detect_periods_multi_scale(corr, min_lag=2):
    """Multi-scale period detection with mathematical insights"""
    periods = []
    half_len = len(corr) // 2
    
    # Scale 1: High sensitivity for small periods
    threshold1 = 0.1
    for i in range(min_lag, min(half_len, 2000)):
        if (corr[i] > threshold1 and 
            corr[i] > corr[i-1] and 
            corr[i] > corr[i+1]):
            periods.append(i)
    
    # Scale 2: Mathematical significance filter
    significant_periods = []
    mean_corr = np.mean(corr[min_lag:half_len//2])
    std_corr = np.std(corr[min_lag:half_len//2])
    significance_threshold = mean_corr + 1.0 * std_corr
    
    for p in periods:
        if p < len(corr) and corr[p] > significance_threshold:
            significant_periods.append(p)
    
    # Scale 3: Add divisors of significant periods
    divisor_periods = []
    for p in significant_periods[:10]:  # Top significant periods
        for d in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            if p % d == 0:
                div_p = p // d
                if div_p >= min_lag:
                    divisor_periods.append(div_p)
    
    # Combine all detected periods
    all_periods = periods + significant_periods + divisor_periods
    
    # Remove duplicates and rank by correlation strength
    unique_periods = list(set(all_periods))
    unique_periods.sort(key=lambda p: corr[p] if p < len(corr) else 0, reverse=True)
    
    return unique_periods[:25]

def validate_period_advanced(a, N, period):
    """Advanced period validation with partial period acceptance"""
    if period <= 1:
        return False
    
    # Check exact period
    if pow(a, period, N) == 1:
        return True
    
    # Check if period leads to useful factorization info
    val = pow(a, period, N)
    
    # Check various offsets
    for offset in [-2, -1, 1, 2]:
        candidate = val + offset
        if candidate > 1:
            g = gcd(candidate, N)
            if g > 1 and g < N:
                return True
    
    return False

def factor_ultimate_wavefront(N, verbose=False):
    """Ultimate wavefront factorization using all techniques"""
    
    # Adaptive parameters for efficiency
    if N.bit_length() <= 32:
        seq_length, max_trials = 12288, 20
    elif N.bit_length() <= 36:
        seq_length, max_trials = 16384, 25
    else:
        seq_length, max_trials = 24576, 30
    
    for trial in range(max_trials):
        a = np.random.randint(2, N - 1)
        if gcd(a, N) != 1:
            factor = gcd(a, N)
            if verbose:
                print(f"  Trial {trial+1}: Lucky GCD factor: {factor}")
            return factor
        
        # Generate multi-modal wavefront
        seq = modexp_sequence(a, N, seq_length)
        signal = multi_modal_wavefront_signal(seq, N)
        corr = autocorrelate_fft(signal)
        periods = detect_periods_multi_scale(corr)
        
        if verbose and trial < 3:
            valid_periods = [p for p in periods[:10] if validate_period_advanced(a, N, p)]
            print(f"  Trial {trial+1}: Base {a}, Valid periods: {valid_periods[:5]}")
        
        # Try factorization with detected periods
        for r in periods:
            if not validate_period_advanced(a, N, r):
                continue
            
            # Comprehensive factorization attempts
            attempts = [
                (r // 2, [-2, -1, 1, 2]),
                (r // 3, [-1, 0, 1]),
                (r // 4, [-1, 1]),
                (r // 6, [-1, 1]),
                (r, [-2, -1, 1, 2]),
                ((r * 2) // 3, [-1, 1]),
                ((r * 3) // 4, [-1, 1]),
            ]
            
            for exp, deltas in attempts:
                if exp <= 0:
                    continue
                
                try:
                    y = pow(a, exp, N)
                    for delta in deltas:
                        candidate = y + delta
                        if candidate > 1:
                            f = gcd(candidate, N)
                            if 1 < f < N:
                                if verbose:
                                    print(f"  ✅ FOUND: {f} via period {r}, exp {exp}, delta {delta}")
                                return f
                except:
                    continue
    
    return None

def test_ultimate():
    """Test the ultimate wavefront approach"""
    print("=== ULTIMATE WAVEFRONT FACTORIZATION ===\n")
    
    # Test on challenging RSA numbers
    test_cases = [
        (32, "RSA-32"),
        (36, "RSA-36"), 
        (40, "RSA-40"),
    ]
    
    for bits, label in test_cases:
        N, p, q = generate_rsa_test(bits)
        print(f"{label}: N = {N}")
        print(f"Target: {p} × {q}")
        
        start = time.time()
        result = factor_ultimate_wavefront(N, verbose=True)
        elapsed = time.time() - start
        
        if result and N % result == 0:
            other = N // result
            print(f"🎯 SUCCESS: {result} × {other} ({elapsed:.2f}s)")
        else:
            print(f"💔 Failed ({elapsed:.2f}s)")
        print("-" * 60)

if __name__ == "__main__":
    test_ultimate()
