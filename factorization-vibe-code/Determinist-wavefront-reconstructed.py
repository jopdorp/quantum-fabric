# Wave-Based Factorization – FFT Phase Signal Prototype (Pre-Wavefront CA)

import numpy as np
from sympy import randprime, gcd
from typing import Optional
from numpy.fft import fft
import time

# ---- PARAMETERS ----
def get_adaptive_params(N):
    """Get adaptive parameters based on the size of N"""
    bit_length = N.bit_length()
    if bit_length <= 20:
        return 8192, 30  # Reduced trials for speed
    elif bit_length <= 28:
        return 16384, 40
    elif bit_length <= 32:
        return 32768, 50  # Increased sequence length
    elif bit_length <= 36:
        return 65536, 60
    elif bit_length <= 40:
        return 131072, 80
    else:
        return 262144, 100

# ---- UTILITIES ----
def generate_rsa_test(bits):
    p = randprime(2**(bits//2 - 1), 2**(bits//2))
    q = randprime(2**(bits//2 - 1), 2**(bits//2))
    while q == p:
        q = randprime(2**(bits//2 - 1), 2**(bits//2))
    return p * q, p, q

def modexp_sequence(a, N, length):
    return np.array([pow(a, i, N) for i in range(length)], dtype=np.int64)

def complex_phase_signal(seq, N):
    """Enhanced phase signal with quantum-inspired wavefront superposition"""
    # Primary wavefront signal
    primary = np.exp(2j * np.pi * seq / N)
    
    # Add wavefront harmonics for better period detection
    harmonic2 = np.exp(4j * np.pi * seq / N)  # Second harmonic
    harmonic3 = np.exp(6j * np.pi * seq / N)  # Third harmonic
    harmonic4 = np.exp(8j * np.pi * seq / N)  # Fourth harmonic
    
    # Quantum-inspired superposition with phase modulation
    sqrt_phase = np.exp(2j * np.pi * np.sqrt(seq / N))  # Square root phase
    log_phase = np.exp(2j * np.pi * np.log(seq + 1) / np.log(N))  # Logarithmic phase
    
    # Combine with weighted superposition (wavefront interference)
    signal = (primary + 0.6 * harmonic2 + 0.4 * harmonic3 + 0.2 * harmonic4 + 
              0.3 * sqrt_phase + 0.25 * log_phase)
    
    # Apply wavefront envelope for better coherence
    envelope = np.exp(-0.5 * (np.arange(len(seq)) / len(seq))**2)
    signal = signal * envelope
    
    # Normalize to maintain wavefront coherence
    return signal / np.abs(signal).max()

# ---- AUTOCORRELATION ----
def autocorrelate_fft(signal):
    fft_signal = fft(signal)
    power = fft_signal * np.conj(fft_signal)
    corr = np.real(np.fft.ifft(power))
    # Normalize correlation
    corr = corr / corr[0]
    return corr

def detect_periods_from_autocorr(corr, min_lag=2):
    """Find multiple potential periods from autocorrelation"""
    periods = []
    threshold = 0.3  # Lower threshold for peak detection
    
    # Look for peaks in the first half of the correlation
    half_len = len(corr) // 2
    
    for i in range(min_lag, half_len):
        # Check if this is a local maximum above threshold
        if (corr[i] > threshold and 
            corr[i] > corr[i-1] and 
            corr[i] > corr[i+1]):
            periods.append(i)
    
    # Sort by correlation strength
    periods.sort(key=lambda p: corr[p], reverse=True)
    return periods[:10]  # Return top 10 candidates

def find_order(a, N, max_order=None):
    """Find the multiplicative order of a modulo N"""
    if max_order is None:
        max_order = min(N, 10000)  # Reasonable upper bound
    
    order = 1
    current = a % N
    
    while order < max_order:
        if current == 1:
            return order
        current = (current * a) % N
        order += 1
    
    return None

def validate_period_wavefront(a, N, period, seq=None):
    """Wavefront-based period validation using phase coherence"""
    if period <= 1:
        return False
    
    # Primary validation: a^period ≡ 1 (mod N)
    if pow(a, period, N) == 1:
        return True
    
    # Wavefront phase coherence check
    if seq is not None and len(seq) > period:
        # Check if the sequence shows periodic behavior
        phase_diff = 0
        for i in range(min(5, len(seq) - period)):
            phase_diff += abs(seq[i] - seq[i + period])
        
        # If phase difference is small, period might be valid
        if phase_diff < N * 0.1:  # Threshold based on N
            return True
    
    # Check if the period divides the actual order
    actual_order = find_order(a, N, max_order=min(N, 20000))
    if actual_order and actual_order % period == 0:
        return True
    
    # Wavefront interference check - even partial periods can be useful
    val = pow(a, period, N)
    gcd_val = gcd(val - 1, N)
    return gcd_val > 1

# ---- FACTORIZATION ----
def factor_fft_phase(N, verbose=False):
    # Get adaptive parameters
    sequence_length, max_trials = get_adaptive_params(N)
    
    for trial in range(max_trials):
        a = np.random.randint(2, N - 1)
        if gcd(a, N) != 1:
            factor = gcd(a, N)
            if verbose:
                print(f"  Trial {trial+1}: Found factor via GCD: {factor}")
            return factor
            
        # Generate sequence and multi-scale signals
        seq = modexp_sequence(a, N, sequence_length)
        
        # Process at multiple wavefront scales for better detection
        scales = [1.0, 0.8, 1.2, 0.6, 1.5]  # Different scale factors
        all_periods = []
        
        for scale in scales:
            scaled_seq = seq[:int(len(seq) * scale)]
            signal = complex_phase_signal(scaled_seq, N)
            corr = autocorrelate_fft(signal)
            periods = detect_periods_enhanced(corr)
            all_periods.extend(periods)
        
        # Add cellular automata wavefront analysis
        ca_corr = wavefront_cellular_analysis(seq, N)
        ca_periods = detect_periods_enhanced(ca_corr)
        all_periods.extend(ca_periods)
        
        # Remove duplicates and sort by frequency of occurrence
        from collections import Counter
        period_counts = Counter(all_periods)
        periods = [p for p, count in period_counts.most_common(30)]
        
        if verbose and periods and trial < 5:  # Only show first few trials
            print(f"  Trial {trial+1}: Base {a}, Found periods: {periods[:5]}")
        
        for r in periods:
            if not validate_period_wavefront(a, N, r, seq):
                continue
                
            # Enhanced approaches with more fractional periods
            approaches = [
                (r // 2, [-1, 1]),  # Classic Shor
                (r // 3, [-1, 0, 1]),  # Third
                (r // 4, [-2, -1, 0, 1, 2]),  # Quarter
                (r // 5, [-1, 0, 1]),  # Fifth
                (r // 6, [-1, 1]),  # Sixth
                (r, [-1, 1]),  # Full period
                (r * 2, [-1, 1]),  # Double period
            ]
            
            for exp, deltas in approaches:
                if exp <= 0:
                    continue
                    
                try:
                    y = pow(a, exp, N)
                    
                    for delta in deltas:
                        candidate = y + delta
                        if candidate <= 1:
                            continue
                            
                        f = gcd(candidate, N)
                        if 1 < f < N:
                            if verbose:
                                print(f"  Found factor: {f} using period {r}, exp {exp}, delta {delta}")
                            return f
                            
                except (ValueError, OverflowError):
                    continue
    
    return None

def wavefront_cellular_analysis(seq, N):
    """Wavefront cellular automata analysis for period detection"""
    # Convert sequence to cellular automata rules
    rules = []
    for i in range(len(seq) - 1):
        rules.append((seq[i] + seq[i+1]) % N)
    
    # Find periods in the cellular automata evolution
    ca_signal = np.exp(2j * np.pi * np.array(rules) / N)
    return autocorrelate_fft(ca_signal)

def detect_periods_enhanced(corr, min_lag=2):
    """Enhanced period detection with wavefront interference analysis"""
    periods = []
    
    # Method 1: Wavefront peak detection with adaptive threshold
    mean_corr = np.mean(corr[min_lag:len(corr)//2])
    std_corr = np.std(corr[min_lag:len(corr)//2])
    threshold = mean_corr + 0.3 * std_corr  # Lowered for more sensitivity
    
    half_len = len(corr) // 2
    
    # Detect primary wavefront peaks
    for i in range(min_lag, half_len):
        if (corr[i] > threshold and 
            corr[i] > corr[i-1] and 
            corr[i] > corr[i+1]):
            periods.append(i)
    
    # Method 2: Wavefront harmonic analysis
    for base_period in periods[:8]:  # More candidates
        for harmonic in [2, 3, 4, 5, 6]:
            harmonic_period = base_period * harmonic
            if harmonic_period < half_len:
                periods.append(harmonic_period)
    
    # Method 3: Subharmonic wavefront analysis (divisors)
    strong_periods = [p for p in periods if corr[p] > threshold * 1.2]
    for period in strong_periods:
        for divisor in [2, 3, 4, 5, 6, 7, 8]:
            if period % divisor == 0:
                sub_period = period // divisor
                if sub_period >= min_lag:
                    periods.append(sub_period)
    
    # Method 4: Wavefront interference pattern detection
    # Look for periods that create constructive interference
    for i in range(min_lag, min(half_len, 1000)):  # Optimized range
        if i not in periods:
            # Check if this creates interference patterns
            interference_score = 0
            for j in range(2, 5):  # Reduced multiples for speed
                if i * j < len(corr):
                    interference_score += corr[i * j]
            if interference_score > threshold * 2.0:  # Higher threshold
                periods.append(i)
    
    # Method 5: Fibonacci-like wavefront sequences
    # Look for periods that follow natural growth patterns
    fib_periods = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]
    for fib_p in fib_periods:
        if fib_p < half_len and fib_p not in periods:
            if corr[fib_p] > threshold * 0.8:  # Lower threshold for natural patterns
                periods.append(fib_p)
    
    # Method 6: Wavefront phase velocity analysis
    # Detect periods based on wave propagation characteristics
    for i in range(min_lag, min(half_len, 500)):  # Focus on smaller periods
        if i not in periods:
            # Calculate phase velocity indicators
            phase_velocity = 0
            for k in range(1, min(4, half_len // i)):
                if i * k < len(corr):
                    phase_velocity += corr[i * k] * np.exp(-0.1 * k)  # Exponential decay
            
            if phase_velocity > threshold * 1.5:
                periods.append(i)
    
    # Remove duplicates and sort by wavefront strength
    periods = list(set(periods))
    periods.sort(key=lambda p: corr[p] if p < len(corr) else 0, reverse=True)
    
    return periods[:30]  # Return top 30 candidates

# ---- TEST LOOP ----
def test_many():
    for bits in [32, 36, 40, 44, 48]:  # Focus on larger numbers
        N, p, q = generate_rsa_test(bits)
        print(f"\n[RSA-{bits}] N = {N}")
        print(f"Expected: {p} × {q}")
        start = time.time()
        f = factor_fft_phase(N, verbose=True)
        elapsed = time.time() - start
        if f and (N % f == 0):
            other_factor = N // f
            print(f"✅ Success: {f} × {other_factor} in {elapsed:.3f}s")
        else:
            print(f"❌ Failed after {elapsed:.3f}s")

if __name__ == "__main__":
    test_many()
