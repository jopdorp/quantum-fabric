# Advanced Wavefront Factorization - Optimized for RSA-32+
import numpy as np
from sympy import randprime, gcd
from numpy.fft import fft
import time

def get_adaptive_params(N):
    """Optimized parameters for larger numbers"""
    bit_length = N.bit_length()
    if bit_length <= 32:
        return 16384, 25  # Shorter sequence, fewer trials
    elif bit_length <= 36:
        return 32768, 30
    else:
        return 65536, 35

def generate_rsa_test(bits):
    p = randprime(2**(bits//2 - 1), 2**(bits//2))
    q = randprime(2**(bits//2 - 1), 2**(bits//2))
    while q == p:
        q = randprime(2**(bits//2 - 1), 2**(bits//2))
    return p * q, p, q

def modexp_sequence(a, N, length):
    return np.array([pow(a, i, N) for i in range(length)], dtype=np.int64)

def quantum_wavefront_signal(seq, N):
    """Quantum-inspired wavefront with entanglement-like correlations"""
    # Primary phase signal
    primary = np.exp(2j * np.pi * seq / N)
    
    # Quantum entanglement simulation - correlate distant elements
    entangled = np.zeros_like(primary, dtype=complex)
    for i in range(len(seq)):
        # Create "entanglement" with elements at Fibonacci distances
        fib_distances = [1, 2, 3, 5, 8, 13, 21]
        for d in fib_distances:
            if i + d < len(seq):
                entangled[i] += 0.1 * np.exp(2j * np.pi * seq[i + d] / N)
    
    # Superposition principle
    signal = primary + 0.3 * entangled
    
    # Apply quantum decoherence envelope
    decoherence = np.exp(-0.1 * np.arange(len(seq)) / len(seq))
    return signal * decoherence

def autocorrelate_fft(signal):
    fft_signal = fft(signal)
    power = fft_signal * np.conj(fft_signal)
    corr = np.real(np.fft.ifft(power))
    return corr / corr[0]

def detect_periods_quantum(corr, min_lag=2):
    """Quantum-inspired period detection"""
    periods = []
    half_len = len(corr) // 2
    
    # Adaptive threshold based on quantum uncertainty principle
    mean_corr = np.mean(corr[min_lag:half_len])
    std_corr = np.std(corr[min_lag:half_len])
    threshold = mean_corr + 0.4 * std_corr
    
    # Primary peak detection
    for i in range(min_lag, min(half_len, 8000)):  # Focus on reasonable range
        if (corr[i] > threshold and 
            corr[i] > corr[i-1] and 
            corr[i] > corr[i+1]):
            periods.append(i)
    
    # Quantum harmonic resonance
    resonance_periods = []
    for p in periods[:10]:  # Top periods only
        for harmonic in [2, 3, 4]:  # Fewer harmonics for speed
            h_period = p * harmonic
            if h_period < half_len:
                resonance_periods.append(h_period)
    
    periods.extend(resonance_periods)
    
    # Remove duplicates and sort by strength
    periods = list(set(periods))
    periods.sort(key=lambda p: corr[p] if p < len(corr) else 0, reverse=True)
    
    return periods[:20]  # Top 20 candidates

def validate_period_quantum(a, N, period):
    """Fast quantum period validation"""
    if period <= 1:
        return False
    
    # Primary check
    if pow(a, period, N) == 1:
        return True
    
    # Quantum interference check
    val = pow(a, period, N)
    return gcd(val - 1, N) > 1

def factor_quantum_wavefront(N, verbose=False):
    """Quantum-inspired wavefront factorization"""
    sequence_length, max_trials = get_adaptive_params(N)
    
    for trial in range(max_trials):
        a = np.random.randint(2, N - 1)
        if gcd(a, N) != 1:
            factor = gcd(a, N)
            if verbose:
                print(f"  Trial {trial+1}: Found factor via GCD: {factor}")
            return factor
        
        # Generate quantum wavefront signal
        seq = modexp_sequence(a, N, sequence_length)
        signal = quantum_wavefront_signal(seq, N)
        corr = autocorrelate_fft(signal)
        periods = detect_periods_quantum(corr)
        
        if verbose and periods and trial < 3:
            print(f"  Trial {trial+1}: Base {a}, Top periods: {periods[:5]}")
        
        for r in periods:
            if not validate_period_quantum(a, N, r):
                continue
            
            # Enhanced factorization approaches
            approaches = [
                (r // 2, [-1, 1]),
                (r // 3, [-1, 0, 1]),
                (r // 4, [-1, 1]),
                (r, [-1, 1]),
                ((2 * r) // 3, [-1, 1]),  # Two-thirds
            ]
            
            for exp, deltas in approaches:
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
                                    print(f"  ✅ Found factor: {f} using period {r}")
                                return f
                except:
                    continue
    
    return None

def test_rsa32_focus():
    """Focus on RSA-32 testing"""
    print("=== Quantum Wavefront RSA-32+ Factorization ===\n")
    
    for bits in [32, 36, 40]:
        N, p, q = generate_rsa_test(bits)
        print(f"[RSA-{bits}] N = {N}")
        print(f"Expected: {p} × {q}")
        
        start = time.time()
        f = factor_quantum_wavefront(N, verbose=True)
        elapsed = time.time() - start
        
        if f and (N % f == 0):
            other_factor = N // f
            print(f"✅ SUCCESS: {f} × {other_factor} in {elapsed:.3f}s")
        else:
            print(f"❌ Failed after {elapsed:.3f}s")
        print()

if __name__ == "__main__":
    test_rsa32_focus()
