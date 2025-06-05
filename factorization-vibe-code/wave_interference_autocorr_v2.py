#!/usr/bin/env python3
"""
Wave Interference Autocorrelation Factorization (v2)
====================================================

A mathematically sound wave-based factorization algorithm that uses:
- Complex modular signals: ψᵢ = e^(2πi·aⁱ mod N / N)
- Autocorrelation-based period detection via phase alignment  
- True wave interference through self-overlap measurement
- Eliminates FFT artifacts and frequency spectrum misinterpretation

This preserves the wave-driven, biologically-inspired computation while
avoiding the flawed use of FFT as a factorization oracle.
"""

import numpy as np
import time
import math
from sympy import randprime
from typing import List, Tuple, Optional

class WaveInterferenceFactorizer:
    """
    Wave-based integer factorization using autocorrelation interference.
    
    Core principle: Encode modular exponentiation as rotating phases on the 
    unit circle, then detect periods through constructive interference when
    the wave overlaps with itself after phase shifts.
    """
    
    def __init__(self, max_signal_length: int = 8192, max_shift_search: int = 2048):
        self.max_signal_length = max_signal_length
        self.max_shift_search = max_shift_search
        
    def fast_modular_sequence(self, base: int, modulus: int, length: int) -> np.ndarray:
        """
        Generate modular exponentiation sequence: [a^0 mod N, a^1 mod N, ..., a^(length-1) mod N]
        Uses Python's arbitrary precision for large numbers, numpy for smaller ones.
        """
        # Check if we can use numpy (faster) or need Python lists (handles large ints)
        if modulus < 2**63:  # Safe for numpy int64
            sequence = np.zeros(length, dtype=np.int64)
            current = 1
            sequence[0] = current
            
            for i in range(1, length):
                current = (current * base) % modulus
                sequence[i] = current
                
            return sequence
        else:
            # Use Python lists for very large integers
            sequence = []
            current = 1
            sequence.append(current)
            
            for i in range(1, length):
                current = (current * base) % modulus
                sequence.append(current)
                
            return np.array(sequence, dtype=object)
    
    def complex_modular_signal(self, base: int, modulus: int, length: int) -> np.ndarray:
        """
        Create complex wave signal: ψᵢ = e^(2πi·aⁱ mod N / N)
        
        This encodes the modular exponentiation as a rotating phase on the unit circle.
        Periodic behavior in the exponents creates constructive interference patterns.
        """
        mod_sequence = self.fast_modular_sequence(base, modulus, length)
        # Convert to float64 for phase calculation, handling both numpy arrays and object arrays
        if mod_sequence.dtype == object:
            # Handle large integers stored as Python objects
            phases = 2 * np.pi * np.array([float(x) for x in mod_sequence], dtype=np.float64) / float(modulus)
        else:
            # Handle regular numpy int64 arrays
            phases = 2 * np.pi * mod_sequence.astype(np.float64) / modulus
        return np.exp(1j * phases)
    
    def autocorrelation_interference(self, signal: np.ndarray, max_shift: int) -> List[Tuple[int, float]]:
        """
        Detect periods via phase alignment autocorrelation.
        
        For each shift d, compute: |∑ᵢ ψᵢ · ψ̄ᵢ₊ₐ|
        
        If the signal has true period r, this sum peaks at d = r due to
        constructive interference when the wave overlaps with itself.
        """
        signal_length = len(signal)
        interference_scores = []
        
        for d in range(1, min(max_shift, signal_length // 2)):
            # Compute overlap between signal and its shifted version
            overlap_length = signal_length - d
            original_segment = signal[:overlap_length]
            shifted_segment = signal[d:d + overlap_length]
            
            # Autocorrelation: sum of element-wise products with conjugate
            correlation = np.sum(original_segment * np.conj(shifted_segment))
            interference_strength = np.abs(correlation)
            
            # Normalize by overlap length for fair comparison across shifts
            normalized_strength = interference_strength / overlap_length
            
            interference_scores.append((d, normalized_strength))
        
        # Sort by interference strength (highest first)
        return sorted(interference_scores, key=lambda x: -x[1])
    
    def validate_period_candidates(self, base: int, modulus: int, 
                                 period_candidates: List[Tuple[int, float]]) -> List[int]:
        """
        Validate period candidates by checking if a^r ≡ 1 (mod N).
        This eliminates false positives from noise in the autocorrelation.
        """
        valid_periods = []
        
        for period, strength in period_candidates[:10]:  # Check top 10 candidates
            if period == 0:
                continue
                
            # Check if a^period ≡ 1 (mod N)
            if pow(base, period, modulus) == 1:
                valid_periods.append(period)
                print(f"✓ Valid period found: {period} (strength: {strength:.4f})")
            # else:
                # print(f"✗ False period rejected: {period} (strength: {strength:.4f})")
        
        return valid_periods
    
    def extract_factors_from_period(self, base: int, modulus: int, period: int) -> Optional[Tuple[int, int]]:
        """
        Extract factors using the classical method: gcd(a^(r/2) ± 1, N)
        
        This is the same approach as Shor's algorithm for quantum factorization.
        """
        if period % 2 != 0:
            print(f"Period {period} is odd, cannot use a^(r/2) method")
            return None
        
        half_period = period // 2
        y = pow(base, half_period, modulus)
        
        if y == 1 or y == modulus - 1:
            print(f"Trivial case: a^(r/2) = {y}, trying different approach")
            return None
        
        # Try both gcd(y-1, N) and gcd(y+1, N)
        factor1 = math.gcd(y - 1, modulus)
        factor2 = math.gcd(y + 1, modulus)
        
        for factor in [factor1, factor2]:
            if 1 < factor < modulus:
                complement = modulus // factor
                if factor * complement == modulus:
                    return (factor, complement)
        
        return None
    
    def multi_base_analysis(self, modulus: int, num_bases: int = 20) -> Optional[Tuple[int, int]]:
        """
        Try multiple bases to increase chances of finding a useful period.
        
        Different bases may have different multiplicative orders, giving us
        multiple chances to find a period that leads to successful factorization.
        """
        bases_to_try = []
        
        # Generate diverse bases coprime to N - try smaller bases first as they often have smaller periods
        small_bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        for base in small_bases:
            if base < modulus and math.gcd(base, modulus) == 1:
                bases_to_try.append(base)
                if len(bases_to_try) >= num_bases // 2:
                    break
        
        # Add some random bases
        for attempt in range(num_bases * 3):
            base = np.random.randint(2, min(1000, modulus))
            if math.gcd(base, modulus) == 1 and base not in bases_to_try:
                bases_to_try.append(base)
                if len(bases_to_try) >= num_bases:
                    break
        
        print(f"Testing bases: {bases_to_try[:10]}{'...' if len(bases_to_try) > 10 else ''}")
        
        # Adaptive signal length based on number size
        bit_length = modulus.bit_length()
        if bit_length <= 20:
            signal_length = self.max_signal_length
            max_shift = self.max_shift_search
        elif bit_length <= 30:
            signal_length = self.max_signal_length // 2
            max_shift = self.max_shift_search // 2
        else:
            signal_length = self.max_signal_length // 4
            max_shift = self.max_shift_search // 4
        
        print(f"Using signal_length={signal_length}, max_shift={max_shift} for {bit_length}-bit number")
        
        for base in bases_to_try:
            # Generate complex wave signal
            signal = self.complex_modular_signal(base, modulus, signal_length)
            
            # Detect periods via autocorrelation interference
            interference_scores = self.autocorrelation_interference(signal, max_shift)
            
            if not interference_scores:
                continue
            
            # Validate period candidates
            valid_periods = self.validate_period_candidates(base, modulus, interference_scores)
            
            # Try to extract factors from valid periods
            for period in valid_periods:
                factors = self.extract_factors_from_period(base, modulus, period)
                if factors:
                    print(f"🎯 SUCCESS! Found factors using base {base}, period {period}")
                    return factors
        
        return None
    
    def factorize(self, number: int) -> Optional[Tuple[int, int]]:
        """
        Main factorization routine using wave interference autocorrelation.
        
        Returns (p, q) if successful, None if factorization fails.
        """
        print(f"\n🌊 Wave Interference Factorization of {number}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Quick checks for trivial cases
        if number % 2 == 0:
            return (2, number // 2)
        
        # Try small prime factors first
        for p in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            if number % p == 0:
                return (p, number // p)
        
        # Main wave-based analysis
        result = self.multi_base_analysis(number)
        
        elapsed = time.time() - start_time
        
        if result:
            p, q = result
            print(f"\n✅ FACTORIZATION SUCCESSFUL!")
            print(f"   {number} = {p} × {q}")
            print(f"   Time: {elapsed:.3f}s")
            print(f"   Verification: {p * q == number}")
        else:
            print(f"\n❌ Factorization failed after {elapsed:.3f}s")
        
        return result


# --- Test Case Generator ---
def generate_test_case(bit_size: int) -> Tuple[int, int, int]:
    half = bit_size // 2
    p = randprime(2 ** (half - 1), 2 ** half)
    q = randprime(2 ** (half - 1), 2 ** half)
    while q == p:
        q = randprime(2 ** (half - 1), 2 ** half)
    return p * q, p, q


def test_wave_interference_factorization():
    """Test the wave interference factorization on various numbers."""
    
    print("🧪 Testing Wave Interference Autocorrelation Factorization")
    print("=" * 70)
    
    factorizer = WaveInterferenceFactorizer()
    
    # Focus on smaller test cases where the algorithm has a better chance
    # Start with 12-bit numbers and gradually increase
    test_cases = [generate_test_case(n) for n in range(12, 32, 2)]
    
    successes = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases):
        number, expected_p, expected_q = test_case
        print(f"\n[{i+1}/{total_tests}] Testing {number.bit_length()}-bit number: {number}")
        print(f"Expected factors: {expected_p} × {expected_q}")
        
        result = factorizer.factorize(number)
        if result:
            successes += 1
            actual_p, actual_q = result
            print(f"✅ SUCCESS: Found {actual_p} × {actual_q}")
        else:
            print(f"❌ FAILED: Could not factor {number}")
        print("\n" + "-" * 50)
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Successful factorizations: {successes}/{total_tests}")
    print(f"   Success rate: {100 * successes / total_tests:.1f}%")
    
    # If success rate is too low, show a few manual examples
    if successes / total_tests < 0.3:
        print(f"\n🔍 Testing a few known small examples:")
        small_tests = [143, 221, 323, 437, 493, 667, 713]  # Products of small primes
        for num in small_tests:
            print(f"\nTesting {num}...")
            result = factorizer.factorize(num)
            if result:
                print(f"✅ {num} = {result[0]} × {result[1]}")

if __name__ == "__main__":
    test_wave_interference_factorization()
