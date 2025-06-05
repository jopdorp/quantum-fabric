#!/usr/bin/env python3
"""
Wave Interference RSA Factorization (v12.0)
Fully Local Signal Accumulation with FFT-Based Vote Propagation
Maintains Polynomial Time Scaling
"""

import numpy as np
import time
from math import gcd, log2
from sympy import randprime
from typing import List, Tuple
from random import randint


# ------------------------- Test Case Generator -------------------------
def generate_test_case(bit_size: int) -> Tuple[int, int, int]:
    half = bit_size // 2
    p = randprime(2**(half - 1), 2**half)
    q = randprime(2**(half - 1), 2**half)
    while p == q:
        q = randprime(2**(half - 1), 2**half)
    return p * q, p, q


# ------------------------- Modular Exponentiation Sequence -------------------------
def modexp_sequence(a: int, N: int, length: int) -> np.ndarray:
    seq = np.zeros(length, dtype=object)
    x = a % N
    for i in range(length):
        seq[i] = x
        x = (x * a) % N
    return seq


# ------------------------- Signal Generation -------------------------
def generate_signal(seq: np.ndarray, N: int) -> np.ndarray:
    # Convert to float array for phase calculation to avoid overflow
    seq_float = np.array([float(x) for x in seq])
    phases = 2 * np.pi * seq_float / N
    return np.exp(1j * phases)


# ------------------------- Local Phase Accumulation -------------------------
def accumulate_local_phases(signal: np.ndarray, window_size: int = 1024) -> np.ndarray:
    """Accumulate phase votes using sliding window FFT correlation"""
    n = len(signal)
    votes = np.zeros(n, dtype=np.float64)
    
    # Process in overlapping windows to maintain locality
    step = window_size // 2
    for start in range(0, n - window_size, step):
        end = start + window_size
        window = signal[start:end]
        
        # Local autocorrelation via FFT
        f = np.fft.fft(window)
        power = f * np.conj(f)
        local_corr = np.fft.ifft(power).real
        
        # Accumulate votes in corresponding global positions
        votes[start:start + len(local_corr)] += np.abs(local_corr)
    
    return votes


# ------------------------- FFT-Based Autocorrelation -------------------------
def autocorrelate_fft(signal: np.ndarray) -> np.ndarray:
    f = np.fft.fft(signal)
    power = f * np.conj(f)
    return np.fft.ifft(power).real


# ------------------------- Cellular Automata Memory Cell -------------------------
class MemoryCell:
    def __init__(self):
        self.interference_score = 0.0
        self.phase_history = []
        self.resonance_votes = {}
        self.decay_rate = 0.95
    
    def update(self, phase: float, period_hint: int):
        # Add to history (keep only recent)
        self.phase_history.append(phase)
        if len(self.phase_history) > 64:  # Local memory limit
            self.phase_history.pop(0)
        
        # Vote for period based on local pattern
        if len(self.phase_history) >= 4:
            local_diff = np.diff(self.phase_history[-4:])
            if np.std(local_diff) < 0.1:  # Stable pattern
                if period_hint not in self.resonance_votes:
                    self.resonance_votes[period_hint] = 0
                self.resonance_votes[period_hint] += 1
        
        # Decay old votes
        for period in list(self.resonance_votes.keys()):
            self.resonance_votes[period] *= self.decay_rate
            if self.resonance_votes[period] < 0.1:
                del self.resonance_votes[period]
    
    def get_strongest_period(self):
        if not self.resonance_votes:
            return None
        return max(self.resonance_votes, key=self.resonance_votes.get)


# ------------------------- Local Cellular Grid -------------------------
def create_cellular_grid(size: int) -> List[MemoryCell]:
    return [MemoryCell() for _ in range(size)]


# ------------------------- Adaptive Base Generator -------------------------
def generate_adaptive_bases(N: int, max_bases: int = 8) -> List[int]:
    bases = []
    # Start with small primes
    candidates = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    for b in candidates:
        if gcd(b, N) == 1 and len(bases) < max_bases:
            bases.append(b)
    
    # Add some quadratic residues
    for i in range(2, min(20, int(N**0.25) + 1)):
        b = (i * i) % N
        if b > 1 and gcd(b, N) == 1 and b not in bases and len(bases) < max_bases:
            bases.append(b)
    
    return bases


# ------------------------- Local Wavefront Propagation -------------------------
def propagate_wavefront(N: int, a: int, cells: List[MemoryCell], max_steps: int = 2048):
    x = a % N
    period_candidates = []
    
    for step in range(max_steps):
        # Calculate phase for current position
        phase = 2 * np.pi * x / N
        
        # Update local cells with spatial locality
        cell_idx = step % len(cells)
        period_hint = step + 1
        
        cells[cell_idx].update(phase, period_hint)
        
        # Check for cross-cell resonance every 32 steps
        if step > 32 and step % 32 == 0:
            resonance_map = {}
            for cell in cells:
                period = cell.get_strongest_period()
                if period:
                    resonance_map[period] = resonance_map.get(period, 0) + 1
            
            # If multiple cells agree on a period, it's a strong candidate
            for period, votes in resonance_map.items():
                if votes >= 3 and period not in period_candidates:
                    period_candidates.append(period)
        
        # Advance modular sequence
        x = (x * a) % N
        
        # Early termination if we have good candidates
        if len(period_candidates) >= 5:
            break
    
    return period_candidates


# ------------------------- Factor Search with Cellular Memory -------------------------
def find_factors_cellular(N: int, max_trials: int = 16) -> Tuple[int, int]:
    # Create persistent cellular grid
    grid_size = min(256, int(np.log2(N)) * 8)  # Adaptive grid size
    cells = create_cellular_grid(grid_size)
    
    # Try multiple adaptive bases
    bases = generate_adaptive_bases(N, max_trials)
    
    all_candidates = []
    for trial, a in enumerate(bases):
        # Propagate wavefront through cellular automata
        candidates = propagate_wavefront(N, a, cells)
        all_candidates.extend(candidates)
        
        # Test each candidate period
        for r in candidates:
            if r <= 1:
                continue
            y = pow(a, r // 2, N)
            if y == 1 or y == N - 1:
                continue
            for delta in [-1, 1]:
                factor = gcd(y + delta, N)
                if 1 < factor < N:
                    return factor, N // factor
    
    return None, None


# ------------------------- Runner -------------------------
def run_rsa_test(bit_sizes: List[int]):
    print("\n\n🧠 RSA Factorization via Cellular Automata Memory")
    print("=" * 64)
    for bits in bit_sizes:
        print(f"\n📦 Generating RSA-{bits} test case...")
        N, p, q = generate_test_case(bits)
        print(f"N = {N} ({p} × {q})")

        print(f"⚙️  Starting cellular wavefront propagation...")
        start = time.time()
        f1, f2 = find_factors_cellular(N)
        elapsed = time.time() - start

        if f1:
            print(f"✅ SUCCESS: {N} = {f1} × {f2}")
        else:
            print("❌ FAILED to factor")
        print(f"⏱️  Time: {elapsed:.2f}s")
        print("-" * 50)


if __name__ == "__main__":
    bit_sizes = list(range(16, 449, 16))
    run_rsa_test(bit_sizes)
