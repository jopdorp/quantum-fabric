import numpy as np
import time
from math import gcd, log2
from random import randint
from sympy import randprime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Constants

MAX_THREADS = 12

# Generate RSA test case
def generate_test_case(bit_size):
    half = bit_size // 2
    p = randprime(2**(half - 1), 2**half)
    q = randprime(2**(half - 1), 2**half)
    while q == p:
        q = randprime(2**(half - 1), 2**half)
    return p * q, p, q

# Modular exponentiation signal
def modular_phase_signal(a, N, length):
    x = a % N
    seq = []
    for _ in range(length):
        seq.append(np.exp(2j * np.pi * x / N))
        x = (x * a) % N
    return np.array(seq)

# LSTM-style feedback update for each cell
class Cell:
    def __init__(self):
        self.state = 0 + 0j
        self.memory = []

    def update(self, signal, step):
        self.memory.append(signal[step])
        if len(self.memory) > 10:
            self.memory.pop(0)
        self.state += sum(self.memory) / len(self.memory)
        return self.state 

# Simulate wavefront with attention-inspired accumulation
def simulate_wavefront(a, N, length):
    signal = modular_phase_signal(a, N, length)
    cells = [Cell() for _ in range(length)]
    interference = np.zeros(length)
    for step in range(length):
        for i, cell in enumerate(cells):
            interference[i] += abs(cell.update(signal, step))
    return interference

# Estimate period via accumulated interference
def detect_period(interference, top_k=5):
    peaks = sorted(((i, val) for i, val in enumerate(interference)), key=lambda x: -x[1])
    return [idx for idx, _ in peaks[:top_k] if idx > 1]

# Main factorization routine
def factor_with_wavefront(N, timeout=60):
    def try_base(a, length):
        interference = simulate_wavefront(a, N, length)
        for r in detect_period(interference):
            y = pow(a, r // 2, N)
            if y != 1 and y != N - 1:
                for delta in [-1, 1]:
                    f = gcd(y + delta, N)
                    if 1 < f < N:
                        return f
        return None

    length = min(max(int(N ** 0.3), 16384), 2**20)
    print(f"[📐] Using signal length: {length}")
    start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(try_base, randint(2, N - 2), length) for _ in range(60)]
        for future in as_completed(futures):
            if time.time() - start > timeout:
                print("🕒 Timeout")
                return None
            factor = future.result()
            if factor:
                return factor
    return None 

# Run full test
def run_rsa430():
    print("\n🔐 Running RSA-430 Test")
    N, p, q = generate_test_case(430)
    print(f"N = {N}")
    start = time.time()
    f = factor_with_wavefront(N)
    end = time.time()
    if f:
        print(f"✅ SUCCESS: {N} = {f} × {N//f}")
        print("🎉 CORRECT FACTORS FOUND!")
    else:
        print("❌ FAILED")
    print(f"⏱️ Time: {end - start:.2f} seconds")

if __name__ == "__main__":
    run_rsa430()
