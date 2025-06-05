import numpy as np
from sympy import randprime, gcd
import matplotlib.pyplot as plt

# ---- PARAMETERS ----
GRID_SIZE = 256
TIME_STEPS = 64
PHASE_MOD = 2 * np.pi
BUFFER_SIZE = 16

# ---- UTILITIES ----
def generate_rsa_test(bits):
    p = randprime(2**(bits//2 - 1), 2**(bits//2))
    q = randprime(2**(bits//2 - 1), 2**(bits//2))
    while q == p:
        q = randprime(2**(bits//2 - 1), 2**(bits//2))
    return p * q, p, q

def modexp_sequence(a, N, length):
    seq = np.zeros(length, dtype=np.int64)
    x = a % N
    for i in range(length):
        seq[i] = x
        x = (x * a) % N
    return seq

def compute_alignment(buffer, modN):
    phases = np.exp(2j * np.pi * np.array(buffer) / modN)
    return np.abs(np.sum(phases)) / len(phases)

# ---- CELLULAR AUTOMATON ----
def ca_phase_update(cells, residues, modN, history):
    new_cells = np.copy(cells)
    for i in range(1, len(cells)-1):
        phase = residues[i] / modN * PHASE_MOD
        delta = np.exp(1j * phase)
        avg = (cells[i-1] + cells[i] + cells[i+1]) / 3

        if not np.isfinite(avg):
            avg = 1.0 + 0j

        if len(history[i]) >= BUFFER_SIZE:
            history[i] = history[i][1:]
        history[i].append(residues[i])

        if len(history[i]) >= 2:
            alignment = compute_alignment(history[i], modN)
        else:
            alignment = 1.0

        signal_strength = np.clip(alignment, 0, 1.0)

        new_value = avg * delta * (1 + 0.5 * signal_strength)
        if np.isfinite(new_value) and abs(new_value) < 1e10:
            new_cells[i] = new_value
        else:
            new_cells[i] = avg * delta
    return new_cells

def run_wave_ca(N, a, grid_size=GRID_SIZE, steps=TIME_STEPS):
    residues = modexp_sequence(a, N, grid_size)
    cells = np.ones(grid_size, dtype=np.complex128)
    history = [[] for _ in range(grid_size)]
    history_energy = []

    for t in range(steps):
        cells = ca_phase_update(cells, residues, N, history)
        energy = np.abs(cells)**2
        energy = np.clip(energy, 0, 1e10)
        history_energy.append(energy)

    summed = np.sum(history_energy, axis=0)
    return summed, residues

# ---- HASH-BASED CYCLE DETECTION ----
def detect_cycle(residues):
    seen = {}
    for i, val in enumerate(residues):
        if val in seen:
            return i - seen[val]
        seen[val] = i
    return None

# ---- FACTORIZATION ----
def try_order(N, a, r):
    if r < 2 or r % 2 != 0:
        return None
    try:
        y = pow(a, r // 2, N)
        for delta in [-1, 1]:
            candidate = y + delta
            if candidate > 0:
                f = gcd(candidate, N)
                if 1 < f < N:
                    return f
    except (ValueError, OverflowError):
        return None
    return None

def factor_via_ca(N, trials=50):
    for _ in range(trials):
        a = np.random.randint(2, N - 1)
        if gcd(a, N) != 1:
            factor = gcd(a, N)
            if factor > 1:
                return factor

        response, residues = run_wave_ca(N, a)
        r_peak = np.argmax(response)
        f1 = try_order(N, a, r_peak)
        if f1:
            return f1

        # Try detected cycle length
        r_hash = detect_cycle(residues)
        if r_hash:
            f2 = try_order(N, a, r_hash)
            if f2:
                return f2
    return None

# ---- TEST ----
def test_ca_factorizer(bits):
    N, p, q = generate_rsa_test(bits)
    print(f"[RSA-{bits}] N = {N}\nExpected: {p} × {q}")
    f = factor_via_ca(N)
    if f:
        print(f"✅ Found factor: {f} × {N//f}")
    else:
        print("❌ Failed to factor")

# ---- MAIN ----
if __name__ == "__main__":
    for bits in range(16, 68, 4):
        test_ca_factorizer(bits)
