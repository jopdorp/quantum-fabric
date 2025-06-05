import numpy as np
from sympy import gcd, randprime

def modexp_sequence(a, N, max_len):
    """Compute a^x mod N for x in [1, max_len]"""
    x = 1
    result = []
    for _ in range(max_len):
        x = (x * a) % N
        result.append(x)
    return result

def detect_cycle_hash(residues):
    """Use a simple hash map to find the first repeating value"""
    seen = {}
    for i, val in enumerate(residues):
        if val in seen:
            return i - seen[val]
        seen[val] = i
    return None

def phase_alignment_score(residues, N):
    """Compute interference magnitude from phase coherence"""
    phases = np.exp(2j * np.pi * np.array(residues) / N)
    score = np.abs(np.sum(phases)) / len(phases)
    return score

def try_factor(N, trials=10, max_len=2048, alignment_threshold=0.95):
    for _ in range(trials):
        a = np.random.randint(2, N)
        if gcd(a, N) != 1:
            return gcd(a, N)

        residues = modexp_sequence(a, N, max_len)
        r = detect_cycle_hash(residues)

        if not r:
            # Try fallback phase detection (not guaranteed to return r)
            score = phase_alignment_score(residues, N)
            if score < alignment_threshold:
                continue
            r = np.argmax([score])  # crude placeholder

        if r % 2 != 0:
            continue

        x = pow(a, r // 2, N)
        for delta in [-1, 1]:
            factor = gcd(x + delta, N)
            if 1 < factor < N:
                return factor
    return None

def test_factorizer(bits):
    p = randprime(2**(bits//2 - 1), 2**(bits//2))
    q = randprime(2**(bits//2 - 1), 2**(bits//2))
    N = p * q
    print(f"[RSA-{bits}] N = {N} = {p} × {q}")
    f = try_factor(N)
    if f:
        print(f"✅ Found factor: {f} × {N//f}")
    else:
        print("❌ Failed to factor")

if __name__ == "__main__":
    for bits in range(16, 36, 4):
        test_factorizer(bits)
