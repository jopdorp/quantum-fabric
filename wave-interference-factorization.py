#!/usr/bin/env python3
"""
Wave Interference Factorization (v6.4)
Autocorrelation‑Based Period Detection using Complex Modular Signals

Changes in 6.4
--------------
• **Single success print‑out** – suppress noisy duplicate prints coming from parallel
  threads.  Only the *first* thread to find a factor prints, then every other
  worker is cancelled.
• **Clean early‑exit** – after a factor is discovered, all still‑queued futures are
  cancelled and the executor is shut down so the harness moves immediately to
  the next RSA size.
• **Higher default depth cap** – lifted to `2**20` so RSA‑36+ has a better chance
  without manual tweaking.
• **Optional verbosity** – `wave_autocorr_factor()` now has a `verbose` flag; the
  worker threads call it with `False`, while the main thread prints the single
  success line.
"""

import time
from math import gcd, log2
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional

import numpy as np
from numba import jit
from sympy import randprime

# ──────────────────────────────────────────────────────────────────────────────
#  ❖  RSA test‑case generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_test_case(bit_size: int) -> Tuple[int, int, int]:
    """Return (N, p, q) with p·q == N and ~bit_size bits total."""
    half_bits = bit_size // 2
    min_p = 1 << (half_bits - 1)
    max_p = (1 << half_bits) - 1
    p = randprime(min_p, max_p)
    q = randprime(min_p, max_p)
    while q == p:
        q = randprime(min_p, max_p)
    return p * q, p, q

# ──────────────────────────────────────────────────────────────────────────────
#  ❖  Core math helpers
# ──────────────────────────────────────────────────────────────────────────────

@jit(nopython=True)
def fast_modmul_sequence(a: int, N: int, length: int):
    """Return [a, a², … a^length] mod N."""
    out = [0] * length
    x = a % N
    for i in range(length):
        out[i] = x
        x = (x * a) % N
    return out


def complex_mod_signal(a: int, N: int, length: int) -> np.ndarray:
    """Map the mod‑exp sequence onto the unit circle as complex phases."""
    seq = fast_modmul_sequence(a, N, length)
    return np.exp(2j * np.pi * np.asarray(seq) / N)


def top_autocorr_peaks(sig: np.ndarray, max_shift: int, top_k: int = 6):
    """Return [(shift, score)] of top autocorrelation magnitudes."""
    scores: List[Tuple[int, float]] = []
    for d in range(1, max_shift):
        shifted = np.roll(sig, d)
        score = abs(np.vdot(sig, shifted))  # vdot = conj() on first arg
        scores.append((d, score))
    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


# ──────────────────────────────────────────────────────────────────────────────
#  ❖  Parameter heuristics
# ──────────────────────────────────────────────────────────────────────────────

def estimate_max_depth(N: int, cap: int = 2 ** 20) -> int:
    """Empirical depth ≈ N^0.30  (cap at 1 048 576)."""
    return min(max(int(N ** 0.30), 8192), cap)


# ──────────────────────────────────────────────────────────────────────────────
#  ❖  Wave autocorrelation factor finder (single a)
# ──────────────────────────────────────────────────────────────────────────────

def wave_autocorr_factor(N: int, a: int, max_depth: int, *, verbose=False) -> Optional[int]:
    sig = complex_mod_signal(a, N, max_depth)
    for r, _score in top_autocorr_peaks(sig, max_shift=max_depth // 2):
        if r <= 0:
            continue
        y = pow(a, r // 2, N)
        if y in (1, N - 1):
            continue
        for delta in (-1, 1):
            f = gcd(y + delta, N)
            if 1 < f < N:
                if verbose:
                    print(f"[🎯] Factor via wave autocorrelation: {f} (period~{r})")
                return f
    return None


# ──────────────────────────────────────────────────────────────────────────────
#  ❖  Parallel sweep across many bases a
# ──────────────────────────────────────────────────────────────────────────────

def sweep_wave_autocorr(N: int, timeout_sec: int = 120) -> Optional[int]:
    """Return a non‑trivial factor of N (or None)."""

    max_depth = estimate_max_depth(N)
    print(f"[📐] Using max_depth = 2^{{{int(log2(max_depth))}}} = {max_depth} for N = {N}")

    # Helper so we can cancel fast once a factor appears
    def try_base(a: int):
        return a, wave_autocorr_factor(N, a, max_depth, verbose=False)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(try_base, a) for a in range(2, 512)]
        for fut in as_completed(futures):
            if time.time() - t0 > timeout_sec:
                break
            a, factor = fut.result()
            if factor:
                # Print once, cancel the rest, return.
                print(f"✅ SUCCESS: {N} = {factor} × {N // factor} (base {a})")
                # Cancel anything not yet started
                pool.shutdown(wait=False, cancel_futures=True)
                return factor

    # Optional deeper retry
    print("⚠️  First sweep failed – expanding depth …")
    max_depth *= 2
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(try_base, a) for a in range(2, 512)]
        for fut in as_completed(futures):
            if time.time() - t0 > timeout_sec * 2:
                break
            a, factor = fut.result()
            if factor:
                print(f"✅ SUCCESS: {N} = {factor} × {N // factor} (base {a})")
                pool.shutdown(wait=False, cancel_futures=True)
                return factor

    print("❌ FAILED: No factor found")
    return None


# ──────────────────────────────────────────────────────────────────────────────
#  ❖  Test‑harness across several RSA sizes
# ──────────────────────────────────────────────────────────────────────────────

def test_wave_autocorr():
    print("🌊 Wave Interference Factorization via Autocorrelation")
    print("=" * 64)

    for bits in (16, 20, 24, 28, 32, 36):
        print(f"\n🎯 Generating RSA-{bits} test case…")
        N, p, q = generate_test_case(bits)
        print(f"N = {N:,}")
        print(f"Expected factors: {p} × {q}")

        t0 = time.time()
        factor = sweep_wave_autocorr(N)
        dt = time.time() - t0
        print(f"⏱️  Time: {dt:.2f}s")

        if factor:
            other = N // factor
            ok = {factor, other} == {p, q}
            print("🎉 CORRECT FACTORS FOUND!" if ok else "⚠️  Wrong factors 🙃")
        print("-" * 50)


if __name__ == "__main__":
    test_wave_autocorr()
