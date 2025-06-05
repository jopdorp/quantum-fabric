#!/usr/bin/env python3
"""
Wave Interference Factorization (v9.1)
Hybrid LSTM-Attention + Phase Interference + Local Memory Cells
"""

import numpy as np
import time
from math import gcd, log2
from sympy import randprime
from typing import Optional
from random import randint
from numba import jit
import torch
import torch.nn as nn

# ----- CONFIG -----
MAX_WORKERS = 12
MAX_TIMEOUT = 180  # seconds

# ----- UTILITIES -----
def generate_test_case(bit_size):
    half = bit_size // 2
    p = randprime(2**(half - 1), 2**half)
    q = randprime(2**(half - 1), 2**half)
    while p == q:
        q = randprime(2**(half - 1), 2**half)
    return p * q, p, q

@jit(nopython=True)
def fast_modexp_seq(a, N, length):
    res = [0] * length
    x = a % N
    for i in range(length):
        res[i] = x
        x = (x * a) % N
    return res

class MemoryCell(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_dim, batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1, batch_first=True)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        score = self.out(attn_out)
        return score.squeeze(-1)


def prepare_input(sequence, N):
    normed = np.array(sequence) / N
    phase = np.exp(2j * np.pi * normed)
    real_imag = np.stack([phase.real, phase.imag], axis=-1)
    return torch.tensor(real_imag, dtype=torch.float32).unsqueeze(0)  # [1, L, 2]


def detect_period_with_nn(signal_tensor, model):
    with torch.no_grad():
        score_seq = model(signal_tensor).squeeze(0).numpy()
        top = np.argsort(score_seq)[-10:][::-1]
        return top


def wave_lstm_attention_factor(N, a, model, depth):
    seq = fast_modexp_seq(a, N, depth)
    x = prepare_input(seq, N)
    peaks = detect_period_with_nn(x, model)
    for r in peaks:
        if r <= 0:
            continue
        r = int(r)  # Convert numpy int64 to Python int
        y = pow(a, r // 2, N)
        if y != 1 and y != N - 1:
            for delta in [-1, 1]:
                f = gcd(y + delta, N)
                if 1 < f < N:
                    print(f"[✅] Factor via NN hybrid wave: {f} (period ~{r})")
                    return f
    return None


def random_coprime(N):
    while True:
        a = randint(2, N - 2)
        if gcd(a, N) == 1:
            return a


def factor_rsa(N, trials=90, depth=None):
    if depth is None:
        depth = min(max(2**13, int(N**0.3)), 2**18)

    print(f"[📐] depth = {depth} for N = {N}")
    model = MemoryCell(hidden_dim=64)
    model.eval()  # untrained, used as feature extractor

    start = time.time()
    for _ in range(trials):
        a = random_coprime(N)
        f = wave_lstm_attention_factor(N, a, model, depth)
        if f:
            print(f"🎯 {N} = {f} × {N // f} (base {a})")
            return f
        if time.time() - start > MAX_TIMEOUT:
            print("❌ Timeout")
            break
    return None


def run_wave_until(bits=256, step=16):
    print("🌊 Neural Wave Factorization")
    for b in range(16, bits + 1, step):
        print("=" * 60)
        print(f"🔢 RSA-{b} test case")
        N, p, q = generate_test_case(b)
        print(f"N = {N}\nExpected: {p} × {q}")
        start = time.time()
        f = factor_rsa(N)
        elapsed = time.time() - start
        if f:
            print(f"✅ Success in {elapsed:.2f}s")
        else:
            print(f"❌ Failed in {elapsed:.2f}s")


if __name__ == "__main__":
    run_wave_until(400, step=4)
