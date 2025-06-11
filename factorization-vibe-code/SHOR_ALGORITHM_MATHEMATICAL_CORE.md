# Shor's Algorithm: Mathematical Core

## Problem Statement

**Input**: Composite integer $N = pq$ where $p, q$ are unknown prime factors  
**Goal**: Find $p$ and $q$ such that $N = pq$

## Mathematical Foundation

### Order Finding Problem
For a given base $a$ coprime to $N$, find the multiplicative order $r$ such that:
$$a^r \equiv 1 \pmod{N}$$

The order $r$ is the smallest positive integer satisfying this congruence.

### Key Theorem
If $r$ is even and $a^{r/2} \not\equiv \pm 1 \pmod{N}$, then:
$$\gcd(a^{r/2} - 1, N) \text{ and } \gcd(a^{r/2} + 1, N)$$
are non-trivial factors of $N$.

**Proof Sketch**: 
Since $a^r \equiv 1 \pmod{N}$, we have $(a^{r/2})^2 \equiv 1 \pmod{N}$, which means $(a^{r/2} - 1)(a^{r/2} + 1) \equiv 0 \pmod{N}$. If neither factor is $\pm 1 \pmod{N}$, then each shares a non-trivial factor with $N$.

## Shor's Quantum Algorithm

### Step 1: Base Selection
Choose random $a \in \{2, 3, \ldots, N-1\}$ such that $\gcd(a, N) = 1$.

### Step 2: Quantum Period Finding
Use quantum Fourier transform to find the period $r$ of the function:
$$f(x) = a^x \bmod N$$

**Quantum Circuit**:
1. Initialize superposition: $\frac{1}{\sqrt{Q}} \sum_{x=0}^{Q-1} |x\rangle |0\rangle$ where $Q = 2^n \geq N^2$
2. Compute function: $\frac{1}{\sqrt{Q}} \sum_{x=0}^{Q-1} |x\rangle |a^x \bmod N\rangle$
3. Measure second register, collapsing to: $\frac{1}{\sqrt{r}} \sum_{j=0}^{r-1} |x_0 + jr\rangle |a^{x_0} \bmod N\rangle$
4. Apply QFT to first register: $\frac{1}{\sqrt{rQ}} \sum_{j=0}^{r-1} \sum_{k=0}^{Q-1} e^{2\pi i k(x_0 + jr)/Q} |k\rangle$
5. Measure first register to obtain $k$ such that $k/Q \approx s/r$ for some $s < r$

### Step 3: Classical Post-Processing
From measurement outcome $k$, use continued fractions to find $r$ such that:
$$\left|\frac{k}{Q} - \frac{s}{r}\right| < \frac{1}{2Q}$$

### Step 4: Factor Extraction
If $r$ is even and $a^{r/2} \not\equiv \pm 1 \pmod{N}$:
$$p = \gcd(a^{r/2} - 1, N), \quad q = \gcd(a^{r/2} + 1, N)$$

## Complexity Analysis

**Time Complexity**: $O((\log N)^3)$ - polynomial in input size  
**Space Complexity**: $O(\log N)$ quantum bits  
**Success Probability**: $\geq 1/2$ per iteration  
**Expected Iterations**: $O(\log \log N)$

## Quantum Fourier Transform

The QFT maps:
$$|x\rangle \mapsto \frac{1}{\sqrt{Q}} \sum_{k=0}^{Q-1} e^{2\pi i xk/Q} |k\rangle$$

**Matrix Representation**:
$$\text{QFT} = \frac{1}{\sqrt{Q}} \begin{pmatrix}
1 & 1 & 1 & \cdots & 1 \\
1 & \omega & \omega^2 & \cdots & \omega^{Q-1} \\
1 & \omega^2 & \omega^4 & \cdots & \omega^{2(Q-1)} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & \omega^{Q-1} & \omega^{2(Q-1)} & \cdots & \omega^{(Q-1)^2}
\end{pmatrix}$$

where $\omega = e^{2\pi i/Q}$ is a primitive $Q$-th root of unity.

## Periodicity Detection

The quantum algorithm exploits interference patterns in the amplitude function:
$$A(k) = \frac{1}{\sqrt{rQ}} \left|\sum_{j=0}^{r-1} e^{2\pi i k(x_0 + jr)/Q}\right|$$

Constructive interference occurs when $kr/Q \approx$ integer, giving peaks at:
$$k \approx \frac{sQ}{r} \text{ for } s = 0, 1, 2, \ldots, r-1$$

## Example: Factoring $N = 15$

**Step 1**: Choose $a = 7$ (since $\gcd(7, 15) = 1$)  
**Step 2**: Find order of $7$ modulo $15$:
- $7^1 \equiv 7 \pmod{15}$
- $7^2 \equiv 4 \pmod{15}$  
- $7^3 \equiv 13 \pmod{15}$
- $7^4 \equiv 1 \pmod{15}$

So $r = 4$.

**Step 3**: Since $r = 4$ is even, compute:
- $7^{r/2} = 7^2 = 49 \equiv 4 \pmod{15}$
- Since $4 \not\equiv \pm 1 \pmod{15}$, proceed

**Step 4**: Extract factors:
- $\gcd(4 - 1, 15) = \gcd(3, 15) = 3$
- $\gcd(4 + 1, 15) = \gcd(5, 15) = 5$

Therefore $N = 15 = 3 \times 5$.

## Theoretical Significance

Shor's algorithm demonstrates that:
1. **Integer factorization** is in **BQP** (bounded-error quantum polynomial time)
2. If $\text{P} \neq \text{BQP}$, then factorization is not in **P** classically
3. **RSA cryptography** becomes vulnerable with sufficiently large quantum computers
4. **Quantum supremacy** exists for certain computational problems

The algorithm's polynomial scaling represents a **exponential speedup** over the best known classical algorithms (GNFS with complexity $O(\exp((\log N)^{1/3}))$).
