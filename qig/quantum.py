"""Exact emulation of amplitude-amplification subroutines in the query model, and their same-law classical twins.

For a deterministic oracle marking M of N register codes, k Grover iterations on the uniform superposition give a
marked code with probability sin^2((2k+1) theta), sin(theta) = sqrt(M/N), uniformly distributed over the marked codes;
otherwise a uniformly random unmarked code. BBHT (Boyer, Brassard, Hoyer & Tapp 1998) handles unknown M by drawing k
uniformly in [0, m) and growing m by lambda = 1.2. Each round costs `cpi` oracle calls per Grover iteration (1 by
default, the usual convention; 2 counts the oracle and its uncomputation separately) plus one classical evaluation to
verify the measured candidate.

Noise model (exploratory, Amendment 4): with eps > 0 the k iterations of a round stay coherent with probability
(1 - eps)^k; otherwise the register is fully decohered and the measurement returns a uniformly random code (the k
calls are still charged). eps = 0 reproduces the noiseless emulation exactly (no extra random draw)."""
import math, numpy as np
from .flowshop import all_insertions, shift_move

def quantum_argmin(c, rng, cq, cpi=1, eps=0.0):
    """Durr & Hoyer (1996) quantum minimum finding over the values c[0..n-1], on a register of 2^ceil(log2 n) codes.
    Each BBHT search below the current threshold stops after cq*sqrt(N) effort without success.
    Returns (index, value, charged calls)."""
    n = c.shape[0]; w = max(1, math.ceil(math.log2(n))); Nreg = 2 ** w; cap = cq * math.sqrt(Nreg)
    y = int(rng.integers(n)); cur = int(c[y]); charged = 1
    while True:
        marked = np.flatnonzero(c < cur); M = int(marked.size)
        theta = math.asin(math.sqrt(M / Nreg)) if M else 0.0; mm, eff, found, seen = 1.0, 0, None, set()
        while eff <= cap:
            k = int(rng.integers(0, int(math.ceil(mm)))); charged += cpi * k; eff += k + 1
            pm = math.sin((2 * k + 1) * theta) ** 2
            if eps > 0 and k > 0 and rng.random() >= (1 - eps) ** k: pm = M / Nreg     # decohered: uniform code
            if M and rng.random() < pm:
                found = int(marked[int(rng.integers(M))]); charged += 1; break
            if rng.random() < (n - M) / (Nreg - M):                       # measured a valid but unmarked code
                unm = np.flatnonzero(c >= cur); q = int(unm[int(rng.integers(unm.size))])
                if q not in seen: seen.add(q); charged += 1
            mm = min(1.2 * mm, math.sqrt(Nreg))
        if found is None: return y, cur, charged
        y = found; cur = int(c[y])

def random_argmin(c, rng, cq):
    """Classical twin with the same stopping rule: uniform draws without replacement, one evaluation each,
    stop after cq*sqrt(N) draws without improvement."""
    n = c.shape[0]; w = max(1, math.ceil(math.log2(n))); cap = cq * math.sqrt(2 ** w)
    order = rng.permutation(n); best = int(order[0]); charged, since = 1, 0
    for q in order[1:]:
        if since >= cap: break
        charged += 1; since += 1
        if c[q] < c[best]: best, since = int(q), 0
    return best, int(c[best]), charged

def insertion_descent(p, x, xc, rng, left, cq, twin, cpi=1, eps=0.0, rec=None):
    """Descent over the whole insertion neighbourhood. Each step searches for an improving move with BBHT
    (twin=False) or with uniform draws without replacement (twin=True, same output law). The move register uses
    i on w_i bits, a direction bit and a distance on w_m bits; a local optimum is declared after cq*sqrt(N) effort
    without success. `left` is the remaining budget (list of one int), decremented in place."""
    n = p.shape[0]
    wi = max(1, math.ceil(math.log2(n))); wm = max(1, math.ceil(math.log2(n - 1))); pw = 2 ** np.arange(wm - 1, -1, -1)
    Nreg = 2 ** (wi + 1 + wm); Nv = n * (n - 1); cap = cq * math.sqrt(Nreg)
    def move():
        while True:
            i = int(rng.integers(0, 2 ** wi)); sb = rng.random() < 0.5; m = int((rng.random(wm) < 0.5).astype(int) @ pw); j = i + (1 if sb else -1) * (m + 1)
            if i < n and 0 <= j < n: return i, j
    while left[0] > 0:
        C = all_insertions(p, x); mask = C < xc; np.fill_diagonal(mask, False); marked = np.flatnonzero(mask); M = int(marked.size)
        seen, eff, found = set(), 0, None
        if not twin:
            theta = math.asin(math.sqrt(M / Nreg)) if M else 0.0; mm = 1.0
            while eff <= cap and left[0] > 0:
                k = int(rng.integers(0, int(math.ceil(mm)))); left[0] -= cpi * k; eff += k + 1
                pm = math.sin((2 * k + 1) * theta) ** 2
                if eps > 0 and k > 0 and rng.random() >= (1 - eps) ** k: pm = M / Nreg  # decohered: uniform code
                if M and rng.random() < pm:
                    found = divmod(int(marked[int(rng.integers(M))]), n); left[0] -= 1; break
                if rng.random() < (Nv - M) / (Nreg - M):
                    while True:
                        ij = move()
                        if not mask[ij]: break
                    if ij not in seen: seen.add(ij); left[0] -= 1
                mm = min(1.2 * mm, math.sqrt(Nreg))
        else:
            while eff < cap and left[0] > 0:
                ij = move(); eff += 1
                if ij in seen: continue
                seen.add(ij); left[0] -= 1
                if mask[ij]: found = ij; break
        if found is None: return x, xc
        i, j = found; xc = int(C[i, j]); x = shift_move(x, i, j)
        if rec is not None: rec(xc)
    return x, xc
