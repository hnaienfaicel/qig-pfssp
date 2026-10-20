"""Iterated Greedy (Ruiz & Stutzle 2007) with interchangeable classical or quantum search subroutines.

Variants (see experiments/arms.json):
  IG        best insertion by full scan (n evaluations), IG local search
  IG-Rmin   best insertion by random search for the minimum (twin of the quantum minimum)
  Q-IG-min  best insertion by quantum minimum finding (Durr-Hoyer), IG local search
  IG-R      full-scan construction, descent over the insertion neighbourhood by uniform draws (twin of Grover)
  IG-RR     random-search construction + uniform-draw descent (complete twin of Q-IG-G)
  Q-IG-G    quantum-minimum construction + Grover (BBHT) descent
"""
import math, numpy as np
from .flowshop import makespan, insertion_scan, neh
from .quantum import quantum_argmin, random_argmin, insertion_descent

def best_insertion(p, x, job, rng, mode, cq, cpi=1):
    c = insertion_scan(p, x, int(job))
    if mode == "random": k, _, ch = random_argmin(c, rng, cq)
    elif mode == "scan": k = int(np.argmin(c)); ch = x.shape[0] + 1
    else: k, _, ch = quantum_argmin(c, rng, cq, cpi)
    return np.concatenate([x[:k], [job], x[k:]]).astype(np.int64), int(c[k]), ch

def ig_local_search(p, perm, c, rng, left, mode, cq, cpi=1):
    improved = True; perm = np.asarray(perm, dtype=np.int64)
    while improved and left[0] > 0:
        improved = False
        for job in rng.permutation(perm).tolist():
            if left[0] <= 0: break
            rest = perm[perm != job]; cand, val, ev = best_insertion(p, rest, job, rng, mode, cq, cpi); left[0] -= ev
            if val < c: perm, c, improved = cand, val, True
    return perm, c

def iterated_greedy(p, budget, seed, cq_min=None, rmin=None, ls="ig", cq_ls=3.0, cpi=1, d=4, Tf=0.4, return_perm=False):
    """Returns (best makespan, calls used, iterations). cq_min -> quantum minimum; rmin -> random-search twin;
    ls in {'ig', 'gls' (Grover descent), 'rls' (uniform-draw twin)}; cpi = oracle calls charged per Grover iteration;
    return_perm=True also returns the best permutation (0-based job indices)."""
    mode = "random" if rmin is not None else ("scan" if cq_min is None else "quantum"); cq = rmin if rmin is not None else cq_min
    rng = np.random.default_rng(seed); n, m = p.shape; T = Tf * p.sum() / (n * m * 10.0)
    nperm, c0, ev = neh(p); left = [budget - ev]; perm = np.asarray(nperm, dtype=np.int64); cmax = int(c0)
    def LS(x, c):
        if ls == "ig": return ig_local_search(p, x, c, rng, left, mode, cq, cpi)
        return insertion_descent(p, x, c, rng, left, cq_ls, twin=(ls == "rls"), cpi=cpi)
    perm, cmax = LS(perm, cmax); best, best_c = perm.copy(), cmax; it = 0
    while left[0] > 0:
        it += 1; cur = perm.copy(); rem = rng.choice(len(cur), size=min(d, len(cur) - 1), replace=False)
        jobs = [int(cur[i]) for i in sorted(rem, reverse=True)]; cur = np.array([j for j in cur if j not in jobs], dtype=np.int64)
        for j in jobs:
            cur, c, k = best_insertion(p, cur, j, rng, mode, cq, cpi); left[0] -= k
        cur, c = LS(cur, c)
        if c < cmax or rng.random() < math.exp(-(c - cmax) / max(T, 1e-9)): perm, cmax = cur, c
        if c < best_c: best, best_c = cur.copy(), c
    assert int(makespan(p, best)) == best_c and sorted(best.tolist()) == list(range(n))
    if return_perm: return int(best_c), budget - left[0], it, best
    return int(best_c), budget - left[0], it
