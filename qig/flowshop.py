"""Permutation flow shop kernels: makespan, Taillard-accelerated insertion scan, insertion neighbourhood, NEH."""
import numpy as np
from numba import njit

@njit(cache=True)
def makespan(p, perm):
    m = p.shape[1]; c = np.zeros(m, dtype=np.int64)
    for idx in range(perm.shape[0]):
        j = perm[idx]; c[0] += p[j, 0]
        for k in range(1, m):
            if c[k - 1] > c[k]: c[k] = c[k - 1]
            c[k] += p[j, k]
    return c[m - 1]

@njit(cache=True)
def shift_move(perm, i, j):
    """Remove the job at position i and reinsert it at position j."""
    if i == j: return perm.copy()
    n = perm.shape[0]; out = np.empty(n, dtype=perm.dtype); x = perm[i]; k = 0
    for t in range(n):
        if t != i: out[k] = perm[t]; k += 1
    res = np.empty(n, dtype=perm.dtype)
    for t in range(j): res[t] = out[t]
    res[j] = x
    for t in range(j, n - 1): res[t + 1] = out[t]
    return res

@njit(cache=True)
def insertion_scan(p, rest, job):
    """Makespans of the n1+1 insertion positions of `job` into `rest` (Taillard 1990 acceleration)."""
    n1 = rest.shape[0]; m = p.shape[1]
    e = np.zeros((n1 + 1, m + 1), np.int64)
    for i in range(1, n1 + 1):
        j = rest[i - 1]
        for k in range(1, m + 1):
            a = e[i - 1, k]; b = e[i, k - 1]; e[i, k] = (a if a > b else b) + p[j, k - 1]
    q = np.zeros((n1 + 2, m + 2), np.int64)
    for i in range(n1, 0, -1):
        j = rest[i - 1]
        for k in range(m, 0, -1):
            a = q[i + 1, k]; b = q[i, k + 1]; q[i, k] = (a if a > b else b) + p[j, k - 1]
    out = np.empty(n1 + 1, np.int64); f = np.zeros(m + 1, np.int64)
    for i in range(n1 + 1):
        best = 0
        for k in range(1, m + 1):
            a = e[i, k]; b = f[k - 1]; f[k] = (a if a > b else b) + p[job, k - 1]
            v = f[k] + q[i + 1, k]
            if v > best: best = v
        out[i] = best
    return out

@njit(cache=True)
def all_insertions(p, base):
    """C[i, j] = makespan of shift_move(base, i, j). Used only to emulate the quantum oracle (never charged)."""
    n = base.shape[0]; C = np.empty((n, n), np.int64); rest = np.empty(n - 1, np.int64)
    for i in range(n):
        k = 0
        for t in range(n):
            if t != i: rest[k] = base[t]; k += 1
        C[i, :] = insertion_scan(p, rest, base[i])
    return C

def neh(p):
    """NEH (Nawaz, Enscore & Ham 1983). Returns (perm, cmax, evaluations); one scan of k positions costs k."""
    order = list(np.argsort(-p.sum(axis=1), kind="stable")); perm = np.zeros(0, np.int64); ev = 0; cmax = 0
    for job in order:
        if perm.shape[0] == 0:
            perm = np.array([job], np.int64); cmax = int(p[job].sum()); ev += 1; continue
        c = insertion_scan(p, perm, int(job)); k = int(np.argmin(c))
        perm = np.concatenate([perm[:k], [job], perm[k:]]).astype(np.int64); cmax = int(c[k]); ev += perm.shape[0]
    return perm, cmax, ev
