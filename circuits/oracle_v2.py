"""Oracle v2 for Grover search over the insertion neighbourhood of a PFSSP permutation (makespan threshold oracle).

The base permutation pi is CLASSICAL (it is fixed during one descent step); only the move register is quantum.
Hence the job at position k of the child sequence is a function of (k, move) that can be expressed with a few
flag qubits, and the processing times are classical constants fanned out with CNOTs (Clifford) under those flags.

Single insertion, move (i, j) on 2w qubits (w = ceil(log2 n)), child = shift(pi, i, j):
    sigma_k = pi_k                 (stay)
            = pi_{k+1}  if L_k = [i <= k < j]
            = pi_{k-1}  if R_k = [j < k <= i]
            = pi_i      if E_k = [k == j]
  * one-hot e^i, e^j by unary iteration (in-place splitting of prefix flags, ~n-2 temporary ANDs each);
  * D_k = XOR_{t<=k} (e^i_t ^ e^j_t) = [min(i,j) <= k < max(i,j)]  (CNOT chain, Clifford);
  * dir = [i < j] (w-bit comparator), L_k = D_k AND dir, R_{k+1} = D_k XOR L_k  (n-1 ANDs);
  * Q_r = p(pi_i, r): table lookup over i done by CNOT fan-out from e^i (Clifford, no extra Toffoli);
  * P_{k,r} = x ^ L(x^y) ^ R(x^z) ^ E x ^ (E AND Q_r) with x,y,z = p(pi_k|pi_{k+1}|pi_{k-1}, r)  (bp ANDs per cell).
DP C[k][r] = max(C[k-1][r], C[k][r-1]) + P_{k,r} on b-bit registers kept until the inverse oracle (Bennett), with
  * Gidney (2018) temporary logical-AND: an AND writes a|0> target (one CCZ/4T), its uncomputation is an X-basis
    measurement + classically controlled CZ (no T).  Gates here: 'and' (compute) and 'unand' (measurement-based).
  * comparator f = [A < B] = carry-out of B + ~A (b ANDs), select M = A ^ f(A^B) (b ANDs, f fanned out),
    in-place adder M += P (b-1 ANDs).  All carry chains uncomputed by measurement.
  * wavefront: cells of an anti-diagonal are independent; each machine column r owns dedicated temporaries
    (input copies, carry chains, P), so the ASAP schedule runs a whole anti-diagonal in parallel.
  * width b = ceil(log2(U+1)) with U a permutation-independent bound on every path length (see path_bound).
Mark = valid AND [Cmax < thr], valid = [i<n][j<n][i!=j] (invalid codes are never marked).

Double insertion, move (i, j, k, l) on 4w qubits: child = shift(shift(pi, i, j), k, l).  Row q of the child is
tau_{q'} with q' in {q, q+1, q-1} (second move) and tau_{q'} in {pi_{q'}, pi_{q'+1}, pi_{q'-1}, pi_i}; the 12
products of first/second-move flags select classical constants; two quantum-data terms remain: G_q AND Q1 (job pi_i
not moved again) and E2_q AND Q2 with Q2 = p(tau_k, .) obtained by a flag-weighted lookup over k.

The circuit is recorded as a flat gate list (op, a, b, c) and can be (i) simulated exactly on computational basis
states for all move codes at once (vectorised), with assertions that every AND target is |0> and every
measurement-based uncomputation sees target == a AND b, (ii) scheduled ASAP to get depth / Toffoli depth,
(iii) exported to Qiskit (and/unand -> ccx, or unand -> H, measure, conditional CZ, X) for statevector checks.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List

import numpy as np

X, CX, CCX, AND, UNAND, CZ, Z, H, CSWAP = range(9)
NAMES = ["x", "cx", "ccx", "and", "unand", "cz", "z", "h", "cswap"]
INV = {AND: UNAND, UNAND: AND}


def bits(v: int, nb: int) -> List[int]:
    return [t for t in range(nb) if (int(v) >> t) & 1]


# ----------------------------------------------------------------------------------------------- circuit recorder
class Circ:
    def __init__(self):
        self.nq = 0
        self.ops: List[tuple] = []
        self.blk: List[int] = []
        self.block_names: List[str] = []
        self._b = self._bid("misc")
        self.regs: Dict[str, List[int]] = {}

    def _bid(self, name):
        if name not in self.block_names:
            self.block_names.append(name)
        return self.block_names.index(name)

    def block(self, name):
        self._b = self._bid(name)

    def alloc(self, size, name=None):
        r = list(range(self.nq, self.nq + size)); self.nq += size
        if name is not None:
            self.regs[name] = r
        return r

    def one(self, name=None):
        return self.alloc(1, name)[0]

    def _g(self, op, a=-1, b=-1, c=-1):
        self.ops.append((op, a, b, c)); self.blk.append(self._b)

    def x(self, q): self._g(X, q)
    def z(self, q): self._g(Z, q)
    def h(self, q): self._g(H, q)
    def cx(self, c, t): self._g(CX, c, t)
    def cz(self, a, b): self._g(CZ, a, b)
    def ccx(self, a, b, t): self._g(CCX, a, b, t)
    def cswap(self, c, a, b): self._g(CSWAP, c, a, b)
    def and_(self, a, b, t): self._g(AND, a, b, t)
    def unand(self, a, b, t): self._g(UNAND, a, b, t)

    def mark(self):
        return len(self.ops)

    def emit_inverse(self, start, stop=None):
        """Append the inverse of ops[start:stop] (reverse order, AND <-> measurement-based UNAND)."""
        seg = self.ops[start:stop]; sb = self.blk[start:stop]
        for (op, a, b, c), bl in zip(reversed(seg), reversed(sb)):
            self.ops.append((INV.get(op, op), a, b, c)); self.blk.append(bl)


def inverse_ops(ops: Iterable[tuple]):
    return [(INV.get(op, op), a, b, c) for (op, a, b, c) in reversed(list(ops))]


# ----------------------------------------------------------------------------------------------- analysis tools
def counts(ops, blk=None, block_names=None):
    tot = np.zeros(len(NAMES), dtype=np.int64)
    per = {}
    for idx, g in enumerate(ops):
        tot[g[0]] += 1
        if blk is not None:
            nm = block_names[blk[idx]]
            per.setdefault(nm, np.zeros(len(NAMES), dtype=np.int64))[g[0]] += 1
    out = {NAMES[k]: int(tot[k]) for k in range(len(NAMES))}
    if blk is not None:
        return out, {k: {NAMES[t]: int(v[t]) for t in range(len(NAMES))} for k, v in per.items()}
    return out


def schedule(ops: Iterable[tuple], nq: int, dur, want_layers=False):
    """ASAP schedule. dur[op] = duration of each op type (0 allowed). Returns depth (and, if want_layers, a dict
    layer -> number of Toffoli-like ops starting at that layer, for ops with dur > 0 among CCX/AND/CSWAP)."""
    ready = [0] * nq
    layers = {} if want_layers else None
    tof = (CCX, AND, CSWAP)
    for (op, a, b, c) in ops:
        d = dur[op]
        if b < 0:
            s = ready[a]; ready[a] = s + d
        elif c < 0:
            s = ready[a] if ready[a] > ready[b] else ready[b]; ready[a] = ready[b] = s + d
        else:
            s = ready[a]
            if ready[b] > s: s = ready[b]
            if ready[c] > s: s = ready[c]
            ready[a] = ready[b] = ready[c] = s + d
        if want_layers and op in tof:
            layers[s] = layers.get(s, 0) + 1
    D = max(ready) if ready else 0
    return (D, layers) if want_layers else D


DUR_ALL = [1] * 9                                  # every gate one layer (Toffoli = one layer)
DUR_TOF = [0, 0, 1, 1, 0, 0, 0, 0, 1]              # Toffoli depth: CCX / AND / CSWAP one layer, Clifford free


def simulate(ops, nq, init: Dict[int, np.ndarray], N: int, check=True):
    """Exact basis-state simulation for N input states at once (S[q] is a bool vector over the N states).
    AND asserts a |0> target; UNAND asserts target == a & b (validity of measurement-based uncomputation)
    and resets it.  Phase gates (Z, CZ) are ignored (no effect on basis states); H is not allowed."""
    S = np.zeros((nq, N), dtype=bool)
    for q, v in init.items():
        S[q] = v
    for (op, a, b, c) in ops:
        if op == X:
            np.logical_not(S[a], out=S[a])
        elif op == CX:
            S[b] ^= S[a]
        elif op == CCX:
            S[c] ^= S[a] & S[b]
        elif op == AND:
            if check and S[c].any():
                raise AssertionError(f"AND target {c} not |0>")
            np.logical_and(S[a], S[b], out=S[c])
        elif op == UNAND:
            if check and not np.array_equal(S[c], S[a] & S[b]):
                raise AssertionError(f"UNAND precondition violated on {c}")
            S[c] = False
        elif op == CSWAP:
            sw = S[a] & (S[b] ^ S[c]); S[b] ^= sw; S[c] ^= sw
        elif op in (Z, CZ):
            pass
        else:
            raise ValueError(NAMES[op])
    return S


def read_reg(S, reg):
    v = np.zeros(S.shape[1], dtype=np.int64)
    for t, q in enumerate(reg):
        v |= S[q].astype(np.int64) << t
    return v


# ----------------------------------------------------------------------------------------------- widths
def path_bound(p: np.ndarray) -> int:
    """Permutation-independent upper bound on every DP value C[k][r] (any permutation of the jobs).
    A monotone lattice path visits every row; charge the first cell of each row to the row maximum and every
    right step to a distinct column maximum:  U1 = sum_j max_r p(j,r) + sum_{r>=1} max_j p(j,r).  Symmetrically
    U2 = sum_r max_j p(j,r) + sum of the n-1 largest row maxima.  Return min(U1, U2)."""
    p = np.asarray(p, dtype=np.int64)
    rmax = p.max(axis=1); cmax = p.max(axis=0)
    U1 = int(rmax.sum() + cmax[1:].sum())
    U2 = int(cmax.sum() + np.sort(rmax)[::-1][: p.shape[0] - 1].sum())
    return min(U1, U2)


def width_v2(p) -> int:
    return max(1, int(math.ceil(math.log2(path_bound(p) + 1))))


def width_p(p) -> int:
    return max(1, int(math.ceil(math.log2(int(np.max(p)) + 1))))


def width_v1(p) -> int:
    return int(math.ceil(math.log2(int(np.asarray(p).sum()) + 1)))


# ----------------------------------------------------------------------------------------------- primitives
def cmp_lt(c: Circ, A, B, f, tmp):
    """f <- [A < B] (f fresh |0>) as carry-out of B + ~A; Gidney carry chain, b ANDs, b-1 measured.  A, B restored.
    tmp: b-1 fresh qubits, returned to |0>."""
    b = len(A)
    for q in A: c.x(q)
    car = [None] + list(tmp[: b - 1]) + [f]            # car[t] = carry into bit t, car[b] = f
    c.and_(A[0], B[0], car[1])
    for t in range(1, b):
        ct = car[t]
        c.cx(ct, A[t]); c.cx(ct, B[t]); c.and_(A[t], B[t], car[t + 1]); c.cx(ct, car[t + 1])
    for s in range(b - 1, 0, -1):                       # restore A_s, B_s, then uncompute car[s]
        cs = car[s]
        c.cx(cs, A[s]); c.cx(cs, B[s])
        if s >= 2:
            c.cx(car[s - 1], cs)
        c.unand(A[s - 1], B[s - 1], cs)
    for q in A: c.x(q)


def select_max(c: Circ, A, B, f, M, fcopies):
    """M <- max(A, B) = A ^ f (A ^ B) given f = [A < B]; M fresh.  f fanned out to fcopies (CNOT tree) so that the
    b ANDs are parallel.  A, B restored, fcopies returned to |0>.  b ANDs."""
    b = len(A)
    st = c.mark()
    fs = [f]; pool = list(fcopies[: b - 1])
    while pool:
        new = []
        for s in list(fs):
            if not pool: break
            q = pool.pop(0); c.cx(s, q); new.append(q)
        fs += new
    fan_end = c.mark()
    for t in range(b): c.cx(A[t], B[t])
    for t in range(b): c.and_(fs[t], B[t], M[t])
    for t in range(b): c.cx(A[t], B[t])
    for t in range(b): c.cx(A[t], M[t])
    c.emit_inverse(st, fan_end)                         # un-fan f


def add_inplace(c: Circ, P, M, tmp):
    """M <- M + P mod 2^b (Gidney 2018), P of bp <= b bits restored; tmp: b-1 fresh qubits; b-1 ANDs."""
    b = len(M); bp = len(P)
    car = [None] + list(tmp[: b - 1])
    for t in range(0, b - 1):
        if t == 0:
            c.and_(P[0], M[0], car[1])
        elif t < bp:
            ct = car[t]
            c.cx(ct, P[t]); c.cx(ct, M[t]); c.and_(P[t], M[t], car[t + 1]); c.cx(ct, car[t + 1])
        else:
            c.and_(M[t], car[t], car[t + 1])
    if b >= 2:
        c.cx(car[b - 1], M[b - 1])
    if b - 1 < bp:
        c.cx(P[b - 1], M[b - 1])
    for t in range(b - 2, -1, -1):
        if t == 0:
            c.unand(P[0], M[0], car[1]); c.cx(P[0], M[0])
        elif t < bp:
            ct = car[t]
            c.cx(ct, car[t + 1]); c.unand(P[t], M[t], car[t + 1]); c.cx(ct, P[t]); c.cx(P[t], M[t])
        else:
            c.unand(M[t], car[t], car[t + 1]); c.cx(car[t], M[t])


def lt_const(c: Circ, Xr, K: int, g, tmp):
    """g <- [X < K] for a classical K (g fresh). [X < K] = NOT carry-out(X + 2^b - K); <= b-1 ANDs, measured."""
    b = len(Xr)
    if K <= 0:
        return
    if K >= (1 << b):
        c.x(g); return
    kb = [((1 << b) - K) >> t & 1 for t in range(b)]
    st = c.mark(); cur = None; ti = 0
    for t in range(b):
        if cur is None:
            if kb[t]:
                cur = Xr[t]                              # c_{t+1} = X_t (alias, no gate)
            continue
        new = tmp[ti]; ti += 1
        if kb[t] == 0:                                   # c_{t+1} = X_t AND c_t
            c.and_(Xr[t], cur, new)
        else:                                            # c_{t+1} = X_t OR c_t
            c.x(Xr[t]); c.x(cur); c.and_(Xr[t], cur, new); c.x(new); c.x(Xr[t]); c.x(cur)
        cur = new
    en = c.mark()
    c.cx(cur, g); c.x(g)
    c.emit_inverse(st, en)


def eq_flag(c: Circ, A, B, out, tmp):
    """out <- [A == B] (out fresh), w-1 ANDs (intermediates measured at once). A, B restored."""
    w = len(A)
    for t in range(w): c.cx(A[t], B[t]); c.x(B[t])
    if w == 1:
        c.cx(B[0], out)
    else:
        prev = B[0]; chain = []
        for t in range(1, w):
            tgt = out if t == w - 1 else tmp[t - 1]
            c.and_(prev, B[t], tgt); chain.append((prev, B[t], tgt)); prev = tgt
        for (a, bq, tg) in reversed(chain[:-1]):
            c.unand(a, bq, tg)
    for t in range(w): c.x(B[t]); c.cx(A[t], B[t])


def and_many(c: Circ, ctrls, out, tmp, neg=()):
    """out <- AND of ctrls (out fresh), negated controls in `neg`; len-1 ANDs; intermediates measured."""
    for q in neg: c.x(q)
    if len(ctrls) == 1:
        c.cx(ctrls[0], out)
    else:
        prev = ctrls[0]; chain = []
        for t in range(1, len(ctrls)):
            tgt = out if t == len(ctrls) - 1 else tmp[t - 1]
            c.and_(prev, ctrls[t], tgt); chain.append((prev, ctrls[t], tgt)); prev = tgt
        for (a, bq, tg) in reversed(chain[:-1]):
            c.unand(a, bq, tg)
    for q in neg: c.x(q)


def onehot(c: Circ, xr, n):
    """Unary iteration by in-place splitting of prefix flags (MSB first): node q=[prefix] splits into
    r = q AND x_t (fresh) and q <- q ^ r = q AND NOT x_t.  Nodes whose code range lies entirely at or above n are
    not split (kept as 'garbage' flags, uncomputed by the inverse).  Returns (e[0..n-1], garbage)."""
    w = len(xr); top = w - 1
    q1 = c.one(); c.cx(xr[top], q1)
    q0 = c.one(); c.cx(xr[top], q0); c.x(q0)
    nodes = [(q0, 0), (q1, 1 << top)]; garbage = []
    for t in range(top - 1, -1, -1):
        new = []
        for (q, lo) in nodes:
            if lo >= n:
                garbage.append(q); continue
            r = c.one(); c.and_(q, xr[t], r); c.cx(r, q)
            new += [(q, lo), (r, lo + (1 << t))]
        nodes = new
    e = [None] * n
    for (q, lo) in nodes:
        if lo < n: e[lo] = q
        else: garbage.append(q)
    assert all(v is not None for v in e)
    return e, garbage


def parity_flag(c: Circ, qs):
    v = c.one()
    for q in qs: c.cx(q, v)
    return v


def move_flags(c: Circ, xi, xj, ei, ej, n, w):
    """Flags of one insertion move: L[k] = [i <= k < j] (k=0..n-2), R[k] = [j < k <= i] (k=1..n-1)."""
    D = c.alloc(n - 1)
    for k in range(n - 1):
        if k > 0: c.cx(D[k - 1], D[k])
        c.cx(ei[k], D[k]); c.cx(ej[k], D[k])
    dirq = c.one(); tmp = c.alloc(max(1, w - 1))
    cmp_lt(c, xi, xj, dirq, tmp)                         # dir = [i < j]
    L = c.alloc(n - 1)
    for k in range(n - 1):
        c.and_(D[k], dirq, L[k]); c.cx(L[k], D[k])       # D[k] now = R[k+1]
    R = {k + 1: D[k] for k in range(n - 1)}
    return {k: L[k] for k in range(n - 1)}, R, dirq


def validity(c: Circ, pairs, w):
    """valid = AND over moves (xi, xj, ei, ej) of [i<n][j<n][i!=j]."""
    terms, neg = [], []
    for (xi, xj, ei, ej) in pairs:
        vi = parity_flag(c, ei); vj = parity_flag(c, ej)
        eq = c.one(); eq_flag(c, xi, xj, eq, c.alloc(max(1, w - 2)))
        terms += [vi, vj, eq]; neg.append(eq)
    valid = c.one(); and_many(c, terms, valid, c.alloc(len(terms) - 2), neg=neg)
    return valid


# ----------------------------------------------------------------------------------------------- DP engine
def _dp(c: Circ, n, m, b, bp, load, nP=1, serial=False):
    """Wavefront DP. load(k, r, Ts) writes p(sigma_k, r) into fresh registers Ts (list of nP bp-bit registers, the
    addend being Ts[0]) and must be self-contained so that its inverse unloads.  Returns C, F registers."""
    C = [[c.alloc(b) for r in range(m)] for k in range(n)]
    F = [[c.one() if (k > 0 and r > 0) else None for r in range(m)] for k in range(n)]
    slot = []
    for r in range(1 if serial else m):
        slot.append(dict(A=c.alloc(b), B=c.alloc(b), ch=c.alloc(max(1, b - 1)), fc=c.alloc(max(1, b - 1)),
                         P=[c.alloc(bp) for _ in range(nP)], ad=c.alloc(max(1, b - 1))))
    for d in range(n + m - 1):
        for k in range(max(0, d - m + 1), min(n, d + 1)):
            r = d - k; S = slot[0 if serial else r]; Cc = C[k][r]
            if k == 0 and r == 0:
                c.block("load_p"); load(k, r, [Cc[:bp]] + S["P"][1:], final=True)
                continue
            c.block("load_p"); st = c.mark(); load(k, r, S["P"]); en = c.mark()
            c.block("dp")
            if k == 0 or r == 0:
                prev = C[k][r - 1] if k == 0 else C[k - 1][r]
                for t in range(b): c.cx(prev[t], Cc[t])
            else:
                A, B = C[k - 1][r], C[k][r - 1]
                for t in range(b): c.cx(A[t], S["A"][t]); c.cx(B[t], S["B"][t])
                cmp_lt(c, S["A"], S["B"], F[k][r], S["ch"])
                select_max(c, S["A"], S["B"], F[k][r], Cc, S["fc"])
                for t in range(b): c.cx(A[t], S["A"][t]); c.cx(B[t], S["B"][t])
            add_inplace(c, S["P"][0], Cc, S["ad"])
            c.block("load_p"); c.emit_inverse(st, en)
    return C, F, slot


# ----------------------------------------------------------------------------------------------- oracles
def build_single(p, base, thr, b=None, serial=False):
    p = np.asarray(p, dtype=np.int64); n, m = p.shape; base = [int(v) for v in base]
    w = max(1, int(math.ceil(math.log2(n)))); b = b or width_v2(p); bp = width_p(p)
    c = Circ()
    I = c.alloc(w, "i"); J = c.alloc(w, "j")
    c.block("unary"); ei, gi = onehot(c, I, n); ej, gj = onehot(c, J, n)
    c.block("flags"); L, R, dirq = move_flags(c, I, J, ei, ej, n, w); E = ej
    valid = validity(c, [(I, J, ei, ej)], w)
    c.block("qrom"); Q = [c.alloc(bp) for r in range(m)]
    for q in range(n):
        for r in range(m):
            for t in bits(p[base[q], r], bp): c.cx(ei[q], Q[r][t])

    def load(k, r, Ts, final=False):
        T = Ts[0]; x = int(p[base[k], r])
        for t in range(bp): c.and_(E[k], Q[r][t], T[t])
        for t in bits(x, bp): c.x(T[t])
        if k in L:
            for t in bits(x ^ int(p[base[k + 1], r]), bp): c.cx(L[k], T[t])
        if k in R:
            for t in bits(x ^ int(p[base[k - 1], r]), bp): c.cx(R[k], T[t])
        for t in bits(x, bp): c.cx(E[k], T[t])

    C, F, slots = _dp(c, n, m, b, bp, load, serial=serial)
    c.block("threshold")
    lt = c.one(); lt_const(c, C[n - 1][m - 1], thr, lt, c.alloc(max(1, b - 1)))
    mk = c.one("mark"); c.and_(valid, lt, mk)
    return dict(circ=c, S=I + J, I=I, J=J, C=C, F=F, slots=slots, cmax=C[n - 1][m - 1], valid=valid, lt=lt,
                mark=mk, n=n, m=m, w=w, b=b, bp=bp, kind="single")


def build_double(p, base, thr, b=None):
    p = np.asarray(p, dtype=np.int64); n, m = p.shape; base = [int(v) for v in base]
    w = max(1, int(math.ceil(math.log2(n)))); b = b or width_v2(p); bp = width_p(p)
    c = Circ()
    I = c.alloc(w, "i"); J = c.alloc(w, "j"); K = c.alloc(w, "k"); Lr = c.alloc(w, "l")
    c.block("unary")
    ei, _ = onehot(c, I, n); ej, _ = onehot(c, J, n); ek, _ = onehot(c, K, n); el, _ = onehot(c, Lr, n)
    c.block("flags")
    L1, R1, _ = move_flags(c, I, J, ei, ej, n, w); E1 = ej
    L2, R2, _ = move_flags(c, K, Lr, ek, el, n, w); E2 = el

    def stay(Lf, Rf, Ef):
        s = []
        for q in range(n):
            v = c.one()
            if q in Lf: c.cx(Lf[q], v)
            if q in Rf: c.cx(Rf[q], v)
            c.cx(Ef[q], v); c.x(v); s.append(v)
        return s
    s1 = stay(L1, R1, E1); s2 = stay(L2, R2, E2)
    valid = validity(c, [(I, J, ei, ej), (K, Lr, ek, el)], w)
    c.block("qrom")
    Q1 = [c.alloc(bp) for r in range(m)]
    for q in range(n):
        for r in range(m):
            for t in bits(p[base[q], r], bp): c.cx(ei[q], Q1[r][t])
    # Q2 = p(tau_k, .): tau_k = pi_k (s1_k) | pi_{k+1} (L1_k) | pi_{k-1} (R1_k) | pi_i (E1_k)
    Xq = {q: c.one() for q in L1}; Yq = {q: c.one() for q in R1}; Zq = {q: c.one() for q in range(n)}
    for q in L1: c.and_(ek[q], L1[q], Xq[q])
    for q in R1: c.and_(ek[q], R1[q], Yq[q])
    for q in range(n): c.and_(ek[q], E1[q], Zq[q])
    Sq = {}
    for q in range(n):
        v = c.one(); c.cx(ek[q], v)
        if q in Xq: c.cx(Xq[q], v)
        if q in Yq: c.cx(Yq[q], v)
        c.cx(Zq[q], v); Sq[q] = v
    eqkj = parity_flag(c, [Zq[q] for q in range(n)])
    Q2 = [c.alloc(bp) for r in range(m)]
    for r in range(m):
        for t in range(bp): c.and_(eqkj, Q1[r][t], Q2[r][t])
    for q in range(n):
        for r in range(m):
            for t in bits(p[base[q], r], bp): c.cx(Sq[q], Q2[r][t])
            if q in Xq:
                for t in bits(p[base[q + 1], r], bp): c.cx(Xq[q], Q2[r][t])
            if q in Yq:
                for t in bits(p[base[q - 1], r], bp): c.cx(Yq[q], Q2[r][t])
    c.block("flags")
    prods = {q: [] for q in range(n)}; G = {}
    for q in range(n):
        cand = [(s2[q], s1[q], q), (s2[q], L1.get(q), q + 1), (s2[q], R1.get(q), q - 1), (s2[q], E1[q], "G")]
        if q in L2:
            cand += [(L2[q], s1[q + 1], q + 1), (L2[q], L1.get(q + 1), q + 2), (L2[q], R1.get(q + 1), q),
                     (L2[q], E1[q + 1], "G")]
        if q in R2:
            cand += [(R2[q], s1[q - 1], q - 1), (R2[q], L1.get(q - 1), q), (R2[q], R1.get(q - 1), q - 2),
                     (R2[q], E1[q - 1], "G")]
        gq = []
        for (a, f2, tgt) in cand:
            if f2 is None: continue
            v = c.one(); c.and_(a, f2, v)
            (gq if tgt == "G" else prods[q]).append((v, tgt) if tgt != "G" else v)
        G[q] = parity_flag(c, gq)

    def load(k, r, Ts, final=False):
        T1, T2 = Ts
        for t in range(bp): c.and_(G[k], Q1[r][t], T1[t])
        for t in range(bp): c.and_(E2[k], Q2[r][t], T2[t])
        for t in range(bp): c.cx(T2[t], T1[t])
        if final:                                        # corner cell: T1 is kept, clean the slot register T2
            for t in range(bp): c.unand(E2[k], Q2[r][t], T2[t])
        for (v, tgt) in prods[k]:
            for t in bits(p[base[tgt], r], bp): c.cx(v, T1[t])

    C, F, slots = _dp(c, n, m, b, bp, load, nP=2)
    c.block("threshold")
    lt = c.one(); lt_const(c, C[n - 1][m - 1], thr, lt, c.alloc(max(1, b - 1)))
    mk = c.one("mark"); c.and_(valid, lt, mk)
    return dict(circ=c, S=I + J + K + Lr, I=I, J=J, K=K, L=Lr, C=C, F=F, slots=slots, cmax=C[n - 1][m - 1],
                valid=valid, lt=lt, mark=mk, n=n, m=m, w=w, b=b, bp=bp, kind="double")


# ----------------------------------------------------------------------------------------------- Grover iteration
def diffusion_ops(S, tmp_start):
    """H X MCZ X H on the move register; MCZ by an AND chain (q-2 ANDs, measured).  Ancillas numbered from
    tmp_start (q-2 of them).  Returns (ops, n_ancilla)."""
    c = Circ(); c.nq = tmp_start; q = len(S)
    for s in S: c.h(s)
    for s in S: c.x(s)
    if q == 1:
        c.z(S[0])
    elif q == 2:
        c.cz(S[0], S[1])
    else:
        tmp = c.alloc(q - 2); prev = S[0]; chain = []
        for t in range(1, q - 1):
            c.and_(prev, S[t], tmp[t - 1]); chain.append((prev, S[t], tmp[t - 1])); prev = tmp[t - 1]
        c.cz(prev, S[q - 1])
        for g in reversed(chain): c.unand(*g)
    for s in S: c.x(s)
    for s in S: c.h(s)
    return c.ops, max(0, q - 2)


def iteration_ops(B):
    """Ops of one Grover iteration: oracle, Z(mark), inverse oracle, diffusion.  Returns (ops, nq)."""
    c = B["circ"]; D, na = diffusion_ops(B["S"], c.nq)
    ops = list(c.ops) + [(Z, B["mark"], -1, -1)] + inverse_ops(c.ops) + D
    return ops, c.nq + na


# ----------------------------------------------------------------------------------------------- classical refs
def shift(perm, i, j):
    x = list(perm); v = x.pop(i); x.insert(j, v); return x


def dp_table(p, seq):
    n = len(seq); m = p.shape[1]; C = np.zeros((n, m), dtype=np.int64)
    for k in range(n):
        for r in range(m):
            a = C[k - 1][r] if k else 0; bb = C[k][r - 1] if r else 0
            C[k][r] = max(a, bb) + p[seq[k], r]
    return C
