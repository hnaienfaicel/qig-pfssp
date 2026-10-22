"""Closed-form Toffoli-like (AND) counts of oracle v2 (single insertion), derived by hand from the construction,
used to cross-check the recorded gate lists:
  forward = 2 S(n,w) + (2w + n) + 1 + n m bp + (n-1)(m-1)(3b-1) + (n+m-2)(b-1) + T(b,thr) + 1
            [unary iteration for i, j] [dir: w, L flags: n-1, [i==j]: w-1, valid: 2] [E-term loads]
            [interior cells: comparator b + select b + adder b-1] [border cells: adder] [threshold chain] [mark]
  inverse = (w-1) + (w-2) + 1 + (n-1)(m-1)(2b-2+bp) + (n+m-2)(b-1+bp) + T(b,thr)
            (re-computation of every AND that the forward pass uncomputed by measurement)
  iteration = forward + inverse + (2w-2) (diffusion multi-controlled Z on 2w qubits)
S(n,w) = number of prefix splits of the unary iteration; T(b,thr) = carry-chain ANDs of the constant comparator."""
def splits(n, w):
    cnt = 0; nodes = [0, 1 << (w - 1)]
    for t in range(w - 2, -1, -1):
        new = []
        for lo in nodes:
            if lo >= n: continue
            cnt += 1; new += [lo, lo + (1 << t)]
        nodes = new
    return cnt


def thr_ands(b, thr):
    if thr <= 0 or thr >= (1 << b): return 0
    k = (1 << b) - thr; t0 = (k & -k).bit_length() - 1
    return b - 1 - t0


def analytic_single(n, m, b, bp, w, thr):
    T = thr_ands(b, thr)
    fwd = 2 * splits(n, w) + (w + (n - 1) + (w - 1) + 2) + n * m * bp + (n - 1) * (m - 1) * (3 * b - 1) + (n + m - 2) * (b - 1) + T + 1
    inv = ((w - 1) + max(0, w - 2) + 1) + (n - 1) * (m - 1) * (2 * b - 2 + bp) + (n + m - 2) * (b - 1 + bp) + T
    return fwd, inv, fwd + inv + (2 * w - 2)


def test_analytic_matches_recorder():
    import os, sys
    import numpy as np
    here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(here), "circuits"))
    sys.path.insert(0, here)
    import oracle_v2 as V
    import test_oracle_v2 as T
    for name in ("tai20_5_0", "tai20_20_0", "tai50_5_0"):
        p = T.taillard(name); n, m = p.shape; base = list(range(n)); thr = int(V.dp_table(p, base)[-1, -1])
        B = V.build_single(p, base, thr); c = B["circ"]; it, _ = V.iteration_ops(B)
        tl = lambda ops: sum(1 for g in ops if g[0] in (V.AND, V.CCX, V.CSWAP))
        assert analytic_single(n, m, B["b"], B["bp"], B["w"], thr) == (tl(c.ops), tl(V.inverse_ops(c.ops)), tl(it)), name


if __name__ == "__main__":
    test_analytic_matches_recorder(); print("analytic counts == recorded counts")
