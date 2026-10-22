"""Exhaustive basis-state validation of oracle v2 (single and double insertion).

For every code of the move register (2w or 4w qubits) the reversible circuit is simulated exactly on the
computational basis (all codes at once, vectorised).  Checked for every code:
  * every DP register C[k][r] equals the classical DP table of the child sequence (valid codes);
  * the mark qubit equals valid AND [Cmax(child) < thr]; invalid codes (i>=n, j>=n, i==j, same for k,l) unmarked;
  * the move register is unchanged; all per-column temporaries (copies, carry chains, P) are |0> after the oracle;
  * every AND writes a |0> target and every measurement-based uncomputation sees target == a AND b (asserted
    inside simulate);
  * oracle followed by its inverse returns every non-move qubit to |0>.
Run: python tests/test_oracle_v2.py  (or pytest tests/test_oracle_v2.py)"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "circuits"))
import oracle_v2 as V


def taillard(name):
    with open(os.path.join(ROOT, "data", "taillard", f"{name}.fsp")) as f:
        lines = [l.strip() for l in f if l.strip()]
    n, m = (int(v) for v in lines[1].split()[:2])
    k = next(i for i, l in enumerate(lines) if l.lower().startswith("processing times")) + 1
    return np.asarray([[int(v) for v in l.split()] for l in lines[k:k + m]], dtype=np.int64).T


def children(base, n, w, double):
    N = 1 << ((4 if double else 2) * w); codes = np.arange(N); mask = (1 << w) - 1
    mv = [(codes >> (t * w)) & mask for t in range(4 if double else 2)]
    valid = (mv[0] < n) & (mv[1] < n) & (mv[0] != mv[1])
    if double:
        valid &= (mv[2] < n) & (mv[3] < n) & (mv[2] != mv[3])
    kids = {}
    for c in np.flatnonzero(valid):
        x = V.shift(base, int(mv[0][c]), int(mv[1][c]))
        if double:
            x = V.shift(x, int(mv[2][c]), int(mv[3][c]))
        kids[int(c)] = x
    return codes, valid, kids


def check(p, base, thrs, double=False):
    """Returns a list of dict rows (one per threshold) with the number of mismatches (must be 0)."""
    n, m = p.shape; w = max(1, int(np.ceil(np.log2(n))))
    codes, valid, kids = children(list(base), n, w, double)
    tabs = {c: V.dp_table(p, x) for c, x in kids.items()}
    rows = []
    for thr in thrs:
        B = (V.build_double if double else V.build_single)(p, base, thr); c = B["circ"]; N = codes.size
        init = {q: ((codes >> t) & 1).astype(bool) for t, q in enumerate(B["S"])}
        S = V.simulate(c.ops, c.nq, init, N)
        bad = 0
        bad += sum(int(not np.array_equal(S[q], init[q])) for q in B["S"])                  # move register intact
        for sl in B["slots"]:                                                                # temporaries clean
            for key, reg in sl.items():
                regs = reg if key == "P" else [reg]
                bad += sum(int(S[q].any()) for rr in regs for q in rr)
        bad += int(not np.array_equal(S[B["valid"]], valid))
        vc = np.flatnonzero(valid)
        Ct = np.stack([tabs[int(cc)] for cc in vc])                                          # (V, n, m)
        for k in range(n):
            for r in range(m):
                bad += int(np.sum(V.read_reg(S, B["C"][k][r])[vc] != Ct[:, k, r]))
        cmax = Ct[:, n - 1, m - 1]
        mark_c = np.zeros(N, dtype=bool); mark_c[vc] = cmax < thr
        bad += int(np.sum(S[B["mark"]] != mark_c))
        S2 = V.simulate(c.ops + V.inverse_ops(c.ops), c.nq, init, N)                         # O^-1 O = identity
        keep = set(B["S"])
        bad += int(sum(S2[q].any() for q in range(c.nq) if q not in keep))
        rows.append(dict(variant="v2-double" if double else "v2", n=n, m=m, base=" ".join(map(str, base)), thr=int(thr),
                         codes=int(N), valid_codes=int(valid.sum()), marked=int(mark_c.sum()), qubits=c.nq,
                         ands=sum(1 for g in c.ops if g[0] == V.AND), mismatches=bad))
    return rows


def thresholds(p, base, kids):
    cm = sorted(int(V.dp_table(p, x)[-1, -1]) for x in kids.values())
    c0 = int(V.dp_table(p, list(base))[-1, -1])
    return sorted({c0, c0 + 1, cm[0], cm[0] + 1, cm[len(cm) // 2], cm[-1] + 1, 0})


def cases_single():
    P = taillard("tai20_5_0"); rng = np.random.default_rng(2026); out = []
    out.append((P[:4, :3].copy(), (3, 1, 2, 0)))                                             # paper's 4x3 example
    for n in (4, 5, 6):
        for m in (2, 3, 4, 5):
            out.append((P[:n, :m].copy(), tuple(int(v) for v in rng.permutation(n))))
            out.append((rng.integers(1, 100, size=(n, m)), tuple(int(v) for v in rng.permutation(n))))
    return out


def cases_double():
    P = taillard("tai20_5_0"); rng = np.random.default_rng(7919); out = []
    for n in (4, 5):
        for m in (2, 3, 4):
            out.append((P[:n, :m].copy(), tuple(int(v) for v in rng.permutation(n))))
            out.append((rng.integers(1, 100, size=(n, m)), tuple(int(v) for v in rng.permutation(n))))
    return out


def run(double):
    rows = []
    for p, base in (cases_double() if double else cases_single()):
        n = p.shape[0]; w = max(1, int(np.ceil(np.log2(n))))
        _, _, kids = children(list(base), n, w, double)
        rows += check(p, base, thresholds(p, base, kids), double)
    return rows


def test_single():
    assert all(r["mismatches"] == 0 for r in run(False))


def test_double():
    assert all(r["mismatches"] == 0 for r in run(True))


if __name__ == "__main__":
    import pandas as pd, time
    t = time.time(); R = run(False) + run(True)
    D = pd.DataFrame(R); out = os.path.join(ROOT, "results", "oracle_v2_validation.csv"); D.to_csv(out, index=False)
    print(D.groupby(["variant", "n", "m"]).agg(cases=("thr", "size"), codes=("codes", "max"), valid=("valid_codes", "max"),
                                                 marked=("marked", "sum"), mismatches=("mismatches", "sum")).to_string())
    print("total mismatches", int(D.mismatches.sum()), "rows", len(D), f"{time.time() - t:.0f}s")
