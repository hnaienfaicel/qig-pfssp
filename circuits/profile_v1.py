"""Gate-for-gate replica of the paper's Grover oracle (circuits/grover_circuit.build, which composes
oracle_makespan.build(p, "onehot")), recorded with block labels so that counts and depths can be split by block:
  perm_network   coherent insertion: 2(n-1) conditional swaps of w-bit words (comparisons of i, j with constants by
                 multi-controlled X in V-chain, Fredkin swaps)
  onehot_decode  n^2 multi-controlled X from the permutation words to the one-hot table
  load_p         CNOT loading of p(pi_k, i) into the shared register P (twice per cell)
  dp             makespan DP: Cuccaro comparator (computed and uncomputed), Toffoli selection into T, Cuccaro adder
  threshold      comparison of Cmax with the threshold (geq_const) and X on the mark qubit
The same primitives of oracle_makespan.py are called (maj, uma, add_inplace, less_than, geq_const), so the gate
sequence is identical to the Qiskit circuit; totals are checked against results/oracle_resources_taillard.csv and
results/oracle_depth.csv.  Usage: python circuits/profile_v1.py -> results/oracle_v1_breakdown.csv"""
import math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import oracle_makespan as OM
import oracle_v2 as V
from oracle_v2 import Circ


def mcx_v(c, ctrls, target, aux, ctrl_state=None):
    k = len(ctrls)
    flips = [ctrls[t] for t in range(k) if ctrl_state is not None and not (ctrl_state >> t) & 1]
    for q in flips: c.x(q)
    if k == 1:
        c.cx(ctrls[0], target)
    elif k == 2:
        c.ccx(ctrls[0], ctrls[1], target)
    else:
        chain = [(ctrls[0], ctrls[1], aux[0])] + [(ctrls[t], aux[t - 2], aux[t - 1]) for t in range(2, k - 1)]
        for g in chain: c.ccx(*g)
        c.ccx(ctrls[k - 1], aux[k - 3], target)
        for g in reversed(chain): c.ccx(*g)
    for q in flips: c.x(q)


def build_v1(p, thr):
    p = np.asarray(p, dtype=np.int64); n, m = p.shape
    w = max(1, math.ceil(math.log2(n))); b = int(math.ceil(math.log2(int(p.sum()) + 1)))
    c = Circ()
    I = c.alloc(w, "i"); J = c.alloc(w, "j"); c1 = c.one(); c2 = c.one(); f = c.one()
    aux = c.alloc(max(1, 2 * w - 3)); Pm = c.alloc(n * w, "perm")
    Xo = [c.alloc(n) for _ in range(n)]                                  # Xo[k][j] = [job j at position k]
    C = [[c.alloc(b) for _ in range(m)] for _ in range(n)]
    P = c.alloc(b); T = c.alloc(b); anc = c.alloc(2); c0, fo = anc
    tmp = c.alloc(b); mark = c.one("mark")

    def lt_k(reg, K, tgt):
        for v in range(min(K, 1 << w)): mcx_v(c, reg, tgt, aux, ctrl_state=v)

    def cond_swap(k, lo, hi):
        lt_k(lo, k + 1, c1)
        lt_k(hi, k + 1, c2); c.x(c2)
        c.ccx(c1, c2, f)
        for t in range(w): c.cswap(f, Pm[k * w + t], Pm[(k + 1) * w + t])
        c.ccx(c1, c2, f)
        c.x(c2); lt_k(hi, k + 1, c2); lt_k(lo, k + 1, c1)

    c.block("perm_network")
    for k in range(n - 1): cond_swap(k, I, J)
    for k in reversed(range(n - 1)): cond_swap(k, J, I)
    c.block("onehot_decode")
    for k in range(n):
        for jj in range(n): mcx_v(c, [Pm[k * w + t] for t in range(w)], Xo[k][jj], aux, ctrl_state=jj)

    def load_p(k, i):
        c.block("load_p")
        for j in range(n):
            for t, bt in enumerate(OM.bits_of(p[j, i], b)):
                if bt: c.cx(Xo[k][j], P[t])
        c.block("dp")

    for k in range(n):
        for i in range(m):
            load_p(k, i)
            if k == 0 and i == 0:
                for t in range(b): c.cx(P[t], C[0][0][t])
            elif k == 0 or i == 0:
                prev = C[k][i - 1] if k == 0 else C[k - 1][i]
                for t in range(b): c.cx(prev[t], C[k][i][t])
                OM.add_inplace(c, P, C[k][i], c0)
            else:
                A, Bq = C[k - 1][i], C[k][i - 1]
                OM.less_than(c, A, Bq, c0, fo)
                c.x(fo)
                for t in range(b): c.ccx(fo, A[t], T[t])
                c.x(fo)
                for t in range(b): c.ccx(fo, Bq[t], T[t])
                for t in range(b): c.cx(T[t], C[k][i][t])
                OM.add_inplace(c, P, C[k][i], c0)
                for t in range(b): c.ccx(fo, Bq[t], T[t])
                c.x(fo)
                for t in range(b): c.ccx(fo, A[t], T[t])
                c.x(fo)
                OM.less_than(c, A, Bq, c0, fo)
            load_p(k, i)
    c.block("threshold")
    OM.geq_const(c, C[n - 1][m - 1], thr, tmp, c0, mark); c.x(mark)
    return dict(circ=c, S=I + J, I=I, J=J, Pm=Pm, cmax=C[n - 1][m - 1], mark=mark, aux=aux, n=n, m=m, w=w, b=b,
                kind="v1")


def diffusion_v1(B):
    """Diffusion of grover_circuit.build: H X H(last) mcx_v(S[:-1] -> last) H(last) X H."""
    c = Circ(); c.nq = B["circ"].nq; S = B["S"]
    for s in S: c.h(s)
    for s in S: c.x(s)
    c.h(S[-1]); mcx_v(c, S[:-1], S[-1], B["aux"]); c.h(S[-1])
    for s in S: c.x(s)
    for s in S: c.h(s)
    return c.ops


def iteration_ops_v1(B):
    c = B["circ"]
    return list(c.ops) + [(V.Z, B["mark"], -1, -1)] + V.inverse_ops(c.ops) + diffusion_v1(B), c.nq


TABLE1 = ("tai20_5_0", "tai20_20_0", "tai50_5_0", "tai50_10_0", "tai50_20_0", "tai100_5_0", "tai100_10_0",
          "tai100_20_0")


def taillard(name):
    root = os.path.dirname(HERE)
    with open(os.path.join(root, "data", "taillard", f"{name}.fsp")) as fh:
        lines = [l.strip() for l in fh if l.strip()]
    n, m = (int(v) for v in lines[1].split()[:2])
    k = next(i for i, l in enumerate(lines) if l.lower().startswith("processing times")) + 1
    return np.asarray([[int(v) for v in l.split()] for l in lines[k:k + m]], dtype=np.int64).T


def block_profile(B):
    """Per-block counts and standalone depths (all gates one layer; Toffoli depth with Clifford free)."""
    c = B["circ"]; tot, per = V.counts(c.ops, c.blk, c.block_names)
    rows = []
    for bi, nm in enumerate(c.block_names):
        ops = [g for g, bl in zip(c.ops, c.blk) if bl == bi]
        if not ops: continue
        cnt = per[nm]
        rows.append(dict(block=nm, x=cnt["x"], cx=cnt["cx"], ccx=cnt["ccx"], cswap=cnt["cswap"], ands=cnt["and"],
                         unand=cnt["unand"], toffoli_like=cnt["ccx"] + cnt["cswap"] + cnt["and"],
                         depth_standalone=V.schedule(ops, c.nq, V.DUR_ALL),
                         toffoli_depth_standalone=V.schedule(ops, c.nq, V.DUR_TOF)))
    return tot, rows


if __name__ == "__main__":
    import pandas as pd
    R = os.path.join(os.path.dirname(HERE), "results")
    ref = pd.read_csv(os.path.join(R, "oracle_resources_taillard.csv")).set_index("instance")
    refd = pd.read_csv(os.path.join(R, "oracle_depth.csv")).set_index("instance")
    out = []
    for name in TABLE1:
        p = taillard(name); n, m = p.shape
        B = build_v1(p, int(p.sum())); c = B["circ"]
        tot, rows = block_profile(B)
        dO = V.schedule(c.ops, c.nq, V.DUR_ALL)
        assert (c.nq, tot["cx"], tot["ccx"], tot["cswap"]) == tuple(int(ref.loc[name, k]) for k in ("qubits", "cx", "ccx", "cswap")), name
        assert dO == int(refd.loc[name, "depth_oracle"]), (name, dO)
        T = sum(r["toffoli_like"] for r in rows)
        for r in rows:
            out.append(dict(instance=name, n=n, m=m, b=B["b"], **r, toffoli_share=r["toffoli_like"] / T,
                            oracle_depth_total=dO))
        print(name, c.nq, tot["cx"], tot["ccx"], tot["cswap"], dO, flush=True)
    pd.DataFrame(out).to_csv(os.path.join(R, "oracle_v1_breakdown.csv"), index=False)
