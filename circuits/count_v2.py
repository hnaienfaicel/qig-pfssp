"""Gate counts, depths and Toffoli depths of the oracle v1 (paper), v2 and v2-double on the 8 Taillard instances of
Table 1, for one oracle call and for one Grover iteration (oracle, Z on the mark qubit, inverse oracle, diffusion).

Counting conventions (all counts are of the recorded gate lists, nothing is estimated):
  toffoli_like = ccx + cswap + and       (one CCZ magic state each; an 'and' is a Gidney temporary logical-AND)
  t_count      = 7 (ccx + cswap) + 4 and (Toffoli/Fredkin with 7 T, Gidney AND with 4 T; measurement-based
                 uncomputation 'unand' = X-basis measurement + classically controlled CZ, no T)
  cx_equiv     = cx + 6 ccx + 8 cswap (paper's convention for v1) ; for v2: cx + 6 and + unand (one CZ fix-up per
                 measured AND, worst case), i.e. an AND is charged like a Toffoli in CNOTs
  depth        = ASAP layers, every gate one layer (Toffoli, Fredkin, AND and the measurement-based uncomputation
                 each one layer), as in results/oracle_depth.csv
  toffoli_depth= ASAP layers with Clifford gates and measurements of zero duration, Toffoli-like gates one layer
  peak_parallel_toffoli = max number of Toffoli-like gates in one layer of the Toffoli-depth schedule
Variants: v1 (paper, thr = sum p as in circuits/resources.py), v2 and v2-double (base = identity, thr = Cmax of the
base), plus two ablations of v2 on the single-insertion oracle: v2-serialDP (one shared set of DP temporaries, i.e.
no wavefront) and v2-bv1 (bit width of v1).  Usage: python circuits/count_v2.py [nproc] -> results/oracle_v2_resources.csv,
results/oracle_v2_breakdown.csv"""
import math, os, sys, time
from multiprocessing import Pool
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import oracle_v2 as V
import profile_v1 as P1

R = os.path.join(os.path.dirname(HERE), "results")


def build(name, variant):
    p = P1.taillard(name); n, m = p.shape; base = list(range(n))
    thr = int(V.dp_table(p, base)[-1, -1])
    if variant == "v1":
        B = P1.build_v1(p, int(p.sum())); it, nq_it = P1.iteration_ops_v1(B)
    else:
        if variant == "v2":
            B = V.build_single(p, base, thr)
        elif variant == "v2-double":
            B = V.build_double(p, base, thr)
        elif variant == "v2-serialDP":
            B = V.build_single(p, base, thr, serial=True)
        elif variant == "v2-bv1":
            B = V.build_single(p, base, thr, b=V.width_v1(p))
        it, nq_it = V.iteration_ops(B)
    return p, B, it, nq_it


def row(args):
    name, variant = args; t0 = time.time()
    p, B, it, nq_it = build(name, variant); c = B["circ"]; n, m = p.shape; w = B["w"]
    q = len(B["S"]); N = 1 << q; Nv = n * (n - 1) if q == 2 * w else (n * (n - 1)) ** 2
    co, per = V.counts(c.ops, c.blk, c.block_names); ci = V.counts(it)
    inv = V.counts(V.inverse_ops(c.ops))
    tl = lambda d: d["ccx"] + d["cswap"] + d["and"]
    tc = lambda d: 7 * (d["ccx"] + d["cswap"]) + 4 * d["and"]
    cxe = lambda d: d["cx"] + 6 * d["ccx"] + 8 * d["cswap"] + 6 * d["and"] + d["unand"]
    dO = V.schedule(c.ops, c.nq, V.DUR_ALL); dOT = V.schedule(c.ops, c.nq, V.DUR_TOF)
    dI = V.schedule(it, nq_it, V.DUR_ALL); dIT, layers = V.schedule(it, nq_it, V.DUR_TOF, want_layers=True)
    r = dict(instance=name, n=n, m=m, variant=variant, w=w, move_qubits=q, codes_N=N, valid_codes=Nv,
             sqrtN=math.sqrt(N), b=B["b"], bp=B.get("bp", np.nan),
             qubits_oracle=c.nq, qubits_iteration=nq_it,
             oracle_x=co["x"], oracle_cx=co["cx"], oracle_ccx=co["ccx"], oracle_cswap=co["cswap"],
             oracle_and=co["and"], oracle_unand=co["unand"], oracle_toffoli=tl(co), oracle_t_count=tc(co),
             oracle_cx_equiv=cxe(co), oracle_depth=dO, oracle_toffoli_depth=dOT,
             inverse_toffoli=tl(inv), inverse_unand=inv["unand"],
             iter_cx=ci["cx"], iter_cz=ci["cz"], iter_ccx=ci["ccx"], iter_cswap=ci["cswap"], iter_and=ci["and"],
             iter_unand=ci["unand"], iter_toffoli=tl(ci), iter_t_count=tc(ci), iter_cx_equiv=cxe(ci),
             iter_depth=dI, iter_toffoli_depth=dIT, iter_peak_parallel_toffoli=max(layers.values()),
             iter_mean_parallel_toffoli=tl(ci) / dIT, build_s=round(time.time() - t0, 1))
    blocks = []
    for nm, d in per.items():
        ops = [g for g, bl in zip(c.ops, c.blk) if c.block_names[bl] == nm]
        blocks.append(dict(instance=name, n=n, m=m, variant=variant, block=nm, x=d["x"], cx=d["cx"], ccx=d["ccx"],
                           cswap=d["cswap"], ands=d["and"], unand=d["unand"], toffoli_like=tl(d),
                           depth_standalone=V.schedule(ops, c.nq, V.DUR_ALL),
                           toffoli_depth_standalone=V.schedule(ops, c.nq, V.DUR_TOF)))
    print(name, variant, c.nq, tl(co), dO, dIT, f"{time.time() - t0:.0f}s", flush=True)
    return r, blocks


if __name__ == "__main__":
    import pandas as pd
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    jobs = [(nm, v) for nm in P1.TABLE1 for v in ("v1", "v2", "v2-double", "v2-serialDP", "v2-bv1")]
    jobs.sort(key=lambda j: -(P1.taillard(j[0]).size * (3 if j[1] == "v1" else 1)))
    with Pool(nproc) as pool:
        res = pool.map(row, jobs, chunksize=1)
    rows = [r for r, _ in res]; blocks = [b for _, bl in res for b in bl]
    D = pd.DataFrame(rows)
    order = {nm: k for k, nm in enumerate(P1.TABLE1)}; vord = {"v1": 0, "v2": 1, "v2-double": 2, "v2-serialDP": 3, "v2-bv1": 4}
    D = D.sort_values(["instance", "variant"], key=lambda s: s.map(order) if s.name == "instance" else s.map(vord))
    ref = pd.read_csv(os.path.join(R, "oracle_resources_taillard.csv")).set_index("instance")
    refd = pd.read_csv(os.path.join(R, "oracle_depth.csv")).set_index("instance")
    for r in D[D.variant == "v1"].itertuples():                       # v1 must reproduce the paper's tables
        assert (r.qubits_oracle, r.oracle_cx, r.oracle_ccx, r.oracle_cswap) == tuple(int(ref.loc[r.instance, k]) for k in ("qubits", "cx", "ccx", "cswap"))
        assert r.oracle_depth == refd.loc[r.instance, "depth_oracle"] and r.iter_depth == refd.loc[r.instance, "depth_iteration"], r.instance
    D.to_csv(os.path.join(R, "oracle_v2_counts_raw.csv"), index=False)
    Bk = pd.DataFrame(blocks); Bk.to_csv(os.path.join(R, "oracle_v2_breakdown.csv"), index=False)
    print(D[["instance", "variant", "qubits_oracle", "oracle_toffoli", "oracle_depth", "iter_toffoli", "iter_depth", "iter_toffoli_depth"]].to_string(index=False))
