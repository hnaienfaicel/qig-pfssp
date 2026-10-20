"""Exploratory record attempt: Q-IG-G and IG at budget x1000 (46,080,000 oracle calls) on the open instances that came
closest to the best-known solution at x100. Not preregistered. Usage: python experiments/run_record.py OUTDIR NWORKERS"""
import os, sys, json, time, multiprocessing as mp, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qig.instances import load
from qig.ig import iterated_greedy
from qig.flowshop import makespan
HERE = os.path.dirname(os.path.abspath(__file__))
INSTANCES = ["VFR40_20_4", "VFR40_20_5", "VFR40_20_1", "tai50_20_9", "tai50_20_8", "VFR40_20_2"]
MULT, RUNS, BUDGET = 1000, 8, 46080

def task(t):
    arm, inst, r, out = t
    fn = os.path.join(out, f"{arm['nom']}_{inst}_{r}.json")
    if os.path.exists(fn): return fn
    p, _ = load(inst); t0 = time.perf_counter()
    kw = {k: arm[k] for k in ("cq_min", "rmin", "ls", "cq_ls", "cpi") if k in arm}
    c, used, it, perm = iterated_greedy(p, BUDGET * MULT, 1000 + 7919 * r, return_perm=True, **kw)
    assert int(makespan(p, perm)) == c
    json.dump(dict(Arm=arm["nom"], Instance=inst, Run=r, Mult=MULT, Cmax=c, Evals=int(used), Iter=int(it),
                   sec=time.perf_counter() - t0, perm=" ".join(str(int(j) + 1) for j in perm)), open(fn, "w"))
    return fn

if __name__ == "__main__":
    out, nw = sys.argv[1], int(sys.argv[2]); os.makedirs(out, exist_ok=True)
    arms = [a for a in json.load(open(os.path.join(HERE, "arms.json"))) if a["nom"] in ("Q-IG-G", "IG")]
    arms.sort(key=lambda a: a["nom"] != "Q-IG-G")          # long Q-IG-G runs first
    T = [(a, i, r, out) for a in arms for r in range(RUNS) for i in INSTANCES]
    with mp.Pool(nw) as pool:
        for k, fn in enumerate(pool.imap_unordered(task, T)): print(k + 1, os.path.basename(fn), flush=True)
