"""Exploratory studies declared in Amendment 4 (not confirmatory). Usage: python experiments/run_extra.py PHASE SHARD NSHARDS OUTDIR
PHASE: noise (Q-IG-min and Q-IG-G with a noisy Grover, eps in EPS, TEST and HARD, 30 runs)
       trace (all six arms, TEST and HARD, 30 runs: best makespan after NEH and at fixed call checkpoints)."""
import os, sys, json, time, random, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qig.instances import load, SETS
from qig.ig import iterated_greedy
HERE = os.path.dirname(os.path.abspath(__file__)); BUDGET = 46080
EPS = (0.001, 0.01, 0.03, 0.1, 0.3)
CHECK = [int(v) for v in np.unique(np.round(np.geomspace(1000, BUDGET, 30)))]
ARMS = json.load(open(os.path.join(HERE, "arms.json")))

def kw(a): return {k: a[k] for k in ("cq_min", "rmin", "ls", "cq_ls", "cpi") if k in a}

def task_noise(a, e, inst, r):
    t0 = time.perf_counter(); p, ub = load(inst)
    c, used, it = iterated_greedy(p, BUDGET, 1000 + 7919 * r, eps=e, **kw(a))
    return dict(Arm=f"{a['nom']} eps{e}", Base=a["nom"], Eps=e, Instance=inst, Run=r, UB=ub, Cmax=c, Gap=100.0 * (c - ub) / ub,
                Evals=used, Iter=it, sec=time.perf_counter() - t0)

def task_trace(a, inst, r):
    t0 = time.perf_counter(); p, ub = load(inst); tr = []
    c, used, it = iterated_greedy(p, BUDGET, 1000 + 7919 * r, trace=tr, **kw(a))
    calls = np.array([t[0] for t in tr]); vals = np.array([t[1] for t in tr])
    row = dict(Arm=a["nom"], Instance=inst, Run=r, UB=ub, Cmax=c, Evals=used, Iter=it, neh_calls=int(calls[0]), neh_c=int(vals[0]))
    for t in CHECK:
        k = np.searchsorted(calls, t, side="right") - 1
        row[f"c{t}"] = int(vals[k]) if k >= 0 else np.nan
    row["sec"] = time.perf_counter() - t0
    return row

if __name__ == "__main__":
    phase, sh, ns, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; os.makedirs(out, exist_ok=True)
    insts = SETS["TEST"] + SETS["HARD"]
    if phase == "noise":
        T = [(a, e, i, r) for r in range(30) for i in insts for e in EPS for a in ARMS if a["nom"] in ("Q-IG-min", "Q-IG-G")]; f = task_noise
        key = lambda t: (f"{t[0]['nom']} eps{t[1]}", t[2], t[3])
    else:
        T = [(a, i, r) for r in range(30) for i in insts for a in ARMS]; f = task_trace
        key = lambda t: (t[0]["nom"], t[1], t[2])
    random.Random(41).shuffle(T)
    fn = os.path.join(out, f"{phase}_{sh}.csv"); rows = pd.read_csv(fn).to_dict("records") if os.path.exists(fn) else []
    done = {(x["Arm"], x["Instance"], int(x["Run"])) for x in rows}
    for k, t in enumerate([t for t in T[sh::ns] if key(t) not in done]):
        rows.append(f(*t))
        if (k + 1) % 5 == 0: pd.DataFrame(rows).to_csv(fn, index=False)
    pd.DataFrame(rows).to_csv(fn, index=False); open(fn + ".done", "w").write("ok")
