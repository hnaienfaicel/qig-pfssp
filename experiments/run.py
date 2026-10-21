"""Run the preregistered Q-IG experiments. Usage: python experiments/run.py PHASE SHARD NSHARDS
PHASE: tune | tune2 | tune3 (TUNE+, 5 runs) | test | hard | ext (30 runs, arms.json) | test_cpi2 | hard_cpi2 (robustness)."""
import os, sys, json, time, random, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qig.instances import load, SETS
from qig.ig import iterated_greedy
BUDGET = 46080
HERE = os.path.dirname(os.path.abspath(__file__))
TUNE = [dict(nom="IG")] + [dict(nom=f"Q-IG-min c{c}", cq_min=c) for c in (1, 2, 3, 4)] + \
       [dict(nom=f"Q-IG-G c{a} l{b}", cq_min=a, ls="gls", cq_ls=b) for a in (2, 3) for b in (1, 3, 6)] + \
       [dict(nom=f"IG-R l{b}", ls="rls", cq_ls=b) for b in (1, 3, 6)]
TUNE2 = [dict(nom=f"Q-IG-min c{c}", cq_min=c) for c in (0.25, 0.5)] + \
        [dict(nom=f"Q-IG-G c{a} l{b}", cq_min=a, ls="gls", cq_ls=b) for a, b in ((1, 0.5), (1, 1), (1, 3), (2, 0.5))] + \
        [dict(nom=f"IG-R l{b}", ls="rls", cq_ls=b) for b in (10, 15)] + \
        [dict(nom=f"IG-Rmin r{c}", rmin=c) for c in (0.5, 1, 2, 3)] + \
        [dict(nom=f"IG-RR r{a} l{b}", rmin=a, ls="rls", cq_ls=b) for a in (0.5, 1, 2) for b in (3, 6, 10)]
TUNE3 = [dict(nom=f"IG-RR r{a} l{b}", rmin=a, ls="rls", cq_ls=b) for a, b in ((2, 15), (3, 6), (3, 10), (3, 15))]

def run(arm, inst, r, budget=BUDGET):
    t0 = time.perf_counter(); p, ub = load(inst); s = 1000 + 7919 * r
    kw = {k: arm[k] for k in ("cq_min", "rmin", "ls", "cq_ls", "cpi") if k in arm}
    c, used, it = iterated_greedy(p, budget, s, **kw)
    return dict(Arm=arm["nom"], Instance=inst, Run=r, UB=ub, Cmax=c, Gap=100.0 * (c - ub) / ub, Evals=used, Iter=it, sec=time.perf_counter() - t0)

def plan(phase):
    if phase.startswith("tune"): return {"tune": TUNE, "tune2": TUNE2, "tune3": TUNE3}[phase], SETS["TUNE+"], 5, 41
    arms = json.load(open(os.path.join(HERE, "arms.json")))
    if phase.endswith("_cpi2"):                       # robustness: 2 oracle calls per Grover iteration, same settings
        arms = [dict(a, nom=a["nom"] + " cpi2", cpi=2) for a in arms if a["nom"].startswith("Q-IG")]
    if phase == "ext": return arms, SETS["EXT"], 30, 41            # replication set (Amendment 4)
    return arms, SETS["TEST" if phase.startswith("test") else "HARD"], 30, 41

if __name__ == "__main__":
    phase, sh, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); out = sys.argv[4] if len(sys.argv) > 4 else os.path.join(HERE, "..", "results", "raw")
    arms, insts, nr, seed = plan(phase); os.makedirs(out, exist_ok=True)
    T = [(a, i, r) for r in range(nr) for i in insts for a in arms]; random.Random(seed).shuffle(T)
    fn = os.path.join(out, f"{phase}_{sh}.csv"); rows = pd.read_csv(fn).to_dict("records") if os.path.exists(fn) else []
    done = {(x["Arm"], x["Instance"], int(x["Run"])) for x in rows}
    for k, t in enumerate([t for t in T[sh::ns] if (t[0]["nom"], t[1], t[2]) not in done]):
        rows.append(run(*t))
        if (k + 1) % 3 == 0: pd.DataFrame(rows).to_csv(fn, index=False)
    pd.DataFrame(rows).to_csv(fn, index=False); open(fn + ".done", "w").write("ok")
