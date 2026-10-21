"""Exploratory studies declared in Amendment 5 (not confirmatory).
Usage: python experiments/run_amend5.py PHASE SHARD NSHARDS OUTDIR
PHASE: abl   (arm Q-IG-LS = full-scan construction + Grover descent, TEST/HARD/EXT, 30 runs)
       grid  (settings grids of Q-IG-G, IG-RR, Q-IG-min, IG-Rmin, TEST/HARD, runs 0-9; the tuned settings are
              not rerun, their runs 0-9 are taken from the campaign)
       equiv (IG at x15, x20, x30, x50 and IG-RR at x5, x7, x15 on the 38 open TEST/HARD instances, runs 0-4)"""
import os, sys, json, random, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qig.instances import SETS
from experiments.run import run, BUDGET, HERE

ABL = dict(nom="Q-IG-LS", ls="gls", cq_ls=1)
TUNED = {"Q-IG-G c2 l1", "IG-RR r3 l6", "Q-IG-min c0.5", "IG-Rmin r1"}
GRID = ([dict(nom=f"Q-IG-G c{a} l{b}", cq_min=a, ls="gls", cq_ls=b) for a in (0.5, 1, 2, 3) for b in (0.5, 1, 2, 3)] +
        [dict(nom=f"IG-RR r{a} l{b}", rmin=a, ls="rls", cq_ls=b) for a in (1, 2, 3, 4, 6) for b in (3, 6, 10, 15)] +
        [dict(nom=f"Q-IG-min c{c}", cq_min=c) for c in (0.25, 0.5, 1, 2)] +
        [dict(nom=f"IG-Rmin r{c}", rmin=c) for c in (0.5, 1, 2, 3)])
EQUIV = [("IG", m) for m in (15, 20, 30, 50)] + [("IG-RR", m) for m in (5, 7, 15)]


def tasks(phase):
    if phase == "abl":
        return [(ABL, i, r, 1) for r in range(30) for s in ("TEST", "HARD", "EXT") for i in SETS[s]]
    if phase == "grid":
        return [(a, i, r, 1) for r in range(10) for s in ("TEST", "HARD") for i in SETS[s] for a in GRID if a["nom"] not in TUNED]
    if phase == "equiv":
        arms = {a["nom"]: a for a in json.load(open(os.path.join(HERE, "arms.json")))}
        insts = open(os.path.join(HERE, "..", "results", "raw_long", "open_instances.txt")).read().split()
        return [(arms[n], i, r, m) for r in range(5) for i in insts for n, m in EQUIV]
    raise SystemExit(f"unknown phase {phase}")


if __name__ == "__main__":
    phase, sh, ns, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; os.makedirs(out, exist_ok=True)
    T = tasks(phase); random.Random(41).shuffle(T)
    fn = os.path.join(out, f"{phase}_{sh}.csv"); rows = pd.read_csv(fn).to_dict("records") if os.path.exists(fn) else []
    done = {(x["Arm"], x["Instance"], int(x["Run"]), int(x.get("Mult", 1))) for x in rows}
    todo = [t for t in T[sh::ns] if (t[0]["nom"], t[1], t[2], t[3]) not in done]
    for k, (a, i, r, m) in enumerate(todo):
        x = run(a, i, r, budget=BUDGET * m); x["Mult"] = m; rows.append(x)
        if (k + 1) % 3 == 0: pd.DataFrame(rows).to_csv(fn, index=False)
    pd.DataFrame(rows).to_csv(fn, index=False); open(fn + ".done", "w").write("ok")
