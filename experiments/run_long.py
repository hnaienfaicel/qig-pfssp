"""Exploratory long-budget runs on the open instances of TEST and HARD (best-known solution not proven optimal).
Not preregistered. Usage: python experiments/run_long.py PHASE SHARD NSHARDS OUTDIR
PHASE: long10 (budget x10, 10 runs) | long100 (budget x100, 5 runs). Arms: IG, IG-RR, Q-IG-G with the tuned settings."""
import os, sys, json, random, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run import run, BUDGET, HERE
OPEN = sys.argv[5].split(",") if len(sys.argv) > 5 else None

def plan(phase):
    arms = [a for a in json.load(open(os.path.join(HERE, "arms.json"))) if a["nom"] in ("IG", "IG-RR", "Q-IG-G")]
    mult, nr = {"long10": (10, 10), "long100": (100, 5)}[phase]
    return arms, mult, nr

if __name__ == "__main__":
    phase, sh, ns, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    insts = open(os.path.join(out, "open_instances.txt")).read().split()
    arms, mult, nr = plan(phase); os.makedirs(out, exist_ok=True)
    T = [(a, i, r) for r in range(nr) for i in insts for a in arms]; random.Random(41).shuffle(T)
    fn = os.path.join(out, f"{phase}_{sh}.csv"); rows = pd.read_csv(fn).to_dict("records") if os.path.exists(fn) else []
    done = {(x["Arm"], x["Instance"], int(x["Run"])) for x in rows}
    for t in [t for t in T[sh::ns] if (t[0]["nom"], t[1], t[2]) not in done]:
        x = run(*t, budget=BUDGET * mult); x["Mult"] = mult; rows.append(x)
        pd.DataFrame(rows).to_csv(fn, index=False)
    pd.DataFrame(rows).to_csv(fn, index=False); open(fn + ".done", "w").write("ok")
