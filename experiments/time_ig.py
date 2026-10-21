"""Wall-clock time per makespan evaluation of the classical IG (Amendment 5, budget equivalence).
IG with its published parameters at 10 x 46,080 calls, 3 runs, on the instances of results/ft_resources.csv, one process.
Usage: python experiments/time_ig.py  ->  results/amend5_ig_timing.csv"""
import os, sys, platform, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run import run, BUDGET
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor())
insts = sorted(pd.read_csv(os.path.join(R, "ft_resources.csv")).instance.unique())
run(dict(nom="IG"), insts[0], 0, budget=BUDGET)                       # compile the numba kernels once
rows = [dict(run(dict(nom="IG"), i, r, budget=10 * BUDGET), cpu=cpu) for i in insts for r in range(3)]
D = pd.DataFrame(rows); D["sec_per_call"] = D.sec / D.Evals
D.to_csv(os.path.join(R, "amend5_ig_timing.csv"), index=False)
print(D.groupby("Instance").sec_per_call.median().map(lambda v: f"{v * 1e6:.2f} us").to_string()); print(cpu)
