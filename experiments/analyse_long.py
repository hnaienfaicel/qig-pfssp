"""Exploratory analysis of the long-budget runs (not preregistered). Unit = open instance (best-known solution not proven
optimal), best of the first 5 runs (seeds 1000 + 7919 r, r = 0..4) at each budget; paired Wilcoxon signed-rank test,
t-based 95% CI of the mean paired difference. Budget x1 = the campaign runs; x10 and x100 = results/raw_long;
x1000 record attempt = results/raw_record (Q-IG-G and IG, 8 runs, six instances).
Usage: python experiments/analyse_long.py"""
import os, sys, glob, json, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qig.instances import INSTANCES, gap_vs_bks
from qig.flowshop import makespan
from qig.instances import load
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"); B0 = 46080; ARMS = ["IG", "IG-RR", "Q-IG-G"]
OPEN = open(os.path.join(R, "raw_long", "open_instances.txt")).read().split()
assert set(OPEN) == set(INSTANCES[(INSTANCES.optimal == 0) & INSTANCES.set.isin(["TEST", "HARD"])].instance)
base = pd.concat([pd.read_csv(os.path.join(R, f)) for f in ("test_runs.csv", "hard_runs.csv")]).assign(Mult=1)
A = pd.concat([base] + [pd.read_csv(f) for f in glob.glob(os.path.join(R, "raw_long", "long*_*.csv"))], ignore_index=True)
A = gap_vs_bks(A[A.Instance.isin(OPEN) & A.Arm.isin(ARMS) & (A.Run < 5)])
P = A.groupby(["Instance", "Mult", "Arm"]).Gap.min().unstack([1, 2]); assert P.shape == (len(OPEN), 9) and not P.isna().any().any()
def ci(x):
    x = np.asarray(x, float); return x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
lev = [dict(Mult=k, Budget=k * B0, Arm=a, Gap=ci(P[(k, a)])[0], CI95=ci(P[(k, a)])[1], Instances=len(OPEN)) for k in (1, 10, 100) for a in ARMS]
pairs = [((k, "Q-IG-G"), (k, b)) for k in (1, 10, 100) for b in ("IG-RR", "IG")] + \
        [((1, "Q-IG-G"), (10, "IG")), ((1, "Q-IG-G"), (10, "IG-RR")), ((10, "Q-IG-G"), (100, "IG")), ((10, "Q-IG-G"), (100, "IG-RR"))]
tst = []
for (ka, a), (kb, b) in pairs:
    d = P[(ka, a)] - P[(kb, b)]; m, h = ci(d)
    tst.append(dict(A=a, MultA=ka, B=b, MultB=kb, Diff_pp=m, CI_low=m - h, CI_high=m + h, A_better=int((d < 0).sum()), B_better=int((d > 0).sum()),
                    ties=int((d == 0).sum()), p=stats.wilcoxon(P[(ka, a)], P[(kb, b)]).pvalue))
pd.DataFrame(lev).to_csv(os.path.join(R, "long_levels.csv"), index=False); pd.DataFrame(tst).to_csv(os.path.join(R, "long_tests.csv"), index=False)
sec = A[A.Mult == 100].groupby("Arm").sec.median().rename("median_sec_x100"); sec.to_csv(os.path.join(R, "long_seconds.csv"))
rec = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "raw_record", "*.json")))]
if rec:
    X = pd.DataFrame(rec); X["UB"] = X.Instance.map(dict(zip(INSTANCES.instance, INSTANCES.ub)))
    for r in X.itertuples():                                   # every reported makespan is re-evaluated from its permutation
        p, _ = load(r.Instance); assert int(makespan(p, np.array([int(v) - 1 for v in r.perm.split()]))) == r.Cmax
    X["Gap"] = 100.0 * (X.Cmax - X.UB) / X.UB; X.drop(columns="perm").to_csv(os.path.join(R, "record_runs.csv"), index=False)
    best = X.loc[X.groupby(["Instance", "Arm"]).Cmax.idxmin()]
    best[["Instance", "Arm", "Run", "Cmax", "UB", "Gap", "perm"]].to_csv(os.path.join(R, "record_best.csv"), index=False)
    print(X.groupby(["Instance", "Arm"]).agg(runs=("Cmax", "size"), best=("Cmax", "min"), bks=("UB", "first"), sec=("sec", "median")).to_string())
print(pd.DataFrame(lev).round(3).to_string(index=False)); print(pd.DataFrame(tst).round(4).to_string(index=False))
# summary over every run of the paper: is any best-known solution improved or matched, and which instance-file bounds are beaten?
fs = [os.path.join(R, f) for f in ("test_runs.csv", "hard_runs.csv", "tune_runs.csv", "tune2_runs.csv", "tune3_runs.csv")] + \
     glob.glob(os.path.join(R, "raw", "*cpi2*.csv")) + glob.glob(os.path.join(R, "raw_long", "long*_*.csv"))
allr = pd.concat([pd.read_csv(f)[["Arm", "Instance", "Cmax"]] for f in fs] + ([pd.DataFrame(rec)[["Arm", "Instance", "Cmax"]]] if rec else []))
S = allr.groupby("Instance").Cmax.min().to_frame("best").join(INSTANCES.set_index("instance")[["set", "ub", "ub_instance_file", "optimal"]])
summ = dict(runs=len(allr), instances=len(S), below_bks=int((S.best < S.ub).sum()), equal_bks=int((S.best == S.ub).sum()),
            equal_bks_open=int(((S.best == S.ub) & (S.optimal == 0)).sum()), below_instance_file=int((S.best < S.ub_instance_file).sum()),
            below_instance_file_proven_optimum=int(((S.best < S.ub_instance_file) & (S.optimal == 1)).sum()),
            below_instance_file_open=int(((S.best < S.ub_instance_file) & (S.optimal == 0)).sum()))
pd.Series(summ).to_csv(os.path.join(R, "bks_summary.csv"), header=["value"]); print(summ)
