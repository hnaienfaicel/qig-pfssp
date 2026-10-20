"""Preregistered analysis (preregistration/preregistration_qig_fr.md). Unit = instance; paired Wilcoxon signed-rank
test; bootstrap 95% CI of the mean paired difference (10,000 resamples); Holm correction within each family.
Usage: python experiments/analyse.py   (reads results/test_runs.csv, results/hard_runs.csv and results/raw/*_cpi2_*.csv)
Gaps are recomputed against the best-known solutions in data/instances.csv (amendment of 20/10/2026, see the preregistration)."""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qig.instances import gap_vs_bks
from scipy.stats import wilcoxon
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty_like(ps); run = 0.0
    for r, i in enumerate(o): run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1.0, run)
    return adj
def boot(d, B=10000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d); m = d[rng.integers(0, len(d), (B, len(d)))].mean(1); return np.percentile(m, [2.5, 97.5])
def cmp(lab, a, b):
    a, b = a.align(b, join="inner"); d = a - b; lo, hi = boot(d.values)
    return dict(Comparison=lab, A=a.mean(), B=b.mean(), Diff_pp=d.mean(), CI_low=lo, CI_high=hi, p=wilcoxon(a, b).pvalue if (d != 0).any() else 1.0,
                A_better=int((d < 0).sum()), B_better=int((d > 0).sum()), ties=int((d == 0).sum()))
FAM = {"primary": [("H1 Q-IG-min - IG", "Q-IG-min", "IG"), ("H2 Q-IG-G - IG-R", "Q-IG-G", "IG-R"), ("H3 Q-IG-G - IG", "Q-IG-G", "IG")],
       "secondary": [("H1b Q-IG-min - IG-Rmin", "Q-IG-min", "IG-Rmin"), ("H2b Q-IG-G - IG-RR", "Q-IG-G", "IG-RR")],
       "robustness_cpi2": [("Q-IG-min cpi2 - IG-Rmin", "Q-IG-min cpi2", "IG-Rmin"), ("Q-IG-min cpi2 - IG", "Q-IG-min cpi2", "IG"),
                           ("Q-IG-G cpi2 - IG-RR", "Q-IG-G cpi2", "IG-RR"), ("Q-IG-G cpi2 - IG", "Q-IG-G cpi2", "IG")]}
out, levels = [], []
for ph, lab in (("test", "TEST"), ("hard", "HARD")):
    A = pd.read_csv(os.path.join(R, f"{ph}_runs.csv"))
    raw = glob.glob(os.path.join(R, "raw", f"{ph}_cpi2_*.csv"))
    if raw: A = pd.concat([A] + [pd.read_csv(f) for f in raw], ignore_index=True).drop_duplicates(["Arm", "Instance", "Run"])
    A = gap_vs_bks(A)                                     # gaps against the current best-known solutions
    cnt = A.groupby("Arm").size(); assert cnt.nunique() == 1, cnt.to_dict()
    T = {"best30": A.groupby(["Arm", "Instance"]).Gap.min().unstack(0), "mean30": A.groupby(["Arm", "Instance"]).Gap.mean().unstack(0)}
    levels.append(pd.DataFrame({f"{lab} {m}": T[m].mean() for m in T}))
    for fam, pairs in FAM.items():
        for met, X in T.items():
            for c_, a, b in pairs:
                if a in X and b in X: out.append(dict(Family=fam, Set=lab, Metric=met, **cmp(c_, X[a], X[b])))
Rt = pd.DataFrame(out)
for fam in FAM:
    for met in ("best30", "mean30"):
        s = (Rt.Family == fam) & (Rt.Metric == met)
        if s.any(): Rt.loc[s, "p_Holm"] = holm(Rt.loc[s, "p"].values)
L = pd.concat(levels, axis=1); L.to_csv(os.path.join(R, "levels.csv")); Rt.to_csv(os.path.join(R, "tests.csv"), index=False)
print(L.round(3).to_string()); print(Rt[Rt.Metric == "best30"].round(4).to_string(index=False))
