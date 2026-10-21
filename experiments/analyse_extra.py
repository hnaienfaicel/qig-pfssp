"""Exploratory analyses of Amendment 4 (not confirmatory): noisy Grover and convergence curves.
Usage: python experiments/analyse_extra.py  (reads results/raw_extra/{noise,trace}_*.csv, or the merged results/{noise,trace}_runs.csv, results/{test,hard}_runs.csv)
Writes results/noise_levels.csv, noise_tests.csv, convergence.csv, convergence_summary.csv."""
import os, sys, glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qig.instances import gap_vs_bks, SETS, UB
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty_like(ps); run = 0.0
    for r, i in enumerate(o): run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1.0, run)
    return adj
SETOF = {i: s for s in ("TEST", "HARD") for i in SETS[s]}
base = gap_vs_bks(pd.concat([pd.read_csv(os.path.join(R, f)) for f in ("test_runs.csv", "hard_runs.csv")]))
base["Set"] = base.Instance.map(SETOF)
# ------------------------------------------------------------------ noise
_nf = glob.glob(os.path.join(R, "raw_extra", "noise_*.csv"))   # shards (not versioned); fall back to the merged file
N = pd.concat([pd.read_csv(f) for f in _nf], ignore_index=True) if _nf else pd.read_csv(os.path.join(R, "noise_runs.csv"))
N = gap_vs_bks(N.drop_duplicates(["Arm", "Instance", "Run"])); N["Set"] = N.Instance.map(SETOF)
assert len(N) == 2 * 5 * 140 * 30, len(N)
N.to_csv(os.path.join(R, "noise_runs.csv"), index=False)
B0 = base.assign(Base=base.Arm, Eps=0.0)
ALL = pd.concat([B0[B0.Arm.isin(["Q-IG-min", "Q-IG-G"])], N], ignore_index=True)
lev, tst = [], []
for s in ("TEST", "HARD"):
    tw = base[base.Set == s].groupby(["Arm", "Instance"]).Gap.min().unstack(0)
    for (b, e), g in ALL[ALL.Set == s].groupby(["Base", "Eps"]):
        x = g.groupby("Instance").Gap.min()
        lev.append(dict(Set=s, Arm=b, Eps=e, best30=x.mean(), mean30=g.groupby("Instance").Gap.mean().mean()))
        for twin in ({"Q-IG-min": "IG-Rmin", "Q-IG-G": "IG-RR"}[b], "IG"):
            y = tw[twin].loc[x.index]; d = x - y
            tst.append(dict(Set=s, Arm=b, Eps=e, Comparator=twin, Diff_pp=d.mean(), A_better=int((d < 0).sum()), B_better=int((d > 0).sum()),
                            ties=int((d == 0).sum()), p=wilcoxon(x, y).pvalue if (d != 0).any() else 1.0))
    for a in ("IG", "IG-Rmin", "IG-RR"):
        lev.append(dict(Set=s, Arm=a, Eps=np.nan, best30=tw[a].mean(), mean30=base[(base.Set == s) & (base.Arm == a)].groupby("Instance").Gap.mean().mean()))
L = pd.DataFrame(lev); T = pd.DataFrame(tst)
m = T.Eps > 0; T.loc[m, "p_Holm"] = holm(T.loc[m, "p"].values)          # one exploratory family over all eps > 0
L.to_csv(os.path.join(R, "noise_levels.csv"), index=False); T.to_csv(os.path.join(R, "noise_tests.csv"), index=False)
print(L.pivot_table(index=["Arm", "Eps"], columns="Set", values="best30").round(3).to_string())
print(T[T.Comparator != "IG"].pivot_table(index=["Arm", "Eps"], columns="Set", values="Diff_pp").round(3).to_string())
# ------------------------------------------------------------------ convergence
fs = glob.glob(os.path.join(R, "raw_extra", "trace_*.csv")) or ([os.path.join(R, "trace_runs.csv")] if os.path.exists(os.path.join(R, "trace_runs.csv")) else [])
if fs:
    C = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates(["Arm", "Instance", "Run"])
    assert len(C) == 6 * 140 * 30, len(C)
    chk = C.merge(base[["Arm", "Instance", "Run", "Cmax", "Evals", "Iter"]], on=["Arm", "Instance", "Run"], suffixes=("", "_c"))
    assert len(chk) == len(C) and (chk.Cmax == chk.Cmax_c).all() and (chk.Evals == chk.Evals_c).all(), "trace runs differ from the campaign"
    C.to_csv(os.path.join(R, "trace_runs.csv"), index=False)
    cols = [c for c in C.columns if c.startswith("c") and c[1:].isdigit()]; C["Set"] = C.Instance.map(SETOF); C["UBv"] = C.Instance.map(UB)
    rows = []
    for t in cols:
        G = C.assign(g=100.0 * (C[t] - C.UBv) / C.UBv)
        for (s, a), g in G.groupby(["Set", "Arm"]):
            per = g.groupby("Instance").g
            if per.count().min() < 30: continue                # NEH not finished for some runs at this checkpoint
            rows.append(dict(Set=s, Arm=a, calls=int(t[1:]), mean_gap=per.mean().mean(), best_gap=per.min().mean()))
    V = pd.DataFrame(rows); V.to_csv(os.path.join(R, "convergence.csv"), index=False)
    S = []
    for s in ("TEST", "HARD"):
        target = V[(V.Set == s) & (V.Arm == "Q-IG-G")]
        for a in V.Arm.unique():
            v = V[(V.Set == s) & (V.Arm == a)].sort_values("calls")
            for lvl_arm in ("IG",):
                fin = V[(V.Set == s) & (V.Arm == lvl_arm)].sort_values("calls").mean_gap.iloc[-1]
                hit = v[v.mean_gap <= fin]
                S.append(dict(Set=s, Arm=a, final_mean_gap=v.mean_gap.iloc[-1], calls_to_reach_IG_final=int(hit.calls.iloc[0]) if len(hit) else np.nan))
    S = pd.DataFrame(S); S.to_csv(os.path.join(R, "convergence_summary.csv"), index=False); print(S.to_string(index=False))
