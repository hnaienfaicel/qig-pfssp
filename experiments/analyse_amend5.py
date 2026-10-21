"""Analysis of the exploratory studies of Amendment 5 (not confirmatory).
Reads results/raw_amend5/{abl,grid,equiv}_*.csv (or the merged results/amend5_*_runs.csv), the campaign runs and the
long-budget runs. Writes results/amend5_*.csv. Usage: python experiments/analyse_amend5.py"""
import os, sys, glob, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qig.instances import INSTANCES, SETS, gap_vs_bks
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"); B0 = 46080
SETOF = {i: s for s in ("TEST", "HARD", "EXT") for i in SETS[s]}
rng = np.random.default_rng(20261021)


def merged(phase):
    fs = glob.glob(os.path.join(R, "raw_amend5", f"{phase}_*.csv")); out = os.path.join(R, f"amend5_{phase}_runs.csv")
    if not fs: return pd.read_csv(out) if os.path.exists(out) else None
    D = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates(["Arm", "Instance", "Run", "Mult"])
    D = gap_vs_bks(D).sort_values(["Arm", "Mult", "Instance", "Run"]); D.to_csv(out, index=False); return D


def boot_ci(d, B=10000):
    d = np.asarray(d, float); idx = rng.integers(0, len(d), (B, len(d))); m = d[idx].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def paired(a, b, label, s, metric):
    d = (a - b).dropna(); lo, hi = boot_ci(d)
    p = stats.wilcoxon(a[d.index], b[d.index]).pvalue if (d != 0).any() else 1.0
    return dict(Comparison=label, Set=s, Metric=metric, Diff_pp=d.mean(), CI_low=lo, CI_high=hi, A_better=int((d < 0).sum()),
                B_better=int((d > 0).sum()), ties=int((d == 0).sum()), p=p, n=len(d))


def holm(df):
    p = df.p.values; o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for k, i in enumerate(o): run = max(run, min(1.0, (len(p) - k) * p[i])); adj[i] = run
    return df.assign(p_Holm=adj)


camp = gap_vs_bks(pd.concat([pd.read_csv(os.path.join(R, f)) for f in ("test_runs.csv", "hard_runs.csv", "ext_runs.csv")], ignore_index=True))
camp["Set"] = camp.Instance.map(SETOF)
ARMS = ["IG", "IG-Rmin", "Q-IG-min", "IG-R", "IG-RR", "Q-IG-LS", "Q-IG-G"]

# ------------------------------------------------------------------ 1. ablation 2x2
A = merged("abl")
if A is not None:
    A["Set"] = A.Instance.map(SETOF); assert len(A) == 30 * 330, len(A)
    C = pd.concat([camp, A.drop(columns="Mult")], ignore_index=True)
    lev, tst = [], []
    for s in ("TEST", "HARD", "EXT"):
        X = C[C.Set == s]; b30 = X.groupby(["Instance", "Arm"]).Gap.min().unstack(); m30 = X.groupby(["Instance", "Arm"]).Gap.mean().unstack()
        for a in ARMS: lev.append(dict(Set=s, Arm=a, best30=b30[a].mean(), mean30=m30[a].mean()))
        for lab, x, y in (("Q-IG-LS - IG-R", "Q-IG-LS", "IG-R"), ("Q-IG-LS - IG", "Q-IG-LS", "IG"), ("Q-IG-G - Q-IG-LS", "Q-IG-G", "Q-IG-LS"),
                          ("Q-IG-G - IG", "Q-IG-G", "IG")):
            tst.append(paired(b30[x], b30[y], lab, s, "best30")); tst.append(paired(m30[x], m30[y], lab, s, "mean30"))
    pd.DataFrame(lev).to_csv(os.path.join(R, "amend5_abl_levels.csv"), index=False)
    T = holm(pd.DataFrame(tst)); T.to_csv(os.path.join(R, "amend5_abl_tests.csv"), index=False)
    print(pd.DataFrame(lev).pivot(index="Arm", columns="Set", values="best30").loc[ARMS].round(3).to_string())
    print(T[T.Metric == "best30"][["Comparison", "Set", "Diff_pp", "CI_low", "CI_high", "A_better", "B_better", "ties", "p_Holm"]].round(4).to_string(index=False))

# ------------------------------------------------------------------ 2. settings grids
G = merged("grid")
if G is not None:
    tuned = {"Q-IG-G": "Q-IG-G c2 l1", "IG-RR": "IG-RR r3 l6", "Q-IG-min": "Q-IG-min c0.5", "IG-Rmin": "IG-Rmin r1"}
    base = camp[camp.Arm.isin(tuned) & (camp.Run < 10) & camp.Set.isin(["TEST", "HARD"])].copy(); base["Arm"] = base.Arm.map(tuned)
    G = pd.concat([G.drop(columns="Mult"), base], ignore_index=True); G["Set"] = G.Instance.map(SETOF)
    G["Family"] = G.Arm.str.split(" ").str[0]
    n_set = G.groupby("Arm").apply(lambda x: x[["Instance", "Run"]].drop_duplicates().shape[0])
    assert (n_set == 1400).all(), n_set[n_set != 1400]
    rows, tst = [], []
    for s in ("TEST", "HARD"):
        X = G[G.Set == s]; b10 = X.groupby(["Instance", "Arm"]).Gap.min().unstack(); m10 = X.groupby(["Instance", "Arm"]).Gap.mean().unstack()
        for a in b10.columns:
            fam = a.split(" ")[0]; par = dict(kv for kv in ((t[0], float(t[1:])) for t in a.split(" ")[1:]))
            rows.append(dict(Set=s, Family=fam, Setting=a, tuned=a in tuned.values(), c_min=par.get("c"), r_min=par.get("r"), c_ls=par.get("l"),
                             best10=b10[a].mean(), mean10=m10[a].mean()))
        Rg = pd.DataFrame([r for r in rows if r["Set"] == s])
        for q, t in (("Q-IG-G", "IG-RR"), ("Q-IG-min", "IG-Rmin")):
            Rq, Rt = Rg[Rg.Family == q], Rg[Rg.Family == t]
            worst_q, best_t = Rq.loc[Rq.best10.idxmax(), "Setting"], Rt.loc[Rt.best10.idxmin(), "Setting"]
            x = paired(b10[worst_q], b10[best_t], f"worst {q} - best {t}", s, "best10"); x.update(A=worst_q, B=best_t); tst.append(x)
            x = paired(b10[tuned[q]], b10[best_t], f"tuned {q} - best {t}", s, "best10"); x.update(A=tuned[q], B=best_t); tst.append(x)
            wm, bm = Rq.loc[Rq.mean10.idxmax(), "Setting"], Rt.loc[Rt.mean10.idxmin(), "Setting"]
            x = paired(m10[wm], m10[bm], f"worst {q} - best {t}", s, "mean10"); x.update(A=wm, B=bm); tst.append(x)
    GR = pd.DataFrame(rows); GR.to_csv(os.path.join(R, "amend5_grid_levels.csv"), index=False)
    GT = holm(pd.DataFrame(tst)); GT.to_csv(os.path.join(R, "amend5_grid_tests.csv"), index=False)
    for s in ("TEST", "HARD"):
        g = GR[GR.Set == s]
        print(s, "Q-IG-G range", g[g.Family == "Q-IG-G"].best10.agg(["min", "max"]).round(3).tolist(), "IG-RR range", g[g.Family == "IG-RR"].best10.agg(["min", "max"]).round(3).tolist(),
              "Q-IG-min", g[g.Family == "Q-IG-min"].best10.agg(["min", "max"]).round(3).tolist(), "IG-Rmin", g[g.Family == "IG-Rmin"].best10.agg(["min", "max"]).round(3).tolist())
    print(GT[["Comparison", "Set", "Metric", "A", "B", "Diff_pp", "CI_low", "CI_high", "A_better", "B_better", "ties", "p_Holm"]].round(4).to_string(index=False))

# ------------------------------------------------------------------ 3. budget equivalence
E = merged("equiv")
if E is not None:
    OPEN = open(os.path.join(R, "raw_long", "open_instances.txt")).read().split()
    lng = gap_vs_bks(pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R, "raw_long", "long*_*.csv"))], ignore_index=True))
    b1 = camp[camp.Instance.isin(OPEN) & camp.Arm.isin(["IG", "IG-RR", "Q-IG-G"])].assign(Mult=1)
    X = pd.concat([b1, lng, E], ignore_index=True); X = X[X.Instance.isin(OPEN) & (X.Run < 5)]
    P = X.groupby(["Instance", "Arm", "Mult"]).Gap.min().unstack([1, 2]); assert len(P) == 38 and not P.isna().any().any(), P.isna().sum()
    target = P[("Q-IG-G", 1)]
    ci95 = lambda x: stats.t.ppf(0.975, len(x) - 1) * np.std(x, ddof=1) / np.sqrt(len(x))

    def kstar(Pm, arm, tgt):
        ks = sorted(k for a, k in Pm.columns if a == arm); g = np.array([Pm[(arm, k)].mean() for k in ks]); t = tgt.mean()
        for i in range(len(ks) - 1):
            if g[i] >= t >= g[i + 1]:
                f = (g[i] - t) / (g[i] - g[i + 1]); return 10 ** (np.log10(ks[i]) + f * (np.log10(ks[i + 1]) - np.log10(ks[i])))
        return np.nan
    eq, curve = [], []
    for arm in ("IG", "IG-RR"):
        k0 = kstar(P, arm, target); boots = []
        for _ in range(2000):
            ii = rng.integers(0, len(P), len(P)); Pb = P.iloc[ii]; boots.append(kstar(Pb, arm, Pb[("Q-IG-G", 1)]))
        boots = np.array(boots); lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        eq.append(dict(Arm=arm, k_star=k0, CI_low=lo, CI_high=hi, boot_nan=int(np.isnan(boots).sum()), target_gap=target.mean()))
        for k in sorted(k for a, k in P.columns if a == arm):
            d = P[("Q-IG-G", 1)] - P[(arm, k)]
            curve.append(dict(Arm=arm, Mult=k, Gap=P[(arm, k)].mean(), CI95=ci95(P[(arm, k)]), QIGG_x1_minus=d.mean(), A_better=int((d < 0).sum()), B_better=int((d > 0).sum()),
                              p=stats.wilcoxon(P[("Q-IG-G", 1)], P[(arm, k)]).pvalue if (d != 0).any() else 1.0))
    curve.append(dict(Arm="Q-IG-G", Mult=1, Gap=target.mean(), CI95=ci95(target)))
    EQ = pd.DataFrame(eq); EQ.to_csv(os.path.join(R, "amend5_equiv.csv"), index=False)
    CU = pd.DataFrame(curve); CU.to_csv(os.path.join(R, "amend5_equiv_curve.csv"), index=False)
    print(CU.round(4).to_string(index=False)); print(EQ.round(3).to_string(index=False))

# ------------------------------------------------------------------ 4. per-group tables
C = camp.copy()
if A is not None: C = pd.concat([C, A.drop(columns="Mult")], ignore_index=True)
C["Group"] = C.Instance.str.rsplit("_", n=1).str[0]
pg = []
for (s, g), X in C.groupby(["Set", "Group"]):
    b30 = X.groupby(["Instance", "Arm"]).Gap.min().unstack(); m30 = X.groupby(["Instance", "Arm"]).Gap.mean().unstack()
    for a in [a for a in ARMS if a in b30.columns]:
        pg.append(dict(Set=s, Group=g, n=int(g.split("_")[0].lstrip("taiVFR")), m=int(g.split("_")[1]), Instances=X.Instance.nunique(),
                       Arm=a, best30=b30[a].mean(), mean30=m30[a].mean()))
PG = pd.DataFrame(pg); PG.to_csv(os.path.join(R, "amend5_per_group.csv"), index=False)
print(PG.pivot_table(index=["Set", "Group"], columns="Arm", values="best30").round(2).head(6).to_string())

# ------------------------------------------------------------------ best-known solutions over the new runs
new = pd.concat([d for d in (A, G, E) if d is not None], ignore_index=True) if any(d is not None for d in (A, G, E)) else None
if new is not None:
    S = new.groupby("Instance").Cmax.min().to_frame("best").join(INSTANCES.set_index("instance")[["ub", "optimal"]])
    print(dict(new_runs=len(new), below_bks=int((S.best < S.ub).sum()), equal_bks_open=int(((S.best == S.ub) & (S.optimal == 0)).sum())))

# ------------------------------------------------------------------ 5. descriptive comparison with Farhani et al. (2024)
from qig.instances import load
from qig.flowshop import neh
FA = pd.read_csv(os.path.join(R, "farhani2024_table23.csv"))
tune = gap_vs_bks(pd.read_csv(os.path.join(R, "tune_runs.csv"))); tune = tune[tune.Arm.isin(["IG", "Q-IG-G c2 l1"])].replace({"Arm": {"Q-IG-G c2 l1": "Q-IG-G"}})
ours = pd.concat([camp[camp.Arm.isin(["IG", "Q-IG-G"])], tune], ignore_index=True)
ours = ours[ours.Instance.isin(FA.instance.unique())]
ub = dict(zip(INSTANCES.instance, INSTANCES.ub)); rows = []
for i in sorted(FA.instance.unique()):
    p, _ = load(i); c = neh(p)[1]; x = ours[ours.Instance == i]
    f = FA[FA.instance == i]
    rows.append(dict(instance=i, group=f.group.iloc[0], optimal=int(INSTANCES.set_index("instance").optimal[i]), NEH=100.0 * (c - ub[i]) / ub[i],
                     IG_mean=x[x.Arm == "IG"].Gap.mean(), QIGG_mean=x[x.Arm == "Q-IG-G"].Gap.mean(), runs=int((x.Arm == "IG").sum()),
                     QUBO_position=f[f.formulation == "Position-based"].ub_gap.iloc[0], QUBO_best6=f.ub_gap.min()))
FX = pd.DataFrame(rows); FX.to_csv(os.path.join(R, "amend5_farhani_instances.csv"), index=False)
FG = FX.groupby("group", sort=False)[["NEH", "IG_mean", "QIGG_mean", "QUBO_position", "QUBO_best6"]].mean()
FG["instances"] = FX.groupby("group", sort=False).size(); FG["runs_min"] = FX.groupby("group", sort=False).runs.min()
FG.loc["all"] = list(FX[["NEH", "IG_mean", "QIGG_mean", "QUBO_position", "QUBO_best6"]].mean()) + [len(FX), FX.runs.min()]
FG.to_csv(os.path.join(R, "amend5_farhani_groups.csv")); print(FG.round(2).to_string())
