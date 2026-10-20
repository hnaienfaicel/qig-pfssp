"""LaTeX tables of the paper, generated from results/*.csv (no number is typed by hand)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qig.instances import gap_vs_bks
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results"); OUT = os.path.join(HERE, "..", "paper", "tables")
os.makedirs(OUT, exist_ok=True)
f2 = lambda v: f"{v:.2f}"; f3 = lambda v: f"{v:.3f}"; s3 = lambda v: f"${v:.3f}$"   # s3: signed, typeset minus
def pfmt(p):
    if p < 1e-4: m, e = f"{p:.1e}".split("e"); return f"${m}\\times 10^{{{int(e)}}}$"
    return f"{p:.4f}"
NAMES = {"IG": "IG (full scan)", "IG-Rmin": "IG-Rmin (twin)", "Q-IG-min": "\\textbf{Q-IG-min}", "IG-R": "IG-R (twin)", "IG-RR": "IG-RR (twin)", "Q-IG-G": "\\textbf{Q-IG-G}",
         "Q-IG-min cpi2": "Q-IG-min, 2 calls/iteration", "Q-IG-G cpi2": "Q-IG-G, 2 calls/iteration"}
L = pd.read_csv(os.path.join(R, "levels.csv"), index_col=0)
order = ["IG", "IG-Rmin", "Q-IG-min", "IG-R", "IG-RR", "Q-IG-G", "Q-IG-min cpi2", "Q-IG-G cpi2"]
rows = [" & ".join([NAMES[a]] + [f3(L.loc[a, c]) for c in ("TEST best30", "TEST mean30", "HARD best30", "HARD mean30")]) + r" \\" for a in order]
rows.insert(6, r"\midrule")
open(os.path.join(OUT, "levels.tex"), "w").write("\n".join(rows) + "\n")
T = pd.read_csv(os.path.join(R, "tests.csv")); T = T[T.Metric == "best30"]
lab = {"H1 Q-IG-min - IG": "H1: Q-IG-min $-$ IG", "H1b Q-IG-min - IG-Rmin": "H1b: Q-IG-min $-$ IG-Rmin", "H2 Q-IG-G - IG-R": "H2: Q-IG-G $-$ IG-R",
       "H2b Q-IG-G - IG-RR": "H2b: Q-IG-G $-$ IG-RR", "H3 Q-IG-G - IG": "H3: Q-IG-G $-$ IG",
       "Q-IG-min cpi2 - IG-Rmin": "Q-IG-min (2 calls) $-$ IG-Rmin", "Q-IG-min cpi2 - IG": "Q-IG-min (2 calls) $-$ IG",
       "Q-IG-G cpi2 - IG-RR": "Q-IG-G (2 calls) $-$ IG-RR", "Q-IG-G cpi2 - IG": "Q-IG-G (2 calls) $-$ IG"}
def block(comps):
    out = []
    for c in comps:
        for s in ("TEST", "HARD"):
            r = T[(T.Comparison == c) & (T.Set == s)].iloc[0]
            out.append(" & ".join([lab[c] if s == "TEST" else "", s, s3(r.Diff_pp), f"[{s3(r.CI_low)}, {s3(r.CI_high)}]", f"{int(r.A_better)}/{int(r.B_better)}/{int(r.ties)}", pfmt(r.p_Holm)]) + r" \\")
    return out
main = block(["H1 Q-IG-min - IG", "H2 Q-IG-G - IG-R", "H3 Q-IG-G - IG"]) + [r"\midrule"] + block(["H1b Q-IG-min - IG-Rmin", "H2b Q-IG-G - IG-RR"])
open(os.path.join(OUT, "tests.tex"), "w").write("\n".join(main) + "\n")
open(os.path.join(OUT, "robust.tex"), "w").write("\n".join(block(["Q-IG-min cpi2 - IG-Rmin", "Q-IG-min cpi2 - IG", "Q-IG-G cpi2 - IG-RR", "Q-IG-G cpi2 - IG"])) + "\n")
# per n
rows = []
for ph, s in (("test", "TEST"), ("hard", "HARD")):
    D = gap_vs_bks(pd.read_csv(os.path.join(R, f"{ph}_runs.csv"))); D["n"] = D.Instance.str.extract(r"^(?:tai|VFR)(\d+)_")[0].astype(int)
    P = D.groupby(["n", "Arm", "Instance"]).Gap.min().groupby(["n", "Arm"]).mean().unstack(); cnt = D.groupby("n").Instance.nunique()
    for n in P.index: rows.append(" & ".join([s, str(n), str(cnt[n])] + [f3(P.loc[n, a]) for a in order[:6]]) + r" \\")
    rows.append(r"\midrule")
open(os.path.join(OUT, "per_n.tex"), "w").write("\n".join(rows[:-1]) + "\n")
# resources
Q = pd.read_csv(os.path.join(R, "oracle_resources_taillard.csv"))
fmt = lambda v: f"{v/1e6:.2f}\\,M" if v >= 1e6 else (f"{v/1e3:.0f}\\,k" if v >= 1e4 else f"{int(v):,}".replace(",", "\\,"))
rows = [" & ".join([f"{int(r.n)}$\\times${int(r.m)}", str(int(r.b)), f"{int(r.qubits):,}".replace(",", "\\,"), fmt(r.cx), fmt(r.ccx), fmt(r.cswap), fmt(r.cx_equiv_oracle), str(int(r.k_opt_M1)), fmt(r.cx_equiv_search_M1).replace("M", "M") if r.cx_equiv_search_M1 < 1e9 else f"{r.cx_equiv_search_M1/1e9:.2f}\\,G"]) + r" \\" for r in Q.itertuples()]
open(os.path.join(OUT, "resources.tex"), "w").write("\n".join(rows) + "\n")
# insertion cost / quality
C = pd.read_csv(os.path.join(R, "insertion_cost_quality.csv"))
nm = {"minimum quantique cq=0,5": "quantum minimum ($c_{\\min}=0.5$)", "jumeau par tirages cq=1": "random-search twin ($c=1$)", "balayage classique": "full scan"}
rows = [" & ".join([str(int(r.n)), nm[r.methode], f"{r.appels_moyens:.1f}", f"{r.rang_moyen:.2f}", f"{100*r.P_meilleure:.0f}", f"{100*r.P_top3:.0f}"]) + r" \\" for r in C.itertuples()]
open(os.path.join(OUT, "insertion.tex"), "w").write("\n".join(rows) + "\n")
# tuning
S = gap_vs_bks(pd.concat([pd.read_csv(os.path.join(R, f)) for f in ("tune_runs.csv", "tune2_runs.csv", "tune3_runs.csv")])).groupby(["Arm", "Instance"]).Gap.min().groupby("Arm").mean()
fams = [("Q-IG-min", "Q-IG-min c"), ("Q-IG-G", "Q-IG-G c"), ("IG-R", "IG-R l"), ("IG-Rmin", "IG-Rmin r"), ("IG-RR", "IG-RR r")]
rows = [f"IG & --- & {f3(S['IG'])} & 1 \\\\"]
for f, pre in fams:
    s = S[[a for a in S.index if a.startswith(pre)]]; best = s.idxmin()
    parts = best.replace(f, "").split(); txt = []
    for t in parts:
        if t[0] in "cr": txt.append(("$c_{\\min}=" if t[0] == "c" else "$c=") + t[1:] + "$")
        elif t[0] == "l": txt.append("$c_{ls}=" + t[1:] + "$")
    rows.append(f"{f} & {', '.join(txt)} & {f3(s.min())} & {len(s)} \\\\")
open(os.path.join(OUT, "tuning.tex"), "w").write("\n".join(rows) + "\n")
print(open(os.path.join(OUT, "tuning.tex")).read())
# long budgets (exploratory)
LV = pd.read_csv(os.path.join(R, "long_levels.csv")).set_index(["Mult", "Arm"]); LT = pd.read_csv(os.path.join(R, "long_tests.csv"))
bud = {1: "46{,}080", 10: "460{,}800", 100: "4{,}608{,}000"}
def tr(a, ka, b, kb):
    r = LT[(LT.A == a) & (LT.MultA == ka) & (LT.B == b) & (LT.MultB == kb)].iloc[0]
    return f"{s3(r.Diff_pp)} ({int(r.A_better)}/{int(r.B_better)}/{int(r.ties)}, {pfmt(r.p)})"
rows = [" & ".join([bud[k]] + [f"{f2(LV.loc[(k, a), 'Gap'])} $\\pm$ {f2(LV.loc[(k, a), 'CI95'])}" for a in ("IG", "IG-RR", "Q-IG-G")]
                   + [tr("Q-IG-G", k, "IG-RR", k), tr("Q-IG-G", k, "IG", k)]) + r" \\" for k in (1, 10, 100)]
open(os.path.join(OUT, "long.tex"), "w").write("\n".join(rows) + "\n")
