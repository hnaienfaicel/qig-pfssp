"""Figures of the paper, from the CSV files in results/. Usage: python figures/make_figures.py"""
import os, sys, numpy as np, pandas as pd, matplotlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qig.instances import gap_vs_bks
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results")
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans", "axes.titlelocation": "left"})
FOC, TW, REF = "#1f5fa8", "#8a8a8a", "#3a3a3a"
def letter(ax, s): ax.text(-0.02, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="bottom")
def boot(x, B=10000, seed=0):
    r = np.random.default_rng(seed); x = np.asarray(x); m = x[r.integers(0, len(x), (B, len(x)))].mean(1); return np.percentile(m, [2.5, 97.5])

# ---------------------------------------------------------------- Fig. 2: gate-level validation of the Grover law
K = pd.read_csv(os.path.join(R, "grover_kscan_n4.csv")); N8 = pd.read_csv(os.path.join(R, "grover_n8_k1.csv")).iloc[0]
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for (M, g), mk in zip(K.groupby("M"), ("o", "s")):
    kk = np.linspace(0, 6, 300); th = np.arcsin(np.sqrt(M / 16))
    ax.plot(kk, np.sin((2 * kk + 1) * th) ** 2, color=FOC if M == 1 else TW, lw=1.0, zorder=1)
    ax.plot(g.k, g.P_succes, mk, ms=5, mfc="white", mec=FOC if M == 1 else TW, mew=1.3, zorder=3, label=f"{M} improving move{'s' if M > 1 else ''} out of 16")
ax.plot(1, N8.P_success_measured, "^", ms=5.5, color="#b5651d", zorder=4, label="8 jobs x 2 machines, 1 of 64 (1,000 shots)")
ax.plot(1, N8.P_theory, "_", ms=10, color="#b5651d", zorder=4)
ax.set_xlabel("Grover iterations k"); ax.set_ylabel("Probability of measuring an improving move")
ax.set_ylim(-0.03, 1.05); ax.set_xlim(-0.3, 6.3)
ax.set_title("Gate-level circuit (markers) matches sin\u00b2((2k+1)\u03b8) (lines)")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, handletextpad=0.3)
fig.savefig(os.path.join(HERE, "fig_validation.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_validation.png"), dpi=300, bbox_inches="tight")

# ---------------------------------------------------------------- Fig. 3: Q-IG against its classical twins
arms = [("IG", "IG (full scan)", REF), ("IG-Rmin", "IG-Rmin (twin)", TW), ("Q-IG-min", "Q-IG-min", FOC),
        ("IG-RR", "IG-RR (twin)", TW), ("IG-R", "IG-R (twin)", TW), ("Q-IG-G", "Q-IG-G", FOC)]
plt.rcParams.update({"xtick.labelsize": 7, "ytick.labelsize": 7})
fig = plt.figure(figsize=(6.0, 5.4)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.6, wspace=0.06)
a0 = fig.add_subplot(gs[0, 0]); a1 = fig.add_subplot(gs[0, 1], sharey=a0); a2 = fig.add_subplot(gs[1, :])
for ax, ph, lab, n, step in ((a0, "test", "TEST", 100, 0.2), (a1, "hard", "HARD", 40, 0.5)):
    D = gap_vs_bks(pd.read_csv(os.path.join(R, f"{ph}_runs.csv"))); B = D.groupby(["Arm", "Instance"]).Gap.min().unstack(0); assert B.shape[0] == n
    for k, (a, l, col) in enumerate(arms):
        mu = B[a].mean(); lo, hi = boot(B[a].values)
        ax.errorbar(mu, k, xerr=[[mu - lo], [hi - mu]], fmt="o", color=col, ecolor=col, ms=5, capsize=2, elinewidth=1.2, mfc=col if col != TW else "white", mew=1.2)
        if a in ("IG", "Q-IG-G"): ax.annotate(f"{mu:.2f}", (hi, k), xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=6, color=col)
    ax.set_title(f"{lab} ({n} instances)"); ax.set_xlabel("Best-of-30 gap to best-known solution (%)"); ax.xaxis.set_major_locator(MultipleLocator(step)); ax.margins(x=0.08)
a0.set_yticks(range(6)); a0.set_yticklabels([l for _, l, _ in arms]); a0.invert_yaxis()
for t, (_, _, col) in zip(a0.get_yticklabels(), arms): t.set_color(col if col != TW else "0.35")
plt.setp(a1.get_yticklabels(), visible=False)
T = pd.read_csv(os.path.join(R, "tests.csv")); T = T[T.Metric == "best30"]
comps = [("H1b Q-IG-min - IG-Rmin", "H1b: Q-IG-min − IG-Rmin"), ("H1 Q-IG-min - IG", "H1: Q-IG-min − IG"),
         ("H2b Q-IG-G - IG-RR", "H2b: Q-IG-G − IG-RR"), ("H2 Q-IG-G - IG-R", "H2: Q-IG-G − IG-R"), ("H3 Q-IG-G - IG", "H3: Q-IG-G − IG")]
for y, (c, l) in enumerate(comps):
    for ens, mk, dy in (("TEST", "o", -0.17), ("HARD", "s", 0.17)):
        r = T[(T.Comparison == c) & (T.Set == ens)].iloc[0]
        a2.errorbar(r.Diff_pp, y + dy, xerr=[[r.Diff_pp - r.CI_low], [r.CI_high - r.Diff_pp]], fmt=mk, color=FOC, ecolor=FOC, ms=4.5, capsize=2, elinewidth=1.2,
                    mfc=FOC if ens == "TEST" else "white", mew=1.2, label=ens if y == 0 else None)
a2.axvline(0, color="0.35", lw=0.9); a2.set_yticks(range(len(comps))); a2.set_yticklabels([l for _, l in comps]); a2.invert_yaxis(); a2.xaxis.set_major_locator(MultipleLocator(0.2))
a2.set_xlabel("Paired difference in best-of-30 gap (points, 95% CI); negative = quantum better")
pmax = T[T.Family.isin(["primary", "secondary"])].p_Holm.max()
a2.set_title(f"Each quantum subroutine beats its twin at equal oracle calls (p_Holm \u2264 {pmax:.1e})")
a2.legend(loc="upper left", frameon=False, handletextpad=0.3); a2.margins(x=0.05)
letter(a0, "a"); letter(a1, "b"); letter(a2, "c")
fig.savefig(os.path.join(HERE, "fig_results.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_results.png"), dpi=300, bbox_inches="tight")
print("pmax", pmax)

# ---------------------------------------------------------------- Fig. 6: larger budgets on the open instances (exploratory)
LV = pd.read_csv(os.path.join(R, "long_levels.csv")); LT = pd.read_csv(os.path.join(R, "long_tests.csv"))
plt.rcParams.update({"xtick.labelsize": 6, "ytick.labelsize": 6})
fig, (b1, b2) = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.42))
style = {"IG": (REF, "IG (full scan)", 1.2, 0.22), "IG-RR": (TW, "IG-RR (twin)", 1.2, -0.22), "Q-IG-G": (FOC, "Q-IG-G", 2.0, 0.0)}
for a, (col, lab, lw, dy) in style.items():
    x = LV[LV.Arm == a].sort_values("Mult"); xb = x.Budget.values
    b1.fill_between(xb, x.Gap - x.CI95, x.Gap + x.CI95, color=col, alpha=0.14, lw=0)
    b1.plot(xb, x.Gap, "-o", color=col, lw=lw, ms=4, zorder=3 if a == "Q-IG-G" else 2)
    b1.text(xb[-1] * 1.3, x.Gap.values[-1] + dy, lab, color=col, va="center", fontsize=7)
b1.axhline(0, color="0.2", lw=0.8, ls="--"); b1.text(46080 * 1.05, 0.1, "best-known solution", fontsize=6, va="bottom")
b1.set_xscale("log"); b1.set_xticks([46080, 460800, 4608000]); b1.set_xticklabels(["46k", "461k", "4.6M"])
b1.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); b1.set_xlim(46080 / 1.6, 4608000 * 9); b1.set_ylim(-0.25, 4.9)
b1.set_xlabel("Oracle calls per run"); b1.set_ylabel("Best-of-5 gap to best-known solution (%)")
b1.set_title("No arm reaches the best-known solutions")
for b, mk, off in (("IG-RR", "o", -0.06), ("IG", "s", 0.06)):
    t = LT[(LT.A == "Q-IG-G") & (LT.B == b) & (LT.MultA == LT.MultB)].sort_values("MultA")
    xx = np.log10(t.MultA.values) + off
    b2.errorbar(xx, t.Diff_pp, yerr=[t.Diff_pp - t.CI_low, t.CI_high - t.Diff_pp], fmt=mk + "-", color=style[b][0], ms=4, lw=1.2, capsize=2,
                mfc=style[b][0] if b == "IG" else "white", mew=1.2)
    b2.text(2.2, t.Diff_pp.values[-1] + (0.09 if b == "IG-RR" else -0.09), f"vs {b}", color=style[b][0], fontsize=7, va="center")
b2.axhline(0, color="0.35", lw=0.9); b2.set_xticks([0, 1, 2]); b2.set_xticklabels(["46k", "461k", "4.6M"]); b2.set_xlim(-0.3, 2.75)
b2.set_ylim(-1.9, 0.15); b2.set_yticks([-1.5, -1.0, -0.5, 0.0]); b2.text(-0.25, 0.03, "negative = quantum better", fontsize=6, va="bottom")
b2.set_xlabel("Oracle calls per run (both arms)"); b2.set_ylabel("Paired difference (points, 95% CI)")
b2.set_title("The quantum advantage persists")
letter(b1, "a"); letter(b2, "b")
fig.savefig(os.path.join(HERE, "fig_budget.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_budget.png"), dpi=300, bbox_inches="tight")
