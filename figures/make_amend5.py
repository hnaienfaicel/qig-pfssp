"""Figures and LaTeX tables of the Amendment 5 studies (exploratory), from results/amend5_*.csv.
Usage: python figures/make_amend5.py   (after experiments/analyse_amend5.py and experiments/time_ig.py)"""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results"); OUT = os.path.join(HERE, "..", "paper", "tables")
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans", "axes.titlelocation": "left"})
FOC, TW, REF = "#1f5fa8", "#8a8a8a", "#3a3a3a"
def letter(ax, s): ax.text(-0.02, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="bottom")
f3 = lambda v: f"{v:.3f}"; s3 = lambda v: f"${v:.3f}$"
def pfmt(p):
    if p < 1e-4: m, e = f"{p:.1e}".split("e"); return f"${m}\\times 10^{{{int(e)}}}$"
    return f"{p:.4f}"
B0 = 46080

# ---------------------------------------------------------------- ablation table
if os.path.exists(os.path.join(R, "amend5_abl_tests.csv")):
    L = pd.read_csv(os.path.join(R, "amend5_abl_levels.csv")); T = pd.read_csv(os.path.join(R, "amend5_abl_tests.csv")); T = T[T.Metric == "best30"]
    rows = []
    for a, lab in (("IG", "IG: full-scan construction, IG local search"), ("IG-R", "IG-R: full-scan construction, random descent (twin)"),
                   ("Q-IG-LS", "\\textbf{Q-IG-LS}: full-scan construction, Grover descent"), ("Q-IG-G", "\\textbf{Q-IG-G}: quantum minimum, Grover descent")):
        rows.append(" & ".join([lab] + [f3(L[(L.Set == s) & (L.Arm == a)].best30.iloc[0]) for s in ("TEST", "HARD", "EXT")]) + r" \\")
    open(os.path.join(OUT, "ablation_levels.tex"), "w").write("\n".join(rows) + "\n")
    lab = {"Q-IG-LS - IG-R": "Q-IG-LS $-$ IG-R (interference in the descent)", "Q-IG-LS - IG": "Q-IG-LS $-$ IG (Grover descent)",
           "Q-IG-G - Q-IG-LS": "Q-IG-G $-$ Q-IG-LS (adding the quantum minimum)", "Q-IG-G - IG": "Q-IG-G $-$ IG (both)"}
    rows = []
    for c in lab:
        for s in ("TEST", "HARD", "EXT"):
            r = T[(T.Comparison == c) & (T.Set == s)].iloc[0]
            rows.append(" & ".join([lab[c] if s == "TEST" else "", s, s3(r.Diff_pp), f"[{s3(r.CI_low)}, {s3(r.CI_high)}]",
                                    f"{int(r.A_better)}/{int(r.B_better)}/{int(r.ties)}", pfmt(r.p_Holm)]) + r" \\")
        rows.append(r"\midrule")
    open(os.path.join(OUT, "ablation_tests.tex"), "w").write("\n".join(rows[:-1]) + "\n")

# ---------------------------------------------------------------- settings grids: figure and table
if os.path.exists(os.path.join(R, "amend5_grid_levels.csv")):
    GL = pd.read_csv(os.path.join(R, "amend5_grid_levels.csv")); GT = pd.read_csv(os.path.join(R, "amend5_grid_tests.csv"))
    fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.0), gridspec_kw=dict(width_ratios=[4, 4], wspace=0.35, hspace=0.62))
    for i, s in enumerate(("TEST", "HARD")):
        g = GL[GL.Set == s]; vmin, vmax = g[g.Family.isin(["Q-IG-G", "IG-RR"])].best10.min(), g[g.Family.isin(["Q-IG-G", "IG-RR"])].best10.max()
        for j, (fam, rk, rl) in enumerate((("Q-IG-G", "c_min", "$c_{\\min}$ (quantum minimum)"), ("IG-RR", "r_min", "$c$ (random minimum)"))):
            ax = axs[i, j]; x = g[g.Family == fam].pivot(index=rk, columns="c_ls", values="best10").sort_index()
            im = ax.imshow(x.values, cmap="cividis", vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
            for (a, b), v in np.ndenumerate(x.values):
                ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v < (vmin + vmax) / 2 else "black")
            tu = g[(g.Family == fam) & g.tuned].iloc[0]
            ax.add_patch(plt.Rectangle((list(x.columns).index(tu.c_ls) - 0.5, list(x.index).index(tu[rk]) - 0.5), 1, 1, fill=False, ec="#d62728", lw=1.6))
            ax.set_xticks(range(len(x.columns))); ax.set_xticklabels([f"{v:g}" for v in x.columns])
            ax.set_yticks(range(len(x.index))); ax.set_yticklabels([f"{v:g}" for v in x.index])
            ax.set_xlabel("$c_{ls}$ (descent stopping constant)"); ax.set_ylabel(rl)
            ax.spines[["top", "right"]].set_visible(True)
            ax.set_title(f"{s}: {fam}{' (twin)' if fam == 'IG-RR' else ''}, {g[g.Family == fam].shape[0]} settings")
            letter(ax, "abcd"[2 * i + j])
        cb = fig.colorbar(im, ax=axs[i, :].tolist(), fraction=0.025, pad=0.02); cb.set_label("Mean best-of-10 gap (%)", fontsize=7); cb.ax.tick_params(labelsize=6)
    fig.savefig(os.path.join(HERE, "fig_grid.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_grid.png"), dpi=300, bbox_inches="tight")
    rows = []
    for q, t in (("Q-IG-G", "IG-RR"), ("Q-IG-min", "IG-Rmin")):
        for s in ("TEST", "HARD"):
            g = GL[GL.Set == s]; rq, rt = g[g.Family == q].best10, g[g.Family == t].best10
            r = GT[(GT.Comparison == f"worst {q} - best {t}") & (GT.Set == s) & (GT.Metric == "best10")].iloc[0]
            rows.append(" & ".join([f"{q} vs {t}" if s == "TEST" else "", s, f"{len(rq)}/{len(rt)}", f"{rq.min():.3f}--{rq.max():.3f}", f"{rt.min():.3f}--{rt.max():.3f}",
                                    s3(r.Diff_pp), f"[{s3(r.CI_low)}, {s3(r.CI_high)}]", f"{int(r.A_better)}/{int(r.B_better)}/{int(r.ties)}", pfmt(r.p_Holm)]) + r" \\")
        rows.append(r"\midrule")
    open(os.path.join(OUT, "grid.tex"), "w").write("\n".join(rows[:-1]) + "\n")

# ---------------------------------------------------------------- budget equivalence: figure and table
if os.path.exists(os.path.join(R, "amend5_equiv.csv")):
    EQ = pd.read_csv(os.path.join(R, "amend5_equiv.csv")).set_index("Arm"); CU = pd.read_csv(os.path.join(R, "amend5_equiv_curve.csv"))
    FT = pd.read_csv(os.path.join(R, "ft_resources.csv")); FT = FT[(FT.eps == 0.1) & (FT.level == "Grover iteration")]
    TM = pd.read_csv(os.path.join(R, "amend5_ig_timing.csv")) if os.path.exists(os.path.join(R, "amend5_ig_timing.csv")) else None
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw=dict(width_ratios=[1.2, 1], wspace=0.38))
    q = CU[CU.Arm == "Q-IG-G"].iloc[0]
    a1.axhspan(q.Gap - q.CI95, q.Gap + q.CI95, color=FOC, alpha=0.12, lw=0); a1.axhline(q.Gap, color=FOC, lw=1.6, label="Q-IG-G at 46,080 calls")
    for arm, col, mk, lab, ky in (("IG", REF, "s", "IG (full scan)", 0.55), ("IG-RR", TW, "o", "IG-RR (twin)", 0.15)):
        x = CU[CU.Arm == arm].sort_values("Mult")
        a1.fill_between(x.Mult, x.Gap - x.CI95, x.Gap + x.CI95, color=col, alpha=0.12, lw=0)
        a1.plot(x.Mult, x.Gap, "-" + mk, color=col, ms=3.5, lw=1.2, mfc="white" if arm == "IG-RR" else col, label=lab)
        k = EQ.loc[arm, "k_star"]; a1.plot([k, k], [0, q.Gap], ls=":", color=col, lw=1.0)
        a1.text(k * 1.08, ky, f"$k^*$ = {k:.1f}", color=col, fontsize=6.5, ha="left", va="bottom")
    a1.legend(frameon=False, fontsize=6.5, loc="upper right")
    a1.set_xscale("log"); a1.set_xticks([1, 10, 100]); a1.set_xticklabels(["1", "10", "100"]); a1.set_xlim(0.8, 130); a1.set_ylim(0, 4.9)
    a1.set_xlabel("Classical budget (multiples of 46,080 calls)"); a1.set_ylabel("Best-of-5 gap to BKS (%)")
    a1.set_title("Classical calls needed to match Q-IG-G"); letter(a1, "a")
    if TM is not None:
        tc = TM.groupby("Instance").sec_per_call.median(); kk = EQ.loc["IG", "k_star"]
        need = (kk * tc).rename("need")
        inst = [i for i in ("tai20_5_0", "tai50_10_0", "tai100_20_0") if i in tc.index]
        xs = np.arange(len(inst))
        for dx, model, col, lab in ((-0.18, "qubit_gate_ns_e3", "#7fa6d6", "superconducting-like (ns, $10^{-3}$)"), (0.18, "qubit_maj_ns_e4", FOC, "Majorana-like (ns, $10^{-4}$)")):
            v = [FT[(FT.instance == i) & (FT.model == model)].runtime_s.iloc[0] for i in inst]
            a2.bar(xs + dx, v, width=0.34, color=col, label=lab)
        a2.scatter(xs, need[inst].values, marker="_", s=300, color="#d62728", lw=2, zorder=3, label="break-even ($k^*$ classical calls)")
        a2.set_yscale("log"); a2.set_xticks(xs); a2.set_xticklabels([i.replace("tai", "").replace("_0", "").replace("_", "\u00d7") for i in inst])
        a2.set_ylabel("Time per Grover iteration (s)"); a2.set_xlabel("Taillard instance size (n\u00d7m)")
        a2.set_title("Required versus estimated iteration time"); a2.set_ylim(1e-6, 1e4)
        a2.legend(frameon=False, fontsize=6, loc="upper left", ncol=1)
        letter(a2, "b")
        def sci(v):
            m, e = f"{v:.1e}".split("e"); return f"{m}\\times10^{{{int(e)}}}"
        rows = []
        for i in sorted(tc.index, key=lambda s: (int(s.split("_")[0][3:]), int(s.split("_")[1]))):
            e3 = FT[(FT.instance == i) & (FT.model == "qubit_gate_ns_e3")].runtime_s.iloc[0]; e4 = FT[(FT.instance == i) & (FT.model == "qubit_maj_ns_e4")].runtime_s.iloc[0]
            rows.append(" & ".join([i.replace("_", "\\_"), f"{tc[i] * 1e6:.2f}", f"{need[i] * 1e6:.1f}", f"{e4:.2f}--{e3:.2f}", f"${sci(e4 / need[i])}$--${sci(e3 / need[i])}$"]) + r" \\")
        open(os.path.join(OUT, "equiv_hw.tex"), "w").write("\n".join(rows) + "\n")
    fig.savefig(os.path.join(HERE, "fig_equiv.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_equiv.png"), dpi=300, bbox_inches="tight")

# ---------------------------------------------------------------- per-group tables (appendix)
if os.path.exists(os.path.join(R, "amend5_per_group.csv")):
    PG = pd.read_csv(os.path.join(R, "amend5_per_group.csv")); arms = [a for a in ["IG", "IG-Rmin", "Q-IG-min", "IG-R", "IG-RR", "Q-IG-LS", "Q-IG-G"] if a in PG.Arm.unique()]
    for met in ("best30", "mean30"):
        rows = []
        for s in ("TEST", "HARD", "EXT"):
            X = PG[PG.Set == s].pivot_table(index=["Group", "n", "m", "Instances"], columns="Arm", values=met).reset_index()
            X["fam"] = X.Group.str.startswith("tai").map({True: 0, False: 1}); X = X.sort_values(["fam", "n", "m"])
            for k, (_, r) in enumerate(X.iterrows()):
                vals = [float(r[a]) for a in arms]; best = min(vals)
                cells = [f"\\textbf{{{v:.2f}}}" if v == best else f"{v:.2f}" for v in vals]
                rows.append(" & ".join([s if k == 0 else "", r["Group"].replace("_", "\\_"), str(int(r["Instances"]))] + cells) + r" \\")
            rows.append(r"\midrule")
        open(os.path.join(OUT, f"pergroup_{met}.tex"), "w").write("\n".join(rows[:-1]) + "\n")
# ---------------------------------------------------------------- comparison with Farhani et al. (2024), descriptive
if os.path.exists(os.path.join(R, "amend5_farhani_groups.csv")):
    FG = pd.read_csv(os.path.join(R, "amend5_farhani_groups.csv"), index_col=0)
    order = ["20_5", "20_10", "20_20", "50_5", "50_10", "50_20", "all"]; rows = []
    for g in order:
        r = FG.loc[g]; lab = "All 60" if g == "all" else g.replace("_", "$\\times$")
        if g == "all": rows.append(r"\midrule")
        rows.append(" & ".join([lab, f"{int(r.instances)}", f"{r.NEH:.2f}", f"{r.IG_mean:.2f}", f"\\textbf{{{r.QIGG_mean:.2f}}}", f"{r.QUBO_position:.2f}", f"{r.QUBO_best6:.2f}"]) + r" \\")
    open(os.path.join(OUT, "farhani.tex"), "w").write("\n".join(rows) + "\n")
print("done")
