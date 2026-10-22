"""Tables of the optimised oracle (v2) for the paper: Table 1 (logical resources on the 8 sizes, noiseless machine and
break-even time), Table 2 (fault-tolerant cost of one Grover iteration), and the v2 table of the detailed-results report. Inputs:
results/oracle_v2_resources.csv and results/ft_v2_resources.csv (circuits/count_v2.py, circuits/ft_v2.py), the IG runs
of the campaign for the classical run time. Usage: python figures/make_oracle_v2.py"""
import os, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results"); OUT = os.path.join(HERE, "..", "paper", "tables")
O = pd.read_csv(os.path.join(R, "oracle_v2_resources.csv")); F = pd.read_csv(os.path.join(R, "ft_v2_resources.csv"))
SIZES = ["tai20_5_0", "tai20_20_0", "tai50_5_0", "tai50_10_0", "tai50_20_0", "tai100_5_0", "tai100_10_0", "tai100_20_0"]
th = lambda v: f"{int(round(v)):,}".replace(",", "\\,")
def k(v):
    if v >= 1e6: return f"{v/1e6:.2f}\\,M"
    if v >= 1e4: return f"{v/1e3:.0f}\\,k"
    return th(v)
KOPT = pd.read_csv(os.path.join(R, "oracle_resources_taillard.csv")).set_index("instance").k_opt_M1
lab = lambda r: f"{int(r.n)}$\\times${int(r.m)}"
g = lambda inst, var: O[(O.instance == inst) & (O.variant == var)].iloc[0]
# Table 1: optimised oracle, one call and one Grover iteration, iterations for the emulated move encoding,
# noiseless machine (1 ns per layer) and break-even time (k* of IG times the classical time per evaluation)
import math
rows = []
for i in SIZES:
    b = g(i, "v2"); n = int(b.n); wi = math.ceil(math.log2(n)); wm = math.ceil(math.log2(n - 1))
    kit = int(math.floor(math.pi / 4 * math.sqrt(2 ** (wi + 1 + wm))))
    rows.append(" & ".join([lab(b), str(int(b.b)), th(b.qubits_iteration), k(b.oracle_toffoli), k(b.iter_toffoli), k(b.iter_depth),
                            th(b.iter_toffoli_depth), str(kit), f"{b.noiseless_1ns_us:.1f}", f"{b.breakeven_IG_us:.1f}"]) + r" \\")
open(os.path.join(OUT, "resources.tex"), "w").write("\n".join(rows) + "\n")
# Table 2: fault-tolerant cost of one Grover iteration
RUNS = pd.concat([pd.read_csv(os.path.join(R, f)) for f in ("test_runs.csv", "hard_runs.csv", "ext_runs.csv", "tune_runs.csv")]); RUNS = RUNS[RUNS.Arm == "IG"]
mname = {"qubit_gate_ns_e3": "gate, ns, $10^{-3}$", "qubit_gate_ns_e4": "gate, ns, $10^{-4}$", "qubit_maj_ns_e4": "Majorana, ns, $10^{-4}$"}
def dur(s):
    if s < 1e-3: return f"{s*1e6:.0f}\\,\\textmu s"
    if s < 1: return f"{s*1e3:.2g}\\,ms" if s < 1e-2 else f"{s*1e3:.0f}\\,ms"
    if s < 60: return f"{s:.2g}\\,s" if s < 10 else f"{s:.0f}\\,s"
    if s < 3600: return f"{s/60:.0f}\\,min"
    if s < 86400 * 2: return f"{s/3600:.1f}\\,h"
    return f"{s/86400:.1f}\\,d" if s < 86400 * 10 else f"{s/86400:.0f}\\,d"
rows = []
for inst in ("tai20_5_0", "tai50_10_0", "tai100_20_0"):
    grp = inst.rsplit("_", 1)[0]; ig = RUNS[RUNS.Instance.str.startswith(grp + "_")].sec.median()
    for j, model in enumerate(("qubit_gate_ns_e3", "qubit_gate_ns_e4", "qubit_maj_ns_e4")):
        f2 = F[(F.instance == inst) & (F.variant == "v2") & (F.model == model)].iloc[0]; be = g(inst, "v2").breakeven_IG_us * 1e-6
        rows.append(" & ".join([lab(f2) if j == 0 else "", mname[model], dur(f2.seq_runtime_s), dur(f2.dl_runtime_s),
                                f"{f2.seq_physical_qubits/1e6:.1f} / {f2.dl_physical_qubits_peak/1e6:.1f}\\,M", f"{dur(46080*f2.seq_runtime_s)} / {dur(46080*f2.dl_runtime_s)}",
                                f"{ig:.2f}\\,s" if j == 0 else "", f"{f2.dl_runtime_s/be:,.0f}".replace(",", "\\,")]) + r" \\")
    rows.append(r"\midrule")
open(os.path.join(OUT, "ft.tex"), "w").write("\n".join(rows[:-1]) + "\n")
# Table S13: one Grover iteration of the optimised oracle on the 8 sizes
rows = []
for i in SIZES:
    a, b = g(i, "v1"), g(i, "v2")
    rows.append(" & ".join([lab(b), th(b.qubits_iteration), k(b.iter_toffoli), k(b.iter_depth), th(b.iter_toffoli_depth), th(b.iter_peak_parallel_toffoli),
                            f"{a.iter_toffoli/b.iter_toffoli:.1f}", f"{a.iter_depth/b.iter_depth:.1f}", f"{a.iter_toffoli_depth/b.iter_toffoli_depth:.0f}",
                            f"{b.noiseless_1ns_us:.1f}", f"{b.noiseless_1ns_us/b.breakeven_IG_us:.2f}"]) + r" \\")
open(os.path.join(OUT, "resources_v2_iter.tex"), "w").write("\n".join(rows) + "\n")
print("done")
