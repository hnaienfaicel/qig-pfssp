"""Time per Grover iteration of oracles v1, v2 and v2-double under three models (counts from
results/oracle_v2_counts_raw.csv, produced by circuits/count_v2.py).

(a) Sequential fault-tolerant estimate (upper estimate), Microsoft/Azure Quantum Resource Estimator run locally
    through the `qdk` package, exactly as circuits/ft_estimate.py: LogicalCounts(numQubits, cczCount,
    measurementCount), error budget 0.1 per Grover iteration, qubit models qubit_gate_ns_e3, qubit_gate_ns_e4 and
    qubit_maj_ns_e4 (floquet code), i.e. the three models of the paper's FT table.  The estimator executes the CCZ
    gates one after the other (3 logical cycles each) and charges 1 logical cycle per measurement.
      v1 : paper convention (numQubits = oracle qubits, cczCount = 2 (ccx + cswap) + 2q, no measurement), which
           reproduces results/ft_resources.csv;
      v2 : exact iteration counts: numQubits = qubits incl. diffusion ancillas, cczCount = Toffoli-like gates of
           oracle + inverse + diffusion (each Gidney AND = one CCZ-type magic state), measurementCount = number of
           measurement-based uncomputations.
(b) Depth-limited estimate (lower estimate).  Assumptions: same code distance d, logical cycle time tau, physical
    qubits per logical qubit and magic-state factory design as in (a) for the same model; Clifford gates and the
    measurement-based uncomputations are absorbed (zero time); each layer of the Toffoli-depth schedule takes ONE
    logical cycle, which requires one magic CCZ state (4 T states in the estimator's factories) per Toffoli of the
    layer to be ready; enough factories run in parallel for the peak layer:
        F_peak = ceil(4 W_peak t_factory / (s tau))      W_peak = max Toffolis in one layer, s = T states per run
    and, for reference, F_mean with the mean parallelism W_mean = Toffolis / Toffoli depth.  Runtime
        T_low = D_T * tau      (D_T = Toffoli depth of one iteration)
    Physical qubits = (algorithmic logical qubits of (a)) * (physical qubits per logical qubit) + F * factory qubits.
    The routing space for parallel lattice-surgery operations is assumed to fit in the layout of (a) (optimistic).
    Also reported: T_all = D * tau with every gate layer (Clifford included) taking one logical cycle.
(c) Noiseless machine: T = D * t_layer with t_layer = 1, 10, 100 ns (D = depth of one iteration, every gate one layer).
Break-even time per Grover iteration (paper): k* x (median measured time per classical IG call), k* = 15.61 (IG) and
9.13 (IG-RR) from results/amend5_equiv.csv, time per call from results/amend5_ig_timing.csv (same machine).
Usage: python circuits/ft_v2.py -> results/ft_v2_resources.csv, results/oracle_v2_resources.csv"""
import math, os
import numpy as np, pandas as pd
from qdk.estimator import LogicalCounts

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
C = pd.read_csv(os.path.join(R, "oracle_v2_counts_raw.csv"))
MODELS = ["qubit_gate_ns_e3", "qubit_gate_ns_e4", "qubit_maj_ns_e4"]
FT_INST = ["tai20_5_0", "tai20_20_0", "tai50_5_0", "tai50_10_0", "tai50_20_0", "tai100_5_0", "tai100_10_0", "tai100_20_0"]; EPS = 0.1
eq = pd.read_csv(os.path.join(R, "amend5_equiv.csv")).set_index("Arm").k_star
tim = pd.read_csv(os.path.join(R, "amend5_ig_timing.csv")).groupby("Instance").sec_per_call.median()


def qdk_est(nq, ccz, meas, model):
    params = {"qubitParams": {"name": model}, "errorBudget": EPS}
    if model.startswith("qubit_maj"): params["qecScheme"] = {"name": "floquet_code"}
    return LogicalCounts({"numQubits": int(nq), "cczCount": int(ccz), "measurementCount": int(meas)}).estimate(params)


rows = []
for r in C[C.instance.isin(FT_INST) & C.variant.isin(["v1", "v2", "v2-double"])].itertuples():
    if r.variant == "v1":
        nq, ccz, meas = r.qubits_oracle, 2 * (r.oracle_ccx + r.oracle_cswap) + 2 * r.move_qubits, 0
    else:
        nq, ccz, meas = r.qubits_iteration, r.iter_toffoli, r.iter_unand
    for model in MODELS:
        e = qdk_est(nq, ccz, meas, model); pc, lq, tf = e["physicalCounts"], e["logicalQubit"], e["tfactory"]
        tau = lq["logicalCycleTime"]; d = lq["codeDistance"]; ppl = lq["physicalQubits"]
        lay = pc["breakdown"]["algorithmicLogicalQubits"]; s = tf["numTstates"]; tF = tf["runtime"]; qF = tf["physicalQubits"]
        Wp = r.iter_peak_parallel_toffoli; Wm = r.iter_mean_parallel_toffoli
        Fp = math.ceil(4 * Wp * tF / (s * tau)); Fm = math.ceil(4 * Wm * tF / (s * tau))
        rows.append(dict(instance=r.instance, n=r.n, m=r.m, variant=r.variant, model=model, error_budget=EPS,
                         qdk_numQubits=int(nq), qdk_cczCount=int(ccz), qdk_measurementCount=int(meas),
                         layout_logical_qubits=lay, code_distance=d, logical_cycle_ns=tau, phys_per_logical=ppl,
                         seq_logical_depth=pc["breakdown"]["logicalDepth"], seq_t_factories=pc["breakdown"]["numTfactories"],
                         seq_physical_qubits=pc["physicalQubits"], seq_runtime_s=pc["runtime"] * 1e-9,
                         factory_physical_qubits=qF, factory_runtime_ns=tF, factory_T_per_run=s,
                         toffoli_depth=r.iter_toffoli_depth, depth=r.iter_depth, W_peak=Wp, W_mean=Wm,
                         dl_runtime_s=r.iter_toffoli_depth * tau * 1e-9, dl_factories_peak=Fp,
                         dl_physical_qubits_peak=lay * ppl + Fp * qF, dl_factories_mean=Fm,
                         dl_physical_qubits_mean=lay * ppl + Fm * qF, alllayers_runtime_s=r.iter_depth * tau * 1e-9))
FT = pd.DataFrame(rows)
# check v1 against the paper's table
old = pd.read_csv(os.path.join(R, "ft_resources.csv"))
old = old[(old.level == "Grover iteration") & (old.eps == EPS)].set_index(["instance", "model"])
v1 = FT[FT.variant == "v1"].set_index(["instance", "model"])
FT["paper_runtime_s"] = [old.runtime_s.get((a, b), np.nan) if v == "v1" else np.nan for a, b, v in zip(FT.instance, FT.model, FT.variant)]
FT.to_csv(os.path.join(R, "ft_v2_resources.csv"), index=False)

# ------------------------------------------------------------------ one row per instance x variant
out = C.copy()
out["breakeven_IG_us"] = [eq["IG"] * tim[i] * 1e6 for i in out.instance]
out["breakeven_IGRR_us"] = [eq["IG-RR"] * tim[i] * 1e6 for i in out.instance]
for t in (1, 10, 100):
    out[f"noiseless_{t}ns_us"] = out.iter_depth * t * 1e-3
for model in MODELS:
    sub = FT[FT.model == model].set_index(["instance", "variant"])
    key = model.replace("qubit_", "")
    out[f"ft_seq_{key}_s"] = [sub.seq_runtime_s.get((i, v), np.nan) for i, v in zip(out.instance, out.variant)]
    out[f"ft_dl_{key}_s"] = [sub.dl_runtime_s.get((i, v), np.nan) for i, v in zip(out.instance, out.variant)]
    out[f"ft_dl_{key}_physical_qubits"] = [sub.dl_physical_qubits_peak.get((i, v), np.nan) for i, v in zip(out.instance, out.variant)]
cols = ["instance", "n", "m", "variant", "w", "move_qubits", "codes_N", "valid_codes", "b", "bp", "qubits_oracle",
        "qubits_iteration", "oracle_toffoli", "oracle_and", "oracle_ccx", "oracle_cswap", "oracle_unand", "oracle_cx",
        "oracle_t_count", "oracle_cx_equiv", "oracle_depth", "oracle_toffoli_depth", "iter_toffoli", "iter_t_count",
        "iter_cx", "iter_unand", "iter_cx_equiv", "iter_depth", "iter_toffoli_depth", "iter_peak_parallel_toffoli",
        "iter_mean_parallel_toffoli", "breakeven_IG_us", "breakeven_IGRR_us", "noiseless_1ns_us", "noiseless_10ns_us",
        "noiseless_100ns_us"] + [c for c in out.columns if c.startswith("ft_")]
out[cols].to_csv(os.path.join(R, "oracle_v2_resources.csv"), index=False)
pd.set_option("display.width", 250)
print(FT[["instance", "variant", "model", "code_distance", "logical_cycle_ns", "seq_runtime_s", "paper_runtime_s", "dl_runtime_s",
          "dl_factories_peak", "dl_physical_qubits_peak", "seq_physical_qubits", "alllayers_runtime_s"]].to_string(index=False))
