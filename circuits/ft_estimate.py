"""Fault-tolerant resource estimates for the Q-IG oracle with the Azure Quantum Resource Estimator (Beverland et al. 2022),
run locally through the `qdk` Python package (pip install qdk). Inputs: results/oracle_resources_taillard.csv (logical
qubits, Toffoli = CCX and Fredkin = CSWAP counts of one oracle call). A Fredkin is one Toffoli plus two CNOTs, so the
non-Clifford cost of one call is ccx + cswap CCZ gates; Clifford gates are not counted in the logical depth (estimator
model, sequential CCZ gates). One Grover iteration = oracle + inverse oracle + diffusion (a multi-controlled Z on the move
register, about 2q Toffolis for q = 2w register qubits). The error budget of a Grover iteration is set to the
probability eps that the iteration is corrupted (Section 7.5 of the paper), so the estimates give the cost of running at
a given eps. A Q-IG-G run charges 46,080 calls, i.e. at most 46,080 Grover iterations executed one after the other.
Usage: python circuits/ft_estimate.py  ->  results/ft_resources.csv"""
import os, math, pandas as pd
from qdk.estimator import LogicalCounts
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
O = pd.read_csv(os.path.join(R, "oracle_resources_taillard.csv"))
MODELS = ["qubit_gate_ns_e3", "qubit_gate_ns_e4", "qubit_gate_us_e4", "qubit_maj_ns_e4"]
EPS = [0.3, 0.1, 0.01]; RUN_ITER = 46080
rows = []
for r in O.itertuples():
    ccz_call = int(r.ccx + r.cswap); q = 2 * math.ceil(math.log2(r.n)); ccz_iter = 2 * ccz_call + 2 * q
    for model in MODELS:
        for eps in EPS:
            for level, ccz, budget in (("oracle call", ccz_call, eps / 2), ("Grover iteration", ccz_iter, eps)):
                params = {"qubitParams": {"name": model}, "errorBudget": budget}
                if model.startswith("qubit_maj"): params["qecScheme"] = {"name": "floquet_code"}
                e = LogicalCounts({"numQubits": int(r.qubits), "cczCount": int(ccz), "measurementCount": 0}).estimate(params)
                pc, lq = e["physicalCounts"], e["logicalQubit"]
                rows.append(dict(instance=r.instance, n=r.n, m=r.m, logical_qubits=int(r.qubits), ccz=int(ccz), model=model, eps=eps, level=level,
                                 error_budget=budget, layout_logical_qubits=pc["breakdown"]["algorithmicLogicalQubits"],
                                 code_distance=lq["codeDistance"], physical_qubits=pc["physicalQubits"], t_factories=pc["breakdown"]["numTfactories"],
                                 logical_cycle_ns=lq["logicalCycleTime"], runtime_s=pc["runtime"] * 1e-9,
                                 run_runtime_h=(pc["runtime"] * 1e-9 * RUN_ITER / 3600.0) if level == "Grover iteration" else float("nan")))
D = pd.DataFrame(rows); D.to_csv(os.path.join(R, "ft_resources.csv"), index=False)
print(D[(D.level == "Grover iteration") & (D.eps == 0.1)][["instance", "model", "code_distance", "physical_qubits", "runtime_s", "run_runtime_h"]].to_string(index=False))
