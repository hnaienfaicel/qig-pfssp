"""Exact Born probabilities (Aer MPS, gate by gate) after each Grover iteration, compared with sin^2((2k+1)theta). Usage: python grover_kscan.py n4|n8"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""Probabilites exactes de Born (Aer MPS, porte par porte) apres chaque iteration de Grover, k = 0..K."""
import time, numpy as np, pandas as pd, grover_circuit as G
from qig.flowshop import makespan as _mk
from qig.instances import load
from qiskit import transpile, QuantumCircuit
from qiskit_aer import AerSimulator
P = load("tai20_5_0")[0]
cas = {"n4": [(4, 3, (3, 1, 2, 0), 6), (4, 3, (1, 0, 2, 3), 6)], "n8": [(8, 2, (7, 0, 3, 6, 4, 2, 1, 5), 6)]}[sys.argv[1]]
sim = AerSimulator(method="matrix_product_state", mps_omp_threads=8); rows = []
for n, m, base, K in cas:
    p = P[:n, :m].copy(); cm = lambda x: int(_mk(p, np.asarray(x, dtype=np.int64))); thr = cm(base)
    B = G.build(p, list(base), thr); w = B["w"]; N = 1 << (2 * w)
    marked = [c for c in range(N) if cm(G.shift(base, *G.code_to_move(c, w))) < thr]; M = len(marked); th = np.arcsin(np.sqrt(M / N))
    qc = G.grover_circuit(B, list(base), 0, measure=False); qc.save_probabilities(qubits=B["S"], label="p0")
    Oi = B["O"].inverse()
    for k in range(1, K + 1):
        qc.compose(B["O"], inplace=True); qc.z(B["mark"][0]); qc.compose(Oi, inplace=True); qc.compose(B["D"], inplace=True)
        qc.save_probabilities(qubits=B["S"], label=f"p{k}")
    tq = qc                                   # deja en portes de base (x, h, z, cx, ccx, cswap)
    assert set(tq.count_ops()) <= set(G.BG) | {"save_probabilities"}, tq.count_ops()
    t = time.perf_counter(); res = sim.run(tq).result().data(0); dt = time.perf_counter() - t
    ops = B["O"].count_ops()
    for k in range(K + 1):
        pr = np.asarray(res[f"p{k}"]); ps = float(pr[marked].sum())
        unm = [c for c in range(N) if c not in marked]
        rows.append(dict(cas=f"n{n}m{m}", base=" ".join(map(str, base)), seuil=thr, N=N, M=M, k=k, P_succes=ps, P_theorie=float(np.sin((2 * k + 1) * th) ** 2),
                         ecart_max_marques=float(np.ptp(pr[marked])) if M > 1 else 0.0, ecart_max_non_marques=float(np.ptp(pr[unm])),
                         qubits=B["O"].num_qubits, cx_oracle=ops.get("cx", 0), ccx_oracle=ops.get("ccx", 0), cswap_oracle=ops.get("cswap", 0),
                         appels_oracle=2 * k, temps_total_s=round(dt, 1)))
        print(rows[-1]["cas"], M, k, round(ps, 5), round(rows[-1]["P_theorie"], 5), flush=True)
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"grover_kscan_{sys.argv[1]}.csv"), index=False)
