"""Logical-qubit and gate counts of the full Grover oracle (coherent insertion + one-hot decoding + reversible makespan
+ threshold comparison) on Taillard instances. CX-equivalents count a Toffoli as 6 CX and a Fredkin as 8 CX."""
import os, sys, math, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grover_circuit as G
from qig.instances import load
rows = []
for name in ("tai20_5_0", "tai20_20_0", "tai50_5_0", "tai50_10_0", "tai50_20_0", "tai100_5_0", "tai100_10_0", "tai100_20_0"):
    p = load(name)[0]; n, m = p.shape; B = G.build(p, list(range(n)), int(p.sum()))
    c = B["O"].count_ops(); cx_eq = c.get("cx", 0) + 6 * c.get("ccx", 0) + 8 * c.get("cswap", 0)
    N = 1 << (2 * B["w"]); k = math.floor(math.pi / 4 * math.sqrt(N))
    rows.append(dict(instance=name, n=n, m=m, b=B["b"], qubits=B["O"].num_qubits, cx=c.get("cx", 0), ccx=c.get("ccx", 0), cswap=c.get("cswap", 0),
                     cx_equiv_oracle=cx_eq, codes_N=N, k_opt_M1=k, cx_equiv_search_M1=2 * k * cx_eq))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "oracle_resources_taillard.csv"), index=False)
