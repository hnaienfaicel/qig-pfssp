"""Cost and exactness of the emulated quantum minimum (Durr-Hoyer + BBHT, register 2^ceil(log2 n)) against a full scan.
Distinct random values, 20,000 searches per cell. Writes results/quantum_minimum_calls_vs_n.csv."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qig.quantum import quantum_argmin
rng = np.random.default_rng(2026); rows = []
for n in (20, 50, 100, 200, 500):
    for cq in (0.5, 1, 2, 3):
        ch, ok = [], []
        for _ in range(20000):
            c = rng.permutation(n).astype(np.int64); y, v, k = quantum_argmin(c, rng, cq); ch.append(k); ok.append(v == 0)
        rows.append(dict(n=n, c_min=cq, mean_calls=np.mean(ch), P_exact_minimum=np.mean(ok), full_scan_calls=n, call_ratio_scan_over_quantum=n / np.mean(ch)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "quantum_minimum_calls_vs_n.csv"), index=False)
