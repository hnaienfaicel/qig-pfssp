"""Depth of the Grover oracle and of one Grover iteration (oracle, phase flip, inverse oracle, diffusion) on the first
instance of each Taillard group of Table 1. Depth counts gate layers, a Toffoli or a Fredkin counting as one layer.
Used for the noiseless-machine bound of Section 5.2 and Table S9. Usage: python circuits/depth.py"""
import os, sys, math, time, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grover_circuit as G
from qiskit import QuantumCircuit
from qig.instances import load
rows = []
for name in ("tai20_5_0", "tai20_20_0", "tai50_5_0", "tai50_10_0", "tai50_20_0", "tai100_5_0", "tai100_10_0", "tai100_20_0"):
    t0 = time.time(); p = load(name)[0]; n, m = p.shape; B = G.build(p, list(range(n)), int(p.sum()))
    it = QuantumCircuit(*B["regs"]); it.compose(B["O"], inplace=True); it.z(B["mark"][0]); it.compose(B["O"].inverse(), inplace=True); it.compose(B["D"], inplace=True)
    rows.append(dict(instance=name, n=n, m=m, qubits=B["O"].num_qubits, depth_oracle=B["O"].depth(), depth_diffusion=B["D"].depth(), depth_iteration=it.depth()))
    print(rows[-1], f"{time.time() - t0:.0f}s", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "oracle_depth.csv"), index=False)
