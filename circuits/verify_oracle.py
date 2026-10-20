"""Exhaustive check of the Grover oracle on every basis move (permutation, makespan and mark registers)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, grover_circuit as G, oracle_makespan as OM
from qig.flowshop import makespan as _mk
from qig.instances import load
from qiskit import transpile, ClassicalRegister
from qiskit_aer import AerSimulator
def verif(p, base, thr, sim=AerSimulator(method="matrix_product_state")):
    n = p.shape[0]; w = max(1, int(np.ceil(np.log2(n))))
    cm = lambda x: int(_mk(p, np.asarray(x, dtype=np.int64)))
    B = G.build(p, list(base), thr); o = OM.build(p, "onehot"); idx = {q: t for t, q in enumerate(o["qc"].qubits)}
    BG = ["ccx", "cswap", "cx", "u", "x", "h", "z", "measure"]; out = []
    for code in range(1 << (2 * w)):
        qc = G.grover_circuit(B, list(base), 0, basis_code=code, measure=False)
        om = [r for r in qc.qregs if r.name == "om"][0]; perm = [r for r in qc.qregs if r.name == "perm"][0]
        crc = ClassicalRegister(len(B["S"]), "c"); qc.add_register(crc); qc.measure(B["S"], crc)
        cr2 = ClassicalRegister(B["b"], "cm"); qc.add_register(cr2); qc.measure([om[idx[q]] for q in o["cmax"]], cr2)
        cr3 = ClassicalRegister(len(perm), "pm"); qc.add_register(cr3); qc.measure(perm, cr3)
        key = list(sim.run(transpile(qc, basis_gates=BG, optimization_level=0), shots=1).result().get_counts())[0].split()
        pmv, cmv, cd, mk = key; i, j = G.code_to_move(code, w)
        child = G.shift(base, i, j) if (i < n and j < n) else None
        words = [int(pmv[::-1][k * w:(k + 1) * w][::-1], 2) for k in range(n)]
        out.append(dict(code=code, i=i, j=j, perm_q=words, perm_c=child, cmax_q=int(cmv, 2), cmax_c=cm(child), mark_q=int(mk), mark_c=int(cm(child) < thr), code_lu=int(cd, 2)))
    return out
if __name__ == "__main__":
    p = load("tai20_5_0")[0][:4, :3].copy()
    base = (3, 1, 2, 0); thr = int(_mk(p, np.asarray(base, dtype=np.int64)))
    bad = 0
    for r in verif(p, base, thr):
        ok = r["perm_q"] == r["perm_c"] and r["cmax_q"] == r["cmax_c"] and r["mark_q"] == r["mark_c"] and r["code_lu"] == r["code"]
        bad += not ok
        if not ok: print(r)
    print("ecarts", bad)
