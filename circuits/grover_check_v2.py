"""Gate-level Grover check of oracle v2 (Qiskit Aer, matrix-product-state method, exact for this circuit since the
state is a sum over the 2^q move codes, so every Schmidt rank is <= 2^q).

The recorded gate list is exported to Qiskit in two forms:
  unitary : AND and its uncomputation both as CCX (the unitary that the Gidney gadgets implement);
  mbu     : AND as CCX on a |0> target, uncomputation as Gidney's measurement-based gadget
            (H, mid-circuit measurement, classically controlled CZ on the controls, X reset of the target).
Born probabilities of the move register are saved after every iteration and compared with sin^2((2k+1) theta),
sin theta = sqrt(M/N) (M marked codes counted classically).  In mbu mode the probabilities are saved per shot,
so each random branch of the measurements is checked.
Usage: python circuits/grover_check_v2.py [single|double] -> results/oracle_v2_grover_check_<kind>.csv"""
import math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import oracle_v2 as V
import profile_v1 as P1
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator


def to_qiskit(qc, ops, mode, cbit):
    for (op, a, b, c) in ops:
        if op == V.X: qc.x(a)
        elif op == V.CX: qc.cx(a, b)
        elif op == V.CZ: qc.cz(a, b)
        elif op == V.Z: qc.z(a)
        elif op == V.H: qc.h(a)
        elif op in (V.CCX, V.AND): qc.ccx(a, b, c)
        elif op == V.CSWAP: qc.cswap(a, b, c)
        elif op == V.UNAND:
            if mode == "unitary":
                qc.ccx(a, b, c)
            else:
                qc.h(c); qc.measure(c, cbit)
                with qc.if_test((cbit, 1)):
                    qc.cz(a, b); qc.x(c)
        else:
            raise ValueError(op)


def run(kind, p, base, K, mode, shots=1, threads=16):
    n = p.shape[0]; thr = int(V.dp_table(p, list(base))[-1, -1])
    B = (V.build_double if kind == "double" else V.build_single)(p, list(base), thr)
    c = B["circ"]; S = B["S"]; q = len(S); N = 1 << q; w = B["w"]
    Dops, na = V.diffusion_ops(S, c.nq)
    import test_oracle_v2 as T
    codes, valid, kids = T.children(list(base), n, w, kind == "double")
    marked = [cc for cc, x in kids.items() if V.dp_table(p, x)[-1, -1] < thr]; M = len(marked)
    th = math.asin(math.sqrt(M / N))
    qr = QuantumRegister(c.nq + na, "q"); cr = ClassicalRegister(1, "m"); qc = QuantumCircuit(qr, cr)
    qc.h(S); qc.save_probabilities(qubits=S, label="p0", pershot=(mode == "mbu"))
    inv = V.inverse_ops(c.ops)
    for k in range(1, K + 1):
        to_qiskit(qc, c.ops, mode, cr[0]); qc.z(B["mark"]); to_qiskit(qc, inv, mode, cr[0]); to_qiskit(qc, Dops, mode, cr[0])
        qc.save_probabilities(qubits=S, label=f"p{k}", pershot=(mode == "mbu"))
    sim = AerSimulator(method="matrix_product_state", mps_omp_threads=threads)
    t = time.perf_counter(); res = sim.run(qc, shots=shots, seed_simulator=11).result().data(0); dt = time.perf_counter() - t
    rows = []
    for k in range(K + 1):
        pr = np.atleast_2d(np.asarray(res[f"p{k}"]))
        for s in range(pr.shape[0]):
            ps = float(pr[s][marked].sum()); unm = [x for x in range(N) if x not in set(marked)]
            rows.append(dict(kind=kind, mode=mode, case=f"n{n}m{p.shape[1]}", base=" ".join(map(str, base)), thr=thr,
                             N=N, M=M, k=k, shot=s, P_success=ps, P_theory=math.sin((2 * k + 1) * th) ** 2,
                             abs_err=abs(ps - math.sin((2 * k + 1) * th) ** 2),
                             spread_marked=float(np.ptp(pr[s][marked])) if M > 1 else 0.0,
                             spread_unmarked=float(np.ptp(pr[s][unm])), qubits=c.nq + na,
                             oracle_and=sum(1 for g in c.ops if g[0] == V.AND), sim_s=round(dt, 1)))
    return rows


if __name__ == "__main__":
    import pandas as pd
    kind = sys.argv[1] if len(sys.argv) > 1 else "single"
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tests"))
    P = P1.taillard("tai20_5_0"); rows = []
    if kind == "single":
        for base in ((3, 1, 2, 0), (1, 0, 2, 3)):                 # ~52 s per iteration on 16 threads
            rows += run("single", P[:4, :3].copy(), base, 3, "unitary")
        rows += run("single", P[:4, :3].copy(), (3, 1, 2, 0), 1, "mbu", shots=3)
    else:
        rows += run("double", P[:4, :3].copy(), (3, 1, 2, 0), 2, "unitary")
        rows += run("double", P[:4, :3].copy(), (3, 1, 2, 0), 1, "mbu", shots=4)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(os.path.dirname(HERE), "results", f"oracle_v2_grover_check_{kind}.csv"), index=False)
    print(D.groupby(["mode", "base", "k"]).agg(M=("M", "first"), N=("N", "first"), P=("P_success", "mean"), Pth=("P_theory", "first"),
                                               err=("abs_err", "max"), sm=("spread_marked", "max"), su=("spread_unmarked", "max"),
                                               shots=("shot", "size"), t=("sim_s", "first")).to_string())
