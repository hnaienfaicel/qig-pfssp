"""Exact statevector check of Grover with oracle v2, using a sparse state vector {basis state: amplitude}.
The state is a superposition of at most 2^q move codes (times 2 transiently under an H), so it is stored exactly.
Gate semantics: X, CX, CCX, Z, CZ, H as usual; AND = CCX on a |0> target (asserted); UNAND in mode
  'unitary': CCX;  'mbu': Gidney's measurement-based uncomputation (H on the target, projective measurement with the
  Born probability, if the outcome is 1 apply CZ on the two controls and X on the target), sampled with a seeded RNG.
After every Grover iteration the check compares the SIGNED amplitudes with the ideal two-dimensional rotation
(-1)^k (marked: sin((2k+1)theta)/sqrt(M), unmarked: cos((2k+1)theta)/sqrt(N-M)); the factor (-1)^k is the
global phase of the diffusion H X MCZ X H = I - 2|s><s| and asserts that every non-move qubit is |0>.
Usage: python circuits/sparse_grover_check.py -> results/oracle_v2_grover_sparse.csv"""
import math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "tests"))
import oracle_v2 as V
import test_oracle_v2 as T

S2 = 1.0 / math.sqrt(2.0)


def apply(state, ops, mode, rng, stats):
    for (op, a, b, c) in ops:
        if op == V.X:
            m = 1 << a; state = {k ^ m: v for k, v in state.items()}
        elif op == V.CX:
            m = 1 << b; state = {(k ^ m) if (k >> a) & 1 else k: v for k, v in state.items()}
        elif op == V.CCX or op == V.AND or (op == V.UNAND and mode == "unitary"):
            if op == V.AND:
                assert all(not (k >> c) & 1 for k in state), "AND target not |0>"
            if op == V.UNAND:
                assert all(((k >> c) & 1) == (((k >> a) & (k >> b)) & 1) for k in state), "UNAND precondition"
            m = 1 << c; state = {(k ^ m) if ((k >> a) & (k >> b)) & 1 else k: v for k, v in state.items()}
        elif op == V.Z:
            state = {k: (-v if (k >> a) & 1 else v) for k, v in state.items()}
        elif op == V.CZ:
            state = {k: (-v if ((k >> a) & (k >> b)) & 1 else v) for k, v in state.items()}
        elif op == V.H:
            new = {}; m = 1 << a
            for k, v in state.items():
                k0 = k & ~m; s = -1.0 if (k >> a) & 1 else 1.0
                new[k0] = new.get(k0, 0.0) + v * S2; new[k0 | m] = new.get(k0 | m, 0.0) + s * v * S2
            state = {k: v for k, v in new.items() if abs(v) > 1e-14}
        elif op == V.UNAND:                      # measurement-based uncomputation (Gidney 2018)
            assert all(((k >> c) & 1) == (((k >> a) & (k >> b)) & 1) for k in state), "UNAND precondition"
            state = apply(state, [(V.H, c, -1, -1)], mode, rng, stats)
            p1 = sum(abs(v) ** 2 for k, v in state.items() if (k >> c) & 1)
            out = int(rng.random() < p1); stats["meas"] += 1; stats["ones"] += out
            norm = math.sqrt(p1 if out else 1.0 - p1)
            state = {k: v / norm for k, v in state.items() if ((k >> c) & 1) == out}
            if out:
                state = apply(state, [(V.CZ, a, b, -1), (V.X, c, -1, -1)], mode, rng, stats)
        else:
            raise ValueError(op)
    return state


def run(kind, p, base, K, mode, seed):
    n = p.shape[0]; thr = int(V.dp_table(p, list(base))[-1, -1])
    B = (V.build_double if kind == "double" else V.build_single)(p, list(base), thr)
    c = B["circ"]; S = B["S"]; q = len(S); N = 1 << q; w = B["w"]
    Dops, na = V.diffusion_ops(S, c.nq); inv = V.inverse_ops(c.ops)
    _, _, kids = T.children(list(base), n, w, kind == "double")
    marked = sorted(cc for cc, x in kids.items() if V.dp_table(p, x)[-1, -1] < thr); M = len(marked)
    th = math.asin(math.sqrt(M / N)); mk = np.zeros(N, dtype=bool); mk[marked] = True
    smask = sum(1 << s for s in S)
    rng = np.random.default_rng(seed); stats = dict(meas=0, ones=0)
    state = apply({0: 1.0}, [(V.H, s, -1, -1) for s in S], mode, rng, stats)
    rows = []
    for k in range(K + 1):
        if k > 0:
            state = apply(state, c.ops + [(V.Z, B["mark"], -1, -1)] + inv + Dops, mode, rng, stats)
        assert all((key & ~smask) == 0 for key in state), "ancilla not returned to |0>"
        amp = np.zeros(N)
        for key, v in state.items():
            code = sum(((key >> s) & 1) << t for t, s in enumerate(S)); amp[code] += v
        ideal = (-1) ** k * np.where(mk, math.sin((2 * k + 1) * th) / math.sqrt(M) if M else 0.0,
                                     math.cos((2 * k + 1) * th) / math.sqrt(N - M))   # diffusion = I - 2|s><s|
        rows.append(dict(kind=kind, mode=mode, seed=seed, case=f"n{n}m{p.shape[1]}", base=" ".join(map(str, base)), thr=thr,
                         N=N, M=M, k=k, P_success=float((amp[mk] ** 2).sum()), P_theory=math.sin((2 * k + 1) * th) ** 2,
                         max_amp_err=float(np.max(np.abs(amp - ideal))), terms=len(state), qubits=c.nq + na,
                         measurements=stats["meas"], outcomes_one=stats["ones"]))
    return rows


if __name__ == "__main__":
    import pandas as pd, time
    P = T.taillard("tai20_5_0"); rows = []; t0 = time.time()
    for base in ((3, 1, 2, 0), (1, 0, 2, 3)):
        rows += run("single", P[:4, :3].copy(), base, 6, "unitary", 0)
        for seed in range(4):
            rows += run("single", P[:4, :3].copy(), base, 6, "mbu", seed)
    rows += run("single", P[:5, :3].copy(), (2, 4, 1, 0, 3), 6, "mbu", 0)
    rows += run("single", P[:6, :4].copy(), (5, 3, 0, 2, 4, 1), 6, "mbu", 1)
    rows += run("double", P[:4, :3].copy(), (3, 1, 2, 0), 4, "unitary", 0)
    rows += run("double", P[:4, :3].copy(), (3, 1, 2, 0), 4, "mbu", 0)
    D = pd.DataFrame(rows); D.to_csv(os.path.join(ROOT, "results", "oracle_v2_grover_sparse.csv"), index=False)
    pd.set_option("display.width", 200)
    print(D.groupby(["kind", "case", "base", "mode", "k"]).agg(N=("N", "first"), M=("M", "first"), P=("P_success", "mean"),
          Pth=("P_theory", "first"), err=("max_amp_err", "max"), runs=("seed", "size"), meas=("measurements", "max")).to_string())
    print("max amplitude error", D.max_amp_err.max(), f"{time.time() - t0:.0f}s")
