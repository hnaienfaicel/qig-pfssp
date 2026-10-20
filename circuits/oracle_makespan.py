"""
Oracle reversible du makespan PFSSP, en portes X / CX / CCX / MCX uniquement
(verifiable par simulation classique bit a bit, puis compte apres transpilation).

C'est la brique dont ont besoin QAOA a cout exact, Grover Adaptive Search et
toute methode qui doit faire interferer les solutions selon leur makespan :
le makespan doit etre calcule DANS le circuit, de facon coherente.

Deux encodages du registre de recherche :
  onehot : x[j][k] = 1 si le job j est en position k        (n^2 qubits)
  lehmer : chiffres L_k sur ceil(log2(n-k)) bits, projection L_k <- min(L_k, n-1-k)
          , puis decodage coherent vers la table one-hot y[j][k]
           (n^2 qubits auxiliaires) avant le meme calcul du makespan.

Calcul : C[k][i] = max(C[k-1][i], C[k][i-1]) + p(pi_k, i), un registre de b bits par
case (Bennett : tout est garde puis decalcule), b = ceil(log2(sum p + 1)).
Construction directe, non optimisee : les comptes sont ceux de CETTE construction.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister


# ----------------------------------------------------------------- primitives
def maj(qc, c, b, a):
    qc.cx(a, b); qc.cx(a, c); qc.ccx(c, b, a)


def maj_inv(qc, c, b, a):
    qc.ccx(c, b, a); qc.cx(a, c); qc.cx(a, b)


def uma(qc, c, b, a):
    qc.ccx(c, b, a); qc.cx(a, c); qc.cx(c, b)


def add_inplace(qc, A, B, c0):
    """B <- B + A (mod 2^n), Cuccaro et al. 2004 ; A restaure, c0 = |0> auxiliaire."""
    n = len(A)
    maj(qc, c0, B[0], A[0])
    for t in range(1, n):
        maj(qc, A[t - 1], B[t], A[t])
    for t in range(n - 1, 0, -1):
        uma(qc, A[t - 1], B[t], A[t])
    uma(qc, c0, B[0], A[0])


def less_than(qc, A, B, c0, f):
    """f ^= [A < B] : retenue de B + ~A. Tout est restaure sauf f."""
    n = len(A)
    for q in A:
        qc.x(q)
    maj(qc, c0, B[0], A[0])
    for t in range(1, n):
        maj(qc, A[t - 1], B[t], A[t])
    qc.cx(A[n - 1], f)
    for t in range(n - 1, 0, -1):
        maj_inv(qc, A[t - 1], B[t], A[t])
    maj_inv(qc, c0, B[0], A[0])
    for q in A:
        qc.x(q)


def geq_const(qc, L, const, tmp, c0, g):
    """g ^= [L >= const] (registre de w bits) : retenue de L + (2^w - const)."""
    w = len(L)
    k = (1 << w) - const
    if k >= (1 << w):            # const = 0 : toujours vrai
        qc.x(g); return
    bits = [(k >> t) & 1 for t in range(w)]
    for t, bt in enumerate(bits):
        if bt: qc.x(tmp[t])
    maj(qc, c0, L[0], tmp[0])
    for t in range(1, w):
        maj(qc, tmp[t - 1], L[t], tmp[t])
    qc.cx(tmp[w - 1], g)
    for t in range(w - 1, 0, -1):
        maj_inv(qc, tmp[t - 1], L[t], tmp[t])
    maj_inv(qc, c0, L[0], tmp[0])
    for t, bt in enumerate(bits):
        if bt: qc.x(tmp[t])


def ctrl_increment(qc, ctrl, R, sign=+1):
    """R += ctrl (sign=+1) ou R -= ctrl (sign=-1), modulo 2^len(R)."""
    w = len(R)
    order = range(w - 1, -1, -1) if sign > 0 else range(w)
    for t in order:
        qc.mcx([ctrl] + list(R[:t]), R[t])


def bits_of(v, b):
    return [(int(v) >> t) & 1 for t in range(b)]


# ----------------------------------------------------------------- oracle
def lehmer_widths(n):
    return [int(math.ceil(math.log2(n - k))) if n - k > 1 else 0 for k in range(n)]


class _Reg(list):
    def __init__(self, size, name, start):
        super().__init__(range(start, start + size)); self.name = name; self.size = size


class Counter:
    """Remplace QuantumCircuit en mode comptage : meme appels, aucun stockage de portes."""
    def __init__(self):
        self.n_qubits = 0; self.ops = {}
    def reg(self, size, name):
        r = _Reg(size, name, self.n_qubits); self.n_qubits += size; return r
    def _add(self, key): self.ops[key] = self.ops.get(key, 0) + 1
    def x(self, q): self._add("x")
    def cx(self, a, b): self._add("cx")
    def ccx(self, a, b, c): self._add("ccx")
    def mcx(self, ctrls, tgt, ctrl_state=None):
        k = len(ctrls); self._add("cx" if k == 1 else "ccx" if k == 2 else f"mcx{k}")


def build(p: np.ndarray, encoding: str = "onehot", count_only: bool = False) -> Dict:
    p = np.asarray(p, dtype=np.int64)
    if count_only:
        cnt = Counter()
        QuantumRegister = lambda size, name: cnt.reg(size, name)          # noqa: N806
        QuantumCircuit = lambda *regs: cnt                                  # noqa: N806
    else:
        from qiskit import QuantumCircuit, QuantumRegister                  # noqa: N806
    n, m = p.shape
    b = int(math.ceil(math.log2(int(p.sum()) + 1)))
    regs: List[QuantumRegister] = []
    if encoding == "onehot":
        X = [QuantumRegister(n, f"x{k}") for k in range(n)]      # X[k][j]
        regs += X
        search = [q for r in X for q in r]
    elif encoding == "lehmer":
        W = lehmer_widths(n)
        Lr = [QuantumRegister(w, f"L{k}") for k, w in enumerate(W) if w > 0]
        Lmap = {k: r for k, r in zip([k for k, w in enumerate(W) if w > 0], Lr)}
        X = [QuantumRegister(n, f"y{k}") for k in range(n)]      # table one-hot decodee
        avail = QuantumRegister(n, "avail")
        wr = int(math.ceil(math.log2(n + 1)))
        R = QuantumRegister(wr, "rank")
        wmax = max(W)
        tmpL = QuantumRegister(max(wmax, 1), "tmpL")
        gq = QuantumRegister(1, "clamp")
        regs += Lr + X + [avail, R, tmpL, gq]
        search = [q for r in Lr for q in r]
    else:
        raise ValueError(encoding)
    C = [[QuantumRegister(b, f"C{k}_{i}") for i in range(m)] for k in range(n)]
    P = QuantumRegister(b, "P"); T = QuantumRegister(b, "T")
    anc = QuantumRegister(2, "anc")                               # c0, f
    regs += [r for row in C for r in row] + [P, T, anc]
    qc = QuantumCircuit(*regs)
    c0, f = anc[0], anc[1]

    # --- decodage coherent Lehmer -> table one-hot
    if encoding == "lehmer":
        for q in avail:
            qc.x(q)
        for k in range(n):
            if W[k] == 0:                      # derniere position : le job restant
                for j in range(n):
                    qc.cx(avail[j], X[k][j])
                for j in range(n):
                    qc.cx(X[k][j], avail[j])
                continue
            L = Lmap[k]; w = W[k]
            clamp = (1 << w) > (n - k)
            if clamp:
                geq_const(qc, L, n - k, tmpL, c0, gq[0])
            for j in range(n):
                for t in range(w):
                    qc.cx(L[t], R[t])
                qc.mcx([avail[j]] + list(R), X[k][j], ctrl_state="0" * wr + "1")
                for t in range(w):
                    qc.cx(L[t], R[t])
                if clamp:
                    cs = "".join(str(v) for v in reversed(bits_of(n - 1 - k, wr))) + "11"
                    qc.mcx([avail[j], gq[0]] + list(R), X[k][j], ctrl_state=cs)
                ctrl_increment(qc, avail[j], R, +1)
            for j in range(n - 1, -1, -1):
                ctrl_increment(qc, avail[j], R, -1)
            if clamp:
                geq_const(qc, L, n - k, tmpL, c0, gq[0])
            for j in range(n):
                qc.cx(X[k][j], avail[j])

    # --- makespan
    def load_p(k, i):                         # P ^= p(pi_k, i) : CX seulement (one-hot)
        for j in range(n):
            for t, bt in enumerate(bits_of(p[j, i], b)):
                if bt: qc.cx(X[k][j], P[t])

    for k in range(n):
        for i in range(m):
            load_p(k, i)
            if k == 0 and i == 0:
                for t in range(b): qc.cx(P[t], C[0][0][t])
            elif k == 0 or i == 0:
                prev = C[k][i - 1] if k == 0 else C[k - 1][i]
                for t in range(b): qc.cx(prev[t], C[k][i][t])
                add_inplace(qc, P, C[k][i], c0)
            else:
                A, B = C[k - 1][i], C[k][i - 1]
                less_than(qc, A, B, c0, f)           # f = [A < B]
                qc.x(f)
                for t in range(b): qc.ccx(f, A[t], T[t])
                qc.x(f)
                for t in range(b): qc.ccx(f, B[t], T[t])
                for t in range(b): qc.cx(T[t], C[k][i][t])
                add_inplace(qc, P, C[k][i], c0)
                for t in range(b): qc.ccx(f, B[t], T[t])   # decalcul de T puis f
                qc.x(f)
                for t in range(b): qc.ccx(f, A[t], T[t])
                qc.x(f)
                less_than(qc, A, B, c0, f)
            load_p(k, i)                              # P remis a 0
    return dict(qc=qc, search=search, cmax=list(C[n - 1][m - 1]), b=b,
                onehot=X, encoding=encoding)


# ----------------------------------------------------------------- simulation
def simulate(qc: QuantumCircuit, ones: List) -> np.ndarray:
    idx = {q: i for i, q in enumerate(qc.qubits)}
    s = np.zeros(qc.num_qubits, dtype=bool)
    for q in ones:
        s[idx[q]] = True
    for ins in qc.data:
        op, qs = ins.operation, [idx[q] for q in ins.qubits]
        nm = op.name
        if nm == "x":
            s[qs[0]] ^= True
        elif nm in ("cx", "ccx", "mcx") or nm.startswith("mcx"):
            ctrls, tgt = qs[:-1], qs[-1]
            cs = getattr(op, "ctrl_state", (1 << len(ctrls)) - 1)
            want = [(cs >> i) & 1 for i in range(len(ctrls))]
            if all(s[c] == bool(w_) for c, w_ in zip(ctrls, want)):
                s[tgt] ^= True
        else:
            raise ValueError(nm)
    return s


def read(s, qc, qubits):
    idx = {q: i for i, q in enumerate(qc.qubits)}
    return sum(int(s[idx[q]]) << t for t, q in enumerate(qubits))
