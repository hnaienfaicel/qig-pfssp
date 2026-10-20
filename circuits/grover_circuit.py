"""Circuit de Grover COMPLET, porte par porte, pour la recherche d'une insertion ameliorante (PFSSP) :
registre de mouvement (i, j) en superposition -> insertion coherente sur un registre de permutation (mots de w bits)
-> decodage one-hot -> makespan reversible (oracle_makespan) -> marque [Cmax < seuil] -> phase -> decalcul -> diffusion.
Toutes les portes de l'oracle sont reversibles classiques (X, CX, CCX, CSWAP) avec ancillas propres explicites
(multi-controles en V-chain) : simulable exactement sur Qiskit Aer, methode a etats produits matriciels (MPS)."""
import math, numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
import oracle_makespan as OM

BG = ["ccx", "cswap", "cx", "u", "x", "h", "z", "measure"]

def shift(perm, i, j):
    x = list(perm); v = x.pop(i); x.insert(j, v); return x

def code_to_move(code, w):
    return code & ((1 << w) - 1), (code >> w) & ((1 << w) - 1)

def mcx_v(qc, ctrls, target, aux, ctrl_state=None):
    """X sur target si ctrls == ctrl_state (bit t de ctrl_state <-> ctrls[t]); V-chain, ancillas propres rendues a 0."""
    c = list(ctrls); k = len(c)
    flips = [c[t] for t in range(k) if ctrl_state is not None and not (ctrl_state >> t) & 1]
    for q in flips: qc.x(q)
    if k == 1: qc.cx(c[0], target)
    elif k == 2: qc.ccx(c[0], c[1], target)
    else:
        assert len(aux) >= k - 2
        chain = [(c[0], c[1], aux[0])] + [(c[t], aux[t - 2], aux[t - 1]) for t in range(2, k - 1)]
        for g in chain: qc.ccx(*g)
        qc.ccx(c[k - 1], aux[k - 3], target)
        for g in reversed(chain): qc.ccx(*g)
    for q in flips: qc.x(q)

def build(p, base, thr):
    n, m = p.shape; w = max(1, math.ceil(math.log2(n)))
    om = OM.build(p, "onehot"); omqc = om["qc"]; b = om["b"]
    I = QuantumRegister(w, "i"); J = QuantumRegister(w, "j"); Pm = QuantumRegister(n * w, "perm")
    c1 = QuantumRegister(1, "c1"); c2 = QuantumRegister(1, "c2"); f = QuantumRegister(1, "flag")
    aux = QuantumRegister(max(1, 2 * w - 3), "aux")
    OMQ = QuantumRegister(omqc.num_qubits, "om"); tmp = QuantumRegister(b, "tmp"); mark = QuantumRegister(1, "mark")
    idx = {q: t for t, q in enumerate(omqc.qubits)}
    X = [[OMQ[idx[om["onehot"][k][jj]]] for jj in range(n)] for k in range(n)]
    cmax = [OMQ[idx[q]] for q in om["cmax"]]; anc = [r for r in omqc.qregs if r.name == "anc"][0]; c0 = OMQ[idx[anc[0]]]
    regs = [I, J, c1, c2, f, aux, Pm, OMQ, tmp, mark]
    O = QuantumCircuit(*regs)
    def less_than(reg, K, tgt):          # tgt ^= [reg < K]  (valeurs mutuellement exclusives : XOR = OU)
        for v in range(min(K, 1 << w)): mcx_v(O, reg, tgt, aux, ctrl_state=v)
    def cond_swap(k, lo, hi):            # echange des mots k, k+1 si lo <= k < hi
        less_than(lo, k + 1, c1[0])
        less_than(hi, k + 1, c2[0]); O.x(c2[0])                      # c2 = [hi >= k+1]
        O.ccx(c1[0], c2[0], f[0])
        for t in range(w): O.cswap(f[0], Pm[k * w + t], Pm[(k + 1) * w + t])
        O.ccx(c1[0], c2[0], f[0])
        O.x(c2[0]); less_than(hi, k + 1, c2[0]); less_than(lo, k + 1, c1[0])
    for k in range(n - 1): cond_swap(k, I, J)                         # montee : i < j
    for k in reversed(range(n - 1)): cond_swap(k, J, I)               # descente : j < i
    for k in range(n):
        for jj in range(n):
            mcx_v(O, [Pm[k * w + t] for t in range(w)], X[k][jj], aux, ctrl_state=jj)
    O.compose(omqc, qubits=list(OMQ), inplace=True)
    OM.geq_const(O, cmax, thr, list(tmp), c0, mark[0]); O.x(mark[0])  # mark = [Cmax < seuil]
    O = transpile(O, basis_gates=BG, optimization_level=0, qubits_initially_zero=False)
    S = list(I) + list(J); D = QuantumCircuit(*regs)
    D.h(S); D.x(S); D.h(S[-1]); mcx_v(D, S[:-1], S[-1], aux); D.h(S[-1]); D.x(S); D.h(S)
    return dict(regs=regs, O=O, D=D, S=S, Pm=Pm, OMQ=OMQ, mark=mark, w=w, b=b, n=n, om=om, idx=idx)

def grover_circuit(B, base, k, basis_code=None, measure=True):
    qc = QuantumCircuit(*B["regs"]); w = B["w"]
    for pos, job in enumerate(base):
        for t in range(w):
            if (int(job) >> t) & 1: qc.x(B["Pm"][pos * w + t])
    if basis_code is None: qc.h(B["S"])
    else:
        for t, q in enumerate(B["S"]):
            if (basis_code >> t) & 1: qc.x(q)
    Oi = B["O"].inverse()
    for _ in range(k):
        qc.compose(B["O"], inplace=True); qc.z(B["mark"][0]); qc.compose(Oi, inplace=True); qc.compose(B["D"], inplace=True)
    if basis_code is not None:
        cm = ClassicalRegister(1, "m"); qc.add_register(cm); qc.compose(B["O"], inplace=True); qc.measure(B["mark"][0], cm[0])
    if measure:
        cr = ClassicalRegister(len(B["S"]), "c"); qc.add_register(cr); qc.measure(B["S"], cr)
    return qc
