"""Benchmark instances (Taillard 1993; Vallada, Ruiz & Framinan 2015) and their best-known solutions (BKS).
Gaps are measured against the BKS of Brunner (2026, Zenodo record 20529962, v10), column `ub` of data/instances.csv;
the bounds of the original instance files are kept in column `ub_instance_file`."""
import os, numpy as np, pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INSTANCES = pd.read_csv(os.path.join(DATA, "instances.csv"))
SETS = {s: INSTANCES[INSTANCES.set == s].instance.tolist() for s in ("TEST", "HARD", "TUNE+", "EXT")}
UB = dict(zip(INSTANCES.instance, INSTANCES.ub))

def gap_vs_bks(df):
    """Recompute columns UB and Gap of a run table against the best-known solutions of data/instances.csv."""
    df = df.copy(); df["UB"] = df.Instance.map(UB).astype(int); df["Gap"] = 100.0 * (df.Cmax - df.UB) / df.UB
    assert (df.Gap >= 0).all(), "a run beats a best-known solution: check it before analysing"
    return df

def _taillard(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    n, m = (int(v) for v in lines[1].split()[:2])
    k = next(i for i, l in enumerate(lines) if l.lower().startswith("processing times")) + 1
    rows = [[int(v) for v in l.split()] for l in lines[k:k + m]]
    p = np.asarray(rows, dtype=np.int64).T
    assert p.shape == (n, m), (path, p.shape)
    return p

def _vrf(path):
    with open(path) as f:
        n, m = (int(v) for v in f.readline().split()[:2])
        rows = [[int(t) for t in l.split()[1::2]] for l in f if l.strip()]
    p = np.asarray(rows, dtype=np.int64)
    assert p.shape == (n, m), (path, p.shape)
    return p

def load(name):
    """Processing times p[job, machine] (int64) and the best-known makespan used to compute gaps."""
    if name.startswith("tai"):
        p = _taillard(os.path.join(DATA, "taillard", f"{name}.fsp"))
    else:
        p = _vrf(os.path.join(DATA, "vrf", f"{name}_Gap.txt"))
    return p, int(UB[name])
