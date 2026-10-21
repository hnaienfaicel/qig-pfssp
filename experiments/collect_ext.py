"""Merge the EXT shards (results/raw/ext_*.csv) into results/ext_runs.csv and check completeness."""
import os, glob, json, pandas as pd, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qig.instances import SETS
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
_fs = sorted(glob.glob(os.path.join(R, "raw", "ext_*.csv")))   # shards are not versioned; results/ext_runs.csv is
if not _fs: sys.exit("no shards in results/raw/ext_*.csv: results/ext_runs.csv is used as is")
A = pd.concat([pd.read_csv(f) for f in _fs], ignore_index=True)
A = A.drop_duplicates(["Arm", "Instance", "Run"]).sort_values(["Arm", "Instance", "Run"])
assert len(A) == 6 * len(SETS["EXT"]) * 30, len(A)
A.to_csv(os.path.join(R, "ext_runs.csv"), index=False); print(len(A))
