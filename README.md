# Q-IG: quantum search subroutines inside Iterated Greedy for the permutation flow shop

Code, data and results for the paper *"Quantum search subroutines inside Iterated Greedy for the permutation flow
shop: a query-model evaluation against same-law classical twins"* (manuscript under submission, not included here).

**What is studied.** The Iterated Greedy algorithm of Ruiz & Stützle (2007) for F|prmu|Cmax, with its two search
operations replaced by amplitude-amplification subroutines:
- **Q-IG-min**: the best-insertion scan becomes Dürr–Høyer quantum minimum finding (with BBHT searches);
- **Q-IG-G**: quantum minimum for the construction, plus a BBHT–Grover descent over the whole insertion neighbourhood.

Each variant is compared with **same-law classical twins**. A twin is the same algorithm with the same stopping rules,
in which every amplified search is replaced by uniform sampling with the same output law, at the same number of
oracle calls. Only interference differs between a variant and its twin.

**How the quantum part is evaluated.** In the query model: one oracle call is one makespan evaluation, and a Taillard
scan of n positions costs n calls. The output law of amplitude amplification is emulated exactly: the makespans of
all branches are computed classically, not charged, and the measurement is drawn from sin²((2k+1)θ). The emulated law
is validated gate by gate on Qiskit Aer (matrix-product-state method) with the full reversible oracle, at 4×3 and 8×2.
The advantage reported is a **query advantage**: it requires a fault-tolerant machine and is not a wall-clock speed-up.

## Main results (46,080 oracle calls per run, best-of-30 gap to the best-known solution, %)

| Arm | TEST (100 instances) | HARD (40 instances) | EXT (190 instances, replication) |
|---|---|---|---|
| IG (classical, full scans) | 0.900 | 2.724 | 1.093 |
| IG-Rmin (twin of Q-IG-min) | 0.827 | 2.565 | 0.982 |
| **Q-IG-min** | **0.746** | **2.348** | **0.887** |
| IG-R (twin of Q-IG-G, preregistered) | 0.773 | 2.486 | 0.897 |
| IG-RR (complete twin of Q-IG-G) | 0.762 | 2.513 | 0.902 |
| **Q-IG-G** | **0.505** | **1.661** | **0.553** |
| Q-IG-G, 2 oracle calls per Grover iteration | 0.580 | 1.867 | – |

All ten preregistered tests are significant on TEST and HARD (Holm-adjusted p ≤ 3.3e-4); see `results/tests.csv`.
The preregistered replication on EXT (Amendment 4: tai50_10 numbers 5–9 and the 185 remaining VRF small instances,
registered before any run, with its own Holm families) confirms every hypothesis (Holm-adjusted p ≤ 8.8e-6). For example,
Q-IG-G − IG-RR = −0.349 points (108 instances better, 4 worse, 78 ties). Gaps are measured against the
best-known solutions (BKS) of Brunner (2026, Zenodo record 20529962, v10; files in `data/bks/`), re-evaluated on our instance
files. Of the 350 instances used, 280 are proven optimal. The preregistration specified the bounds of the instance files, which are outdated for 64 of the 160 TEST, HARD and TUNE+ instances (80 of all 350); the
results against those bounds are in `results/tests_instance_file_bounds.csv` and `results/levels_instance_file_bounds.csv`
(Amendment 3: every paired difference moves by less than 0.01 points and no conclusion changes).

**Larger budgets (exploratory, not preregistered).** On the 38 open TEST/HARD instances, at 10× and 100× the budget
(5 runs), the advantage of Q-IG-G over its complete twin shrinks but persists (−1.16, −0.92, −0.75 points), and Q-IG-G
is at least as good as IG and IG-RR given ten times more calls. No best-known solution is improved
(`results/long_levels.csv`, `results/long_tests.csv`, `results/record_runs.csv`).

**Convergence (exploratory, Amendment 4).** With the best solution recorded at 30 call checkpoints, Q-IG-G reaches the
final mean gap of IG after about 7,300 calls (6.4 times fewer) on TEST and HARD. It is the best arm, and IG the worst, at
every checkpoint from 5,568 calls (`results/convergence.csv`, `results/convergence_summary.csv`).

**Noise (exploratory, Amendment 4).** If each Grover iteration is corrupted with probability ε (a round of k iterations stays coherent
with probability (1−ε)^k, otherwise the measurement returns a uniformly random code), Q-IG-G still beats its complete twin up to ε = 0.1. The break-even point is about ε ≈ 0.18
(TEST) and 0.21 (HARD). Q-IG-min is much less sensitive (`results/noise_levels.csv`, `results/noise_tests.csv`).

**Fault-tolerant cost (exploratory, Amendment 4).** Azure Quantum Resource Estimator, run locally with `qdk`. On a
100×20 Taillard instance with the `qubit_gate_ns_e3` model and an error budget of 0.1 per Grover iteration, one
iteration needs code distance 23, about 95.6 million physical qubits and 48 s. A run of 46,080 iterations would take
about 26 days, against a median of 0.66 s for a classical IG run. The query advantage is therefore not a wall-clock
advantage on foreseeable hardware (`results/ft_resources.csv`).

## Repository layout

```
qig/            emulation library: instances, flow-shop kernels (Taillard scan, NEH), quantum subroutines, IG variants
experiments/    run.py (all phases, sharded), analyse.py (statistics), arms.json (tuned settings),
                quantum_minimum_cost.py (cost and exactness of the quantum minimum),
                run_long.py, run_record.py and analyse_long.py (exploratory larger-budget runs)
                collect_ext.py (EXT replication), run_extra.py and analyse_extra.py (noise and convergence studies)
circuits/       reversible oracle (oracle_makespan.py, grover_circuit.py), exhaustive oracle check (verify_oracle.py),
                gate-level Born probabilities (grover_kscan.py), logical resource counts (resources.py),
                fault-tolerant estimates (ft_estimate.py, Azure Quantum Resource Estimator through qdk)
data/           the 350 instances used (Taillard 1993; Vallada, Ruiz & Framinan 2015), instances.csv (sets, best-known
                solutions, original bounds) and bks/ (best-known solutions with schedules, Brunner 2026, CC BY 4.0)
results/        raw runs (test/hard/tune*, raw_long/, raw_record/), ext_runs.csv, tests.csv, levels.csv, long_*.csv,
                noise_*.csv, trace_runs.csv, convergence*.csv, ft_resources.csv, circuit validation and resource tables
figures/        make_figures.py and make_tables.py: every figure and table of the paper is generated from results/
                (figures are written to figures/, LaTeX tables to paper/tables/; generated files are not versioned)
preregistration/  preregistration with its four dated amendments (French original)
```

## Reproduce

```bash
pip install -r requirements.txt                      # numpy, numba, pandas, scipy, matplotlib (+ qiskit for circuits/)
python experiments/run.py test 0 1                   # one shard; use: for s in 0..K-1: run.py PHASE s K
python experiments/run.py hard 0 1
python experiments/run.py test_cpi2 0 1              # robustness: 2 oracle calls per Grover iteration
for s in $(seq 0 39); do python experiments/run.py ext $s 40 & done; wait; python experiments/collect_ext.py   # EXT replication
python experiments/analyse.py                        # -> results/tests.csv, results/levels.csv
for s in $(seq 0 49); do python experiments/run_long.py long10 $s 50 results/raw_long & done; wait   # then long100
python experiments/run_record.py results/raw_record 54 && python experiments/analyse_long.py
for s in $(seq 0 39); do python experiments/run_extra.py noise $s 40 results/raw_extra & done; wait   # then: trace
python experiments/analyse_extra.py                  # -> noise_*.csv, trace_runs.csv, convergence*.csv
pip install qdk && python circuits/ft_estimate.py     # -> results/ft_resources.csv
python figures/make_figures.py && python figures/make_tables.py
cd circuits && python verify_oracle.py && python grover_kscan.py n4 && python resources.py
```

Phases: `tune`, `tune2`, `tune3` (TUNE+, 5 runs), `test`, `hard`, `ext` (30 runs), `test_cpi2`, `hard_cpi2`. Seeds are
1000 + 7919 r. Runs are deterministic: re-running any (arm, instance, run) reproduces `Cmax`, `Evals` and `Iter` exactly.
The full campaign takes about 15 CPU-hours (6 arms × 140 instances × 30 runs).

## Citation

Please cite the paper (reference to be added on publication) if you use this code. Gaps use the best-known solutions of Brunner (2026),
https://doi.org/10.5281/zenodo.20529962, listed in `data/instances.csv`.

## License

MIT (see `LICENSE`).
