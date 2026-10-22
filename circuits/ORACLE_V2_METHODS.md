# Optimised oracle (v2) for the Q-IG insertion neighbourhood: design, validation and cost

Optimised oracle of the Q-IG paper (Section 5.1 of the main text, Section S12 of the supplementary material). The
double-insertion variant (v2-double) is an extension that is not used in the paper. All numbers come from the scripts
listed at the end.

## 1. What is simulated, what is counted

| object | simulated | counted only |
|---|---|---|
| v1 oracle (paper) | – (re-recorded gate for gate; counts/depths reproduce `oracle_resources_taillard.csv`, `oracle_depth.csv` and the qdk rows of `ft_resources.csv` exactly) | blocks, depth, FT time |
| v2 / v2-double reversible part | exact basis-state simulation of **all** move codes, n = 4-6 (single), n = 4-5 (double) | all 8 Taillard sizes |
| Gidney measurement-based uncomputation | (i) basis-state: asserted precondition target = a AND b at every uncomputation; (ii) exact sparse state vector with sampled measurements and CZ fix-ups; (iii) Qiskit Aer MPS with mid-circuit measurement and feed-forward | T-free, 1 layer, 1 logical cycle in qdk |
| Grover iterations | exact sparse state vector (single n = 4, 5, 6; double n = 4) and Aer MPS (single n = 4) | – |
| FT time | – | qdk estimator (sequential) and a depth-limited model |

## 2. Profile of the paper's oracle (v1)

The v1 oracle is a gate-for-gate replica (`circuits/profile_v1.py`) of `grover_circuit.build` + `oracle_makespan.build(p, "onehot")`.
Toffoli-like = CCX + CSWAP; depths are standalone depths of each block (all gates one layer), so they do not add up to
the oracle depth (blocks overlap a little in the ASAP schedule).

| instance | Toffoli-like (CCX+CSWAP) | perm_network % | onehot_decode % | dp % | threshold % | depth (oracle) | depth perm_network | depth onehot_decode | depth load_p | depth dp |
|---|---|---|---|---|---|---|---|---|---|---|
| tai20_5_0 | 24,210 | 45 | 11.6 | 43.3 | 0.1 | 28,893 | 12,588 | 3,281 | 2,646 | 15,965 |
| tai20_20_0 | 69,026 | 15.8 | 4.1 | 80.1 | 0 | 94,857 | 12,588 | 3,281 | 9,765 | 81,608 |
| tai50_5_0 | 140,436 | 63.4 | 16 | 20.6 | 0 | 145,412 | 100,470 | 25,602 | 13,966 | 43,754 |
| tai50_10_0 | 179,404 | 49.6 | 12.5 | 37.8 | 0 | 202,171 | 100,470 | 25,602 | 28,768 | 100,558 |
| tai50_20_0 | 262,652 | 33.9 | 8.6 | 57.5 | 0 | 323,672 | 100,470 | 25,602 | 54,870 | 221,522 |
| tai100_5_0 | 609,902 | 71.7 | 18 | 10.2 | 0 | 608,537 | 485,102 | 122,401 | 50,241 | 94,163 |
| tai100_10_0 | 693,430 | 63.1 | 15.9 | 21.1 | 0 | 709,750 | 485,102 | 122,401 | 104,422 | 215,472 |
| tai100_20_0 | 871,198 | 50.2 | 12.6 | 37.2 | 0 | 967,584 | 485,102 | 122,401 | 207,323 | 472,896 |

The conditional-swap permutation network (comparisons of i, j with constants by enumerating up to k+1 values with
V-chain multi-controlled X) costs O(n^2 w) Toffolis. Above n = 50 it alone carries 34-72 % of the Toffolis, and
one-hot decoding another 9-18 %. The DP, with Cuccaro comparators computed and uncomputed plus a Toffoli selection, costs
about 10b Toffolis per cell with b = ceil(log2(sum p + 1)) = 13-17. All cells share the registers P, T and the carry
ancilla, so the oracle is sequential: Toffoli parallelism is 1.1-1.4.

## 3. Design of oracle v2 (single insertion, move register (i, j) on 2w qubits)

The base permutation pi is classical during one descent step, so the circuit is compiled for pi: its classical data
(processing times of pi) are CNOT fan-out patterns. Only the 2w-qubit move register is quantum.

1. **Unary iteration** (Babbush et al., PRX 8, 041015, 2018, doi:10.1103/PhysRevX.8.041015): one-hot flags e^i, e^j by
   in-place splitting of prefix flags (node q splits into q AND x_t and q AND NOT x_t with one temporary AND), about
   n-2 ANDs per register. Prefixes that lie entirely at or above n are not split.
2. **Thermometer flags**: D_k = XOR_(t<=k)(e^i_t XOR e^j_t) = [min(i,j) <= k < max(i,j)] (CNOT chain);
   dir = [i<j] (w-bit comparator); L_k = D_k AND dir = [i <= k < j]; R_(k+1) = D_k XOR L_k = [j < k+1 <= i];
   E_k = e^j_k. This costs n-1+w ANDs instead of the 2(n-1) conditional swaps and n^2 decoders of v1.
3. **Processing times**: P_(k,r) = x XOR L_k(x XOR y) XOR R_k(x XOR z) XOR E_k x XOR (E_k AND Q_r), with x, y, z =
   p(pi_k|pi_(k+1)|pi_(k-1), r) classical, so the classical part is CNOT/X only. The moved job's times
   Q_r = p(pi_i, r) are a table lookup over i done by CNOT fan-out from e^i (Clifford, since e^i is already
   computed). The only non-Clifford part is E_k AND Q_r: bp = 7 ANDs per cell.
4. **DP cell** C = max(A, B) + P with Gidney temporary logical-ANDs (Gidney, Quantum 2, 74, 2018,
   doi:10.22331/q-2018-06-18-74): comparator f = [A<B] = carry-out of B + NOT A (b ANDs); selection
   M = A XOR f(A XOR B), with f fanned out to b copies (b parallel ANDs); in-place adder M += P (b-1 ANDs). All carry
   chains are uncomputed by X-basis measurement plus CZ fix-up (no T). f and C are kept until the inverse oracle
   (Bennett).
5. **Wavefront**: each machine column r owns dedicated temporaries (copies of both inputs, comparator chain,
   fan-out of f, P, adder carries). The cells of one anti-diagonal therefore share no qubit except read-only
   sources, and the ASAP schedule runs anti-diagonals in parallel. The extra qubits are m(5b - 3 + bp).
6. **Width**: b = ceil(log2(U+1)) with U = min(sum_j max_r p_jr + sum_(r>=1) max_j p_jr, symmetric bound), a
   permutation-independent bound on every DP value: 11-14 bits instead of 13-17. Values of invalid codes may wrap
   around; this is harmless because they are never marked.
7. **Mark** = valid AND [Cmax < thr], valid = [i<n] AND [j<n] AND [i != j]. [i<n] is the parity of e^i (Clifford),
   [i = j] needs w-1 ANDs, and [Cmax < thr] is a constant comparator with at most b-1 ANDs. Invalid codes are never
   marked, whatever the threshold.

**Closed-form AND count** (`tests/analytic_counts.py`), equal to the recorded gate lists on all 8 sizes:
forward = 2S(n,w) + 2w + n + 1 + n m bp + (n-1)(m-1)(3b-1) + (n+m-2)(b-1) + T(b,thr) + 1;
inverse = 2w - 2 + (n-1)(m-1)(2b-2+bp) + (n+m-2)(b-1+bp) + T(b,thr);
iteration = forward + inverse + (2w-2) (diffusion).
Per interior cell and per Grover iteration this is 5b - 3 + 2bp ANDs (81 at 100x20), against about 20b = 340
Toffolis for v1, before counting v1's permutation network and decoding.

Blocks of v2 and v2-double at 100x20 (forward oracle):

| variant | block | cx | ands | unand | depth_standalone | toffoli_depth_standalone |
|---|---|---|---|---|---|---|
| v2 | unary | 206 | 202 | 0 | 64 | 55 |
| v2 | flags | 644 | 114 | 12 | 301 | 106 |
| v2 | qrom | 6,336 | 0 | 0 | 195 | 0 |
| v2 | load_p | 39,348 | 14,000 | 13,993 | 2,607 | 833 |
| v2 | dp | 467,689 | 78,655 | 50,440 | 31,303 | 4,759 |
| v2 | threshold | 1 | 9 | 8 | 29 | 9 |
| v2-double | unary | 412 | 404 | 0 | 64 | 55 |
| v2-double | flags | 2,182 | 1,417 | 26 | 312 | 112 |
| v2-double | qrom | 25,720 | 438 | 0 | 491 | 143 |
| v2-double | load_p | 140,805 | 28,000 | 27,993 | 3,520 | 833 |
| v2-double | dp | 467,689 | 78,655 | 50,440 | 31,303 | 4,759 |
| v2-double | threshold | 1 | 9 | 8 | 29 | 9 |

## 4. Double insertion (move (i -> j) then (k -> l) on 4w qubits)

Row q of the child is tau_(q') with q' in {q, q+1, q-1} for the second move, and tau_(q') in
{pi_(q'), pi_(q'+1), pi_(q'-1), pi_i}. The 12 products of first- and second-move flags (12n ANDs, computed once)
select classical constants by CNOT fan-out. Two quantum-data terms remain per cell: G_q AND Q1 (job pi_i not moved
again) and E2_q AND Q2, with Q2 = p(tau_k, .) obtained by a flag-weighted lookup over k (3n + m bp ANDs, once).
This gives 2bp ANDs per cell instead of bp. The DP, the width and the threshold are those of the single-insertion
oracle. Validity requires the six conditions [i,j,k,l < n], i != j, k != l, so there are (n(n-1))^2 valid codes out of 2^(4w).

## 5. Validation

* Exhaustive basis-state simulation (`tests/test_oracle_v2.py`): 244 cases (single: n = 4, 5, 6 x m = 2-5, Taillard
  sub-matrices of tai20_5_0 plus random p in [1, 99], random bases, the paper's 4x3 example with base 3 1 2 0;
  double: n = 4, 5 x m = 2-4), 7 thresholds each (0, Cmax(base), Cmax(base)+1, min, min+1, median, max+1 over moves).
  7,792 single and 177,408 double code evaluations; **0 mismatches**.
  For every code the test checks all n x m DP registers against the classical DP of the child, the mark, the validity
  flag, the integrity of the move register and the cleanliness of the column temporaries. It also asserts that every
  AND target is |0> and every measured uncomputation satisfies target = a AND b, and that O^-1 O returns every
  ancilla to |0>.
* Mutation test: 7 injected bugs (E-term dropped, wrong neighbour constant, comparator A<=B, missing [i != j], missing
  MSB carry, double G or Q2 term dropped) were all detected.
* Exact sparse state vector (`circuits/sparse_grover_check.py`) for single n4m3 (bases 3 1 2 0 and 1 0 2 3), n5m3 and
  n6m4, and double n4m3 (N = 256), for k = 0..6 Grover iterations, both with the unitary form and with the
  measurement-based uncomputation (sampled outcomes, up to 7,338 mid-circuit measurements per run).
  The signed amplitudes equal the ideal rotation (-1)^k [sin((2k+1)theta)/sqrt(M), cos((2k+1)theta)/sqrt(N-M)]
  within 8.3e-14, and every ancilla is |0> after every iteration. The P_success values coincide with the
  paper's v1 Aer scan (e.g. 0.472656, 0.908447, 0.961319 for base 3 1 2 0).
* Qiskit Aer MPS, gate by gate (`circuits/grover_check_v2.py`, n4m3, 322 qubits, about 52 s per iteration): unitary
  form k <= 3 for both bases, and the measurement-based form with mid-circuit measurement and if_test feed-forward
  (k = 1, 3 shots). Maximum |P - sin^2((2k+1)theta)| = 2.9e-10.

## 6. Resources per oracle call and per Grover iteration

Iteration = oracle + Z(mark) + inverse oracle + diffusion. Toffoli-like = CCX + CSWAP + AND. Depth: every gate is one
layer (Toffoli, Fredkin, AND and measured uncomputation included), as in `oracle_depth.csv`. Toffoli depth: Clifford
gates and measurements take zero time. Peak = most Toffoli-like gates in one Toffoli layer.

| instance | variant | b | qubits_iteration | oracle_toffoli | iter_toffoli | iter_t_count | iter_cx | iter_depth | iter_toffoli_depth | iter_peak_parallel_toffoli |
|---|---|---|---|---|---|---|---|---|---|---|
| tai20_5_0 | v1 | 13 | 1,862 | 24,210 | 48,435 | 339,045 | 56,794 | 57,805 | 36,947 | 3 |
| tai20_5_0 | v2 | 11 | 1,631 | 3,440 | 5,906 | 23,624 | 37,350 | 9,682 | 1,385 | 14 |
| tai20_5_0 | v2-double | 11 | 2,184 | 4,532 | 7,718 | 30,872 | 50,866 | 9,830 | 1,425 | 18 |
| tai20_20_0 | v1 | 15 | 6,568 | 69,026 | 138,067 | 966,469 | 250,486 | 189,733 | 100,517 | 3 |
| tai20_20_0 | v2 | 12 | 6,707 | 15,934 | 27,113 | 108,452 | 176,216 | 15,058 | 2,209 | 29 |
| tai20_20_0 | v2-double | 12 | 7,470 | 19,231 | 33,230 | 132,920 | 223,938 | 15,186 | 2,249 | 35 |
| tai50_5_0 | v1 | 14 | 6,369 | 140,436 | 280,891 | 1,966,237 | 240,890 | 290,847 | 222,313 | 3 |
| tai50_5_0 | v2 | 13 | 4,081 | 10,011 | 17,126 | 68,504 | 108,132 | 27,449 | 3,948 | 16 |
| tai50_5_0 | v2-double | 13 | 5,332 | 12,697 | 21,586 | 86,344 | 140,542 | 27,725 | 4,021 | 19 |
| tai50_10_0 | v1 | 15 | 10,372 | 179,404 | 358,827 | 2,511,789 | 510,206 | 404,365 | 274,701 | 3 |
| tai50_10_0 | v2 | 13 | 7,956 | 21,130 | 35,934 | 143,736 | 232,750 | 29,059 | 4,196 | 21 |
| tai50_10_0 | v2-double | 13 | 9,277 | 25,601 | 43,929 | 175,716 | 296,086 | 29,309 | 4,268 | 24 |
| tai50_20_0 | v1 | 16 | 18,875 | 262,652 | 525,323 | 3,677,261 | 1,048,262 | 647,367 | 392,133 | 3 |
| tai50_20_0 | v2 | 13 | 15,706 | 43,370 | 73,554 | 294,216 | 481,830 | 32,311 | 4,696 | 29 |
| tai50_20_0 | v2-double | 13 | 17,167 | 51,411 | 88,619 | 354,476 | 604,206 | 32,561 | 4,768 | 34 |
| tai100_5_0 | v1 | 15 | 18,276 | 609,902 | 1,219,827 | 8,538,789 | 808,826 | 1,217,101 | 1,092,181 | 3 |
| tai100_5_0 | v2 | 14 | 8,263 | 21,405 | 36,570 | 146,280 | 230,850 | 58,162 | 8,385 | 17 |
| tai100_5_0 | v2-double | 14 | 10,670 | 26,743 | 45,436 | 181,744 | 295,504 | 58,614 | 8,510 | 19 |
| tai100_10_0 | v1 | 16 | 26,779 | 693,430 | 1,386,883 | 9,708,181 | 1,684,062 | 1,419,527 | 1,093,373 | 3 |
| tai100_10_0 | v2 | 14 | 16,163 | 45,265 | 76,865 | 307,460 | 498,062 | 59,908 | 8,655 | 22 |
| tai100_10_0 | v2-double | 14 | 18,640 | 54,138 | 92,766 | 371,064 | 623,804 | 60,336 | 8,780 | 25 |
| tai100_20_0 | v1 | 17 | 44,782 | 871,198 | 1,742,419 | 12,196,933 | 3,393,558 | 1,935,195 | 1,333,851 | 3 |
| tai100_20_0 | v2 | 14 | 31,963 | 92,980 | 157,445 | 629,780 | 1,028,448 | 63,318 | 9,181 | 30 |
| tai100_20_0 | v2-double | 14 | 34,580 | 108,923 | 187,416 | 749,664 | 1,273,618 | 63,740 | 9,305 | 35 |

Ablations of the single-insertion v2 (same gates): v2-serialDP = one shared set of DP temporaries (no wavefront);
v2-bv1 = the bit width of v1.

| instance | variant | b | qubits_iteration | iter_toffoli | iter_depth | iter_toffoli_depth |
|---|---|---|---|---|---|---|
| tai20_5_0 | v2 | 11 | 1,631 | 5,906 | 9,682 | 1,385 |
| tai20_5_0 | v2-serialDP | 11 | 1,395 | 5,906 | 14,282 | 3,261 |
| tai20_5_0 | v2-bv1 | 13 | 1,883 | 6,762 | 11,262 | 1,641 |
| tai20_20_0 | v2 | 12 | 6,707 | 27,113 | 15,058 | 2,209 |
| tai20_20_0 | v2-serialDP | 12 | 5,491 | 27,113 | 70,226 | 13,728 |
| tai20_20_0 | v2-bv1 | 15 | 8,210 | 32,762 | 18,418 | 2,773 |
| tai50_5_0 | v2 | 13 | 4,081 | 17,126 | 27,449 | 3,948 |
| tai50_5_0 | v2-serialDP | 13 | 3,805 | 17,126 | 41,911 | 9,107 |
| tai50_5_0 | v2-bv1 | 14 | 4,357 | 18,214 | 29,379 | 4,256 |
| tai50_10_0 | v2 | 13 | 7,956 | 35,934 | 29,059 | 4,196 |
| tai50_10_0 | v2-serialDP | 13 | 7,335 | 35,934 | 92,857 | 18,152 |
| tai50_10_0 | v2-bv1 | 15 | 9,058 | 40,580 | 33,139 | 4,852 |
| tai50_20_0 | v2 | 13 | 15,706 | 73,554 | 32,311 | 4,696 |
| tai50_20_0 | v2-serialDP | 13 | 14,395 | 73,554 | 194,751 | 36,096 |
| tai50_20_0 | v2-bv1 | 16 | 19,009 | 87,933 | 39,091 | 5,800 |
| tai100_5_0 | v2 | 14 | 8,263 | 36,570 | 58,162 | 8,385 |
| tai100_5_0 | v2-serialDP | 14 | 7,967 | 36,570 | 90,156 | 19,054 |
| tai100_5_0 | v2-bv1 | 15 | 8,789 | 38,758 | 61,992 | 8,993 |
| tai100_10_0 | v2 | 14 | 16,163 | 76,865 | 59,908 | 8,655 |
| tai100_10_0 | v2-serialDP | 14 | 15,497 | 76,865 | 201,054 | 38,130 |
| tai100_10_0 | v2-bv1 | 16 | 18,265 | 86,211 | 67,788 | 9,911 |
| tai100_20_0 | v2 | 14 | 31,963 | 157,445 | 63,318 | 9,181 |
| tai100_20_0 | v2-serialDP | 14 | 30,557 | 157,445 | 422,734 | 76,123 |
| tai100_20_0 | v2-bv1 | 17 | 38,266 | 186,374 | 75,798 | 11,185 |

## 7. Time per Grover iteration

**Break-even** (paper data): t_BE = k* x t_call, with k* = 15.61 (IG; 9.13 for IG-RR) from `amend5_equiv.csv` and the
median measured time of a classical IG call per instance from `amend5_ig_timing.csv` (Xeon Gold 5120).

**(a) Sequential FT (upper estimate)**: qdk 1.33.1 `LogicalCounts` with error budget 0.1 per Grover iteration and the
paper's three models: qubit_gate_ns_e3, qubit_gate_ns_e4 and qubit_maj_ns_e4 (floquet code). v1 uses the paper's
convention (it reproduces `ft_resources.csv` exactly). v2 uses the exact counts per iteration: numQubits = qubits incl.
diffusion ancillas, cczCount = Toffoli-like gates, measurementCount = measured uncomputations. The estimator runs the CCZs
one after another (3 logical cycles each, 1 per measurement), so parallelism is ignored.

**(b) Depth-limited FT (lower estimate)**, explicit assumptions:
1. same code distance d, logical cycle tau, physical qubits per logical qubit and factory design as (a) for the same model
   (conservative on d, since the shorter run would allow a smaller d);
2. Clifford gates and measured uncomputations take zero time (absorbed in lattice-surgery/Pauli-frame tracking);
3. each layer of the Toffoli-depth schedule takes one logical cycle, so T_low = D_T x tau;
4. one CCZ-type magic state (4 T states in the estimator's 15-to-1 factories) is available for each Toffoli of a
   layer, so the number of factories is F_peak = ceil(4 W_peak t_factory / (s tau)) (s = T states per factory run),
   with physical qubits = layout logical qubits x physical per logical + F_peak x factory qubits; F_mean uses
   the mean parallelism (stalls would then lengthen the run);
5. routing space for parallel operations is assumed to fit in the layout of (a), which is optimistic.

**(c) Noiseless machine**: T = D x t_layer, t_layer = 1, 10, 100 ns, where D is the depth of one iteration (every
gate, including a Toffoli or a mid-circuit measurement with feed-forward, takes one layer).

| instance | variant | break-even IG (us) | noiseless 1 ns (us) | 10 ns (us) | 100 ns (us) | FT seq gate_ns_e3 (s) | FT seq gate_ns_e4 (s) | FT seq maj_ns_e4 (s) | FT depth-lim gate_ns_e3 (s) | FT depth-lim gate_ns_e4 (s) | FT depth-lim maj_ns_e4 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tai20_5_0 | v1 | 16 | 57.8 | 578 | 5,780 | 0.988 | 0.523 | 0.392 | 0.251 | 0.133 | 0.0998 |
| tai20_5_0 | v2 | 16 | 9.68 | 96.8 | 968 | 0.142 | 0.0661 | 0.0638 | 0.00831 | 0.00388 | 0.00374 |
| tai20_5_0 | v2-double | 16 | 9.83 | 98.3 | 983 | 0.185 | 0.0864 | 0.0834 | 0.00855 | 0.00399 | 0.00385 |
| tai50_10_0 | v1 | 7.81 | 404 | 4,044 | 40,436 | 9.04 | 4.74 | 3.55 | 2.31 | 1.21 | 0.907 |
| tai50_10_0 | v2 | 7.81 | 29.1 | 291 | 2,906 | 1.09 | 0.517 | 0.388 | 0.0319 | 0.0151 | 0.0113 |
| tai50_10_0 | v2-double | 7.81 | 29.3 | 293 | 2,931 | 1.34 | 0.633 | 0.474 | 0.0324 | 0.0154 | 0.0115 |
| tai100_20_0 | v1 | 7.08 | 1,935 | 19,352 | 1.94e+05 | 48.1 | 23 | 17.2 | 12.3 | 5.87 | 4.4 |
| tai100_20_0 | v2 | 7.08 | 63.3 | 633 | 6,332 | 5.29 | 2.77 | 2.08 | 0.0771 | 0.0404 | 0.0303 |
| tai100_20_0 | v2-double | 7.08 | 63.7 | 637 | 6,374 | 6.3 | 3.3 | 2.47 | 0.0782 | 0.0409 | 0.0307 |

FT model details (d, tau, factories, physical qubits):

| instance | variant | model | code_distance | logical_cycle_ns | layout_logical_qubits | seq_physical_qubits | seq_t_factories | W_peak | dl_factories_peak | dl_physical_qubits_peak | dl_factories_mean | dl_physical_qubits_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tai20_5_0 | v1 | qubit_gate_ns_e3 | 17 | 6,800 | 3,848 | 2,340,304 | 12 | 3 | 101 | 3,201,824 | 45 | 2,659,744 |
| tai20_5_0 | v1 | qubit_gate_ns_e4 | 9 | 3,600 | 3,848 | 633,376 | 10 | 3 | 87 | 710,376 | 38 | 661,376 |
| tai20_5_0 | v1 | qubit_maj_ns_e4 | 9 | 2,700 | 3,848 | 1,929,824 | 20 | 3 | 175 | 5,315,024 | 77 | 3,174,704 |
| tai20_5_0 | v2 | qubit_gate_ns_e3 | 15 | 6,000 | 3,378 | 1,616,900 | 10 | 14 | 534 | 6,689,220 | 163 | 3,097,940 |
| tai20_5_0 | v2 | qubit_gate_ns_e4 | 7 | 2,800 | 3,378 | 341,044 | 10 | 14 | 520 | 851,044 | 159 | 490,044 |
| tai20_5_0 | v2 | qubit_maj_ns_e4 | 9 | 2,700 | 3,378 | 1,594,584 | 13 | 14 | 679 | 16,140,024 | 207 | 5,831,544 |
| tai20_5_0 | v2-double | qubit_gate_ns_e3 | 15 | 6,000 | 4,502 | 2,122,700 | 10 | 18 | 687 | 8,676,060 | 207 | 4,029,660 |
| tai20_5_0 | v2-double | qubit_gate_ns_e4 | 7 | 2,800 | 4,502 | 451,196 | 10 | 18 | 669 | 1,110,196 | 202 | 643,196 |
| tai20_5_0 | v2-double | qubit_maj_ns_e4 | 9 | 2,700 | 4,502 | 2,030,696 | 13 | 18 | 872 | 20,791,256 | 263 | 7,490,696 |
| tai50_10_0 | v1 | qubit_gate_ns_e3 | 21 | 8,400 | 21,034 | 18,803,988 | 14 | 3 | 119 | 20,693,988 | 52 | 19,487,988 |
| tai50_10_0 | v1 | qubit_gate_ns_e4 | 11 | 4,400 | 21,034 | 5,113,748 | 12 | 3 | 100 | 5,286,228 | 44 | 5,176,468 |
| tai50_10_0 | v1 | qubit_maj_ns_e4 | 11 | 3,300 | 21,034 | 12,212,616 | 16 | 3 | 143 | 14,986,296 | 63 | 13,239,096 |
| tai50_10_0 | v2 | qubit_gate_ns_e3 | 19 | 7,600 | 16,166 | 11,749,292 | 8 | 21 | 633 | 17,799,292 | 258 | 14,169,292 |
| tai50_10_0 | v2 | qubit_gate_ns_e4 | 9 | 3,600 | 16,166 | 2,626,892 | 8 | 21 | 607 | 3,225,892 | 248 | 2,866,892 |
| tai50_10_0 | v2 | qubit_maj_ns_e4 | 9 | 2,700 | 16,166 | 6,600,008 | 15 | 21 | 1,223 | 32,982,728 | 499 | 17,170,568 |
| tai50_10_0 | v2-double | qubit_gate_ns_e3 | 19 | 7,600 | 18,828 | 13,671,256 | 8 | 24 | 723 | 20,592,456 | 310 | 16,594,616 |
| tai50_10_0 | v2-double | qubit_gate_ns_e4 | 9 | 3,600 | 18,828 | 3,058,136 | 8 | 24 | 694 | 3,744,136 | 298 | 3,348,136 |
| tai50_10_0 | v2-double | qubit_maj_ns_e4 | 9 | 2,700 | 18,828 | 7,632,864 | 15 | 24 | 1,398 | 37,837,584 | 600 | 20,409,264 |
| tai100_20_0 | v1 | qubit_gate_ns_e3 | 23 | 9,200 | 90,164 | 95,645,512 | 14 | 3 | 120 | 97,553,512 | 53 | 96,347,512 |
| tai100_20_0 | v1 | qubit_gate_ns_e4 | 11 | 4,400 | 90,164 | 21,843,208 | 12 | 3 | 100 | 22,015,688 | 44 | 21,905,928 |
| tai100_20_0 | v1 | qubit_maj_ns_e4 | 11 | 3,300 | 90,164 | 51,201,936 | 16 | 3 | 143 | 53,975,616 | 63 | 52,228,416 |
| tai100_20_0 | v2 | qubit_gate_ns_e3 | 21 | 8,400 | 64,433 | 56,951,586 | 9 | 30 | 966 | 69,890,226 | 553 | 64,306,466 |
| tai100_20_0 | v2 | qubit_gate_ns_e4 | 11 | 4,400 | 64,433 | 15,610,426 | 9 | 30 | 993 | 17,539,066 | 568 | 16,706,066 |
| tai100_20_0 | v2 | qubit_maj_ns_e4 | 11 | 3,300 | 64,433 | 36,602,292 | 12 | 30 | 1,430 | 67,571,412 | 817 | 54,183,492 |
| tai100_20_0 | v2-double | qubit_gate_ns_e3 | 21 | 8,400 | 69,687 | 61,585,614 | 9 | 35 | 1,127 | 76,700,974 | 649 | 70,238,414 |
| tai100_20_0 | v2-double | qubit_gate_ns_e4 | 11 | 4,400 | 69,687 | 16,881,894 | 9 | 35 | 1,159 | 19,135,894 | 667 | 18,171,574 |
| tai100_20_0 | v2-double | qubit_maj_ns_e4 | 11 | 3,300 | 69,687 | 39,565,548 | 12 | 35 | 1,668 | 75,732,588 | 960 | 60,269,868 |

Noiseless machine at 1 ns per layer, ratio to the break-even (IG):

| instance | breakeven_IG_us | noiseless_1ns_us | ratio v2 noiseless 1ns / break-even | ratio v1 noiseless 1ns / break-even |
|---|---|---|---|---|
| tai20_5_0 | 16 | 9.68 | 0.605 | 3.61 |
| tai20_20_0 | 19.4 | 15.1 | 0.777 | 9.79 |
| tai50_5_0 | 6.93 | 27.4 | 3.96 | 41.9 |
| tai50_10_0 | 7.81 | 29.1 | 3.72 | 51.7 |
| tai50_20_0 | 10.3 | 32.3 | 3.14 | 62.9 |
| tai100_5_0 | 3.91 | 58.2 | 14.9 | 311 |
| tai100_10_0 | 4.79 | 59.9 | 12.5 | 296 |
| tai100_20_0 | 7.08 | 63.3 | 8.94 | 273 |

## 8. Findings

* v2 lowers the cost of one Grover iteration by 5.1-33.4x in Toffoli-like gates (11.1x at 100x20),
  7.6-49.9x in T-count per oracle call,
  6.0-30.6x in depth and 27-145x in Toffoli depth.
  It also uses fewer qubits on 7 of 8 sizes (31,963 vs 44,782 at 100x20;
  at 20x20 v2 uses 2 % more). The two largest effects are the removal of the permutation network and decoding
  (classical base) and the wavefront: without it the Toffoli depth is 2.3-8.3x larger. With v1's width b, the Toffoli count would be
  6-21 % higher (v2-bv1).
* The double-insertion oracle costs only 19-31 % more Toffolis per iteration than the single one, with the
  same depth (+1-2 %) and 8-34 % more qubits, while sqrt(N) grows from 2^w to 2^(2w) (128 -> 16,384 at n = 100). The DP over n x m
  dominates the oracle.
* Break-even: on the noiseless machine at 1 ns per layer, v2 is below the break-even at n = 20
  (0.60x at 20x5, 0.78x at 20x20) but 3.1-14.9x above it for n = 50-100 (v1: 3.6-311x).
  At 10 ns per layer v2 is above the break-even everywhere. Fault-tolerant estimates remain far above: at
  100x20, sequential 5.29 s (gate_ns_e3) vs 48.1 s for v1, and depth-limited
  77 ms, against a break-even of 7.1 us. Over the three sizes and three models, the
  depth-limited v2 time per iteration is at least 233x the break-even
  (logical cycle 2.7-8.4 us, i.e. at least 1,385 x 2.7 us per iteration at 20x5).
  A cheaper oracle does not close the gap: with one logical cycle per Toffoli layer, the Toffoli depth alone exceeds
  the classical time budget per iteration.

## 9. Caveats

* Counts are exact for the constructions written here. Other constructions could do better: carry-lookahead
  comparators (Draper et al., quant-ph/0406142) would shorten the Clifford depth per cell (about 4b layers per
  ripple), and a phase-oracle form would drop the mark qubit. These are not implemented.
* The oracle is compiled for each base permutation: its CNOT pattern changes at every accepted move. This costs
  O(nm bp) classical work per descent step, which is not charged.
* The measured uncomputation of f (the comparator flag) is counted in its deterministic form (b-1 ANDs). Gidney's
  phase fix-up would only be needed with probability 1/2, so the expected count would be lower.
* Depth counts a Toffoli, an AND and a measured uncomputation as one layer each. The noiseless model at 1 ns per layer
  ignores the latency of mid-circuit measurement and feed-forward.
* The depth-limited FT model reuses d and the factory design from qdk's sequential estimate and assumes that the
  routing of parallel operations fits in the sequential layout. Its time is a lower estimate, and the factory count
  needed to reach it is large (F_peak = 520-1,668 factories, 0.85-77 million physical qubits for v2 and v2-double, Table 6).
* Note: v2 (like v1's gate-level circuit) uses the (i, j) register with N = 2^(2w).
  `qig/quantum.py::insertion_descent` instead emulates a register (i, direction, distance) with
  N = 2^(w_i+1+w_m), which is twice as large at all 8 sizes (e.g. 2^15 vs 2^14 at n = 100). The stopping cap
  cq sqrt(N) and theta therefore differ by sqrt(2) between the emulator and the gate-level circuit.
* Qualtran was not used. The cross-check is the closed-form count of Section 3 and the literature: a Gidney adder on
  b bits costs b-1 ANDs (4(b-1) T).

## 10. Files

* `circuits/oracle_v2.py`: recorder, primitives, `build_single`, `build_double`, `iteration_ops`, basis simulator,
  ASAP scheduler.
* `circuits/profile_v1.py`: v1 replica with block labels, giving `results/oracle_v1_breakdown.csv`.
* `circuits/count_v2.py`: `results/oracle_v2_counts_raw.csv` and `results/oracle_v2_breakdown.csv`.
* `circuits/ft_v2.py` (qdk): `results/ft_v2_resources.csv` and `results/oracle_v2_resources.csv`.
* `tests/test_oracle_v2.py` (`results/oracle_v2_validation.csv`), `tests/analytic_counts.py`,
  `circuits/sparse_grover_check.py` (`results/oracle_v2_grover_sparse.csv`) and `circuits/grover_check_v2.py`
  (Aer, `results/oracle_v2_grover_check_single.csv`).

References: Babbush et al. 2018 (PRX 8, 041015); Gidney 2018 (Quantum 2, 74); Cuccaro et al. 2004 (quant-ph/0410184);
Draper et al. 2006 (quant-ph/0406142); Beverland et al. 2022 (arXiv:2211.07629, resource estimator); Boyer, Brassard,
Hoyer and Tapp 1998 (BBHT); Taillard 1990.
