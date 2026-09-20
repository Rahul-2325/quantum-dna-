# Results

Each experiment writes a CSV plus a `_meta.json` sidecar recording the git commit,
library versions, seeds and runtime. Never edit these by hand.

## `noise_degradation_<date>.csv`

Degradation of the weight-register mismatch counter (`src/hwlib.py:mismatch_circuit`)
under parametric noise. Produced by `src/experiments/noise_degradation.py`.

One row per **cell** = (scenario, L, p, m). Within a cell the true mismatch count `m`
is fixed, and `pairs_per_m` independently sampled (read, window) pairs are each run
for `shots` shots; the outcome histograms are pooled, so `n_trials = pairs_per_m * shots`.

### Design columns

| column | meaning |
|---|---|
| `scenario` | `S1` = depolarizing only (1q = p/10, 2q = p), thermal and readout off. `S2` = S1 + thermal relaxation + readout error. |
| `L` | read length in bases. The circuit uses `k + 3L` qubits. |
| `p` | two-qubit (cx) depolarizing probability. One-qubit gates get `p/10`. |
| `m` | true mismatch count, planted exactly (0..L). |
| `pairs_per_m` | independently sampled (read, window) pairs in this cell. |
| `shots` | shots per pair. |
| `n_trials` | `pairs_per_m * shots`, the pooled Bernoulli trial count. |

### Metric columns

| column | meaning |
|---|---|
| `p_correct` | fraction of shots whose decoded control register equals `m`. |
| `wilson_lo`, `wilson_hi` | Wilson 95% score interval for `p_correct` over `n_trials`. |
| `mae` | mean absolute error of the measured count, `sum p(v) * abs(v - m)`. |
| `p_out_of_range` | probability mass on outcomes `> L`, which are impossible noiselessly and so are a pure noise signature. |
| `t_quarter` | the value of `floor(L/4)`, i.e. which threshold the `_tq` columns used. |
| `fn_t0`, `fn_t1`, `fn_tq` | false-negative rate at thresholds `t = 0, 1, floor(L/4)`: true `m <= t` but measured `> t`. Empty when `m > t`. |
| `fp_t0`, `fp_t1`, `fp_tq` | false-positive rate at the same thresholds: true `m > t` but measured `<= t`. Empty when `m <= t`. |

Only one of FP/FN is defined per row, because `m` is fixed within a cell; the other is
written as an empty/NaN field. Note `floor(L/4)` collapses onto `t=0` at `L=2` and onto
`t=1` at `L=4,6`, so `_tq` duplicates another column at those lengths; it is kept as a
separate column so the schema is identical across `L`.

### Provenance columns

| column | meaning |
|---|---|
| `k` | control-register width, `ceil(log2(L+1))`. |
| `num_qubits` | total circuit width, `k + 3L`. |
| `depth`, `cx` | transpiled depth and CX count for a representative circuit at this `L`. |
| `seed_pairs`, `seed_sim` | base seeds for pair sampling and for the simulator. |
| `elapsed_s` | wall-clock seconds for this cell. |

## Limitations

These bound what may be claimed from this data.

1. **All-to-all connectivity.** Circuits are transpiled with `coupling_map=None`, so no
   SWAP overhead is charged. Real hardware would add routing depth, making these
   numbers optimistic. Connectivity-aware runs are not done.
2. **Thermal relaxation is applied after gates only.** There is no idle/decoherence
   noise on qubits while other qubits are being acted on, and no explicit delay
   instructions. Qubits that sit idle through a long circuit are therefore modelled as
   healthier than they would really be — optimistic.
3. **`rz` carries noise** although `rz` is a virtual, error-free gate on IBM hardware.
   These circuits are rz-heavy (rz is ~49% of all gates at every `L`), so this is a
   deliberate pessimistic choice; it keeps `assert_full_coverage` meaningful. S1/S2 are
   therefore not a literal model of any specific device. **Measured sensitivity:**
   rebuilding the S2 `p=0` point with `rz` fully noiseless recovers only +0.011 / +0.011 /
   +0.020 in P(correct) for L = 2 / 4 / 6. The decoherence floor is therefore not an
   artifact of this choice — it is dominated by the 300 ns CX duration (24.3 us of the
   ~31.7 us circuit time at L=6) rather than by the 50 ns rz gates.
4. **Basis-state inputs only.** Reads and windows are classical strings, so the ideal
   output is a single deterministic value. TVD is therefore not reported: it would equal
   `1 - p_correct` and carry no extra information. The distribution helpers in
   `src/aer_helpers.py` are kept for later superposition-input tests, where TVD becomes
   meaningful.
5. **Simulation method.** The sweep runs on Aer's matrix-product-state backend, which is
   exact for these low-entanglement circuits (`test_mps_matches_statevector` pins MPS
   against dense statevector to 1e-12, noiseless and noisy). MPS removes the dense
   memory wall — L=8 costs about 9 MB rather than the 4.29 GB a 28-qubit statevector
   needs — but MPS cost grows with entanglement, so this advantage is a property of
   *these* circuits and must be re-verified before trusting it on any new construction.
6. **Pooled confidence intervals.** The Wilson interval treats all `n_trials` as i.i.d.
   Bernoulli draws, but shots within one pair share a circuit. The interval is therefore
   for the pooled success rate *conditional on the sampled pairs*; between-pair
   variability is not captured and the true uncertainty over the population of reads is
   wider. Do not quote these intervals as read-population error bars.
7. **One window per circuit.** The reference window is loaded classically per shift, so
   these costs are per-window and exclude any superposition-over-offsets scheme. Paper 3
   stores the whole reference in qubits; any comparison must state this difference.
8. **No baseline yet.** These are absolute degradation curves for one counter, not a
   comparison. The Paper-1 Section 2 "plus-one" Toffoli/CNOT counter and the Paper-3
   scheme are separate, later work.
