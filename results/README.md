# Results

Each experiment writes a CSV plus a `_meta.json` sidecar recording the git commit,
library versions, seeds and runtime. Never edit these by hand.

## Runs to date

| file | grid | cells | runtime |
|---|---|---|---|
| `noise_degradation_2026-09-20.csv` | L = 2, 4, 6 | 240 | 19.0 min |
| `noise_degradation_2026-09-20_L8.csv` | L = 8 | 144 | 38.6 min |

Both use the same scenarios, `p` grid, seeds, 8 pairs per m and 1024 shots, so
they concatenate directly. `--tag` keeps same-day runs from overwriting each other.

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
| `p_out_of_range` | probability mass on outcomes `> L`. These are impossible by construction, so a non-zero rate means noise. See "Reading `p_out_of_range`" below before drawing conclusions from it. |
| `bias` | mean **signed** error, `sum p(v) * (v - m)`. Negative means the count is under-reported. `mae` cannot show direction; this can. |
| `boot_lo`, `boot_hi` | percentile bootstrap 95% CI for P(correct), resampling **pairs** (the independent unit), 10000 resamples. |
| `pair_sd` | standard deviation of P(correct) across the pairs in the cell. |
| `pairs_used` | number of pairs the bootstrap resampled. |
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

## Reading `p_out_of_range`

The control register holds `k = ceil(log2(L+1))` qubits, so it can express `2^k`
values while only `0..L` are reachable noiselessly. The out-of-range rate is therefore
**bounded above by the unused fraction of the codespace**:

| L | k | values | valid | ceiling `(2^k-(L+1))/2^k` |
|---|---|---|---|---|
| 2 | 2 | 4 | 3 | 25.0% |
| 4 | 3 | 8 | 5 | 37.5% |
| 6 | 3 | 8 | 7 | 12.5% |
| 8 | 4 | 16 | 9 | 43.8% |

Two consequences:

1. **The ceiling is 0 when `L+1` is a power of two** (L = 1, 3, 7, 15): the codespace is
   exactly filled, so there are no impossible outcomes and this diagnostic carries no
   information at all at those lengths.
2. **Ordering across L is arithmetic, not an empirical result.** Ranking lengths by
   out-of-range rate mostly reproduces the ranking of their ceilings, including the
   non-monotonicity in L. That is a property of `ceil(log2(L+1))`, not a discovery.

So treat this column as a **cheap consistency check**: values must stay under the
ceiling, and a value approaching the ceiling means the output distribution is
approaching uniform over the whole register, i.e. near-total information loss.
It is not, on its own, evidence of a useful error-detection mechanism.

> **Superseded claim.** Commit `54f8c73` described this as a "free error-detection
> signal" confirmed by rank agreement across L. That reading gave empirical weight to
> what is really a counting bound, and should not be relied on. Whether post-selecting
> on in-range outcomes actually buys accuracy is a separate question, not yet measured.

## Limitations

These bound what may be claimed from this data.

1. **All-to-all connectivity.** Circuits are transpiled with `coupling_map=None`, so no
   SWAP overhead is charged. Real hardware would add routing depth, making these
   numbers optimistic. Connectivity-aware runs are not done.
2. **Thermal relaxation is applied after gates only.** There is no idle/decoherence
   noise on qubits while other qubits are being acted on, and no explicit delay
   instructions. Qubits that sit idle through a long circuit are therefore modelled as
   healthier than they would really be — optimistic.
3. **`rz` handling changed between runs.** `rz` is a virtual gate on IBM hardware: a
   zero-duration frame change carrying essentially no error. Runs dated **2026-09-21 and
   later** model it that way (`virtual_rz=True`, the default), giving rz no depolarizing
   and no thermal relaxation. The **2026-09-20** runs charged rz the same error and 50 ns
   duration as `sx`/`x`, which was strictly pessimistic: rz is ~49% of all gates at every
   `L`, so that added 170 x 50 ns = 8.5 us of fictitious decay at L=8. The `virtual_rz`
   flag reproduces the old behaviour. `meta.json` records which convention a run used, so
   check it before comparing files across dates.
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
6. **Two different intervals are reported; pick the right one.** `wilson_lo/hi` treats
   all `n_trials` as i.i.d. Bernoulli draws, but shots within one pair share a circuit,
   so it answers only "how precisely do we know the rate *for these particular pairs*"
   and is far too narrow as a statement about reads in general. `boot_lo/hi` resamples
   **pairs**, the actual independent unit, and is the one to quote for read-population
   error bars. `pair_sd` shows the between-pair spread directly. Caveat: the bootstrap
   resamples only `pairs_used` values, so with a small pair count it is coarse and its
   endpoints are granular.
7. **One window per circuit.** The reference window is loaded classically per shift, so
   these costs are per-window and exclude any superposition-over-offsets scheme. Paper 3
   stores the whole reference in qubits; any comparison must state this difference.
8. **No baseline yet.** These are absolute degradation curves for one counter, not a
   comparison. The Paper-1 Section 2 "plus-one" Toffoli/CNOT counter and the Paper-3
   scheme are separate, later work.
