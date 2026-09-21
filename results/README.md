# Results

Each experiment writes a CSV plus a `_meta.json` sidecar recording the git commit,
library versions, seeds and runtime. Never edit these by hand.

## `counter_resources.csv`

Produced by `src/experiments/counter_resources.py`. Resource comparison of the two
counters ALONE (`plus_one_counter(L)` vs `weight_unitary(L)`, acting on `L` bits
directly), decoupled from the DNA-mismatch prefix (read load / XOR / OR-flag) that
both `mismatch_circuit` and `mismatch_circuit_adder` share -- the prefix is common
infrastructure, not part of what differs between the counters.

Columns: `{adder,phase}_{qubits,cx,depth}_i` = the counter as actually used in our
experiments (compute the weight, read it out). `{adder,phase}_{qubits,cx,depth}_ii` =
compute, then uncompute the counter's own output register too -- the cost of using
the counter as a borrowed subroutine inside a larger coherent oracle that must
release its qubits clean. `{adder,phase}_uncompute_adds_cx` = `cx_ii - cx_i`.
`adder_temp_qubits` = the adder's carry/scratch width (`max(0, k-2)`); note this
scratch is *already* self-cleaning after every single increment (proven in
`tests/test_plus_one_counter.py`) regardless of whether the OUTPUT register is
later uncomputed -- (ii) is about the output, not this scratch.

**A methodology note worth keeping:** `qc.compose(qc.inverse())` with nothing
between the two halves is mathematically the identity operation, and Qiskit's
optimizer can and does find that -- an early version of this script measured
exactly that (no barrier between forward and inverse) and got a circuit
transpiled down to *fewer* CX than the forward-only version at L=2, which is not
a real result, just the transpiler being good at algebra. A `barrier` now sits
between compute and uncompute, standing in for the real payload operation that
any genuine oracle usage would have there, which blocks that cancellation and
makes (ii) mean "compute, then separately uncompute" rather than "an
optimizer-dependent fraction of the identity."

Current numbers (2026-09-22): phase's (ii) is exactly 2x (i) at every `L` -- a
clean, symmetric round trip once cross-boundary cancellation is blocked. The
adder's (ii) is consistently ~2.25-2.33x (i), not exactly double, across all
four `L`. Both counters' own ancilla structure is already garbage-free forward
(phase has none beyond its output register; the adder's carry scratch
self-cleans per increment), so (i) qubit counts already reflect that.

## `counter_comparison_2026-09-22.csv`

C4: the adder (Paper 1) vs phase (Paper 2) mismatch counters, run through the
identical noise harness as `noise_degradation.py` (same scenarios, p-grid,
strata, seeds, shots, `virtual_rz`) with only the circuit builder swapped.
480 rows = 2 counters x 2 scenarios x 8 p-values x sum(L+1 for L in 2,4,6).
Same columns as `noise_degradation_*.csv` plus a `counter` column
(`"adder"`/`"phase"`). All 30 S1 p=0 control cells are exactly 1.0.
`figures/6_counter_comparison_noise.png` plots P(correct) vs p per L, S2.

**Result: not a clean win for either counter.** The adder wins at every p
for L=2 and L=4 (e.g. L=4, p=0.01: adder 0.677 vs phase 0.606). At L=6 the
two curves are close and **cross**: adder leads at p<=0.001, phase overtakes
from p=0.005 up (L=6, p=0.05: adder 0.177 vs phase 0.198). This is consistent
with the resource preview in `figures/5_counter_comparison.png`: the adder
has fewer CX at small L but the phase counter's shallower depth (which
matters more as thermal relaxation time accumulates) starts to dominate as L
grows. **L=8 is not yet run** for this comparison (queued as a separate leg,
same reasoning as the L=8 noise-degradation runs: the adder's mismatch
circuit is measurably slower per cell than the phase counter's, and the two
counters together triple the total grid size versus a single-counter sweep).

## `depth_optimal_counter.py` (task 2: depth-optimal rewrite)

CLAUDE.md flagged the old v1 depth-optimal attempt (`Untitled5.ipynb`) as
WRONG -- success ~8%, needing "a real rewrite: GHZ/repetition-code fan-out +
semiclassical IQFT with classically controlled Rz (Qiskit dynamic circuits)."
`src/depth_optimal_counter.py` is that rewrite, verified against
`weight_unitary` (Algorithm 2) on random superposition inputs, not just basis
states -- `tests/test_depth_optimal_counter.py`, 10/10 passing.

**Three things needed empirical resolution, not just a literal reading of
Definition IV.1** (documented in the module docstring): the rotation index
used per round is REVERSED relative to round number (round 1, unconditioned,
uses the largest rotation), the classically-conditioned phase correction
needs a NEGATIVE sign, and round 1's extracted bit is the LEAST significant
bit of the weight. Each was found by brute-forcing the 8 sign/order
combinations against the classical weight (zero mismatches for the winning
combination, n=2,3,4 exhaustive), not re-derived analytically. An initial
naive reading gave near-random output (~13-25% success) -- the same failure
signature the old v1 attempt had, which is itself evidence the naive reading
is a genuinely different, wrong circuit rather than a relabeling of a
correct one.

**A second, unrelated bug surfaced during verification: Aer's
`matrix_product_state` method gives visibly wrong results on this circuit's
superposition inputs** (TVD ~0.14-0.20 against the true distribution) even
though every basis-state input is still exactly correct under MPS. Dense
`statevector` gives TVD ~0.003-0.02 (shot-noise-consistent) on the identical
circuit. This is isolated to MPS's handling of THIS circuit's mid-circuit
measurement + GHZ-entangled parity + classical feedback -- every other
circuit in this project (weight_unitary, plus_one_counter, mismatch_circuit*)
is purely unitary with a single final measurement, and MPS is proven exact
for those (`test_mps_matches_statevector`). Use `statevector`, not MPS, for
this circuit with anything but classical basis-state inputs;
`test_mps_is_wrong_on_this_circuit_regression_guard` pins this so it isn't
silently "optimized" back onto MPS later.

**One documented simplification, not a silent one:** the repetition-code
"fan-out" step (encoding one control qubit into n physical copies) uses a
standard O(log n)-depth CNOT tree, not the paper's own constant-depth
construction (which needs machinery from a separate paper, Quek-Kaur-Wilde,
not reimplemented here). This is why the depth numbers below beat Algorithm
2 by a growing margin but likely don't hit the paper's literal O(log n)
bound -- the qualitative "depth-optimal beats Algorithm 2 and the gap grows
with n" claim is intact; the exact asymptotic is not independently confirmed.

Depth preview (basis gates cx/rz/sx/x/h/reset/measure, optimization_level=1):

| n | k | depth-opt qubits | depth-opt depth | Alg. 2 qubits | Alg. 2 depth |
|---|---|---|---|---|---|
| 8 | 4 | 16 | 49 | 12 | 67 |
| 16 | 5 | 32 | 69 | 21 | 103 |
| 32 | 6 | 64 | 92 | 38 | 163 |

**Update: now wired into a mismatch circuit.** `src/mismatch_depth_optimal.py`
builds `mismatch_circuit_depth_optimal`, the same verified-identical
load/XOR/OR-flag prefix as `mismatch_circuit`/`mismatch_circuit_adder`,
counted with `depth_optimal_weight`. The MPS caveat above does NOT apply
here: mismatch circuits only ever load classical read/window strings via X
gates, never superpositions, and MPS is exact for basis-state inputs to this
counter (exhaustively verified). Tests: matches the classical count, and
agrees with the phase counter on all `2^L` flag patterns for L=1..5.

Resource preview (transpiled depth, no noise yet):

| L | depth-opt qubits | depth-opt depth | adder qubits | adder depth | phase qubits | phase depth |
|---|---|---|---|---|---|---|
| 2 | 8 | 26 | 8 | 24 | 8 | 36 |
| 4 | 16 | 39 | 16 | 61 | 15 | 54 |
| 6 | 24 | 41 | 22 | 117 | 21 | 60 |
| 8 | 32 | 55 | 30 | 192 | 28 | 78 |

The depth-optimal counter has the **lowest transpiled depth of all three at
every L from 4 up**, and the gap grows with L (at L=8: 55 vs phase's 78 vs
adder's 192) -- consistent with it being the one circuit here specifically
built to minimize depth. It costs qubits comparable to the adder (both need
extra ancilla beyond the phase counter's minimal k+n).

Still not done: a noise comparison. This is a depth/qubit preview only,
using the same caveat as every other preview in this project -- whether
lower depth translates into better noise robustness (it plausibly does,
since S2's thermal relaxation accumulates with circuit time) is an empirical
question the C4 harness could answer directly, not yet run for this third
counter.

## Post-selection (task 3, first piece)

`src/experiments/post_selection.py` computes `P(correct | in-range) =
p_correct / (1 - p_out_of_range)` directly from the existing
`noise_degradation_*.csv` and `counter_comparison_*.csv` columns -- no new
circuit runs, since every correct outcome is already in-range by
construction (`P(correct AND in-range) = P(correct)`). This is the
post-selection analysis "Reading p_out_of_range" above said had not been
measured yet.

Real, modest gains, tracking the codespace-ceiling pattern from that
section (larger wasted codespace -> more error-detection signal -> bigger
gain): at L=8, p=0.01, post-selection recovers 0.347 -> 0.446 (+0.099) at
the cost of discarding 22.1% of shots. At L=6 (smallest wasted codespace,
12.5%) the gain is much smaller: 0.521 -> 0.548 (+0.027) at only 4.9%
discard. Full table, all L and p, in the script's own output.

Applied to the C4 adder-vs-phase data: post-selection helps BOTH counters,
but helps the phase counter more in absolute terms (L=4, p=0.01: adder
+0.033, phase +0.085) -- consistent with the phase counter's k-qubit
register also wasting codespace. It does not change which counter wins at
any L or p already reported: the L=6 crossover (adder ahead at low p, phase
ahead from p=0.005 up) is unchanged after post-selection.

## Runs to date

| file | grid | rz convention | cells | runtime |
|---|---|---|---|---|
| `noise_degradation_2026-09-20.csv` | L = 2, 4, 6 | legacy (rz noisy) | 240 | 19.0 min |
| `noise_degradation_2026-09-20_L8.csv` | L = 8 | legacy (rz noisy) | 144 | 38.6 min |
| `noise_degradation_2026-09-21_virtualrz.csv` | L = 2, 4, 6 | virtual_rz (default) | 240 | 16.8 min |
| `noise_degradation_2026-09-22_virtualrz_L8.csv` | L = 8 | virtual_rz (default) | 144 | 54.6 min |

All four use the same scenarios, `p` grid, seeds, 8 pairs per m and 1024 shots, so
each pair (legacy / virtualrz) concatenates directly. `--tag` keeps same-day and
same-convention runs from overwriting each other. **The virtualrz files are the
current numbers** -- see "rz handling changed between runs" in Limitations below
before comparing across the two conventions.

The L=8 virtualrz run needed two retries: the first attempt (started 2026-09-21)
was interrupted by a session restart; the second crashed with a genuine
`MemoryError: bad allocation` inside Aer's C++ backend at 134/144 cells,
concurrent with other memory-heavy work in the same session. Both failures are
why `run_sweep` now writes and flushes each CSV row immediately rather than
batching the whole run in memory -- see the docstring in
`src/experiments/noise_degradation.py`. `meta.json` is written only on full
success, so a CSV without a matching `meta.json` next to it is an incomplete run
and should not be used.

### Old-vs-new: effect of virtual_rz on P(correct), mean over m, S2 scenario

| L | p | legacy (rz noisy) | virtual_rz | delta |
|---|---|---|---|---|
| 2 | 0.0 | 0.904 | 0.915 | +0.011 |
| 2 | 0.001 | 0.890 | 0.904 | +0.014 |
| 2 | 0.01 | 0.787 | 0.812 | +0.025 |
| 4 | 0.0 | 0.805 | 0.816 | +0.011 |
| 4 | 0.001 | 0.779 | 0.793 | +0.014 |
| 4 | 0.01 | 0.577 | 0.606 | +0.029 |
| 6 | 0.0 | 0.750 | 0.770 | +0.019 |
| 6 | 0.001 | 0.718 | 0.737 | +0.019 |
| 6 | 0.01 | 0.496 | 0.521 | +0.025 |
| 8 | 0.0 | 0.619 | 0.647 | +0.027 |
| 8 | 0.001 | 0.578 | 0.601 | +0.023 |
| 8 | 0.01 | 0.319 | 0.347 | +0.029 |

The shift at `p=0` is small (+0.011 to +0.027, growing with L because rz's
*absolute* count grows even though its *share* of gates stays ~49% at every L) --
this isolates rz's thermal-relaxation contribution, since S1 depolarizing is off
at p=0. The larger shift at `p=0.01` (+0.023 to +0.029) is rz's depolarizing
contribution, which only exists once gate error is present. Both are consistent
with the direct sensitivity check in Limitations below (rz fully noiseless at
S2 p=0 recovers +0.011/+0.011/+0.020 for L=2/4/6): that check used a slightly
different construction (only S2's thermal term toggled, not the full
`virtual_rz` flag) but lands on the same numbers, which is the cross-check that
matters.

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
