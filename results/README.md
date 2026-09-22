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

## Connectivity-aware resimulation: a real reversal, not just an extension

`results/counter_comparison_2026-09-22_hex.csv`, 720 rows: the exact same
3-way comparison as immediately below, with ONLY the transpile target
changed -- `src/connectivity.py`'s heavy-hex coupling map
(`CouplingMap.from_heavy_hex(5)`, 57 qubits, generated natively by Qiskit,
no IBM account or hardware access needed) instead of all-to-all. Same
scenarios/p-grid/seeds/pairs/shots. All 45 S1 p=0 control cells exactly
1.0. `figures/7_counter_comparison_noise_hex.png`.

**Building this exposed a serious bug before any real numbers came out of
it.** Handing the full 57-qubit map straight to `transpile(coupling_map=...)`
makes the transpiled circuit as wide as the WHOLE map, not just the qubits
the logical circuit needs -- an 8-qubit circuit silently became a 57-qubit
one. This blew up MPS (over 1GB RAM and still climbing, had to be
force-killed) and would have made dense statevector uncomputable (2^57).
Fixed in `connectivity.py` by extracting a connected N-qubit subgraph via
BFS + `CouplingMap.reduce()` instead of handing over the full map -- pinned
by a regression test so this can't silently recur.

**Result: this reverses the previous conclusion, not just extends it.**
Under all-to-all routing, depth-optimal never won a single cell (see the
section immediately below). Under realistic heavy-hex routing, it wins
several: L=4 at p=0.01/0.02/0.05, and L=6 at p=0.02/0.05. Everyone gets
worse under real routing, as expected, but depth-optimal degrades LEAST:
at L=6, p=0.01, adder drops -0.243 and phase drops -0.224 from their
all-to-all values, while depth-optimal drops only -0.181. Its already-lower
depth (see the earlier resource-only preview) leaves less room for routing
to add further depth on top -- the property that looked irrelevant under
the idealized comparison turns out to be the deciding factor once realistic
connectivity is accounted for.

Full picture per L (S2, mean over m):

| L | p | adder | phase | depth-optimal | winner |
|---|---|---|---|---|---|
| 2 | 0.01 | 0.816 | 0.727 | 0.795 | adder |
| 2 | 0.05 | 0.535 | 0.413 | 0.524 | adder |
| 4 | 0.01 | 0.459 | 0.416 | **0.485** | depth-optimal |
| 4 | 0.05 | 0.173 | 0.150 | **0.181** | depth-optimal |
| 6 | 0.01 | 0.256 | **0.297** | 0.292 | phase (barely) |
| 6 | 0.05 | 0.130 | 0.131 | **0.134** | depth-optimal |

Not yet done: L=8 for this comparison, and whether a different heavy-hex
distance/layout choice or a different SABRE seed changes these specific
crossover points (the qualitative reversal is unlikely to be a seed
artifact given the consistent pattern across 4 separate (L,p) cells, but
the exact numbers have not been checked for seed sensitivity).

## 3-way noise comparison: depth-optimal added (item 1 of the follow-up list)

`results/counter_comparison_2026-09-22_3way.csv`, 720 rows: the C4 harness
extended to all three counters (adder, phase, depth-optimal) at L=2,4,6.
Same scenarios/p-grid/seeds/pairs/shots as the 2-way run. All 45 S1 p=0
control cells are exactly 1.0. Figure:
`figures/6_counter_comparison_noise_3way.png`.

**Depth-optimal's depth advantage does not translate into a noise-robustness
advantage -- it loses at every (L, p) tested here, never winning even once.**
The adder wins every point at L=2 and L=4; at L=6 the earlier
adder/phase crossover survives with depth-optimal added, and depth-optimal
sits at or below both of them throughout (e.g. L=4, p=0.01: adder 0.677,
phase 0.606, depth-optimal 0.597; L=6, p=0.01: phase 0.521, adder 0.499,
depth-optimal 0.473).

The likely reason, and it is a real, checkable trade-off rather than a
guess: `mismatch_circuit_depth_optimal` needs a FULL `L`-qubit control
register (reused/reset across rounds, but still `L` physical qubits present
throughout), not just the phase counter's `k = ceil(log2(L+1))` qubits --
at L=8 the bare counter alone needs 16 qubits (2x its input size) against
the phase counter's 12 and the adder's 14 (see the depth-optimal resource
table above). More physical qubits under S2's per-qubit thermal relaxation
and readout error is more surface area for noise to act on, and here that
appears to outweigh the benefit of a shallower circuit. This is a concrete
illustration that "lower depth" and "more noise-robust" are not the same
claim, and a resource preview using depth/CX counts alone (as the earlier
sections did, honestly, before this run existed) can be misleading about
which counter would actually perform better on noisy hardware.

Not yet done: L=8 for this 3-way comparison, and a check of whether this
holds under the connectivity-aware (heavy-hex) transpilation discussed
earlier -- routing overhead could plausibly change this picture again,
since it would fall differently on circuits with different qubit counts.

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

## Truncated IQFT (task 3, second piece)

`src/truncated_weight.py`: `weight_unitary_truncated(n, k_used)`, the same
construction as `weight_unitary` but with a `k_used`-qubit control register
instead of the full `k = ceil(log2(n+1))`.

**Which reading of "truncated IQFT" this is, and why.** Two standard
readings exist. The more literal one -- dropping the smallest-angle
controlled-phase gates from inside `QFTGate`'s own internal decomposition --
was attempted first and abandoned: matching Qiskit's exact internal
convention (swap placement, control/target roles, angle signs) by hand
proved error-prone within the time available (multiple non-matching
attempts against `Operator` comparison), and shipping a circuit that LOOKS
like a truncated QFT but silently computes something else is exactly the
failure mode task 2 already hit once this session -- task 2's fix required
extensive superposition testing to catch, and there wasn't budget to repeat
that level of verification here. This module implements the other standard
reading instead: fewer control qubits, a genuinely smaller internal IQFT.

**Verified, not assumed:** `k_used` equal to the full `k` reproduces
`weight_unitary` exactly (all n=3,5,7, exhaustive). For `k_used` less than
the full `k`, the result **aliases exactly `(true weight) mod 2^k_used`**,
deterministically (checked exhaustively for n=7, k_used=1,2) -- this is a
clean modular wraparound, not a rounded approximation or a mix of
candidates. One direct consequence, also verified: for true weights strictly
below `2^k_used`, truncation gives the exact true weight with **no**
aliasing, which is what would make a truncated register usable for
low-threshold screening (the `fn_t0`/`fn_t1` metrics already in this
project) PROVIDED the true weight is known to stay under `2^k_used` --
aliasing a genuinely high weight down into the "looks low" range would
silently create new false negatives, so this is not a safe substitute for
the full register without that guarantee.

Resource savings from dropping just one qubit (`k_used = k - 1`):

| n | k (full) | full CX | full depth | truncated CX | truncated depth | CX savings |
|---|---|---|---|---|---|---|
| 4 | 3 | 33 | 43 | 21 | 31 | 36.4% |
| 6 | 3 | 45 | 49 | 29 | 36 | 35.6% |
| 8 | 4 | 82 | 67 | 57 | 55 | 30.5% |
| 16 | 5 | 186 | 103 | 146 | 91 | 21.5% |

Not yet done: a noise comparison (does the CX/depth saving translate into
better accuracy under S2, for the specific use case of low-threshold
screening where aliasing is provably harmless below `2^k_used`?) and the
CPhase-dropping interpretation, abandoned above for lack of verification
budget -- flagged as open, not silently dropped.

## Zero-noise extrapolation (task 3, third piece)

`src/zne.py`: global unitary folding (`U -> U(U^dagger U)^m` for odd scale
`2m+1`; scale=1 is the circuit unmodified) plus two extrapolation methods
(`linear_extrapolate`, `exponential_extrapolate`, the latter via
`scipy.optimize.curve_fit` rather than a hand-derived closed-form formula --
an earlier attempt at a 3-point closed-form exponential formula was
abandoned mid-derivation as too easy to get subtly wrong, the same concern
that shaped the truncated-IQFT scope decision above). Folding is verified to
exactly preserve the circuit's unitary at every scale tested
(`Operator(folded).equiv(Operator(original))`, scale = 1,3,5,7) while
scaling gate count exactly linearly with `scale`.

**Real, substantial recovery -- exponential extrapolation far outperforms
linear.** Measured at S2, scale = 1, 3, 5, comparing against this project's
own known ground truth (S1 p=0 gives exactly 1.0, so "how close does the
scale=0 estimate get to 1.0" is a real check, not a plausibility guess):

| L | p | raw (scale=1) | scale=3 | scale=5 | linear ZNE | exponential ZNE |
|---|---|---|---|---|---|---|
| 4 | 0.01 | 0.604 | 0.362 | 0.245 | 0.673 | **0.811** |
| 6 | 0.01 | 0.509 | 0.257 | 0.178 | 0.563 | **0.796** |
| 8 | 0.01 | 0.351 | 0.115 | 0.082 | 0.384 | **0.814** |
| 4 | 0.02 | 0.456 | 0.217 | 0.165 | 0.498 | **0.802** |

Linear extrapolation gives only a small, honest improvement (it systematically
under-corrects, since the decay visibly isn't linear in scale -- each
successive drop is larger than the last, e.g. L=8: -0.236 then -0.033, not a
constant slope). Exponential extrapolation is dramatically closer to the true
value at every point tested, more than doubling raw accuracy at L=8
(0.351 -> 0.814). It does not fully reach 1.0, which is expected: ZNE
approximates the noiseless limit from a small number of scaled measurements,
it does not reconstruct it exactly, and the true noise process likely isn't a
single clean exponential either.

**Cost:** folding at scale=5 quintuples circuit depth (and gate count), so
each ZNE estimate costs `sum(scales)`-times the circuits of a single raw
measurement (1+3+5=9x here) -- a real resource cost for a real accuracy gain,
not a free lunch.

Not yet done: a full sweep across the (scenario, L, p, m) grid (this is a
small, targeted probe on a handful of points, not a systematic run), and a
side-by-side comparison against post-selection and truncated IQFT's own
cost/benefit numbers above.

## Amplitude-damping bias, corrected

The `bias` column was built (see the commit that introduced it) specifically
to test one hypothesis: that amplitude damping pulls the control register
toward `|0>`, so low counts would read low and the measured count would be
systematically UNDER-reported (`bias < 0`) at every true weight `m`. That
was never actually checked until now. **It is wrong.**

The real pattern, checked against `noise_degradation_2026-09-21_virtualrz.csv`
and the L=8 file: `bias` is strongly **positive** for small true `m` and
**negative** for large true `m`, crossing zero not near `m=0` but near the
**center of the full `2^k` codespace**, `(2^k-1)/2` -- for L=6 (k=3, codespace
0..7, center 3.5) the crossing sits exactly between m=3 (+0.14) and m=4
(-0.14); for L=8 (k=4, codespace 0..15, center 7.5) it sits between m=7
(+0.54) and m=8 (-0.43). This is regression toward a UNIFORM distribution over
the whole representable range, not a directional pull toward zero. Fit at the
strongest noise tested (S2, p=0.05): `bias` regressed against
`(codespace_center - true_m)` gives a slope of 0.58 (L=2) rising to 0.89
(L=8) -- consistent with the measured distribution moving toward (but not
reaching) uniform as depolarizing/thermal/readout noise accumulates, with
slope 1.0 meaning fully random output and zero remaining signal.

This makes sense in hindsight: the dominant noise channels here
(depolarizing, thermal relaxation applied per-gate, symmetric readout error)
have no reason to prefer `|0>` over any other computational basis state --
depolarizing noise is symmetric by construction, and there is nothing in S1
or S2 that biases toward `|0>` specifically. A pull toward `|0>` would be the
signature of a *dominant, uncorrected* amplitude-damping channel, which this
project's noise model does not include as a separate term (see the gate
treatment in "How build_noise_model treats each gate type" above). The
"regression toward the codespace center" pattern is the generic signature of
noise that flattens the outcome distribution, which is what depolarizing
noise does.

## Combined mitigation: post-selection stacked with ZNE (task 3, extension)

`src/experiments/combined_mitigation.py`: at EACH ZNE noise scale (1, 3, 5),
discard out-of-range shots before computing that scale's P(correct), then
exponentially extrapolate the resulting POST-SELECTED series to scale=0 --
rather than extrapolating the raw series, which is what the standalone ZNE
probe above did. Whether stacking actually beats either technique alone
(rather than, say, folding's higher out-of-range rate at large scales eating
into the sample used for extrapolation) is checked directly below, not
assumed.

**It does compose, and substantially so.** Same probe points as the
standalone ZNE section above:

| L | p | raw | ZNE alone | **post-select + ZNE** |
|---|---|---|---|---|
| 4 | 0.01 | 0.604 | 0.811 | **0.878** |
| 6 | 0.01 | 0.509 | 0.796 | **0.821** |
| 8 | 0.01 | 0.351 | 0.814 | **0.941** |
| 4 | 0.02 | 0.456 | 0.802 | **0.935** |

At L=8, p=0.01, stacking gets to 0.941 against a true value of 1.0 -- within
6 percentage points, from a raw measurement that was wrong more than half the
time. Discard rates grow with fold scale as expected (more folding exposes
more noise, so more impossible outcomes to filter): at L=8 the rate goes
20% -> 34% -> 39% across scales 1/3/5, and post-selecting BEFORE
extrapolating gives the fit a cleaner per-scale signal to work with than
extrapolating the raw, noise-inflated series would.

Same scope caveat as the standalone pieces above: this is a small, targeted
probe (4 points), not a systematic grid sweep.

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
| `bias` | mean **signed** error, `sum p(v) * (v - m)`. `mae` cannot show direction; this can. Its actual pattern is not what a naive "amplitude damping pulls toward \|0>" guess predicts -- see "Amplitude-damping bias, corrected" below before interpreting it. |
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
