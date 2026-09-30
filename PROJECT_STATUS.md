# Project Status: Noise-Aware Quantum DNA Mismatch Counting

**Last updated:** 2026-09-30 (Grover diffusion+search built and verified; connectivity-aware
4-way sweep completed; noise-aware Grover search experiment running). This document explains what has been built, verified,
and found so far, what is running right now, and what comes next — written to be
handed to someone (e.g. a supervisor) who was not in the room for the work itself.

---

## 1. The goal

Write a publishable paper on **noise-aware, low-ancilla mismatch counting for
approximate DNA read alignment**, using quantum circuits that count how many bases
differ between a short "read" and a window of a longer reference sequence — a
quantum version of Hamming distance, applied to genomics. A secondary thread
(`hwb`, hidden weighted bit) reuses the same core primitive for a different,
unrelated problem, as a resource-tradeoff case study.

## 2. The three reference papers, and their role here

| Paper | What it contributes | Role in this project |
|---|---|---|
| **Bravyi–Yoder–Maslov** (IEEE TC 2022) | An ancilla-free reversible/quantum circuit for `hwb`, and — the piece we actually use — a "plus-one" CNOT/Toffoli ripple-carry technique for computing Hamming weight | Source of one of our four mismatch-counting circuits (`plus_one_counter.py`) |
| **Rethinasamy–LaBorde–Wilde** (PRA 110, 052401) | A phase-kickback quantum circuit that measures Hamming weight coherently, including a depth-optimal variant using dynamic circuits | Source of our main counter (`weight_unitary` in `hwlib.py`, pre-existing) and of the depth-optimal rewrite (`depth_optimal_counter.py`, built this session) |
| **Jamaludin–Lee–Kim** (IEEE Access 2025) | DNA alignment on simulated noisy hardware, with an error-detection scheme and zero-noise extrapolation (ZNE) | The application domain and the noise-realism bar to match — **used only as a conceptual reference and comparison target, not reimplemented**; we do not copy its circuit, only borrow the idea that noise and mitigation need to be taken seriously |

**A fourth, independently discovered source:** a literature search this session
found **QShift-SA** (arXiv:2602.23848, Feb 2026), which counts mismatches using a
Draper/QFT-style adder combined with Grover search over read positions. This
became the direct motivation for two pieces of this session's novelty work (the
QFT-adder counter, and the Grover-oracle feasibility check) — see Section 4.

## 3. What is implemented and verified

Every claim below has an automated test behind it (currently **119 tests**,
`tests/`), and every result file has a matching `results/README.md` entry
documenting exactly what it measures and its limitations. Nothing here is
reported without having been checked against a known-correct answer first —
this discipline mattered in practice: several real bugs (wrong circuit
conventions, broken test harnesses, a memory-blowup bug) were caught and fixed
specifically because of this checking, not despite it. See Section 6.

### 3.1 Four independent ways to count mismatches

All four take the same input (a read and a reference window) and are checked
against each other and against the classical answer, exhaustively for small
sizes and via distribution comparisons for larger ones.

1. **Phase counter** (`hwlib.py`, pre-existing) — Paper 2's coherent phase-kickback
   + inverse-QFT technique. The project's original/main method.
2. **Adder counter** (`plus_one_counter.py` + `mismatch_adder.py`) — Paper 1's
   ripple-carry "plus-one" technique, adapted to DNA mismatch counting.
3. **Depth-optimal counter** (`depth_optimal_counter.py` + `mismatch_depth_optimal.py`)
   — a from-scratch rewrite of Paper 2's dynamic-circuit construction, which an
   earlier attempt (before this session) had gotten wrong (~8% success rate).
   Rebuilding it required finding and fixing two separate real bugs (a
   sign/ordering error in the classically-controlled correction, and an Aer
   simulator bug specific to this circuit's dynamic-circuit pattern) before it
   passed verification against random quantum superposition inputs, not just
   simple test cases.
4. **QFT-adder counter** (`qft_adder_counter.py` + `mismatch_qft_adder.py`) — a
   Draper/Fourier-domain sequential accumulator, the style QShift-SA uses. This
   closes a real gap: no prior work compares a phase-kickback counter, a
   ripple-carry counter, and a QFT-adder counter under one noise model.

### 3.2 Realistic noise modeling

`src/noise_models.py`, `src/experiments/noise_degradation.py`: parametric
depolarizing, thermal-relaxation, and readout-error noise, calibrated to be as
close to real hardware behavior as a simulation can be:

- `rz` gates are modeled as virtual (zero-duration, error-free), matching real
  IBM hardware, after we found and corrected an earlier overly-pessimistic
  assumption.
- Circuits are additionally re-simulated under a **realistic connectivity
  constraint** (a heavy-hex qubit layout, the same topology real IBM devices
  use) instead of assuming every qubit can talk to every other qubit —
  because that assumption is known to be unrealistic and we wanted to know
  whether it was hiding anything. It was (see Section 4).

### 3.3 Mitigation techniques

Three noise-mitigation strategies were implemented and measured, individually
and combined:

- **Flag post-selection** — discard measurement outcomes that are logically
  impossible, for free (a pure calculation on data already collected).
- **Truncated readout** — use a smaller output register to save circuit
  resources, at a documented resolution cost.
- **Zero-noise extrapolation (ZNE)** — deliberately run the circuit at
  amplified noise levels and extrapolate back to the zero-noise answer.
- **Combined (post-selection + ZNE stacked)** — the strongest result:
  recovers a measurement that was wrong more than half the time up to 94%
  accuracy against a known-correct answer of 100%.

## 4. Novelty work: what is genuinely new here, not just re-implemented

This is the part meant to differentiate the project from a re-implementation
exercise. Three pieces:

### 4.1 The first four-way noise comparison of counting methods

Nobody has published a noise-realistic comparison between a phase-kickback
counter, a ripple-carry adder counter, a depth-optimal counter, and a
QFT-adder counter for this problem. We now have one, completed
2026-09-23 (`results/counter_comparison_2026-09-23_4way_v2.csv`, 960 rows,
verified: all 60 noiseless control cells return exactly the correct answer).
Under idealized (all-to-all) connectivity, the new QFT-adder counter **never
wins** at any read length or error rate tested — it is the outright worst
counter for short reads, and stays near the bottom for longer ones. This
tracks its resource cost directly: it has the highest gate count of all four
counters at every length. Whether that holds up once realistic qubit
connectivity is factored in (see 4.2, where a *different* counter's ranking
did flip) is not yet tested — flagged as the natural next run, not assumed
either way.

### 4.2 Connectivity-aware re-simulation

Every published comparison of these circuit styles (including our own first
pass) implicitly assumes any qubit can interact with any other qubit for
free. Real hardware cannot. We re-ran the noise comparison under a realistic
heavy-hex qubit layout (generated natively, no real hardware account needed)
and found that it **reverses** which counter wins, not just makes everything
uniformly worse. This is a substantive, checkable finding, not a footnote.

### 4.3 Grover search: oracle, diffusion, and amplification loop, all built and verified

QShift-SA's approach searches over *all* possible read alignment positions
at once using Grover's algorithm, rather than checking each position one at
a time (which is what every one of our four counters currently does). Nobody
has combined a *phase-kickback* counter with this kind of search.

The full search is now built and verified as real project code
(`src/grover_oracle.py`, `src/grover_search.py`, `tests/test_grover_oracle.py`
+ `tests/test_grover_search.py`, 15 tests total). The ORACLE does three
things coherently, in one circuit: counts mismatches at *every* candidate
shift position simultaneously (while all positions are held in quantum
superposition together), marks shifts whose count is at or below a chosen
threshold with the phase flip Grover's algorithm needs, and cleans up every
register used along the way. On top of that, the DIFFUSION operator
(inversion about the mean) and the amplification loop (repeated
oracle+diffusion) are built and verified against the textbook formula and
against real amplification behaviour.

Two real bugs were found and fixed while building this, both the kind that
would have silently produced wrong results without the specific check that
caught them:

1. **Leftover shift-register branches were being spuriously marked "good".**
   Whenever the number of real alignment positions isn't itself a power of
   two (i.e. almost always), the shift register has to be padded to the next
   power of two, leaving "leftover" branches with no real alignment behind
   them. Because those branches never get a reference window loaded, their
   mismatch count comes out as zero by default — which the original oracle
   then marked as "good" for any non-negative threshold, exactly as if it
   were a real match. This wouldn't have shown up in the oracle-only tests
   (which only ever checked real shifts), but it would have silently broken
   every Grover amplification run whenever the shift count wasn't a power of
   two. Fixed with a persistent `valid` ancilla (computed the same way as the
   existing per-shift decoder, but left set through the marking step instead
   of immediately uncomputed) that gates the marking so leftover branches are
   never touched.
2. **The diffuser's own test caught two mistakes — one in the circuit, one in
   the test.** The textbook diffuser construction realized the correct
   operator up to an overall global phase of -1 (physically meaningless, but
   worth matching to the textbook formula exactly, since it's a citable
   circuit); fixed with an explicit `global_phase` correction. Separately, the
   first draft of the end-to-end amplification test asserted leftover
   branches must end up at ~zero probability — which is actually FALSE: standard
   Grover dynamics keeps every *unmarked* state (bad real shifts and leftover
   branches alike) at exactly equal, generally nonzero probability throughout,
   never driven to zero. The test was rewritten to check the property that
   actually matters (leftover and bad shifts carry identical per-state
   probability; good shifts carry far more), not a superficially plausible
   but wrong one.

Both things that could plausibly have gone wrong with the oracle itself were
checked directly rather than assumed: (1) every register except the "which
shift" register returns to a single clean state with probability exactly
1.0 after cleanup — no leftover entanglement — checked across 6 different
reference/read/threshold combinations; (2) the "which shift" register's raw
quantum amplitude (not just probability, which cannot see a phase flip)
carries the opposite sign for "good" vs. "bad" shifts. For the full search,
end-to-end amplification was checked directly against known-good/bad shifts
for cases with genuine leftover branches, confirming strong amplification of
the true good shift(s) and correct (non-privileged) treatment of leftover
branches.

**Open question, now being tested (Section 7): does this amplification
survive realistic noise?** The verification above is all at zero circuit
noise. A quick feasibility check found the amplification circuit needs
roughly an order of magnitude more CX gates than any of the four direct
counters at comparable read length (~1100 CX at the noiseless-optimal 2
iterations, vs. low hundreds for the direct counters) — enough, on a rough
estimate, to matter a great deal once the same depolarizing/thermal/readout
noise model used everywhere else in this project is applied. `src/experiments/grover_noise.py`
is built to measure this directly; see Section 7 for its current status.

## 5. Key findings so far, in order of how far they'd move the needle in a paper

1. **At L=8 (an 8-letter read) with a realistic gate error rate, the main
   phase-kickback counter is right only ~32–35% of the time**, uncorrected.
   This is the central motivation for the mitigation work: without
   correction, the circuit is not usable at that length under a realistic
   error rate.

2. **Combined mitigation (post-selection + ZNE) recovers that to ~94%**
   accuracy at the same read length and error rate, against a known-correct
   answer of 100%. A single mitigation technique alone (either one) recovers
   less; combining them compounds the benefit.

3. **Which counter is "best" depends entirely on whether qubit connectivity
   is realistic.** Under an idealized (all-to-all) simulation, the adder
   counter wins at short reads and the phase counter wins at longer ones;
   the depth-optimal counter never wins. Under a realistic (heavy-hex)
   connectivity constraint, that changes: the depth-optimal counter — the
   one that never won before — starts winning at several read
   lengths/error-rate combinations, because its shallower circuit has less
   room for the extra routing overhead real connectivity imposes. **A
   resource comparison done without a real connectivity model can reach the
   wrong conclusion about which circuit design is actually better.**

4. **A worked demonstration confirms the whole pipeline detects real,
   imperfect alignments** — a read with a deliberately planted single-base
   error was still correctly matched to its true origin in a longer
   reference, with a clear margin over every wrong position, at every noise
   level tested.

5. **The new QFT-adder counter does not outperform the existing three
   counters under idealized connectivity** — it loses at every read length
   and error rate tested, consistent with it having the highest gate count
   of the four. Finding #3 raised the open question of whether this survives
   realistic connectivity, the way depth-optimal's own all-to-all loss did
   not. **Answer: partially.** Under heavy-hex routing, qft-adder still never
   wins a single cell, but at L=6 it stops being the clear loser and closes
   most of the gap — beating the adder counter outright at L=6, p=0.01
   (0.276 vs. 0.255) and landing within 0.003 of all three other counters at
   L=6, p=0.05. At L=2 and L=4 it remains clearly last. Same qualitative
   story as depth-optimal's reversal (real routing overhead falls differently
   on high-CX-count circuits than the idealized picture suggests), but here
   it closes the gap rather than crossing all the way to a win.

## 6. Process notes worth knowing about (things caught and fixed, not swept under the rug)

- An early attempt at the depth-optimal counter passed every simple test but
  failed badly on genuine quantum superposition inputs — caught only because
  superposition testing (not just simple test cases) was used as the
  verification standard throughout.
- The QFT-adder counter appeared badly broken (up to 63% error) during
  development; the circuit was actually correct, and the *test script*
  checking it had a subtle ordering bug. Both were found and the fix applied
  to the test, not the (already-correct) circuit — a reminder that a failing
  check does not always mean the thing being checked is wrong.
- Extending the noise-comparison tooling to a realistic qubit layout initially
  caused a >1000x qubit-count blow-up (a bug in how the coupling map was
  constrained) that would have made every subsequent timing and result
  meaningless. Caught before any real results were produced from it.
- A crash-recovery design (results are saved to disk incrementally, one row
  at a time, rather than only at the end of a run) was added specifically
  after a long run was interrupted mid-way; it has already prevented losing
  hours of computed results more than once since.

## 7. What is running right now

The four-way connectivity-aware comparison (item 1 of the prior next-steps
list) completed successfully on 2026-09-30 (960/960 rows, 143.5 minutes,
zero crashes needed thanks to the new crash-resilient supervisor described
below) and is committed — see finding #5 above and `results/README.md`.

The Grover diffusion operator and amplification loop (prior item 2) are
also now built and verified — see Section 4.3.

**Currently running: `src/experiments/grover_noise.py`**, the noise-aware
Grover search experiment described at the end of Section 4.3 — does the
verified-correct amplification survive the same S1/S2 noise model used
throughout this project? Sweeps `iterations` (0-4) x the standard P_GRID,
on a fixed test case with one known good shift among six real shifts (two
leftover). Results not yet in; see `results/README.md` once complete.

Two reliability items built this session, both proven necessary in
practice, not speculative:
- **`counter_comparison.py --resume`**: the connectivity-aware 4-way sweep
  had crashed repeatedly (genuine, non-deterministic segfaults, confirmed by
  reproducing and ruling out any specific circuit as the cause) before this
  was built. A segfault kills the whole process, so no in-process
  try/except can catch and retry it — `--resume` lets a restarted process
  skip cells already written to the CSV instead of recomputing the whole
  grid.
- **`supervised_run.py`**: the outer loop that actually does the restarting
  — invokes `counter_comparison.py` as a subprocess, and relaunches it with
  `--resume` if it dies before writing `meta.json`, up to a bounded retry
  count. The 2026-09-30 four-way run needed zero retries once this was in
  place, but the machinery exists because three prior unsupervised attempts
  all crashed.

## 8. Next steps, and why

In rough priority order:

1. **Finish and analyze the noise-aware Grover search experiment** (running
   now, Section 7) — the natural next question now that the search itself
   is verified correct: does it survive realistic noise, or is its
   dramatically higher gate count (an order of magnitude more CX gates than
   any direct counter at comparable length) enough on its own to erase the
   amplification advantage at noise levels the direct counters handle
   comfortably? This is a genuinely new empirical question — not answered by
   QShift-SA or any of the three reference papers.
2. **Extend the missing L=8 legs** for both the plain and connectivity-aware
   four-way comparisons, to match the depth of data already collected for
   the original three-way and single-counter noise sweeps.
3. **Only after 1–2:** begin drafting the paper's results section, per this
   project's own working rule that no narrative claims should be written
   before their supporting data exists and is saved.

## 9. Where things live

- `src/` — all circuit and experiment code, one file per counter/technique.
- `tests/` — the verification suite (119 tests); run with `pytest -q` from the
  project's `.venv`.
- `results/` — every experiment's raw output (`.csv`), metadata (`.json`,
  including exact seeds and the git commit that produced it), and a detailed
  `README.md` explaining every column and every limitation.
- `figures/` — generated circuit diagrams and comparison plots.
- Every unit of work above corresponds to one or more Git commits, each with
  a detailed message explaining what changed, what was verified, and why —
  visible via `git log`.
