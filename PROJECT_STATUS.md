# Project Status: Noise-Aware Quantum DNA Mismatch Counting

**Last updated:** 2026-09-23. This document explains what has been built, verified,
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

Every claim below has an automated test behind it (currently **93 tests**,
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

### 4.1 The first three-way (now four-way) noise comparison of counting methods

Nobody has published a noise-realistic comparison between a phase-kickback
counter, a ripple-carry adder counter, and a QFT-adder counter for this
problem. We now have one — see Section 5 for the headline finding, which is
genuinely surprising (a counter that looks worse on paper turns out to be
the best choice under realistic conditions).

### 4.2 Connectivity-aware re-simulation

Every published comparison of these circuit styles (including our own first
pass) implicitly assumes any qubit can interact with any other qubit for
free. Real hardware cannot. We re-ran the noise comparison under a realistic
heavy-hex qubit layout (generated natively, no real hardware account needed)
and found that it **reverses** which counter wins, not just makes everything
uniformly worse. This is a substantive, checkable finding, not a footnote.

### 4.3 Grover-oracle feasibility check (in progress)

QShift-SA's approach searches over *all* possible read alignment positions
at once using Grover's algorithm, rather than checking each position one at
a time (which is what every one of our four counters currently does). Nobody
has combined a *phase-kickback* counter with this kind of search. Before
committing to building the full search algorithm (a substantial undertaking),
we validated the hardest and riskiest sub-piece in isolation: can a circuit
correctly count mismatches at *every* candidate alignment position
*simultaneously*, while all positions are held in quantum superposition
together? A small test case (checking all 6 possible alignment positions of
a 2-letter read against a 7-letter reference, at once) came back with the
exactly correct mismatch count at every position, each with the exact
probability quantum mechanics predicts. This is real evidence the idea is
sound, though the full search algorithm (marking good matches, undoing
intermediate computation, and the repeated amplification steps Grover's
method requires) is not yet built.

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

A **four-way noise comparison** (adder vs. phase vs. depth-optimal vs.
QFT-adder, under realistic — not yet connectivity-constrained — noise) is
running in the background. It was interrupted once by an environment restart
(no data lost, thanks to the incremental-save design above) and has been
restarted. Expected runtime: roughly two hours from restart. Results will be
written to `results/counter_comparison_2026-09-23_4way_v2.csv` plus a
matching `_meta.json`, and reported here once complete.

## 8. Next steps, and why

In rough priority order:

1. **Finish and report the four-way comparison** (running now) — establishes
   whether the QFT-adder counter is competitive with the other three under
   realistic noise, closing the loop on the novelty work in Section 4.1.
2. **Re-run the four-way comparison under realistic connectivity** — given
   Section 5's finding #3, this is not optional if the four-way result is
   going to be trusted; connectivity has already been shown to change the
   ranking once.
3. **Decide on the Grover-oracle direction** — the feasibility check in
   Section 4.3 passed. The next decision is whether to invest in building the
   full search algorithm (a substantial piece of work: threshold marking,
   uncomputation, and the repeated amplification steps) now that the
   riskiest assumption behind it has been validated, or to prioritize
   finishing the more bounded four-way comparison work first.
4. **Extend the missing L=8 legs** for both the plain and connectivity-aware
   four-way comparisons, to match the depth of data already collected for
   the original three-way and single-counter noise sweeps.
5. **Only after 1–4:** begin drafting the paper's results section, per this
   project's own working rule that no narrative claims should be written
   before their supporting data exists and is saved.

## 9. Where things live

- `src/` — all circuit and experiment code, one file per counter/technique.
- `tests/` — the verification suite (93 tests); run with `pytest -q` from the
  project's `.venv`.
- `results/` — every experiment's raw output (`.csv`), metadata (`.json`,
  including exact seeds and the git commit that produced it), and a detailed
  `README.md` explaining every column and every limitation.
- `figures/` — generated circuit diagrams and comparison plots.
- Every unit of work above corresponds to one or more Git commits (25 so
  far), each with a detailed message explaining what changed, what was
  verified, and why — visible via `git log`.
