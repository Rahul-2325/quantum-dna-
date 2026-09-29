# Project Status: Noise-Aware Quantum DNA Mismatch Counting

**Last updated:** 2026-09-29 (Grover oracle verified). This document explains what has been built, verified,
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

Every claim below has an automated test behind it (currently **100 tests**,
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

### 4.3 Grover-oracle: the hard half is built and verified

QShift-SA's approach searches over *all* possible read alignment positions
at once using Grover's algorithm, rather than checking each position one at
a time (which is what every one of our four counters currently does). Nobody
has combined a *phase-kickback* counter with this kind of search.

This is no longer just a feasibility probe — the full ORACLE (the part of
Grover's algorithm specific to this problem, as opposed to the generic
search machinery around it) is built and verified as real project code
(`src/grover_oracle.py`, `tests/test_grover_oracle.py`, 7 tests). It does
three things coherently, in one circuit: counts mismatches at *every*
candidate shift position simultaneously (while all positions are held in
quantum superposition together), marks shifts whose count is at or below a
chosen threshold with the phase flip Grover's algorithm needs, and cleans up
every register used along the way. Both things that could plausibly have
gone wrong were checked directly rather than assumed: (1) every register
except the "which shift" register returns to a single clean state with
probability exactly 1.0 after cleanup — no leftover entanglement — checked
across 6 different reference/read/threshold combinations; (2) the "which
shift" register's raw quantum amplitude (not just probability, which cannot
see a phase flip) carries the opposite sign for "good" vs. "bad" shifts,
exactly as the marking is supposed to produce.

What remains is the generic part: the "diffusion" step that amplifies the
marked shifts' likelihood of being observed, and repeating oracle+diffusion
the right number of times. That is standard, well-documented machinery, not
something this project needs to invent — but it is not yet built, so there
is not yet a full working search, only a verified-correct oracle for it to
be built on.

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

5. **The new QFT-adder counter (the main piece of this session's novelty
   work) does not outperform the existing three counters** under idealized
   connectivity — it loses at every read length and error rate tested,
   consistent with it having the highest gate count of the four. This is a
   real, useful negative result (it directly answers the question the
   literature search raised), not a failure of the work; the open question
   it leaves is whether that changes under realistic qubit connectivity,
   given finding #3 already showed connectivity can reverse a ranking.

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

Nothing. The four-way noise comparison (Section 4.1/5.5) completed on
2026-09-23 (960/960 rows, all sanity checks passed, 87.7 minutes) and is
committed. It was interrupted once mid-run by an environment restart before
that — no data was lost (the incremental-save design saved 584 of 960 rows
before the interruption), and it was simply restarted from scratch rather
than resumed, since the harness does not yet support resuming a partial grid.

## 8. Next steps, and why

In rough priority order:

1. **Re-run the four-way comparison under realistic (heavy-hex) connectivity**
   — now the top priority. Section 5's finding #3 already showed connectivity
   can reverse which counter wins; finding #5 (QFT-adder losing under
   idealized connectivity) has not yet been checked against that same effect,
   and it is the one piece of this session's headline novelty work still
   resting on an idealized noise assumption.
2. **Build the diffusion operator and amplification loop for Grover** — the
   oracle it would act on is now verified correct (Section 4.3), which is
   what makes this the natural next step rather than a leap of faith. This
   is standard, well-documented machinery, not a new risk area like the
   oracle was.
3. **Extend the missing L=8 legs** for both the plain and connectivity-aware
   four-way comparisons, to match the depth of data already collected for
   the original three-way and single-counter noise sweeps.
4. **Only after 1–3:** begin drafting the paper's results section, per this
   project's own working rule that no narrative claims should be written
   before their supporting data exists and is saved.

## 9. Where things live

- `src/` — all circuit and experiment code, one file per counter/technique.
- `tests/` — the verification suite (100 tests); run with `pytest -q` from the
  project's `.venv`.
- `results/` — every experiment's raw output (`.csv`), metadata (`.json`,
  including exact seeds and the git commit that produced it), and a detailed
  `README.md` explaining every column and every limitation.
- `figures/` — generated circuit diagrams and comparison plots.
- Every unit of work above corresponds to one or more Git commits (25 so
  far), each with a detailed message explaining what changed, what was
  verified, and why — visible via `git log`.
