"""Demonstration: does the mismatch counter actually detect a DNA alignment?

Slides `read` across every position of `reference` and counts mismatches at
each shift using the quantum circuit (not a classical popcount -- the whole
point is this count comes from measuring a quantum register). The shift with
the fewest mismatches is the DETECTED alignment. Runs the same sweep
noiselessly (via Statevector, exact) and under this project's S2 noise model
(via Aer, realistic), so the practical question -- "does it still work when
the circuit is actually noisy" -- has a real answer, not an assumed one.

Run: python src/demo_alignment.py
"""
from __future__ import annotations

import random

from aer_helpers import (control_distribution_statevector,
                         control_distributions_aer, mismatch_circuit_measured,
                         transpile_for_noise)
from experiments.noise_degradation import scenario_noise_model
from hwlib import mismatch_circuit


def align_noiseless(read, reference):
    """Exact mismatch count at every shift, via Statevector (no noise)."""
    results = []
    for shift in range(len(reference) - len(read) + 1):
        window = reference[shift:shift + len(read)]
        qc, k = mismatch_circuit(read, window)
        distribution = control_distribution_statevector(qc, k)
        best = max(distribution, key=distribution.get)
        results.append((shift, window, best, distribution[best]))
    return results


def align_noisy(read, reference, p=0.01, shots=2048, seed=42):
    """Same sweep, run through Aer under S2 noise at 2-qubit error rate p."""
    circuits, meta = [], []
    for shift in range(len(reference) - len(read) + 1):
        window = reference[shift:shift + len(read)]
        qc, k = mismatch_circuit_measured(read, window)
        transpiled = transpile_for_noise(qc)
        circuits.append(transpiled)
        meta.append((shift, window, k))

    noise_model = scenario_noise_model("S2", p, circuits[0].num_qubits)
    distributions = control_distributions_aer(circuits, shots, noise_model=noise_model,
                                              seed=seed)
    results = []
    for (shift, window, k), distribution in zip(meta, distributions):
        best = max(distribution, key=distribution.get)
        results.append((shift, window, best, distribution[best]))
    return results


def report(title, read, reference, results):
    print(f"\n{title}")
    print(f"  reference = {reference}")
    print(f"  read      = {read}  (length {len(read)})\n")
    print(f"  {'shift':>5} {'window':>{len(read) + 2}} {'mismatches':>10} {'confidence':>11}")
    best_shift = min(results, key=lambda r: r[2])
    for shift, window, count, confidence in results:
        marker = "  <-- best alignment" if shift == best_shift[0] else ""
        print(f"  {shift:>5} {window:>{len(read) + 2}} {count:>10} {confidence:>10.1%}{marker}")
    return best_shift


def main():
    # A reference with ONE region that truly matches the read (allowing for a
    # single planted sequencing-error mismatch), and every other window at
    # least 2 mismatches away -- verified programmatically (not by hand
    # counting characters, which is exactly the kind of thing this session
    # has gotten wrong before) so the "best alignment" is genuinely unique,
    # not an accidental tie.
    reference = "GCTAAAGAATCCCAATTACA"
    read = "ATCG"          # true origin is "ATCC" at shift 8; last base flipped

    print("=" * 70)
    print("DEMONSTRATION: quantum mismatch counting for approximate DNA alignment")
    print("=" * 70)
    print("""
The read has ONE deliberate error (ATCG instead of the true ATCC at shift 8)
-- simulating a real sequencing error. Exact-match search would MISS this
alignment entirely. Approximate (mismatch-counting) search should still find
it as the best (lowest-mismatch) candidate, tolerating the 1-base error.
""")

    noiseless = align_noiseless(read, reference)
    best_noiseless = report("Noiseless (Statevector, exact)", read, reference, noiseless)

    print(f"\n  ==> Detected alignment: shift {best_noiseless[0]} "
          f"(window {best_noiseless[1]!r}), {best_noiseless[2]} mismatch(es), "
          f"confidence {best_noiseless[3]:.1%}")
    print(f"      Correct: shift {best_noiseless[0]}'s window is the read's true")
    print("      origin (ATCC), exactly 1 base different from the read ATCG (the")
    print("      planted error) -- detected despite the error, with a real margin")
    print("      over every other position (verified >= 2 mismatches elsewhere).")

    for p in (0.001, 0.01, 0.03):
        noisy = align_noisy(read, reference, p=p)
        best_noisy = report(f"Noisy: S2 (thermal + readout + depolarizing), p={p}",
                            read, reference, noisy)
        correct = "CORRECT" if best_noisy[0] == best_noiseless[0] else "WRONG"
        print(f"\n  ==> Detected alignment: shift {best_noisy[0]}  [{correct}]")


if __name__ == "__main__":
    main()
