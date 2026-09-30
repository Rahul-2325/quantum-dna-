import pathlib
import sys

import numpy as np
import pytest
from qiskit.quantum_info import Operator, Statevector

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from grover_search import build_grover_diffuser, build_grover_search, optimal_iterations


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_diffuser_matches_explicit_inversion_about_the_mean(k):
    """2|s><s| - I, |s> = uniform superposition -- checked against the exact
    matrix, not just trusted from the standard H-X-mcx-X-H construction."""
    qc = build_grover_diffuser(k)
    actual = Operator(qc).data

    n = 2 ** k
    s = np.full((n, 1), 1 / np.sqrt(n))
    expected = 2 * (s @ s.T) - np.eye(n)

    assert np.allclose(actual, expected, atol=1e-9)


def test_optimal_iterations_matches_known_grover_case():
    # Textbook example: N=16, M=1 -> optimal iterations = round(pi/4 * 4) = 3.
    assert optimal_iterations(16, 1) == 3
    # No marked states or all marked: degenerate, must not divide by zero.
    assert optimal_iterations(16, 0) == 1
    assert optimal_iterations(16, 16) == 1


@pytest.mark.parametrize("reference,read,threshold", [
    ("GCTAAAGAATCC", "ATCC", 0),   # 9 shifts, k_shift=4 -> 7 leftover branches
    ("TACCGAT", "TA", 0),          # 6 shifts, k_shift=3 -> 2 leftover branches
])
def test_search_amplifies_good_shifts_and_never_marks_leftover_branches(reference, read, threshold):
    """The critical end-to-end check: with a non-power-of-two shift count
    (so leftover branches genuinely exist), Grover iterations must
    concentrate probability on the TRUE good shifts.

    Standard Grover dynamics treats every UNMARKED basis state symmetrically:
    all unmarked states start with equal amplitude and are never phase-
    flipped, so the well-known two-dimensional-subspace argument behind
    Grover's algorithm guarantees they carry EXACTLY equal probability to
    each other after any number of iterations -- not driven to zero, just
    kept equal. So the correctness property to check is not "leftover
    probability is ~0" (false in general: unmarked probability mass has to
    go somewhere and is never fully eliminated except at special N/M
    ratios) but "leftover branches are treated exactly like bad (unmarked)
    shifts, never privileged like good (marked) ones" -- which is exactly
    what the oracle's `valid` ancilla is for: without it, every leftover
    branch's unloaded read counts as zero mismatches and would be spuriously
    MARKED, breaking this per-state symmetry and riding along with the
    amplification instead of matching the bad shifts.
    """
    qc, shift, k_shift, shifts, good_shifts, iterations = build_grover_search(
        reference, read, threshold)
    assert good_shifts, "test case must have at least one genuinely good shift"
    assert len(shifts) < 2 ** k_shift, "test case must actually have leftover branches"

    state = Statevector(qc)
    dist = state.probabilities_dict(qargs=shift)

    leftover = [s for s in range(2 ** k_shift) if s not in shifts]
    bad_shifts = [s for s in shifts if s not in good_shifts]
    assert bad_shifts, "test case must have at least one genuinely bad (unmarked) real shift"

    good_prob = sum(dist.get(format(s, f"0{k_shift}b"), 0.0) for s in good_shifts)
    bad_prob = sum(dist.get(format(s, f"0{k_shift}b"), 0.0) for s in bad_shifts)
    leftover_prob = sum(dist.get(format(s, f"0{k_shift}b"), 0.0) for s in leftover)

    good_avg = good_prob / len(good_shifts)
    bad_avg = bad_prob / len(bad_shifts)
    leftover_avg = leftover_prob / len(leftover)

    assert leftover_avg == pytest.approx(bad_avg, abs=1e-9), (
        "leftover branches must carry EXACTLY the same per-state probability "
        f"as genuinely bad shifts (both unmarked): leftover_avg={leftover_avg}, "
        f"bad_avg={bad_avg}")
    assert good_avg > 10 * bad_avg, (
        f"expected strong amplification of the {len(good_shifts)} good shift(s) "
        f"after {iterations} iteration(s): good_avg={good_avg}, bad_avg={bad_avg}")
    assert good_prob > 0.8, (
        f"expected strong amplification of the {len(good_shifts)} good shift(s) "
        f"after {iterations} iteration(s), got P(good)={good_prob}")


def test_search_with_zero_iterations_is_just_flat_superposition():
    """Sanity check on the harness itself: forcing iterations=0 must skip the
    oracle/diffuser loop entirely and leave every real shift (not leftover)
    at the flat 1/2^k_shift starting probability."""
    reference, read, threshold = "TACCGAT", "TA", 0
    qc, shift, k_shift, shifts, good_shifts, _ = build_grover_search(
        reference, read, threshold, iterations=0)
    state = Statevector(qc)
    dist = state.probabilities_dict(qargs=shift)
    expected = 1 / (2 ** k_shift)
    for s in shifts:
        assert dist.get(format(s, f"0{k_shift}b"), 0.0) == pytest.approx(expected, abs=1e-9)
