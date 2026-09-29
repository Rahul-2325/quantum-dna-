import pathlib
import sys

import pytest
from qiskit.quantum_info import Statevector

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from grover_oracle import build_grover_oracle


def _true_mismatch(read, reference, shift):
    window = reference[shift:shift + len(read)]
    return sum(a != b for a, b in zip(read, window))


@pytest.mark.parametrize("reference,read,threshold", [
    ("TACCGAT", "TA", 0),
    ("TACCGAT", "TA", 1),
    ("TACCGAT", "TA", 2),
    ("GCTAAAGAATCC", "ATCC", 0),
    ("GCTAAAGAATCC", "ATCC", 1),
    ("AAAA", "A", 0),
])
def test_oracle_marks_good_shifts_and_cleans_up_everything_else(reference, read, threshold):
    """The two things a Grover oracle must do, both checked directly:

    1. Every register except the shift register returns to a SINGLE clean
       computational basis outcome with probability 1.0 -- zero leftover
       entanglement, which is what lets Grover's diffusion step interfere
       correctly on the shift register alone.
    2. The shift register's raw COMPLEX AMPLITUDE (not probability, which
       is phase-blind and cannot see a sign flip) carries one consistent
       sign for every shift whose true mismatch count is <= threshold, and
       the opposite consistent sign for every shift whose true count is
       > threshold.
    """
    qc, shift, k_shift, shifts = build_grover_oracle(reference, read, threshold)
    state = Statevector(qc)

    non_shift = list(range(k_shift, qc.num_qubits))
    non_shift_dist = state.probabilities_dict(qargs=non_shift)
    nonzero = {bits: p for bits, p in non_shift_dist.items() if p > 1e-9}
    assert len(nonzero) == 1, (
        f"expected exactly one clean non-shift outcome, got {len(nonzero)}: {nonzero}")
    clean_bits = next(iter(nonzero))
    assert nonzero[clean_bits] == pytest.approx(1.0, abs=1e-9)

    full = state.data
    good_signs, bad_signs = set(), set()
    for s in shifts:
        index = int(clean_bits + format(s, f"0{k_shift}b"), 2)
        amplitude = full[index]
        assert abs(amplitude) > 1e-9, f"shift {s} has near-zero amplitude, expected nonzero"
        sign = 1 if amplitude.real > 0 else -1
        if _true_mismatch(read, reference, s) <= threshold:
            good_signs.add(sign)
        else:
            bad_signs.add(sign)

    assert len(good_signs) == 1, f"good shifts should share one sign, got {good_signs}"
    assert len(bad_signs) <= 1, f"bad shifts should share one sign, got {bad_signs}"
    if bad_signs:
        assert good_signs != bad_signs, "good and bad shifts must carry OPPOSITE signs"


def test_oracle_is_flat_probability_across_shifts_before_marking_bias():
    """Marking must change PHASE only, never the magnitude -- every shift
    should still carry equal probability mass after the full oracle, exactly
    matching the equal-superposition input (this is what "phase oracle,
    not amplitude-changing" means, and is what makes Grover's later
    amplification the only place probability actually shifts)."""
    reference, read, threshold = "TACCGAT", "TA", 1
    qc, shift, k_shift, shifts = build_grover_oracle(reference, read, threshold)
    state = Statevector(qc)
    dist = state.probabilities_dict(qargs=shift)
    valid_probs = [p for bits, p in dist.items() if int(bits, 2) in shifts]
    assert len(valid_probs) == len(shifts)
    expected = 1 / (2 ** k_shift)
    for p in valid_probs:
        assert p == pytest.approx(expected, abs=1e-9)
