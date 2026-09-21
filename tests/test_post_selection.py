import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "experiments"))

from post_selection import with_post_selection


def test_post_selection_formula_matches_direct_conditional():
    """P(correct | in-range) computed via the shortcut formula must match a
    direct conditional-probability calculation from a synthetic outcome
    histogram, since correct outcomes are always in-range by construction."""
    # L=4, true m=1: 600 correct(=1), 250 wrong-but-in-range, 150 out-of-range.
    trials = 1000
    counts = {1: 600, 2: 250, 6: 150}          # value 6 > L=4, impossible noiselessly
    p_correct = counts[1] / trials
    p_oor = counts[6] / trials

    row = {"p_correct": str(p_correct), "p_out_of_range": str(p_oor)}
    post, discard = with_post_selection(row)

    in_range_trials = trials - counts[6]
    direct = counts[1] / in_range_trials
    assert post == pytest.approx(direct)
    assert discard == pytest.approx(p_oor)


def test_post_selection_never_decreases_accuracy():
    """Discarding only wrong answers can only help or leave accuracy unchanged."""
    for p_correct, p_oor in ((0.5, 0.1), (0.9, 0.01), (0.3, 0.4), (0.0, 0.0)):
        row = {"p_correct": str(p_correct), "p_out_of_range": str(p_oor)}
        post, _ = with_post_selection(row)
        assert post >= p_correct - 1e-12


def test_post_selection_is_nan_when_everything_is_out_of_range():
    row = {"p_correct": "0.0", "p_out_of_range": "1.0"}
    post, discard = with_post_selection(row)
    assert post != post   # NaN
    assert discard == 1.0
