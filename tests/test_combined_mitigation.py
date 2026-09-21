import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "experiments"))

from combined_mitigation import raw_and_post_selected


def test_raw_and_post_selected_matches_direct_conditional_probability():
    """Same check as test_post_selection.py's formula test, but against this
    module's own per-scale helper (it recomputes p_out_of_range from a raw
    distribution rather than reading it from a CSV column, so it needs its
    own direct verification)."""
    # L=4: value 1 is correct (m=1), value 2 is wrong-but-in-range, value 6
    # is impossible (> L) and must be excluded from the post-selected mean.
    distribution = {1: 0.6, 2: 0.25, 6: 0.15}
    raw, post, discard = raw_and_post_selected(distribution, L=4, m=1)
    assert raw == pytest.approx(0.6)
    assert discard == pytest.approx(0.15)
    assert post == pytest.approx(0.6 / 0.85)


def test_raw_and_post_selected_handles_no_out_of_range_mass():
    distribution = {0: 0.7, 1: 0.2, 2: 0.1}
    raw, post, discard = raw_and_post_selected(distribution, L=4, m=0)
    assert discard == 0.0
    assert post == pytest.approx(raw)


def test_raw_and_post_selected_is_nan_when_everything_discarded():
    distribution = {5: 1.0}   # L=4, so 5 is out of range: everything discarded
    raw, post, discard = raw_and_post_selected(distribution, L=4, m=0)
    assert raw == 0.0
    assert discard == 1.0
    assert post != post   # NaN
