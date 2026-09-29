import csv
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "experiments"))

from counter_comparison import COUNTERS, run_comparison


def _run(tmp_path, levels, **kwargs):
    return run_comparison(levels, pairs_per_m=1, shots=64, results_dir=str(tmp_path),
                          tag="resumetest", **kwargs)


def test_resume_skips_completed_cells_and_appends_the_rest(tmp_path, monkeypatch):
    """A --resume run must not recompute cells already in the CSV, must not
    duplicate the header, and must end up with exactly the full grid's worth
    of rows -- this is the exact mechanism a crash-and-restart depends on.

    Simulates a crash honestly: patches run_cell to raise after a fixed
    number of calls (mimicking a segfault killing the process mid-grid,
    which is the actual failure mode this feature exists for), catches that
    exception the same way an outer supervisor script would see the process
    die, and checks the CSV is genuinely partial (no meta.json) before
    resuming.
    """
    from noise_degradation import P_GRID
    import counter_comparison as cc_module

    expected_per_counter = 2 * len(P_GRID) * 3   # 2 scenarios * 8 p-values * (L=2 -> 3 m-values)
    crash_after = expected_per_counter * 2 + 5   # partway through the 3rd of 4 counters

    real_run_cell = cc_module.run_cell
    calls = {"n": 0}

    def flaky_run_cell(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > crash_after:
            raise RuntimeError("simulated crash (stands in for a real segfault)")
        return real_run_cell(*args, **kwargs)

    monkeypatch.setattr(cc_module, "run_cell", flaky_run_cell)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(tmp_path, levels=[2])

    csv_path = tmp_path / "counter_comparison_2026-09-30_resumetest.csv"
    meta_path = tmp_path / "counter_comparison_2026-09-30_resumetest_meta.json"
    assert not meta_path.exists()  # partial run: no meta.json yet (matches the real convention)
    rows_after_crash = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert 0 < len(rows_after_crash) < expected_per_counter * len(COUNTERS)

    # "Restart": remove the fault injection and resume.
    monkeypatch.setattr(cc_module, "run_cell", real_run_cell)
    csv_path2, meta_path2 = _run(tmp_path, levels=[2], resume=True)
    assert csv_path2 == csv_path
    assert meta_path2.exists()   # full grid now complete

    final_rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(final_rows) == expected_per_counter * len(COUNTERS)

    # No duplicate header, no duplicate cells.
    keys = [(r["counter"], r["scenario"], r["L"], r["p"], r["m"]) for r in final_rows]
    assert len(keys) == len(set(keys)), "resume must not recompute/duplicate a completed cell"

    # Every counter present, each with its full share of rows.
    for counter in COUNTERS:
        assert sum(1 for k in keys if k[0] == counter) == expected_per_counter


def test_resume_with_no_existing_file_behaves_like_a_fresh_run(tmp_path, monkeypatch):
    first_counter = next(iter(COUNTERS))
    monkeypatch.setitem(sys.modules["counter_comparison"].__dict__, "COUNTERS",
                        {first_counter: COUNTERS[first_counter]})
    csv_path, meta_path = _run(tmp_path, levels=[2], resume=True)
    assert meta_path.exists()
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) > 0


def test_without_resume_a_second_run_still_refuses_to_overwrite(tmp_path, monkeypatch):
    first_counter = next(iter(COUNTERS))
    monkeypatch.setitem(sys.modules["counter_comparison"].__dict__, "COUNTERS",
                        {first_counter: COUNTERS[first_counter]})
    _run(tmp_path, levels=[2])
    with pytest.raises(FileExistsError):
        _run(tmp_path, levels=[2])   # no resume=True, no force=True
