import csv
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "experiments"))

import grover_noise as gn


def test_cell_metrics_partitions_good_bad_leftover():
    # k_shift=3 -> 8 basis states; shifts=0..5 real, 6/7 leftover; good=[0]
    dist = {0: 0.5, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.05, 6: 0.025, 7: 0.025}
    metrics = gn.cell_metrics(dist, k_shift=3, shifts=list(range(6)), good_shifts=[0])
    assert metrics["p_good"] == pytest.approx(0.5)
    assert metrics["p_bad"] == pytest.approx(0.45)
    assert metrics["p_leftover"] == pytest.approx(0.05)
    assert metrics["top_shift_is_good"] == 1


def test_cell_metrics_top_shift_not_good():
    dist = {0: 0.1, 1: 0.5, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1}
    metrics = gn.cell_metrics(dist, k_shift=3, shifts=list(range(6)), good_shifts=[0])
    assert metrics["top_shift_is_good"] == 0


def test_run_sweep_writes_csv_and_meta_for_a_tiny_grid(tmp_path, monkeypatch):
    """Shrinks P_GRID to a single value so this stays fast (real Aer calls,
    but only 2 cells: S1/S2 at p=0.0, iterations=0 -- 0 CX gates, near-instant)."""
    monkeypatch.setattr(gn, "P_GRID", (0.0,))
    csv_path, meta_path = gn.run_sweep([0], results_dir=tmp_path, tag="tinytest", shots=64)
    assert meta_path.exists()
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 2  # 2 scenarios x 1 p-value x 1 iteration count
    for row in rows:
        assert row["iterations"] == "0"
        assert row["num_qubits"] == "14"


def test_run_sweep_refuses_to_overwrite_without_force(tmp_path, monkeypatch):
    monkeypatch.setattr(gn, "P_GRID", (0.0,))
    gn.run_sweep([0], results_dir=tmp_path, tag="tinytest2", shots=64)
    with pytest.raises(FileExistsError):
        gn.run_sweep([0], results_dir=tmp_path, tag="tinytest2", shots=64)


def test_run_sweep_enforces_noiseless_correctness_at_iterations_one(tmp_path, monkeypatch):
    """At iterations>=1, S1/p=0.0 (zero circuit noise) must still measure the
    true good shift as most likely -- this is grover_search.py's own
    noiseless guarantee, re-checked through the sampling harness."""
    monkeypatch.setattr(gn, "P_GRID", (0.0,))
    csv_path, meta_path = gn.run_sweep([1], results_dir=tmp_path, tag="tinytest3", shots=256)
    assert meta_path.exists()
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    noiseless = [r for r in rows if r["scenario"] == "S1"][0]
    assert noiseless["top_shift_is_good"] == "1"
