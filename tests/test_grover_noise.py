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


def test_theoretical_p_good_matches_known_grover_oscillation_values():
    """Closed-form sin^2((2t+1)*theta) for N=8, M=1 (theta=arcsin(1/sqrt(8))
    ~ 0.3614 rad) -- known values, including the over-rotation trough at
    t=4 that a zero-noise measurement in this experiment confirmed
    empirically (theory ~1.2%, measured ~1.0%, well within shot noise)."""
    n_total, n_marked = 8, 1
    assert gn.theoretical_p_good(n_total, n_marked, 0) == pytest.approx(0.125, abs=1e-6)
    assert gn.theoretical_p_good(n_total, n_marked, 1) == pytest.approx(0.78125, abs=1e-6)
    assert gn.theoretical_p_good(n_total, n_marked, 2) == pytest.approx(0.9453125, abs=1e-6)
    assert gn.theoretical_p_good(n_total, n_marked, 3) == pytest.approx(0.330078125, abs=1e-6)
    assert gn.theoretical_p_good(n_total, n_marked, 4) == pytest.approx(0.01220703125, abs=1e-6)


def test_run_sweep_matches_theory_at_zero_circuit_noise(tmp_path, monkeypatch):
    """At S1/p=0.0 (zero circuit noise, only shot noise), measured p_good
    must track the closed-form theory at EVERY iteration count -- including
    iterations past the optimum, where theory predicts a trough far below
    the flat baseline (the "over-rotation" phenomenon: running the correct
    algorithm for too many rounds can be worse than not searching at all).
    This replaces an earlier, wrong version of this test that assumed more
    iterations always finds the good shift -- false in general, and it
    crashed the real sweep the first time it ran against iterations=4."""
    monkeypatch.setattr(gn, "P_GRID", (0.0,))
    csv_path, meta_path = gn.run_sweep([0, 1, 2, 3, 4], results_dir=tmp_path,
                                       tag="tinytest3", shots=512)
    assert meta_path.exists()
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    noiseless = [r for r in rows if r["scenario"] == "S1"]
    assert len(noiseless) == 5
    # iterations=2 (the peak) must be clearly high; iterations=4 (the
    # trough) must be clearly low -- the oscillation itself, not just that
    # the harness ran without crashing.
    by_it = {int(r["iterations"]): r for r in noiseless}
    assert float(by_it[2]["p_good"]) > 0.8
    assert float(by_it[4]["p_good"]) < 0.1
