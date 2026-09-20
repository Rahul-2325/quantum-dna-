import pathlib
import random
import sys
from datetime import date

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from aer_helpers import (control_distribution_aer, control_distribution_statevector,
                         mismatch_circuit_measured, transpile_for_noise)
from experiments.noise_degradation import (cell_metrics, sample_stratified_pair,
                                           wilson_interval)
from hwlib import mismatch_circuit, read_control_value


def test_statevector_distribution_is_full_and_normalised():
    qc, k = mismatch_circuit("ACGT", "AGGT")
    distribution = control_distribution_statevector(qc, k)
    assert abs(sum(distribution.values()) - 1.0) < 1e-9
    assert max(distribution, key=distribution.get) == 1
    # read_control_value only reports the mode; the helper keeps every outcome.
    assert read_control_value(qc, k)[0] == 1


def test_aer_noiseless_matches_statevector():
    for read, window in (("ACG", "ACG"), ("ACG", "TCG"), ("ACGT", "TGCA")):
        exact_qc, k = mismatch_circuit(read, window)
        exact = control_distribution_statevector(exact_qc, k)
        measured_qc, _ = mismatch_circuit_measured(read, window)
        sampled = control_distribution_aer(transpile_for_noise(measured_qc),
                                           shots=2048, seed=7)
        for value in set(exact) | set(sampled):
            assert abs(exact.get(value, 0.0) - sampled.get(value, 0.0)) < 0.05


def test_measured_circuit_reproduces_classical_counts():
    rng = random.Random(0)
    for L in (2, 3, 4):
        for m in range(L + 1):
            read, window = sample_stratified_pair(L, m, rng)
            assert sum(a != b for a, b in zip(read, window)) == m
            qc, _ = mismatch_circuit_measured(read, window)
            distribution = control_distribution_aer(transpile_for_noise(qc),
                                                    shots=512, seed=11)
            assert distribution.get(m, 0.0) > 0.999


def test_sweep_cell_path_with_coverage_verification():
    """Exercise run_cell exactly as the sweep calls it, including the coverage check.

    Regression: the S1 p=0 control has an empty noise model by construction, so
    verifying coverage there used to abort the sweep on its first cell.
    """
    from experiments.noise_degradation import (P_GRID, is_noiseless_control,
                                               run_cell, scenario_noise_model)

    L = 2
    probe, _ = mismatch_circuit_measured("A" * L, "A" * L)
    for scenario in ("S1", "S2"):
        for p in (0.0, P_GRID[1]):
            noise_model = scenario_noise_model(scenario, p, probe.num_qubits)
            control = is_noiseless_control(scenario, p)
            assert bool(noise_model.to_dict()["errors"]) != control
            counts, per_pair, _ = run_cell(
                scenario, L, p, 1, pairs_per_m=1, shots=64,
                rng=random.Random(0), noise_model=noise_model,
                verify_coverage=not control)
            assert sum(counts.values()) == 64
            assert len(per_pair) == 1 and sum(per_pair[0].values()) == 64
            if control:
                assert counts == {1: 64}


def test_sweep_refuses_to_overwrite_existing_results(tmp_path):
    """A second run on the same date must not clobber the first run's data."""
    from experiments.noise_degradation import run_sweep

    existing = tmp_path / f"noise_degradation_{date.today().isoformat()}.csv"
    existing.write_text("precious,data\n1,2\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        run_sweep([2], pairs_per_m=1, shots=8, results_dir=tmp_path)
    assert existing.read_text(encoding="utf-8") == "precious,data\n1,2\n"

    # A tag names the run separately, so it writes without touching the original.
    run_sweep([2], pairs_per_m=1, shots=8, results_dir=tmp_path, tag="other")
    assert existing.read_text(encoding="utf-8") == "precious,data\n1,2\n"
    assert (tmp_path / f"noise_degradation_{date.today().isoformat()}_other.csv").exists()


def test_mps_matches_statevector():
    """The sweep runs on MPS; it must agree with dense statevector, noisy and clean."""
    from experiments.noise_degradation import scenario_noise_model

    for L, m in ((2, 1), (3, 2), (4, 2)):
        read, window = sample_stratified_pair(L, m, random.Random(L))
        qc, _ = mismatch_circuit_measured(read, window)
        transpiled = transpile_for_noise(qc)
        for noise_model in (None, scenario_noise_model("S2", 0.02, qc.num_qubits)):
            mps = control_distribution_aer(transpiled, shots=2048, seed=23,
                                           noise_model=noise_model,
                                           method="matrix_product_state")
            dense = control_distribution_aer(transpiled, shots=2048, seed=23,
                                             noise_model=noise_model,
                                             method="statevector")
            for value in set(mps) | set(dense):
                assert mps.get(value, 0.0) == pytest.approx(dense.get(value, 0.0),
                                                            abs=1e-12)


def test_stratified_sampling_plants_exact_mismatch_count():
    rng = random.Random(3)
    for L in (1, 2, 5, 8):
        for m in range(L + 1):
            for _ in range(5):
                read, window = sample_stratified_pair(L, m, rng)
                assert len(read) == len(window) == L
                assert sum(a != b for a, b in zip(read, window)) == m


def test_bias_shows_direction_where_mae_cannot():
    """Signed error distinguishes under- from over-counting; MAE cannot."""
    from experiments.noise_degradation import cell_metrics

    under = cell_metrics({4: 500, 2: 500}, L=8, m=4, trials=1000)
    over = cell_metrics({4: 500, 6: 500}, L=8, m=4, trials=1000)
    assert under["mae"] == pytest.approx(over["mae"])       # MAE is blind to sign
    assert under["bias"] == pytest.approx(-1.0)
    assert over["bias"] == pytest.approx(+1.0)
    assert cell_metrics({4: 1000}, L=8, m=4, trials=1000)["bias"] == 0.0


def test_pair_bootstrap_widens_with_between_pair_spread():
    """Pooled Wilson ignores between-pair variance; the pair bootstrap must not."""
    from experiments.noise_degradation import pair_statistics, wilson_interval

    shots = 1000
    # Same pooled mean (0.5), but one case is consistent and the other is split.
    consistent = [{0: 500, 1: 500} for _ in range(8)]
    split = [{0: shots} if i % 2 else {1: shots} for i in range(8)]

    agree = pair_statistics(consistent, m=0, shots=shots)
    disagree = pair_statistics(split, m=0, shots=shots)

    assert agree["pairs_used"] == disagree["pairs_used"] == 8
    assert agree["pair_sd"] == pytest.approx(0.0, abs=1e-12)
    assert disagree["pair_sd"] > 0.4
    assert (disagree["boot_hi"] - disagree["boot_lo"]) > (agree["boot_hi"] - agree["boot_lo"])

    # The pooled interval is identical for both, which is exactly the problem.
    pooled = wilson_interval(4000, 8000)
    assert pooled[1] - pooled[0] < disagree["boot_hi"] - disagree["boot_lo"]


def test_wilson_interval_brackets_estimate():
    low, high = wilson_interval(50, 100)
    assert low < 0.5 < high
    tight_low, tight_high = wilson_interval(5000, 10000)
    assert (tight_high - tight_low) < (high - low)
    assert wilson_interval(100, 100)[1] == pytest.approx(1.0, abs=1e-9)


def test_cell_metrics_flags_errors_and_thresholds():
    # L=4, true m=0, 90% correct, 10% reported as 3 mismatches.
    counts = {0: 900, 3: 100}
    row = cell_metrics(counts, L=4, m=0, trials=1000)
    assert row["p_correct"] == pytest.approx(0.9)
    assert row["mae"] == pytest.approx(0.3)
    assert row["p_out_of_range"] == 0.0
    # m=0 <= every threshold, so only false negatives are defined.
    assert row["fn_t0"] == pytest.approx(0.1)
    assert row["fn_t1"] == pytest.approx(0.1)
    assert row["fp_t0"] != row["fp_t0"]  # NaN

    # true m=3 above threshold t=1: mass at or below t counts as a false positive.
    row = cell_metrics({3: 800, 1: 150, 7: 50}, L=4, m=3, trials=1000)
    assert row["fp_t1"] == pytest.approx(0.15)
    assert row["p_out_of_range"] == pytest.approx(0.05)
    assert row["fn_t1"] != row["fn_t1"]  # NaN
