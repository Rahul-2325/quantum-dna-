import pathlib
import random
import sys

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
