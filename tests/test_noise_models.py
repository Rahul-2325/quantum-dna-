import itertools
import pathlib
import sys

import pytest
from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel, depolarizing_error

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from aer_helpers import (control_distribution_aer, mismatch_circuit_measured,
                         transpile_for_noise)
from noise_models import (BASIS_GATES, VIRTUAL_GATES, assert_full_coverage,
                          build_noise_model)


def _probe_circuit(gate, num_qubits):
    """Circuit exercising `gate` on every qubit, with a deterministic noiseless outcome."""
    qc = QuantumCircuit(num_qubits, num_qubits)
    if gate == "cx":
        for control in range(0, num_qubits, 2):
            qc.x(control)                    # noiseless prep; only cx carries error here
            qc.cx(control, control + 1)
    elif gate == "sx":
        for qubit in range(num_qubits):      # sx twice == x, so the outcome is deterministic
            qc.sx(qubit)
            qc.sx(qubit)
    elif gate == "x":
        for qubit in range(num_qubits):
            qc.x(qubit)
    elif gate == "rz":
        for qubit in range(num_qubits):      # rz alone leaves |0>, so any drift is noise
            qc.rz(0.7, qubit)
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def _single_gate_noise_model(gate, num_qubits, probability):
    """Depolarizing error on exactly one basis gate, across all qubits."""
    noise_model = NoiseModel(basis_gates=BASIS_GATES)
    if gate == "cx":
        error = depolarizing_error(probability, 2)
        for pair in itertools.permutations(range(num_qubits), 2):
            noise_model.add_quantum_error(error, "cx", list(pair))
    else:
        error = depolarizing_error(probability, 1)
        for qubit in range(num_qubits):
            noise_model.add_quantum_error(error, gate, [qubit])
    return noise_model


def _marginal_ones(distribution, num_qubits):
    """P(qubit reads 1) for each qubit, from an int-keyed outcome distribution."""
    return [sum(p for value, p in distribution.items() if value >> qubit & 1)
            for qubit in range(num_qubits)]


def test_coverage_passes_on_a_full_model():
    qc, _ = mismatch_circuit_measured("ACG", "AGG")
    transpiled = transpile_for_noise(qc)
    noise_model = build_noise_model(qc.num_qubits, p_1q=0.001, p_2q=0.01)
    assert_full_coverage(noise_model, transpiled, exempt=VIRTUAL_GATES)


def test_coverage_includes_measure_only_when_readout_is_modelled():
    qc, _ = mismatch_circuit_measured("ACG", "AGG")
    transpiled = transpile_for_noise(qc)

    without_readout = build_noise_model(qc.num_qubits, p_1q=0.001, p_2q=0.01)
    with pytest.raises(AssertionError, match="measure"):
        assert_full_coverage(without_readout, transpiled, check_measure=True,
                             exempt=VIRTUAL_GATES)

    with_readout = build_noise_model(qc.num_qubits, p_1q=0.001, p_2q=0.01,
                                     readout_error=0.01)
    assert_full_coverage(with_readout, transpiled, check_measure=True,
                         exempt=VIRTUAL_GATES)


def test_coverage_raises_when_only_a_subset_of_qubits_carries_noise():
    qc, _ = mismatch_circuit_measured("ACGT", "AGGT")
    transpiled = transpile_for_noise(qc)
    bare = list(range(qc.num_qubits - 3, qc.num_qubits))
    partial = build_noise_model(qc.num_qubits, p_1q=0.001, p_2q=0.01,
                                qubits=[q for q in range(qc.num_qubits) if q not in bare])

    with pytest.raises(AssertionError) as excinfo:
        assert_full_coverage(partial, transpiled, exempt=VIRTUAL_GATES)
    message = str(excinfo.value)
    assert "does not cover" in message
    assert any(str(qubit) in message for qubit in bare)


def test_virtual_rz_leaves_rz_completely_noiseless():
    """Default virtual_rz must give rz no depolarizing and no thermal relaxation."""
    from noise_models import covered_operations, noisy_one_qubit_gates

    assert noisy_one_qubit_gates(True) == ("sx", "x")
    assert noisy_one_qubit_gates(False) == ("rz", "sx", "x")

    thermal = dict(t1=100_000.0, t2=80_000.0, gate_time_1q=50.0, gate_time_2q=300.0)
    virtual = build_noise_model(4, p_1q=0.01, p_2q=0.02, **thermal)
    assert "rz" not in covered_operations(virtual)
    for gate in ("sx", "x", "cx"):
        assert gate in covered_operations(virtual)

    legacy = build_noise_model(4, p_1q=0.01, p_2q=0.02, virtual_rz=False, **thermal)
    assert "rz" in covered_operations(legacy)


def test_coverage_exempts_virtual_gates_but_still_catches_real_gaps():
    """Regression: virtual rz must not be reported as an uncovered gate."""
    from noise_models import VIRTUAL_GATES

    qc, _ = mismatch_circuit_measured("ACG", "AGG")
    transpiled = transpile_for_noise(qc)
    assert transpiled.count_ops().get("rz", 0) > 0

    model = build_noise_model(qc.num_qubits, p_1q=0.001, p_2q=0.01)
    with pytest.raises(AssertionError, match="rz"):
        assert_full_coverage(model, transpiled)            # unexempted: rz looks missing
    assert_full_coverage(model, transpiled, exempt=VIRTUAL_GATES)

    # Exempting rz must not mask a genuinely uncovered qubit.
    bare = list(range(qc.num_qubits - 2, qc.num_qubits))
    partial = build_noise_model(
        qc.num_qubits, p_1q=0.001, p_2q=0.01,
        qubits=[q for q in range(qc.num_qubits) if q not in bare])
    with pytest.raises(AssertionError) as excinfo:
        assert_full_coverage(partial, transpiled, exempt=VIRTUAL_GATES)
    assert "rz" not in str(excinfo.value)
    assert any(str(q) in str(excinfo.value) for q in bare)


def test_thermal_relaxation_rejects_t2_above_twice_t1():
    with pytest.raises(ValueError, match="T2"):
        build_noise_model(3, t1=100.0, t2=300.0)


@pytest.mark.parametrize("gate", BASIS_GATES)
def test_large_error_on_each_basis_gate_changes_every_qubit(gate):
    """Empirical counterpart to assert_full_coverage: the noise must actually bite, everywhere."""
    num_qubits = 4
    # optimization_level=0 so the gate under test survives to the simulator.
    transpiled = transpile_for_noise(_probe_circuit(gate, num_qubits),
                                     optimization_level=0)
    assert transpiled.count_ops().get(gate, 0) > 0, f"{gate} optimized out of the probe"

    clean = control_distribution_aer(transpiled, shots=4096, seed=5)
    noisy = control_distribution_aer(
        transpiled, shots=4096, seed=5,
        noise_model=_single_gate_noise_model(gate, num_qubits, 0.75))

    clean_marginals = _marginal_ones(clean, num_qubits)
    noisy_marginals = _marginal_ones(noisy, num_qubits)
    for qubit, (before, after) in enumerate(zip(clean_marginals, noisy_marginals)):
        assert abs(after - before) > 0.05, (
            f"{gate}: qubit {qubit} unchanged ({before:.3f} -> {after:.3f})")
