import pathlib
import sys

import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from hwlib import mismatch_circuit
from zne import exponential_extrapolate, fold_circuit, linear_extrapolate


@pytest.mark.parametrize("scale", [1, 3, 5, 7])
def test_fold_circuit_preserves_the_unitary(scale):
    """Folding must be logically the identity operation added on top -- same
    unitary, `scale` times the gate count."""
    qc, _ = mismatch_circuit("AC", "AG")
    folded = fold_circuit(qc, scale)
    assert folded.size() == scale * qc.size()
    assert Operator(folded).equiv(Operator(qc))


def test_fold_circuit_rejects_even_or_nonpositive_scale():
    qc, _ = mismatch_circuit("A", "A")
    for bad_scale in (0, 2, -1, -3):
        with pytest.raises(ValueError):
            fold_circuit(qc, bad_scale)


def test_fold_circuit_scale_1_is_the_circuit_itself():
    qc, _ = mismatch_circuit("ACG", "AGG")
    folded = fold_circuit(qc, 1)
    assert folded.size() == qc.size()
    assert Statevector(folded).equiv(Statevector(qc))


def test_linear_extrapolate_recovers_exact_line():
    scales = [1, 3, 5]
    intercept, slope = 0.87, -0.05
    values = [intercept + slope * s for s in scales]
    assert linear_extrapolate(scales, values) == pytest.approx(intercept, abs=1e-9)


def test_exponential_extrapolate_recovers_exact_exponential():
    """Synthetic data generated from a KNOWN a + b*exp(-c*scale) must recover
    that exact a+b at scale=0, not just "something plausible."""
    a, b, c = 1.0, -0.8, 0.3
    scales = [1, 3, 5]
    values = [a + b * 2.718281828 ** (-c * s) for s in scales]
    estimate = exponential_extrapolate(scales, values)
    assert estimate == pytest.approx(a + b, abs=1e-4)


def test_exponential_extrapolate_falls_back_to_linear_on_flat_data():
    """Flat (noiseless-looking) data has no decay to fit; must not crash,
    and should behave sanely (close to the flat value)."""
    scales = [1, 3, 5]
    values = [0.5, 0.5, 0.5]
    estimate = exponential_extrapolate(scales, values)
    assert estimate == pytest.approx(0.5, abs=0.05)
