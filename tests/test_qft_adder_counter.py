import itertools
import pathlib
import sys

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_statevector

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from hwlib import weight_unitary
from qft_adder_counter import qft_adder_counter


def test_qft_of_zero_equals_hadamard_of_zero():
    """The construction's premise: QFT|0> has no relative phases to add, so
    it is exactly H^k|0>, the same starting point weight_unitary uses.
    If this weren't true the whole "QFT once at the start" framing would
    need a different justification."""
    from qiskit.circuit.library import QFTGate

    for k in (1, 2, 3, 4):
        qft_start = QuantumCircuit(k)
        qft_start.append(QFTGate(k), range(k))
        hadamard_start = QuantumCircuit(k)
        hadamard_start.h(range(k))
        assert Statevector(qft_start).equiv(Statevector(hadamard_start))


@pytest.mark.parametrize("n", range(1, 9))
def test_qft_adder_counter_matches_classical_weight_exhaustively(n):
    qc, k = qft_adder_counter(n)
    for bits in itertools.product((0, 1), repeat=n):
        prep = QuantumCircuit(k + n)
        for i, bit in enumerate(bits):
            if bit:
                prep.x(k + i)
        prep.compose(qc, inplace=True)
        d = Statevector(prep).probabilities_dict()
        best = max(d, key=d.get)
        assert d[best] > 0.999
        reversed_bits = best[::-1]
        assert int(reversed_bits[:k][::-1], 2) == sum(bits)


def test_qft_adder_counter_matches_weight_unitary_on_superpositions():
    """Same benchmark Paper 2's own Section VI and test_hwlib.py use:
    P(x) = ||P_x psi||^2, checked via TVD against the true distribution.

    The embedding here (np.kron(psi.data, np.eye(2**k)[0]), data as the
    OUTER/more-significant kron factor) matters and was gotten wrong once
    during development -- an inverted kron order made a correct circuit
    look badly broken (TVD up to 0.63) purely from a bad test harness. This
    is the corrected, verified-consistent form, matching
    test_hwlib.py::test_weight_distribution_on_superposition's own pattern.
    """
    for n in (3, 4, 5):
        qc, k = qft_adder_counter(n)
        phase_qc, k_phase = weight_unitary(n)
        assert k == k_phase

        for seed in range(10):
            psi = random_statevector(2 ** n, seed=100 * n + seed)
            full_data = np.kron(psi.data, np.eye(2 ** k)[0])
            out = Statevector(full_data).evolve(qc)
            got = {int(bits, 2): p for bits, p in
                  out.probabilities_dict(qargs=list(range(k))).items()}

            true_dist = {}
            for x, amplitude in enumerate(psi.data):
                w = bin(x).count("1")
                true_dist[w] = true_dist.get(w, 0) + abs(amplitude) ** 2

            tvd = 0.5 * sum(abs(got.get(w, 0.0) - true_dist.get(w, 0.0))
                            for w in range(2 ** k))
            assert tvd < 1e-9, f"n={n} seed={seed}: TVD={tvd:.2e}"


def test_qft_adder_counter_agrees_with_weight_unitary_pointwise():
    """Both counters must give the SAME distribution for the same input,
    not merely each be individually correct -- checked directly against
    each other, not just each against the classical formula."""
    for n in (3, 4, 5, 6):
        qc, k = qft_adder_counter(n)
        phase_qc, k_phase = weight_unitary(n)
        for bits in itertools.product((0, 1), repeat=n):
            prep_a = QuantumCircuit(k + n)
            prep_b = QuantumCircuit(k_phase + n)
            for i, bit in enumerate(bits):
                if bit:
                    prep_a.x(k + i)
                    prep_b.x(k_phase + i)
            prep_a.compose(qc, inplace=True)
            prep_b.compose(phase_qc, inplace=True)
            d_a = Statevector(prep_a).probabilities_dict(qargs=list(range(k)))
            d_b = Statevector(prep_b).probabilities_dict(qargs=list(range(k_phase)))
            assert int(max(d_a, key=d_a.get), 2) == int(max(d_b, key=d_b.get), 2)
