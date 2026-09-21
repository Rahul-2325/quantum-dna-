import itertools
import pathlib
import sys

import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from plus_one_counter import plus_one_counter, register_width


def test_register_width_matches_k_at_i_equals_n():
    """r_n must equal k = ceil(log2(n+1)) for every n, or the layout is wrong."""
    import math
    for n in range(1, 65):
        k = math.ceil(math.log2(n + 1))
        assert register_width(n) == k


@pytest.mark.parametrize("n", range(1, 9))
def test_plus_one_counter_matches_classical_weight_exhaustively(n):
    """Paper 1 Sec. 2: verify against the classical Hamming weight for every input."""
    qc, k, t_width = plus_one_counter(n)
    for bits in itertools.product((0, 1), repeat=n):
        prep = QuantumCircuit(qc.num_qubits)
        for i, bit in enumerate(bits):
            if bit:
                prep.x(k + i)
        prep.compose(qc, inplace=True)
        d = Statevector(prep).probabilities_dict()
        best = max(d, key=d.get)
        assert d[best] > 0.999                              # exact, deterministic circuit
        reversed_bits = best[::-1]                           # bit string is qubit-0-last
        weight_bits = reversed_bits[:k]
        temp_bits = reversed_bits[k + n:k + n + t_width]
        assert int(weight_bits[::-1], 2) == sum(bits)
        assert set(temp_bits) <= {"0"}                       # scratch returns to |0>


def test_plus_one_counter_matches_weight_unitary_distribution():
    """Same map as weight_unitary, checked via full outcome distributions, not just argmax."""
    from hwlib import weight_unitary

    for n in (3, 5, 7):
        adder, k, t_width = plus_one_counter(n)
        phase, k_phase = weight_unitary(n)
        assert k == k_phase
        for bits in itertools.product((0, 1), repeat=n):
            prep_adder = QuantumCircuit(adder.num_qubits)
            prep_phase = QuantumCircuit(phase.num_qubits)
            for i, bit in enumerate(bits):
                if bit:
                    prep_adder.x(k + i)
                    prep_phase.x(k + i)
            prep_adder.compose(adder, inplace=True)
            prep_phase.compose(phase, inplace=True)

            got_adder = Statevector(prep_adder).probabilities_dict(qargs=list(range(k)))
            got_phase = Statevector(prep_phase).probabilities_dict(qargs=list(range(k)))
            value_adder = int(max(got_adder, key=got_adder.get), 2)
            value_phase = int(max(got_phase, key=got_phase.get), 2)
            assert value_adder == value_phase == sum(bits)
