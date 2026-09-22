import itertools
import pathlib
import random
import sys

import pytest
from qiskit.quantum_info import Statevector

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from hwlib import mismatch_circuit
from mismatch_qft_adder import mismatch_circuit_qft_adder


def test_prefix_matches_hwlib_mismatch_circuit():
    """Same check as mismatch_adder.py's own test: the duplicated prefix must
    produce IDENTICAL flag-register state to hwlib.mismatch_circuit's own
    prefix, on every input."""
    rng = random.Random(0)
    for L in (1, 2, 3, 4):
        pairs = list(itertools.product("ACGT", repeat=L))
        for r, d in rng.sample(list(itertools.product(pairs, pairs)), min(20, len(pairs) ** 2)):
            read, window = "".join(r), "".join(d)

            reference, k = mismatch_circuit(read, window)
            flags_ref = Statevector(reference).probabilities_dict(
                qargs=list(range(k, k + L)))

            qft_adder, k_adder = mismatch_circuit_qft_adder(read, window)
            from qiskit import QuantumCircuit

            from mismatch_adder import _load_and_flag
            prefix_only = QuantumCircuit(qft_adder.num_qubits)
            flags = list(range(k_adder, k_adder + L))
            rd = list(range(k_adder + L, k_adder + 3 * L))
            _load_and_flag(prefix_only, read, window, flags, rd)
            flags_dup = Statevector(prefix_only).probabilities_dict(
                qargs=list(range(k_adder, k_adder + L)))

            assert k == k_adder
            assert flags_ref.keys() == flags_dup.keys()
            for bits in flags_ref:
                assert flags_ref[bits] == pytest.approx(flags_dup[bits], abs=1e-9)


def test_matches_classical_count():
    random.seed(0)
    for L in (1, 2, 3):
        pairs = list(itertools.product("ACGT", repeat=L))
        allpairs = list(itertools.product(pairs, pairs))
        for r, d in random.sample(allpairs, min(25, len(allpairs))):
            qc, k = mismatch_circuit_qft_adder("".join(r), "".join(d))
            distribution = Statevector(qc).probabilities_dict(qargs=list(range(k)))
            best = max(distribution, key=distribution.get)
            assert distribution[best] > 0.999
            assert int(best, 2) == sum(a != b for a, b in zip(r, d))


def _pair_for_flag_pattern(pattern):
    read = "A" * len(pattern)
    window = "".join("A" if bit == 0 else "C" for bit in pattern)
    return read, window


def test_agrees_with_phase_counter_on_all_flag_patterns():
    """Same benchmark as the adder and depth-optimal counters: exhaustive
    over all 2^L flag patterns."""
    for L in (1, 2, 3, 4, 5, 6):
        for pattern in itertools.product((0, 1), repeat=L):
            read, window = _pair_for_flag_pattern(pattern)
            truth = sum(pattern)

            qc_qft, k_qft = mismatch_circuit_qft_adder(read, window)
            d_qft = Statevector(qc_qft).probabilities_dict(qargs=list(range(k_qft)))
            best_qft = max(d_qft, key=d_qft.get)

            qc_phase, k_phase = mismatch_circuit(read, window)
            d_phase = Statevector(qc_phase).probabilities_dict(qargs=list(range(k_phase)))
            best_phase = max(d_phase, key=d_phase.get)

            assert d_qft[best_qft] > 0.999
            assert d_phase[best_phase] > 0.999
            assert int(best_qft, 2) == truth
            assert int(best_phase, 2) == truth
