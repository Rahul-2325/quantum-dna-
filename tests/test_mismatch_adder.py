import itertools
import pathlib
import random
import sys

import pytest
from qiskit.quantum_info import Statevector

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from hwlib import mismatch_circuit
from mismatch_adder import mismatch_circuit_adder


def test_prefix_matches_hwlib_mismatch_circuit():
    """The duplicated load/XOR/OR-flag prefix must produce IDENTICAL flag-register
    state to hwlib.mismatch_circuit's own prefix, on every input -- this is what
    licenses calling the two circuits' counting stages a fair, apples-to-apples
    comparison. Checked via the full pre-counter statevector, not just gate lists,
    so it catches any semantic drift even if someone edits one copy but not the other.
    """
    rng = random.Random(0)
    for L in (1, 2, 3, 4):
        pairs = list(itertools.product("ACGT", repeat=L))
        for r, d in rng.sample(list(itertools.product(pairs, pairs)), min(20, len(pairs) ** 2)):
            read, window = "".join(r), "".join(d)

            reference, k = mismatch_circuit(read, window)
            flags_ref = Statevector(reference).probabilities_dict(
                qargs=list(range(k, k + L)))

            adder, k_adder = mismatch_circuit_adder(read, window)
            # Strip the counter back off: rebuild just through the shared prefix
            # by importing the private helper directly (same object under test).
            from qiskit import QuantumCircuit

            from mismatch_adder import _load_and_flag
            prefix_only = QuantumCircuit(adder.num_qubits)
            flags = list(range(k_adder, k_adder + L))
            rd = list(range(k_adder + L, k_adder + 3 * L))
            _load_and_flag(prefix_only, read, window, flags, rd)
            flags_dup = Statevector(prefix_only).probabilities_dict(
                qargs=list(range(k_adder, k_adder + L)))

            assert k == k_adder
            assert flags_ref.keys() == flags_dup.keys()
            for bits in flags_ref:
                assert flags_ref[bits] == pytest.approx(flags_dup[bits], abs=1e-9)


def test_mismatch_circuit_adder_matches_classical_count():
    random.seed(0)
    for L in (1, 2, 3):
        pairs = list(itertools.product("ACGT", repeat=L))
        allpairs = list(itertools.product(pairs, pairs))
        for r, d in random.sample(allpairs, min(25, len(allpairs))):
            qc, k = mismatch_circuit_adder("".join(r), "".join(d))
            distribution = Statevector(qc).probabilities_dict(qargs=list(range(k)))
            best = max(distribution, key=distribution.get)
            assert distribution[best] > 0.999
            assert int(best, 2) == sum(a != b for a, b in zip(r, d))


def _pair_for_flag_pattern(pattern):
    """A concrete (read, window) whose OR-flags are exactly `pattern` (tuple of 0/1).

    Every flag pattern is reachable: position i mismatches (flag=1) by picking
    any window base != read base, or matches (flag=0) by copying the read base.
    """
    read = "A" * len(pattern)
    window = "".join("A" if bit == 0 else "C" for bit in pattern)
    return read, window


def test_mismatch_circuit_adder_agrees_with_phase_counter_on_all_flag_patterns():
    """Both counters must agree with the classical count on EVERY flag pattern
    (all 2^L OR-flag outcomes, each realized by one representative read/window
    pair), not just sampled ones."""
    for L in (1, 2, 3, 4, 5, 6):
        for pattern in itertools.product((0, 1), repeat=L):
            read, window = _pair_for_flag_pattern(pattern)
            truth = sum(pattern)

            qc_adder, k_adder = mismatch_circuit_adder(read, window)
            d_adder = Statevector(qc_adder).probabilities_dict(qargs=list(range(k_adder)))
            best_adder = max(d_adder, key=d_adder.get)

            qc_phase, k_phase = mismatch_circuit(read, window)
            d_phase = Statevector(qc_phase).probabilities_dict(qargs=list(range(k_phase)))
            best_phase = max(d_phase, key=d_phase.get)

            assert d_adder[best_adder] > 0.999
            assert d_phase[best_phase] > 0.999
            assert int(best_adder, 2) == truth
            assert int(best_phase, 2) == truth
