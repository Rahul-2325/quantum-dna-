import itertools
import pathlib
import random
import sys

import pytest
from qiskit import transpile
from qiskit_aer import AerSimulator

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from hwlib import mismatch_circuit
from mismatch_depth_optimal import mismatch_circuit_depth_optimal

# Classical basis-state inputs only (DNA reads are never superpositions), so
# MPS is exact here -- see module docstring for why the MPS caveat in
# depth_optimal_counter.py doesn't apply to this file.
SIM = AerSimulator(method="matrix_product_state")


def _measured(read, window, shots=256, seed=11):
    qc, k = mismatch_circuit_depth_optimal(read, window)
    transpiled = transpile(qc, SIM, optimization_level=1, seed_transpiler=seed)
    counts = SIM.run(transpiled, shots=shots, seed_simulator=seed).result().get_counts()
    return counts, k


def test_matches_classical_count():
    random.seed(0)
    for L in (1, 2, 3):
        pairs = list(itertools.product("ACGT", repeat=L))
        allpairs = list(itertools.product(pairs, pairs))
        for r, d in random.sample(allpairs, min(20, len(allpairs))):
            read, window = "".join(r), "".join(d)
            counts, k = _measured(read, window)
            truth = sum(a != b for a, b in zip(r, d))
            total = sum(counts.values())
            correct = sum(c for key, c in counts.items() if int(key[-k:], 2) == truth)
            assert correct / total > 0.999, f"{read} vs {window}: {counts}"


def _pair_for_flag_pattern(pattern):
    read = "A" * len(pattern)
    window = "".join("A" if bit == 0 else "C" for bit in pattern)
    return read, window


def test_agrees_with_phase_counter_on_all_flag_patterns():
    """Same benchmark as the adder counter: exhaustive over all 2^L flag
    patterns, each realized by one representative read/window pair."""
    for L in (1, 2, 3, 4, 5):
        for pattern in itertools.product((0, 1), repeat=L):
            read, window = _pair_for_flag_pattern(pattern)
            truth = sum(pattern)

            counts, k = _measured(read, window)
            total = sum(counts.values())
            correct = sum(c for key, c in counts.items() if int(key[-k:], 2) == truth)
            assert correct / total > 0.999, f"depth-optimal {read} vs {window}: {counts}"

            phase_qc, k_phase = mismatch_circuit(read, window)
            from qiskit.quantum_info import Statevector
            d_phase = Statevector(phase_qc).probabilities_dict(qargs=list(range(k_phase)))
            best_phase = max(d_phase, key=d_phase.get)
            assert d_phase[best_phase] > 0.999
            assert int(best_phase, 2) == truth
