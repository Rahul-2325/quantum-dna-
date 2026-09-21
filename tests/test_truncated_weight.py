import itertools
import math
import pathlib
import sys

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from hwlib import weight_unitary
from truncated_weight import weight_unitary_truncated


def test_full_k_matches_weight_unitary_exactly():
    """k_used == the full k must reproduce weight_unitary exactly (it's the
    same construction with no truncation applied)."""
    for n in (3, 5, 7):
        k_full = math.ceil(math.log2(n + 1))
        truncated, k = weight_unitary_truncated(n, k_full)
        reference, k_ref = weight_unitary(n)
        assert k == k_ref == k_full

        for bits in itertools.product((0, 1), repeat=n):
            prep_t = QuantumCircuit(k_full + n)
            prep_r = QuantumCircuit(k_full + n)
            for i, bit in enumerate(bits):
                if bit:
                    prep_t.x(k_full + i)
                    prep_r.x(k_full + i)
            prep_t.compose(truncated, inplace=True)
            prep_r.compose(reference, inplace=True)
            d_t = Statevector(prep_t).probabilities_dict(qargs=list(range(k_full)))
            d_r = Statevector(prep_r).probabilities_dict(qargs=list(range(k_full)))
            assert int(max(d_t, key=d_t.get), 2) == int(max(d_r, key=d_r.get), 2)


def test_truncation_aliases_exactly_mod_2_to_the_k_used():
    """Dropping qubits must alias to (true weight) mod 2^k_used, deterministically
    -- not a rounded/binned approximation, and not a superposition of candidates."""
    n = 7
    for k_used in (1, 2):
        qc, k = weight_unitary_truncated(n, k_used)
        assert k == k_used
        for bits in itertools.product((0, 1), repeat=n):
            prep = QuantumCircuit(k_used + n)
            for i, bit in enumerate(bits):
                if bit:
                    prep.x(k_used + i)
            prep.compose(qc, inplace=True)
            distribution = Statevector(prep).probabilities_dict(qargs=list(range(k_used)))
            best = max(distribution, key=distribution.get)
            assert distribution[best] > 0.999          # still deterministic, not a mix
            truth = sum(bits)
            assert int(best, 2) == truth % (2 ** k_used)


def test_truncation_preserves_low_end_exactly():
    """For true weights strictly below 2^k_used, truncation must give the
    EXACT true weight (no aliasing yet) -- this is what makes truncated
    registers usable for low-threshold screening (e.g. t=0, t=1)."""
    n = 8
    k_used = 2   # register holds 0..3, so weights 0..3 are alias-free
    qc, k = weight_unitary_truncated(n, k_used)
    for bits in itertools.product((0, 1), repeat=n):
        truth = sum(bits)
        if truth >= 2 ** k_used:
            continue
        prep = QuantumCircuit(k_used + n)
        for i, bit in enumerate(bits):
            if bit:
                prep.x(k_used + i)
        prep.compose(qc, inplace=True)
        distribution = Statevector(prep).probabilities_dict(qargs=list(range(k_used)))
        best = max(distribution, key=distribution.get)
        assert int(best, 2) == truth
