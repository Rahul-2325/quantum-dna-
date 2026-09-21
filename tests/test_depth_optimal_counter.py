import itertools
import pathlib
import sys

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, random_statevector
from qiskit_aer import AerSimulator

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from depth_optimal_counter import depth_optimal_weight, ghz_fanout
from hwlib import weight_unitary

SIM = AerSimulator(method="matrix_product_state")
# MPS gives wrong results on superposition inputs for THIS circuit specifically
# (see module docstring); basis-state tests can safely use the fast MPS
# simulator above, but superposition tests must use dense statevector.
SIM_EXACT = AerSimulator(method="statevector")


def _run_basis(n, bits, shots=256, seed=11):
    qc, k = depth_optimal_weight(n)
    prep = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    for i, bit in enumerate(bits):
        if bit:
            prep.x(i)  # data qubits are [0..n-1]
    prep.compose(qc, inplace=True)
    transpiled = transpile(prep, SIM, optimization_level=1, seed_transpiler=seed)
    counts = SIM.run(transpiled, shots=shots, seed_simulator=seed).result().get_counts()
    return counts, k


def test_ghz_fanout_is_a_clean_repetition_code():
    """Only |0..0> and |1..1> should appear; no other basis states, any n."""
    for n in (2, 3, 5, 8):
        qc = QuantumCircuit(n)
        qc.ry(0.7, 0)
        ghz_fanout(qc, list(range(n)))
        probs = Statevector(qc).probabilities_dict()
        assert set(probs) <= {"0" * n, "1" * n}


@pytest.mark.parametrize("n", range(1, 7))
def test_depth_optimal_weight_matches_classical_weight_exhaustively(n):
    """Every basis-state input must give the correct weight with ~unit probability."""
    for bits in itertools.product((0, 1), repeat=n):
        counts, k = _run_basis(n, bits)
        truth = sum(bits)
        total = sum(counts.values())
        # out register is the LAST k classical bits in the printed key; reading
        # them as a plain binary string is correct as-is (out[0] = LSB, matching
        # weight_unitary's convention) -- see module docstring for how that was
        # established empirically.
        correct = sum(c for key, c in counts.items() if int(key[-k:], 2) == truth)
        assert correct / total > 0.999, (
            f"n={n} bits={bits} truth={truth}: only {correct}/{total} correct, "
            f"counts={counts}")


def test_depth_optimal_weight_matches_weight_unitary_on_superpositions():
    """Full outcome distribution must match weight_unitary's, the same benchmark
    Paper 2's own Section VI and test_hwlib.py use (P(x) = ||P_x psi||^2)."""
    for n in (3, 4, 5):
        do_qc, k = depth_optimal_weight(n)
        phase_qc, k_phase = weight_unitary(n)
        assert k == k_phase

        for seed in range(5):
            psi = random_statevector(2 ** n, seed=1000 * n + seed)

            prep = QuantumCircuit(do_qc.num_qubits, do_qc.num_clbits)
            prep.initialize(psi.data, list(range(n)))
            prep.compose(do_qc, inplace=True)
            transpiled = transpile(prep, SIM_EXACT, optimization_level=1, seed_transpiler=1)
            counts = SIM_EXACT.run(transpiled, shots=4000,
                                   seed_simulator=1).result().get_counts()
            got = {}
            for key, c in counts.items():
                value = int(key[-k:], 2)
                got[value] = got.get(value, 0) + c / 4000

            true_dist = {}
            for x, amplitude in enumerate(psi.data):
                w = bin(x).count("1")
                true_dist[w] = true_dist.get(w, 0) + abs(amplitude) ** 2

            tvd = 0.5 * sum(abs(got.get(w, 0.0) - true_dist.get(w, 0.0))
                            for w in range(k_phase and 2 ** k))
            # Shot noise at 4000 shots: allow a generous but real tolerance,
            # not the ~1e-16 exactness of the noiseless algebraic counters.
            assert tvd < 0.06, f"n={n} seed={seed}: TVD={tvd:.4f} too high, got={got}"


def test_mps_is_wrong_on_this_circuit_regression_guard():
    """Documents and pins the MPS-vs-statevector discrepancy for this circuit.

    If this ever starts FAILING (i.e. MPS agrees with statevector), that's
    good news -- it means a newer Aer fixed the underlying issue, and the
    other tests' use of SIM_EXACT can likely switch back to the faster MPS
    method. Until then, this stays red-flagged so nobody "optimizes" the
    superposition tests back onto MPS and reintroduces silently wrong results.
    """
    n = 3
    do_qc, k = depth_optimal_weight(n)
    psi = random_statevector(2 ** n, seed=3000)

    tvds = {}
    for label, sim in (("mps", SIM), ("statevector", SIM_EXACT)):
        prep = QuantumCircuit(do_qc.num_qubits, do_qc.num_clbits)
        prep.initialize(psi.data, list(range(n)))
        prep.compose(do_qc, inplace=True)
        transpiled = transpile(prep, sim, optimization_level=1, seed_transpiler=1)
        counts = sim.run(transpiled, shots=4000, seed_simulator=1).result().get_counts()
        got = {}
        for key, c in counts.items():
            value = int(key[-k:], 2)
            got[value] = got.get(value, 0) + c / 4000
        true_dist = {}
        for x, amplitude in enumerate(psi.data):
            w = bin(x).count("1")
            true_dist[w] = true_dist.get(w, 0) + abs(amplitude) ** 2
        tvds[label] = 0.5 * sum(abs(got.get(w, 0.0) - true_dist.get(w, 0.0))
                                for w in range(2 ** k))

    assert tvds["statevector"] < 0.06, "statevector should be accurate"
    assert tvds["mps"] > 0.1, (
        "MPS TVD dropped below 0.1 -- Aer may have fixed the dynamic-circuit "
        "issue this test guards against; see module docstring before removing")


def test_depth_optimal_weight_uses_dynamic_circuit_features():
    """Sanity check that this really is exercising mid-circuit measurement and
    classical feedback, not silently degenerating into something trivial."""
    qc, k = depth_optimal_weight(4)
    ops = qc.count_ops()
    assert ops.get("reset", 0) >= k - 1        # reset between rounds 2..k
    assert ops.get("if_else", 0) >= 1          # at least one conditioned round
    assert "store" in ops
    assert k == 3
