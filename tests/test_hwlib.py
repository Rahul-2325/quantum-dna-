import itertools, random, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from hwlib import *
from qiskit.quantum_info import random_statevector

def test_weight_distribution_on_superposition():
    """Paper 2 benchmark: coherent weight distribution == ||P_x psi||^2."""
    for n in (3, 4):
        W, k = weight_unitary(n)
        psi = random_statevector(2**n, seed=n)
        out = Statevector(np.kron(psi.data, np.eye(2**k)[0])).evolve(W)
        got = {int(b, 2): p for b, p in out.probabilities_dict(qargs=list(range(k))).items()}
        true = {}
        for x, a in enumerate(psi.data):
            w = bin(x).count("1"); true[w] = true.get(w, 0) + abs(a) ** 2
        tvd = 0.5 * sum(abs(got.get(w, 0) - true.get(w, 0)) for w in range(2**k))
        assert tvd < 1e-9

def test_mismatch_counter_matches_classical():
    random.seed(0)
    for L in (1, 2, 3):
        pairs = list(itertools.product("ACGT", repeat=L))
        allpairs = list(itertools.product(pairs, pairs))
        for r, d in random.sample(allpairs, min(25, len(allpairs))):
            qc, k = mismatch_circuit("".join(r), "".join(d))
            v, p = read_control_value(qc, k)
            assert p > 0.999 and v == sum(a != b for a, b in zip(r, d))

def test_paper3_example_TA_vs_TACCG():
    ref, read = "TACCG", "TA"
    counts = []
    for s in range(len(ref) - len(read) + 1):
        qc, k = mismatch_circuit(read, ref[s:s + len(read)])
        counts.append(read_control_value(qc, k)[0])
    assert counts == [0, 2, 2, 2]

def test_hwb_all_inputs_small_n():
    assert hwb_classical((1, 0, 0)) == (0, 1, 0)      # Paper 1 truth table
    for n in (3, 4, 5):
        qc, k = hwb_circuit(n)
        for x in itertools.product((0, 1), repeat=n):
            prep = QuantumCircuit(k + n)
            for i, b in enumerate(x):
                if b: prep.x(k + i)
            prep.compose(qc, inplace=True)
            d = Statevector(prep).probabilities_dict(); best = max(d, key=d.get)
            bits = best[::-1]
            assert d[best] > 0.999 and set(bits[:k]) == {"0"}
            assert tuple(int(c) for c in bits[k:]) == hwb_classical(x)
