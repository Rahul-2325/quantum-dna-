"""mismatch_circuit, but counted with Paper 1's plus-one adder instead of
Paper 2's phase/QFT weight_unitary -- the baseline for the C4 comparison.

hwlib.py is left untouched (see the project notes on why). To satisfy "shares
EXACTLY the same read loading, XOR and OR-flag code" as mismatch_circuit
without editing hwlib.py, `_load_and_flag` below is a byte-for-byte copy of
that prefix, not an independent reimplementation. This is checked, not just
asserted: test_mismatch_adder.py compares the flag-register state of this
circuit against hwlib.mismatch_circuit's, gate list included, on random
inputs, so any future drift between the two copies is caught.

If a single shared prefix function (refactoring hwlib.py) is preferred over
this verified-duplicate approach, that is a separate decision -- flagged, not
made here.
"""
from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit

from hwlib import ENC
from plus_one_counter import plus_one_counter


def _load_and_flag(qc, read, ref_window, flags, rd):
    """Exact copy of hwlib.mismatch_circuit's prefix: load, XOR, OR-flag.

    Verified identical to hwlib.mismatch_circuit by
    test_mismatch_adder.py::test_prefix_matches_hwlib_mismatch_circuit.
    """
    for i, b in enumerate(read):
        for t, bit in enumerate(ENC[b]):
            if bit:
                qc.x(rd[2 * i + t])
    for i, b in enumerate(ref_window):
        for t, bit in enumerate(ENC[b]):
            if bit:
                qc.x(rd[2 * i + t])
    for i in range(len(read)):
        a, b = rd[2 * i], rd[2 * i + 1]
        qc.x(a); qc.x(b); qc.ccx(a, b, flags[i]); qc.x(flags[i]); qc.x(a); qc.x(b)


def mismatch_circuit_adder(read, ref_window):
    """mismatch_circuit's read/XOR/OR-flag prefix, counted with plus_one_counter.

    Same qubit layout as mismatch_circuit: [control k | flags L | read 2L],
    plus plus_one_counter's own temp scratch appended at the end (returns to
    |0>, so it needs no separate uncompute -- see plus_one_counter.py).
    """
    L = len(read)
    counter, k, t_width = plus_one_counter(L)
    qc = QuantumCircuit(k + L + 2 * L + t_width)
    flags = list(range(k, k + L))
    rd = list(range(k + L, k + 3 * L))

    _load_and_flag(qc, read, ref_window, flags, rd)
    # plus_one_counter's own layout is [weight k | data L | temp]; its data
    # register is exactly our flag register, so compose it onto [control | flags | temp].
    temp = list(range(k + 3 * L, k + 3 * L + t_width))
    qc.compose(counter, qubits=list(range(k)) + flags + temp, inplace=True)
    return qc, k


def mismatch_circuit_adder_measured(read, ref_window):
    """mismatch_circuit_adder with the k control qubits measured into a classical register."""
    qc, k = mismatch_circuit_adder(read, ref_window)
    creg = ClassicalRegister(k, "c")
    qc.add_register(creg)
    qc.measure(list(range(k)), list(creg))
    return qc, k
