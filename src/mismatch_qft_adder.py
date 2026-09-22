"""mismatch_circuit, counted with the Draper/QFT-adder (qft_adder_counter.py)
instead of Paper 2's weight_unitary, Paper 1's plus_one_counter, or the
depth-optimal rewrite -- the fourth counter, closing the 3-paradigm
comparison the literature search flagged as missing (QShift-SA uses a
QFT-adder style oracle; nobody has compared it against a phase-kickback
counter and a ripple-carry counter under one noise model).

Same pattern as mismatch_adder.py and mismatch_depth_optimal.py: hwlib.py
stays untouched, and `_load_and_flag` is the same verified-identical prefix
duplicate (checked against hwlib.mismatch_circuit's own prefix by
test_mismatch_qft_adder.py, not just assumed).
"""
from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit

from mismatch_adder import _load_and_flag
from qft_adder_counter import qft_adder_counter


def mismatch_circuit_qft_adder(read, ref_window):
    """mismatch_circuit's read/XOR/OR-flag prefix, counted with qft_adder_counter.

    Same qubit layout as mismatch_circuit and mismatch_circuit_adder:
    [control k | flags L | read 2L]. qft_adder_counter needs no separate
    scratch register (unlike plus_one_counter's temp bits or
    depth_optimal_weight's reused control block), so this is the same
    total qubit count as the phase counter's mismatch_circuit.
    """
    L = len(read)
    counter, k = qft_adder_counter(L)
    qc = QuantumCircuit(k + L + 2 * L)
    flags = list(range(k, k + L))
    rd = list(range(k + L, k + 3 * L))

    _load_and_flag(qc, read, ref_window, flags, rd)
    qc.compose(counter, qubits=list(range(k)) + flags, inplace=True)
    return qc, k


def mismatch_circuit_qft_adder_measured(read, ref_window):
    """mismatch_circuit_qft_adder with the k control qubits measured into a
    classical register."""
    qc, k = mismatch_circuit_qft_adder(read, ref_window)
    creg = ClassicalRegister(k, "c")
    qc.add_register(creg)
    qc.measure(list(range(k)), list(creg))
    return qc, k
