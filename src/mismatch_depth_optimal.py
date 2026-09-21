"""mismatch_circuit, counted with the depth-optimal circuit (task 2,
src/depth_optimal_counter.py) instead of Paper 2's Algorithm 2 (weight_unitary)
or Paper 1's ripple-carry adder (plus_one_counter) -- a third counter for the
noise comparison.

Note on the MPS caveat in depth_optimal_counter.py: that module documents
Aer's matrix_product_state method giving wrong results for SUPERPOSITION
inputs to the bare counter. It does not apply here -- every mismatch circuit
in this project loads reads/windows as classical basis states via X gates
(never superpositions), and depth_optimal_weight is exactly correct under
MPS for basis-state inputs (exhaustively verified in
tests/test_depth_optimal_counter.py). MPS is safe to use for this module.

Like mismatch_adder.py, hwlib.py is left untouched; `_load_and_flag` is a
byte-for-byte copy of mismatch_circuit's prefix, checked against the
original by test_mismatch_depth_optimal.py, not just asserted.
"""
from __future__ import annotations

from qiskit import QuantumCircuit

from depth_optimal_counter import depth_optimal_weight
from mismatch_adder import _load_and_flag


def mismatch_circuit_depth_optimal(read, ref_window):
    """mismatch_circuit's read/XOR/OR-flag prefix, counted with depth_optimal_weight.

    Qubit layout: [flags L | control L | read 2L]. depth_optimal_weight's own
    "data" register is exactly our flag register (both length L); its
    "control" register is fresh ancilla, reused/reset internally across
    rounds. Its own classical registers (out: k bits, temp: L bits) are
    added to this circuit and populated by depth_optimal_weight's own
    mid-circuit measurements -- there is no separate ".measure()" step to
    add afterwards, unlike the other two counters, because this circuit
    already produces its answer via classical `store`/`measure` internally.

    Returns (circuit, k). The answer is in the circuit's first k classical
    bits (register "out"); the remaining L bits (register "temp") are
    per-round measurement scratch and should be ignored by callers.
    """
    L = len(read)
    counter, k = depth_optimal_weight(L)

    qc = QuantumCircuit(4 * L)   # flags(L) + control(L) + read(2L)
    flags = list(range(0, L))
    control = list(range(L, 2 * L))
    rd = list(range(2 * L, 4 * L))

    _load_and_flag(qc, read, ref_window, flags, rd)

    for creg in counter.cregs:
        qc.add_register(creg)

    qc.compose(counter, qubits=flags + control,
              clbits=list(range(counter.num_clbits)), inplace=True)
    return qc, k
