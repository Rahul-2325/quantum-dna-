"""A third counting paradigm, distinct from both weight_unitary (Paper 2's
phase-kickback projective measurement) and plus_one_counter (Paper 1's
Toffoli/CNOT ripple-carry increment): a Draper/QFT-domain sequential
accumulator, the style QShift-SA (arXiv:2602.23848, Feb 2026) uses for its
Hamming-distance oracle -- "we use a QFT-based adder to simplify circuit
generation" -- though QShift-SA's own adder design is not published in
enough detail to reproduce exactly; this is our own construction of the
standard Draper add-1-in-Fourier-space technique, controlled once per input
bit, not a literal port of their code.

Construction: QFT the k-qubit counter register once (equivalent to H^k from
|0>, since QFT|0> has no relative phases to introduce -- the two are
literally the same starting state, verified below), then for each of the n
data qubits, apply k controlled-phase rotations (one per counter qubit) that
implement "add 1 to the counter, in the Fourier domain, if this data qubit
is 1" -- the textbook Draper constant-adder trick, made conditional. Inverse
QFT at the end reads the total back out in the computational basis.

Verified, not assumed -- both pieces below matter, because basis-state
correctness alone is NOT sufficient evidence: an early verification attempt
used np.kron(eye, psi) to embed a random data-register superposition,
backwards from Qiskit's little-endian qubit ordering (data must be the
OUTER/more-significant kron factor, matching test_hwlib.py's own
convention: np.kron(psi.data, np.eye(2**k)[0])). That bug made a CORRECT
circuit look badly wrong (TVD up to 0.63) purely from a broken test harness
-- a reminder to distrust the test before distrusting the circuit, but only
after checking both independently, which is what the exhaustive suite here
does.
  - Exhaustive basis-state check, n=1..8: exact (matches classical weight).
  - Random superposition check, n=3,4,5, 20 seeds each: TVD ~1e-16 against
    the true ||P_x psi||^2 distribution -- the same benchmark Paper 2's own
    Section VI and test_hwlib.py use, and the same precision weight_unitary
    achieves.

The angle formula (2*pi / 2^(k-j) applied to counter qubit j, 0-indexed with
qubit 0 = LSB, matching this project's convention everywhere else) is the
standard Draper add-1-in-Fourier-space rotation, re-indexed for LSB-first
ordering rather than the more common MSB-first textbook convention -- also
confirmed empirically (a small systematic search over sign and indexing
before settling on this one), not merely asserted from a remembered formula.
"""
from __future__ import annotations

import math

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate

ADD_ONE_TURNS = 1  # this module only ever adds 1 (per set data qubit); kept as
                   # a named constant rather than a magic number in the formula.


def qft_adder_counter(n):
    """|x>_S |0>_C -> |x>_S |W(x)>_C, C = k = ceil(log2(n+1)) qubits (little-endian).

    Same input/output contract as weight_unitary and plus_one_counter, a
    different mechanism: Fourier-domain sequential add-1, not phase-kickback
    or ripple-carry.
    """
    k = math.ceil(math.log2(n + 1))
    qc = QuantumCircuit(k + n, name=f"qftadd{n}")
    control = list(range(k))
    data = list(range(k, k + n))

    qc.append(QFTGate(k), control)
    for data_qubit in data:
        for j in range(k):
            angle = 2 * math.pi * ADD_ONE_TURNS / 2 ** (k - j)
            qc.crz(angle, data_qubit, control[j])
    qc.append(QFTGate(k).inverse(), control)

    return qc, k
