"""Paper 1 (Bravyi-Yoder-Maslov, IEEE TC 2022), Section 2 and Fig. 1:
the "plus-one" CNOT/Toffoli weight counter, used here as a baseline to
compare against Paper 2's phase/QFT weight_unitary.

Section 2 computes the Hamming weight of x1..xn by applying, for each xi in
turn, a controlled "increment the running weight register w by 1" using a
standard reversible ripple-carry adder: a Toffoli chain computes carries
forward into scratch qubits t, a half-adder applies the top bit, then a
second Toffoli/CNOT chain uncomputes the carries on the way back while
writing the remaining sum bits. Because increment-by-one always uncomputes
its own carries, `t` returns to |0> after every single call and the same
scratch qubits are reused for all n increments: only O(log n) temp qubits
are needed in total, matching k-2 where k = ceil(log2(n+1)).

The register width used for the i-th increment is r_i = floor(log2(i)) + 1,
which grows only as needed; the full k-wide weight register is reached only
at i = n, since ceil(log2(n+1)) == floor(log2(n)) + 1 for every n >= 1.
"""
from __future__ import annotations

import math

from qiskit import QuantumCircuit


def register_width(i):
    """r_i = floor(log2(i)) + 1: weight-register bits needed after i increments."""
    return i.bit_length()


def _increment_by_one(qc, ctrl, w, t):
    """w += 1, controlled on `ctrl`, via Paper 1 Section 2's ripple-carry increment.

    `w` is the (prefix of the) weight register in use for this call, `w[0]`
    the LSB. `t` is the temp/carry register, needing len(w) - 2 qubits when
    len(w) >= 3 and none otherwise. `t` is returned to |0> by this call.
    """
    r = len(w)
    if r == 1:                                   # i == 1: trivial single bit
        qc.cx(ctrl, w[0])
        return
    if r == 2:                                    # i in {2, 3}: direct half-adder
        qc.ccx(ctrl, w[0], w[1])
        qc.cx(ctrl, w[0])
        return

    # r >= 3: ripple carry forward into t[0..r-3].
    qc.ccx(ctrl, w[0], t[0])
    for j in range(1, r - 2):
        qc.ccx(t[j - 1], w[j], t[j])

    # Half-adder at the top: may extend the weight into bit w[r-1].
    qc.ccx(t[r - 3], w[r - 2], w[r - 1])
    qc.cx(t[r - 3], w[r - 2])

    # Uncompute carries on the way back, writing the remaining sum bits.
    for j in range(r - 3, 0, -1):
        qc.ccx(t[j - 1], w[j], t[j])
        qc.cx(t[j - 1], w[j])
    qc.ccx(ctrl, w[0], t[0])
    qc.cx(ctrl, w[0])


def plus_one_counter(n):
    """Paper 1 Sec. 2 weight counter: |x>|0> -> |x>|W(x)>, ancilla returned to |0>.

    Qubit layout mirrors weight_unitary: [0..k-1] weight/control (little-endian),
    [k..k+n-1] data, [k+n..k+n+t_width-1] temp scratch (garbage-free: always |0>
    both before and after the circuit, so it never needs separate uncomputation).
    Returns (circuit, k, t_width).
    """
    k = max(1, math.ceil(math.log2(n + 1))) if n > 0 else 0
    t_width = max(0, k - 2)
    qc = QuantumCircuit(k + n + t_width, name=f"plus_one_{n}")
    w = list(range(k))
    x = list(range(k, k + n))
    t = list(range(k + n, k + n + t_width))

    for i in range(1, n + 1):
        r = register_width(i)
        _increment_by_one(qc, x[i - 1], w[:r], t[:max(0, r - 2)])

    return qc, k, t_width
