"""Diffusion operator + amplification loop on top of grover_oracle.py's
verified phase oracle -- the "generic Grover machinery" half of the novelty
direction described in grover_oracle.py's module docstring and PROJECT_STATUS.md
Section 4.3.

build_grover_diffuser(k) is the textbook inversion-about-the-mean operator
2|s><s| - I, where |s> = H^k|0> is the uniform superposition over all 2^k
computational basis states of a k-qubit register: H^k, X^k, an (k-1)-controlled
Z realized via H-mcx-H phase kickback on the last qubit, X^k, H^k. Verified
directly against the explicit matrix (tests/test_grover_search.py) for k up to
4, not just trusted by construction.

build_grover_search(reference, read, threshold, iterations=None) assembles:
  H on the shift register (state prep, applied ONCE)
  repeat `iterations` times: oracle marking (grover_oracle.append_oracle_marking)
                              then the diffuser on the shift register alone
`iterations` defaults to the standard Grover-optimal round(pi/4 * sqrt(N/M)),
N = 2^k_shift, M = number of real shifts with true mismatch count <= threshold
(computed classically here purely to pick the iteration count -- the search
circuit itself never uses this classical knowledge of M for anything but
choosing how many times to repeat, which is how every Grover-search paper
picks it too; a real deployment would need amplitude estimation or a
doubling schedule when M is unknown, out of scope here).

This module answers "does the marked oracle, correctly excluding leftover
branches (see grover_oracle.py's `valid` ancilla), actually amplify toward
the true good shifts when wired into standard Grover machinery" -- the
remaining question after grover_oracle.py's own verification, and the one
that would have silently failed before the `valid`-ancilla fix whenever
len(shifts) wasn't a power of two (i.e. almost always).
"""
from __future__ import annotations

import math

from qiskit import QuantumCircuit

from grover_oracle import append_oracle_marking, oracle_layout


def _true_mismatch(reference, read, shift):
    window = reference[shift:shift + len(read)]
    return sum(a != b for a, b in zip(read, window))


def build_grover_diffuser(k):
    """2|s><s| - I on a k-qubit register, |s> = H^k|0>.

    The textbook H-X-(multi-controlled-Z via H/mcx/H sandwich)-X-H
    construction below realizes this operator up to an overall global phase
    of -1 (verified by direct matrix comparison in
    test_diffuser_matches_explicit_inversion_about_the_mean -- a global
    phase has no effect on any measured probability or on Grover's
    correctness, but is corrected here anyway so this circuit matches the
    textbook formula exactly rather than merely up to a citable footnote).
    """
    qc = QuantumCircuit(k, name=f"diffuser{k}")
    qc.h(range(k))
    qc.x(range(k))
    if k == 1:
        qc.z(0)
    else:
        qc.h(k - 1)
        qc.mcx(list(range(k - 1)), k - 1)
        qc.h(k - 1)
    qc.x(range(k))
    qc.h(range(k))
    qc.global_phase += math.pi
    return qc


def optimal_iterations(n_total, n_marked):
    """Standard Grover-optimal repeat count, floor(0) clamped to at least 1
    (zero iterations would just return the flat superposition, useless for a
    search demonstration even when n_marked is large)."""
    if n_marked <= 0 or n_marked >= n_total:
        return 1
    return max(1, round(math.pi / 4 * math.sqrt(n_total / n_marked)))


def build_grover_search(reference, read, threshold, iterations=None):
    """Full search circuit: H state-prep + `iterations` rounds of
    (oracle marking, diffuser). Returns (circuit, shift_qubits, k_shift,
    shifts, good_shifts, iterations) where good_shifts is the classically
    computed list of shifts with true mismatch count <= threshold (used only
    to pick the default iteration count and by tests, never by the circuit
    itself to decide what to mark -- that's the oracle's job, done
    coherently).
    """
    from hwlib import ENC

    L = len(read)
    shifts = list(range(len(reference) - L + 1))
    k_shift = max(1, (len(shifts) - 1).bit_length())
    layout = oracle_layout(L, k_shift)
    good_shifts = [s for s in shifts if _true_mismatch(reference, read, s) <= threshold]

    if iterations is None:
        iterations = optimal_iterations(2 ** k_shift, len(good_shifts))

    qc = QuantumCircuit(layout["num_qubits"], name=f"grover_search_L{L}_t{threshold}")

    for i, base in enumerate(read):
        for t, bit in enumerate(ENC[base]):
            if bit:
                qc.x(layout["rd"][2 * i + t])

    qc.h(layout["shift"])
    diffuser = build_grover_diffuser(k_shift)
    for _ in range(iterations):
        append_oracle_marking(qc, reference, read, threshold, shifts, k_shift, layout)
        qc.compose(diffuser, qubits=layout["shift"], inplace=True)

    return qc, layout["shift"], k_shift, shifts, good_shifts, iterations
