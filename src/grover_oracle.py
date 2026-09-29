"""Grover-oracle feasibility work (novelty direction, see PROJECT_STATUS.md
Section 4.3): can a single circuit correctly count DNA mismatches at EVERY
possible read-alignment shift simultaneously, in superposition, and then mark
"good" shifts (mismatch count <= threshold) with the phase flip Grover's
algorithm needs -- fully coherently, with every working register cleanly
uncomputed afterward?

This module builds and verifies the ORACLE HALF of that idea. It does NOT
implement the diffusion operator or the repeated amplification loop -- those
are comparatively standard, generic Grover machinery. The oracle
(coherent controlled-shift + compare + count + threshold-mark + uncompute)
is the hard, novel, and risky part specific to this project, and the part
verified here.

Architecture:
  1. Load the read once, unconditionally (X gates) -- same for every shift.
  2. Superpose the shift register over 2^k_shift computational basis states
     (some may be "leftover"/invalid if the number of valid shifts isn't a
     power of 2 -- those branches simply never get a window loaded and are
     excluded from correctness checks, since they aren't real alignments).
  3. For each valid shift s: decode a single reusable ancilla "active_s" via
     a multi-controlled X matching the shift register to s, use it as the
     SOLE control for that shift's window-XOR gates (so only shift s's
     branch loads shift s's reference window), then uncompute the decoder.
  4. Shared, unconditional: OR-flag the loaded window against the read
     (the same prefix hwlib.mismatch_circuit and every other counter in this
     project use), then count via weight_unitary (Paper 2's phase-kickback
     counter -- the specific counter this Grover work is paired with, per
     the novelty framing: nobody has combined THIS counter with Grover
     search).
  5. Mark: prepare an ancilla in |-> (phase-kickback trick: X|-> = -|->), and
     for every count value v <= threshold, apply a multi-controlled X
     (matching the count register to v) onto that ancilla. Only the actual
     count value's block ever fires, so this correctly ORs together
     "count == 0", "count == 1", ..., "count == threshold" into a single
     "count <= threshold" condition -- and, via phase kickback, applies a
     -1 phase to exactly those branches.
  6. Uncompute count, then flags, then the window XOR, in reverse -- all of
     these operations happen to be self-inverse (weight_unitary needs its
     own .inverse(); the OR-flag and window-XOR gates are literally their
     own inverse), so "uncompute" is calling the same code again.

Verified (tests/test_grover_oracle.py), not assumed:
  - After step 6, EVERY qubit except the shift register collapses to a
    single computational basis outcome with probability 1.0 (zero leftover
    entanglement) -- checked by inspecting the full outcome distribution of
    every non-shift qubit, not just spot-checking a few states.
  - The shift register's raw complex amplitude (not just probability, which
    cannot see a phase flip) has one consistent sign for every shift whose
    true mismatch count is <= threshold, the OPPOSITE consistent sign for
    every shift whose true count is > threshold, checked across multiple
    threshold values and reference/read combinations.

NOT yet built: the diffusion operator and the amplification loop (repeat
oracle + diffusion ~pi/4*sqrt(N/M) times) that would turn this oracle into an
actual working search. This module answers "is the hard part correct",
not "does the full algorithm find the best alignment faster than checking
shifts one at a time" -- that is future work, tracked in PROJECT_STATUS.md.
"""
from __future__ import annotations

from qiskit import QuantumCircuit

from hwlib import ENC, weight_unitary


def _multi_controlled_x(qc, control_qubits, pattern_bits, target):
    """Flip `target` iff control_qubits == pattern_bits (a tuple of 0/1)."""
    flip = [q for q, bit in zip(control_qubits, pattern_bits) if bit == 0]
    for q in flip:
        qc.x(q)
    qc.mcx(control_qubits, target)
    for q in flip:
        qc.x(q)


def build_grover_oracle(reference, read, threshold):
    """The oracle circuit: phase-marks shifts with mismatch count <= threshold.

    Qubit layout: [shift (k_shift) | decoder (1) | flags (L) | rd (2L) |
    count (k_count) | marker (1)]. Returns (circuit, shift_qubits, k_shift,
    valid_shifts) -- valid_shifts lists which of the 2^k_shift shift-register
    values correspond to a real alignment position (the rest, if any, are
    unused "leftover" branches from padding to a power of two).
    """
    L = len(read)
    shifts = list(range(len(reference) - L + 1))
    k_shift = max(1, (len(shifts) - 1).bit_length())
    weight_circuit, k_count = weight_unitary(L)

    shift = list(range(k_shift))
    decoder = k_shift
    flags = list(range(k_shift + 1, k_shift + 1 + L))
    rd = list(range(k_shift + 1 + L, k_shift + 1 + L + 2 * L))
    count = list(range(k_shift + 1 + L + 2 * L, k_shift + 1 + L + 2 * L + k_count))
    marker = k_shift + 1 + L + 2 * L + k_count

    qc = QuantumCircuit(marker + 1, name=f"grover_oracle_L{L}_t{threshold}")

    for i, base in enumerate(read):
        for t, bit in enumerate(ENC[base]):
            if bit:
                qc.x(rd[2 * i + t])

    qc.h(shift)

    def window_xor_block():
        """Self-inverse: calling this twice returns rd to just the read."""
        for s in shifts:
            pattern = tuple((s >> b) & 1 for b in range(k_shift))
            _multi_controlled_x(qc, shift, pattern, decoder)
            window = reference[s:s + L]
            for i, base in enumerate(window):
                for t, bit in enumerate(ENC[base]):
                    if bit:
                        qc.cx(decoder, rd[2 * i + t])
            _multi_controlled_x(qc, shift, pattern, decoder)

    def flag_block():
        """Self-inverse OR-flag computation, same as hwlib.mismatch_circuit's prefix."""
        for i in range(L):
            a, b = rd[2 * i], rd[2 * i + 1]
            qc.x(a); qc.x(b); qc.ccx(a, b, flags[i]); qc.x(flags[i]); qc.x(a); qc.x(b)

    window_xor_block()
    flag_block()
    qc.compose(weight_circuit, qubits=count + flags, inplace=True)

    qc.x(marker)
    qc.h(marker)
    for value in range(min(threshold, 2 ** k_count - 1) + 1):
        pattern = tuple((value >> b) & 1 for b in range(k_count))
        _multi_controlled_x(qc, count, pattern, marker)
    qc.h(marker)
    qc.x(marker)

    qc.compose(weight_circuit.inverse(), qubits=count + flags, inplace=True)
    flag_block()
    window_xor_block()

    return qc, shift, k_shift, shifts
