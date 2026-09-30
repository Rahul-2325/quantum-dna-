"""Grover-oracle feasibility work (novelty direction, see PROJECT_STATUS.md
Section 4.3): can a single circuit correctly count DNA mismatches at EVERY
possible read-alignment shift simultaneously, in superposition, and then mark
"good" shifts (mismatch count <= threshold) with the phase flip Grover's
algorithm needs -- fully coherently, with every working register cleanly
uncomputed afterward?

This module builds and verifies the ORACLE HALF of that idea; grover_search.py
builds the diffusion operator and the amplification loop on top of it.

Architecture:
  1. Load the read once, unconditionally (X gates) -- same for every shift.
  2. Superpose the shift register over 2^k_shift computational basis states.
     Whenever the number of real alignment positions (`shifts`) is not itself
     a power of two, padding to k_shift = ceil(log2(len(shifts))) leaves some
     basis-state branches ("leftover") with no corresponding real shift.
  3. For each valid shift s: decode a single reusable ancilla "active_s" via
     a multi-controlled X matching the shift register to s, use it as the
     SOLE control for that shift's window-XOR gates (so only shift s's
     branch loads shift s's reference window), then uncompute the decoder.
     Leftover branches never trigger any of these, so `rd` stays as just the
     loaded read for them.
  4. Shared, unconditional: OR-flag the loaded window against the read
     (the same prefix hwlib.mismatch_circuit and every other counter in this
     project use), then count via weight_unitary (Paper 2's phase-kickback
     counter -- the specific counter this Grover work is paired with, per
     the novelty framing: nobody has combined THIS counter with Grover
     search).
  5. Compute a persistent "valid" ancilla: 1 iff the shift register's current
     basis state matches one of the REAL shifts, 0 for leftover branches.
     Built the same way as the per-shift decoder (mutually-exclusive
     pattern-matched multi-controlled X), but left set (not immediately
     uncomputed) through the marking step below -- this is what stops
     leftover branches from being spuriously marked "good" just because
     their unloaded `rd` happens to count as zero mismatches, which would
     otherwise corrupt every Grover amplification computation whenever
     len(shifts) isn't a power of two (i.e. almost always).
  6. Mark: prepare an ancilla in |-> (phase-kickback trick: X|-> = -|->), and
     for every count value v <= threshold, apply a multi-controlled X
     (matching the count register to v AND valid=1) onto that ancilla. Only
     a genuinely valid, in-threshold branch ever fires, so this correctly
     ANDs "this is a real shift" with "count <= threshold", and applies a
     -1 phase to exactly those branches via phase kickback.
  7. Uncompute valid, count, flags, then the window XOR, in reverse -- all of
     these operations happen to be self-inverse (weight_unitary needs its
     own .inverse(); the OR-flag, window-XOR, and valid-computation blocks
     are each literally their own inverse), so "uncompute" is calling the
     same code again.

Verified (tests/test_grover_oracle.py), not assumed:
  - After step 7, EVERY qubit except the shift register collapses to a
    single computational basis outcome with probability 1.0 (zero leftover
    entanglement) -- checked by inspecting the full outcome distribution of
    every non-shift qubit, not just spot-checking a few states.
  - The shift register's raw complex amplitude (not just probability, which
    cannot see a phase flip) has one consistent sign for every shift whose
    true mismatch count is <= threshold, the OPPOSITE consistent sign for
    every shift whose true count is > threshold, checked across multiple
    threshold values and reference/read combinations.
  - (tests/test_grover_search.py) Leftover branches (when len(shifts) is not
    a power of two) are NEVER marked, regardless of threshold -- the defect
    step 5 exists to prevent.
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


def oracle_layout(L, k_shift):
    """Qubit index layout shared by build_grover_oracle and grover_search.py's
    iterated search circuit: [shift (k_shift) | decoder (1) | valid (1) |
    flags (L) | rd (2L) | count (k_count) | marker (1)]."""
    weight_circuit, k_count = weight_unitary(L)
    shift = list(range(k_shift))
    decoder = k_shift
    valid = k_shift + 1
    flags = list(range(k_shift + 2, k_shift + 2 + L))
    rd = list(range(k_shift + 2 + L, k_shift + 2 + L + 2 * L))
    count = list(range(k_shift + 2 + L + 2 * L, k_shift + 2 + L + 2 * L + k_count))
    marker = k_shift + 2 + L + 2 * L + k_count
    return dict(weight_circuit=weight_circuit, k_count=k_count, shift=shift,
               decoder=decoder, valid=valid, flags=flags, rd=rd, count=count,
               marker=marker, num_qubits=marker + 1)


def append_oracle_marking(qc, reference, read, threshold, shifts, k_shift, layout):
    """Append the marking transformation (steps 3-7 above) to `qc` in place.
    Does NOT include the initial H on the shift register -- callers that want
    a single phase-oracle application (build_grover_oracle) apply H once
    before calling this; grover_search.py's iterated loop applies H once as
    state prep and then calls this repeatedly, interleaved with the diffuser,
    without re-applying H.
    """
    L = len(read)
    shift, decoder, valid = layout["shift"], layout["decoder"], layout["valid"]
    flags, rd, count, marker = layout["flags"], layout["rd"], layout["count"], layout["marker"]
    weight_circuit, k_count = layout["weight_circuit"], layout["k_count"]

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

    def valid_block():
        """Self-inverse: sets `valid` to 1 iff the shift register matches one
        of the real `shifts` (at most one pattern can match a given basis
        state, so applying this sequence a second time always undoes it)."""
        for s in shifts:
            pattern = tuple((s >> b) & 1 for b in range(k_shift))
            _multi_controlled_x(qc, shift, pattern, valid)

    window_xor_block()
    flag_block()
    qc.compose(weight_circuit, qubits=count + flags, inplace=True)
    valid_block()

    qc.x(marker)
    qc.h(marker)
    for value in range(min(threshold, 2 ** k_count - 1) + 1):
        pattern = tuple((value >> b) & 1 for b in range(k_count))
        _multi_controlled_x(qc, count + [valid], pattern + (1,), marker)
    qc.h(marker)
    qc.x(marker)

    valid_block()
    qc.compose(weight_circuit.inverse(), qubits=count + flags, inplace=True)
    flag_block()
    window_xor_block()


def build_grover_oracle(reference, read, threshold):
    """The oracle circuit: phase-marks shifts with mismatch count <= threshold.

    Returns (circuit, shift_qubits, k_shift, valid_shifts) -- valid_shifts
    lists which of the 2^k_shift shift-register values correspond to a real
    alignment position (the rest, if any, are unused "leftover" branches from
    padding to a power of two, which this oracle never marks -- see
    oracle_layout's `valid` ancilla).
    """
    L = len(read)
    shifts = list(range(len(reference) - L + 1))
    k_shift = max(1, (len(shifts) - 1).bit_length())
    layout = oracle_layout(L, k_shift)

    qc = QuantumCircuit(layout["num_qubits"], name=f"grover_oracle_L{L}_t{threshold}")

    for i, base in enumerate(read):
        for t, bit in enumerate(ENC[base]):
            if bit:
                qc.x(layout["rd"][2 * i + t])

    qc.h(layout["shift"])
    append_oracle_marking(qc, reference, read, threshold, shifts, k_shift, layout)

    return qc, layout["shift"], k_shift, shifts
