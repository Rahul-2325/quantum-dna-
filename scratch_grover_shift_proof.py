"""Proof of concept ONLY, not production code: does a coherent controlled-shift
+ compare + count sub-circuit correctly give the right mismatch count for
EVERY shift simultaneously, when the shift register is in superposition?

No Grover amplification yet, no uncompute of the output (count is meant to be
read), just: can we correctly correlate (shift, mismatch_count) coherently.

Architecture: per shift s, decode a single ancilla "active_s" (multi-controlled
X on the shift register matching binary(s)), use it as the SOLE control for
that shift's window-XOR gates (only), then the shared OR-flag + weight_unitary
count logic runs ONCE, unconditionally, on whatever ended up XORed into the
comparison register. Uncompute active_s after each shift's block so the
decoder ancilla is reusable and returns to |0> (doesn't leak which-shift info
outside the algorithm).
"""
import itertools
import sys

sys.path.insert(0, r"C:\Users\Punnam Rahul\Downloads\dna-hamming-qc-starter\dna-hamming-qc\src")

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from hwlib import ENC, weight_unitary

REFERENCE = "TAC"   # 3 bases -> 3 possible 1-base windows (shifts 0,1,2)
READ = "A"          # 1 base


def multi_controlled_x(qc, control_qubits, pattern_bits, target):
    """Flip `target` iff control_qubits == pattern_bits (0/1 tuple)."""
    flip = [q for q, bit in zip(control_qubits, pattern_bits) if bit == 0]
    for q in flip:
        qc.x(q)
    qc.mcx(control_qubits, target)
    for q in flip:
        qc.x(q)


def build(reference, read):
    L = len(read)
    shifts = list(range(len(reference) - L + 1))
    k_shift = max(1, (len(shifts) - 1).bit_length())
    W, k_count = weight_unitary(L)

    shift = list(range(k_shift))
    decoder = k_shift
    flags = list(range(k_shift + 1, k_shift + 1 + L))
    rd = list(range(k_shift + 1 + L, k_shift + 1 + L + 2 * L))
    count = list(range(k_shift + 1 + L + 2 * L, k_shift + 1 + L + 2 * L + k_count))

    qc = QuantumCircuit(k_shift + 1 + L + 2 * L + k_count)

    # Load the read once, unconditionally -- same for every shift.
    for i, base in enumerate(read):
        for t, bit in enumerate(ENC[base]):
            if bit:
                qc.x(rd[2 * i + t])

    # Superpose the shift register over 2^k_shift values (some may be invalid
    # "leftover" shifts if len(shifts) isn't a power of 2 -- fine for this
    # proof of concept; correctness is only checked on the VALID shifts).
    qc.h(shift)

    for s in shifts:
        pattern = tuple((s >> b) & 1 for b in range(k_shift))
        multi_controlled_x(qc, shift, pattern, decoder)
        window = reference[s:s + L]
        for i, base in enumerate(window):
            for t, bit in enumerate(ENC[base]):
                if bit:
                    qc.cx(decoder, rd[2 * i + t])   # XOR this window base, ONLY on this shift's branch
        multi_controlled_x(qc, shift, pattern, decoder)   # uncompute the decoder ancilla

    # Shared, unconditional: OR-flag then count -- same code as
    # hwlib.mismatch_circuit's own prefix, operating on whatever window ended
    # up XORed into `rd` for each branch.
    for i in range(L):
        a, b = rd[2 * i], rd[2 * i + 1]
        qc.x(a); qc.x(b); qc.ccx(a, b, flags[i]); qc.x(flags[i]); qc.x(a); qc.x(b)
    qc.compose(W, qubits=count + flags, inplace=True)  # weight_unitary expects [control|data]

    return qc, shift, flags, count, k_shift, k_count, shifts


def main():
    qc, shift, flags, count, k_shift, k_count, shifts = build(REFERENCE, READ)
    print(f"reference={REFERENCE!r} read={READ!r} shifts={shifts} "
          f"k_shift={k_shift} k_count={k_count} total_qubits={qc.num_qubits}")

    state = Statevector(qc)
    # Marginal over (shift register, count register) jointly.
    joint = state.probabilities_dict(qargs=shift + count)
    # Aggregate by (shift_value, count_value).
    from collections import defaultdict
    agg = defaultdict(float)
    for bits, p in joint.items():
        # qargs order: bits string has LAST-listed qarg leftmost per Qiskit convention;
        # shift+count listed in that order means count is printed first (leftmost).
        count_bits = bits[:k_count]
        shift_bits = bits[k_count:]
        agg[(int(shift_bits, 2), int(count_bits, 2))] += p

    print("\n(shift, count) -> probability, with classical truth for valid shifts:")
    for (s, c), p in sorted(agg.items()):
        if p < 1e-6:
            continue
        truth = None
        if s in shifts:
            window = REFERENCE[s:s + len(READ)]
            truth = sum(a != b for a, b in zip(READ, window))
        flag = "" if truth is None else ("OK" if truth == c else "MISMATCH!!")
        print(f"  shift={s} count={c}  p={p:.4f}  truth={truth}  {flag}")


if __name__ == "__main__":
    main()
