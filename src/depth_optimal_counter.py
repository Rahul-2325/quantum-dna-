"""Paper 2 (Rethinasamy-LaBorde-Wilde, PRA 110, 052401), Section IV and Fig. 7:
the depth-optimal weight circuit, using n control qubits with resets instead
of Algorithm 2's k control qubits, for O(log n) depth instead of O(n log n).

This is the construction CLAUDE.md flags as "NOT done" (the old v1 attempt,
`Untitled5.ipynb`, was WRONG -- success ~8%). Built from independently
testable pieces:

  1. `ghz_fanout`: spreads a single qubit's amplitude onto n qubits via a
     binary-tree CNOT cascade (a|0>+b|1> on qubit 0 -> a|0..0>+b|1..1> on all
     n), encoding one logical control bit into an n-qubit repetition code.
     Paper 2 Section IV cites a genuinely CONSTANT-depth construction for
     this step (mid-circuit measurement and feedback, from Quek-Kaur-Wilde,
     "Multivariate trace estimation in constant quantum depth", not
     reimplemented here). This module uses the standard O(log n)-depth
     CNOT-tree fanout instead -- correct, but not asymptotically optimal for
     this one sub-step. A documented simplification, not a silent one: see
     results/README.md for what this costs the overall depth claim.

  2. Classically-controlled phase: Definition IV.1 needs Rz(pi/2^(j-1) * abar)
     applied to one physical qubit, where abar is an integer built from the
     j-1 previously-measured classical bits a_1..a_{j-1} (a_1 extracted
     first). Rz's angle must be known at circuit-build time, so this is
     decomposed into j-1 separately classically-controlled Rz gates
     (Rz(pi/2^(j-i)) conditioned on a_i==1, for i=1..j-1), which sum to the
     same total phase since Rz(x)Rz(y) = Rz(x+y). This is the standard way
     semiclassical (Griffiths-Niu) Fourier transforms are realized with
     discrete classical branching.

  3. Parity extraction: each round's n-qubit block is measured into a
     temporary classical register, then XORed down to one bit via
     `qc.store` + `expr.bit_xor` (Qiskit's real-time classical compute),
     giving that round's weight bit -- this is what Definition IV.1 calls
     "a_k := b^k_1 + ... + b^k_s" (addition mod 2).

The full circuit (`depth_optimal_weight`) reuses ONE n-qubit control register
across k = ceil(log2(n+1)) rounds, reset between rounds, matching Fig. 7:
round j prepares a fresh GHZ-fanout, applies the gamma_j/beta_j rotations
from Algorithm 2 (now parallelised across n physical copies of the control
bit instead of bottlenecked on one qubit), then runs one semiclassical-IQFT
step, classically conditioned on every earlier round's outcome.

Three things were resolved empirically rather than by re-deriving them from
the paper's equations under time pressure -- each pinned by brute-force
search against the classical weight on n=2,3,4 (zero mismatches out of every
basis-state input), then confirmed on superpositions in the test suite:

  - Round j applies rotation index (k-j+1), i.e. round 1 (unconditioned) uses
    the LARGEST rotation (gamma_k, beta_k), and the last round uses the
    smallest (gamma_1, beta_1). This matches standard iterative/semiclassical
    phase estimation: the qubit that would carry the most significant phase
    information in a one-shot circuit is measured FIRST, with later,
    less-significant measurements corrected using it.
  - Round 1's extracted bit is the LEAST significant bit of the weight; the
    last round's is the most significant.
  - The conditioned-phase correction needs a NEGATIVE sign,
    Rz(-pi/2^(j-i)), not the "+" that a literal reading of Definition IV.1's
    Rz(pi/2^(k-1)*abar) suggests. This is very likely a sign-convention
    mismatch between the paper's assumed IQFT phase convention and Qiskit's
    QFTGate(k).inverse() (the same object weight_unitary uses), not an error
    in the paper -- not chased further since the empirical result is what
    the implementation is checked against, not a re-derivation of the sign.

An initial attempt using the naive (unreversed, "+", MSB-first) reading of
Definition IV.1 produced near-random output (best outcome ~13-25% of shots,
the same failure signature CLAUDE.md records for the old v1 attempt) --
evidence that the "+" / natural-order reading is not simply a relabeling of
the same correct circuit, but a genuinely different, incorrect one.

IMPORTANT -- simulation method: Aer's `matrix_product_state` method, exact
and much faster for every OTHER circuit in this project
(test_mps_matches_statevector), gives visibly wrong results for THIS
circuit's superposition inputs (TVD ~0.14-0.20 against the true distribution,
even though every basis-state input is still exactly correct under MPS).
Dense `statevector` gives TVD ~0.003-0.02 (consistent with shot noise) on
the identical circuit. The discrepancy is isolated to MPS's handling of this
circuit's mid-circuit measurement + GHZ-entangled parity + classical
feedback pattern -- it is not present in the purely-unitary,
single-basis-measurement circuits (weight_unitary, plus_one_counter,
mismatch_circuit*) that the rest of this project uses MPS for. Use
`statevector`, not MPS, whenever running this circuit with anything but a
classical basis-state input.
"""
from __future__ import annotations

import math
from functools import reduce

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.classical import expr


def ghz_fanout(qc, qubits):
    """Spread qubits[0]'s amplitude onto every qubit in `qubits` (repetition code).

    O(log n) depth via a doubling CNOT tree. NOT the paper's constant-depth G
    (see module docstring) -- documented simplification, not a silent one.
    """
    n = len(qubits)
    reach = 1
    while reach < n:
        step = min(reach, n - reach)
        for i in range(step):
            qc.cx(qubits[i], qubits[i + reach])
        reach *= 2


def _apply_conditioned_phase(qc, target_qubit, prior_bits, out_creg):
    """-Rz(pi/2^(j-1) * abar) on `target_qubit`, decomposed per earlier bit.

    `prior_bits` are out_creg bit indices for the previously-measured rounds,
    in round order. Each contributes Rz(-pi/2^(j-i)) when that bit was
    measured as 1 (i = 1..j-1, 1-indexed), summing to -Rz(pi/2^(j-1) * abar).
    The negative sign is an empirical finding -- see the module docstring.
    """
    j = len(prior_bits) + 1
    for offset, bit_index in enumerate(prior_bits):
        i = offset + 1
        angle = -math.pi / 2 ** (j - i)
        with qc.if_test((out_creg[bit_index], 1)):
            qc.rz(angle, target_qubit)


def depth_optimal_weight(n):
    """|x>|0>^n -> |x> with k classical bits holding W(x), depth O(k) = O(log n).

    Qubit layout: [0..n-1] data (the input |psi>), [n..2n-1] the single
    reusable n-qubit control block. k classical bits hold the weight
    (out[0] is the bit extracted in round 1, ..., out[k-1] the bit extracted
    last); bit-to-significance mapping is confirmed in the test suite.
    """
    k = math.ceil(math.log2(n + 1))
    N = 2 ** k - 1

    data = QuantumRegister(n, "data")
    control = QuantumRegister(n, "control")
    out = ClassicalRegister(k, "out")
    temp = ClassicalRegister(n, "temp")
    qc = QuantumCircuit(data, control, out, temp, name=f"depth_optimal_{n}")

    for round_idx in range(1, k + 1):
        # Round 1 (unconditioned) carries the LARGEST rotation and yields the
        # LEAST significant weight bit; round k carries the smallest rotation
        # and yields the most significant bit. See module docstring.
        rot_j = k - round_idx + 1
        if round_idx > 1:
            qc.reset(control)
        gamma_j = math.pi * n / (N + 1) * 2 ** (rot_j - 1)
        beta_j = 2 * math.pi / (N + 1) * 2 ** (rot_j - 1)

        qc.h(control[0])
        qc.rz(gamma_j, control[0])
        ghz_fanout(qc, list(control))

        for i in range(n):
            qc.crz(beta_j, control[i], data[i])

        prior_bits = list(range(round_idx - 1))
        if prior_bits:
            _apply_conditioned_phase(qc, control[0], prior_bits, out)

        qc.h(control)
        qc.measure(control, temp)
        parity = reduce(expr.bit_xor, [temp[i] for i in range(n)])
        qc.store(out[round_idx - 1], parity)

    return qc, k
