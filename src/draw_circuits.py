"""Draw the implemented circuits and map each part back to the source papers.

Writes PNGs to figures/ and prints text diagrams. Run:
    python src/draw_circuits.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from qiskit import transpile

from aer_helpers import mismatch_circuit_measured, transpile_for_noise
from hwlib import ENC, hwb_circuit, mismatch_circuit, weight_unitary
from plus_one_counter import plus_one_counter, register_width

FIGURES = Path(__file__).resolve().parents[1] / "figures"


def save(qc, name, title):
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / f"{name}.png"
    qc.draw("mpl", fold=90, style="iqp").savefig(path, dpi=150, bbox_inches="tight")
    print(f"  wrote {path.relative_to(FIGURES.parent)}  ({title})")
    return path


def weight_figure(n=3):
    """Paper 2 (PRA 110, 052401) Algorithm 2 / Figure 3."""
    qc, k = weight_unitary(n)
    N = 2 ** k - 1
    print(f"""
================================================================================
1. weight_unitary(n={n})  ->  Paper 2, Algorithm 2 (Fig. 3, Eqs. 47-51)
================================================================================
Computes |x>|0> -> |x>|W(x)>: writes the Hamming weight of the n data qubits
into a k-qubit control register.  k = ceil(log2(n+1)) = {k},  N = 2^k - 1 = {N}.

Qubits 0..{k - 1}  = control register C (holds the weight, little-endian)
Qubits {k}..{k + n - 1}  = data register S (the n bits whose weight we count)

Step-by-step, each line matching the paper:
  Step 2 (Eq. 47): H on every control qubit       -> uniform superposition over y
  Step 3 (Eq. 48): rz(gamma_j) on control qubit j -> gamma_j = n*pi/(N+1) * 2^(j-1)
                   = the single rotation that replaces Algorithm 1's controlled
                     rotations on the zero-padded qubits (Lemma II.1), which is
                     exactly why Algorithm 2 needs no padding register.
  Step 4 (Eq. 50): crz(beta_j) from control j to EVERY data qubit
                   beta_j = 2*pi/(N+1) * 2^(j-1)   (Eq. 51)
                   = the controlled R_z^(x)n(beta_j), decomposed into n CRz gates.
  Step 5:          inverse QFT on the control register, then measure.

Note the code indexes j from 0 while the paper indexes from 1, so code 2**j
equals paper 2^(j-1). The angles are otherwise identical.
""")
    print(qc.draw("text", fold=100))
    save(qc, "1_weight_unitary", "Paper 2 Alg. 2")


def mismatch_figure(read="AC", window="AG"):
    """The DNA application: Paper 3's problem solved with Paper 2's primitive."""
    qc, k = mismatch_circuit(read, window)
    L = len(read)
    print(f"""
================================================================================
2. mismatch_circuit(read={read!r}, window={window!r})  ->  our contribution
================================================================================
Counts how many BASES differ between a read and a reference window.
This is the Paper 3 problem (DNA alignment) solved with the Paper 2 primitive.

Encoding: 2 qubits per base, {', '.join(f'{b}={a}{c}' for b, (a, c) in ENC.items())}
Total qubits = k + L + 2L = {k} + {L} + {2 * L} = {qc.num_qubits}

Qubits 0..{k - 1}      = control C: the MISMATCH COUNT is read from here
Qubits {k}..{k + L - 1}      = flags: one qubit per base, 1 if that base differs
Qubits {k + L}..{qc.num_qubits - 1}      = read register: 2 qubits per base

How it works:
  a) X gates load the read            ({read})
  b) X gates XOR in the window        ({window})  -> a base's 2 qubits are now
     both 0 only if that base matched
  c) flag_i = OR of the base's two XOR qubits, built as X-X-CCX-X-X-X
     (De Morgan: NOT(NOT a AND NOT b)). flag_i = 1 exactly when base i mismatched.
  d) weight_unitary on the flag register: the number of mismatches IS the
     Hamming weight of the flags, so Paper 2's circuit counts them into k qubits.

That is the whole idea: mismatch counting = Hamming weight of a flag register.
Paper 3 needs 3(L+N)+L(N-L+1) qubits to hold the whole reference; we need
{qc.num_qubits} per window, because we load one window classically per shift.
""")
    print(qc.draw("text", fold=100))
    save(qc, "2_mismatch_circuit", "DNA mismatch counter")

    measured, _ = mismatch_circuit_measured(read, window)
    transpiled = transpile_for_noise(measured)
    ops = transpiled.count_ops()
    print(f"""
--------------------------------------------------------------------------------
2b. The SAME circuit as actually executed (transpiled to cx/rz/sx/x)
--------------------------------------------------------------------------------
This is what the noise model sees: depth {transpiled.depth()}, gates {dict(ops)}.
The CCX and CRZ gates above are not hardware gates; they decompose into the
one- and two-qubit gates below. This is why the noise results depend on CX count.
""")
    save(transpiled, "2b_mismatch_transpiled", "as executed")


def hwb_figure(n=3):
    """Paper 1 (IEEE TC 2022): hidden weighted bit, built from the weight register."""
    qc, k = hwb_circuit(n)
    print(f"""
================================================================================
3. hwb_circuit(n={n})  ->  Paper 1 (Bravyi-Yoder-Maslov) case study
================================================================================
hwb(x) = cyclic shift of x to the right by |x| (its Hamming weight).
Paper 1 gives an ANCILLA-FREE O(n^2) circuit. We trade {k} ancillas for fewer
gates, which is the space-time trade-off their conclusion invites.

Construction (three blocks in the drawing below):
  1) weight_unitary  : compute |x| into the {k} control qubits
  2) controlled shifts: for each control qubit j, cyclically rotate the data by
     2^j positions, controlled on that qubit. Summing 2^j over the set bits of
     the weight gives a total rotation of exactly |x|.
     Each rotation is 3 reversals of the register (~n Fredkin/cswap gates).
  3) weight_unitary.inverse(): uncompute the control register back to |0>.

Why step 3 is valid: a cyclic rotation does not change the Hamming weight, so
after the shift the control register still holds the weight of the (rotated)
data, and the inverse computation cleanly disentangles it.
""")
    print(qc.draw("text", fold=100))
    save(qc, "3_hwb_circuit", "Paper 1 case study")


def plus_one_figure(n=7):
    """Paper 1 (IEEE TC 2022), Section 2 / Fig. 1: the CNOT/Toffoli 'plus-one' counter.

    n=7 is the paper's own worked example, drawn here the same size so the
    qubit count and gate pattern can be checked directly against their Fig. 1.
    """
    qc, k, t_width = plus_one_counter(n)
    print(f"""
================================================================================
4. plus_one_counter(n={n})  ->  Paper 1, Section 2 and Fig. 1 (baseline)
================================================================================
Computes |x>|0> -> |x>|W(x)> the OTHER way: instead of phase kickback + QFT
(Paper 2), this runs a controlled ripple-carry INCREMENT once per input bit,
exactly Paper 1's Fig. 1 (drawn there for n=7, hence the default here).

Qubits 0..{k - 1}        = weight register w (little-endian, same meaning as weight_unitary's C)
Qubits {k}..{k + n - 1}        = data x1..x{n} (the bits being counted)
Qubits {k + n}..{qc.num_qubits - 1}        = temp/carry scratch t (returns to |0> — see below)

Paper 1's own layout for n=7 (Fig. 1 caption) is
  |x1 x2 x3 x4 x5 x6 x7 w1 t1 w2 w3>
i.e. weight and temp qubits interleaved differently from ours, but the GATE
SEQUENCE is what we verify against, not the wire ordering, since ordering is
just a drawing choice.

For each input bit x_i (i=1..{n}), "increment w by 1 if x_i=1":
  i=1            : a single CNOT(x1, w1) -- the 1-bit case needs no carry.
  i=2,3          : Toffoli(xi,w1,w2) then CNOT(xi,w1) -- a plain 2-bit half adder.
  i>3            : ripple the carry forward through t (a Toffoli chain), apply
                   a half-adder at the current top bit (this is what may GROW
                   the register into a new bit), then ripple back, UNCOMPUTING
                   each carry the instant it has been used to set a sum bit.
                   That is why t is back at |0> after every single increment --
                   not just at the end of all {n} of them -- so the same scratch
                   qubits are reused for every i without ever needing a separate
                   'restore ancilla' pass. Verified exhaustively for n=1..8 in
                   tests/test_plus_one_counter.py (matches the classical weight
                   on every input, and t stays |0>).

Register width grows only as needed: at step i it is floor(log2(i))+1, so the
full k-wide register is only reached at the LAST increment (i={n}) -- earlier
steps in the drawing below touch fewer wires, which is visible as gates that
start appearing only partway across.
""")
    print(qc.draw("text", fold=110))
    save(qc, "4_plus_one_counter", "Paper 1 Sec. 2 baseline")


def counter_comparison_figure(levels=(2, 4, 6, 8)):
    """Side by side: Paper 1's adder counter vs Paper 2's phase counter, same n."""
    print(f"""
================================================================================
5. Resource comparison: plus_one_counter (Paper 1) vs weight_unitary (Paper 2)
================================================================================
Same transpile target (cx/rz/sx/x, all-to-all), optimization_level=1.
This is a quick preview -- the full comparison (same noise models, same
strata, same seeds, matching mismatch_circuit_adder against mismatch_circuit)
is part C4 of the baseline work, not yet run.
""")
    header = (f"{'n':>3} {'k':>3} | {'adder q':>7} {'adder cx':>8} {'adder depth':>11} | "
              f"{'phase q':>7} {'phase cx':>8} {'phase depth':>11}")
    print(header)
    print("-" * len(header))
    rows = []
    for n in levels:
        adder, k, t_width = plus_one_counter(n)
        adder_t = transpile(adder, basis_gates=["cx", "rz", "sx", "x"],
                            coupling_map=None, optimization_level=1, seed_transpiler=1234)
        phase, _ = weight_unitary(n)
        phase_t = transpile(phase, basis_gates=["cx", "rz", "sx", "x"],
                            coupling_map=None, optimization_level=1, seed_transpiler=1234)
        row = (n, k, adder.num_qubits, adder_t.count_ops().get("cx", 0), adder_t.depth(),
               phase.num_qubits, phase_t.count_ops().get("cx", 0), phase_t.depth())
        rows.append(row)
        print(f"{row[0]:>3} {row[1]:>3} | {row[2]:>7} {row[3]:>8} {row[4]:>11} | "
              f"{row[5]:>7} {row[6]:>8} {row[7]:>11}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    ns = [r[0] for r in rows]
    axes[0].plot(ns, [r[3] for r in rows], "o-", label="adder (Paper 1)")
    axes[0].plot(ns, [r[6] for r in rows], "s-", label="phase (Paper 2)")
    axes[0].set_xlabel("n"); axes[0].set_ylabel("CX count"); axes[0].legend()
    axes[0].set_title("CX count")
    axes[1].plot(ns, [r[4] for r in rows], "o-", label="adder (Paper 1)")
    axes[1].plot(ns, [r[7] for r in rows], "s-", label="phase (Paper 2)")
    axes[1].set_xlabel("n"); axes[1].set_ylabel("transpiled depth"); axes[1].legend()
    axes[1].set_title("Depth")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / "5_counter_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  wrote {path.relative_to(FIGURES.parent)}  (preview only, not the full C4 run)")


def main():
    print("Drawing the implemented circuits and mapping them to the papers.")
    weight_figure()
    mismatch_figure()
    hwb_figure()
    plus_one_figure()
    counter_comparison_figure()
    print(f"""
================================================================================
Summary of which paper each piece comes from
================================================================================
  Paper 2 (PRA 110, 052401)  -> weight_unitary: the core primitive, Alg. 2.
  Paper 3 (IEEE Access 2025) -> the problem mismatch_circuit solves, and the
                                baseline we compare qubit counts against.
  Paper 1 (IEEE TC 2022)     -> hwb_circuit: the secondary case study.

PNGs are in {FIGURES.relative_to(FIGURES.parent)}/ .
""")


if __name__ == "__main__":
    main()
