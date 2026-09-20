# Project context (read this first)

## Goal
Write a publishable paper on **noise-aware, low-ancilla mismatch counting for approximate DNA read alignment**,
using coherent Hamming-weight circuits. Secondary case study: `hwb` (hidden weighted bit) from the same primitive.

## The three base papers (PDFs go in `papers/`)
1. **Bravyi-Yoder-Maslov (IEEE TC 2022)**: ancilla-free hwb; O(n^2) quantum circuit, O(n^6.42) reversible. Role: space-time trade-off case study.
2. **Rethinasamy-LaBorde-Wilde (PRA 110, 052401)**: coherent Hamming-weight measurement. Alg.1 (zero padding), Alg.2 (no padding),
   depth-width trade-off via repetition-code fan-out + semiclassical encoded IQFT, O(log n) depth with n control qubits (resets). Role: core primitive.
3. **Jamaludin-Lee-Kim (IEEE Access 2025)**: DNA alignment (Varsamis baseline) + 3rd error-detection qubit per base + FakeJakartaV2 noise + ZNE with confidence early-stop. Role: application domain and baseline to beat.

## Verified so far (noiseless simulation only) - see `src/hwlib.py`, `tests/`, `notebooks/01_verification_v2.ipynb`
- `weight_unitary(n)` = Paper-2 Alg.2 as a unitary |x>|0> -> |x>|W(x)>; matches ||P_x psi||^2 on random superpositions (~1e-16).
- `mismatch_circuit(read, window)`: 2 bits/base (A=00,C=01,G=10,T=11), flag_i = OR of XOR bits, weight circuit on flags -> mismatch count. 466/466 random pairs exact.
- `hwb_circuit(n)`: weight -> controlled cyclic shifts by 2^j -> uncompute. Correct for all inputs n=3..7; CX ~ 10-12.6 * n*ceil(log2(n+1)).

## NOT done / known issues
- The v1 "depth-optimal" circuit (old Untitled5.ipynb) is WRONG (success ~8%). Needs a real rewrite: GHZ/repetition-code fan-out + semiclassical IQFT with classically controlled Rz (Qiskit dynamic circuits).
- NO noise results yet. Nothing about robustness may be claimed until there are saved results in `results/`.
- Paper 3 checks: FakeJakartaV2 is (as far as we know) a 7-qubit device model but its circuits use ~27 qubits - verify how noise reaches the extra qubits before using it as a baseline. Its evaluation has only 6 test cases.
- Fairness caveat: Paper 3 stores the whole reference in qubits; ours loads one window classically per shift. State this in any comparison.

## Novelty warnings
Hamming-distance quantum read alignment already exists (QiBAM, Sarkar et al. 2019; QShift-SA 2026 uses adders). The contribution must be the
phase/QFT counter vs adder counter vs Paper-3 scheme, judged on qubits, depth AND noise robustness. Do a literature check before writing claims.

## Conventions
- Qiskit >= 2.1: use `QFTGate(k).inverse()`, not the deprecated `QFT`. Qubit order is little-endian (qubit 0 = LSB).
- Every experiment = a script in `src/` or a notebook cell that writes a CSV to `results/` (with seeds, shots, noise-model params). Never paste numbers by hand.
- Run `pytest -q` before and after any change to `src/hwlib.py`.
- Notebooks must stay self-contained: a notebook running on a *Colab kernel from VS Code cannot import local files*. Keep the library code inside the notebook
  (or `git clone` the repo in the first cell) and keep it in sync with `src/hwlib.py`.
- Commit after each experiment; describe what changed and which result file it produced.

## Next tasks (in order)
1. Noise experiments: baseline (Paper-3-style compare) vs weight-register counter under depolarizing + thermal-relaxation + readout noise; metrics = success prob, FP/FN at a mismatch threshold, TVD.
2. Rewrite depth-optimal circuit with dynamic circuits; verify against Alg.2 on random inputs.
3. Mitigation: ZNE (on P(outcome=w)), truncated IQFT, flag post-selection. Compare cost vs benefit.
4. Many reads (hundreds) with planted mismatches; report rates with confidence intervals.
5. Draft paper structure only after 1-4 have saved results.
