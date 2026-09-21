"""C3: resource table for the adder (Paper 1) vs phase (Paper 2) mismatch counters.

For each L, reports qubits/CX/depth for the COUNTER ITSELF -- plus_one_counter(L)
vs weight_unitary(L), acting on the L flag qubits, decoupled from the
mismatch-circuit-specific prefix (read load / XOR / OR-flag), which is shared
infrastructure common to both and not part of what's being compared here:

  (i)  count only     -- the counter as used in our actual experiments: compute
                          the weight into k qubits and read it out (measure or
                          pass downstream). This is what mismatch_circuit and
                          mismatch_circuit_adder actually run.
  (ii) count + uncompute -- append the counter's own inverse, returning EVERY
                          qubit the counter touched (including its k-qubit
                          weight/control register) to |0>. This is the cost of
                          using the counter as a borrowed subroutine inside a
                          larger coherent oracle that cannot leave any residual
                          entanglement behind when it releases the qubits --
                          different from just letting plus_one_counter's OWN
                          carry/temp scratch self-clean (which it already does,
                          per tests/test_plus_one_counter.py; that is not what
                          (ii) is measuring -- (ii) uncomputes the OUTPUT too).

An earlier version of this script mistakenly uncomputed the full mismatch
circuit (prefix included), which roughly doubles CX for both counters
regardless of any real difference between them and so measured nothing
about the counters specifically. Fixed to isolate the counter alone.
"""
from __future__ import annotations

import csv
from pathlib import Path

from qiskit import QuantumCircuit, transpile

from hwlib import weight_unitary
from plus_one_counter import plus_one_counter

BASIS_GATES = ["cx", "rz", "sx", "x"]
LEVELS = (2, 4, 6, 8)


def _stats(qc, seed=1234):
    t = transpile(qc, basis_gates=BASIS_GATES, coupling_map=None,
                  optimization_level=1, seed_transpiler=seed)
    return {"qubits": qc.num_qubits, "cx": t.count_ops().get("cx", 0), "depth": t.depth()}


def _stats_with_uncompute(qc, seed=1234):
    """count + its own inverse appended.

    A `barrier` sits between the two halves. Without it, `qc.compose(qc.inverse())`
    with nothing in between is mathematically the identity, and Qiskit's
    optimizer can find and cancel that (fully, for some L; only partially for
    others, and even a spurious *negative* apparent cost at L=2 -- an early
    version of this function hit exactly that and it was a transpiler
    cancellation artifact, not a real measurement). Any genuine oracle usage
    has a real payload operation between compute and uncompute that blocks
    this cancellation, so the barrier is the physically honest stand-in for
    that -- it forces (ii) to actually mean "compute, then separately
    uncompute", which any real use of this counter would.
    """
    round_trip = qc.copy()
    round_trip.barrier()
    round_trip.compose(qc.inverse(), inplace=True)
    return _stats(round_trip, seed=seed)


def build_table(levels=LEVELS):
    rows = []
    for L in levels:
        adder, k_a, t_width = plus_one_counter(L)
        phase, k_p = weight_unitary(L)
        assert k_a == k_p

        row = {"L": L, "k": k_a, "adder_temp_qubits": t_width}
        for label, qc in (("adder", adder), ("phase", phase)):
            count_only = _stats(qc)
            with_uncompute = _stats_with_uncompute(qc)
            for key, value in count_only.items():
                row[f"{label}_{key}_i"] = value
            for key, value in with_uncompute.items():
                row[f"{label}_{key}_ii"] = value
            row[f"{label}_uncompute_adds_cx"] = with_uncompute["cx"] - count_only["cx"]
        rows.append(row)
    return rows


def print_table(rows):
    header = (f"{'L':>3} {'k':>3} | {'adder q':>8} {'adder cx(i)':>11} {'adder cx(ii)':>12} "
              f"{'adder depth(i)':>14} | {'phase q':>8} {'phase cx(i)':>11} "
              f"{'phase cx(ii)':>12} {'phase depth(i)':>14}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['L']:>3} {row['k']:>3} | {row['adder_qubits_i']:>8} "
              f"{row['adder_cx_i']:>11} {row['adder_cx_ii']:>12} {row['adder_depth_i']:>14} | "
              f"{row['phase_qubits_i']:>8} {row['phase_cx_i']:>11} {row['phase_cx_ii']:>12} "
              f"{row['phase_depth_i']:>14}")
    print()
    for row in rows:
        note = (f" ({row['adder_temp_qubits']} temp qubits, already self-cleaning per "
                f"increment)" if row["adder_temp_qubits"] else " (no temp qubits needed)")
        print(f"L={row['L']}: uncompute adds {row['adder_uncompute_adds_cx']} CX to adder"
              f"{note}, {row['phase_uncompute_adds_cx']} CX to phase")


def main():
    rows = build_table()
    print_table(rows)
    out_dir = Path(__file__).resolve().parents[2] / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "counter_resources.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
