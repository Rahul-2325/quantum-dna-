"""C3: resource table for the adder (Paper 1) vs phase (Paper 2) mismatch counters.

For each L, reports qubits/CX/depth for:
  (i)  count only            -- the circuit as built (mismatch_circuit / mismatch_circuit_adder)
  (ii) count + garbage uncompute -- needed whenever a counter is used coherently
                                     inside a larger oracle (its own qubits must
                                     return to |0> so they can be reused / don't
                                     leak phase into an outer computation)

Both counters turn out garbage-free already for (i): weight_unitary's control
register is the intended OUTPUT (not garbage) so nothing needs uncomputing,
and plus_one_counter's temp/carry scratch is proven (in
tests/test_plus_one_counter.py) to already return to |0> after every
increment, without any extra gates. So (ii) reported here is the SAME
circuit run forward then immediately inverted and appended -- a literal
compute+uncompute pair -- which is what "used inside an oracle and then
released" actually costs, even though the counter's own scratch needed no
help getting back to |0> on its own.
"""
from __future__ import annotations

import csv
from pathlib import Path

from qiskit import transpile

from hwlib import mismatch_circuit
from mismatch_adder import mismatch_circuit_adder

BASIS_GATES = ["cx", "rz", "sx", "x"]
LEVELS = (2, 4, 6, 8)


def _stats(qc, seed=1234):
    t = transpile(qc, basis_gates=BASIS_GATES, coupling_map=None,
                  optimization_level=1, seed_transpiler=seed)
    return {"qubits": qc.num_qubits, "cx": t.count_ops().get("cx", 0), "depth": t.depth()}


def _stats_with_uncompute(qc, seed=1234):
    """count + its own inverse appended, transpiled together (not two separate transpiles,
    since gate cancellation across the seam is exactly what a real oracle usage gets)."""
    round_trip = qc.compose(qc.inverse())
    return _stats(round_trip, seed=seed)


def build_table(levels=LEVELS):
    rows = []
    for L in levels:
        read, window = "A" * L, ("C" * (L // 2)) + ("A" * (L - L // 2))
        adder, k_a = mismatch_circuit_adder(read, window)
        phase, k_p = mismatch_circuit(read, window)
        assert k_a == k_p

        row = {"L": L, "k": k_a}
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
        print(f"L={row['L']}: uncompute adds {row['adder_uncompute_adds_cx']} CX to adder, "
              f"{row['phase_uncompute_adds_cx']} CX to phase "
              f"(both counters' own scratch is already garbage-free without it -- "
              f"see module docstring)")


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
