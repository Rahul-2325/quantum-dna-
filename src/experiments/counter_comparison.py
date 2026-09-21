"""C4: side-by-side noise comparison of the adder (Paper 1) vs phase (Paper 2)
mismatch counters. Same transpile target, same noise models (virtual_rz),
same stratified sampling, same seeds and shots as noise_degradation.py --
this script reuses that module's run_cell/circuit_stats/cell_metrics/
pair_statistics machinery with only the circuit builder swapped, so any
difference in the results is attributable to the counter, not the harness.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt

from aer_helpers import mismatch_circuit_measured
from mismatch_adder import mismatch_circuit_adder_measured
from noise_degradation import (CSV_FIELDNAMES, P_GRID, READOUT_ERROR, SEED_PAIRS, SEED_SIM,
                               THERMAL, VIRTUAL_RZ, cell_metrics, circuit_stats,
                               git_commit_hash, is_noiseless_control, pair_statistics,
                               run_cell, scenario_noise_model)

COUNTERS = {"adder": mismatch_circuit_adder_measured, "phase": mismatch_circuit_measured}
COMPARISON_FIELDNAMES = ["counter"] + CSV_FIELDNAMES


def run_comparison(levels, pairs_per_m, shots, results_dir, tag=None, force=False):
    """Same incremental-write-per-row design as run_sweep, and for the same reason:
    a long Aer run at L=8 has twice raised a hard MemoryError partway through, so
    batching all rows into memory and writing once at the end risks losing a full
    run's worth of completed cells to one crash near the finish line."""
    import qiskit
    import qiskit_aer

    results_dir = Path(results_dir)
    stamp = date.today().isoformat()
    suffix = f"_{tag}" if tag else ""
    csv_path = results_dir / f"counter_comparison_{stamp}{suffix}.csv"
    meta_path = results_dir / f"counter_comparison_{stamp}{suffix}_meta.json"
    if not force:
        for path in (csv_path, meta_path):
            if path.exists():
                raise FileExistsError(f"{path} already exists; pass --tag or --force")

    started = time.perf_counter()
    stats = {(counter, L): circuit_stats(L, circuit_builder=builder)
             for counter, builder in COUNTERS.items() for L in levels}
    results_dir.mkdir(parents=True, exist_ok=True)

    row_count = 0
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDNAMES)
        writer.writeheader()
        handle.flush()

        for counter, builder in COUNTERS.items():
            for scenario in ("S1", "S2"):
                for L in levels:
                    num_qubits = stats[(counter, L)]["num_qubits"]
                    for p in P_GRID:
                        noise_model = scenario_noise_model(scenario, p, num_qubits)
                        noiseless_control = is_noiseless_control(scenario, p)
                        for m in range(L + 1):
                            rng = random.Random(SEED_PAIRS + 7919 * L + 104729 * m)
                            started_cell = time.perf_counter()
                            counts, per_pair, k = run_cell(
                                scenario, L, p, m, pairs_per_m, shots, rng, noise_model,
                                verify_coverage=(m == 0 and not noiseless_control),
                                circuit_builder=builder)
                            trials = pairs_per_m * shots
                            metrics = cell_metrics(counts, L, m, trials)
                            metrics.update(pair_statistics(per_pair, m, shots))
                            if (scenario == "S1" and p == 0.0
                                    and metrics["p_correct"] != 1.0):
                                raise RuntimeError(
                                    f"{counter} noiseless S1 cell L={L} m={m} gave "
                                    f"P(correct)={metrics['p_correct']!r}, expected "
                                    f"exactly 1.0")
                            row = {"counter": counter, "scenario": scenario, "L": L,
                                   "p": p, "m": m, "pairs_per_m": pairs_per_m,
                                   "shots": shots, "n_trials": trials, **metrics,
                                   **stats[(counter, L)], "seed_pairs": SEED_PAIRS,
                                   "seed_sim": SEED_SIM,
                                   "elapsed_s": time.perf_counter() - started_cell}
                            writer.writerow(row)
                            handle.flush()
                            row_count += 1
                            print(f"  {counter:>5} {scenario} L={L} p={p:<6} m={m}  "
                                  f"P(correct)={metrics['p_correct']:.4f}  "
                                  f"({time.perf_counter() - started_cell:.1f}s)",
                                  flush=True)

    total_seconds = time.perf_counter() - started
    meta = {"date": stamp, "git_commit": git_commit_hash(),
            "qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__,
            "seed_pairs": SEED_PAIRS, "seed_sim": SEED_SIM, "shots": shots,
            "pairs_per_m": pairs_per_m, "levels": list(levels), "p_grid": list(P_GRID),
            "counters": list(COUNTERS), "virtual_rz": VIRTUAL_RZ,
            "thermal": THERMAL, "readout_error": READOUT_ERROR,
            "circuit_stats": {f"{c}_{L}": stats[(c, L)] for c, L in stats},
            "total_runtime_s": total_seconds, "rows": row_count}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nwrote {csv_path} ({row_count} rows) and {meta_path} "
          f"in {total_seconds / 60:.1f} min")
    return csv_path, meta_path


def make_figure(csv_path, out_path, levels):
    """P(correct) vs p, per L, both counters overlaid (S2, mean over m)."""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    fig, axes = plt.subplots(1, len(levels), figsize=(3.2 * len(levels), 3.2), sharey=True)
    if len(levels) == 1:
        axes = [axes]
    for ax, L in zip(axes, levels):
        for counter, marker in (("adder", "o-"), ("phase", "s-")):
            xs, ys = [], []
            for p in P_GRID:
                sel = [float(r["p_correct"]) for r in rows
                      if r["counter"] == counter and r["scenario"] == "S2"
                      and int(r["L"]) == L and float(r["p"]) == p]
                if sel:
                    xs.append(p); ys.append(sum(sel) / len(sel))
            ax.plot(xs, ys, marker, label=counter)
        ax.set_title(f"L={L}"); ax.set_xlabel("2-qubit error rate p")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("P(correct), mean over m"); axes[0].legend()
    fig.suptitle("Adder (Paper 1) vs phase (Paper 2) mismatch counter, S2 noise")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--pairs-per-m", type=int, default=8)
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--results-dir", default=str(
        Path(__file__).resolve().parents[2] / "results"))
    parser.add_argument("--figures-dir", default=str(
        Path(__file__).resolve().parents[2] / "figures"))
    parser.add_argument("--tag", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    csv_path, _ = run_comparison(args.levels, args.pairs_per_m, args.shots,
                                 args.results_dir, tag=args.tag, force=args.force)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    make_figure(csv_path, figures_dir / f"6_counter_comparison_noise{suffix}.png",
               args.levels)


if __name__ == "__main__":
    main()
