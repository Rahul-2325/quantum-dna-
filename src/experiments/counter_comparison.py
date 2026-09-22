"""C4: side-by-side noise comparison of the adder (Paper 1), phase (Paper 2),
and depth-optimal (task 2 rewrite) mismatch counters. Same transpile target,
same noise models (virtual_rz), same stratified sampling, same seeds and
shots as noise_degradation.py -- this script reuses that module's
run_cell/circuit_stats/cell_metrics/pair_statistics machinery with only the
circuit builder swapped, so any difference in the results is attributable to
the counter, not the harness.

Adding the depth-optimal counter required two small fixes to shared
infrastructure, both backward compatible: aer_helpers.control_distributions_aer
grew a `num_bits` parameter (depth_optimal_weight uses two classical
registers -- "out" and a scratch "temp" -- so Aer's count keys come back
space-separated, which plain int(bits,2) cannot parse; the other two counters
use a single register and are unaffected by leaving num_bits unset), and
noise_models.assert_full_coverage now skips "if_else"/"store" (classical
control-flow constructs from the classically-conditioned rotations, not
physical gates) the same way it already skipped barrier/delay/reset. See
those modules' docstrings for the important caveat this surfaced: gates
INSIDE an if_else block are not visible to assert_full_coverage's flat
iteration, so whether Aer's noise model actually reaches them at simulation
time is a separate, unverified question.
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

from aer_helpers import mismatch_circuit_measured, transpile_for_noise
from mismatch_adder import mismatch_circuit_adder_measured
from mismatch_depth_optimal import mismatch_circuit_depth_optimal
from mismatch_qft_adder import mismatch_circuit_qft_adder_measured
from noise_degradation import (CSV_FIELDNAMES, P_GRID, READOUT_ERROR, SEED_PAIRS, SEED_SIM,
                               THERMAL, VIRTUAL_RZ, cell_metrics, circuit_stats,
                               git_commit_hash, is_noiseless_control, pair_statistics,
                               run_cell, scenario_noise_model)

COUNTERS = {"adder": mismatch_circuit_adder_measured, "phase": mismatch_circuit_measured,
           "depth_optimal": mismatch_circuit_depth_optimal,
           "qft_adder": mismatch_circuit_qft_adder_measured}
COMPARISON_FIELDNAMES = ["counter"] + CSV_FIELDNAMES


def run_comparison(levels, pairs_per_m, shots, results_dir, tag=None, force=False,
                   transpile_fn=transpile_for_noise):
    """Same incremental-write-per-row design as run_sweep, and for the same reason:
    a long Aer run at L=8 has twice raised a hard MemoryError partway through, so
    batching all rows into memory and writing once at the end risks losing a full
    run's worth of completed cells to one crash near the finish line.

    `transpile_fn` selects the transpile target; defaults to the existing
    all-to-all transpile_for_noise. Pass
    connectivity.transpile_connectivity_aware for a realistic heavy-hex
    coupling map instead.
    """
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
    stats = {(counter, L): circuit_stats(L, circuit_builder=builder, transpile_fn=transpile_fn)
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
                                circuit_builder=builder, transpile_fn=transpile_fn)
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
            "transpile": transpile_fn.__name__,
            "total_runtime_s": total_seconds, "rows": row_count}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nwrote {csv_path} ({row_count} rows) and {meta_path} "
          f"in {total_seconds / 60:.1f} min")
    return csv_path, meta_path


MARKERS = {"adder": "o-", "phase": "s-", "depth_optimal": "^-", "qft_adder": "d-"}


def make_figure(csv_path, out_path, levels):
    """P(correct) vs p, per L, all counters present in the CSV overlaid (S2, mean over m)."""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    present = [c for c in MARKERS if any(r["counter"] == c for r in rows)]
    fig, axes = plt.subplots(1, len(levels), figsize=(3.2 * len(levels), 3.2), sharey=True)
    if len(levels) == 1:
        axes = [axes]
    for ax, L in zip(axes, levels):
        for counter in present:
            xs, ys = [], []
            for p in P_GRID:
                sel = [float(r["p_correct"]) for r in rows
                      if r["counter"] == counter and r["scenario"] == "S2"
                      and int(r["L"]) == L and float(r["p"]) == p]
                if sel:
                    xs.append(p); ys.append(sum(sel) / len(sel))
            ax.plot(xs, ys, MARKERS[counter], label=counter)
        ax.set_title(f"L={L}"); ax.set_xlabel("2-qubit error rate p")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("P(correct), mean over m"); axes[0].legend()
    fig.suptitle(f"{' vs '.join(present)} mismatch counter, S2 noise")
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
    parser.add_argument("--connectivity-aware", action="store_true",
                        help="route onto a heavy-hex coupling map instead of all-to-all")
    args = parser.parse_args()

    if args.connectivity_aware:
        from connectivity import transpile_connectivity_aware
        transpile_fn = transpile_connectivity_aware
    else:
        transpile_fn = transpile_for_noise

    csv_path, _ = run_comparison(args.levels, args.pairs_per_m, args.shots,
                                 args.results_dir, tag=args.tag, force=args.force,
                                 transpile_fn=transpile_fn)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    make_figure(csv_path, figures_dir / f"6_counter_comparison_noise{suffix}.png",
               args.levels)


if __name__ == "__main__":
    main()
