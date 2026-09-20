"""Noise degradation of the weight-register mismatch counter.

Scenarios:
  S1 = depolarizing only (1q = p/10, 2q = p), thermal relaxation and readout off.
  S2 = S1 + thermal relaxation (T1/T2, 50ns/300ns gates) + symmetric readout error.

Pairs are stratified by true mismatch count m: for each (L, m) the reference
window is the read with exactly m planted substitutions.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from datetime import date
from pathlib import Path

from aer_helpers import (control_distribution_aer, mismatch_circuit_measured,
                         transpile_for_noise)
from noise_models import assert_full_coverage, build_noise_model

BASES = "ACGT"
P_GRID = (0.0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05)
THERMAL = dict(t1=100_000.0, t2=80_000.0, gate_time_1q=50.0, gate_time_2q=300.0)
READOUT_ERROR = 0.01
SEED_PAIRS = 20260920
SEED_SIM = 761
Z_95 = 1.959963984540054


def sample_stratified_pair(L, m, rng):
    """Random read of length L plus a window differing in exactly m positions."""
    read = [rng.choice(BASES) for _ in range(L)]
    window = list(read)
    for position in rng.sample(range(L), m):
        window[position] = rng.choice([b for b in BASES if b != read[position]])
    return "".join(read), "".join(window)


def wilson_interval(successes, trials, z=Z_95):
    """Wilson score interval for a binomial proportion."""
    if trials == 0:
        return float("nan"), float("nan")
    p_hat = successes / trials
    denominator = 1 + z ** 2 / trials
    centre = (p_hat + z ** 2 / (2 * trials)) / denominator
    spread = z * math.sqrt(p_hat * (1 - p_hat) / trials
                           + z ** 2 / (4 * trials ** 2)) / denominator
    return centre - spread, centre + spread


def scenario_noise_model(scenario, p, num_qubits):
    """S1 = depolarizing only; S2 = depolarizing + thermal relaxation + readout."""
    if scenario == "S1":
        return build_noise_model(num_qubits, p_1q=p / 10, p_2q=p)
    if scenario == "S2":
        return build_noise_model(num_qubits, p_1q=p / 10, p_2q=p,
                                 readout_error=READOUT_ERROR, **THERMAL)
    raise ValueError(f"unknown scenario {scenario!r}")


def thresholds_for(L):
    """The thresholds t in {0, 1, floor(L/4)}, deduplicated but reported in fixed columns."""
    return {"t0": 0, "t1": 1, "tq": L // 4}


def cell_metrics(counts, L, m, trials):
    """P(correct), Wilson CI, mean absolute error, P(out of range) and FP/FN rates."""
    correct = counts.get(m, 0)
    p_correct = correct / trials
    low, high = wilson_interval(correct, trials)
    mae = sum(count * abs(value - m) for value, count in counts.items()) / trials
    out_of_range = sum(count for value, count in counts.items() if value > L) / trials

    row = {"p_correct": p_correct, "wilson_lo": low, "wilson_hi": high,
           "mae": mae, "p_out_of_range": out_of_range}
    for label, t in thresholds_for(L).items():
        # Only one of FP/FN is defined per cell, because m is fixed within a cell.
        if m <= t:
            above = sum(c for v, c in counts.items() if v > t) / trials
            row[f"fn_{label}"], row[f"fp_{label}"] = above, float("nan")
        else:
            below = sum(c for v, c in counts.items() if v <= t) / trials
            row[f"fn_{label}"], row[f"fp_{label}"] = float("nan"), below
    row["t_quarter"] = L // 4
    return row


def run_cell(scenario, L, p, m, pairs_per_m, shots, rng, noise_model, verify_coverage=False):
    """Pool `pairs_per_m` circuits x `shots` shots into one outcome histogram."""
    counts = {}
    k = None
    for index in range(pairs_per_m):
        read, window = sample_stratified_pair(L, m, rng)
        qc, k = mismatch_circuit_measured(read, window)
        transpiled = transpile_for_noise(qc)
        if verify_coverage:
            assert_full_coverage(noise_model, transpiled,
                                 check_measure=(scenario == "S2"))
        distribution = control_distribution_aer(
            transpiled, shots, noise_model=noise_model,
            seed=SEED_SIM + 1000 * m + index)
        for value, probability in distribution.items():
            counts[value] = counts.get(value, 0) + round(probability * shots)
    return counts, k


def circuit_stats(L, seed=SEED_PAIRS):
    """Qubit count, transpiled depth and CX count for a representative circuit."""
    rng = random.Random(seed)
    read, window = sample_stratified_pair(L, L // 2, rng)
    qc, k = mismatch_circuit_measured(read, window)
    transpiled = transpile_for_noise(qc)
    return {"k": k, "num_qubits": qc.num_qubits, "depth": transpiled.depth(),
            "cx": transpiled.count_ops().get("cx", 0)}


def time_one_circuit(L, shots, scenario, p=0.01):
    """Wall-clock breakdown for a single circuit: build, transpile, simulate."""
    rng = random.Random(SEED_PAIRS)
    read, window = sample_stratified_pair(L, L // 2, rng)

    start = time.perf_counter()
    qc, k = mismatch_circuit_measured(read, window)
    build_s = time.perf_counter() - start

    start = time.perf_counter()
    transpiled = transpile_for_noise(qc)
    transpile_s = time.perf_counter() - start

    start = time.perf_counter()
    noise_model = scenario_noise_model(scenario, p, qc.num_qubits)
    model_s = time.perf_counter() - start

    start = time.perf_counter()
    control_distribution_aer(transpiled, shots, noise_model=noise_model, seed=SEED_SIM)
    run_s = time.perf_counter() - start

    return {"L": L, "scenario": scenario, "shots": shots, "k": k,
            "num_qubits": qc.num_qubits, "depth": transpiled.depth(),
            "cx": transpiled.count_ops().get("cx", 0), "build_s": build_s,
            "transpile_s": transpile_s, "model_s": model_s, "run_s": run_s,
            "per_shot_ms": 1000 * run_s / shots}


def estimate_grid_seconds(timings, pairs_per_m, shots, levels):
    """Extrapolate full-grid runtime per scenario (overhead + shots x per-shot)."""
    total = 0.0
    for L in levels:
        cells_per_scenario = (L + 1) * len(P_GRID)
        for scenario in ("S1", "S2"):
            matching = [t for t in timings if t["L"] == L and t["scenario"] == scenario]
            if not matching:
                continue
            per_shot = max(t["per_shot_ms"] for t in matching) / 1000
            overhead = max(t["build_s"] + t["transpile_s"] for t in matching)
            total += cells_per_scenario * pairs_per_m * (overhead + shots * per_shot)
    return total


def git_commit_hash():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_sweep(levels, pairs_per_m, shots, results_dir):
    """Full grid over scenarios x p x L x m; writes CSV and metadata JSON."""
    import csv

    import qiskit
    import qiskit_aer

    started = time.perf_counter()
    stats = {L: circuit_stats(L) for L in levels}
    rows = []

    for scenario in ("S1", "S2"):
        for L in levels:
            num_qubits = stats[L]["num_qubits"]
            for p in P_GRID:
                noise_model = scenario_noise_model(scenario, p, num_qubits)
                for m in range(L + 1):
                    rng = random.Random(SEED_PAIRS + 7919 * L + 104729 * m)
                    cell_started = time.perf_counter()
                    counts, k = run_cell(scenario, L, p, m, pairs_per_m, shots,
                                         rng, noise_model,
                                         verify_coverage=(m == 0))
                    trials = pairs_per_m * shots
                    metrics = cell_metrics(counts, L, m, trials)
                    if scenario == "S1" and p == 0.0 and metrics["p_correct"] != 1.0:
                        raise RuntimeError(
                            f"noiseless S1 cell L={L} m={m} gave "
                            f"P(correct)={metrics['p_correct']!r}, expected exactly 1.0")
                    rows.append({"scenario": scenario, "L": L, "p": p, "m": m,
                                 "pairs_per_m": pairs_per_m, "shots": shots,
                                 "n_trials": trials, **metrics, **stats[L],
                                 "seed_pairs": SEED_PAIRS, "seed_sim": SEED_SIM,
                                 "elapsed_s": time.perf_counter() - cell_started})
                    print(f"  {scenario} L={L} p={p:<6} m={m}  "
                          f"P(correct)={metrics['p_correct']:.4f}  "
                          f"MAE={metrics['mae']:.3f}  "
                          f"({time.perf_counter() - cell_started:.1f}s)", flush=True)

    total_seconds = time.perf_counter() - started
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    csv_path = results_dir / f"noise_degradation_{stamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    meta = {"date": stamp, "git_commit": git_commit_hash(),
            "qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__,
            "seed_pairs": SEED_PAIRS, "seed_sim": SEED_SIM, "shots": shots,
            "pairs_per_m": pairs_per_m, "levels": list(levels),
            "p_grid": list(P_GRID), "scenarios": {"S1": "depolarizing only",
                                                  "S2": "depolarizing + thermal + readout"},
            "thermal": THERMAL, "readout_error": READOUT_ERROR,
            "circuit_stats": {str(L): stats[L] for L in levels},
            "total_runtime_s": total_seconds, "rows": len(rows)}
    meta_path = results_dir / f"noise_degradation_{stamp}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nwrote {csv_path} ({len(rows)} rows) and {meta_path} "
          f"in {total_seconds / 60:.1f} min")
    return csv_path, meta_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true",
                        help="time single circuits and estimate the grid runtime")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--pairs-per-m", type=int, default=8)
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--results-dir", default=str(
        Path(__file__).resolve().parents[2] / "results"))
    args = parser.parse_args()

    if args.benchmark:
        timings = []
        for L in args.levels:
            for scenario in ("S1", "S2"):
                timing = time_one_circuit(L, args.shots, scenario)
                timings.append(timing)
                print(f"L={timing['L']} {timing['scenario']}  "
                      f"qubits={timing['num_qubits']:<3} depth={timing['depth']:<5} "
                      f"cx={timing['cx']:<5} transpile={timing['transpile_s']:.2f}s "
                      f"run({args.shots} shots)={timing['run_s']:.2f}s "
                      f"per_shot={timing['per_shot_ms']:.3f}ms", flush=True)
        estimate = estimate_grid_seconds(timings, args.pairs_per_m, args.shots, args.levels)
        print(f"\nestimated full grid (levels={args.levels}, "
              f"pairs_per_m={args.pairs_per_m}, shots={args.shots}): "
              f"{estimate / 60:.1f} min")
        return

    run_sweep(args.levels, args.pairs_per_m, args.shots, args.results_dir)


if __name__ == "__main__":
    main()
