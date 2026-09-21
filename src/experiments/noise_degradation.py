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

import numpy as np

from aer_helpers import (control_distribution_aer, control_distributions_aer,
                         mismatch_circuit_measured, transpile_for_noise)
from noise_models import (VIRTUAL_GATES, assert_full_coverage, build_noise_model,
                          noisy_one_qubit_gates)

BASES = "ACGT"
P_GRID = (0.0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05)
THERMAL = dict(t1=100_000.0, t2=80_000.0, gate_time_1q=50.0, gate_time_2q=300.0)
READOUT_ERROR = 0.01
SEED_PAIRS = 20260920
SEED_SIM = 761
SEED_BOOTSTRAP = 90210
BOOTSTRAP_RESAMPLES = 10_000
Z_95 = 1.959963984540054

# rz is a zero-duration frame change on IBM hardware, so it carries no noise.
VIRTUAL_RZ = True
COVERAGE_EXEMPT = VIRTUAL_GATES if VIRTUAL_RZ else ()


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


def is_noiseless_control(scenario, p):
    """S1 at p=0 is the exactness control: no depolarizing, no thermal, no readout."""
    return scenario == "S1" and p == 0.0


def scenario_noise_model(scenario, p, num_qubits, virtual_rz=VIRTUAL_RZ):
    """S1 = depolarizing only; S2 = depolarizing + thermal relaxation + readout."""
    if scenario == "S1":
        return build_noise_model(num_qubits, p_1q=p / 10, p_2q=p,
                                 virtual_rz=virtual_rz)
    if scenario == "S2":
        return build_noise_model(num_qubits, p_1q=p / 10, p_2q=p,
                                 readout_error=READOUT_ERROR,
                                 virtual_rz=virtual_rz, **THERMAL)
    raise ValueError(f"unknown scenario {scenario!r}")


def thresholds_for(L):
    """The thresholds t in {0, 1, floor(L/4)}, deduplicated but reported in fixed columns."""
    return {"t0": 0, "t1": 1, "tq": L // 4}


def bootstrap_mean_interval(values, resamples=BOOTSTRAP_RESAMPLES,
                            seed=SEED_BOOTSTRAP, alpha=0.05):
    """Percentile bootstrap CI for the mean, resampling the independent units.

    Pairs, not shots, are the independent units here: shots within one pair
    share a circuit, so a pooled binomial interval understates uncertainty
    across the population of reads.
    """
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[draws].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def pair_statistics(per_pair_counts, m, shots, seed=SEED_BOOTSTRAP):
    """Bootstrap CI and spread of P(correct) across pairs, the independent units."""
    per_pair = [counts.get(m, 0) / shots for counts in per_pair_counts]
    low, high = bootstrap_mean_interval(per_pair, seed=seed)
    spread = float(np.std(per_pair, ddof=1)) if len(per_pair) > 1 else float("nan")
    return {"boot_lo": low, "boot_hi": high, "pair_sd": spread,
            "pairs_used": len(per_pair)}


def cell_metrics(counts, L, m, trials):
    """P(correct), Wilson CI, error metrics, P(out of range) and FP/FN rates."""
    correct = counts.get(m, 0)
    p_correct = correct / trials
    low, high = wilson_interval(correct, trials)
    mae = sum(count * abs(value - m) for value, count in counts.items()) / trials
    # Signed error shows DIRECTION: amplitude damping pulls the register toward
    # |0>, which would under-report the count and make this negative.
    bias = sum(count * (value - m) for value, count in counts.items()) / trials
    out_of_range = sum(count for value, count in counts.items() if value > L) / trials

    row = {"p_correct": p_correct, "wilson_lo": low, "wilson_hi": high,
           "mae": mae, "bias": bias, "p_out_of_range": out_of_range}
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


def run_cell(scenario, L, p, m, pairs_per_m, shots, rng, noise_model, verify_coverage=False,
             circuit_builder=mismatch_circuit_measured):
    """Run `pairs_per_m` circuits x `shots` shots.

    `circuit_builder(read, window) -> (qc, k)` selects the counter under test;
    defaults to the phase counter (mismatch_circuit_measured) so existing
    callers are unaffected. Pass mismatch_circuit_adder_measured to sweep the
    Paper-1 adder counter instead -- same sampling, metrics and noise model.

    Returns the pooled histogram, the per-pair histograms (kept because pairs
    are the independent unit for the bootstrap) and the control width k.
    """
    circuits = []
    k = None
    for _ in range(pairs_per_m):
        read, window = sample_stratified_pair(L, m, rng)
        qc, k = circuit_builder(read, window)
        transpiled = transpile_for_noise(qc)
        if verify_coverage:
            assert_full_coverage(noise_model, transpiled,
                                 check_measure=(scenario == "S2"),
                                 exempt=COVERAGE_EXEMPT)
        circuits.append(transpiled)

    distributions = control_distributions_aer(
        circuits, shots, noise_model=noise_model, seed=SEED_SIM + 7919 * L + 1000 * m,
        num_bits=k)

    per_pair_counts = [{value: round(probability * shots)
                        for value, probability in distribution.items()}
                       for distribution in distributions]
    counts = {}
    for pair_counts in per_pair_counts:
        for value, count in pair_counts.items():
            counts[value] = counts.get(value, 0) + count
    return counts, per_pair_counts, k


def circuit_stats(L, seed=SEED_PAIRS, circuit_builder=mismatch_circuit_measured):
    """Qubit count, transpiled depth and CX count for a representative circuit."""
    rng = random.Random(seed)
    read, window = sample_stratified_pair(L, L // 2, rng)
    qc, k = circuit_builder(read, window)
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


def noise_audit(levels=(2, 4, 6, 8)):
    """A1: transpiled gate mix per L, and exactly how each gate type is treated."""
    print("Transpiled gate counts (basis cx/rz/sx/x, all-to-all connectivity)\n")
    header = f"{'L':>3} {'qubits':>7} {'depth':>6} {'cx':>5} {'rz':>5} {'sx':>4} {'x':>4} {'rz share':>9}"
    print(header)
    print("-" * len(header))
    for L in levels:
        rng = random.Random(SEED_PAIRS)
        read, window = sample_stratified_pair(L, L // 2, rng)
        qc, _ = mismatch_circuit_measured(read, window)
        transpiled = transpile_for_noise(qc)
        ops = transpiled.count_ops()
        gates = {g: ops.get(g, 0) for g in ("cx", "rz", "sx", "x")}
        total = sum(gates.values())
        print(f"{L:>3} {qc.num_qubits:>7} {transpiled.depth():>6} {gates['cx']:>5} "
              f"{gates['rz']:>5} {gates['sx']:>4} {gates['x']:>4} "
              f"{gates['rz'] / total:>8.1%}")

    noisy = noisy_one_qubit_gates(VIRTUAL_RZ)
    print(f"""
How build_noise_model treats each gate type (VIRTUAL_RZ={VIRTUAL_RZ}):

  cx      depolarizing(p_2q, 2 qubits)
          composed with thermal_relaxation(T1={THERMAL['t1']:.0f}ns, T2={THERMAL['t2']:.0f}ns,
          duration={THERMAL['gate_time_2q']:.0f}ns) on EACH of the two qubits   [S2 only]

  sx, x   depolarizing(p_1q, 1 qubit)
          composed with thermal_relaxation(T1, T2, duration={THERMAL['gate_time_1q']:.0f}ns)  [S2 only]

  rz      {'NO depolarizing, NO thermal relaxation (virtual gate: a zero-duration'
           if VIRTUAL_RZ else 'treated exactly like sx and x (legacy pessimistic mode)'}
          {'frame change in software on IBM hardware, so it costs no time and no fidelity)'
           if VIRTUAL_RZ else ''}

  measure symmetric ReadoutError({READOUT_ERROR})                              [S2 only]

  Noisy one-qubit gates: {', '.join(noisy) if noisy else 'none'}
  p_1q = p/10 and p_2q = p, where p is the swept two-qubit error rate.

Why this matters: rz is ~49% of every circuit. Charging it a {THERMAL['gate_time_1q']:.0f}ns duration
added fictitious decay (170 x 50ns = 8.5us at L=8) on top of a circuit whose
real cost is dominated by cx ({THERMAL['gate_time_2q']:.0f}ns each).
""")


def git_commit_hash():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


CSV_FIELDNAMES = (
    ["scenario", "L", "p", "m", "pairs_per_m", "shots", "n_trials",
     "p_correct", "wilson_lo", "wilson_hi", "mae", "bias", "p_out_of_range",
     "fn_t0", "fp_t0", "fn_t1", "fp_t1", "fn_tq", "fp_tq", "t_quarter",
     "boot_lo", "boot_hi", "pair_sd", "pairs_used",
     "k", "num_qubits", "depth", "cx", "seed_pairs", "seed_sim", "elapsed_s"])


def run_sweep(levels, pairs_per_m, shots, results_dir, tag=None, force=False):
    """Full grid over scenarios x p x L x m; writes CSV incrementally, metadata at the end.

    Aer's C++ backend can raise a hard MemoryError partway through a long grid
    (hit twice on the L=8 run at ~28 qubits/S2). Writing and flushing each row
    as it is produced, rather than batching everything into memory and writing
    once at the end, means a crash loses at most the in-flight cell instead of
    every row computed so far. `meta.json` is written only on a full, successful
    completion, so its absence next to a CSV is the signal that a run is partial.
    """
    import csv

    import qiskit
    import qiskit_aer

    # Name the outputs before doing 20 minutes of work, so a name collision
    # fails now rather than silently overwriting a previous run's data.
    results_dir = Path(results_dir)
    stamp = date.today().isoformat()
    suffix = f"_{tag}" if tag else ""
    csv_path = results_dir / f"noise_degradation_{stamp}{suffix}.csv"
    meta_path = results_dir / f"noise_degradation_{stamp}{suffix}_meta.json"
    if not force:
        for path in (csv_path, meta_path):
            if path.exists():
                raise FileExistsError(
                    f"{path} already exists; pass --tag to name this run separately "
                    f"(or --force to overwrite)")

    started = time.perf_counter()
    stats = {L: circuit_stats(L) for L in levels}
    results_dir.mkdir(parents=True, exist_ok=True)

    row_count = 0
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        handle.flush()

        for scenario in ("S1", "S2"):
            for L in levels:
                num_qubits = stats[L]["num_qubits"]
                for p in P_GRID:
                    noise_model = scenario_noise_model(scenario, p, num_qubits)
                    # S1 at p=0 is the noiseless control: depolarizing_error(0) is
                    # the identity channel, which Aer drops, so the model is empty
                    # by design and there is no coverage to check.
                    noiseless_control = is_noiseless_control(scenario, p)
                    if noiseless_control and noise_model.to_dict()["errors"]:
                        raise RuntimeError(
                            f"{scenario} p={p} was expected to be noiseless but carries errors")
                    for m in range(L + 1):
                        rng = random.Random(SEED_PAIRS + 7919 * L + 104729 * m)
                        cell_started = time.perf_counter()
                        counts, per_pair, k = run_cell(
                            scenario, L, p, m, pairs_per_m, shots, rng, noise_model,
                            verify_coverage=(m == 0 and not noiseless_control))
                        trials = pairs_per_m * shots
                        metrics = cell_metrics(counts, L, m, trials)
                        metrics.update(pair_statistics(per_pair, m, shots))
                        if scenario == "S1" and p == 0.0 and metrics["p_correct"] != 1.0:
                            raise RuntimeError(
                                f"noiseless S1 cell L={L} m={m} gave "
                                f"P(correct)={metrics['p_correct']!r}, expected exactly 1.0")
                        row = {"scenario": scenario, "L": L, "p": p, "m": m,
                               "pairs_per_m": pairs_per_m, "shots": shots,
                               "n_trials": trials, **metrics, **stats[L],
                               "seed_pairs": SEED_PAIRS, "seed_sim": SEED_SIM,
                               "elapsed_s": time.perf_counter() - cell_started}
                        writer.writerow(row)
                        handle.flush()
                        row_count += 1
                        print(f"  {scenario} L={L} p={p:<6} m={m}  "
                              f"P(correct)={metrics['p_correct']:.4f}  "
                              f"MAE={metrics['mae']:.3f}  "
                              f"({time.perf_counter() - cell_started:.1f}s)", flush=True)

    total_seconds = time.perf_counter() - started
    meta = {"date": stamp, "git_commit": git_commit_hash(),
            "qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__,
            "seed_pairs": SEED_PAIRS, "seed_sim": SEED_SIM, "shots": shots,
            "pairs_per_m": pairs_per_m, "levels": list(levels),
            "p_grid": list(P_GRID), "scenarios": {"S1": "depolarizing only",
                                                  "S2": "depolarizing + thermal + readout"},
            "thermal": THERMAL, "readout_error": READOUT_ERROR,
            "virtual_rz": VIRTUAL_RZ,
            "noisy_one_qubit_gates": list(noisy_one_qubit_gates(VIRTUAL_RZ)),
            "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": SEED_BOOTSTRAP,
                          "unit": "pair"},
            "circuit_stats": {str(L): stats[L] for L in levels},
            "total_runtime_s": total_seconds, "rows": row_count}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nwrote {csv_path} ({row_count} rows) and {meta_path} "
          f"in {total_seconds / 60:.1f} min")
    return csv_path, meta_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true",
                        help="time single circuits and estimate the grid runtime")
    parser.add_argument("--audit", action="store_true",
                        help="print the gate mix and how each gate type is made noisy")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--pairs-per-m", type=int, default=8)
    parser.add_argument("--levels", type=int, nargs="+", default=None)
    parser.add_argument("--results-dir", default=str(
        Path(__file__).resolve().parents[2] / "results"))
    parser.add_argument("--tag", default=None,
                        help="suffix for the output filenames, e.g. L8")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing output files")
    args = parser.parse_args()

    if args.audit:
        noise_audit(args.levels or (2, 4, 6, 8))
        return

    levels = args.levels or [2, 4, 6]
    if args.benchmark:
        timings = []
        for L in levels:
            for scenario in ("S1", "S2"):
                timing = time_one_circuit(L, args.shots, scenario)
                timings.append(timing)
                print(f"L={timing['L']} {timing['scenario']}  "
                      f"qubits={timing['num_qubits']:<3} depth={timing['depth']:<5} "
                      f"cx={timing['cx']:<5} transpile={timing['transpile_s']:.2f}s "
                      f"run({args.shots} shots)={timing['run_s']:.2f}s "
                      f"per_shot={timing['per_shot_ms']:.3f}ms", flush=True)
        estimate = estimate_grid_seconds(timings, args.pairs_per_m, args.shots, levels)
        print(f"\nestimated full grid (levels={levels}, "
              f"pairs_per_m={args.pairs_per_m}, shots={args.shots}): "
              f"{estimate / 60:.1f} min")
        return

    run_sweep(levels, args.pairs_per_m, args.shots, args.results_dir,
              tag=args.tag, force=args.force)


if __name__ == "__main__":
    main()
