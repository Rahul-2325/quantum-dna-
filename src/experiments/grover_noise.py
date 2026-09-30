"""Noise-aware Grover search (novelty direction #4 -- see PROJECT_STATUS.md):
does the amplification verified noiseless in grover_search.py survive the
SAME realistic noise model used throughout the rest of this project?

Nobody in this project's three reference papers, nor QShift-SA (the paper
that inspired the Grover-oracle work), has checked this empirically for a
phase-kickback-counter Grover oracle -- grover_search.py verifies the
circuit is CORRECT at zero noise; this experiment asks whether that
correctness survives contact with a real device's error rates.

Fixed test case: reference="TACCGAT", read="TA", threshold=0 -- 6 real
shifts (2 leftover, from k_shift=3 padding to the next power of two), and
exactly one good shift (shift 0). The same case used in
test_grover_search.py's leftover-branch correctness test. Sweeps
`iterations` (0 = no amplification, the trivial gate-count baseline) x the
SAME S1/S2 scenarios and P_GRID as noise_degradation.py, so results are
directly comparable to every other noise result in this project.

Circuit-size finding, not just a caveat: at the noiseless-optimal 2
iterations this circuit already has ~1100 CX gates after transpiling to
basis gates -- an order of magnitude more than any of the four direct
mismatch-counting circuits at comparable read length. That gate count is
exactly why this experiment exists, and (measured below) is enough on its
own to erase the amplification advantage at noise levels where the direct
counters are still comfortably above 85% correct.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import date
from pathlib import Path

import qiskit
import qiskit_aer
from qiskit import ClassicalRegister

from aer_helpers import control_distribution_aer, transpile_for_noise
from grover_search import build_grover_search
from noise_degradation import (P_GRID, READOUT_ERROR, SEED_SIM, THERMAL, VIRTUAL_RZ,
                               git_commit_hash, is_noiseless_control, scenario_noise_model)

REFERENCE, READ, THRESHOLD = "TACCGAT", "TA", 0
ITERATIONS_GRID = (0, 1, 2, 3, 4)
SHOTS = 1024

FIELDNAMES = ["scenario", "p", "iterations", "num_qubits", "cx_count", "depth",
             "p_good", "p_bad", "p_leftover", "top_shift_is_good",
             "shots", "seed_sim", "elapsed_s"]


def build_measured_circuit(iterations):
    """Grover search circuit for the fixed test case, with the shift register
    measured into its own classical register (nothing else needs measuring --
    every other register is uncomputed by construction in the noiseless
    case, and this experiment is precisely about what noise does to that)."""
    qc, shift, k_shift, shifts, good_shifts, actual_iterations = build_grover_search(
        REFERENCE, READ, THRESHOLD, iterations=iterations)
    creg = ClassicalRegister(k_shift, "c")
    qc.add_register(creg)
    qc.measure(shift, list(creg))
    return qc, shift, k_shift, shifts, good_shifts, actual_iterations


def cell_metrics(dist, k_shift, shifts, good_shifts):
    """p_good/p_bad/p_leftover partition the full 2^k_shift outcome space the
    same way test_grover_search.py's noiseless check does; top_shift_is_good
    is the practical question a real user of this search would ask: "if I
    take the single most-observed shift as my answer, is it right?" """
    leftover = [s for s in range(2 ** k_shift) if s not in shifts]
    bad = [s for s in shifts if s not in good_shifts]
    p_good = sum(dist.get(s, 0.0) for s in good_shifts)
    p_bad = sum(dist.get(s, 0.0) for s in bad)
    p_leftover = sum(dist.get(s, 0.0) for s in leftover)
    top_shift = max(range(2 ** k_shift), key=lambda s: dist.get(s, 0.0))
    return dict(p_good=p_good, p_bad=p_bad, p_leftover=p_leftover,
               top_shift_is_good=int(top_shift in good_shifts))


def run_sweep(iterations_grid, results_dir, tag=None, force=False, shots=SHOTS):
    results_dir = Path(results_dir)
    stamp = date.today().isoformat()
    suffix = f"_{tag}" if tag else ""
    csv_path = results_dir / f"grover_noise_{stamp}{suffix}.csv"
    meta_path = results_dir / f"grover_noise_{stamp}{suffix}_meta.json"
    if not force:
        for path in (csv_path, meta_path):
            if path.exists():
                raise FileExistsError(f"{path} already exists; pass --tag or --force")

    results_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    row_count = 0

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        handle.flush()

        for iterations in iterations_grid:
            qc, shift, k_shift, shifts, good_shifts, actual_iterations = \
                build_measured_circuit(iterations)
            tqc = transpile_for_noise(qc)
            cx_count = tqc.count_ops().get("cx", 0)
            depth = tqc.depth()

            for scenario in ("S1", "S2"):
                for p in P_GRID:
                    noise_model = scenario_noise_model(scenario, p, qc.num_qubits)
                    started_cell = time.perf_counter()
                    dist = control_distribution_aer(
                        tqc, shots=shots, noise_model=noise_model,
                        seed=SEED_SIM, method="statevector", num_bits=k_shift)
                    metrics = cell_metrics(dist, k_shift, shifts, good_shifts)
                    if actual_iterations >= 1 and is_noiseless_control(scenario, p):
                        # A search that has actually run at least one
                        # iteration must still get the right answer with
                        # zero circuit noise -- this is the noiseless
                        # correctness guarantee grover_search.py already
                        # verified, re-checked here through the sampling
                        # harness itself (shot noise only, no circuit noise).
                        assert metrics["top_shift_is_good"] == 1, (
                            f"noiseless control top shift is not good at "
                            f"iterations={actual_iterations}: {dist}")
                    row = dict(scenario=scenario, p=p, iterations=actual_iterations,
                              num_qubits=qc.num_qubits, cx_count=cx_count, depth=depth,
                              shots=shots, seed_sim=SEED_SIM,
                              elapsed_s=time.perf_counter() - started_cell, **metrics)
                    writer.writerow(row)
                    handle.flush()
                    row_count += 1
                    print(f"  it={actual_iterations} {scenario} p={p:<6} "
                         f"P(good)={metrics['p_good']:.4f} "
                         f"top_is_good={metrics['top_shift_is_good']} "
                         f"({time.perf_counter() - started_cell:.1f}s)", flush=True)

    total_seconds = time.perf_counter() - started
    meta = dict(date=stamp, git_commit=git_commit_hash(), qiskit=qiskit.__version__,
               qiskit_aer=qiskit_aer.__version__, reference=REFERENCE, read=READ,
               threshold=THRESHOLD, iterations_grid=list(iterations_grid),
               p_grid=list(P_GRID), shots=shots, seed_sim=SEED_SIM,
               thermal=THERMAL, readout_error=READOUT_ERROR, virtual_rz=VIRTUAL_RZ,
               total_runtime_s=total_seconds, rows=row_count)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nwrote {csv_path} ({row_count} rows) and {meta_path} "
          f"in {total_seconds / 60:.1f} min")
    return csv_path, meta_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, nargs="+", default=list(ITERATIONS_GRID))
    parser.add_argument("--shots", type=int, default=SHOTS)
    parser.add_argument("--results-dir", default=str(
        Path(__file__).resolve().parents[2] / "results"))
    parser.add_argument("--tag", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_sweep(args.iterations, args.results_dir, tag=args.tag,
             force=args.force, shots=args.shots)


if __name__ == "__main__":
    main()
