"""Task 3, first piece: flag post-selection, analyzed from data already collected.

An outcome > L is impossible noiselessly (see "Reading p_out_of_range" in
results/README.md -- this is a bounded, cheap consistency signal, not
evidence of a discovery). Discarding those shots is free error detection:
every CORRECT shot is already in-range (P(correct AND in-range) = P(correct)),
so

    P(correct | in-range) = p_correct / (1 - p_out_of_range)

is computable directly from columns already in noise_degradation_*.csv and
counter_comparison_*.csv -- no new circuit runs needed. This script reports
that recovery and its cost (the fraction of shots discarded), it does not
re-simulate anything.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def with_post_selection(row):
    """Add p_correct_post_select and discard_rate to one CSV row (as floats)."""
    p_correct = float(row["p_correct"])
    p_oor = float(row["p_out_of_range"])
    kept = 1 - p_oor
    post_selected = p_correct / kept if kept > 0 else float("nan")
    return post_selected, p_oor


def summarize(csv_path, group_keys=("scenario", "L", "p")):
    rows = list(csv.DictReader(Path(csv_path).open(encoding="utf-8")))
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        post, discard = with_post_selection(row)
        groups.setdefault(key, {"p_correct": [], "post": [], "discard": []})
        groups[key]["p_correct"].append(float(row["p_correct"]))
        groups[key]["post"].append(post)
        groups[key]["discard"].append(discard)
    return {key: {"p_correct": sum(v["p_correct"]) / len(v["p_correct"]),
                 "post_select": sum(v["post"]) / len(v["post"]),
                 "discard_rate": sum(v["discard"]) / len(v["discard"])}
           for key, v in groups.items()}


def print_report(csv_path, levels, scenario="S2"):
    summary = summarize(csv_path)
    print(f"Post-selection recovery, {scenario}, mean over m (from {csv_path})\n")
    header = f"{'L':>3} {'p':>7} | {'raw':>7} {'post-select':>12} {'gain':>7} {'discard %':>10}"
    print(header)
    print("-" * len(header))
    p_grid = sorted({float(p) for (s, L, p) in summary if s == scenario})
    for L in levels:
        for p in p_grid:
            key = (scenario, str(L), str(p) if p != int(p) else str(p))
            # p may be stored as "0.0" or "0" etc; find matching key robustly.
            match = next((k for k in summary
                         if k[0] == scenario and k[1] == str(L) and float(k[2]) == p), None)
            if match is None:
                continue
            row = summary[match]
            gain = row["post_select"] - row["p_correct"]
            print(f"{L:>3} {p:>7} | {row['p_correct']:>7.3f} {row['post_select']:>12.3f} "
                  f"{gain:>+7.3f} {row['discard_rate'] * 100:>9.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--scenario", default="S2")
    args = parser.parse_args()
    print_report(args.csv_path, args.levels, args.scenario)


if __name__ == "__main__":
    main()
