"""Outer retry loop for counter_comparison.py.

A segfault kills the whole process and cannot be caught by any in-process
try/except -- see counter_comparison.run_comparison's `resume` docstring for
why --resume exists. This script is the other half of that fix: it invokes
counter_comparison.py as a SEPARATE subprocess, and if that process dies
(any exit, clean or otherwise) without having written meta.json, relaunches
it with --resume so the next attempt continues from the last completed cell
instead of restarting the grid.

Success is judged only by meta.json's existence, never by the subprocess's
exit code. counter_comparison.py's own main() generates a matplotlib figure
AFTER run_comparison() returns and meta.json is written; a crash during
figure generation would give a nonzero exit code despite the actual sweep
data being complete and trustworthy, which would be a false negative for
the part that matters. A crash BEFORE meta.json exists is a true failure
and triggers a retry regardless of exit code, since a segfault often exits
with no distinguishing code at all.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def meta_path_for(results_dir, tag):
    stamp = date.today().isoformat()
    suffix = f"_{tag}" if tag else ""
    return Path(results_dir) / f"counter_comparison_{stamp}{suffix}_meta.json"


def run_supervised(extra_args, results_dir, tag, max_attempts=10,
                   python=sys.executable, repo_root=REPO_ROOT,
                   run_subprocess=subprocess.run):
    """Run counter_comparison.py with `extra_args`, retrying (with --resume)
    up to `max_attempts` times until meta_path_for(results_dir, tag) exists.

    `run_subprocess` is injectable so tests can simulate a crashing/succeeding
    subprocess without actually shelling out to Aer.
    """
    meta_path = meta_path_for(results_dir, tag)
    script = Path(repo_root) / "src" / "experiments" / "counter_comparison.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(repo_root) / "src")

    for attempt in range(1, max_attempts + 1):
        # --resume is safe even on a from-scratch attempt 1 (no existing CSV):
        # run_comparison treats "resume requested but nothing to resume" as an
        # ordinary fresh run -- see test_resume_with_no_existing_file_behaves_like_a_fresh_run.
        args = [python, str(script), *extra_args, "--resume"]
        print(f"\n=== attempt {attempt}/{max_attempts}: {' '.join(args)} ===", flush=True)
        started = time.time()
        proc = run_subprocess(args, cwd=str(repo_root), env=env)
        elapsed = time.time() - started
        print(f"=== attempt {attempt} exited with code {proc.returncode} "
              f"after {elapsed:.1f}s ===", flush=True)

        if meta_path.exists():
            print(f"success: {meta_path} exists")
            return True

        print(f"meta.json not yet written after attempt {attempt}"
              + (" -- retrying" if attempt < max_attempts else ""), flush=True)

    print(f"gave up after {max_attempts} attempts; {meta_path} still missing", flush=True)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--pairs-per-m", type=int, default=8)
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--figures-dir", default=str(REPO_ROOT / "figures"))
    parser.add_argument("--tag", default=None)
    parser.add_argument("--connectivity-aware", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=10)
    args = parser.parse_args()

    extra = ["--shots", str(args.shots), "--pairs-per-m", str(args.pairs_per_m),
             "--levels", *map(str, args.levels),
             "--results-dir", args.results_dir, "--figures-dir", args.figures_dir]
    if args.tag:
        extra += ["--tag", args.tag]
    if args.connectivity_aware:
        extra.append("--connectivity-aware")

    ok = run_supervised(extra, args.results_dir, args.tag, max_attempts=args.max_attempts)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
