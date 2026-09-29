import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "experiments"))

from supervised_run import meta_path_for, run_supervised


def _fake_subprocess(succeed_on_attempt, meta_path, side_effect_calls):
    """Returns a run_subprocess stand-in that writes meta_path (simulating a
    successful counter_comparison.py run) only on the Nth call, and otherwise
    returns as if the process died without writing it -- standing in for a
    segfault, since real segfaults are what --resume exists to survive."""

    def fake_run(args, cwd, env):
        side_effect_calls.append(args)
        if len(side_effect_calls) >= succeed_on_attempt:
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text("{}", encoding="utf-8")
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=139)  # segfault-like exit code

    return fake_run


def test_retries_until_meta_json_appears_then_stops(tmp_path):
    meta_path = meta_path_for(tmp_path, "sometag")
    calls = []
    ok = run_supervised(["--tag", "sometag"], results_dir=tmp_path, tag="sometag",
                        max_attempts=10, repo_root=tmp_path,
                        run_subprocess=_fake_subprocess(3, meta_path, calls))
    assert ok is True
    assert len(calls) == 3
    # every invocation (including the first) carries --resume -- safe even
    # from scratch, and required for attempts after the first to not restart.
    assert all("--resume" in call for call in calls)


def test_gives_up_after_max_attempts_if_meta_json_never_appears(tmp_path):
    meta_path = meta_path_for(tmp_path, "nevertag")
    calls = []
    ok = run_supervised(["--tag", "nevertag"], results_dir=tmp_path, tag="nevertag",
                        max_attempts=4, repo_root=tmp_path,
                        run_subprocess=_fake_subprocess(99, meta_path, calls))
    assert ok is False
    assert len(calls) == 4
    assert not meta_path.exists()


def test_succeeds_immediately_if_first_attempt_writes_meta_json(tmp_path):
    meta_path = meta_path_for(tmp_path, "firsttry")
    calls = []
    ok = run_supervised(["--tag", "firsttry"], results_dir=tmp_path, tag="firsttry",
                        max_attempts=10, repo_root=tmp_path,
                        run_subprocess=_fake_subprocess(1, meta_path, calls))
    assert ok is True
    assert len(calls) == 1
