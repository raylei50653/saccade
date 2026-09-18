"""Fail-closed contracts for tools/resctl.py, the cross-worktree resource lease CLI.

Every test builds a throwaway repository with two linked worktrees and drives
the real CLI as subprocesses, because the guarantees under test -- one holder
per resource, cross-resource conflicts, release on crash -- are properties of
kernel ``flock`` across *processes*, not of in-process bookkeeping.  The lease
JSON is deliberately corrupted or orphaned in places to pin the rule that
metadata can never make a resource look free.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
RESCTL = ROOT / "tools" / "resctl.py"

EXIT_LOCK_FAILED = 75


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("resctl", RESCTL)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["resctl"] = (
        mod  # dataclasses resolve string annotations via sys.modules
    )
    spec.loader.exec_module(mod)
    return mod


resctl = _load_module()


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> dict[str, Path]:
    """A repo with worktrees ``a`` (main) and ``b`` (branch ``feature``)."""
    a = tmp_path / "a"
    a.mkdir()
    _git("init", "-q", "-b", "main", cwd=a)
    _git("config", "user.email", "t@example.com", cwd=a)
    _git("config", "user.name", "t", cwd=a)
    (a / "README.md").write_text("x\n", encoding="utf-8")
    _git("add", "README.md", cwd=a)
    _git("commit", "-q", "-m", "init", cwd=a)
    b = tmp_path / "b"
    _git("worktree", "add", "-q", "-b", "feature", str(b), cwd=a)
    return {"a": a, "b": b, "common": a / ".git"}


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["RESCTL_NVIDIA_SMI"] = "/nonexistent/nvidia-smi"  # keep tests hardware-free
    return env


def _cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RESCTL), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=_env(),
        timeout=60,
    )


def _lease_path(repo: dict[str, Path], resource: str) -> Path:
    return repo["common"] / "worktree-runtime" / "leases" / f"{resource}.json"


def _hold(resource: str, cwd: Path, repo: dict[str, Path]) -> subprocess.Popen[str]:
    """Start ``resctl run RESOURCE -- sleep 60`` and wait until the lease is written."""
    proc = subprocess.Popen(
        [sys.executable, str(RESCTL), "run", "--quiet", resource, "--", "sleep", "60"],
        cwd=cwd,
        env=_env(),
        text=True,
    )
    lease = _lease_path(repo, resource)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if lease.exists() and proc.poll() is None:
            return proc
        if proc.poll() is not None:
            raise AssertionError(f"holder exited early with {proc.returncode}")
        time.sleep(0.05)
    raise AssertionError("holder never acquired the lease")


def _stop(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        proc.terminate()
    proc.wait(timeout=20)


def _status(cwd: Path) -> dict[str, object]:
    res = _cli("status", "--json", cwd=cwd)
    assert res.returncode == 0, res.stderr
    out: dict[str, object] = json.loads(res.stdout)
    return out


def _lease_states(cwd: Path) -> dict[str, dict[str, object]]:
    leases = _status(cwd)["leases"]
    assert isinstance(leases, list)
    return {row["resource"]: row for row in leases}


# --------------------------------------------------------------------------- #
# Locking and conflicts
# --------------------------------------------------------------------------- #


def test_two_worktrees_cannot_both_hold_gpu0(repo: dict[str, Path]) -> None:
    holder = _hold("gpu0", repo["a"], repo)
    try:
        res = _cli("run", "gpu0", "--", "true", cwd=repo["b"])
        assert res.returncode == EXIT_LOCK_FAILED
        # The refusal names the current owner: worktree, branch and PID.
        assert str(repo["a"]) in res.stderr
        assert "[main]" in res.stderr
        assert f"pid {holder.pid}" in res.stderr
    finally:
        _stop(holder)


@pytest.mark.parametrize("victim", ["gpu0", "cpu-heavy"])
def test_machine_bench_blocks_gpu_and_cpu(repo: dict[str, Path], victim: str) -> None:
    holder = _hold("machine-bench", repo["a"], repo)
    try:
        res = _cli("run", victim, "--", "true", cwd=repo["b"])
        assert res.returncode == EXIT_LOCK_FAILED
        assert "machine-bench is BUSY" in res.stderr
    finally:
        _stop(holder)


@pytest.mark.parametrize("active", ["gpu0", "cpu-heavy"])
def test_gpu_or_cpu_blocks_machine_bench(repo: dict[str, Path], active: str) -> None:
    holder = _hold(active, repo["a"], repo)
    try:
        res = _cli("run", "machine-bench", "--", "true", cwd=repo["b"])
        assert res.returncode == EXIT_LOCK_FAILED
        assert f"{active} is BUSY" in res.stderr
    finally:
        _stop(holder)


def test_gpu0_and_cpu_heavy_do_not_conflict(repo: dict[str, Path]) -> None:
    holder = _hold("gpu0", repo["a"], repo)
    try:
        res = _cli("run", "--quiet", "cpu-heavy", "--", "true", cwd=repo["b"])
        assert res.returncode == 0, res.stderr
    finally:
        _stop(holder)


def test_lock_released_after_normal_completion(repo: dict[str, Path]) -> None:
    res = _cli("run", "gpu0", "--", "sh", "-c", "echo ran", cwd=repo["a"])
    assert res.returncode == 0
    assert "ran" in res.stdout
    assert not _lease_path(repo, "gpu0").exists()
    assert _lease_states(repo["b"])["gpu0"]["state"] == "FREE"
    # And the next holder gets it immediately.
    assert _cli("run", "--quiet", "gpu0", "--", "true", cwd=repo["b"]).returncode == 0


def test_command_failure_propagates_exit_and_releases(repo: dict[str, Path]) -> None:
    res = _cli("run", "--quiet", "gpu0", "--", "sh", "-c", "exit 3", cwd=repo["a"])
    assert res.returncode == 3
    assert not _lease_path(repo, "gpu0").exists()
    assert _lease_states(repo["a"])["gpu0"]["state"] == "FREE"


def test_command_killed_by_signal_releases(repo: dict[str, Path]) -> None:
    holder = _hold("gpu0", repo["a"], repo)
    holder.send_signal(signal.SIGTERM)  # forwarded to the child; resctl then releases
    rc = holder.wait(timeout=20)
    assert rc == 128 + signal.SIGTERM
    assert not _lease_path(repo, "gpu0").exists()
    assert _lease_states(repo["b"])["gpu0"]["state"] == "FREE"


def test_resctl_sigkill_leaves_no_permanent_lock(repo: dict[str, Path]) -> None:
    """SIGKILL skips every ``finally``: the flock still dies with the process."""
    holder = _hold("gpu0", repo["a"], repo)
    holder.kill()
    holder.wait(timeout=20)
    # The lease file survives (nobody could delete it) but must read as stale ...
    row = _lease_states(repo["b"])["gpu0"]
    assert row["state"] == "FREE"
    assert row["stale_lease"] is True
    assert row["owner_alive"] is False
    # ... and a new holder reclaims it without any manual override.
    res = _cli("run", "gpu0", "--", "true", cwd=repo["b"])
    assert res.returncode == 0, res.stderr
    assert "reclaimed stale lease" in res.stderr
    assert not _lease_path(repo, "gpu0").exists()


def test_wait_blocks_until_release(repo: dict[str, Path]) -> None:
    holder = _hold("gpu0", repo["a"], repo)
    waiter = subprocess.Popen(
        [
            sys.executable,
            str(RESCTL),
            "run",
            "--wait",
            "--poll",
            "0.1",
            "--timeout",
            "30",
            "machine-bench",
            "--",
            "true",
        ],
        cwd=repo["b"],
        env=_env(),
        text=True,
        stderr=subprocess.PIPE,
    )
    time.sleep(0.5)
    assert waiter.poll() is None  # still waiting: gpu0 blocks machine-bench
    _stop(holder)
    _, err = waiter.communicate(timeout=30)
    assert waiter.returncode == 0, err
    assert "waiting for machine-bench" in err
    assert "acquired machine-bench" in err


def test_timeout_gives_up_with_owner(repo: dict[str, Path]) -> None:
    holder = _hold("gpu0", repo["a"], repo)
    try:
        res = _cli(
            "run",
            "--timeout",
            "0.5",
            "--poll",
            "0.1",
            "gpu0",
            "--",
            "true",
            cwd=repo["b"],
        )
        assert res.returncode == EXIT_LOCK_FAILED
        assert "timed out waiting for gpu0" in res.stderr
        assert str(repo["a"]) in res.stderr
    finally:
        _stop(holder)


# --------------------------------------------------------------------------- #
# Stale / corrupt metadata never makes a resource look free
# --------------------------------------------------------------------------- #


def test_corrupt_lease_with_held_lock_is_busy_unknown_owner(
    repo: dict[str, Path],
) -> None:
    holder = _hold("gpu0", repo["a"], repo)
    try:
        _lease_path(repo, "gpu0").write_text("{not json", encoding="utf-8")
        row = _lease_states(repo["b"])["gpu0"]
        assert row["state"] == "BUSY"
        assert row["owner"] is None
        assert "corrupt" in str(row["lease_error"])
        who = _cli("who", "gpu0", cwd=repo["b"])
        assert who.returncode == 1
        assert "owner unknown" in who.stdout
        # Acquisition still fails: the flock, not the JSON, is the lock.
        assert (
            _cli("run", "gpu0", "--", "true", cwd=repo["b"]).returncode
            == EXIT_LOCK_FAILED
        )
    finally:
        _stop(holder)


def test_missing_lease_with_held_lock_is_busy(repo: dict[str, Path]) -> None:
    holder = _hold("gpu0", repo["a"], repo)
    try:
        _lease_path(repo, "gpu0").unlink()
        row = _lease_states(repo["b"])["gpu0"]
        assert row["state"] == "BUSY"
        assert row["owner"] is None
        assert (
            _cli("run", "gpu0", "--", "true", cwd=repo["b"]).returncode
            == EXIT_LOCK_FAILED
        )
    finally:
        _stop(holder)


def test_orphan_lease_without_lock_is_stale_and_cleanable(
    repo: dict[str, Path],
) -> None:
    _cli("status", cwd=repo["a"])  # creates the runtime dir
    lease = _lease_path(repo, "gpu0")
    lease.write_text(
        json.dumps(
            {
                "resource": "gpu0",
                "pid": 2**22 - 1,
                "worktree": str(repo["a"]),
                "command_str": "x",
            }
        ),
        encoding="utf-8",
    )
    row = _lease_states(repo["b"])["gpu0"]
    assert row["state"] == "FREE"
    assert row["stale_lease"] is True
    who = _cli("who", "gpu0", cwd=repo["b"])
    assert who.returncode == 0
    assert "stale lease" in who.stdout
    clean = _cli("clean", cwd=repo["b"])
    assert clean.returncode == 0
    assert "gpu0" in clean.stdout
    assert not lease.exists()


def test_clean_never_touches_an_active_lease(repo: dict[str, Path]) -> None:
    holder = _hold("gpu0", repo["a"], repo)
    try:
        assert "no stale leases" in _cli("clean", cwd=repo["b"]).stdout
        assert _lease_path(repo, "gpu0").exists()
        assert _lease_states(repo["b"])["gpu0"]["state"] == "BUSY"
    finally:
        _stop(holder)


# --------------------------------------------------------------------------- #
# Status, worktrees, handoffs
# --------------------------------------------------------------------------- #


def test_status_lists_worktrees_and_lease_owner(repo: dict[str, Path]) -> None:
    (repo["b"] / "scratch.txt").write_text("dirty\n", encoding="utf-8")
    holder = _hold("cpu-heavy", repo["b"], repo)
    try:
        status = _status(repo["a"])
        wts = {w["path"]: w for w in status["worktrees"]}  # type: ignore[union-attr]
        assert set(wts) == {str(repo["a"]), str(repo["b"])}
        assert wts[str(repo["a"])]["branch"] == "main"
        assert wts[str(repo["a"])]["dirty"] is False
        assert wts[str(repo["b"])]["branch"] == "feature"
        assert wts[str(repo["b"])]["dirty"] is True
        assert wts[str(repo["b"])]["head"] == _git("rev-parse", "HEAD", cwd=repo["b"])

        row = _lease_states(repo["a"])["cpu-heavy"]
        assert row["state"] == "BUSY"
        owner = row["owner"]
        assert isinstance(owner, dict)
        assert owner["worktree"] == str(repo["b"])
        assert owner["branch"] == "feature"
        assert owner["pid"] == holder.pid
        assert owner["command"] == ["sleep", "60"]
        assert row["owner_alive"] is True
        assert isinstance(row["elapsed_s"], float)

        # Human rendering carries the same facts and runs without a GPU.
        text = _cli("status", cwd=repo["a"])
        assert text.returncode == 0
        assert "cpu-heavy      BUSY" in text.stdout
        assert str(repo["b"]) in text.stdout
        assert "unavailable" in text.stdout  # GPU section, nvidia-smi absent
        assert status["gpu"]["available"] is False  # type: ignore[index]
    finally:
        _stop(holder)


def test_handoff_readable_from_other_worktree(repo: dict[str, Path]) -> None:
    res = _cli(
        "handoff",
        "finished X; next run Y",
        "--done",
        "X",
        "--next",
        "run Y",
        "--last-result",
        "IDF1 80.4",
        cwd=repo["b"],
    )
    assert res.returncode == 0, res.stderr

    shown = _cli("handoff-show", "--json", cwd=repo["a"])
    assert shown.returncode == 0
    records = json.loads(shown.stdout)
    assert len(records) == 1
    rec = records[0]["record"]
    assert rec["worktree"] == str(repo["b"])
    assert rec["branch"] == "feature"
    assert rec["head"] == _git("rev-parse", "HEAD", cwd=repo["b"])
    assert rec["dirty"] is False
    assert rec["note"] == "finished X; next run Y"
    assert rec["done"] == "X"
    assert rec["next_action"] == "run Y"
    assert rec["last_result"] == "IDF1 80.4"
    assert rec["pending"] == ""
    assert rec["safe_to_remove"] is False

    # Updating merges: unspecified fields survive, git state is refreshed.
    (repo["b"] / "wip.txt").write_text("wip\n", encoding="utf-8")
    res = _cli(
        "handoff", "Y running", "--pending", "Y", "--safe-to-remove", cwd=repo["b"]
    )
    assert res.returncode == 0, res.stderr
    rec = json.loads(_cli("handoff-show", "--json", cwd=repo["a"]).stdout)[0]["record"]
    assert rec["done"] == "X"
    assert rec["pending"] == "Y"
    assert rec["safe_to_remove"] is True
    assert rec["dirty"] is True

    # --here filters to the calling worktree; a has none.
    assert "no handoff recorded" in _cli("handoff-show", "--here", cwd=repo["a"]).stdout
    assert "Y running" in _cli("handoff-show", "--here", cwd=repo["b"]).stdout

    # --clear drops the merged fields.
    _cli("handoff", "fresh", "--clear", cwd=repo["b"])
    rec = json.loads(_cli("handoff-show", "--json", cwd=repo["a"]).stdout)[0]["record"]
    assert rec["done"] == "" and rec["safe_to_remove"] is False


def test_runtime_state_lives_in_git_common_dir_and_is_untracked(
    repo: dict[str, Path],
) -> None:
    _cli("handoff", "n", cwd=repo["b"])
    holder = _hold("gpu0", repo["b"], repo)
    try:
        runtime = repo["common"] / "worktree-runtime"
        assert (runtime / "locks" / "gpu0.lock").exists()
        assert (runtime / "leases" / "gpu0.json").exists()
        assert list((runtime / "handoffs").glob("*.json"))
        # Nothing appears in either worktree's status.
        assert _git("status", "--porcelain", cwd=repo["a"]) == ""
        assert _git("status", "--porcelain", cwd=repo["b"]) == ""
    finally:
        _stop(holder)


def test_unknown_resource_and_missing_command_are_usage_errors(
    repo: dict[str, Path],
) -> None:
    assert _cli("run", "gpu1", "--", "true", cwd=repo["a"]).returncode == 2
    assert _cli("run", "gpu0", cwd=repo["a"]).returncode == 2
    assert _cli("who", "gpu1", cwd=repo["a"]).returncode == 2


def test_outside_git_repo_is_an_environment_error(tmp_path: Path) -> None:
    res = _cli("status", cwd=tmp_path)
    assert res.returncode == 2
    assert "resctl:" in res.stderr


def test_conflict_table_is_symmetric() -> None:
    for res, others in resctl.CONFLICTS.items():
        for other in others:
            assert res in resctl.CONFLICTS[other], f"{res}->{other} conflict is one-way"
    assert set(resctl.CONFLICTS) == set(resctl.RESOURCES)
