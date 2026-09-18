#!/usr/bin/env python3
"""resctl -- local resource leases for parallel git worktrees / agents.

Several worktrees of this repository share one machine.  A benchmark in one
worktree and a ``cmake --build`` in another silently corrupt each other's
measurements, and two GPU jobs racing for the same device fail in confusing
ways.  ``resctl`` gives every agent one CLI to answer *who is using what right
now, and may I start?* without a daemon, a server, or any tracked file.

Resources (see ``RESOURCES`` / ``CONFLICTS``):

    gpu0           ordinary CUDA / GPU work
    cpu-heavy      big builds, pytest, data generation
    machine-bench  formal performance measurement: exclusive use of the box;
                   conflicts with gpu0 *and* cpu-heavy

Mechanism:

* The lock is a kernel ``flock(2)`` on ``<git-common-dir>/worktree-runtime/
  locks/<resource>.lock``.  The kernel drops it when the holder dies, so a
  crashed command can never leave a permanent lock behind.
* A JSON lease next to it (``leases/<resource>.json``) records *who* holds the
  lock (worktree, branch, HEAD, PID, command, start time).  The lease is
  metadata only: a lease whose flock is free is *stale* and is reported as
  such; a held flock with a missing or corrupt lease is reported as BUSY with
  an unknown owner.  Metadata can never make a resource look free.
* Cross-resource conflicts are resolved under a short ``.acquire.lock``
  critical section: after taking our own lock we probe every conflicting lock
  with a non-blocking flock and back out if any probe fails.
* Handoffs (``handoffs/<worktree-key>.json``) are short-lived execution notes
  per worktree; ``docs/HANDOFF.md``-style research state stays in the repo.

All state lives under the shared git common directory, so every worktree of
the repository sees the same locks, leases and handoffs, and nothing is ever
committed.

Usage:
    tools/resctl.py status [--json]
    tools/resctl.py who RESOURCE
    tools/resctl.py run [--wait] [--timeout S] RESOURCE -- COMMAND ...
    tools/resctl.py handoff [--done ..] [--pending ..] [--last-result ..]
                            [--next ..] [--safe-to-remove|--not-safe-to-remove]
                            [--clear] NOTE
    tools/resctl.py handoff-show [--here] [--json]
    tools/resctl.py clean

Exit codes: 0 ok / resource free; 1 resource busy (``who``); 2 usage or
environment error; 75 lock not acquired (``run``); otherwise the command's own
exit status.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# --------------------------------------------------------------------------- #
# Resource model
# --------------------------------------------------------------------------- #

RESOURCES: tuple[str, ...] = ("gpu0", "cpu-heavy", "machine-bench")

# Which other resources must be *free* for a resource to be acquired.  A
# resource always conflicts with itself (that is the flock on its own file).
CONFLICTS: dict[str, tuple[str, ...]] = {
    "gpu0": ("machine-bench",),
    "cpu-heavy": ("machine-bench",),
    "machine-bench": ("gpu0", "cpu-heavy"),
}

RUNTIME_DIRNAME = "worktree-runtime"
LEASE_VERSION = 1
HANDOFF_VERSION = 1

EXIT_BUSY = 1
EXIT_USAGE = 2
EXIT_LOCK_FAILED = 75  # EX_TEMPFAIL


class ResctlError(Exception):
    """Environment or usage error; reported on stderr with exit 2."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_ts(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _fmt_elapsed(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# --------------------------------------------------------------------------- #
# Git helpers
# --------------------------------------------------------------------------- #


def _git(args: list[str], cwd: Path | None = None) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise ResctlError("git is not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise ResctlError(f"git {' '.join(args)} timed out") from exc
    if proc.returncode != 0:
        where = f" (in {cwd})" if cwd else ""
        raise ResctlError(
            f"git {' '.join(args)} failed{where}: {proc.stderr.strip() or proc.returncode}"
        )
    return proc.stdout.strip()


def git_common_dir(cwd: Path | None = None) -> Path:
    """The shared ``.git`` directory of the repository owning ``cwd``."""
    return Path(_git(["rev-parse", "--path-format=absolute", "--git-common-dir"], cwd))


@dataclass(frozen=True)
class WorktreeInfo:
    path: str
    branch: str  # "detached" when not on a branch
    head: str
    dirty: bool | None  # None = unknown (git status failed / path missing)
    dirty_summary: str
    exists: bool = True

    @property
    def key(self) -> str:
        return worktree_key(self.path)


def worktree_key(path: str) -> str:
    """Stable, filesystem-safe identifier for a worktree path."""
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
    base = "".join(c if c.isalnum() or c in "-_." else "_" for c in Path(path).name)
    return f"{base or 'root'}-{digest}"


def _worktree_dirty(path: Path) -> tuple[bool | None, str]:
    try:
        out = _git(["status", "--porcelain"], path)
    except ResctlError as exc:
        return None, f"unknown ({exc})"
    lines = [line for line in out.splitlines() if line.strip()]
    if not lines:
        return False, "clean"
    untracked = sum(1 for line in lines if line.startswith("??"))
    modified = len(lines) - untracked
    parts: list[str] = []
    if modified:
        parts.append(f"{modified} modified")
    if untracked:
        parts.append(f"{untracked} untracked")
    return True, "dirty: " + ", ".join(parts)


def current_worktree(cwd: Path | None = None) -> WorktreeInfo:
    """Describe the worktree that contains ``cwd`` (default: the process cwd)."""
    top = Path(_git(["rev-parse", "--show-toplevel"], cwd))
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], top)
    if branch == "HEAD":
        branch = "detached"
    head = _git(["rev-parse", "HEAD"], top)
    dirty, summary = _worktree_dirty(top)
    return WorktreeInfo(str(top), branch, head, dirty, summary)


def list_worktrees(cwd: Path | None = None) -> list[WorktreeInfo]:
    """All worktrees registered with the repository owning ``cwd``."""
    raw = _git(["worktree", "list", "--porcelain"], cwd)
    entries: list[WorktreeInfo] = []
    block: dict[str, str] = {}

    def flush() -> None:
        if not block:
            return
        path = block.get("worktree", "")
        if "bare" in block:
            block.clear()
            return
        branch = block.get("branch", "")
        branch = branch.removeprefix("refs/heads/") if branch else "detached"
        head = block.get("HEAD", "")
        p = Path(path)
        if p.is_dir():
            dirty, summary = _worktree_dirty(p)
            entries.append(WorktreeInfo(path, branch, head, dirty, summary, True))
        else:
            entries.append(
                WorktreeInfo(path, branch, head, None, "path missing", False)
            )
        block.clear()

    for line in raw.splitlines():
        if not line.strip():
            flush()
            continue
        key, _, value = line.partition(" ")
        block[key] = value
    flush()
    return entries


# --------------------------------------------------------------------------- #
# Runtime directory
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Runtime:
    root: Path

    @property
    def locks(self) -> Path:
        return self.root / "locks"

    @property
    def leases(self) -> Path:
        return self.root / "leases"

    @property
    def handoffs(self) -> Path:
        return self.root / "handoffs"

    def lock_path(self, resource: str) -> Path:
        return self.locks / f"{resource}.lock"

    def lease_path(self, resource: str) -> Path:
        return self.leases / f"{resource}.json"

    @property
    def acquire_lock_path(self) -> Path:
        return self.locks / ".acquire.lock"

    def ensure(self) -> None:
        for d in (self.locks, self.leases, self.handoffs):
            d.mkdir(parents=True, exist_ok=True)


def runtime_for(cwd: Path | None = None) -> Runtime:
    rt = Runtime(git_common_dir(cwd) / RUNTIME_DIRNAME)
    rt.ensure()
    return rt


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Return (payload, error).  Missing file -> (None, None); corrupt -> (None, why)."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        return None, f"unreadable: {exc}"
    try:
        data = json.loads(text)
    except ValueError as exc:
        return None, f"corrupt JSON: {exc}"
    if not isinstance(data, dict):
        return None, "corrupt: not a JSON object"
    return data, None


# --------------------------------------------------------------------------- #
# Locks and leases
# --------------------------------------------------------------------------- #


def _try_flock(path: Path) -> int | None:
    """Take an exclusive non-blocking flock on ``path``; return the fd or None.

    The fd is non-inheritable, so a command run under the lease cannot carry
    the lock past resctl's own lifetime.
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, PermissionError):
        os.close(fd)
        return None
    return fd


def _flock_is_held(path: Path) -> bool:
    """True when some process currently holds the flock on ``path``."""
    if not path.exists():
        return False
    fd = _try_flock(path)
    if fd is None:
        return True
    os.close(fd)
    return False


@contextlib.contextmanager
def _acquire_section(rt: Runtime, shared: bool = False) -> Iterator[None]:
    """Serialise acquisition (exclusive) or observation (shared) of the lock set."""
    fd = os.open(rt.acquire_lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


@dataclass(frozen=True)
class LeaseState:
    resource: str
    held: bool  # ground truth: the flock is taken
    lease: dict[str, Any] | None  # metadata (may be absent or stale)
    lease_error: str | None  # why the lease file could not be parsed

    @property
    def stale(self) -> bool:
        return (not self.held) and (
            self.lease is not None or self.lease_error is not None
        )

    @property
    def owner_alive(self) -> bool | None:
        if self.lease is None:
            return None
        pid = self.lease.get("pid")
        return _pid_alive(pid) if isinstance(pid, int) else None

    def elapsed(self) -> float | None:
        if self.lease is None:
            return None
        started = _parse_ts(str(self.lease.get("start_time", "")))
        if started is None:
            return None
        return (datetime.now(timezone.utc) - started).total_seconds()

    def owner_line(self) -> str:
        if self.lease is None:
            if self.lease_error:
                return f"owner unknown (lease {self.lease_error})"
            return "owner unknown (no lease metadata)"
        wt = self.lease.get("worktree", "?")
        pid = self.lease.get("pid", "?")
        branch = self.lease.get("branch", "?")
        cmd = self.lease.get("command_str", "?")
        alive = self.owner_alive
        alive_s = "" if alive is None else (" alive" if alive else " DEAD")
        return f"{wt} [{branch}] pid {pid}{alive_s}: {cmd}"


def inspect_resource(rt: Runtime, resource: str) -> LeaseState:
    if resource not in RESOURCES:
        raise ResctlError(
            f"unknown resource {resource!r}; choose from {', '.join(RESOURCES)}"
        )
    held = _flock_is_held(rt.lock_path(resource))
    lease, err = _read_json(rt.lease_path(resource))
    return LeaseState(resource, held, lease, err)


def inspect_all(rt: Runtime) -> list[LeaseState]:
    with _acquire_section(rt, shared=True):
        return [inspect_resource(rt, r) for r in RESOURCES]


@dataclass
class Acquisition:
    resource: str
    fd: int | None = None
    blocked_by: LeaseState | None = None
    reclaimed_stale: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.fd is not None


def _make_lease(resource: str, wt: WorktreeInfo, command: list[str]) -> dict[str, Any]:
    return {
        "lease_version": LEASE_VERSION,
        "resource": resource,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "worktree": wt.path,
        "branch": wt.branch,
        "head": wt.head,
        "dirty": wt.dirty,
        "start_time": _utcnow(),
        "command": command,
        "command_str": shlex.join(command),
    }


def try_acquire(
    rt: Runtime, resource: str, wt: WorktreeInfo, command: list[str]
) -> Acquisition:
    """One non-blocking acquisition attempt.  Never waits, never overrides."""
    if resource not in RESOURCES:
        raise ResctlError(
            f"unknown resource {resource!r}; choose from {', '.join(RESOURCES)}"
        )
    acq = Acquisition(resource)
    with _acquire_section(rt):
        fd = _try_flock(rt.lock_path(resource))
        if fd is None:
            acq.blocked_by = inspect_resource(rt, resource)
            return acq
        for other in CONFLICTS[resource]:
            probe = _try_flock(rt.lock_path(other))
            if probe is None:
                os.close(fd)
                acq.blocked_by = inspect_resource(rt, other)
                return acq
            os.close(probe)
        # We own the flock; any lease left here belongs to a dead holder.
        old, _ = _read_json(rt.lease_path(resource))
        if old is not None or rt.lease_path(resource).exists():
            acq.reclaimed_stale = old or {"note": "corrupt lease"}
        _write_json_atomic(rt.lease_path(resource), _make_lease(resource, wt, command))
        acq.fd = fd
    return acq


def release(rt: Runtime, acq: Acquisition) -> None:
    """Drop the lease *then* the flock, so nobody can see a lease without a lock."""
    if acq.fd is None:
        return
    with contextlib.suppress(FileNotFoundError):
        rt.lease_path(acq.resource).unlink()
    os.close(acq.fd)
    acq.fd = None


def clean_stale(rt: Runtime) -> list[str]:
    """Remove lease files whose flock is free.  Active leases are never touched."""
    removed: list[str] = []
    with _acquire_section(rt):
        for resource in RESOURCES:
            state = inspect_resource(rt, resource)
            if state.stale:
                with contextlib.suppress(FileNotFoundError):
                    rt.lease_path(resource).unlink()
                removed.append(resource)
    return removed


# --------------------------------------------------------------------------- #
# Handoffs
# --------------------------------------------------------------------------- #


@dataclass
class Handoff:
    worktree: str
    branch: str
    head: str
    dirty: bool | None
    dirty_summary: str
    note: str
    done: str = ""
    pending: str = ""
    last_result: str = ""
    next_action: str = ""
    safe_to_remove: bool = False
    updated_at: str = field(default_factory=_utcnow)
    handoff_version: int = HANDOFF_VERSION


def handoff_path(rt: Runtime, wt_path: str) -> Path:
    return rt.handoffs / f"{worktree_key(wt_path)}.json"


def write_handoff(
    rt: Runtime,
    wt: WorktreeInfo,
    note: str,
    *,
    done: str | None,
    pending: str | None,
    last_result: str | None,
    next_action: str | None,
    safe_to_remove: bool | None,
    clear: bool,
) -> Handoff:
    """Merge the given fields into the worktree's handoff (or start fresh with ``clear``)."""
    previous: dict[str, Any] = {}
    if not clear:
        prev, _ = _read_json(handoff_path(rt, wt.path))
        previous = prev or {}

    def pick(new: str | None, key: str) -> str:
        if new is not None:
            return new
        value = previous.get(key, "")
        return value if isinstance(value, str) else ""

    prev_safe = previous.get("safe_to_remove", False)
    record = Handoff(
        worktree=wt.path,
        branch=wt.branch,
        head=wt.head,
        dirty=wt.dirty,
        dirty_summary=wt.dirty_summary,
        note=note,
        done=pick(done, "done"),
        pending=pick(pending, "pending"),
        last_result=pick(last_result, "last_result"),
        next_action=pick(next_action, "next_action"),
        safe_to_remove=(
            safe_to_remove if safe_to_remove is not None else bool(prev_safe)
        ),
    )
    _write_json_atomic(handoff_path(rt, wt.path), asdict(record))
    return record


def read_handoffs(rt: Runtime) -> list[tuple[Path, dict[str, Any] | None, str | None]]:
    out: list[tuple[Path, dict[str, Any] | None, str | None]] = []
    for path in sorted(rt.handoffs.glob("*.json")):
        data, err = _read_json(path)
        out.append((path, data, err))
    return out


# --------------------------------------------------------------------------- #
# Hardware / system snapshot
# --------------------------------------------------------------------------- #

_GPU_FIELDS = (
    "index",
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
)


def _nvidia_smi_binary() -> str | None:
    override = os.environ.get("RESCTL_NVIDIA_SMI")
    if override:
        return override if Path(override).exists() else None
    return shutil.which("nvidia-smi")


def gpu_snapshot() -> dict[str, Any]:
    binary = _nvidia_smi_binary()
    if binary is None:
        return {"available": False, "reason": "nvidia-smi not found", "gpus": []}
    try:
        proc = subprocess.run(
            [
                binary,
                f"--query-gpu={','.join(_GPU_FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": f"nvidia-smi failed: {exc}", "gpus": []}
    if proc.returncode != 0:
        reason = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        return {"available": False, "reason": f"nvidia-smi: {reason}", "gpus": []}
    gpus: list[dict[str, str]] = []
    for line in proc.stdout.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) != len(_GPU_FIELDS):
            continue
        gpus.append(dict(zip(_GPU_FIELDS, cells)))
    if not gpus:
        return {"available": False, "reason": "nvidia-smi returned no GPUs", "gpus": []}
    procs: list[dict[str, str]] = []
    try:
        pq = subprocess.run(
            [
                binary,
                "--query-compute-apps=pid,used_memory,process_name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if pq.returncode == 0:
            for line in pq.stdout.splitlines():
                cells = [c.strip() for c in line.split(",")]
                if len(cells) == 3:
                    procs.append(
                        {"pid": cells[0], "used_memory": cells[1], "name": cells[2]}
                    )
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"available": True, "gpus": gpus, "compute_apps": procs}


def system_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {"cpus": os.cpu_count()}
    try:
        snap["load"] = list(os.getloadavg())
    except OSError:
        snap["load"] = None
    mem: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                mem[key] = int(rest.split()[0]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    snap["ram_total_mb"] = mem.get("MemTotal")
    snap["ram_available_mb"] = mem.get("MemAvailable")
    return snap


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #


def collect_status(rt: Runtime, cwd: Path | None = None) -> dict[str, Any]:
    worktrees = list_worktrees(cwd)
    leases = inspect_all(rt)
    handoffs = read_handoffs(rt)
    known_paths = {w.path for w in worktrees}
    return {
        "runtime_dir": str(rt.root),
        "worktrees": [asdict(w) for w in worktrees],
        "leases": [
            {
                "resource": s.resource,
                "state": "BUSY" if s.held else "FREE",
                "stale_lease": s.stale,
                "owner": s.lease,
                "owner_alive": s.owner_alive,
                "lease_error": s.lease_error,
                "elapsed_s": s.elapsed(),
            }
            for s in leases
        ],
        "gpu": gpu_snapshot(),
        "system": system_snapshot(),
        "handoffs": [
            {
                "file": str(p),
                "record": d,
                "error": e,
                "worktree_known": bool(d and d.get("worktree") in known_paths),
            }
            for p, d, e in handoffs
        ],
    }


def _short(sha: str) -> str:
    return sha[:8] if sha else "?"


def render_status(status: dict[str, Any]) -> str:
    out: list[str] = []
    out.append(f"runtime: {status['runtime_dir']}")
    out.append("")
    out.append("WORKTREES")
    for w in status["worktrees"]:
        flag = "" if w["exists"] else "  (PATH MISSING)"
        out.append(f"- {w['path']}{flag}")
        out.append(
            f"    branch {w['branch']}  HEAD {_short(w['head'])}  {w['dirty_summary']}"
        )
    out.append("")
    out.append("LEASES")
    for lease in status["leases"]:
        owner = lease["owner"]
        if lease["state"] == "BUSY":
            elapsed = lease["elapsed_s"]
            el = f"  elapsed {_fmt_elapsed(elapsed)}" if elapsed is not None else ""
            if owner is None:
                why = lease["lease_error"] or "no lease metadata"
                out.append(f"- {lease['resource']:<14} BUSY  owner unknown ({why})")
            else:
                alive = lease["owner_alive"]
                alive_s = (
                    "" if alive is None else ("" if alive else "  (owner pid DEAD?)")
                )
                out.append(
                    f"- {lease['resource']:<14} BUSY  {owner.get('worktree', '?')} "
                    f"[{owner.get('branch', '?')}]  pid {owner.get('pid', '?')}{el}{alive_s}"
                )
                out.append(f"    cmd: {owner.get('command_str', '?')}")
        elif lease["stale_lease"]:
            if owner is None:
                out.append(
                    f"- {lease['resource']:<14} FREE  (stale lease: {lease['lease_error'] or 'corrupt'};"
                    " run `resctl clean`)"
                )
            else:
                out.append(
                    f"- {lease['resource']:<14} FREE  (stale lease from pid {owner.get('pid', '?')} "
                    f"{owner.get('worktree', '?')}; run `resctl clean`)"
                )
        else:
            out.append(f"- {lease['resource']:<14} FREE")
    out.append("")
    out.append("GPU")
    gpu = status["gpu"]
    if not gpu["available"]:
        out.append(f"- unavailable ({gpu['reason']})")
    else:
        for g in gpu["gpus"]:
            out.append(
                f"- [{g['index']}] {g['name']}  util {g['utilization.gpu']}%  "
                f"vram {g['memory.used']}/{g['memory.total']} MiB  "
                f"{g['temperature.gpu']}C  {g['power.draw']} W"
            )
        apps = gpu.get("compute_apps") or []
        if apps:
            out.append(
                "    compute apps: "
                + ", ".join(f"pid {a['pid']} ({a['used_memory']} MiB)" for a in apps)
            )
    out.append("")
    out.append("SYSTEM")
    s = status["system"]
    load = s["load"]
    load_s = "  ".join(f"{x:.2f}" for x in load) if load else "?"
    out.append(f"- cpus {s['cpus']}  load(1/5/15) {load_s}")
    total, avail = s["ram_total_mb"], s["ram_available_mb"]
    if total is not None and avail is not None:
        out.append(f"- ram {total - avail}/{total} MiB used  ({avail} MiB available)")
    else:
        out.append("- ram unknown")
    out.append("")
    out.append("HANDOFF")
    if not status["handoffs"]:
        out.append("- (none)")
    for h in status["handoffs"]:
        rec = h["record"]
        if rec is None:
            out.append(f"- {h['file']}: {h['error']}")
            continue
        gone = "" if h["worktree_known"] else "  (WORKTREE GONE)"
        out.append(
            f"- {rec.get('worktree', '?')}{gone}  [{rec.get('branch', '?')} @ {_short(str(rec.get('head', '')))}"
            f", {rec.get('dirty_summary', '?')}]  updated {rec.get('updated_at', '?')}"
        )
        out.append(f"    note: {rec.get('note', '')}")
        for key, label in (
            ("done", "done"),
            ("pending", "pending"),
            ("last_result", "last result"),
            ("next_action", "next"),
        ):
            if rec.get(key):
                out.append(f"    {label}: {rec[key]}")
        out.append(
            f"    safe_to_remove: {'yes' if rec.get('safe_to_remove') else 'no'}"
        )
    return "\n".join(out)


def render_handoff(rec: dict[str, Any]) -> str:
    lines = [
        f"worktree:       {rec.get('worktree', '?')}",
        f"branch:         {rec.get('branch', '?')}",
        f"HEAD:           {rec.get('head', '?')}",
        f"dirty:          {rec.get('dirty_summary', '?')}",
        f"updated:        {rec.get('updated_at', '?')}",
        f"note:           {rec.get('note', '')}",
        f"done:           {rec.get('done', '')}",
        f"pending:        {rec.get('pending', '')}",
        f"last result:    {rec.get('last_result', '')}",
        f"next action:    {rec.get('next_action', '')}",
        f"safe_to_remove: {'yes' if rec.get('safe_to_remove') else 'no'}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def _log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(f"[resctl] {msg}", file=sys.stderr, flush=True)


def cmd_status(args: argparse.Namespace) -> int:
    rt = runtime_for()
    status = collect_status(rt)
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print(render_status(status))
    return 0


def cmd_who(args: argparse.Namespace) -> int:
    rt = runtime_for()
    with _acquire_section(rt, shared=True):
        state = inspect_resource(rt, args.resource)
    if state.held:
        el = state.elapsed()
        el_s = f" for {_fmt_elapsed(el)}" if el is not None else ""
        print(f"{args.resource}: BUSY{el_s} -- {state.owner_line()}")
        return EXIT_BUSY
    if state.stale:
        print(f"{args.resource}: FREE (stale lease present: {state.owner_line()})")
    else:
        print(f"{args.resource}: FREE")
    return 0


def _die_with_parent() -> None:
    """Best effort: have the kernel SIGTERM the command if resctl itself dies.

    The flock is released the moment resctl dies (that is the whole point of
    using flock), so without this an orphaned command would keep using the
    hardware while the resource already reads FREE.  Linux only; grandchildren
    that re-parent themselves are not covered.
    """
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, int(signal.SIGTERM), 0, 0, 0)  # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass


def cmd_run(args: argparse.Namespace) -> int:
    command: list[str] = list(args.command)
    if not command:
        raise ResctlError("no command given; usage: resctl run RESOURCE -- COMMAND ...")
    rt = runtime_for()
    wt = current_worktree()
    wait = bool(args.wait) or args.timeout is not None
    deadline = None if args.timeout is None else time.monotonic() + args.timeout
    acq = try_acquire(rt, args.resource, wt, command)
    while not acq.ok:
        blocker = acq.blocked_by
        assert blocker is not None
        timed_out = deadline is not None and time.monotonic() >= deadline
        if not wait or timed_out:
            why = "timed out waiting for" if timed_out else "cannot acquire"
            _log(
                f"{why} {args.resource}: {blocker.resource} is BUSY -- {blocker.owner_line()}",
                args.quiet,
            )
            if not wait:
                _log(
                    "(use --wait / --timeout S to block until it is released)",
                    args.quiet,
                )
            return EXIT_LOCK_FAILED
        _log(
            f"waiting for {args.resource}: {blocker.resource} is BUSY -- {blocker.owner_line()}",
            args.quiet,
        )
        time.sleep(args.poll)
        acq = try_acquire(rt, args.resource, wt, command)

    if acq.reclaimed_stale is not None:
        stale = acq.reclaimed_stale
        _log(
            f"reclaimed stale lease on {args.resource} "
            f"(pid {stale.get('pid', '?')}, {stale.get('worktree', '?')})",
            args.quiet,
        )
    _log(
        f"acquired {args.resource}  worktree {wt.path} [{wt.branch}]  pid {os.getpid()}",
        args.quiet,
    )
    started = time.monotonic()
    rc = 0
    try:
        try:
            proc = subprocess.Popen(command, preexec_fn=_die_with_parent)
        except OSError as exc:
            _log(f"failed to start {shlex.join(command)}: {exc}", args.quiet)
            return 127

        def forward(signum: int, _frame: Any) -> None:
            if proc.poll() is None:
                with contextlib.suppress(OSError):
                    proc.send_signal(signum)

        previous = {
            s: signal.signal(s, forward)
            for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        }
        try:
            rc = proc.wait()
        finally:
            for s, handler in previous.items():
                signal.signal(s, handler)
    finally:
        release(rt, acq)
        how = f"exit {rc}" if rc >= 0 else f"signal {-rc}"
        _log(
            f"released {args.resource} after {_fmt_elapsed(time.monotonic() - started)} ({how})",
            args.quiet,
        )
    return rc if rc >= 0 else 128 - rc


def cmd_handoff(args: argparse.Namespace) -> int:
    rt = runtime_for()
    wt = current_worktree()
    safe: bool | None = None
    if args.safe_to_remove:
        safe = True
    elif args.not_safe_to_remove:
        safe = False
    record = write_handoff(
        rt,
        wt,
        args.note,
        done=args.done,
        pending=args.pending,
        last_result=args.last_result,
        next_action=args.next,
        safe_to_remove=safe,
        clear=args.clear,
    )
    print(f"handoff written: {handoff_path(rt, wt.path)}")
    print(render_handoff(asdict(record)))
    return 0


def cmd_handoff_show(args: argparse.Namespace) -> int:
    rt = runtime_for()
    entries = read_handoffs(rt)
    if args.here:
        here = current_worktree().path
        entries = [
            e for e in entries if e[1] is not None and e[1].get("worktree") == here
        ]
        if not entries:
            print(f"no handoff recorded for {here}")
            return 0
    if args.json:
        print(
            json.dumps(
                [{"file": str(p), "record": d, "error": e} for p, d, e in entries],
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not entries:
        print("no handoffs recorded")
        return 0
    known = {w.path for w in list_worktrees()}
    blocks: list[str] = []
    for path, data, err in entries:
        if data is None:
            blocks.append(f"{path}: {err}")
            continue
        header = "" if data.get("worktree") in known else "  (WORKTREE GONE)"
        blocks.append(f"== {data.get('worktree', '?')}{header}\n{render_handoff(data)}")
    print("\n\n".join(blocks))
    return 0


def cmd_clean(_args: argparse.Namespace) -> int:
    rt = runtime_for()
    removed = clean_stale(rt)
    if removed:
        print("removed stale lease(s): " + ", ".join(removed))
    else:
        print("no stale leases")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="resctl",
        description="Lease GPU / CPU / whole-machine resources across git worktrees.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("status", help="worktrees, leases, GPU, system and handoffs")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("who", help="who holds RESOURCE (exit 1 when busy)")
    p.add_argument("resource", choices=RESOURCES)
    p.set_defaults(func=cmd_who)

    p = sub.add_parser("run", help="run COMMAND while holding RESOURCE")
    p.add_argument("resource", choices=RESOURCES)
    p.add_argument(
        "--wait", action="store_true", help="poll until the resource is free"
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="wait at most S seconds (implies --wait)",
    )
    p.add_argument(
        "--poll", type=float, default=2.0, help="seconds between attempts (with --wait)"
    )
    p.add_argument(
        "--quiet", action="store_true", help="suppress [resctl] lines on stderr"
    )
    p.set_defaults(func=cmd_run, command=[])

    p = sub.add_parser("handoff", help="record this worktree's execution handoff")
    p.add_argument("note", help="free-text summary, e.g. 'finished X; next run Y'")
    p.add_argument("--done")
    p.add_argument("--pending")
    p.add_argument("--last-result")
    p.add_argument("--next")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--safe-to-remove", action="store_true")
    g.add_argument("--not-safe-to-remove", action="store_true")
    p.add_argument(
        "--clear", action="store_true", help="drop previously recorded fields"
    )
    p.set_defaults(func=cmd_handoff)

    p = sub.add_parser("handoff-show", help="show handoffs of every worktree")
    p.add_argument("--here", action="store_true", help="only this worktree")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_handoff_show)

    p = sub.add_parser(
        "clean", help="remove stale lease metadata (never an active lock)"
    )
    p.set_defaults(func=cmd_clean)
    return parser


def _split_command(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split ``run ... -- COMMAND`` at the first ``--`` so argparse never sees the command."""
    if argv and argv[0] == "run" and "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1 :]
    return argv, []


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    own, command = _split_command(list(sys.argv[1:] if argv is None else argv))
    args = parser.parse_args(own)
    if args.cmd == "run":
        if not command:
            parser.error("run needs `-- COMMAND ...` after the resource")
        args.command = command
    try:
        result: int = args.func(args)
    except ResctlError as exc:
        print(f"resctl: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return result


if __name__ == "__main__":
    sys.exit(main())
