"""Fail-closed ``run_manifest.json`` writer for produced experiment artifacts.

ADR 021 (AP-1 / AP-2).  An artifact directory that carries no manifest cannot be
cited safely and cannot be deleted safely, which is why ``runs/`` and
``results/`` grew to tens of gigabytes that nothing can account for.

The contract this module exists to enforce is an **ordering** one, not a
presence one:

    the manifest must be durably on disk *before* the first result byte is
    written, so that a crash — or a kill, or a CUDA OOM — cannot leave an
    anonymous result directory behind.

Checking "the directory ended up with a manifest" would not catch that: the
crash path is exactly the path where the ending never happens.  Callers
therefore invoke :func:`open_run` as their first side effect, and it raises
:class:`ManifestError` rather than returning a partial write.

Prior art: ``scripts/eval/diagnostics/bridge_gate_breakpoints.py`` binds a
*content* fingerprint to a measurement.  A manifest is deliberately weaker and
cheaper — identity, not equivalence — and makes no claim that two runs sharing
one are bit-identical.

This module records mechanical facts only.  Verdicts live in the claim-state
registry and in ADR 020 terminal slots; ``claims`` here holds registry object
ids and nothing else (link, never restate).
"""

# status: stable

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "run_manifest.json"

# Bumped only by an append-only schema change.  Present from v1 precisely
# because unknown fields are fail-closed: without a version, adding a field
# later would make every older reader reject every newer manifest.
SCHEMA_VERSION = 1

PRODUCED_BY = frozenset({"eval", "train", "diagnostic", "ad-hoc"})

# Required: absence is a bug in the caller, not a property of the environment.
REQUIRED_FIELDS = (
    "schema_version",
    "run_id",
    "commit",
    "dirty",
    "produced_by",
    "started_at",
    "host",
    "cmdline",
)

# Optional: legitimately unknown for some producers (a training run has no
# detector; an ad-hoc probe has no preset).  Absent means unknown; it never
# means "does not apply", and nothing downstream may infer a default.
OPTIONAL_FIELDS = (
    "preset",
    "detector",
    "dataset",
    "gpu",
    "claims",
)

ALLOWED_FIELDS = frozenset(REQUIRED_FIELDS + OPTIONAL_FIELDS)


class ManifestError(RuntimeError):
    """The manifest could not be built, validated, or durably written.

    Raised before any result writer starts.  Callers must not catch this and
    continue: continuing is exactly how an anonymous directory is produced.
    """


def _git(*args: str) -> str | None:
    """Run a git command in the repo, or return None when git cannot answer.

    Git being unavailable is a property of the environment (a container, an
    exported tree), not a caller bug, so it yields an explicit null rather than
    a failure.  A null ``commit`` is an honest "unknown", and readers must treat
    it as such instead of assuming the working tree matched some HEAD.
    """
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _git_head() -> str | None:
    return _git("rev-parse", "HEAD")


def _git_dirty() -> bool | None:
    status = _git("status", "--porcelain")
    if status is None:
        return None
    return bool(status.strip())


def _gpu_identity() -> str | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    lines = [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()]
    return " | ".join(lines) or None


def build_manifest(
    run_id: str,
    *,
    produced_by: str,
    preset: str | None = None,
    detector: str | None = None,
    dataset: str | None = None,
    cmdline: Iterable[str] | None = None,
    claims: Iterable[str] = (),
) -> dict[str, Any]:
    """Assemble the manifest payload for one produced artifact directory."""
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "commit": _git_head(),
        "dirty": _git_dirty(),
        "produced_by": produced_by,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": f"{socket.gethostname()} | {platform.platform()}",
        "cmdline": list(cmdline) if cmdline is not None else list(sys.argv),
    }
    gpu = _gpu_identity()
    if gpu is not None:
        payload["gpu"] = gpu
    if preset is not None:
        payload["preset"] = preset
    if detector is not None:
        payload["detector"] = detector
    if dataset is not None:
        payload["dataset"] = dataset
    claim_list = list(claims)
    if claim_list:
        payload["claims"] = claim_list
    return payload


def validate_manifest(payload: Mapping[str, Any]) -> None:
    """Fail-closed schema check.  Raises ManifestError on the first violation.

    Unknown fields are rejected for the same reason the ADR 020 terminal slot
    rejects them: a tolerated stray field is how a vocabulary drifts until two
    spellings of one concept coexist and neither is authoritative.
    """
    if not isinstance(payload, Mapping):
        raise ManifestError(
            f"manifest must be a JSON object, got {type(payload).__name__}"
        )

    unknown = sorted(set(payload) - ALLOWED_FIELDS)
    if unknown:
        raise ManifestError(
            "unknown manifest field(s): "
            + ", ".join(unknown)
            + f" (allowed: {', '.join(sorted(ALLOWED_FIELDS))})"
        )

    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ManifestError("missing required manifest field(s): " + ", ".join(missing))

    version = payload["schema_version"]
    if version != SCHEMA_VERSION:
        raise ManifestError(
            f"unsupported manifest schema_version {version!r}; this reader speaks {SCHEMA_VERSION}"
        )

    produced_by = payload["produced_by"]
    if produced_by not in PRODUCED_BY:
        raise ManifestError(
            f"produced_by must be one of {sorted(PRODUCED_BY)}, got {produced_by!r}"
        )

    run_id = payload["run_id"]
    if not isinstance(run_id, str) or not run_id.strip():
        raise ManifestError(f"run_id must be a non-empty string, got {run_id!r}")

    commit = payload["commit"]
    if commit is not None and not isinstance(commit, str):
        raise ManifestError(
            f"commit must be a string or null, got {type(commit).__name__}"
        )

    dirty = payload["dirty"]
    if dirty is not None and not isinstance(dirty, bool):
        raise ManifestError(f"dirty must be a bool or null, got {type(dirty).__name__}")

    cmdline = payload["cmdline"]
    if not isinstance(cmdline, list) or not all(
        isinstance(item, str) for item in cmdline
    ):
        raise ManifestError("cmdline must be a list of strings")

    claims = payload.get("claims", [])
    if not isinstance(claims, list) or not all(
        isinstance(item, str) for item in claims
    ):
        raise ManifestError("claims must be a list of registry object id strings")


def _write_atomically(path: Path, text: str) -> None:
    """Write via temp file + fsync + rename, then fsync the directory.

    A half-written manifest is worse than none: it would satisfy a presence
    check while carrying an unusable identity.  The rename makes the file
    appear whole or not at all.
    """
    directory = path.parent
    handle, tmp_name = tempfile.mkstemp(
        dir=directory, prefix=".run_manifest.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    dir_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def open_run(
    output_dir: str | os.PathLike[str],
    *,
    produced_by: str,
    preset: str | None = None,
    detector: str | None = None,
    dataset: str | None = None,
    cmdline: Iterable[str] | None = None,
    claims: Iterable[str] = (),
) -> Path:
    """Claim an artifact directory by landing its manifest first.

    Call this as the **first** side effect of a producing entry point, before
    creating sub-directories, opening result files, or dispatching workers.  It
    returns only once the manifest is durably on disk and reads back valid; on
    any failure it raises :class:`ManifestError` and leaves no manifest behind.

    Re-running into an existing directory overwrites the manifest: one
    directory describes one run, and the newest run is the one whose outputs
    are there.  Keep separate runs in separate directories.
    """
    directory = Path(output_dir)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ManifestError(
            f"cannot create artifact directory {directory}: {exc}"
        ) from exc

    payload = build_manifest(
        directory.name,
        produced_by=produced_by,
        preset=preset,
        detector=detector,
        dataset=dataset,
        cmdline=cmdline,
        claims=claims,
    )
    validate_manifest(payload)

    path = directory / MANIFEST_FILENAME
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        _write_atomically(path, text)
    except OSError as exc:
        raise ManifestError(f"cannot write manifest {path}: {exc}") from exc

    # Read back: a write that the filesystem accepted but that cannot be parsed
    # again is a failure, and the caller must learn that now rather than when
    # someone tries to cite the directory months later.
    readback = read_manifest(directory)
    if readback != payload:
        raise ManifestError(f"manifest at {path} did not read back as written")
    return path


def read_manifest(output_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and validate the manifest of an artifact directory."""
    path = Path(output_dir) / MANIFEST_FILENAME
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest {path} is not valid JSON: {exc}") from exc
    validate_manifest(payload)
    return payload


def require_manifest(output_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Assert that a directory is already claimed, for writers downstream of open_run."""
    directory = Path(output_dir)
    if not (directory / MANIFEST_FILENAME).exists():
        raise ManifestError(
            f"{directory} carries no {MANIFEST_FILENAME}; call open_run() before writing results"
        )
    return read_manifest(directory)
