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

There are two ways a manifest can come to stand over a directory, and v2 makes
them distinguishable in the file itself (``provenance_mode``):

* ``production`` — :func:`open_run`, before the first result byte.  This is the
  mode that carries the ordering guarantee above.
* ``reconstructed`` — :func:`attach_reconstructed_manifest`, afterwards, from
  named sources (ADR 021 AP-4).  It carries no ordering guarantee at all, and a
  reader that could not tell the two apart would extend the first one's trust
  to the second.

A v1 manifest carries no mode field because when it was written there was only
one writer.  It reads as legacy production; see :func:`provenance_mode_of`.

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

# What this module *writes*.  v2 adds ``provenance_mode`` and
# ``backfill_sources`` (ADR 021 AP-4).
SCHEMA_VERSION = 2

# What this module *reads*.  The bump is append-only, and the reason is not
# caution: v1 has an unambiguous meaning.  The only way to create a v1 manifest
# was ``open_run``, which writes before the first result byte — no
# reconstruction writer existed — so a v1 file **is** a production manifest, and
# reading it as one recovers a fact rather than picking a default.
#
# Rejecting v1 instead would also have opened a transition race with no upside:
# AP-2 has been live on ``main`` since #330, so any run between the survey that
# observed ``0 manifested`` and this change lands writes a perfectly valid v1
# manifest, which would turn ``invalid`` the moment this merged and fail the
# AP-3 check closed on a directory that did nothing wrong.
SUPPORTED_SCHEMA_VERSIONS = (1, 2)
LEGACY_SCHEMA_VERSION = 1

PRODUCED_BY = frozenset({"eval", "train", "diagnostic", "ad-hoc"})

# How the manifest came to stand over these bytes.
#
# ``production``    — written by the run itself, before its first result byte.
#                     Carries the ordering guarantee ``open_run`` enforces.
# ``reconstructed`` — attached afterwards from named sources (ADR 021 AP-4).
#                     Carries no ordering guarantee whatsoever.
#
# These must be distinguishable in the file.  If they were not, a downstream
# reader could not tell an identity captured at production time from one
# assembled later by archaeology, and would extend the trust earned by the
# first to the second.
PROVENANCE_MODES = frozenset({"production", "reconstructed"})

# Present in both v2 modes.
CORE_REQUIRED_FIELDS = (
    "schema_version",
    "run_id",
    "provenance_mode",
)

# Production: every one of these is knowable at the moment the run starts, so
# absence is a bug in the caller, not a property of the environment.
PRODUCTION_REQUIRED_FIELDS = CORE_REQUIRED_FIELDS + (
    "produced_by",
    "commit",
    "dirty",
    "started_at",
    "host",
    "cmdline",
)

# v1 had no mode field and exactly one producer, ``open_run``.  Its required set
# is the production set minus the field that did not exist yet.
LEGACY_REQUIRED_FIELDS = tuple(
    field for field in PRODUCTION_REQUIRED_FIELDS if field != "provenance_mode"
)

# Reconstruction: fewer fields are required, and this is not a lower bar.
#
# Nothing observed the run, so ``started_at`` / ``host`` / ``cmdline`` /
# ``dirty`` are establishable only if some record happens to state them.  The
# alternative to allowing their absence is filling them — with the current
# clock, an empty string, an empty list — which produces a manifest that is
# schema-valid and factually false.  Absence already means "unknown" here
# (see OPTIONAL_FIELDS), so absence is the honest encoding and the required
# set shrinks to what a reconstruction must nonetheless establish:
#
#   * ``commit`` — non-null.  A reconstructed manifest whose commit is unknown
#     accounts for nothing; it is not worth writing.
#   * ``backfill_sources`` — where each stated fact came from, so that the
#     reconstruction is auditable rather than merely plausible.
#
# ``produced_by`` is deliberately *not* in this set.  It is a closed vocabulary,
# and nothing outside the run itself records which term applies; a rule of the
# form "this file has a preset and a detector, therefore it was an eval" is an
# inference, and writing an inference into a field that reads like an observed
# fact is the exact failure this module exists to prevent.  It is written only
# when a record inside the directory states it.
RECONSTRUCTED_REQUIRED_FIELDS = CORE_REQUIRED_FIELDS + (
    "commit",
    "backfill_sources",
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

ALLOWED_FIELDS = frozenset(
    PRODUCTION_REQUIRED_FIELDS + RECONSTRUCTED_REQUIRED_FIELDS + OPTIONAL_FIELDS
)

# Backwards-compatible alias: the production set is what a producing entry
# point must supply, and that is what the name meant in v1.
REQUIRED_FIELDS = PRODUCTION_REQUIRED_FIELDS


def required_fields(provenance_mode: str) -> tuple[str, ...]:
    """The fields a v2 manifest of this mode must carry."""
    if provenance_mode == "reconstructed":
        return RECONSTRUCTED_REQUIRED_FIELDS
    return PRODUCTION_REQUIRED_FIELDS


def provenance_mode_of(payload: Mapping[str, Any]) -> str:
    """How this manifest came to stand over its bytes, v1 included.

    A v1 file carries no mode because when it was written there was only one:
    ``open_run`` had no counterpart.  Reading it as production is recovering
    what it meant, not supplying a default.
    """
    if payload.get("schema_version") == LEGACY_SCHEMA_VERSION:
        return "production"
    return str(payload["provenance_mode"])


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
        "provenance_mode": "production",
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

    if "schema_version" not in payload:
        raise ManifestError("missing required manifest field(s): schema_version")

    version = payload["schema_version"]
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ManifestError(
            f"unsupported manifest schema_version {version!r}; this reader speaks "
            + ", ".join(str(item) for item in SUPPORTED_SCHEMA_VERSIONS)
        )

    if version == LEGACY_SCHEMA_VERSION:
        # A v1 file is legacy production and can be nothing else: the only
        # writer that ever produced one was open_run.  Both of these fields
        # postdate it, so a v1 file carrying either was not written by that
        # writer, and reading it as a v1 manifest would be reading a forgery.
        for impossible, why in (
            ("provenance_mode", "postdates v1"),
            ("backfill_sources", "postdates v1, and v1 is never a reconstruction"),
        ):
            if impossible in payload:
                raise ManifestError(
                    f"a v1 manifest may not carry {impossible}: the field {why}. "
                    "Write schema_version 2 instead."
                )
        mode = "production"
        missing = [field for field in LEGACY_REQUIRED_FIELDS if field not in payload]
        if missing:
            raise ManifestError(
                "missing required manifest field(s) for schema_version 1: "
                + ", ".join(missing)
            )
    else:
        missing_core = [field for field in CORE_REQUIRED_FIELDS if field not in payload]
        if missing_core:
            raise ManifestError(
                "missing required manifest field(s): " + ", ".join(missing_core)
            )

        mode = payload["provenance_mode"]
        if mode not in PROVENANCE_MODES:
            raise ManifestError(
                f"provenance_mode must be one of {sorted(PROVENANCE_MODES)}, got {mode!r}"
            )

        missing = [field for field in required_fields(mode) if field not in payload]
        if missing:
            raise ManifestError(
                f"missing required manifest field(s) for provenance_mode {mode!r}: "
                + ", ".join(missing)
            )

        if mode == "production" and "backfill_sources" in payload:
            raise ManifestError(
                "backfill_sources is meaningless on a production manifest: the run "
                "itself is the source. Its presence would suggest the identity was "
                "assembled afterwards."
            )

    if mode == "reconstructed":
        sources = payload["backfill_sources"]
        if (
            not isinstance(sources, list)
            or not sources
            or not all(isinstance(item, str) and item.strip() for item in sources)
        ):
            raise ManifestError(
                "backfill_sources must be a non-empty list of non-empty strings "
                "naming where each reconstructed fact came from"
            )
        if not isinstance(payload["commit"], str) or not payload["commit"].strip():
            raise ManifestError(
                "a reconstructed manifest must name a commit: an unknown commit "
                "accounts for nothing, and this manifest should not have been written"
            )

    if "produced_by" in payload:
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

    # Absent is only reachable in reconstructed mode, where it means the fact
    # was never established.  Present still has to be well typed.
    if "dirty" in payload:
        dirty = payload["dirty"]
        if dirty is not None and not isinstance(dirty, bool):
            raise ManifestError(
                f"dirty must be a bool or null, got {type(dirty).__name__}"
            )

    for text_field in ("started_at", "host"):
        if text_field in payload and not isinstance(payload[text_field], str):
            raise ManifestError(
                f"{text_field} must be a string, got {type(payload[text_field]).__name__}"
            )

    if "cmdline" in payload:
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


def _publish_exclusively(path: Path, text: str) -> None:
    """Write via temp file + fsync + link, so publication *is* the existence check.

    ``os.replace`` cannot fail on an existing destination, so a manifest written
    by a producer between the caller's check and the rename is silently
    displaced — and non-reattribution, the one rule ``open_run`` and
    ``attach_reconstructed_manifest`` share, is exactly the rule that must not
    depend on a window being narrow.  ``os.link`` refuses to overwrite, in the
    same syscall that publishes, so a losing writer raises instead.

    The temp file still carries the atomicity: the destination appears whole or
    not at all, and a crash mid-write leaves only the dot-file behind.  A
    filesystem without hard links fails here rather than degrading to a racy
    write, which is the fail-closed choice for a provenance record.
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
        try:
            os.link(tmp_path, path)
        except FileExistsError as exc:
            raise ManifestError(
                f"{path} already exists; a manifest is never replaced, because "
                "the one already there may be the record written when the run "
                "happened"
            ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
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
    """Claim an empty or not-yet-existing artifact directory, manifest first.

    Call this as the **first** side effect of a producing entry point, before
    creating sub-directories, opening result files, or dispatching workers.  It
    returns only once the manifest is durably on disk and reads back valid; on
    any failure it raises :class:`ManifestError` and leaves no manifest behind.

    A directory that already holds anything — old sequence outputs, a dispatch
    plan, a previous ``run_manifest.json`` — is refused.  Overwriting the
    manifest would be worse than having none: producers here do not clear the
    directory first, so run B's manifest would come to stand over whichever of
    run A's files B never happened to overwrite, and the result is confident,
    plausible, wrong provenance.  Wrong provenance is harder to detect than
    absent provenance, and everything downstream — citation, disposal — trusts
    it.

    v1 therefore has no ``overwrite`` and no resume: a re-run goes to a fresh
    directory.  Run-continuation semantics (what a resumed run inherits, and
    what it may claim about bytes it did not produce) is a separate design
    question, and guessing at it here would bake the answer into a hundred
    directories before anyone chose it.
    """
    directory = Path(output_dir)
    if directory.exists():
        if not directory.is_dir():
            raise ManifestError(
                f"artifact path {directory} exists and is not a directory"
            )
        try:
            occupied = next(directory.iterdir(), None)
        except OSError as exc:
            raise ManifestError(
                f"cannot inspect artifact directory {directory}: {exc}"
            ) from exc
        if occupied is not None:
            raise ManifestError(
                f"artifact directory {directory} is not empty (found {occupied.name}); "
                "a run may only claim an empty or new directory, because a manifest "
                "written over existing files would misattribute them to this run. "
                "Point the run at a fresh directory."
            )
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
        _publish_exclusively(path, text)
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


def build_reconstructed_manifest(
    run_id: str,
    *,
    commit: str,
    backfill_sources: Iterable[str],
    produced_by: str | None = None,
    dirty: bool | None = None,
    started_at: str | None = None,
    host: str | None = None,
    preset: str | None = None,
    detector: str | None = None,
    dataset: str | None = None,
    gpu: str | None = None,
    cmdline: Iterable[str] | None = None,
    claims: Iterable[str] = (),
) -> dict[str, Any]:
    """Assemble a manifest for bytes that were produced before it existed.

    Every optional argument left at ``None`` is **omitted from the payload**
    rather than written as a null, an empty string, or an empty list.  That is
    the whole discipline of this function: a reconstruction states what it can
    source and stays silent about the rest, because a field filled with a
    placeholder reads downstream exactly like a field filled with a fact.

    ``backfill_sources`` must name where the stated facts came from, one entry
    per source, concretely enough for someone else to check them.
    """
    sources = [str(item) for item in backfill_sources]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "provenance_mode": "reconstructed",
        "commit": commit,
        "backfill_sources": sources,
    }
    for key, value in (
        ("produced_by", produced_by),
        ("dirty", dirty),
        ("started_at", started_at),
        ("host", host),
        ("preset", preset),
        ("detector", detector),
        ("dataset", dataset),
        ("gpu", gpu),
    ):
        if value is not None:
            payload[key] = value
    if cmdline is not None:
        payload["cmdline"] = list(cmdline)
    claim_list = list(claims)
    if claim_list:
        payload["claims"] = claim_list
    validate_manifest(payload)
    return payload


def attach_reconstructed_manifest(
    output_dir: str | os.PathLike[str], payload: Mapping[str, Any]
) -> Path:
    """Attach a reconstructed identity to a directory that already holds bytes.

    This is the deliberate mirror image of :func:`open_run`, which refuses any
    directory that is not empty.  The two rules do not contradict each other:
    ``open_run`` refuses because a *production* manifest claims to have been
    written by the run that produced those bytes, and over foreign files that
    claim is false.  A reconstructed manifest makes no such claim — accounting
    for pre-existing bytes is its entire purpose — which is exactly why it must
    be a separate mode and a separate function rather than an ``overwrite=True``
    flag on the first one.

    The one rule both share is non-reattribution: an existing manifest is never
    replaced.  A directory that already carries an identity is not a backfill
    candidate, and overwriting one would let archaeology quietly displace a
    record made at production time.  The check below reports that case in the
    terms the caller needs; the rule itself is enforced one layer down, by
    :func:`_publish_exclusively`, so that a manifest appearing *after* the check
    is refused rather than overwritten.
    """
    directory = Path(output_dir)
    if not directory.is_dir():
        raise ManifestError(f"{directory} is not a directory; nothing to account for")

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(
            "a reconstruction is written at the current schema version "
            f"({SCHEMA_VERSION}); v1 exists only to be read"
        )

    if (directory / MANIFEST_FILENAME).exists():
        raise ManifestError(
            f"{directory} already carries a {MANIFEST_FILENAME}; a reconstruction "
            "never replaces an existing manifest, because the one already there "
            "may be the record written when the run happened"
        )

    if next(directory.iterdir(), None) is None:
        raise ManifestError(
            f"{directory} is empty; there are no bytes here for a reconstructed "
            "manifest to account for. A new run should call open_run() instead."
        )

    validate_manifest(payload)
    mode = provenance_mode_of(payload)
    if mode != "reconstructed":
        raise ManifestError(
            f"attach_reconstructed_manifest refuses a {mode!r} manifest: only a "
            "reconstruction may stand over bytes it did not produce"
        )

    path = directory / MANIFEST_FILENAME
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        _publish_exclusively(path, text)
    except OSError as exc:
        raise ManifestError(f"cannot write manifest {path}: {exc}") from exc

    readback = read_manifest(directory)
    if readback != dict(payload):
        raise ManifestError(f"manifest at {path} did not read back as written")
    return path
