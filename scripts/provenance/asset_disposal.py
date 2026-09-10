"""Workspace-local projection of which accounted-for units are disposal candidates.

ADR 021 (AP-5).  Turns the AP-3 inventory plus existing provenance / citation
facts into a list of assets worth an owner's review.  That list is not a
deletion verdict, not an approval, and not authorization to remove anything.

**Candidate ≠ deletion.**  A candidate is an inventory unit that current
mechanical conditions say is worth handing to an owner.  Nothing this module
emits means safe to delete, approved for deletion, disposable by default, or
auto-delete eligible.  Owner approval is a separate authorized action this
tool does not record and does not perform; absence of that action is not
approval.

**This tool deletes nothing.**  There is no ``rm``, no trash, no age-based
cleanup, and no unattended deletion job.  A real deletion flow, if one is
ever added, is a different authorized action sitting on the other side of
the approval boundary.

**This is a projection, not a committed authority.**  Same reason as AP-3:
the asset roots are gitignored, so a clean clone holds none of them.
Committing the rendered view would make one machine's reading of 82 GB look
like a repository fact.  So:

* ``--emit`` renders for a human, to a **gitignored** path outside ``docs/``.
  Emitting into ``docs/`` is refused outright (``build_master_map`` rglobs
  that tree).
* ``--check`` validates rather than compares.  Zero candidates on a clean
  clone is the correct answer.  An invalid manifest still fails closed —
  a broken producer is not a candidate, and this tool must not walk past it.

The predicate consumes AP-3 facts (unit, cited, manifest_state, orphan) and
AP-4 evidence (self-attesting records, covering seals, multi-run container
shape).  It does not invent a second asset-state taxonomy, and it does not
write back into inventory classification.

Age is a filter on top of those facts, never a substitute.  It is the
inventory unit directory's own POSIX ``st_mtime``, compared to an injected
UTC clock, in whole days, against ``MIN_AGE_DAYS``.  mtime, directory name,
size, and "looks like an old experiment" are not candidate conditions on
their own.
"""

# status: stable

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve()
if str(_HERE.parents[2]) not in sys.path:
    sys.path.insert(0, str(_HERE.parents[2]))

from scripts.provenance.asset_inventory import (  # noqa: E402
    ASSET_ROOTS,
    InventoryError,
    Unit,
    invalid_manifests,
    scan,
)
from scripts.provenance.backfill import (  # noqa: E402
    multi_run_container_evidence,
    self_attesting_evidence,
)

DEFAULT_OUTPUT = Path(".provenance/asset_disposal_candidates.generated.md")

# Policy constant.  Changing it is a spec change, not a CLI flag; the tool
# does not offer a switch that loosens the threshold.
MIN_AGE_DAYS = 90

KNOWN_MANIFEST_STATES = frozenset({"valid", "invalid", "absent"})

BLOCK_UNKNOWN_STATE = "unknown_manifest_state"
BLOCK_INVALID = "invalid_manifest"
BLOCK_CITED = "cited"
BLOCK_MANIFESTED = "manifested"
BLOCK_NOT_ORPHAN = "not_orphan"
BLOCK_UNSAFE_PATH = "unsafe_path"
BLOCK_UNREADABLE = "unreadable"
BLOCK_NOT_DIRECTORY = "not_a_directory"
BLOCK_MTIME_FUTURE = "mtime_in_future"
BLOCK_AGE = "age_below_threshold"
BLOCK_SELF_ATTESTING = "self_attesting_record"
BLOCK_CONTAINER = "unit_granularity_unclear"


class DisposalError(RuntimeError):
    """The candidate set cannot be derived from this workspace."""


@dataclass(frozen=True)
class Candidate:
    """One unit that existing mechanical conditions say is worth owner review.

    Deliberately not a verdict.  There is no approval field: storing one here
    would make this projection the owner of a decision it is not allowed to
    make.  Absence of approval is not encoded, because encoding it would be
    read as a state.
    """

    path: str
    root: str
    age_days: int


@dataclass(frozen=True)
class Evaluation:
    """The derivation for one unit: candidate or the reasons it is not."""

    path: str
    candidate: bool
    age_days: int | None
    blocked_by: tuple[str, ...]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_as_of(value: str) -> datetime:
    """Parse an ISO-8601 timestamp into an aware UTC datetime.

    A naive value is taken as UTC rather than as local time — local time
    would make the same ``--as-of`` produce different candidate sets on
    two machines.
    """
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise DisposalError(f"invalid --as-of timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _require_aware(now: datetime) -> None:
    if now.tzinfo is None:
        raise DisposalError(
            "candidate derivation requires a timezone-aware clock; "
            "a naive datetime would silently shift the age threshold"
        )


def _directory_for(unit: Unit, repo_root: Path) -> Path | None:
    """The unit directory, or None if the path is not a safe inventory unit."""
    rel = Path(unit.path)
    if rel.is_absolute() or ".." in rel.parts:
        return None
    if unit.root not in ASSET_ROOTS or not rel.parts or rel.parts[0] != unit.root:
        return None
    return repo_root / rel


def evaluate_unit(unit: Unit, repo_root: Path, now: datetime) -> Evaluation:
    """Apply the AP-5 predicate to one inventory unit.

    Blocking conditions are collected rather than short-circuited, so a test
    can assert the specific fail-closed reason rather than only "not a
    candidate".  Filesystem probes that cannot run (unreadable / not a
    directory / unsafe path) skip the probes they cannot make; those already
    block.
    """
    _require_aware(now)
    blocks: list[str] = []

    if unit.manifest_state not in KNOWN_MANIFEST_STATES:
        blocks.append(BLOCK_UNKNOWN_STATE)
    if unit.manifest_state == "invalid":
        blocks.append(BLOCK_INVALID)
    if unit.cited:
        blocks.append(BLOCK_CITED)
    if unit.manifest_state == "valid":
        blocks.append(BLOCK_MANIFESTED)
    if not unit.orphan:
        blocks.append(BLOCK_NOT_ORPHAN)

    directory = _directory_for(unit, repo_root)
    if directory is None:
        blocks.append(BLOCK_UNSAFE_PATH)
        return Evaluation(unit.path, False, None, tuple(blocks))

    try:
        st = directory.stat()
    except OSError:
        blocks.append(BLOCK_UNREADABLE)
        return Evaluation(unit.path, False, None, tuple(blocks))

    if not directory.is_dir():
        blocks.append(BLOCK_NOT_DIRECTORY)
        return Evaluation(unit.path, False, None, tuple(blocks))

    mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
    if mtime > now:
        blocks.append(BLOCK_MTIME_FUTURE)
        age_days: int | None = None
    else:
        age_days = (now - mtime).days
        if age_days < MIN_AGE_DAYS:
            blocks.append(BLOCK_AGE)

    try:
        if self_attesting_evidence(directory):
            blocks.append(BLOCK_SELF_ATTESTING)
    except OSError:
        blocks.append(BLOCK_UNREADABLE)

    try:
        if multi_run_container_evidence(directory):
            blocks.append(BLOCK_CONTAINER)
    except OSError:
        blocks.append(BLOCK_UNREADABLE)

    return Evaluation(
        path=unit.path,
        candidate=not blocks,
        age_days=age_days,
        blocked_by=tuple(blocks),
    )


def derive_candidates(
    units: tuple[Unit, ...],
    repo_root: str | os.PathLike[str],
    now: datetime,
) -> tuple[Candidate, ...]:
    """Pure projection: inventory units × filesystem facts × clock → candidates.

    Does not read a previous generated view.  Does not modify ``units``.
    Nested children never appear: they are not inventory units, and this
    function does not invent them.
    """
    root = Path(repo_root)
    candidates: list[Candidate] = []
    for unit in units:
        evaluation = evaluate_unit(unit, root, now)
        if evaluation.candidate:
            assert evaluation.age_days is not None
            candidates.append(
                Candidate(
                    path=unit.path,
                    root=unit.root,
                    age_days=evaluation.age_days,
                )
            )
    return tuple(candidates)


def render(
    candidates: tuple[Candidate, ...],
    *,
    repo_root: Path,
    now: datetime,
    unit_count: int,
) -> str:
    generated_at = now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "<!-- Generated by scripts/provenance/asset_disposal.py; "
        "workspace-local projection, not committed, not an authority, "
        "not a deletion verdict. -->",
        "# Asset disposal candidates (workspace projection)",
        "",
        f"Workspace: `{repo_root}`",
        f"As of (UTC): `{generated_at}`",
        f"Age policy: directory `st_mtime` ≥ **{MIN_AGE_DAYS}** whole days.",
        "",
        "This file is a reading of the current workspace, not a record. "
        "Re-run the generator against the workspace as it is now; a copy "
        "left on disk is not current authority, and this tool never reads "
        "it back as an input.",
        "",
        "**A candidate is not a deletion verdict.** It is an inventory unit "
        "that existing mechanical conditions say is worth owner review. "
        "Nothing listed here is approved, and owner approval is a separate "
        "authorized action this tool does not record and does not perform. "
        "Absence of that action is not approval. This tool deletes nothing.",
        "",
        "Inputs are AP-3 inventory facts (unit = immediate child of "
        f"{', '.join(ASSET_ROOTS)}; `orphan` = not cited and no manifest at "
        "all) plus AP-4 self-attesting / covering-seal evidence and "
        "multi-run container shape. Invalid manifests, cited units, "
        "manifested units, sealed records, and units whose granularity is "
        "unclear are not candidates, regardless of age. Age never stands "
        "in for provenance or citation.",
        "",
        "Citation is a literal substring match against tracked, "
        "non-`*.generated.md` documents, so this file cannot cite its own "
        "subjects — even if it were committed, the `*.generated.md` suffix "
        "keeps it out of the corpus.",
        "",
        "## Summary",
        "",
        "| | |",
        "|:--|--:|",
        f"| inventory units | {unit_count} |",
        f"| candidates for owner review | {len(candidates)} |",
        "",
        "## Candidates",
        "",
    ]
    if not candidates:
        lines += [
            "None. Zero candidates is a correct answer on a clean clone, "
            "and also when every unit is cited, manifested, too young, "
            "sealed, or otherwise blocked.",
            "",
        ]
        return "\n".join(lines)

    lines += [
        "| Unit | Root | Age (days) |",
        "|:--|:--|--:|",
    ]
    for candidate in candidates:
        lines.append(
            f"| `{candidate.path}` | {candidate.root} | {candidate.age_days} |"
        )
    lines.append("")
    return "\n".join(lines)


def _refuse_docs(target: Path, repo_root: Path) -> str | None:
    docs_root = (repo_root / "docs").resolve()
    if target == docs_root or docs_root in target.parents:
        return (
            f"asset disposal: refusing to emit into {docs_root} — build_master_map "
            "collects documents with rglob rather than git, so an untracked view "
            "there fails the checked-in master map on this machine while CI stays "
            f"green (ADR 021 §3 AP-5). Emit outside docs/, e.g. {DEFAULT_OUTPUT}."
        )
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Project which artifact directories are disposal candidates for "
            "owner review. This tool does not delete, approve, or authorize."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="UTC ISO-8601 clock for age (default: now). Naive values are UTC.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail closed on manifests that exist but are not valid. An empty "
        "candidate set (clean clone, no local assets) is a correct answer. "
        "Does not delete anything.",
    )
    parser.add_argument(
        "--emit",
        type=Path,
        nargs="?",
        const=DEFAULT_OUTPUT,
        help=f"Render the view to a gitignored path (default: {DEFAULT_OUTPUT}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()

    try:
        now = parse_as_of(args.as_of) if args.as_of else utc_now()
    except DisposalError as exc:
        print(f"asset disposal: {exc}", file=sys.stderr)
        return 2

    try:
        units = scan(root)
    except InventoryError as exc:
        print(f"asset disposal: {exc}", file=sys.stderr)
        return 2

    candidates = derive_candidates(units, root, now)

    if args.emit is not None:
        target = args.emit if args.emit.is_absolute() else root / args.emit
        target = target.resolve()
        refusal = _refuse_docs(target, root)
        if refusal is not None:
            print(refusal, file=sys.stderr)
            return 2
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            render(
                candidates,
                repo_root=root,
                now=now,
                unit_count=len(units),
            ),
            encoding="utf-8",
        )
        print(f"asset disposal: wrote {target}")

    broken = invalid_manifests(units)
    if broken:
        print(
            f"asset disposal: {len(broken)} invalid manifest(s) — a manifest that "
            "exists but does not validate is a broken producer, not a candidate:",
            file=sys.stderr,
        )
        for unit in broken:
            print(f"  {unit.path}: {unit.detail}", file=sys.stderr)
        return 1

    print(
        f"asset disposal: {len(candidates)} candidate(s) for owner review "
        f"({len(units)} unit(s); this tool deletes nothing)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
