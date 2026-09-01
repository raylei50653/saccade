"""Which cited artifact directories can honestly be given a manifest, and which cannot.

ADR 021 (AP-4).  The ADR originally described this step as "pay-on-use backfill":
walk the directories the evidence ledger cites and write them a manifest.  That
description was wrong in three ways, and this module exists as the correction.

**1. A cited path is not a run.**  ``results/dual_stability_ablation_20260709/``
is one accounting unit to :mod:`asset_inventory`, but it holds four ablation arms
run twice each.  Writing one manifest at that root would assert that eight runs
were one, which is the same non-reattribution failure ``open_run`` refuses at
production time, arriving by a different door.

**2. The ledger does not name artifact paths.**  It maps
``commit / preset / metrics`` to a *source document*; the literal output path
appears in that document, if anywhere.  Discovery therefore has to follow the
chain — authority → linked source doc → literal path — rather than grep the
ledger and stop.

**3. "Leave the unknown fields empty" is not available.**  A manifest's required
fields are required because a *producing* run knows all of them.  Reconstruction
knows almost none, and filling them with the current clock or an empty string
yields a manifest that passes validation and states things that are false.  So
the schema grew a mode instead (``provenance_mode``, schema v2), and the rule
here is: **a fact with no named source is not written, and a candidate whose
required facts cannot be sourced is not backfilled at all.**  It stays cited and
unmanifested, which is an accurate description of it.

The consequence is that this tool is expected to write nothing on most runs, and
that a candidate landing in ``insufficient_identity`` is a result rather than a
failure.  Nothing here is allowed to move a directory into the writable class in
order to make :mod:`asset_inventory`'s ``manifested`` count go up; that count is
a smoke signal for whether AP-2 is live, not a coverage metric to optimise.

Selection comes from the authority chain.  **Facts come from inside the
directory.**  That split is deliberate: a document that names a path in prose and
a commit in a table has bound them only in a reader's head, and this tool has no
way to check that binding.  A metadata file written by the run, sitting in the
run's own directory, is a binding a machine can check.

Disposal is AP-5 and is not here.  This module deletes nothing and recommends
deleting nothing.
"""

# status: stable

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve()
if str(_HERE.parents[2]) not in sys.path:
    sys.path.insert(0, str(_HERE.parents[2]))

from scripts.provenance.run_manifest import (  # noqa: E402
    MANIFEST_FILENAME,
    ManifestError,
    attach_reconstructed_manifest,
    build_reconstructed_manifest,
    read_manifest,
)

# The two documents that decide whether an artifact is worth accounting for.
# Neither is a list of paths; both are entry points into a chain (see module
# docstring).  A missing one is fail-closed: an empty authority set would make
# every candidate list trivially correct and trivially useless.
AUTHORITIES = (
    "docs/research/evidence_ledger.md",
    "docs/research/contracts/claim_state_registry.md",
)

ASSET_ROOTS = ("runs", "results", "out", "output")

_ASSET_TOKEN = re.compile(
    r"\b(?:" + "|".join(ASSET_ROOTS) + r")/[A-Za-z0-9_][A-Za-z0-9_.\-/]*"
)
_MD_LINK = re.compile(r"\]\(([^)\s#]+\.md)(?:#[^)]*)?\)")

# Files that mean "a run wrote its output directly here".  Used only to tell a
# run apart from a directory of runs; never to infer what the run was.
RUN_OUTPUT_MARKERS = (
    "run_meta.txt",
    "_fps_summary.txt",
    "_latency_profile.json",
    "_global_id_map.txt",
    MANIFEST_FILENAME,
)
RUN_OUTPUT_GLOBS = ("MOT17-*.txt",)

# Directories that already carry an identity of their own.  Adding a second,
# weaker one is refused for two separate reasons, and either alone is enough:
# a checksum file means the directory's contents are digested somewhere, so
# adding a file can invalidate that seal; an H2 identity record already
# declares its own authority, and a run manifest sitting beside it would be a
# competing account of the same bytes.
SEAL_MARKER_GLOBS = ("SHA256SUMS*", "*.sha256", "*.sha256sum", "checksums.sha256")
IDENTITY_RECORD_MARKERS = (
    "runtime_inputs.json",
    "behavior_probe.json",
    "runtime_identity.json",
    "layer_p.json",
)

# Classification outcomes.  Exactly one of these is writable.
ELIGIBLE = "single_run_reconstructable"
CONTAINER = "multi_run_container"
INSUFFICIENT = "insufficient_identity"
SELF_ATTESTING = "self_attesting_record"
ALREADY_MANIFESTED = "already_manifested"
INVALID_MANIFEST = "invalid_manifest"
NOT_A_RUN_DIRECTORY = "not_a_run_directory"
ABSENT = "absent_from_workspace"


class BackfillError(RuntimeError):
    """A precondition of the backfill survey does not hold."""


@dataclass(frozen=True)
class Candidate:
    """One literal asset path named somewhere in the authority chain."""

    path: str
    classification: str
    reason: str
    cited_by: tuple[str, ...] = ()
    facts: dict[str, object] = field(default_factory=dict)
    sources: tuple[str, ...] = ()

    @property
    def writable(self) -> bool:
        return self.classification == ELIGIBLE


def authority_chain(repo_root: Path) -> dict[str, str]:
    """The documents a candidate may be discovered in, and why each is in scope.

    Depth one: the two authorities, plus the documents they link to.  Not
    transitive — a document two hops out was cited by something that was itself
    only cited, and treating that as authority would eventually pull in the
    whole corpus and make "cited by an authority" mean nothing.
    """
    chain: dict[str, str] = {}
    for name in AUTHORITIES:
        authority = repo_root / name
        if not authority.is_file():
            raise BackfillError(
                f"authority document {name} is missing; the candidate set is "
                "defined by the authority chain and cannot be built without it"
            )
        chain[name] = "authority"

    for name in AUTHORITIES:
        text = (repo_root / name).read_text(encoding="utf-8", errors="ignore")
        for match in _MD_LINK.finditer(text):
            target = ((repo_root / name).parent / match.group(1)).resolve()
            try:
                rel = target.relative_to(repo_root).as_posix()
            except ValueError:
                continue  # outside the repo; not part of the chain
            if target.is_file():
                chain.setdefault(rel, f"linked from {name}")
    return chain


def discover(repo_root: Path) -> dict[str, tuple[str, ...]]:
    """Literal asset paths named in the chain, mapped to the documents naming them."""
    found: dict[str, set[str]] = {}
    for document in authority_chain(repo_root):
        text = (repo_root / document).read_text(encoding="utf-8", errors="ignore")
        for match in _ASSET_TOKEN.finditer(text):
            token = match.group(0).rstrip("/.,;:`")
            if not token or "/" not in token:
                continue
            found.setdefault(token, set()).add(document)
    return {path: tuple(sorted(docs)) for path, docs in sorted(found.items())}


def _has_run_output(directory: Path) -> bool:
    for marker in RUN_OUTPUT_MARKERS:
        if (directory / marker).is_file():
            return True
    return any(
        next(directory.glob(pattern), None) is not None for pattern in RUN_OUTPUT_GLOBS
    )


def _seal_markers(directory: Path) -> list[str]:
    """Seals belonging to *this* directory, not to something buried under it.

    Depth-limited to two levels on purpose.  A seal three levels down belongs to
    a descendant run, and a manifest written up here would neither touch it nor
    be covered by it; treating that as a refusal would misdescribe a plain
    container as a sealed record.  (It would also mean rglob-ing an 82 GB tree
    to answer a question about one directory.)
    """
    hits = []
    for pattern in SEAL_MARKER_GLOBS:
        for depth in (pattern, f"*/{pattern}"):
            hits.extend(
                item.relative_to(directory).as_posix()
                for item in sorted(directory.glob(depth))
                if item.is_file()
            )
    hits.extend(
        marker for marker in IDENTITY_RECORD_MARKERS if (directory / marker).is_file()
    )
    return sorted(set(hits))


def parse_run_meta(path: Path) -> dict[str, str]:
    """Read the ``key=value`` metadata a run driver leaves beside its output.

    Tolerant of the two shapes actually on disk: one pair per line, and several
    space-separated pairs on one line (``preset=... detector=... ...``).
    """
    facts: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        parts = line.split() if line.count("=") > 1 and " " in line else [line]
        for part in parts:
            if "=" not in part:
                continue
            key, _, value = part.partition("=")
            key, value = key.strip(), value.strip()
            if key and value:
                facts.setdefault(key, value)
    return facts


_SHA = re.compile(r"^[0-9a-f]{7,40}$")


def _source_facts(
    directory: Path, rel: str
) -> tuple[dict[str, object], list[str], str]:
    """Establish what can be established, naming the source of each fact.

    Returns ``(facts, sources, reason)``.  ``reason`` is non-empty exactly when
    the facts are insufficient, and says which required fact is missing.
    """
    meta_path = directory / "run_meta.txt"
    if not meta_path.is_file():
        return (
            {},
            [],
            "no in-directory run record (run_meta.txt) binds this directory to a "
            "commit; the citing document states a commit in prose, which is a "
            "binding only a reader can make, not one this tool can check",
        )

    meta = parse_run_meta(meta_path)
    source = f"{rel}/run_meta.txt"

    commit = meta.get("git_sha") or meta.get("commit")
    if not commit or not _SHA.match(commit):
        return (
            {},
            [],
            f"{source} names no usable commit (git_sha=/commit=); a reconstructed "
            "manifest without a commit accounts for nothing",
        )

    # produced_by is derived by a stated rule, not guessed: run_meta.txt records
    # a preset and a detector only for an evaluation invocation.  The rule is
    # written into backfill_sources so a reviewer can check it rather than
    # trust it.
    if "preset" in meta and "detector" in meta:
        produced_by = "eval"
        rule = f"{source}: preset= and detector= present => produced_by=eval"
    else:
        return (
            {},
            [],
            f"{source} does not identify what kind of run this was; produced_by "
            "is a closed vocabulary and may not be guessed",
        )

    facts: dict[str, object] = {"commit": commit, "produced_by": produced_by}
    sources = [f"{source}: git_sha=", rule]
    for key, manifest_key in (
        ("host", "host"),
        ("gpu", "gpu"),
        ("preset", "preset"),
        ("detector", "detector"),
        ("date", "started_at"),
    ):
        if key in meta:
            facts[manifest_key] = meta[key]
            sources.append(f"{source}: {key}=")
    # dirty and cmdline are intentionally absent: nothing on disk records them,
    # and absence is how this schema says "not established".
    return facts, sources, ""


def classify(repo_root: Path, rel: str, cited_by: tuple[str, ...]) -> Candidate:
    root = Path(repo_root)
    target = root / rel

    def make(classification: str, reason: str, **extra: object) -> Candidate:
        return Candidate(
            path=rel,
            classification=classification,
            reason=reason,
            cited_by=cited_by,
            **extra,  # type: ignore[arg-type]
        )

    if not target.exists():
        return make(
            ABSENT,
            "named by the authority chain but not present in this workspace "
            "(the asset roots are gitignored, so this is expected on a clone)",
        )
    if not target.is_dir():
        return make(
            NOT_A_RUN_DIRECTORY,
            "the chain names a file, not a run directory; a file cannot carry a "
            "manifest, and promoting it to its parent would be this tool "
            "inventing a citation the document did not make",
        )

    if (target / MANIFEST_FILENAME).exists():
        try:
            payload = read_manifest(target)
        except ManifestError as exc:
            return make(INVALID_MANIFEST, str(exc))
        return make(
            ALREADY_MANIFESTED,
            f"already carries a {payload['provenance_mode']} manifest",
        )

    seals = _seal_markers(target)
    if seals:
        return make(
            SELF_ATTESTING,
            "already carries its own identity/attestation record "
            f"({', '.join(seals[:4])}); adding a run manifest could invalidate a "
            "self-seal, and would in any case put two competing accounts of the "
            "same bytes side by side",
        )

    children = [child for child in sorted(target.iterdir()) if child.is_dir()]
    run_children = [child for child in children if _has_run_output(child)]
    if len(run_children) >= 2:
        return make(
            CONTAINER,
            f"holds {len(run_children)} directories that each contain run output "
            f"({', '.join(child.name for child in run_children[:4])}...); one "
            "manifest here would assert that several runs were one",
        )
    if len(children) >= 2 and not _has_run_output(target):
        return make(
            CONTAINER,
            f"nothing is written at this level; the {len(children)} directories "
            "below it are where runs live",
        )
    if not _has_run_output(target):
        return make(
            INSUFFICIENT,
            "no recognisable run output directly at this level, so this tool "
            "cannot even establish that it is one run",
        )

    facts, sources, reason = _source_facts(target, rel)
    if reason:
        return make(INSUFFICIENT, reason)
    return make(
        ELIGIBLE,
        "single run shape, and every required fact has a named in-directory source",
        facts=facts,
        sources=tuple(sources),
    )


def survey(repo_root: str | os.PathLike[str]) -> tuple[Candidate, ...]:
    root = Path(repo_root).resolve()
    return tuple(
        classify(root, rel, cited_by) for rel, cited_by in discover(root).items()
    )


def backfill(repo_root: str | os.PathLike[str], candidate: Candidate) -> Path:
    """Write the reconstructed manifest for one eligible candidate."""
    if not candidate.writable:
        raise BackfillError(
            f"{candidate.path} is {candidate.classification}, not {ELIGIBLE}; "
            "only a candidate whose required facts all have a named source may "
            "be given a manifest"
        )
    facts = dict(candidate.facts)
    payload = build_reconstructed_manifest(
        Path(candidate.path).name,
        produced_by=str(facts.pop("produced_by")),
        commit=str(facts.pop("commit")),
        backfill_sources=[
            *candidate.sources,
            *(f"cited by {doc}" for doc in candidate.cited_by),
        ],
        **facts,  # type: ignore[arg-type]
    )
    return attach_reconstructed_manifest(Path(repo_root) / candidate.path, payload)


_ORDER = (
    ELIGIBLE,
    INVALID_MANIFEST,
    CONTAINER,
    SELF_ATTESTING,
    INSUFFICIENT,
    ALREADY_MANIFESTED,
    NOT_A_RUN_DIRECTORY,
    ABSENT,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Survey which cited artifact directories can be given a reconstructed "
            "manifest. Reports by default; --write is required to write anything."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"Attach manifests to the {ELIGIBLE} candidates. Nothing else is "
        "touched, and no directory is ever deleted or modified otherwise.",
    )
    args = parser.parse_args(argv)

    root = args.repo_root.resolve()
    try:
        candidates = survey(root)
    except BackfillError as exc:
        print(f"backfill: {exc}", file=sys.stderr)
        return 2

    by_class: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_class.setdefault(candidate.classification, []).append(candidate)

    print(f"backfill survey: {len(candidates)} candidate(s) from the authority chain")
    for classification in _ORDER:
        group = by_class.get(classification, [])
        if not group:
            continue
        print(f"\n{classification} ({len(group)}):")
        for candidate in group:
            print(f"  {candidate.path}")
            print(f"    {candidate.reason}")

    written = []
    if args.write:
        for candidate in by_class.get(ELIGIBLE, []):
            try:
                written.append(backfill(root, candidate))
            except (BackfillError, ManifestError) as exc:
                print(f"backfill: {candidate.path}: {exc}", file=sys.stderr)
                return 2
        print(f"\nbackfill: wrote {len(written)} reconstructed manifest(s)")
        for path in written:
            print(f"  {path.relative_to(root)}")
    elif by_class.get(ELIGIBLE):
        print("\n(dry run — pass --write to attach these manifests)")

    broken = by_class.get(INVALID_MANIFEST, [])
    if broken:
        print(
            f"backfill: {len(broken)} cited directory/ies carry a manifest that "
            "does not validate; that is a broken producer, not a backfill task",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
