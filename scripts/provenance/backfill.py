"""Which cited artifact directories can honestly be given a manifest, and which cannot.

ADR 021 (AP-4).  The ADR originally described this step as "pay-on-use backfill":
walk the directories the evidence ledger cites and write them a manifest.  That
description was wrong in several ways, and this module exists as the correction.

**1. A cited path is not a run.**  ``results/dual_stability_ablation_20260709/``
is one accounting unit to :mod:`asset_inventory`, but it holds four ablation arms
run twice each.  Writing one manifest at that root would assert that eight runs
were one, which is the same non-reattribution failure ``open_run`` refuses at
production time, arriving by a different door.

**2. The ledger does not name artifact paths.**  It maps
``commit / preset / metrics`` to a *source document*; the literal output path
appears in that document, if anywhere.  Discovery therefore has to follow the
chain — authority row → the document that row **cites as its source** → a path
that document writes **as a path** — rather than grep the ledger and stop.

Both of those bindings are load-bearing, and loosening either one turns
discovery back into a grep with extra steps.  Following every link instead of
the cited sources put a rule document in the chain because a row named the rule
it obeys, and brought an unrelated ``out/signal_study`` in with it.  Reading
paths out of running prose turned the sentence "宣告/runner/results/packet 同一
commit 落地" — four project roles, no path — into ``results/packet``, a
directory this tool then offered to create a manifest in.

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
import json
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
    PRODUCED_BY,
    ManifestError,
    attach_reconstructed_manifest,
    build_reconstructed_manifest,
    provenance_mode_of,
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

# A citation is a *relation*, not a link.  An authority row names where its
# evidence lives — the ledger in its ``Source`` column, the registry in
# ``supporting_declaration`` / ``accepting_review`` — and only that document is
# in the chain.  Following every link instead makes the chain mean "mentioned
# near an authority": the registry links the doc-structure contract to name a
# **rule it obeys**, and following that link imported an unrelated
# ``out/signal_study`` as a candidate.  A rule a row obeys is not a source of
# the row's evidence, and neither is a navigation link to a status snapshot.
SOURCE_RELATIONS = frozenset({"source", "supporting_declaration", "accepting_review"})

_SEPARATOR_CELL = re.compile(r"^:?-{2,}:?$")
_RELATION_FIELD = re.compile(r"^\s*([a-z_]+):\s*(\S.*)$")
_MARKUP = re.compile(r"[*_`]")

# A document reference inside a source-relation value: a Markdown link target,
# or the bare relative path the registry writes (``supporting_declaration:
# ../../modules/.../x.md``).
_DOC_REF = re.compile(
    r"\]\(([^)\s#]+\.md)(?:#[^)]*)?\)|([A-Za-z0-9_.][A-Za-z0-9_./\-]*\.md)"
)

# An asset path counts only where the document writes it **as a path** — inside
# a code fence, an inline code span, or a link target.  In running prose the
# same characters are usually not a path at all: the registry sentence
# "宣告/runner/results/packet 同一 commit 落地" enumerates four project roles,
# and grepping it yielded ``results/packet`` as a directory to write a manifest
# into.  The leading look-behind is the other half of that: without it the
# token could start in the middle of somebody else's path.
_ASSET_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_./\-])(?:"
    + "|".join(ASSET_ROOTS)
    + r")/[A-Za-z0-9_][A-Za-z0-9_.\-/]*"
)
_CODE_SPAN = re.compile(r"`([^`]+)`")
_LINK_TARGET = re.compile(r"\]\(([^)\s]+)\)")
_FENCE = re.compile(r"^\s*(?:```|~~~)")

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

# Directories that already carry an identity of their own get no second, weaker
# one.  Two independent grounds, each established by evidence rather than by
# where a file happens to sit:
#
#  (a) a record in the directory **declares its own authority** — an H2 identity
#      record does this in so many words — so a run manifest beside it would be
#      a competing account of the same bytes;
#  (b) a checksum manifest whose coverage **actually reaches** this directory's
#      bytes, which is decided by reading it.
#
# Depth is not evidence.  ``run/archive/checksums.sha256`` listing four files in
# ``archive/`` proves the closure of ``archive/`` and says nothing about the run
# root or its siblings; treating "a seal exists somewhere below" as "this
# directory is sealed" would refuse ordinary runs for no reason, and would be
# reasoning about identity semantics from filesystem layout.
SEAL_FILE_GLOBS = ("SHA256SUMS*", "*.sha256", "*.sha256sum")

# Keys whose presence at the top level of a JSON record mean the record speaks
# for itself.  ``_latency_profile.json`` and friends carry neither.
AUTHORITY_DECLARING_KEYS = ("authority", "certificate")

# Scan budget, not a rule: a directory's own seal sits at its root or one level
# in, and rglob over an 82 GB tree to answer a question about one directory is
# not affordable.  Whether a seal that *is* found covers this directory is then
# decided by parsing it, never by where it was found.
SEAL_SEARCH_DEPTH = 2

# A record that declares its own authority is read to check that it does; the
# cap keeps that from meaning "parse a 4 MB runtime-inputs dump".
AUTHORITY_RECORD_MAX_BYTES = 1 << 20

# Classification outcomes.  Exactly one of these is writable.
ELIGIBLE = "single_run_reconstructable"
CONTAINER = "multi_run_container"
INSUFFICIENT = "insufficient_identity"
SELF_ATTESTING = "self_attesting_record"
ALREADY_MANIFESTED = "already_manifested"
INVALID_MANIFEST = "invalid_manifest"
NOT_A_RUN_DIRECTORY = "not_a_run_directory"
ABSENT = "absent_from_workspace"
UNSAFE_PATH = "unsafe_path"


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


def source_relation_values(text: str) -> list[str]:
    """The values of the source relations in a document, and nothing else.

    Two shapes, because the two authorities are written differently and neither
    is going to be rewritten for this tool's convenience:

    * the ledger is a Markdown table whose header names a ``Source`` column, so
      the relation is that column's cell on each body row;
    * the registry is a YAML-ish record whose ``supporting_declaration:`` and
      ``accepting_review:`` fields hold the same relation as bare paths.

    A field's continuation lines are deliberately **not** read.  Those lines are
    where a record explains itself ("charter 收尾見 …"), and prose in a
    continuation is prose whatever field it hangs under.  The cost is a source
    that is only reachable from a wrapped line, which is a candidate not found
    rather than a candidate invented.
    """
    values: list[str] = []
    previous: list[str] | None = None
    columns: list[int] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("|") and line.endswith("|") and len(line) > 1:
            cells = [cell.strip() for cell in line[1:-1].split("|")]
            if any(cells) and all(
                _SEPARATOR_CELL.match(cell) for cell in cells if cell
            ):
                columns = [
                    index
                    for index, name in enumerate(previous or [])
                    if _MARKUP.sub("", name).strip().lower() in SOURCE_RELATIONS
                ]
            elif columns:
                values.extend(cells[index] for index in columns if index < len(cells))
            previous = cells
            continue
        previous, columns = None, []
        field = _RELATION_FIELD.match(raw)
        if field and field.group(1) in SOURCE_RELATIONS:
            values.append(field.group(2))
    return values


def authority_chain(repo_root: Path) -> dict[str, str]:
    """The documents a candidate may be discovered in, and why each is in scope.

    Depth one: the two authorities, plus the documents their rows **cite as the
    source of that row's evidence**.  Not every document they link to, and not
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
        authority = repo_root / name
        text = authority.read_text(encoding="utf-8", errors="ignore")
        for value in source_relation_values(text):
            for match in _DOC_REF.finditer(value):
                ref = match.group(1) or match.group(2)
                target = (authority.parent / ref).resolve()
                try:
                    rel = target.relative_to(repo_root).as_posix()
                except ValueError:
                    continue  # outside the repo; not part of the chain
                if target.is_file():
                    chain.setdefault(rel, f"cited as a source by {name}")
    return chain


def explicit_path_spans(text: str) -> list[str]:
    """The regions of a document where a path is written as a path.

    Code fences (a recorded command line: ``--output out/frozen_v2``), inline
    code spans, and link targets.  Running prose is excluded on purpose — see
    :data:`_ASSET_TOKEN`.
    """
    spans: list[str] = []
    fenced = False
    for line in text.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
            continue
        if fenced:
            spans.append(line)
            continue
        spans.extend(match.group(1) for match in _CODE_SPAN.finditer(line))
        spans.extend(match.group(1) for match in _LINK_TARGET.finditer(line))
    return spans


def discover(repo_root: Path) -> dict[str, tuple[str, ...]]:
    """Literal asset paths named in the chain, mapped to the documents naming them."""
    found: dict[str, set[str]] = {}
    for document in authority_chain(repo_root):
        text = (repo_root / document).read_text(encoding="utf-8", errors="ignore")
        for span in explicit_path_spans(text):
            for match in _ASSET_TOKEN.finditer(span):
                token = match.group(0).rstrip("/.,;:`")
                if not token or "/" not in token:
                    continue
                found.setdefault(token, set()).add(document)
    return {path: tuple(sorted(docs)) for path, docs in sorted(found.items())}


def _is_contained(root: Path, target: Path) -> bool:
    """Is ``target`` inside ``root`` and under one of the asset roots?"""
    try:
        rel = target.relative_to(root)
    except ValueError:
        return False
    parts = rel.parts
    return len(parts) >= 1 and parts[0] in ASSET_ROOTS


def _require_containment(root: Path, target: Path, named: str) -> None:
    """Re-check containment at the moment of writing, after resolution.

    The authority documents are trusted, so this is not a defence against them.
    It is a defence against this being a fail-closed *writer*: a token that
    survives discovery reaches ``--write`` as a path to create a file at, and a
    writer should never take a caller's word for where it is pointing.
    """
    if not _is_contained(root, target):
        raise BackfillError(
            f"{named} resolves to {target}, which is outside the asset roots "
            f"({', '.join(ASSET_ROOTS)}) under {root}; refusing to write there"
        )


def _has_run_output(directory: Path) -> bool:
    for marker in RUN_OUTPUT_MARKERS:
        if (directory / marker).is_file():
            return True
    return any(
        next(directory.glob(pattern), None) is not None for pattern in RUN_OUTPUT_GLOBS
    )


def _authority_declaring_records(directory: Path) -> list[str]:
    """JSON records directly here that declare an authority of their own."""
    hits = []
    for item in sorted(directory.glob("*.json")):
        if not item.is_file() or item.stat().st_size > AUTHORITY_RECORD_MAX_BYTES:
            continue
        try:
            payload = json.loads(item.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and any(
            key in payload for key in AUTHORITY_DECLARING_KEYS
        ):
            hits.append(item.name)
    return hits


def _checksum_entries(path: Path) -> list[str] | None:
    """The paths a checksum manifest lists, or None if it cannot be read.

    Handles the two shapes on disk: ``sha256sum`` text lines, and a JSON pack
    with a ``files`` list or a flat ``{path: digest}`` mapping.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    if path.suffix == ".json":
        try:
            payload = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        if isinstance(payload.get("files"), list):
            return [
                str(item["file"])
                for item in payload["files"]
                if isinstance(item, dict) and "file" in item
            ]
        return [str(key) for key in payload]
    entries = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            return None  # not a checksum listing this understands
        entries.append(parts[1].lstrip("*").strip())
    return entries


def _covering_seals(directory: Path) -> list[str]:
    """Checksum manifests that demonstrably cover this directory's own bytes.

    A seal covers ``directory`` when one of the files it lists resolves to a
    file sitting directly in ``directory`` — the seal at the root listing its
    siblings — or to anything outside the seal's own directory, which is a seal
    further in reaching back out.  A seal that lists only its own neighbours
    covers only them, however deep or shallow it sits.

    An unreadable seal is treated as covering.  Refusing to write is the
    reversible mistake; writing into something whose coverage could not be
    established is not.

    Two known imprecisions, both erring toward refusal and neither able to
    produce a false identity:

    * an unreadable seal *nested* one level in is attributed to this directory
      even though a readable one in the same place would not be, so a directory
      can be refused for a seal that does not actually cover it;
    * a seal listing files at this level proves those bytes are digested; it
      does not prove the file *set* is closed, so "adding a file breaks it" is
      the conservative reading rather than a demonstrated fact.
    """
    hits = []
    for depth in range(SEAL_SEARCH_DEPTH):
        prefix = "*/" * depth
        for pattern in SEAL_FILE_GLOBS:
            for seal in sorted(directory.glob(prefix + pattern)):
                if not seal.is_file():
                    continue
                rel = seal.relative_to(directory).as_posix()
                entries = _checksum_entries(seal)
                if entries is None:
                    hits.append(f"{rel} (unreadable, treated as covering)")
                    continue
                home = seal.parent.resolve()
                here = directory.resolve()
                for entry in entries:
                    resolved = (seal.parent / entry).resolve()
                    reaches_out = home != resolved and home not in resolved.parents
                    if resolved.parent == here or reaches_out:
                        hits.append(rel)
                        break
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

# In preference order: the first one a record actually carries is the one
# read, and the one named as the source.
COMMIT_KEYS = ("git_sha", "commit")


def _source_facts(
    directory: Path, rel: str
) -> tuple[dict[str, object], list[str], str]:
    """Establish what can be established, naming the source of each fact.

    Returns ``(facts, sources, reason)``.  ``reason`` is non-empty exactly when
    the facts are insufficient, and says which required fact is missing.

    ``produced_by`` is read only if the record states it.  There is a tempting
    rule available here — this file has a preset and a detector, therefore it
    was an eval — and it is exactly the move this module exists to refuse.
    ``run_meta.txt`` is not a versioned schema and nothing contracts what its
    fields imply, so that rule would be an inference wearing the clothes of an
    observed fact.  ``produced_by`` is optional on a reconstruction precisely so
    that it can be left unsaid.
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

    # Which key was read is part of the source, not an implementation detail:
    # backfill_sources is an audit pointer, and naming git_sha= over a record
    # that wrote commit= sends the next reader to a field that is not there.
    commit_key, commit = "", ""
    for key in COMMIT_KEYS:
        value = meta.get(key, "")
        if value:
            commit_key, commit = key, value
            break
    if not commit or not _SHA.match(commit):
        keys = "/".join(f"{key}=" for key in COMMIT_KEYS)
        return (
            {},
            [],
            f"{source} names no usable commit ({keys}); a reconstructed "
            "manifest without a commit accounts for nothing",
        )

    facts: dict[str, object] = {"commit": commit}
    sources = [f"{source}: {commit_key}="]

    stated_kind = meta.get("produced_by")
    if stated_kind in PRODUCED_BY:
        facts["produced_by"] = stated_kind
        sources.append(f"{source}: produced_by=")

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

    if ".." in Path(rel).parts or Path(rel).is_absolute():
        return make(
            UNSAFE_PATH,
            "the token walks out of the path it names; a citation is a name, not "
            "a traversal, and this tool writes files at the paths it is given",
        )
    if not _is_contained(root.resolve(), (root / rel).resolve()):
        return make(
            UNSAFE_PATH,
            f"resolves outside the asset roots ({', '.join(ASSET_ROOTS)})",
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
        # provenance_mode_of, not payload["provenance_mode"]: a v1 manifest is
        # valid and has no such field. Reading it directly would mean the reader
        # accepts v1 and the consumer crashes on it — which is precisely the
        # transition-window file the v1 compatibility exists for.
        return make(
            ALREADY_MANIFESTED,
            f"already carries a {provenance_mode_of(payload)} manifest",
        )

    declared = _authority_declaring_records(target)
    if declared:
        return make(
            SELF_ATTESTING,
            "already holds a record that declares its own authority "
            f"({', '.join(declared[:4])}); a run manifest beside it would be a "
            "second, weaker account of the same bytes",
        )

    covering = _covering_seals(target)
    if covering:
        return make(
            SELF_ATTESTING,
            "a checksum manifest covers this directory's own bytes "
            f"({', '.join(covering[:4])}); adding a file to a sealed set can "
            "invalidate the seal",
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
    """Write the reconstructed manifest for one eligible candidate.

    The candidate handed in is a *snapshot* — it was classified when the survey
    ran, printed for a human to read, and only then acted on.  Everything it
    asserts is re-established here against the directory as it is now, and any
    divergence is refused rather than reconciled: what the operator approved was
    the surveyed directory, and a directory whose record changed underneath is a
    different one.  Without this the tool will write a manifest naming commit A
    over a run whose ``run_meta.txt`` now says B — reconstructed, sourced,
    plausible, and wrong.

    This narrows the window; it does not abolish it.  The last of it —
    "the manifest appeared between the check and the write" — is not closed by
    checking earlier, and is closed instead by
    :func:`~scripts.provenance.run_manifest.attach_reconstructed_manifest`
    publishing exclusively, so that a losing writer fails rather than replaces.
    """
    root = Path(repo_root).resolve()
    target = (root / candidate.path).resolve()
    _require_containment(root, target, candidate.path)

    if not candidate.writable:
        raise BackfillError(
            f"{candidate.path} is {candidate.classification}, not {ELIGIBLE}; "
            "only a candidate whose required facts all have a named source may "
            "be given a manifest"
        )
    fresh = classify(root, candidate.path, candidate.cited_by)
    if not fresh.writable:
        raise BackfillError(
            f"{candidate.path} is now {fresh.classification}, not {ELIGIBLE}; "
            f"it stopped being writable after the survey ran ({fresh.reason})"
        )
    if fresh.facts != candidate.facts or fresh.sources != candidate.sources:
        raise BackfillError(
            f"{candidate.path} changed between the survey and the write: it was "
            f"{candidate.facts} from {list(candidate.sources)} and is now "
            f"{fresh.facts} from {list(fresh.sources)}; re-run the survey rather "
            "than writing a manifest for a directory nobody surveyed"
        )

    facts = dict(fresh.facts)
    payload = build_reconstructed_manifest(
        Path(candidate.path).name,
        commit=str(facts.pop("commit")),
        backfill_sources=[
            *fresh.sources,
            *(f"cited by {doc}" for doc in fresh.cited_by),
        ],
        **facts,  # type: ignore[arg-type]
    )
    return attach_reconstructed_manifest(target, payload)


_ORDER = (
    ELIGIBLE,
    INVALID_MANIFEST,
    CONTAINER,
    SELF_ATTESTING,
    INSUFFICIENT,
    ALREADY_MANIFESTED,
    NOT_A_RUN_DIRECTORY,
    UNSAFE_PATH,
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
