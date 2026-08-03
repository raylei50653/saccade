#!/usr/bin/env python3
"""Corpus checker for committed H2 Layer-M evidence roots.

A new checker rather than a change to `check_h0_phase_a_archives.py`: that tool
keeps verifying the frozen H0 v1 corpus under the v1 schema and never sees an H2
root (§ 9 item 3), which is why the evidence prefix is `h2_measure_` in the first
place.

This module is the canonical corpus admission owner. Archive verification is a
different question with a different answer: `verify_h2_measurement` decides
whether one root is internally consistent, and a root can be perfectly
consistent and still not belong to this corpus.

Four things live here because § C3.1 and § C3.5.1 put them here, and nowhere
else in the unit enforces them:

  * **classification.** Every root is exactly one of `complete`, `envelope`,
    `unterminated` (§ C3.5.1's three verify classes) or `inadmissible` (step 4 —
    a refused admission gate, which is Layer-P class and never a consumed
    attempt). "Verifies" means the class's own integrity condition: a terminal-4
    attempt is never required to produce the artifacts its failure is defined by
    the absence of.
  * **root identity.** The `F64` in a Phase-B root name is recomputed from the
    recorded freeze record, so two attempts cannot share a root even at a
    byte-identical head.
  * **`prior_attempts`.** Complete, ordered, every named root present and
    verified — and the § C3.5 ban: a terminal 2 or 3 (or an unterminated attempt
    whose survivors already show one) may not be re-attempted against the same
    bound measurement surface, and a changed surface is admissible only with a
    named defect repair in H0 § 6's own repair vocabulary.
  * **execution-domain admission.** Every consumed attempt was consumed under
    the controlled host's authorization ledger, and not some other one. A run
    against a disposable ledger — a rehearsal — produces an archive of exactly
    the canonical shape whose every internal binding holds, so nothing inside
    the root can refuse it and the corpus must.

An empty corpus passes: a checker that failed on an empty corpus would be
reporting the absence of a measurement as a defect in the archive.

Usage:
  uv run python scripts/tools/check_h2_measure_archives.py
"""
# status: stable

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = REPO_ROOT / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import h2_measurement_evidence as evidence  # noqa: E402
import h2_terminal_partition as partition  # noqa: E402
import verify_h2_execution as successor_verifier  # noqa: E402
import verify_h2_measurement as verifier  # noqa: E402

EVIDENCE_ROOT = REPO_ROOT / evidence.EVIDENCE_REL
CONTROLLED_HOST_DOMAIN_PATH = (
    REPO_ROOT / "docs/research/contracts/h2_controlled_host_execution_domain_v1.json"
)

# Every semantic name below is imported, never restated. Which terminals are
# properties of the measurement surface, and which repairs may reopen one, decide
# what a successor attempt is allowed to be — ruler facts, and this file is
# `plumbing_only`, so a copy here could be edited without moving an axis (§ C3.9).
CONSUMED_CLASSES = partition.VERIFY_CLASSES
INADMISSIBLE = partition.INADMISSIBLE_CLASS
SURFACE_TERMINALS = partition.SURFACE_BAN_TERMINALS
REPAIR_VOCABULARY = partition.REPAIR_VOCABULARY

# The § C3.5.1 classification itself lives with the verifier, which needs it to
# verify a prior attempt in its own class before the admission gate that binds it
# can be recomputed. This is where it is applied to the corpus.
classify = verifier.classify


# The chain rules are the verifier's, because § C3.6(e) makes them part of one
# root's admissibility: a root whose predecessors are incomplete had no right to
# consume `S_B`, and that verdict cannot differ depending on whether it was
# reached through the corpus or through the root. This module raises the same
# error type and adds only what is genuinely corpus-wide — the § C3.5 ban, which
# needs each predecessor's verified report.
CorpusError = verifier.CorpusError


class Attempt(NamedTuple):
    root: Path
    name: evidence.RootName
    verify_class: str
    report: dict[str, Any]
    freeze: dict[str, Any]

    @property
    def terminal(self) -> str | None:
        return self.report.get("terminal")

    @property
    def surface(self) -> str | None:
        value = self.freeze.get("measurement_surface_digest")
        return value if isinstance(value, str) else None

    @property
    def prior_attempts(self) -> tuple[str, ...]:
        value = self.freeze.get("prior_attempts", ())
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            raise CorpusError(
                f"{self.root.name}: prior_attempts is not a list of roots"
            )
        return tuple(value)

    @property
    def phase_a_evidence_root(self) -> str | None:
        section = self.freeze.get("phase_a_evidence")
        if not isinstance(section, Mapping):
            return None
        value = section.get("evidence_root")
        return value if isinstance(value, str) else None

    @property
    def bans_re_attempt(self) -> bool:
        if self.terminal in SURFACE_TERMINALS:
            return True
        # § C3.5.1's kill-switch: an attempt that died before it could record a
        # terminal still carries the ban if what survived already shows one.
        return bool(
            self.report.get("perturbation_observed")
            or self.report.get("invalid_packet_observed")
        )


class SuccessorAttempt(NamedTuple):
    """One admitted four-artifact successor execution.

    The successor archive deliberately has no legacy freeze, root-name identity
    or verify class.  Its identity and terminal live in the result and the
    independently reproduced verification record instead.
    """

    root: Path
    result: dict[str, Any]
    verification: dict[str, Any]

    @property
    def verify_class(self) -> str:
        return "successor"

    @property
    def terminal(self) -> str | None:
        value = self.result.get("terminal")
        return value if isinstance(value, str) else None


def controlled_host_execution_domain() -> dict[str, Any]:
    """The one execution domain the canonical corpus admits attempts from."""
    return evidence.load_document(
        CONTROLLED_HOST_DOMAIN_PATH.parent,
        CONTROLLED_HOST_DOMAIN_PATH.name,
        schema=evidence.AUTHORIZATION_DOMAIN_SCHEMA,
    )


def execution_domain_admission_reasons(
    root: Path, verify_class: str, phase: str
) -> tuple[str, ...]:
    """Why `root` is not an attempt of the controlled host, if it is not.

    The reusable policy predicate behind `check_corpus`, kept separate because
    more than one reader decides what the corpus contains and they must not
    answer differently. Two archived objects are compared — the tracked anchor
    and the root's own record — so the verdict is the same on every host, which
    is the property the 2026-07-29 verifier repair established and this must
    not undo.

    Formatting is not identity: both sides are parsed, then their member sets
    and values are required to be equal. A reordered or re-serialized document
    with the same content is the same domain.

    Scope. This is a provenance/admission guard, not an authority proof. It
    refuses an attempt consumed under some other ledger — a rehearsal, another
    operator, another host — including one that is perfectly self-consistent and
    that the archive verifier therefore accepts. It cannot refuse a forgery: an
    author able to rewrite a grant, a receipt and the digest chain can write the
    anchor's bytes too. Unforgeable issuance needs a signature, which nothing in
    this repository supplies.
    """
    # An `inadmissible` root is a refused admission gate: no authorization was
    # consumed and no execution domain was ever recorded, so there is nothing
    # here to judge and its absence is not a defect.
    if verify_class == INADMISSIBLE:
        return ()
    # Phase A only, and deliberately. § C3.5.1 step 5 makes the receipt the whole
    # of a Phase-B consumption, and that record's shape is not specified yet — no
    # Phase-B attempt exists to specify it against. Judging it here would be this
    # file inventing a contract instead of applying one. When Phase-B
    # consumption is specified, its own domain binding belongs in this predicate.
    if phase != "a":
        return ()
    expected = controlled_host_execution_domain()
    reasons: list[str] = []
    receipt = evidence.load_document(
        root, evidence.AUTHORIZATION_NAME, schema=evidence.AUTHORIZATION_SCHEMA
    )
    if receipt.get("execution_domain") != evidence.digest(expected):
        reasons.append(
            f"{root.name}: the consumption receipt was not written against the "
            "controlled host's authorization ledger"
        )
    archived = evidence.load_document(
        root,
        evidence.AUTHORIZATION_DOMAIN_NAME,
        schema=evidence.AUTHORIZATION_DOMAIN_SCHEMA,
    )
    if set(archived) != set(expected):
        reasons.append(
            f"{root.name}: archived execution domain members "
            f"{sorted(archived)} are not the controlled host's {sorted(expected)}"
        )
    else:
        reasons.extend(
            f"{root.name}: archived execution domain {member} is not the "
            "controlled host's"
            for member in sorted(expected)
            if archived[member] != expected[member]
        )
    return tuple(reasons)


def archive_roots(root: Path = EVIDENCE_ROOT) -> list[Path]:
    if not root.is_dir():
        return []
    legacy = {
        path
        for path in root.glob(f"{evidence.PHASE_A_ROOT_PREFIX}*")
        if path.is_dir() or path.is_symlink()
    }
    # Successor root names are audit metadata, not validity gates (Correction
    # 5).  Discovery therefore uses the artifact family, not a new name prefix.
    # Any one family-specific member is enough to make an incomplete root
    # visible to the fail-closed verifier rather than letting it disappear.
    successor_markers = {
        *successor_verifier.PRODUCER_ARTIFACTS,
        successor_verifier.VERIFICATION_NAME,
    }
    successor = {
        path
        for path in root.iterdir()
        if (path.is_dir() or path.is_symlink())
        and any((path / name).exists() for name in successor_markers)
    }
    return sorted(legacy | successor, key=lambda item: item.name.encode("utf-8"))


def _is_successor_archive(root: Path) -> bool:
    markers = {
        *successor_verifier.PRODUCER_ARTIFACTS,
        successor_verifier.VERIFICATION_NAME,
    }
    return any((root / name).exists() for name in markers)


def successor_admission_reasons(root: Path) -> tuple[str, ...]:
    """Why a successor archive cannot enter the measurement corpus.

    The independent verifier answers internal consistency.  This owner adds the
    one corpus fact it alone can add: a diagnostic is non-qualifying even when
    every predicate and every verifier check passes.  The authority tokens come
    from the ruler rather than being restated here.
    """
    reasons: list[str] = []
    verdict_path = root / successor_verifier.VERIFICATION_NAME
    inventory_path = root / successor_verifier.CHECKSUMS_NAME
    if not verdict_path.is_file() or not inventory_path.is_file():
        missing = [
            name
            for name, path in (
                (successor_verifier.VERIFICATION_NAME, verdict_path),
                (successor_verifier.CHECKSUMS_NAME, inventory_path),
            )
            if not path.is_file()
        ]
        reasons.append(
            f"{root.name}: successor archive is not closed; missing {missing}"
        )

    try:
        documents, _ = successor_verifier.load_archive(root)
        verification = successor_verifier.verify_archive(root)
        successor_verifier.validate_verification(verification)
    except (
        successor_verifier.ExecutionVerificationError,
        evidence.EvidenceError,
        OSError,
    ) as exc:
        reasons.append(
            f"{root.name}: successor archive verification cannot be formed: {exc}"
        )
        return tuple(reasons)

    if verification.get("valid") is not True:
        detail = verification.get("reasons")
        reasons.append(
            f"{root.name}: successor archive is not independently valid: {detail}"
        )

    result = documents.get("result.json")
    if not isinstance(result, Mapping):
        reasons.append(f"{root.name}: successor result is not an object")
    elif result.get("authority") != partition.MEASUREMENT_AUTHORITY:
        reasons.append(
            f"{root.name}: authority {result.get('authority')!r} is not the "
            "exactly-once measurement authority; a diagnostic is never canonical "
            "measurement evidence"
        )
    return tuple(reasons)


def _load_successor_attempt(root: Path) -> SuccessorAttempt:
    reasons = successor_admission_reasons(root)
    if reasons:
        raise CorpusError("; ".join(reasons))
    documents, _ = successor_verifier.load_archive(root)
    result = documents["result.json"]
    verification = successor_verifier.verify_archive(root)
    if not isinstance(result, dict):  # held by the admission verdict above
        raise CorpusError(f"{root.name}: successor result is not an object")
    return SuccessorAttempt(root, result, verification)


def _load_attempt(root: Path) -> Attempt:
    name = evidence.parse_root_name(root.name)
    verify_class = classify(root)
    freeze = evidence.load_document(
        root, evidence.FREEZE_NAME, schema=evidence.FREEZE_SCHEMA
    )
    # Every class verifies, `inadmissible` included: it spent no authorization
    # and asserts nothing about the world, but it is still an artifact with a
    # root identity, a freeze record and a checksum inventory to stand behind.
    report = verifier.VERIFIERS[verify_class](root)
    return Attempt(root, name, verify_class, report, freeze)


def _check_prior_attempts(attempts: Iterable[Attempt]) -> None:
    """The § C3.5 ban, over chains the verifier has already proved complete."""
    known = {attempt.root.name: attempt for attempt in attempts}
    for attempt in attempts:
        if attempt.name.phase != "b" or attempt.verify_class == INADMISSIBLE:
            continue
        # Completeness, order, consumed-only membership and per-class verification
        # are one rule and live in one place; the corpus is where it is applied to
        # every root rather than where it is restated.
        verifier.verify_prior_chain(attempt.root, attempt.freeze, visiting=frozenset())
        for prior_name in attempt.prior_attempts:
            prior = known.get(prior_name) or _load_attempt(
                attempt.root.parent / prior_name
            )
            _check_re_attempt(attempt, prior)


def _check_re_attempt(attempt: Attempt, prior: Attempt) -> None:
    if not prior.bans_re_attempt:
        return
    if attempt.surface is None or prior.surface is None:
        raise CorpusError(
            f"{attempt.root.name}: F_B binds no measurement_surface_digest, so the "
            f"§ C3.5 ban carried by {prior.root.name} cannot be decided"
        )
    if attempt.surface == prior.surface:
        raise CorpusError(
            f"{attempt.root.name}: re-attempt against the same measurement surface "
            f"after {prior.root.name} ({prior.terminal or 'unterminated'}); § C3.5 "
            "forbids it"
        )
    repair = attempt.freeze.get("defect_repair")
    if not isinstance(repair, Mapping):
        raise CorpusError(
            f"{attempt.root.name}: a changed measurement surface is necessary but "
            f"never sufficient — no named defect repair is bound for {prior.root.name}"
        )
    if repair.get("prior_attempt") != prior.root.name:
        raise CorpusError(
            f"{attempt.root.name}: the bound defect repair does not name "
            f"{prior.root.name}"
        )
    defect_class = repair.get("defect_class")
    if defect_class not in REPAIR_VOCABULARY:
        raise CorpusError(
            f"{attempt.root.name}: defect_class {defect_class!r} is outside H0 § 6's "
            f"repair vocabulary {sorted(REPAIR_VOCABULARY)}"
        )


def check_corpus(roots: Iterable[Path]) -> list[Attempt | SuccessorAttempt]:
    """The canonical corpus admission owner.

    Archive-verifier success alone has no canonical-admission meaning: it says
    a root is internally consistent, not that this corpus may contain it.
    """
    attempts: list[Attempt | SuccessorAttempt] = []
    legacy_attempts: list[Attempt] = []
    for root in roots:
        if _is_successor_archive(root):
            attempts.append(_load_successor_attempt(root))
        else:
            attempt = _load_attempt(root)
            attempts.append(attempt)
            legacy_attempts.append(attempt)
    reasons = [
        reason
        for attempt in legacy_attempts
        for reason in execution_domain_admission_reasons(
            attempt.root, attempt.verify_class, attempt.name.phase
        )
    ]
    if reasons:
        raise CorpusError("; ".join(reasons))
    _check_prior_attempts(legacy_attempts)
    return attempts


def main(argv: Iterable[str] | None = None) -> int:
    del argv
    roots = archive_roots()
    if not roots:
        print("H2 measurement corpus: PASS (no evidence roots)")
        return 0
    try:
        attempts = check_corpus(roots)
    except (
        CorpusError,
        evidence.EvidenceError,
        partition.PartitionError,
        verifier.VerificationError,
        OSError,
    ) as exc:
        print(f"H2 measurement corpus rejected: {exc}", file=sys.stderr)
        return 1
    counts: dict[str, int] = {}
    for attempt in attempts:
        counts[attempt.verify_class] = counts.get(attempt.verify_class, 0) + 1
    summary = ", ".join(f"{name}={counts[name]}" for name in sorted(counts))
    print(f"H2 measurement corpus: PASS ({len(attempts)} roots; {summary})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
