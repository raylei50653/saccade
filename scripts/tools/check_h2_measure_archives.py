#!/usr/bin/env python3
"""Corpus checker for committed H2 Layer-M evidence roots.

A new checker rather than a change to `check_h0_phase_a_archives.py`: that tool
keeps verifying the frozen H0 v1 corpus under the v1 schema and never sees an H2
root (§ 9 item 3), which is why the evidence prefix is `h2_measure_` in the first
place.

Three things live here because § C3.1 and § C3.5.1 put them here, and nowhere
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

An empty corpus passes. No Layer-M authorization has ever been issued, so there
is nothing to verify; a checker that failed here would be reporting the absence
of a measurement as a defect in the archive.

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
import verify_h2_measurement as verifier  # noqa: E402

EVIDENCE_ROOT = REPO_ROOT / evidence.EVIDENCE_REL

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


def archive_roots(root: Path = EVIDENCE_ROOT) -> list[Path]:
    if not root.is_dir():
        return []
    return [
        path
        for path in sorted(
            root.glob(f"{evidence.PHASE_A_ROOT_PREFIX}*"),
            key=lambda item: item.name.encode("utf-8"),
        )
        if path.is_dir()
    ]


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


def check_corpus(roots: Iterable[Path]) -> list[Attempt]:
    attempts = [_load_attempt(root) for root in roots]
    _check_prior_attempts(attempts)
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
