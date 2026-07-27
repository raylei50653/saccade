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

CONSUMED_CLASSES = ("complete", "envelope", "unterminated")
INADMISSIBLE = "inadmissible"

# The terminals § C3.5 makes properties of the measurement surface rather than
# of the attempt. A re-attempt against the same surface is forbidden.
SURFACE_TERMINALS = frozenset({"H2_CAPTURE_PERTURBS_POLICY", "H2_PACKET_INVALID"})

# H0 § 6, verbatim: "Only repairs that leave all those semantics unchanged —
# compilation, capacity sizing, serialization, or implementation bugs — may
# proceed under the same seal." § C3.5's first guard consumes that vocabulary
# unchanged; anything outside it is not a repair.
REPAIR_VOCABULARY = frozenset(
    {"compilation", "capacity_sizing", "serialization", "implementation_bug"}
)


class CorpusError(RuntimeError):
    """The corpus does not satisfy § C3.1 / § C3.5 / § C3.5.1. Fail-closed."""


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


def classify(root: Path) -> str:
    """Exactly one class per root; an unclassifiable root is a defect."""
    admission = root / evidence.ADMISSION_NAME
    authorization = root / evidence.AUTHORIZATION_NAME
    terminal = root / evidence.TERMINAL_NAME
    if admission.is_file():
        record = evidence.load_document(
            root, evidence.ADMISSION_NAME, schema=evidence.ADMISSION_SCHEMA
        )
        verdict = partition.evaluate_admission(record, phase="b")
        if not verdict.admitted:
            if authorization.is_file():
                raise CorpusError(
                    f"{root.name}: S_B was consumed after a refused admission gate "
                    "(§ C3.5.1 steps 4-5)"
                )
            return INADMISSIBLE
    name = evidence.parse_root_name(root.name)
    if name.phase == "b" and not authorization.is_file():
        raise CorpusError(
            f"{root.name}: a phase-B root records neither an admission refusal nor "
            "the authorization_consumed write that spends S_B"
        )
    if not terminal.is_file():
        return "unterminated"
    if (root / evidence.MANIFEST_NAME).is_file() and (
        root / evidence.OBSERVATION_NAME
    ).is_file():
        return "complete"
    return "envelope"


def _load_attempt(root: Path) -> Attempt:
    name = evidence.parse_root_name(root.name)
    verify_class = classify(root)
    freeze = evidence.load_document(
        root, evidence.FREEZE_NAME, schema=evidence.FREEZE_SCHEMA
    )
    if verify_class == INADMISSIBLE:
        # Verified only as far as its own class goes: an inadmissible root spent
        # nothing and asserts nothing about the world.
        report: dict[str, Any] = {"verify_class": INADMISSIBLE, "valid": True}
    else:
        report = verifier.VERIFIERS[verify_class](root)
    return Attempt(root, name, verify_class, report, freeze)


def _check_prior_attempts(attempts: Iterable[Attempt]) -> None:
    """Complete, ordered, and § C3.5-admissible, per Phase-A result."""
    by_group: dict[str, list[Attempt]] = {}
    known = {attempt.root.name: attempt for attempt in attempts}
    for attempt in attempts:
        if attempt.name.phase != "b" or attempt.verify_class == INADMISSIBLE:
            continue
        group = attempt.phase_a_evidence_root
        if group is None:
            raise CorpusError(
                f"{attempt.root.name}: F_B binds no phase_a_evidence root (§ C3.2 item 7)"
            )
        by_group.setdefault(group, []).append(attempt)

    for group, members in sorted(by_group.items()):
        chain = sorted(members, key=lambda item: len(item.prior_attempts))
        for position, attempt in enumerate(chain):
            priors = attempt.prior_attempts
            # Existence first: a root naming an attempt that is not in the corpus
            # is a more basic defect than an out-of-order chain, and reporting the
            # ordering instead would send a reader looking in the wrong place.
            for prior_name in priors:
                prior = known.get(prior_name)
                if prior is None:
                    raise CorpusError(
                        f"{attempt.root.name}: prior attempt {prior_name} does not exist"
                    )
                if prior.verify_class == INADMISSIBLE:
                    raise CorpusError(
                        f"{attempt.root.name}: {prior_name} was inadmissible and is "
                        "not a consumed attempt (§ C3.5.1 step 4)"
                    )
                _check_re_attempt(attempt, prior)
            if len(priors) != position:
                raise CorpusError(
                    f"{attempt.root.name}: prior_attempts is incomplete for the "
                    f"Phase-A result {group}: {len(priors)} bound, {position} exist"
                )
            expected = [item.root.name for item in chain[:position]]
            if list(priors) != expected:
                raise CorpusError(
                    f"{attempt.root.name}: prior_attempts is not the ordered list of "
                    f"preceding consumed attempts (expected {expected})"
                )


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
