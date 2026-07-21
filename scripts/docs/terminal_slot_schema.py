"""Fail-closed validator for the ADR 020 terminal-slot schema v0.

The validator is intentionally independent from a document parser: callers
hand it one YAML mapping and receive a stable error class if it is invalid.
That lets the canonical fixtures exercise the schema without making the
reconciled map a special case.
"""
# status: stable

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # Supports both ``python -m`` and direct script execution.
    from .strict_yaml import StrictYamlError, strict_safe_load
except ImportError:  # pragma: no cover - exercised by direct CLI use
    from strict_yaml import StrictYamlError, strict_safe_load


LINE_TYPES = frozenset(
    {
        "math-closed",
        "local-math-claim",
        "scoped-empirical",
        "engineering-ablation",
    }
)
EMPIRICAL_CLAIM_VERDICTS = frozenset(
    {
        "VERIFIED",
        "FALSIFIED",
        "NOT_IDENTIFIABLE",
        "INCONCLUSIVE",
        "NOT_EVALUATED",
    }
)
DEDUCTIVE_CLAIM_VERDICTS = frozenset({"PROVED", "REFUTED"})
CLAIM_VERDICTS = EMPIRICAL_CLAIM_VERDICTS | DEDUCTIVE_CLAIM_VERDICTS
DECISION_OUTCOMES = frozenset(
    {"POSITIVE", "NET_NEGATIVE", "NO_PRODUCTION_ADVANTAGE", "NOT_ASSESSED"}
)
MODEL_RELATIONS = frozenset({"current", "superseded"})
LIFECYCLE_DISPOSITIONS = frozenset({"SEALED"})
LIVE_LIFECYCLE_STATES = frozenset(
    {"proposed", "active", "parked", "PROPOSED", "ACTIVE", "PARKED"}
)

REQUIRED_SLOT_FIELDS = frozenset(
    {
        "study_id",
        "line_type",
        "claim_verdict",
        "decision_outcome",
        "lifecycle_disposition",
        "verdict_locus",
        "evidence_owner",
        "process_disposition",
    }
)
SLOT_FIELDS = REQUIRED_SLOT_FIELDS | {"model_relation"}

LOCUS_FIELDS_BY_LINE_TYPE = {
    "math-closed": frozenset({"assumptions", "domain", "model_ref", "model_version"}),
    "local-math-claim": frozenset({"assumptions", "domain", "claim"}),
    "scoped-empirical": frozenset({"assumptions", "domain", "protocol_ref"}),
    "engineering-ablation": frozenset({"attribution"}),
}

EXPECTED_VALID_FIXTURE_COUNT = 5
EXPECTED_INVALID_FIXTURE_COUNT = 9
RECONCILED_STUDY_IDS = frozenset(
    {
        "kappa_d0_proxy_fidelity",
        "kappa_r1_runtime_replay",
        "rho_s0_safe_axis_transfer",
        "ek0_exact_key_recoverability",
        "p0_decision_path_identifiability",
        "door0_t2_ranking_power",
    }
)


class TerminalSlotValidationError(ValueError):
    """A schema error whose class is part of the fixture contract."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


class FixtureValidationError(ValueError):
    """The canonical fixture file did not demonstrate the promised contract."""

    def __init__(self, message: str, *, error_class: str = "invalid_fixture") -> None:
        super().__init__(message)
        self.error_class = error_class


class WorkedExampleValidationError(ValueError):
    """The reconciled navigation map no longer supplies the six valid slots."""

    def __init__(
        self, message: str, *, error_class: str = "invalid_worked_example"
    ) -> None:
        super().__init__(message)
        self.error_class = error_class


@dataclass(frozen=True)
class FixtureValidationSummary:
    valid_count: int
    invalid_count: int


def _fail(error_class: str, message: str) -> None:
    raise TerminalSlotValidationError(error_class, message)


def _require_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail("invalid_field_type", f"{field} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        _fail("invalid_field_type", f"{field} keys must be strings")
    return value


def _require_nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("invalid_field_type", f"{field} must be a non-empty string")
    return value


def _validate_enum(value: object, *, field: str, values: frozenset[str]) -> str:
    text = _require_nonempty_string(value, field=field)
    if text not in values:
        _fail("unknown_enum", f"{field} has unknown value {text!r}")
    return text


def _validate_claim_verdict(value: object, *, line_type: str) -> None:
    verdict = _require_nonempty_string(value, field="claim_verdict")
    if verdict in DECISION_OUTCOMES | LIFECYCLE_DISPOSITIONS | MODEL_RELATIONS:
        _fail(
            "cross_axis_misuse",
            f"claim_verdict cannot use {verdict!r} from another axis",
        )
    if verdict not in CLAIM_VERDICTS:
        _fail("unknown_enum", f"claim_verdict has unknown value {verdict!r}")

    permitted = (
        DEDUCTIVE_CLAIM_VERDICTS
        if line_type == "local-math-claim"
        else EMPIRICAL_CLAIM_VERDICTS
    )
    if verdict not in permitted:
        _fail(
            "line_type_claim_verdict_mismatch",
            f"claim_verdict {verdict!r} is incompatible with {line_type!r}",
        )


def _validate_decision_outcome(value: object) -> None:
    outcome = _require_nonempty_string(value, field="decision_outcome")
    if outcome in CLAIM_VERDICTS | LIFECYCLE_DISPOSITIONS | MODEL_RELATIONS:
        _fail(
            "cross_axis_misuse",
            f"decision_outcome cannot use {outcome!r} from another axis",
        )
    if outcome not in DECISION_OUTCOMES:
        _fail("unknown_enum", f"decision_outcome has unknown value {outcome!r}")


def _validate_lifecycle_disposition(value: object) -> None:
    disposition = _require_nonempty_string(value, field="lifecycle_disposition")
    if disposition in LIVE_LIFECYCLE_STATES:
        _fail(
            "live_state_not_terminal",
            f"lifecycle_disposition {disposition!r} is a live state, not a slot state",
        )
    if disposition not in LIFECYCLE_DISPOSITIONS:
        _fail(
            "unknown_enum",
            f"lifecycle_disposition has unknown value {disposition!r}",
        )


def _validate_model_relation(slot: Mapping[str, object], *, line_type: str) -> None:
    present = "model_relation" in slot
    if line_type != "math-closed":
        if present:
            _fail(
                "model_relation_requires_math_closed",
                "model_relation is meaningful only for math-closed slots",
            )
        return

    if not present:
        _fail("missing_required_field", "math-closed slot is missing model_relation")
    _validate_enum(
        slot["model_relation"], field="model_relation", values=MODEL_RELATIONS
    )


def _validate_verdict_locus(value: object, *, line_type: str) -> None:
    locus = _require_mapping(value, field="verdict_locus")
    permitted = LOCUS_FIELDS_BY_LINE_TYPE[line_type]
    unknown = sorted(set(locus) - permitted)
    if unknown:
        _fail(
            "unknown_field",
            f"verdict_locus contains unknown field {unknown[0]!r} for {line_type!r}",
        )
    missing = sorted(permitted - set(locus))
    if missing:
        _fail(
            "missing_required_field",
            f"verdict_locus is missing required field {missing[0]!r}",
        )

    for field, field_value in locus.items():
        text = _require_nonempty_string(field_value, field=f"verdict_locus.{field}")
        if field == "model_version" and not re.fullmatch(r"v\d+\.\d+", text):
            _fail(
                "invalid_value",
                "verdict_locus.model_version must use vMAJOR.MINOR",
            )


def _validate_process_disposition(value: object) -> None:
    disposition = _require_nonempty_string(value, field="process_disposition")
    if disposition.startswith("deleted-to-git@"):
        _fail(
            "process_disposition_no_inline_sha",
            "deleted-to-git must not pin an inline commit or blob identifier",
        )
    if disposition in {"retained", "deleted-to-git"}:
        return
    if disposition.startswith("folded-to-workspace@"):
        destination = disposition.removeprefix("folded-to-workspace@")
        if destination and not Path(destination).is_absolute():
            return
    _fail("invalid_value", f"invalid process_disposition {disposition!r}")


def validate_terminal_slot(slot: Mapping[str, object]) -> None:
    """Validate one terminal slot, failing closed on every unknown field.

    Returns ``None`` on success.  Consumers should catch
    :class:`TerminalSlotValidationError` and use ``error_class`` for stable
    machine-facing reporting.
    """

    slot = _require_mapping(slot, field="slot")
    unknown = sorted(set(slot) - SLOT_FIELDS)
    if unknown:
        _fail("unknown_field", f"slot contains unknown field {unknown[0]!r}")
    missing = sorted(REQUIRED_SLOT_FIELDS - set(slot))
    if missing:
        _fail(
            "missing_required_field", f"slot is missing required field {missing[0]!r}"
        )

    _require_nonempty_string(slot["study_id"], field="study_id")
    line_type = _validate_enum(slot["line_type"], field="line_type", values=LINE_TYPES)
    _validate_claim_verdict(slot["claim_verdict"], line_type=line_type)
    _validate_decision_outcome(slot["decision_outcome"])
    _validate_lifecycle_disposition(slot["lifecycle_disposition"])
    _validate_model_relation(slot, line_type=line_type)
    _validate_verdict_locus(slot["verdict_locus"], line_type=line_type)
    _require_nonempty_string(slot["evidence_owner"], field="evidence_owner")
    _validate_process_disposition(slot["process_disposition"])


def _load_yaml_mapping(path: Path) -> Mapping[str, object]:
    try:
        document = strict_safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise FixtureValidationError(f"cannot read {path}: {error}") from error
    except StrictYamlError as error:
        raise FixtureValidationError(
            f"cannot parse YAML {path}: {error}", error_class=error.error_class
        ) from error
    if not isinstance(document, Mapping):
        raise FixtureValidationError(f"{path} must contain a YAML mapping")
    return document


def validate_fixture_file(path: str | Path) -> FixtureValidationSummary:
    """Run the canonical valid/invalid fixture contract."""

    document = _load_yaml_mapping(Path(path))
    valid = document.get("valid")
    invalid = document.get("invalid")
    if not isinstance(valid, list) or not isinstance(invalid, list):
        raise FixtureValidationError("fixtures must contain valid and invalid lists")
    if (
        len(valid) != EXPECTED_VALID_FIXTURE_COUNT
        or len(invalid) != EXPECTED_INVALID_FIXTURE_COUNT
    ):
        raise FixtureValidationError(
            "fixture counts changed: expected "
            f"{EXPECTED_VALID_FIXTURE_COUNT} valid and {EXPECTED_INVALID_FIXTURE_COUNT} invalid"
        )

    for entry in valid:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("name"), str):
            raise FixtureValidationError("every valid fixture requires a string name")
        candidate = entry.get("slot")
        if not isinstance(candidate, Mapping):
            raise FixtureValidationError(
                f"valid fixture {entry['name']!r} lacks a slot mapping"
            )
        try:
            validate_terminal_slot(candidate)
        except TerminalSlotValidationError as error:
            raise FixtureValidationError(
                f"valid fixture {entry['name']!r} failed as {error.error_class}: {error}"
            ) from error

    for entry in invalid:
        if not isinstance(entry, Mapping):
            raise FixtureValidationError("every invalid fixture must be a mapping")
        name = entry.get("name")
        expected = entry.get("expect_error")
        candidate = entry.get("slot")
        if (
            not isinstance(name, str)
            or not isinstance(expected, str)
            or not isinstance(candidate, Mapping)
        ):
            raise FixtureValidationError(
                "invalid fixtures require name, expect_error, and slot"
            )
        try:
            validate_terminal_slot(candidate)
        except TerminalSlotValidationError as error:
            if error.error_class != expected:
                raise FixtureValidationError(
                    f"invalid fixture {name!r}: expected {expected!r}, got {error.error_class!r}"
                ) from error
        else:
            raise FixtureValidationError(
                f"invalid fixture {name!r} unexpectedly passed"
            )

    return FixtureValidationSummary(valid_count=len(valid), invalid_count=len(invalid))


def extract_yaml_slots_from_markdown(path: str | Path) -> list[Mapping[str, object]]:
    """Return slot-shaped YAML fence contents from a Markdown document."""

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as error:
        raise WorkedExampleValidationError(f"cannot read {source}: {error}") from error

    slots: list[Mapping[str, object]] = []
    for block in re.findall(r"```yaml\s*\n(.*?)```", text, flags=re.DOTALL):
        # Documents may use YAML fences for their own non-slot records.  Only
        # parse a fence once it declares itself slot-shaped; otherwise an
        # unrelated legacy pseudo-YAML block could prevent a terminal owner
        # from being validated.
        if not re.search(r"(?m)^\s*study_id\s*:", block):
            continue
        try:
            parsed: Any = strict_safe_load(block)
        except StrictYamlError as error:
            raise WorkedExampleValidationError(
                f"invalid YAML fence in {source}: {error}",
                error_class=error.error_class,
            ) from error
        if isinstance(parsed, Mapping) and "study_id" in parsed:
            slots.append(parsed)
    return slots


def validate_reconciled_worked_example(path: str | Path) -> tuple[str, ...]:
    """Validate D0/R1/S0/EK0/P0/T2 with the generic schema validator only."""

    slots = extract_yaml_slots_from_markdown(path)
    study_ids = [slot.get("study_id") for slot in slots]
    if (
        len(slots) != len(RECONCILED_STUDY_IDS)
        or set(study_ids) != RECONCILED_STUDY_IDS
    ):
        raise WorkedExampleValidationError(
            "reconciled map must contain exactly the D0/R1/S0/EK0/P0/T2 terminal slots"
        )

    for slot in slots:
        try:
            validate_terminal_slot(slot)
        except TerminalSlotValidationError as error:
            study_id = slot.get("study_id", "<unknown>")
            raise WorkedExampleValidationError(
                f"reconciled slot {study_id!r} failed as {error.error_class}: {error}"
            ) from error
        if slot["line_type"] != "scoped-empirical":
            raise WorkedExampleValidationError(
                f"reconciled slot {slot['study_id']!r} must be scoped-empirical"
            )
    return tuple(sorted(RECONCILED_STUDY_IDS))


def _default_path(relative: str) -> Path:
    return Path.cwd() / relative


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate ADR 020 terminal-slot schema v0"
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=_default_path("docs/ownership/terminal_slot_fixtures.yaml"),
    )
    parser.add_argument(
        "--worked-example",
        type=Path,
        default=_default_path(
            "docs/modules/semantic/research/bridge_fidelity_reconciled_map_20260715.md"
        ),
    )
    arguments = parser.parse_args()
    try:
        fixture_summary = validate_fixture_file(arguments.fixtures)
        study_ids = validate_reconciled_worked_example(arguments.worked_example)
    except (FixtureValidationError, WorkedExampleValidationError) as error:
        print(f"terminal-slot validation failed: {error}")
        return 1

    print(
        "terminal-slot fixtures green: "
        f"{fixture_summary.valid_count} valid, {fixture_summary.invalid_count} invalid"
    )
    print(f"reconciled worked examples green: {', '.join(study_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
