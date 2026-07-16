from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from scripts.docs.terminal_slot_schema import (
    FixtureValidationError,
    TerminalSlotValidationError,
    WorkedExampleValidationError,
    extract_yaml_slots_from_markdown,
    validate_fixture_file,
    validate_reconciled_worked_example,
    validate_terminal_slot,
)
from scripts.docs.strict_yaml import strict_safe_load


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "docs/ownership/terminal_slot_fixtures.yaml"
RECONCILED_MAP = (
    ROOT / "docs/modules/semantic/research/bridge_fidelity_reconciled_map_20260715.md"
)


def _valid_slot() -> dict[str, object]:
    document = strict_safe_load(FIXTURES.read_text(encoding="utf-8"))
    return deepcopy(document["valid"][0]["slot"])


def test_canonical_terminal_slot_fixtures_are_green() -> None:
    summary = validate_fixture_file(FIXTURES)

    assert summary.valid_count == 5
    assert summary.invalid_count == 9


def test_reconciled_map_six_worked_examples_use_generic_validator() -> None:
    study_ids = validate_reconciled_worked_example(RECONCILED_MAP)

    assert len(study_ids) == 6
    assert "kappa_d0_proxy_fidelity" in study_ids
    assert "door0_t2_ranking_power" in study_ids


def test_verdict_locus_unknown_field_fails_closed() -> None:
    slot = _valid_slot()
    locus = slot["verdict_locus"]
    assert isinstance(locus, dict)
    locus["epistemic_verdict"] = "FALSIFIED"

    with pytest.raises(TerminalSlotValidationError) as raised:
        validate_terminal_slot(slot)

    assert raised.value.error_class == "unknown_field"


def test_terminal_fixture_rejects_duplicate_slot_top_level_key(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures.yaml"
    fixtures.write_text(
        """valid:
  - name: duplicate slot key
    slot:
      study_id: first
      study_id: second
invalid: []
""",
        encoding="utf-8",
    )

    with pytest.raises(FixtureValidationError) as raised:
        validate_fixture_file(fixtures)

    assert raised.value.error_class == "duplicate_yaml_key"


def test_worked_example_rejects_duplicate_nested_verdict_locus_key(
    tmp_path: Path,
) -> None:
    worked_example = tmp_path / "worked-example.md"
    worked_example.write_text(
        """```yaml
study_id: duplicate_nested_locus
line_type: scoped-empirical
claim_verdict: FALSIFIED
decision_outcome: NOT_ASSESSED
lifecycle_disposition: SEALED
verdict_locus:
  assumptions: fixed assumptions
  domain: first domain
  domain: second domain
  protocol_ref: sealed protocol
evidence_owner: docs/results.md
process_disposition: retained
```
""",
        encoding="utf-8",
    )

    with pytest.raises(WorkedExampleValidationError) as raised:
        extract_yaml_slots_from_markdown(worked_example)

    assert raised.value.error_class == "duplicate_yaml_key"


_MISSING = object()


def _structural_diff(
    before: object, after: object, path: tuple[str, ...] = ()
) -> list[tuple[str, ...]]:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        differences: list[tuple[str, ...]] = []
        for key in sorted(set(before) | set(after), key=str):
            left = before.get(key, _MISSING)
            right = after.get(key, _MISSING)
            child_path = path + (str(key),)
            if left is _MISSING or right is _MISSING:
                differences.append(child_path)
            else:
                differences.extend(_structural_diff(left, right, child_path))
        return differences
    if (
        isinstance(before, Sequence)
        and not isinstance(before, (str, bytes))
        and isinstance(after, Sequence)
        and not isinstance(after, (str, bytes))
    ):
        differences = []
        for index in range(max(len(before), len(after))):
            left = before[index] if index < len(before) else _MISSING
            right = after[index] if index < len(after) else _MISSING
            child_path = path + (str(index),)
            if left is _MISSING or right is _MISSING:
                differences.append(child_path)
            else:
                differences.extend(_structural_diff(left, right, child_path))
        return differences
    return [] if before == after else [path]


@pytest.mark.parametrize(
    ("base", "candidate", "expected_path"),
    [
        ({"field": "old"}, {"field": "new"}, ("field",)),
        ({}, {"field": "added"}, ("field",)),
        ({"field": "removed"}, {}, ("field",)),
        (
            {"verdict_locus": {"domain": "old"}},
            {"verdict_locus": {"domain": "new"}},
            ("verdict_locus", "domain"),
        ),
    ],
)
def test_structural_diff_counts_single_replacement_addition_removal_and_nested_change(
    base: dict[str, Any], candidate: dict[str, Any], expected_path: tuple[str, ...]
) -> None:
    assert _structural_diff(base, candidate) == [expected_path]


def test_each_invalid_fixture_is_exactly_one_structural_mutation() -> None:
    document = strict_safe_load(FIXTURES.read_text(encoding="utf-8"))
    base = document["_base_scoped_empirical"]

    for fixture in document["invalid"]:
        differences = _structural_diff(base, fixture["slot"])
        assert len(differences) == 1, f"{fixture['name']}: {differences}"
