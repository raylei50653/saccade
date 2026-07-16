from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from scripts.docs.terminal_slot_schema import (
    TerminalSlotValidationError,
    validate_fixture_file,
    validate_reconciled_worked_example,
    validate_terminal_slot,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "docs/ownership/terminal_slot_fixtures.yaml"
RECONCILED_MAP = (
    ROOT / "docs/modules/semantic/research/bridge_fidelity_reconciled_map_20260715.md"
)


def _valid_slot() -> dict[str, object]:
    document = yaml.safe_load(FIXTURES.read_text(encoding="utf-8"))
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
