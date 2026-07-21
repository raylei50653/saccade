"""Contract for the old-flagship per-study inventory parser."""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.docs.migration_manifest import parse_migration_manifest
from scripts.docs.old_flagship_inventory import (
    OldFlagshipInventoryError,
    OldFlagshipInventory,
    _parse_studies,
    parse_old_flagship_inventory,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs/ownership/doc_migration_manifest.yaml"
INVENTORY = ROOT / "docs/ownership/old_flagship_per_study_inventory.yaml"


def _parse_inventory(inventory: Path = INVENTORY) -> OldFlagshipInventory:
    manifest = parse_migration_manifest(MANIFEST, repo_root=ROOT)
    return parse_old_flagship_inventory(inventory, manifest=manifest, repo_root=ROOT)


def test_old_flagship_inventory_has_one_role_for_every_resolved_file() -> None:
    inventory = _parse_inventory()

    assert len(inventory.file_roles) == 20
    assert len(inventory.terminal_backed_files) == 15
    assert len(inventory.live_owned_files) == 3
    assert len(inventory.shared_support_files) == 2
    assert inventory.unmapped_files == frozenset()
    assert set(inventory.file_roles) == (
        inventory.terminal_backed_files
        | inventory.live_owned_files
        | inventory.shared_support_files
    )


def test_old_flagship_terminal_and_live_studies_have_disjoint_owner_forms() -> None:
    inventory = _parse_inventory()

    terminal_backed = [
        study for study in inventory.studies.values() if study.terminal_ref is not None
    ]
    live_owned = [
        study for study in inventory.studies.values() if study.live_owner is not None
    ]

    assert len(terminal_backed) == 6
    assert {study.study_id for study in terminal_backed} == {
        "kappa_d0_proxy_fidelity",
        "kappa_r1_runtime_replay",
        "rho_s0_safe_axis_transfer",
        "ek0_exact_key_recoverability",
        "p0_decision_path_identifiability",
        "door0_t2_ranking_power",
    }
    assert all(
        study.terminal_ref and study.live_owner is None for study in terminal_backed
    )
    assert {study.study_id for study in live_owned} == {
        "discrete_m_capability_20260712",
        "h0_full_bridge_decision_capture",
    }
    assert all(study.live_owner and study.terminal_ref is None for study in live_owned)


def test_old_flagship_remains_quarantined_b_and_not_disposal_authorized() -> None:
    inventory = _parse_inventory()

    assert inventory.migration_state == "quarantined"
    assert inventory.classification == "B"
    assert inventory.disposal_authorized is False
    assert inventory.shared_support_files


def test_old_flagship_rejects_a_resolved_file_without_a_role(tmp_path: Path) -> None:
    content = INVENTORY.read_text(encoding="utf-8")
    content = content.replace(
        "  - path: docs/modules/semantic/research/m_gate_h_ratio_signal_7seq_20260709.md\n"
        "    kind: shared_support\n",
        "",
    )
    candidate = tmp_path / "inventory.yaml"
    candidate.write_text(content, encoding="utf-8")

    with pytest.raises(OldFlagshipInventoryError) as raised:
        _parse_inventory(candidate)

    assert raised.value.error_class == "unmapped_resolved_file"


def test_old_flagship_rejects_both_terminal_and_live_owners(
    tmp_path: Path,
) -> None:
    content = INVENTORY.read_text(encoding="utf-8")
    content = content.replace(
        "  discrete_m_capability_20260712:\n"
        "    live_owner: docs/modules/semantic/research/discrete_m_capability_declaration_20260712.md\n",
        "  discrete_m_capability_20260712:\n"
        "    live_owner: docs/modules/semantic/research/discrete_m_capability_declaration_20260712.md\n"
        "    terminal_ref: docs/modules/semantic/research/d0_runtime_shadow_fidelity_results_20260712.md\n",
    )
    candidate = tmp_path / "inventory.yaml"
    candidate.write_text(content, encoding="utf-8")

    with pytest.raises(OldFlagshipInventoryError) as raised:
        _parse_inventory(candidate)

    assert raised.value.error_class == "invalid_owner_form"


def test_old_flagship_rejects_a_study_without_an_owner(tmp_path: Path) -> None:
    content = INVENTORY.read_text(encoding="utf-8")
    content = content.replace(
        "  h0_full_bridge_decision_capture:\n"
        "    live_owner: docs/modules/semantic/TODO.md\n",
        "  h0_full_bridge_decision_capture: {}\n",
    )
    candidate = tmp_path / "inventory.yaml"
    candidate.write_text(content, encoding="utf-8")

    with pytest.raises(OldFlagshipInventoryError) as raised:
        _parse_inventory(candidate)

    assert raised.value.error_class == "invalid_owner_form"


def test_terminal_registry_can_own_slots_for_multiple_studies(tmp_path: Path) -> None:
    registry = tmp_path / "docs/ownership/shared_terminal_registry.md"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        """```yaml
study_id: first_study
line_type: scoped-empirical
claim_verdict: FALSIFIED
decision_outcome: NOT_ASSESSED
lifecycle_disposition: SEALED
verdict_locus:
  assumptions: fixed assumptions
  domain: fixed domain
  protocol_ref: fixed protocol
evidence_owner: docs/ownership/shared_terminal_registry.md
process_disposition: retained
```

```yaml
study_id: second_study
line_type: scoped-empirical
claim_verdict: FALSIFIED
decision_outcome: NOT_ASSESSED
lifecycle_disposition: SEALED
verdict_locus:
  assumptions: fixed assumptions
  domain: fixed domain
  protocol_ref: fixed protocol
evidence_owner: docs/ownership/shared_terminal_registry.md
process_disposition: retained
```
""",
        encoding="utf-8",
    )

    studies = _parse_studies(
        {
            "first_study": {
                "terminal_ref": "docs/ownership/shared_terminal_registry.md"
            },
            "second_study": {
                "terminal_ref": "docs/ownership/shared_terminal_registry.md"
            },
        },
        repo_root=tmp_path,
    )

    assert all(study.is_terminal_backed for study in studies.values())
