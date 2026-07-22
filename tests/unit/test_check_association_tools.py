"""Unit tests for scripts/tools/check_association_tools.py.

No GPU. Uses live registry by default plus a tiny synthetic registry for
schema/error paths.
"""

# scope: eval
# function: contract
# lifecycle: active

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "tools" / "check_association_tools.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_association_tools", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def checker():
    return _load_checker()


def test_live_registry_passes(checker):
    """Repo association_tools.yaml should pass path / no_go / schema checks."""
    assert checker.DEFAULT_REGISTRY.is_file()
    rc = checker.main([])
    assert rc == 0


def test_list_and_print_recipe(checker, capsys):
    rc = checker.main(["--list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Door A" in out
    assert "build_relink_candidates" in out

    rc = checker.main(["--print-recipe", "R-A"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "R-A" in out
    assert "build_relink_candidates" in out
    assert "uv run python" in out


def test_print_unknown_recipe(checker):
    assert checker.main(["--print-recipe", "R-NOT-A-THING"]) == 2


def test_synthetic_missing_path_errors(checker, tmp_path):
    reg = {
        "registry_status": "test",
        "allowed_doors": {"A": {"name": "a"}},
        "allowed_roles": ["canonical"],
        "allowed_priorities": ["P0"],
        "forbidden_fields": ["metrics", "verdict"],
        "tools": [
            {
                "id": "ghost_tool",
                "path": "scripts/tools/does_not_exist_assoc_xyz.py",
                "door": "A",
                "roles": ["canonical"],
                "priority": "P0",
                "fact_owner": "docs/modules/semantic/research/"
                "association_recovery_info_source_contract_20260709.md",
                "no_go_ids": ["#99999"],
                "recipes": ["R-X"],
            }
        ],
        "recipes": [
            {
                "id": "R-X",
                "title": "x",
                "door": "A",
                "purpose": "test",
                "steps": ["ghost_tool"],
                "fact_owner": "docs/modules/semantic/research/"
                "association_recovery_info_source_contract_20260709.md",
            }
        ],
    }
    reg_path = tmp_path / "association_tools.yaml"
    reg_path.write_text(yaml.dump(reg), encoding="utf-8")

    rc = checker.main(
        [
            "--registry",
            str(reg_path),
            "--no-go",
            str(checker.DEFAULT_NO_GO),
        ]
    )
    assert rc == 1


def test_forbidden_field_detected(checker):
    data = {
        "allowed_doors": {"A": {}},
        "allowed_roles": ["canonical"],
        "allowed_priorities": ["P0"],
        "forbidden_fields": ["metrics", "verdict"],
        "tools": [
            {
                "id": "bad",
                "path": "scripts/eval/mot17.py",
                "door": "A",
                "roles": ["canonical"],
                "priority": "P0",
                "fact_owner": "docs/modules/semantic/research/"
                "offline_relink_candidate_analysis.md",
                "metrics": {"idf1": 0.99},
            }
        ],
        "recipes": [],
    }
    errors, _warnings = checker.check_registry(
        data, no_go_ids=set(), no_go_path=checker.DEFAULT_NO_GO
    )
    assert any("forbidden field 'metrics'" in e for e in errors)


def test_no_go_id_loader(checker):
    ids = checker.load_no_go_ids(checker.DEFAULT_NO_GO)
    assert "#39" in ids
    assert "#55" in ids
    assert "#57" in ids
