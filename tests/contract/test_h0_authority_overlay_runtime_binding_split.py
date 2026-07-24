"""Contract for Amendment 10 authority-overlay / runtime-binding split.

Pins the S3 provenance defect repair:
  declaration must not be a runtime-bound repository input, must live only in
  the owner authority overlay, and continuous monitoring uses S bytes.
"""

# scope: system
# function: regression
# lifecycle: active

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "scripts/tools"
sys.path.insert(0, TOOLS.as_posix())

import run_h0_phase_a as controller  # noqa: E402
import verify_h0_phase_a as phase_a  # noqa: E402
import verify_h0_preseal_freeze as preseal  # noqa: E402
import check_h0_repair_acceptance_matrix as matrix  # noqa: E402
import h0_launch_hygiene_gate as gate  # noqa: E402

DECLARATION = controller.DECLARATION_PATH
S3 = {
    "I": "5a2d1de509fa64f2e5ce9a4db8182337da215968",
    "F": "7895704c298504b279ae8e1febf19ca2a715637f",
    "S": "3a6a9ec6348f1dccca6acabef8025159c3bec1d3",
}


def _git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=cwd,
        text=True,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"},
    ).strip()


def _commit(root: Path, message: str) -> str:
    _git("add", "-A", cwd=root)
    _git("commit", "-m", message, cwd=root)
    return _git("rev-parse", "HEAD", cwd=root)


def test_repair_unit_is_sole_authority_overlay_split() -> None:
    matrix.validate_matrix(matrix.load_matrix())
    assert matrix.REPAIR_UNITS == ("h0_authority_overlay_runtime_binding_split_v1",)


def test_declaration_absent_from_runtime_required_inputs() -> None:
    assert DECLARATION not in controller.REQUIRED_REPOSITORY_INPUTS
    assert DECLARATION not in preseal.REPOSITORY_INPUTS
    assert DECLARATION not in phase_a.REQUIRED_REPOSITORY_INPUTS
    assert DECLARATION in controller.HISTORICAL_REQUIRED_REPOSITORY_INPUTS
    assert DECLARATION in controller.RUNTIME_EXCLUDED_REPOSITORY_PATHS


def test_runtime_and_overlay_inventories_are_disjoint() -> None:
    assert controller.OWNER_AUTHORITY_OVERLAY_SCHEMA == "h0_owner_authority_overlay_v1"
    assert DECLARATION in controller.RUNTIME_EXCLUDED_REPOSITORY_PATHS
    # Runtime required set and overlay-only path must not intersect.
    assert not (set(controller.REQUIRED_REPOSITORY_INPUTS) & {DECLARATION})


def test_leaked_declaration_in_runtime_inventory_fails_closed() -> None:
    paths = sorted(
        (*controller.REQUIRED_REPOSITORY_INPUTS, DECLARATION),
        key=lambda path: path.encode("utf-8"),
    )
    inventory = {
        "schema": controller.BOUND_INPUTS_SCHEMA,
        "digest": "",
        "repository": [
            {
                "git_object": hashlib.sha1(
                    path.encode("utf-8"), usedforsecurity=False
                ).hexdigest(),
                "git_type": "blob",
                "kind": "regular",
                "length": len(path.encode("utf-8")),
                "mode": "100644",
                "path": path,
                "sha256": hashlib.sha256(path.encode("utf-8")).hexdigest(),
            }
            for path in paths
        ],
        "models_engines": [
            {
                "length": 1,
                "logical_path": logical,
                "realpath": f"/fixture/{index}",
                "sha256": hashlib.sha256(bytes([index])).hexdigest(),
                "symlink_chain": [],
            }
            for index, logical in enumerate(controller.MODEL_LOGICAL_PATHS, start=1)
        ],
        "sequence": {
            "algorithm": "h0_sequence_inputs_v1",
            "digest": "",
            "files": [],
            "root": controller.SEQUENCE_REL,
        },
        "tool_runtime": [],
    }
    inventory["sequence"]["digest"] = hashlib.sha256(
        controller.canonical_json_bytes(
            {"algorithm": "h0_sequence_inputs_v1", "files": []}
        )
    ).hexdigest()
    inventory["digest"] = controller.bound_inventory_digest(inventory)
    with pytest.raises(controller.ContractError, match="leaked into runtime"):
        controller.validate_bound_inventory(inventory)


def test_owner_authority_overlay_shape() -> None:
    overlay = controller.owner_authority_overlay(
        artifact_path="docs/modules/semantic/research/evidence/h0_preseal_freeze_"
        + ("a" * 40)
        + "/h0_preseal_freeze_v3.json",
        declaration_at_f={"length": 12, "sha256": "ab" * 32},
    )
    assert overlay["schema"] == controller.OWNER_AUTHORITY_OVERLAY_SCHEMA
    assert overlay["declaration_path"] == DECLARATION
    assert overlay["declaration_at_f"] == {"length": 12, "sha256": "ab" * 32}
    preseal._verify_landing_shape(overlay, "a" * 40)


def test_positive_i_f_s_topology_with_overlay(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _git("init", cwd=root)
    _git("config", "user.email", "h0@example.invalid", cwd=root)
    _git("config", "user.name", "H0 test", cwd=root)
    declaration = root / DECLARATION
    declaration.parent.mkdir(parents=True)
    base = b"base declaration\n"
    declaration.write_bytes(base)
    instrumentation = _commit(root, "instrumentation")
    freeze_rel = preseal.freeze_path(instrumentation)
    overlay = {
        "schema": preseal.LANDING_SCHEMA,
        "artifact_path": freeze_rel,
        "declaration_path": DECLARATION,
        "declaration_at_f": {
            "length": len(base),
            "sha256": hashlib.sha256(base).hexdigest(),
        },
        "post_head_allowed_paths": [freeze_rel, DECLARATION],
    }
    # Freeze blob must equal the object later verified at landing.
    artifact = {
        "instrumentation_head": instrumentation,
        "authority_landing": overlay,
        "phase_a_controller_input": {"authority_landing": overlay},
    }
    freeze_path = root / freeze_rel
    freeze_path.parent.mkdir(parents=True)
    freeze_path.write_bytes(preseal.canonical_json(artifact) + b"\n")
    freeze = _commit(root, "freeze")
    with declaration.open("ab") as handle:
        handle.write(
            f"| 2026-07-24 | `{instrumentation}` | `{freeze}` | `SEALED` |\n".encode()
        )
    seal = _commit(root, "seal")
    relation = preseal.verify_authority_landing(root, artifact)
    assert relation == {
        "instrumentation_head": instrumentation,
        "freeze_commit": freeze,
        "seal_commit": seal,
        "execution_checkout": seal,
    }


def test_negative_declaration_byte_change_beyond_owner_event(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _git("init", cwd=root)
    _git("config", "user.email", "h0@example.invalid", cwd=root)
    _git("config", "user.name", "H0 test", cwd=root)
    declaration = root / DECLARATION
    declaration.parent.mkdir(parents=True)
    declaration.write_text("base declaration\n", encoding="utf-8")
    instrumentation = _commit(root, "instrumentation")
    artifact_body = {"instrumentation_head": instrumentation}
    freeze_rel = preseal.freeze_path(instrumentation)
    freeze_path = root / freeze_rel
    freeze_path.parent.mkdir(parents=True)
    freeze_path.write_bytes(preseal.canonical_json(artifact_body) + b"\n")
    freeze = _commit(root, "freeze")
    # Mutate earlier content AND append SEALED — not a pure append.
    declaration.write_text(
        f"mutated base\n| 2026-07-24 | `{instrumentation}` | `{freeze}` | `SEALED` |\n",
        encoding="utf-8",
    )
    _commit(root, "bad seal")
    artifact = {"instrumentation_head": instrumentation}
    with pytest.raises(preseal.VerificationError, match="append-only"):
        preseal.verify_authority_landing(root, artifact)


def test_historical_s3_identities_remain_immutable() -> None:
    # Spent chain coordinates must remain the documented exact SHAs.
    assert len(S3["I"]) == 40 and len(S3["F"]) == 40 and len(S3["S"]) == 40
    evidence = (
        ROOT
        / "docs/modules/semantic/research/evidence"
        / f"h0_phase_a_{S3['I']}"
        / "result.json"
    )
    assert evidence.is_file()
    result = json.loads(evidence.read_text(encoding="utf-8"))
    assert result == {
        "result": "provenance_invalid",
        "schema": "h0_phase_a_execution_v1",
    }


def test_launch_hygiene_gate_still_single_source() -> None:
    assert gate.PREDICATE_SOURCE == "run_h0_phase_a.assert_no_preexisting_build_tree"
    assert gate.controller is controller


def test_registration_v3_downstream_structurally_reachable() -> None:
    value = matrix.load_matrix()
    downstream = value["registration_v3_downstream"]
    assert downstream["registration_schema"] == "h0_gctm_guarantee_registration_v3"
    assert (
        downstream["consumer_universe"] == "gctm_runtime_native_candidate_universe_v1"
    )
    assert downstream["actual_guarantee_in_this_pr"] is False
