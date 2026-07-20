"""Prospective H0 repair/qualification gate contracts."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "scripts/tools"
WORKFLOW = ROOT / ".github/workflows/h0_qualification.yml"
sys.path.insert(0, TOOLS.as_posix())

import build_h0_preseal_freeze as freezer  # noqa: E402
import check_h0_phase_a_archives as archive_corpus  # noqa: E402
import check_h0_repair_acceptance_matrix as matrix  # noqa: E402
import qualify_h0_phase_a as qualification  # noqa: E402
import verify_h0_phase_a_archive as archive  # noqa: E402
import run_h0_phase_a as controller  # noqa: E402


def test_repair_acceptance_matrix_is_complete() -> None:
    value = matrix.load_matrix()
    matrix.validate_matrix(value)
    assert [gate["id"] for gate in value["gates"]] == list(matrix.GATES)
    assert value["qualification"]["authority"] == "non_authoritative"


def test_repair_acceptance_matrix_rejects_every_requirement_mutation() -> None:
    source = matrix.load_matrix()
    for gate_index, gate in enumerate(source["gates"]):
        requirements = gate["required"]
        assert isinstance(requirements, list)
        for requirement_index in range(len(requirements)):
            for mutation in ("delete", "replace", "reorder", "add"):
                value = copy.deepcopy(source)
                required = value["gates"][gate_index]["required"]
                assert isinstance(required, list)
                if mutation == "delete":
                    required.pop(requirement_index)
                elif mutation == "replace":
                    required[requirement_index] = "optional_note"
                elif mutation == "reorder":
                    other = (requirement_index + 1) % len(required)
                    required[requirement_index], required[other] = (
                        required[other],
                        required[requirement_index],
                    )
                else:
                    required.append("optional_note")
                with pytest.raises(matrix.MatrixError, match="requirements differ"):
                    matrix.validate_matrix(value)


def test_qualification_workspace_refuses_authoritative_or_ambiguous_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "build").mkdir()
    evidence = root / "docs/modules/semantic/research/evidence"
    evidence.mkdir(parents=True)
    with pytest.raises(qualification.QualificationError, match="authoritative"):
        qualification._workspace(root, root / "build/h0_phase_a")
    with pytest.raises(qualification.QualificationError, match="authoritative"):
        qualification._workspace(root, evidence / "qualification")
    with pytest.raises(qualification.QualificationError, match="below"):
        qualification._workspace(root, root / "scratch")
    workspace = qualification._workspace(root, root / "build/h0_qualification/run-1")
    assert workspace.is_dir()


def test_build_identity_matches_project_enabled_languages(tmp_path: Path) -> None:
    build = tmp_path / "build"
    build.mkdir()
    suffix = qualification.sysconfig.get_config_var("EXT_SUFFIX")
    assert isinstance(suffix, str) and suffix
    extension = build / f"saccade_tracking_ext{suffix}"
    plugin = build / "libsaccade_scan_plugin.so"
    cxx = tmp_path / "cxx"
    cuda = tmp_path / "nvcc"
    cmake = tmp_path / "cmake"
    for path in (extension, plugin, cxx, cuda, cmake):
        path.write_bytes(b"fixture")
    (build / "CMakeCache.txt").write_text(
        "\n".join(
            (
                f"CMAKE_CXX_COMPILER:FILEPATH={cxx}",
                f"CMAKE_CUDA_COMPILER:FILEPATH={cuda}",
                f"CMAKE_COMMAND:INTERNAL={cmake}",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    identity = qualification._build_identity(build, build, Path(sys.executable))

    assert set(identity["compilers"]) == {"cxx", "cuda"}
    assert identity["cmake"]["path"] == cmake.as_posix()


def test_qualification_failure_probe_is_truthful_but_non_authoritative() -> None:
    probe = qualification._failure_probe()
    assert probe["qualification_only"] is True
    assert probe["failure"]["stage"] == "checkpoint_T1"
    assert {
        key: value for key, value in probe["row"].items() if key != "monotonic_ns"
    } == {
        "cause": "inventory_mismatch",
        "digest": None,
        "events_after": [],
        "events_before": [],
        "inventory_comparison_executed": True,
        "inventory_equal": False,
        "name": "T1",
        "observed_digest": "0" * 64,
        "state": "failed",
    }


def test_qualification_runner_uses_an_explicit_synthetic_child() -> None:
    vector = qualification.qualification_runner_argv(
        Path("/tmp/build"),
        Path("/tmp/build/extension.so"),
        Path("/tmp/build/plugin.so"),
    )
    assert vector[:3] == [sys.executable, "-I", "-B"]
    assert vector[3].endswith("scripts/tools/qualify_h0_phase_a_child.py")
    assert "run_h0_phase_a_child.py" not in vector


def test_qualification_t1_verdict_semantics_uses_controller_producer() -> None:
    inventory = {"digest": "a" * 64, "records": []}
    row = controller._checkpoint_inventory_verdict("T1", inventory, dict(inventory))
    assert row["state"] == "completed"
    assert row["inventory_equal"] is True
    assert row["observed_digest"] == inventory["digest"]


def _build_tool_bound_inputs_fixture(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    cmake = {
        "length": 1,
        "path": "/fixture/cmake",
        "sha256": "a" * 64,
    }
    cxx = {
        "length": 2,
        "path": "/fixture/cxx",
        "sha256": "b" * 64,
    }
    loader = {
        "length": 3,
        "logical_path": "/fixture/libloader.so",
        "realpath": "/fixture/libloader.so",
        "sha256": "c" * 64,
        "symlink_chain": [],
    }
    binding = {
        "build_environment_path": f"{tmp_path}/.venv/bin:/usr/bin:/bin",
        "digest": "fixture",
        "loader_closure": [loader],
        "resolver": controller.BUILD_TOOL_BINDING_RESOLVER,
        "schema": controller.BUILD_TOOL_BINDING_SCHEMA,
        "tools": [
            {
                "command": "c++",
                "record": {
                    "logical_path": cxx["path"],
                    "length": cxx["length"],
                    "realpath": cxx["path"],
                    "sha256": cxx["sha256"],
                    "symlink_chain": [],
                },
                "role": "cxx",
            },
            {
                "command": "cmake",
                "record": {
                    "logical_path": cmake["path"],
                    "length": cmake["length"],
                    "realpath": cmake["path"],
                    "sha256": cmake["sha256"],
                    "symlink_chain": [],
                },
                "role": "cmake",
            },
        ],
    }
    binding["digest"] = controller._binding_digest(binding)
    contribution: dict[str, object] = {
        "build_tool_binding": binding,
        "digest": "",
        "schema": freezer.BUILD_TOOL_BOUND_INPUTS_SCHEMA,
        "tool_runtime": sorted(
            [
                *(item["record"] for item in binding["tools"]),
                *binding["loader_closure"],
            ],
            key=lambda record: record["logical_path"].encode("utf-8"),
        ),
    }
    contribution["digest"] = freezer.build_tool_bound_inputs_digest(contribution)
    return contribution, {"cmake": cmake, "compilers": {"cxx": cxx}}


def test_freezer_build_tool_producer_returns_the_exact_tool_runtime_contribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contribution, _identity = _build_tool_bound_inputs_fixture(tmp_path)
    binding = contribution["build_tool_binding"]
    assert isinstance(binding, dict)
    calls: list[object] = []
    monkeypatch.setattr(
        controller,
        "resolve_build_tool_binding",
        lambda root, **kwargs: calls.append((root, kwargs)) or binding,
    )
    monkeypatch.setattr(
        controller,
        "_validate_build_tool_binding_shape",
        lambda observed: calls.append(observed),
    )

    result = freezer.derive_build_tool_bound_inputs(
        tmp_path, ldd_path=Path("/usr/bin/ldd")
    )

    assert result == contribution
    assert len(calls) == 2


def test_qualification_build_tool_dry_run_uses_freezer_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contribution, identity = _build_tool_bound_inputs_fixture(tmp_path)
    calls: list[object] = []
    monkeypatch.setattr(
        freezer,
        "derive_build_tool_bound_inputs",
        lambda root, **kwargs: calls.append((root, kwargs)) or contribution,
    )
    monkeypatch.setattr(
        controller,
        "validate_resolved_build_tool_identity",
        lambda observed, **kwargs: calls.append((observed, kwargs)),
    )
    assert (
        qualification._derive_qualification_build_tool_bound_inputs(
            tmp_path, identity, ldd=Path("/usr/bin/ldd")
        )
        == contribution
    )
    assert len(calls) == 2


@pytest.mark.parametrize("removed_member", ["primary", "closure"])
def test_qualification_rejects_missing_assembler_build_tool_record(
    tmp_path: Path, removed_member: str
) -> None:
    contribution, identity = _build_tool_bound_inputs_fixture(tmp_path)
    binding = contribution["build_tool_binding"]
    assert isinstance(binding, dict)
    tools = binding["tools"]
    closure = binding["loader_closure"]
    assert isinstance(tools, list) and isinstance(closure, list)
    primary_record = tools[0]["record"]
    closure_record = closure[0]
    assert primary_record not in closure
    assert closure_record in closure
    mutated = copy.deepcopy(contribution)
    records = mutated["tool_runtime"]
    assert isinstance(records, list)
    expected_removed = primary_record if removed_member == "primary" else closure_record
    assert expected_removed in records
    removed = records.pop(records.index(expected_removed))
    assert removed == expected_removed
    if removed_member == "primary":
        assert removed == tools[0]["record"]
    else:
        assert removed == closure[0]
    # The proof must fail even if a defective producer recomputed its own digest.
    mutated["digest"] = freezer.build_tool_bound_inputs_digest(mutated)

    with pytest.raises(
        qualification.QualificationError,
        match="does not exactly equal the binding contribution",
    ):
        qualification._validate_qualification_build_tool_bound_inputs(mutated, identity)


def test_qualification_report_binds_resolved_repository_identity(
    tmp_path: Path,
) -> None:
    identity = qualification._repository_identity(
        ROOT, "agent/h0-repair-qualification-gates"
    )
    report = {"result": "passed", **identity}
    path = qualification._write_report(tmp_path, report)
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["repository_head_sha"] == identity["repository_head_sha"]
    assert written["repository_tree_sha"] == identity["repository_tree_sha"]
    assert written["requested_ref"] == identity["requested_ref"]

    with pytest.raises(qualification.QualificationError, match="repository identity"):
        qualification._write_report(
            tmp_path,
            {
                "result": "passed",
                "repository_tree_sha": identity["repository_tree_sha"],
                "requested_ref": identity["requested_ref"],
            },
        )


def test_qualification_failure_report_retains_repository_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    workspace = tmp_path / "workspace"
    identity = {
        "repository_head_sha": "a" * 40,
        "repository_tree_sha": "b" * 40,
        "requested_ref": "repair-head",
    }
    monkeypatch.setattr(qualification, "_repository_identity", lambda *_args: identity)
    monkeypatch.setattr(
        qualification,
        "_tool",
        lambda _name: (_ for _ in ()).throw(
            qualification.QualificationError("missing controlled-host tool")
        ),
    )

    with pytest.raises(qualification.QualificationError, match="controlled-host"):
        qualification.run_qualification(
            root,
            workspace,
            requested_ref="repair-head",
            timeout=1.0,
        )

    report = json.loads(
        (workspace / qualification.REPORT_NAME).read_text(encoding="utf-8")
    )
    assert report["result"] == "failed"
    assert {key: report[key] for key in identity} == identity


def test_qualification_workflow_binds_and_publishes_qualified_head() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "id: qualified_commit" in workflow
    assert 'qualified_sha="$(git rev-parse --verify HEAD^{commit})"' in workflow
    assert '--requested-ref "$REQUESTED_REF"' in workflow
    assert (
        "name: h0-qualification-${{ steps.qualified_commit.outputs.sha }}" in workflow
    )
    assert workflow.count("if: ${{ !cancelled() }}") == 2


def test_archive_registry_verifies_immutable_v1_packet_without_execution_binding() -> (
    None
):
    evidence = (
        ROOT
        / "docs/modules/semantic/research/evidence"
        / "h0_phase_a_6ed30243554edfc898de32916298aa863673fced"
    )
    report = archive.verify_archive(evidence)
    assert report["valid"] is True
    assert report["codec_schema"] == "h0_phase_a_execution_v1"
    execution_paths = {path for path, _identity in freezer.IMPLEMENTATION_IDENTITIES}
    assert "scripts/tools/verify_h0_phase_a_archive.py" not in execution_paths
    assert "scripts/tools/qualify_h0_phase_a.py" not in execution_paths
    assert "scripts/tools/qualify_h0_phase_a_child.py" not in execution_paths


def test_archive_corpus_discovers_every_committed_phase_a_root() -> None:
    roots = archive_corpus.archive_roots()
    assert [root.name for root in roots] == [
        "h0_phase_a_1a8c13a890b3490bb7aa50dc2ab491db89b8b474",
        "h0_phase_a_42121c064cd1a3c4202e114cc6f4d8866a9e6af0",
        "h0_phase_a_6ed30243554edfc898de32916298aa863673fced",
        "h0_phase_a_9712e951bd4b8ce5e5382f48cd0b7ca68686a720",
    ]


def test_preseal_sealability_gate_rejects_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = "a" * 40

    def unsealable(command_line: list[str]) -> dict[str, object]:
        return {
            "schema": "h0_preseal_static_sealability_v1",
            "instrumentation_head": head,
            "instrumentation_tree": "b" * 40,
            "projection_admitted": False,
            "sealable": False,
            "problems": ["projection not admitted; runtime paths outside …"],
        }

    monkeypatch.setattr(freezer, "check_preseal_sealability", unsealable)
    with pytest.raises(
        qualification.QualificationError, match="static sealability failed"
    ):
        qualification._check_preseal_sealability(head)

    def contradictory(command_line: list[str]) -> dict[str, object]:
        # problems non-empty must fail even if sealable claims true.
        return {
            "schema": "h0_preseal_static_sealability_v1",
            "instrumentation_head": head,
            "instrumentation_tree": "b" * 40,
            "projection_admitted": True,
            "sealable": True,
            "problems": ["policy_base_tree mismatch against §2"],
        }

    monkeypatch.setattr(freezer, "check_preseal_sealability", contradictory)
    with pytest.raises(
        qualification.QualificationError, match="static sealability failed"
    ):
        qualification._check_preseal_sealability(head)


def test_preseal_sealability_gate_rejects_head_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def sealable_elsewhere(command_line: list[str]) -> dict[str, object]:
        return {
            "schema": "h0_preseal_static_sealability_v1",
            "instrumentation_head": "c" * 40,
            "instrumentation_tree": "b" * 40,
            "projection_admitted": True,
            "sealable": True,
            "problems": [],
        }

    monkeypatch.setattr(freezer, "check_preseal_sealability", sealable_elsewhere)
    with pytest.raises(qualification.QualificationError, match="different head"):
        qualification._check_preseal_sealability("a" * 40)
    with pytest.raises(qualification.QualificationError, match="different head"):
        qualification._check_preseal_sealability(None)


def test_passing_report_requires_canonical_step_sequence() -> None:
    canonical = [{"name": name, "state": "passed"} for name in qualification.STEP_NAMES]
    qualification._require_canonical_steps(canonical)

    truncated = canonical[:-1]
    with pytest.raises(qualification.QualificationError, match="step sequence drift"):
        qualification._require_canonical_steps(truncated)

    reordered = [canonical[1], canonical[0], *canonical[2:]]
    with pytest.raises(qualification.QualificationError, match="step sequence drift"):
        qualification._require_canonical_steps(reordered)

    renamed = [dict(step) for step in canonical]
    renamed[-1]["name"] = "freeze_assembly"
    with pytest.raises(qualification.QualificationError, match="step sequence drift"):
        qualification._require_canonical_steps(renamed)

    failed_state = [dict(step) for step in canonical]
    failed_state[3]["state"] = "failed"
    with pytest.raises(qualification.QualificationError, match="step sequence drift"):
        qualification._require_canonical_steps(failed_state)


def test_canonical_steps_close_over_matrix_required_steps() -> None:
    # Runner canonical sequence, matrix JSON, and matrix checker must be the
    # same exact tuple — no subsequence tolerance.
    required = tuple(matrix.load_matrix()["qualification"]["required_steps"])
    assert required == qualification.STEP_NAMES
    assert required == matrix.QUALIFICATION_STEPS
    assert required[-1] == "preseal_freeze_assembly"
