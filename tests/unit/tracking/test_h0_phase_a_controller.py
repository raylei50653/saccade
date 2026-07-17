"""A7 + RC1 Phase-A controller substrate tests.

Hermetic only: no real build, capture, or Phase A. Dummy paths and synthetic
evidence exercise the frozen contract, digests, C/D/V matrix, and fail-closed
verifier independently of controller self-report.
"""

from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "tools"))

from run_h0_phase_a import (  # noqa: E402
    CDV_RESULT_MATRIX,
    INOTIFY_MASK,
    INOTIFY_MASK_NAMES,
    PARENT_COMMAND_VECTOR,
    RESULT_ENUM,
    RUN_IDS,
    ContractError,
    assemble_synthetic_evidence,
    bound_inputs_inventory_digest,
    build_bound_inputs_v1,
    build_child_input,
    build_controller_plan,
    cdv_matrix_row,
    cdv_paths_for_result,
    child_command_vector,
    child_environment,
    classify_inotify_event,
    compute_sequence_input_digest,
    empty_mutation_observation,
    environment_digest,
    evaluator_argv,
    load_schema,
    operator_main,
    ordered_run_plan,
    parent_command_vector,
    require_physical_abs_path,
    run_hermetic_contract_session,
    select_controller_result,
    sequence_file_record_bytes,
    validate_controller_input,
)
from run_h0_phase_a_child import (  # noqa: E402
    execute_child_contract,
    parse_child_argv,
    verify_environment_exact,
)
from verify_h0_phase_a import verify_evidence_package  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture()
def physical_root(tmp_path: Path) -> Path:
    """Absolute non-symlink repository-like root with library dirs and sequence."""
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    (root / "scripts" / "tools").mkdir(parents=True)
    (root / "scripts" / "tools" / "run_h0_phase_a_child.py").write_text(
        "# stub\n", encoding="utf-8"
    )
    (root / "build" / "h0_phase_a").mkdir(parents=True)

    for name in ("tensorrt", "pytorch", "cuda_lib64"):
        d = (tmp_path / name).resolve()
        d.mkdir()
        (d / "marker").write_text(name, encoding="utf-8")

    seq = root / "datasets" / "MOT17" / "train" / "MOT17-04-SDP"
    (seq / "img1").mkdir(parents=True)
    (seq / "gt").mkdir()
    (seq / "det").mkdir()
    (seq / "seqinfo.ini").write_text(
        "[Sequence]\nname=MOT17-04-SDP\n", encoding="utf-8"
    )
    (seq / "img1" / "000001.jpg").write_bytes(b"\xff\xd8fakejpeg")
    (seq / "gt" / "gt.txt").write_text("should-be-excluded\n", encoding="utf-8")
    (seq / "det" / "det.txt").write_text("should-be-excluded\n", encoding="utf-8")
    return root


def _lib_dirs(tmp_path: Path) -> dict[str, str]:
    return {
        "tensorrt_lib_dir": str((tmp_path / "tensorrt").resolve()),
        "pytorch_lib_dir": str((tmp_path / "pytorch").resolve()),
        "cuda_lib64_dir": str((tmp_path / "cuda_lib64").resolve()),
    }


def _head() -> str:
    return "a" * 40


def make_controller_input(physical_root: Path, tmp_path: Path) -> dict[str, Any]:
    libs = _lib_dirs(tmp_path)
    seq = compute_sequence_input_digest(
        physical_root / "datasets" / "MOT17" / "train" / "MOT17-04-SDP"
    )
    bi = build_bound_inputs_v1(
        instrumentation_head=_head(),
        repository=[
            {
                "mode": "100644",
                "object_type": "blob",
                "object_id": "b" * 40,
                "path": "scripts/tools/run_h0_phase_a.py",
                "byte_length": 1,
                "content_sha256_or_target": _sha(b"x"),
            }
        ],
        models_engines=[
            {
                "logical_path": "models/dummy.engine",
                "realpath": str((tmp_path / "tensorrt" / "marker").resolve()),
                "symlink_chain": [],
                "byte_length": 8,
                "sha256": _sha(b"tensorrt"),
            }
        ],
        sequence=seq,
        tool_runtime_inputs=[
            {
                "name": "python",
                "path": str(physical_root / ".venv" / "bin" / "python"),
                "sha256": _sha(b"#!/bin/sh\n"),
            }
        ],
    )
    return {
        "schema_version": "h0_phase_a_controller_input_v1",
        "instrumentation_head": _head(),
        "repository_root": str(physical_root),
        "tensorrt_lib_dir": libs["tensorrt_lib_dir"],
        "pytorch_lib_dir": libs["pytorch_lib_dir"],
        "cuda_lib64_dir": libs["cuda_lib64_dir"],
        "cuda_device_uuid": "GPU-00000000-0000-0000-0000-000000000001",
        "cuda_pci_bus_id": "0000:01:00.0",
        "bound_inputs": bi,
        "capture_run_uuids": {
            rid: f"00000000-0000-4000-8000-{i:012d}" for i, rid in enumerate(RUN_IDS)
        },
    }


# ---------------------------------------------------------------------------
# Conforming cases
# ---------------------------------------------------------------------------


def test_schema_file_loads_and_is_strict() -> None:
    schema = load_schema()
    assert schema["schema_version"] == "h0_phase_a_execution_v1"
    assert schema["additionalProperties"] is False
    assert set(schema["path_sets"]["C"])  # non-empty
    assert len(schema["frozen_constants"]["result_enum"]) == 11
    assert schema["frozen_constants"]["inotify_mask"] == 4046


def test_canonical_dry_run_contract_validates(
    physical_root: Path, tmp_path: Path
) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    validated = validate_controller_input(cin)
    assert validated["instrumentation_head"] == _head()
    plan = build_controller_plan(
        repository_root=cin["repository_root"],
        instrumentation_head=cin["instrumentation_head"],
        cuda_device_uuid=cin["cuda_device_uuid"],
        tensorrt_lib_dir=cin["tensorrt_lib_dir"],
        pytorch_lib_dir=cin["pytorch_lib_dir"],
        cuda_lib64_dir=cin["cuda_lib64_dir"],
    )
    assert plan["ordered_run_plan"] == list(RUN_IDS)
    assert plan["deadline_seconds"] == 3600
    assert plan["parent_command_vector"] == list(PARENT_COMMAND_VECTOR)


def test_parent_and_child_identical_command_vectors(
    physical_root: Path, tmp_path: Path
) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    root = cin["repository_root"]
    for run_id in RUN_IDS:
        parent_vec = child_command_vector(root, run_id)
        # Child rebuilds via the same pure function (no free choice).
        child_vec = child_command_vector(root, run_id)
        assert parent_vec == child_vec
        assert parent_vec[0] == f"{root}/.venv/bin/python"
        assert parent_vec[1:3] == ["-I", "-B"]
        assert parent_vec[3] == f"{root}/scripts/tools/run_h0_phase_a_child.py"
        assert parent_vec[4:6] == ["--run-id", run_id]


def test_environment_table_and_digests_rebuild(
    physical_root: Path, tmp_path: Path
) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    plan = build_controller_plan(
        repository_root=cin["repository_root"],
        instrumentation_head=cin["instrumentation_head"],
        cuda_device_uuid=cin["cuda_device_uuid"],
        tensorrt_lib_dir=cin["tensorrt_lib_dir"],
        pytorch_lib_dir=cin["pytorch_lib_dir"],
        cuda_lib64_dir=cin["cuda_lib64_dir"],
    )
    for run_id in RUN_IDS:
        env = plan["environment_tables_by_run"][run_id]
        again = child_environment(
            repository_root=cin["repository_root"],
            run_id=run_id,
            instrumentation_head=cin["instrumentation_head"],
            cuda_device_uuid=cin["cuda_device_uuid"],
            tensorrt_lib_dir=cin["tensorrt_lib_dir"],
            pytorch_lib_dir=cin["pytorch_lib_dir"],
            cuda_lib64_dir=cin["cuda_lib64_dir"],
        )
        assert env == again
        assert environment_digest(env) == environment_digest(again)
        assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert env["PYTHONHASHSEED"] == "0"
        parts = env["LD_LIBRARY_PATH"].split(":")
        assert len(parts) == 4
        assert parts[0].endswith("/build/h0_phase_a")


def test_verifier_accepts_untampered_synthetic_evidence(
    physical_root: Path, tmp_path: Path
) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    evidence = assemble_synthetic_evidence(controller_input=cin, result="phase_a_pass")
    verdict = verify_evidence_package(evidence)
    assert verdict["accepted"] is True
    assert verdict["result"] == "phase_a_pass"
    assert verdict["rejection_codes"] == []


@pytest.mark.parametrize("result", list(RESULT_ENUM))
def test_cdv_matrix_every_terminal_combination(
    physical_root: Path, tmp_path: Path, result: str
) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    row = cdv_matrix_row(result)
    assert row["result"] == result
    required_sets, forbidden_sets = CDV_RESULT_MATRIX[result]
    assert tuple(row["required_sets"]) == required_sets
    assert tuple(row["forbidden_sets"]) == forbidden_sets
    req, forb = cdv_paths_for_result(result)
    assert set(req).isdisjoint(set(forb))
    # Full universe = C ∪ D ∪ V
    universe = set()
    for name in ("C", "D", "V"):
        from run_h0_phase_a import PATH_SETS

        universe |= set(PATH_SETS[name])
    assert set(req) | set(forb) == universe or result in {
        "packet_invalid",
        "phase_a_pass",
    }
    if result in {"packet_invalid", "phase_a_pass"}:
        assert set(req) == universe
        assert forb == []

    child_exit = "completed"
    if result == "runner_nonzero":
        child_exit = "runner_nonzero"
    evidence = assemble_synthetic_evidence(
        controller_input=cin,
        result=result,
        child_exit_class=child_exit,
    )
    # For phase_a_pass / packet_invalid require completed children.
    if result in {"phase_a_pass", "packet_invalid", "capture_perturbs_policy"}:
        evidence = assemble_synthetic_evidence(
            controller_input=cin, result=result, child_exit_class="completed"
        )
    verdict = verify_evidence_package(evidence)
    assert verdict["accepted"] is True, verdict
    assert verdict["result"] == result


def test_sequence_input_digest_excludes_gt_and_det(
    physical_root: Path,
) -> None:
    seq = compute_sequence_input_digest(
        physical_root / "datasets" / "MOT17" / "train" / "MOT17-04-SDP"
    )
    paths = [r["relative_path"] for r in seq["file_records"]]
    assert "seqinfo.ini" in paths
    assert "img1/000001.jpg" in paths
    assert not any(p.startswith("gt/") for p in paths)
    assert not any(p.startswith("det/") for p in paths)
    # Deterministic recompute
    again = compute_sequence_input_digest(
        physical_root / "datasets" / "MOT17" / "train" / "MOT17-04-SDP"
    )
    assert again["aggregate_sha256"] == seq["aggregate_sha256"]


def test_bound_inputs_digest_stable(physical_root: Path, tmp_path: Path) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    bi = cin["bound_inputs"]
    assert bi["inventory_digest"] == bound_inputs_inventory_digest(bi)


def test_child_execute_contract_with_hook(physical_root: Path, tmp_path: Path) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    child_in = build_child_input(
        repository_root=cin["repository_root"],
        instrumentation_head=cin["instrumentation_head"],
        run_id="01_capture_on_1",
        capture_run_uuid=cin["capture_run_uuids"]["01_capture_on_1"],
        cuda_device_uuid=cin["cuda_device_uuid"],
        tensorrt_lib_dir=cin["tensorrt_lib_dir"],
        pytorch_lib_dir=cin["pytorch_lib_dir"],
        cuda_lib64_dir=cin["cuda_lib64_dir"],
        bound_inputs_digest=cin["bound_inputs"]["inventory_digest"],
        sequence_input_digest=cin["bound_inputs"]["sequence"]["aggregate_sha256"],
    )
    called: list[str] = []

    def hook(payload: dict[str, Any]) -> None:
        called.append(payload["run_id"])
        assert payload["trace_lifecycle"]["clear_research_h0_bridge_trace"] is True

    result = execute_child_contract(
        child_in,
        actual_env=child_in["environment"],
        eval_hook=hook,
    )
    assert result["exit_class"] == "completed"
    assert called == ["01_capture_on_1"]


def test_hermetic_session_spawn_hook(physical_root: Path, tmp_path: Path) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    seen: list[str] = []

    def spawn_hook(**kwargs: Any) -> dict[str, Any]:
        run_id = kwargs["run_id"]
        seen.append(run_id)
        spec = kwargs["popen_spec"]
        assert spec["shell"] is False
        assert spec["close_fds"] is True
        assert spec["start_new_session"] is True
        assert spec["cwd"] == cin["repository_root"]
        child_in = build_child_input(
            repository_root=cin["repository_root"],
            instrumentation_head=cin["instrumentation_head"],
            run_id=run_id,
            capture_run_uuid=cin["capture_run_uuids"][run_id],
            cuda_device_uuid=cin["cuda_device_uuid"],
            tensorrt_lib_dir=cin["tensorrt_lib_dir"],
            pytorch_lib_dir=cin["pytorch_lib_dir"],
            cuda_lib64_dir=cin["cuda_lib64_dir"],
            bound_inputs_digest=cin["bound_inputs"]["inventory_digest"],
            sequence_input_digest=cin["bound_inputs"]["sequence"]["aggregate_sha256"],
        )
        return execute_child_contract(
            child_in,
            actual_env=spec["env"],
            eval_hook=lambda _p: None,
        )

    evidence = run_hermetic_contract_session(
        cin, result="phase_a_pass", spawn_hook=spawn_hook
    )
    assert seen == list(RUN_IDS)
    verdict = verify_evidence_package(evidence)
    assert verdict["accepted"] is True


def test_operator_main_refuses_without_authority() -> None:
    assert operator_main([]) == 2
    assert operator_main(["--help"]) == 0
    assert operator_main(["-h"]) == 0
    assert operator_main(["--preset", "x"]) == 2


def test_inotify_mask_frozen() -> None:
    assert INOTIFY_MASK == 4046
    assert len(INOTIFY_MASK_NAMES) == 9


def test_result_enum_selection_order() -> None:
    # First applicable wins.
    assert (
        select_controller_result(
            {"provenance_invalid": True, "build_failed": True, "phase_a_pass": True}
        )
        == "provenance_invalid"
    )
    assert select_controller_result({"phase_a_pass": True}) == "phase_a_pass"
    assert select_controller_result({}) == "unclassified_execution_failure"


# ---------------------------------------------------------------------------
# Fail-closed cases
# ---------------------------------------------------------------------------


def test_missing_required_field(physical_root: Path, tmp_path: Path) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    del cin["cuda_device_uuid"]
    with pytest.raises(ContractError) as ei:
        validate_controller_input(cin)
    assert ei.value.code == "missing_required_field"


def test_extra_unknown_field(physical_root: Path, tmp_path: Path) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    cin["extra_free_choice"] = "nope"
    with pytest.raises(ContractError) as ei:
        validate_controller_input(cin)
    assert ei.value.code == "unknown_field"


def test_illegal_enum_run_id(physical_root: Path) -> None:
    with pytest.raises(ContractError) as ei:
        child_command_vector(str(physical_root), "99_not_a_run")
    assert ei.value.code == "illegal_enum"


def test_command_vector_order_change_rejected_by_verifier(
    physical_root: Path, tmp_path: Path
) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    evidence = assemble_synthetic_evidence(controller_input=cin, result="phase_a_pass")
    # Swap two tokens in parent vector.
    evidence["parent_command_vector"] = list(
        reversed(evidence["parent_command_vector"])
    )
    verdict = verify_evidence_package(evidence)
    assert verdict["accepted"] is False
    assert (
        "command_vector_mismatch" in verdict["rejection_codes"]
        or "command_vector_order" in verdict["rejection_codes"]
    )


def test_executable_path_change_rejected(physical_root: Path, tmp_path: Path) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    evidence = assemble_synthetic_evidence(controller_input=cin, result="phase_a_pass")
    for run_id in RUN_IDS:
        vec = list(evidence["child_command_vectors"][run_id])
        vec[0] = "/usr/bin/python3"
        evidence["child_command_vectors"][run_id] = vec
        evidence["child_results"][run_id]["command_vector"] = vec
    verdict = verify_evidence_package(evidence)
    assert verdict["accepted"] is False
    assert any(
        c in verdict["rejection_codes"]
        for c in ("executable_path_mismatch", "command_vector_mismatch")
    )


def test_argument_insert_delete_reorder_rejected(
    physical_root: Path, tmp_path: Path
) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    evidence = assemble_synthetic_evidence(controller_input=cin, result="phase_a_pass")
    vec = list(evidence["evaluator_argv_by_run"]["00_capture_off"])
    # reorder a pair
    i = vec.index("--max-frames")
    vec[i], vec[i + 2] = vec[i + 2], vec[i]
    evidence["evaluator_argv_by_run"]["00_capture_off"] = vec
    evidence["child_results"]["00_capture_off"]["evaluator_argv"] = vec
    verdict = verify_evidence_package(evidence)
    assert verdict["accepted"] is False
    assert "argument_mismatch" in verdict["rejection_codes"]


def test_environment_missing_key(physical_root: Path, tmp_path: Path) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    env = child_environment(
        repository_root=cin["repository_root"],
        run_id="00_capture_off",
        instrumentation_head=cin["instrumentation_head"],
        cuda_device_uuid=cin["cuda_device_uuid"],
        tensorrt_lib_dir=cin["tensorrt_lib_dir"],
        pytorch_lib_dir=cin["pytorch_lib_dir"],
        cuda_lib64_dir=cin["cuda_lib64_dir"],
    )
    del env["TZ"]
    with pytest.raises(ContractError) as ei:
        environment_digest(env)
    assert ei.value.code == "environment_mismatch"


def test_environment_extra_key(physical_root: Path, tmp_path: Path) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    env = child_environment(
        repository_root=cin["repository_root"],
        run_id="00_capture_off",
        instrumentation_head=cin["instrumentation_head"],
        cuda_device_uuid=cin["cuda_device_uuid"],
        tensorrt_lib_dir=cin["tensorrt_lib_dir"],
        pytorch_lib_dir=cin["pytorch_lib_dir"],
        cuda_lib64_dir=cin["cuda_lib64_dir"],
    )
    with pytest.raises(ContractError):
        verify_environment_exact(env, {**env, "PYTHONPATH": "/evil"})


def test_environment_value_drift(physical_root: Path, tmp_path: Path) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    env = child_environment(
        repository_root=cin["repository_root"],
        run_id="00_capture_off",
        instrumentation_head=cin["instrumentation_head"],
        cuda_device_uuid=cin["cuda_device_uuid"],
        tensorrt_lib_dir=cin["tensorrt_lib_dir"],
        pytorch_lib_dir=cin["pytorch_lib_dir"],
        cuda_lib64_dir=cin["cuda_lib64_dir"],
    )
    drifted = dict(env)
    drifted["PYTHONHASHSEED"] = "1"
    with pytest.raises(ContractError) as ei:
        verify_environment_exact(env, drifted)
    assert ei.value.code == "env_mismatch"


def test_working_directory_in_popen_spec(physical_root: Path, tmp_path: Path) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    from run_h0_phase_a import child_popen_spec

    spec = child_popen_spec(
        repository_root=cin["repository_root"],
        run_id="00_capture_off",
        instrumentation_head=cin["instrumentation_head"],
        cuda_device_uuid=cin["cuda_device_uuid"],
        tensorrt_lib_dir=cin["tensorrt_lib_dir"],
        pytorch_lib_dir=cin["pytorch_lib_dir"],
        cuda_lib64_dir=cin["cuda_lib64_dir"],
        stdout_path="/dev/null",
        stderr_path="/dev/null",
    )
    assert spec["cwd"] == cin["repository_root"]
    # Drifted cwd would be a free choice — verifier reconstructs from vectors.
    evidence = assemble_synthetic_evidence(controller_input=cin, result="phase_a_pass")
    # Corrupt PATH which encodes root.
    evidence["environment_tables_by_run"]["00_capture_off"]["PATH"] = "/evil/bin"
    verdict = verify_evidence_package(evidence)
    assert verdict["accepted"] is False
    assert "environment_mismatch" in verdict["rejection_codes"]


def test_bound_input_digest_tamper(physical_root: Path, tmp_path: Path) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    evidence = assemble_synthetic_evidence(controller_input=cin, result="phase_a_pass")
    evidence["bound_inputs"]["inventory_digest"] = "0" * 64
    verdict = verify_evidence_package(evidence)
    assert verdict["accepted"] is False
    assert "bound_input_digest_mismatch" in verdict["rejection_codes"]


def test_sequence_input_digest_mismatch(physical_root: Path, tmp_path: Path) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    evidence = assemble_synthetic_evidence(controller_input=cin, result="phase_a_pass")
    # Top-level sequence digest must be a distinct object from bound_inputs.sequence.
    evidence["sequence_input_digest"] = copy.deepcopy(evidence["sequence_input_digest"])
    evidence["sequence_input_digest"]["aggregate_sha256"] = "1" * 64
    verdict = verify_evidence_package(evidence)
    assert verdict["accepted"] is False
    assert "sequence_input_digest_mismatch" in verdict["rejection_codes"]


def test_symlink_path_rejected(tmp_path: Path) -> None:
    real = (tmp_path / "real").resolve()
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(ContractError) as ei:
        require_physical_abs_path(str(link), field="repository_root")
    assert ei.value.code == "symlink_or_non_canonical_path"


def test_path_traversal_rejected(physical_root: Path) -> None:
    with pytest.raises(ContractError):
        require_physical_abs_path(str(physical_root / "a" / ".." / "b"), field="x")
    with pytest.raises(ContractError):
        evaluator_argv("../escape/runs/00_capture_off")


def test_inotify_mask_mismatch(physical_root: Path, tmp_path: Path) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    evidence = assemble_synthetic_evidence(controller_input=cin, result="phase_a_pass")
    evidence["mutation_observation"]["inotify_mask"] = 1
    verdict = verify_evidence_package(evidence)
    assert verdict["accepted"] is False
    assert "inotify_mask_mismatch" in verdict["rejection_codes"]


def test_disallowed_mutation(physical_root: Path, tmp_path: Path) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    mut = empty_mutation_observation()
    mut["final_classification"] = "bound_path_mutation"
    mut["events"] = [
        {
            "watch_path": "/x",
            "event_path": "/x/file",
            "mask": INOTIFY_MASK,
            "classification": "bound_path_mutation",
        }
    ]
    evidence = assemble_synthetic_evidence(
        controller_input=cin, result="phase_a_pass", mutation_observation=mut
    )
    verdict = verify_evidence_package(evidence)
    assert verdict["accepted"] is False
    assert "disallowed_mutation" in verdict["rejection_codes"]


def test_mutation_classification_ignores_output_paths() -> None:
    cls = classify_inotify_event(
        mask=INOTIFY_MASK,
        watch_path="/repo/out",
        event_path="/repo/out/log.txt",
        bound_paths=["/repo/src/a.py"],
        output_prefixes=["/repo/out"],
    )
    assert cls == "none"
    cls2 = classify_inotify_event(
        mask=INOTIFY_MASK,
        watch_path="/repo/src",
        event_path="/repo/src/a.py",
        bound_paths=["/repo/src/a.py"],
        output_prefixes=["/repo/out"],
    )
    assert cls2 == "bound_path_mutation"


def test_child_unexpected_exit_vs_phase_a_pass(
    physical_root: Path, tmp_path: Path
) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    evidence = assemble_synthetic_evidence(
        controller_input=cin,
        result="phase_a_pass",
        child_exit_class="runner_nonzero",
    )
    verdict = verify_evidence_package(evidence)
    assert verdict["accepted"] is False
    assert any(
        c in verdict["rejection_codes"]
        for c in ("child_unexpected_exit", "result_enum_inconsistency")
    )


def test_malformed_child_output(physical_root: Path, tmp_path: Path) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    evidence = assemble_synthetic_evidence(controller_input=cin, result="phase_a_pass")
    evidence["child_results"]["00_capture_off"]["exit_class"] = "malformed"
    verdict = verify_evidence_package(evidence)
    assert verdict["accepted"] is False
    assert "malformed_child_output" in verdict["rejection_codes"]


def test_controller_self_report_valid_not_trusted(
    physical_root: Path, tmp_path: Path
) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    evidence = assemble_synthetic_evidence(
        controller_input=cin,
        result="phase_a_pass",
        controller_self_report_valid=True,
    )
    # Tamper command vector while claiming valid.
    evidence["parent_command_vector"] = list(PARENT_COMMAND_VECTOR) + ["--evil"]
    verdict = verify_evidence_package(evidence)
    assert verdict["accepted"] is False
    assert "command_vector_mismatch" in verdict["rejection_codes"]


def test_cdv_matrix_inconsistency_with_result(
    physical_root: Path, tmp_path: Path
) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    evidence = assemble_synthetic_evidence(controller_input=cin, result="phase_a_pass")
    # Claim phase_a_pass but attach provenance_invalid matrix row.
    evidence["cdv_matrix_row"] = cdv_matrix_row("provenance_invalid")
    verdict = verify_evidence_package(evidence)
    assert verdict["accepted"] is False
    assert "cdv_matrix_inconsistency" in verdict["rejection_codes"]


def test_evidence_missing_published_path(physical_root: Path, tmp_path: Path) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    evidence = assemble_synthetic_evidence(controller_input=cin, result="phase_a_pass")
    evidence["published_paths"] = evidence["published_paths"][:-3]
    verdict = verify_evidence_package(evidence)
    assert verdict["accepted"] is False
    assert "evidence_incomplete" in verdict["rejection_codes"]


def test_unrecognized_state_predicate() -> None:
    with pytest.raises(ContractError) as ei:
        select_controller_result({"not_a_real_result": True})
    assert ei.value.code == "unrecognized_state"


def test_child_argv_parser_rejects_extras() -> None:
    with pytest.raises(ContractError):
        parse_child_argv([])
    with pytest.raises(ContractError):
        parse_child_argv(["--run-id"])
    with pytest.raises(ContractError):
        parse_child_argv(["--run-id", "00_capture_off", "--extra"])
    with pytest.raises(ContractError):
        parse_child_argv(["-r", "00_capture_off"])
    assert parse_child_argv(["--run-id", "02_capture_on_2"]) == "02_capture_on_2"


def test_child_refuses_without_eval_hook(physical_root: Path, tmp_path: Path) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    child_in = build_child_input(
        repository_root=cin["repository_root"],
        instrumentation_head=cin["instrumentation_head"],
        run_id="00_capture_off",
        capture_run_uuid=cin["capture_run_uuids"]["00_capture_off"],
        cuda_device_uuid=cin["cuda_device_uuid"],
        tensorrt_lib_dir=cin["tensorrt_lib_dir"],
        pytorch_lib_dir=cin["pytorch_lib_dir"],
        cuda_lib64_dir=cin["cuda_lib64_dir"],
        bound_inputs_digest=cin["bound_inputs"]["inventory_digest"],
        sequence_input_digest=cin["bound_inputs"]["sequence"]["aggregate_sha256"],
    )
    with pytest.raises(ContractError) as ei:
        execute_child_contract(child_in, actual_env=child_in["environment"])
    assert ei.value.code == "unclassified"


def test_ordered_run_plan_immutable() -> None:
    assert ordered_run_plan() == list(RUN_IDS)
    assert parent_command_vector() == list(PARENT_COMMAND_VECTOR)


def test_sequence_record_bytes_format() -> None:
    rec = sequence_file_record_bytes("img1/000001.jpg", 10, "a" * 64)
    assert rec.endswith(b"\n")
    assert b"\0" in rec


def test_four_deliverable_paths_exist() -> None:
    tools = Path(__file__).resolve().parents[3] / "scripts" / "tools"
    assert (tools / "run_h0_phase_a.py").is_file()
    assert (tools / "run_h0_phase_a_child.py").is_file()
    assert (tools / "h0_phase_a_execution_schema_v1.json").is_file()
    assert (tools / "verify_h0_phase_a.py").is_file()


def test_checkpoint_drift_fails_phase_a_pass(
    physical_root: Path, tmp_path: Path
) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    evidence = assemble_synthetic_evidence(
        controller_input=cin, result="phase_a_pass", checkpoint_equal=False
    )
    verdict = verify_evidence_package(evidence)
    assert verdict["accepted"] is False
    assert "bound_input_digest_mismatch" in verdict["rejection_codes"]


def test_unknown_field_in_evidence(physical_root: Path, tmp_path: Path) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    evidence = assemble_synthetic_evidence(controller_input=cin, result="phase_a_pass")
    evidence["free_text_verdict"] = "looks fine"
    verdict = verify_evidence_package(evidence)
    assert verdict["accepted"] is False
    assert "unknown_field" in verdict["rejection_codes"]


def test_child_env_drift_during_execute(physical_root: Path, tmp_path: Path) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    child_in = build_child_input(
        repository_root=cin["repository_root"],
        instrumentation_head=cin["instrumentation_head"],
        run_id="00_capture_off",
        capture_run_uuid=cin["capture_run_uuids"]["00_capture_off"],
        cuda_device_uuid=cin["cuda_device_uuid"],
        tensorrt_lib_dir=cin["tensorrt_lib_dir"],
        pytorch_lib_dir=cin["pytorch_lib_dir"],
        cuda_lib64_dir=cin["cuda_lib64_dir"],
        bound_inputs_digest=cin["bound_inputs"]["inventory_digest"],
        sequence_input_digest=cin["bound_inputs"]["sequence"]["aggregate_sha256"],
    )
    bad_env = dict(child_in["environment"])
    bad_env["LANG"] = "en_US.UTF-8"
    with pytest.raises(ContractError) as ei:
        execute_child_contract(child_in, actual_env=bad_env, eval_hook=lambda _p: None)
    assert ei.value.code == "env_mismatch"


def test_bound_digest_revalidate_drift(physical_root: Path, tmp_path: Path) -> None:
    cin = make_controller_input(physical_root, tmp_path)
    child_in = build_child_input(
        repository_root=cin["repository_root"],
        instrumentation_head=cin["instrumentation_head"],
        run_id="00_capture_off",
        capture_run_uuid=cin["capture_run_uuids"]["00_capture_off"],
        cuda_device_uuid=cin["cuda_device_uuid"],
        tensorrt_lib_dir=cin["tensorrt_lib_dir"],
        pytorch_lib_dir=cin["pytorch_lib_dir"],
        cuda_lib64_dir=cin["cuda_lib64_dir"],
        bound_inputs_digest=cin["bound_inputs"]["inventory_digest"],
        sequence_input_digest=cin["bound_inputs"]["sequence"]["aggregate_sha256"],
    )
    with pytest.raises(ContractError) as ei:
        execute_child_contract(
            child_in,
            actual_env=child_in["environment"],
            revalidate_bound_digest="f" * 64,
            eval_hook=lambda _p: None,
        )
    assert ei.value.code == "bound_input_drift"
