#!/usr/bin/env python3
"""Independent H0 Phase-A aggregate verifier (``h0_phase_a_verifier_v1``).

Authority: Amendment 7 A7.2/A7.8/A7.9 + RC1.4.

Does not trust controller self-report ``valid``/``controller_self_report_valid``.
Does not normalize non-canonical input. Unknown fields, partial evidence, and
warnings-as-acceptance are forbidden — every failure is a rejection code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from run_h0_phase_a import (  # noqa: E402
    BOUND_INPUT_CHECKPOINTS,
    BUILD_BUILD_VECTOR,
    BUILD_CONFIGURE_VECTOR,
    CHILD_RESULT_SCHEMA_VERSION,
    CHILD_SCHEMA_VERSION,
    CONTROLLER_SCHEMA_VERSION,
    ENV_KEY_ORDER,
    EXECUTION_SCHEMA_VERSION,
    INOTIFY_MASK,
    INOTIFY_MASK_NAMES,
    PARENT_COMMAND_VECTOR,
    RESULT_ENUM,
    RUN_IDS,
    VERIFIER_SCHEMA_VERSION,
    ContractError,
    build_controller_plan,
    cdv_matrix_row,
    cdv_paths_for_result,
    environment_digest,
    require_physical_abs_path,
    validate_bound_inputs,
    validate_sequence_digest_object,
)

REJECTION_CODES = (
    "schema_nonconformance",
    "unknown_field",
    "missing_required_field",
    "illegal_enum",
    "command_vector_mismatch",
    "command_vector_order",
    "executable_path_mismatch",
    "argument_mismatch",
    "environment_mismatch",
    "working_directory_drift",
    "bound_input_digest_mismatch",
    "sequence_input_digest_mismatch",
    "symlink_or_non_canonical_path",
    "inotify_mask_mismatch",
    "disallowed_mutation",
    "child_unexpected_exit",
    "malformed_child_output",
    "controller_self_report_untrusted",
    "cdv_matrix_inconsistency",
    "result_enum_inconsistency",
    "evidence_incomplete",
    "unrecognized_state",
    "parent_child_inconsistency",
    "a7_9_free_choice_detected",
)

EVIDENCE_REQUIRED_FIELDS = (
    "schema_version",
    "controller_schema_version",
    "child_schema_version",
    "verifier_schema_version",
    "instrumentation_head",
    "result",
    "controller_self_report_valid",
    "parent_command_vector",
    "build_configure_vector",
    "build_build_vector",
    "ordered_run_plan",
    "child_command_vectors",
    "evaluator_argv_by_run",
    "environment_tables_by_run",
    "environment_digests_by_run",
    "bound_inputs",
    "sequence_input_digest",
    "mutation_observation",
    "checkpoint_records",
    "child_results",
    "cdv_matrix_row",
    "published_paths",
    "artifact_states",
)


class VerificationFailure(Exception):
    def __init__(self, codes: Sequence[str], detail: str = "") -> None:
        self.codes = list(codes)
        self.detail = detail
        super().__init__(",".join(self.codes) + (f": {detail}" if detail else ""))


def _reject(*codes: str, detail: str = "") -> None:
    for c in codes:
        if c not in REJECTION_CODES:
            raise VerificationFailure(["unrecognized_state"], detail=f"bad code {c}")
    raise VerificationFailure(codes, detail=detail)


def _require_mapping(obj: Any, *, what: str) -> MutableMapping[str, Any]:
    if not isinstance(obj, dict):
        _reject("schema_nonconformance", detail=f"{what} not object")
    return obj


def _check_no_extra(
    obj: Mapping[str, Any], allowed: Sequence[str], *, what: str
) -> None:
    extra = set(obj) - set(allowed)
    if extra:
        _reject("unknown_field", detail=f"{what} extra={sorted(extra)}")
    missing = set(allowed) - set(obj)
    if missing:
        _reject("missing_required_field", detail=f"{what} missing={sorted(missing)}")


def verify_evidence_package(package: Mapping[str, Any]) -> dict[str, Any]:
    """Independently verify a complete synthetic or real evidence package.

    Returns ``h0_phase_a_verifier_v1`` result. Never returns acceptance on
    partial evidence or when controller self-report is the sole signal.
    """
    try:
        obj = _require_mapping(package, what="evidence")
        _check_no_extra(obj, EVIDENCE_REQUIRED_FIELDS, what="evidence")

        if obj["schema_version"] != EXECUTION_SCHEMA_VERSION:
            _reject("illegal_enum", detail="schema_version")
        if obj["controller_schema_version"] != CONTROLLER_SCHEMA_VERSION:
            _reject("illegal_enum", detail="controller_schema_version")
        if obj["child_schema_version"] != CHILD_SCHEMA_VERSION:
            _reject("illegal_enum", detail="child_schema_version")
        if obj["verifier_schema_version"] != VERIFIER_SCHEMA_VERSION:
            _reject("illegal_enum", detail="verifier_schema_version")

        result = obj["result"]
        if result not in RESULT_ENUM:
            _reject("illegal_enum", detail=f"result {result!r}")

        # Never trust controller self-report as acceptance.
        if obj["controller_self_report_valid"] is True:
            # Still must independently prove; flag alone is insufficient.
            pass
        elif obj["controller_self_report_valid"] is not False:
            _reject(
                "unrecognized_state", detail="controller_self_report_valid not bool"
            )

        head = obj["instrumentation_head"]
        if (
            not isinstance(head, str)
            or len(head) != 40
            or any(c not in "0123456789abcdef" for c in head)
        ):
            _reject("illegal_enum", detail="instrumentation_head")

        # Command vectors: exact frozen constants.
        if list(obj["parent_command_vector"]) != list(PARENT_COMMAND_VECTOR):
            _reject("command_vector_mismatch", detail="parent")
        if list(obj["build_configure_vector"]) != list(BUILD_CONFIGURE_VECTOR):
            _reject("command_vector_mismatch", detail="build configure")
        if list(obj["build_build_vector"]) != list(BUILD_BUILD_VECTOR):
            _reject("command_vector_mismatch", detail="build build")

        plan = obj["ordered_run_plan"]
        if list(plan) != list(RUN_IDS):
            _reject("command_vector_order", detail=f"ordered_run_plan {plan!r}")

        child_vecs = _require_mapping(
            obj["child_command_vectors"], what="child_command_vectors"
        )
        _check_no_extra(child_vecs, RUN_IDS, what="child_command_vectors")
        eval_vecs = _require_mapping(
            obj["evaluator_argv_by_run"], what="evaluator_argv_by_run"
        )
        _check_no_extra(eval_vecs, RUN_IDS, what="evaluator_argv_by_run")
        env_tables = _require_mapping(
            obj["environment_tables_by_run"], what="environment_tables_by_run"
        )
        _check_no_extra(env_tables, RUN_IDS, what="environment_tables_by_run")
        env_digests = _require_mapping(
            obj["environment_digests_by_run"], what="environment_digests_by_run"
        )
        _check_no_extra(env_digests, RUN_IDS, what="environment_digests_by_run")
        child_results = _require_mapping(obj["child_results"], what="child_results")
        _check_no_extra(child_results, RUN_IDS, what="child_results")

        # Recover repository_root from child vectors (absolute python path).
        sample_vec = child_vecs[RUN_IDS[0]]
        if not isinstance(sample_vec, list) or len(sample_vec) < 1:
            _reject("malformed_child_output", detail="child vector")
        python_path = sample_vec[0]
        if not isinstance(python_path, str) or not python_path.endswith(
            "/.venv/bin/python"
        ):
            _reject("executable_path_mismatch", detail=python_path)
        root = python_path[: -len("/.venv/bin/python")]
        try:
            require_physical_abs_path(root, field="repository_root")
        except ContractError as exc:
            _reject("symlink_or_non_canonical_path", detail=str(exc))

        # Rebuild expected plan independently from recovered parameters.
        # Library dirs from LD_LIBRARY_PATH: build, trt, torch, cuda (order fixed).
        sample_env = env_tables[RUN_IDS[0]]
        if not isinstance(sample_env, Mapping):
            _reject("environment_mismatch", detail="env not mapping")
        try:
            ld_parts = sample_env["LD_LIBRARY_PATH"].split(":")
        except Exception as exc:  # noqa: BLE001
            _reject("environment_mismatch", detail=f"LD_LIBRARY_PATH: {exc}")
        if len(ld_parts) != 4 or any(p == "" for p in ld_parts):
            _reject("environment_mismatch", detail="LD_LIBRARY_PATH arity")
        if ld_parts[0] != f"{root}/build/h0_phase_a":
            _reject("environment_mismatch", detail="build dir in LD_LIBRARY_PATH")
        trt, pth, cuda = ld_parts[1], ld_parts[2], ld_parts[3]
        cuda_uuid = sample_env.get("CUDA_VISIBLE_DEVICES", "")
        try:
            expected_plan = build_controller_plan(
                repository_root=root,
                instrumentation_head=head,
                cuda_device_uuid=cuda_uuid,
                tensorrt_lib_dir=trt,
                pytorch_lib_dir=pth,
                cuda_lib64_dir=cuda,
            )
        except ContractError as exc:
            _reject("a7_9_free_choice_detected", detail=str(exc))

        for run_id in RUN_IDS:
            exp_child = expected_plan["child_command_vectors"][run_id]
            got_child = list(child_vecs[run_id])
            if got_child != exp_child:
                if got_child and got_child[0] != exp_child[0]:
                    _reject("executable_path_mismatch", detail=run_id)
                if len(got_child) != len(exp_child):
                    _reject("argument_mismatch", detail=run_id)
                # order change
                if sorted(got_child) == sorted(exp_child) and got_child != exp_child:
                    _reject("command_vector_order", detail=run_id)
                _reject("command_vector_mismatch", detail=run_id)

            exp_eval = expected_plan["evaluator_argv_by_run"][run_id]
            if list(eval_vecs[run_id]) != exp_eval:
                _reject("argument_mismatch", detail=f"evaluator {run_id}")

            exp_env = expected_plan["environment_tables_by_run"][run_id]
            got_env = env_tables[run_id]
            if set(got_env) != set(ENV_KEY_ORDER):
                _reject("environment_mismatch", detail=f"keys {run_id}")
            for k in ENV_KEY_ORDER:
                if got_env.get(k) != exp_env[k]:
                    _reject("environment_mismatch", detail=f"{run_id}.{k}")
            try:
                dig = environment_digest(got_env)
            except ContractError as exc:
                _reject("environment_mismatch", detail=str(exc))
            if dig != env_digests[run_id]:
                _reject("environment_mismatch", detail=f"digest {run_id}")

            # Child result consistency.
            cr = child_results[run_id]
            if not isinstance(cr, Mapping):
                _reject("malformed_child_output", detail=run_id)
            _verify_child_result(
                cr,
                run_id=run_id,
                expected_command=exp_child,
                expected_eval=exp_eval,
                expected_env_digest=dig,
                expected_bound_digest=obj["bound_inputs"]["inventory_digest"],
                expected_seq_digest=obj["sequence_input_digest"]["aggregate_sha256"],
                result=result,
            )

        # Bound inputs + sequence digest independent recompute.
        try:
            validate_bound_inputs(obj["bound_inputs"], expected_head=head)
            validate_sequence_digest_object(obj["sequence_input_digest"])
        except ContractError as exc:
            code = {
                "bound_input_digest_mismatch": "bound_input_digest_mismatch",
                "sequence_input_digest_mismatch": "sequence_input_digest_mismatch",
                "unknown_field": "unknown_field",
                "missing_required_field": "missing_required_field",
                "illegal_enum": "illegal_enum",
                "schema_nonconformance": "schema_nonconformance",
                "symlink_or_non_canonical_path": "symlink_or_non_canonical_path",
            }.get(exc.code, "bound_input_digest_mismatch")
            _reject(code, detail=exc.message)

        if (
            obj["sequence_input_digest"]["aggregate_sha256"]
            != obj["bound_inputs"]["sequence"]["aggregate_sha256"]
        ):
            _reject(
                "sequence_input_digest_mismatch",
                detail="top-level vs bound_inputs.sequence",
            )

        # Mutation / inotify.
        mut = _require_mapping(obj["mutation_observation"], what="mutation_observation")
        _verify_mutation(mut, result=result)

        # Checkpoints.
        cps = obj["checkpoint_records"]
        if not isinstance(cps, list) or not cps:
            _reject("evidence_incomplete", detail="checkpoint_records")
        seen_cp = [c.get("checkpoint_id") for c in cps if isinstance(c, Mapping)]
        # At least T0 present; for phase_a_pass require all equal_to_t0.
        if "T0" not in seen_cp:
            _reject("evidence_incomplete", detail="T0 missing")
        for c in cps:
            if not isinstance(c, Mapping):
                _reject("schema_nonconformance", detail="checkpoint")
            if c.get("checkpoint_id") not in BOUND_INPUT_CHECKPOINTS:
                _reject("illegal_enum", detail=f"checkpoint {c.get('checkpoint_id')!r}")
            if result == "phase_a_pass" and c.get("equal_to_t0") is not True:
                _reject(
                    "bound_input_digest_mismatch",
                    detail=f"checkpoint {c.get('checkpoint_id')}",
                )

        # C/D/V matrix.
        row = _require_mapping(obj["cdv_matrix_row"], what="cdv_matrix_row")
        expected_row = cdv_matrix_row(result)
        if row.get("result") != result:
            _reject("cdv_matrix_inconsistency", detail="row.result != result")
        if list(row.get("required_sets", [])) != expected_row["required_sets"]:
            _reject("cdv_matrix_inconsistency", detail="required_sets")
        if list(row.get("forbidden_sets", [])) != expected_row["forbidden_sets"]:
            _reject("cdv_matrix_inconsistency", detail="forbidden_sets")
        if list(row.get("required_paths", [])) != expected_row["required_paths"]:
            _reject("cdv_matrix_inconsistency", detail="required_paths")
        if list(row.get("forbidden_paths", [])) != expected_row["forbidden_paths"]:
            _reject("cdv_matrix_inconsistency", detail="forbidden_paths")

        published = obj["published_paths"]
        if not isinstance(published, list):
            _reject("schema_nonconformance", detail="published_paths")
        required_paths, forbidden_paths = cdv_paths_for_result(result)
        if sorted(published) != sorted(required_paths):
            _reject("evidence_incomplete", detail="published_paths != required")
        states = _require_mapping(obj["artifact_states"], what="artifact_states")
        for p in required_paths:
            if states.get(p) != "produced":
                _reject("evidence_incomplete", detail=f"required not produced: {p}")
        for p in forbidden_paths:
            if states.get(p) == "produced":
                _reject("cdv_matrix_inconsistency", detail=f"forbidden produced: {p}")

        # Parent/child consistency on digests for completed children.
        for run_id in RUN_IDS:
            cr = child_results[run_id]
            if cr.get("exit_class") == "completed":
                if (
                    cr.get("bound_inputs_digest")
                    != obj["bound_inputs"]["inventory_digest"]
                ):
                    _reject("parent_child_inconsistency", detail=f"bound {run_id}")
                if (
                    cr.get("sequence_input_digest")
                    != obj["sequence_input_digest"]["aggregate_sha256"]
                ):
                    _reject("parent_child_inconsistency", detail=f"seq {run_id}")

        # Result vs child exits for terminal consistency (mechanical subset).
        if result == "phase_a_pass":
            for run_id in RUN_IDS:
                if child_results[run_id].get("exit_class") != "completed":
                    _reject(
                        "result_enum_inconsistency", detail=f"{run_id} not completed"
                    )
        if result == "runner_nonzero":
            if not any(
                child_results[r].get("exit_class") == "runner_nonzero" for r in RUN_IDS
            ) and not any(
                child_results[r].get("exit_class") not in {"completed", "not_run"}
                for r in RUN_IDS
            ):
                # Allow synthetic packages that declare runner_nonzero with
                # non-completed children.
                if all(
                    child_results[r].get("exit_class") == "completed" for r in RUN_IDS
                ):
                    _reject(
                        "result_enum_inconsistency",
                        detail="runner_nonzero but all children completed",
                    )

        # If controller claims valid but we would reject any structural issue —
        # already covered. If controller claims valid=true with phase_a_pass but
        # mutation present: reject.
        if (
            obj["controller_self_report_valid"] is True
            and result == "phase_a_pass"
            and mut.get("final_classification") not in (None, "none")
        ):
            _reject("controller_self_report_untrusted", detail="mutation vs valid")

        # A7.9: re-derived plan must match all frozen fields with zero free choice.
        if expected_plan["deadline_seconds"] != 3600:
            _reject("a7_9_free_choice_detected", detail="deadline")
        if expected_plan["inotify_mask"] != INOTIFY_MASK:
            _reject("a7_9_free_choice_detected", detail="inotify mask")

        return {
            "schema_version": VERIFIER_SCHEMA_VERSION,
            "accepted": True,
            "result": result,
            "rejection_codes": [],
        }
    except VerificationFailure as exc:
        return {
            "schema_version": VERIFIER_SCHEMA_VERSION,
            "accepted": False,
            "result": None,
            "rejection_codes": list(exc.codes),
        }


def _verify_child_result(
    cr: Mapping[str, Any],
    *,
    run_id: str,
    expected_command: Sequence[str],
    expected_eval: Sequence[str],
    expected_env_digest: str,
    expected_bound_digest: str,
    expected_seq_digest: str,
    result: str,
) -> None:
    required = {
        "schema_version",
        "run_id",
        "exit_class",
        "command_vector",
        "evaluator_argv",
        "environment_digest",
        "bound_inputs_digest",
        "sequence_input_digest",
        "capture_run_uuid",
        "stdout_sha256",
        "stderr_sha256",
    }
    # failure_reason_code is optional; only known fields are admitted.
    allowed = set(required) | {"failure_reason_code"}
    if set(cr) - allowed:
        _reject(
            "unknown_field",
            detail=f"child_result {run_id} extra={sorted(set(cr) - allowed)}",
        )
    for field in (
        "schema_version",
        "run_id",
        "exit_class",
        "command_vector",
        "evaluator_argv",
        "environment_digest",
        "bound_inputs_digest",
        "sequence_input_digest",
        "capture_run_uuid",
        "stdout_sha256",
        "stderr_sha256",
    ):
        if field not in cr:
            _reject("missing_required_field", detail=f"child_result.{field}")

    if cr["schema_version"] != CHILD_RESULT_SCHEMA_VERSION:
        _reject("illegal_enum", detail="child_result schema_version")
    if cr["run_id"] != run_id:
        _reject("parent_child_inconsistency", detail="run_id")
    if list(cr["command_vector"]) != list(expected_command):
        _reject("command_vector_mismatch", detail=f"child_result {run_id}")
    if list(cr["evaluator_argv"]) != list(expected_eval):
        _reject("argument_mismatch", detail=f"child_result eval {run_id}")
    if cr["environment_digest"] != expected_env_digest:
        # completed children must match; failed children may carry invalid digests
        if cr["exit_class"] == "completed":
            _reject("environment_mismatch", detail=f"child digest {run_id}")
    exit_class = cr["exit_class"]
    allowed_exits = {
        "completed",
        "runner_nonzero",
        "runner_timeout",
        "provenance_invalid",
        "unclassified_execution_failure",
        "malformed",
    }
    if exit_class not in allowed_exits:
        _reject("illegal_enum", detail=f"exit_class {exit_class!r}")
    if exit_class == "malformed":
        _reject("malformed_child_output", detail=run_id)
    if result == "phase_a_pass" and exit_class != "completed":
        _reject("child_unexpected_exit", detail=run_id)


def _verify_mutation(mut: Mapping[str, Any], *, result: str) -> None:
    required = {
        "schema_version",
        "inotify_mask",
        "inotify_mask_names",
        "monitor_status",
        "events",
        "final_classification",
    }
    _check_no_extra(mut, list(required), what="mutation_observation")
    if mut["inotify_mask"] != INOTIFY_MASK:
        _reject("inotify_mask_mismatch", detail=str(mut["inotify_mask"]))
    if list(mut["inotify_mask_names"]) != list(INOTIFY_MASK_NAMES):
        _reject("inotify_mask_mismatch", detail="names")
    if mut["monitor_status"] not in {"active", "stopped", "failed"}:
        _reject("illegal_enum", detail="monitor_status")
    cls = mut["final_classification"]
    allowed_cls = {
        "none",
        "bound_path_mutation",
        "ancestor_move_or_delete",
        "watch_install_failure",
        "queue_overflow",
        "ignored_watch",
        "unclassified_monitor_failure",
    }
    if cls not in allowed_cls:
        _reject("illegal_enum", detail=f"mutation class {cls!r}")
    if result == "phase_a_pass" and cls != "none":
        _reject("disallowed_mutation", detail=cls)
    if result == "provenance_invalid" and cls == "none":
        # provenance can be non-mutation; OK
        pass
    if mut["monitor_status"] == "failed" and result == "phase_a_pass":
        _reject("disallowed_mutation", detail="monitor failed")


def verify_path(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "schema_version": VERIFIER_SCHEMA_VERSION,
            "accepted": False,
            "result": None,
            "rejection_codes": ["malformed_child_output"],
            "detail": str(exc),
        }
    return verify_evidence_package(data)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent H0 Phase-A verifier")
    parser.add_argument(
        "evidence",
        type=Path,
        help="Path to evidence package JSON (synthetic or published aggregate)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = verify_path(args.evidence)
    sys.stdout.write(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    )
    return 0 if result.get("accepted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
