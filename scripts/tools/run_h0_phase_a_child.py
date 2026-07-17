#!/usr/bin/env python3
"""H0 Phase-A child executor (``h0_phase_a_child_v1``).

Authority: Amendment 7 RC1 (A7.RC1.1–A7.RC1.3).

Not an operator entry point. The parent controller launches this process with
exactly::

    <ROOT>/.venv/bin/python -I -B <ROOT>/scripts/tools/run_h0_phase_a_child.py --run-id <id>

The child accepts only the two-token suffix ``--run-id <enumerated-id>``.
No interactive mode, manual override, fallback, or best-effort path exists.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from run_h0_phase_a import (  # noqa: E402
    BUILD_DIR_REL,
    CHILD_SCHEMA_VERSION,
    ENV_KEY_ORDER,
    RUN_IDS,
    ContractError,
    build_child_result,
    child_command_vector,
    child_context_path,
    environment_digest,
    evaluator_argv,
    require_physical_abs_path,
    run_dir_rel,
    sha256_bytes,
    validate_child_input,
)

# Schema/version string frozen by RC1.1.
assert CHILD_SCHEMA_VERSION == "h0_phase_a_child_v1"


def parse_child_argv(argv: Sequence[str]) -> str:
    """Accept exactly ``--run-id <one-enumerated-id>``; reject everything else."""
    args = list(argv)
    if len(args) != 2:
        raise ContractError(
            "argv_reject",
            f"child accepts exactly --run-id <id>, got {args!r}",
        )
    if args[0] != "--run-id":
        raise ContractError(
            "argv_reject", f"first token must be --run-id, got {args[0]!r}"
        )
    run_id = args[1]
    if run_id not in RUN_IDS:
        raise ContractError("argv_reject", f"run_id not enumerated: {run_id!r}")
    # Reject abbreviations / duplicates already covered by exact length-2 form.
    return run_id


def verify_environment_exact(
    expected: Mapping[str, str], actual: Mapping[str, str]
) -> None:
    """RC1.2: key set and values must equal the table exactly; no pass-through."""
    exp_keys = set(expected)
    act_keys = set(actual)
    if act_keys != exp_keys:
        missing = sorted(exp_keys - act_keys)
        extra = sorted(act_keys - exp_keys)
        raise ContractError(
            "env_mismatch",
            f"environment key set mismatch missing={missing} extra={extra}",
        )
    for key in ENV_KEY_ORDER:
        if actual[key] != expected[key]:
            raise ContractError(
                "env_mismatch",
                f"environment value drift for {key}: "
                f"expected={expected[key]!r} actual={actual[key]!r}",
            )


def verify_command_vector(
    repository_root: str, run_id: str, observed: Sequence[str] | None = None
) -> list[str]:
    """Rebuild the sole legal child argv and optionally compare."""
    expected = child_command_vector(repository_root, run_id)
    if observed is not None and list(observed) != expected:
        raise ContractError(
            "argv_reject",
            f"command vector mismatch expected={expected!r} observed={list(observed)!r}",
        )
    return expected


def verify_bound_and_sequence_digests(
    child_input: Mapping[str, Any],
    *,
    bound_inputs_digest: str | None = None,
    sequence_input_digest: str | None = None,
) -> None:
    if (
        bound_inputs_digest is not None
        and bound_inputs_digest != child_input["bound_inputs_digest"]
    ):
        raise ContractError(
            "bound_input_drift",
            "bound_inputs_digest drifted after parent validation",
        )
    if (
        sequence_input_digest is not None
        and sequence_input_digest != child_input["sequence_input_digest"]
    ):
        raise ContractError(
            "sequence_digest_mismatch",
            "sequence_input_digest drifted after parent validation",
        )


def load_child_input_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError("bound_input_drift", f"child_input.json missing: {path}")
    if path.is_symlink():
        raise ContractError(
            "symlink_or_traversal", f"child_input.json is symlink: {path}"
        )
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(
            "malformed_output", f"child_input unreadable: {exc}"
        ) from exc
    if not isinstance(obj, dict):
        raise ContractError("malformed_output", "child_input not an object")
    return validate_child_input(obj)


def assert_cwd_is_repository_root(repository_root: str) -> None:
    root = require_physical_abs_path(repository_root, field="repository_root")
    cwd = os.path.realpath(os.getcwd())
    if cwd != root:
        raise ContractError(
            "env_mismatch",
            f"working directory drift cwd={cwd!r} root={root!r}",
        )


def build_trace_lifecycle(run_id: str) -> dict[str, Any]:
    """Frozen H0 trace arm/clear/drain plan (RC1.1). Not executed here without authority."""
    caps = (65536, 16384, 16384, 16384)
    if run_id == "00_capture_off":
        return {
            "set_research_h0_bridge_trace": [False, *caps],
            "clear_research_h0_bridge_trace": False,
            "drain_research_h0_bridge_trace": False,
        }
    return {
        "set_research_h0_bridge_trace": [True, *caps],
        "clear_research_h0_bridge_trace": True,
        "drain_research_h0_bridge_trace": {
            "seq": "MOT17-04-SDP",
            "capture_phase": "phase_a",
            "require_candidate_exposure": True,
            "require_commit_exposure": False,
        },
    }


def execute_child_contract(
    child_input: Mapping[str, Any],
    *,
    actual_env: Mapping[str, str] | None = None,
    revalidate_bound_digest: str | None = None,
    revalidate_sequence_digest: str | None = None,
    eval_hook: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Validate child contract fail-closed and return structured child_result.

    ``eval_hook`` is test-only injection. Production path leaves it ``None`` and
    refuses Phase-A evaluation without sealed authority (no best-effort run).
    """
    cin = validate_child_input(child_input)
    run_id = cin["run_id"]
    root = cin["repository_root"]

    verify_command_vector(root, run_id, cin["command_vector"])
    expected_eval = evaluator_argv(run_dir_rel(cin["instrumentation_head"], run_id))
    if list(cin["evaluator_argv"]) != expected_eval:
        raise ContractError("argv_reject", "evaluator_argv non-canonical")

    env_actual = dict(os.environ) if actual_env is None else dict(actual_env)
    # Child must see only the RC1.2 key set. Compare against expected table.
    verify_environment_exact(
        cin["environment"],
        {k: env_actual[k] for k in ENV_KEY_ORDER if k in env_actual}
        if set(env_actual) >= set(ENV_KEY_ORDER)
        else env_actual,
    )
    # Stricter: exact key set equality against expected (no extras).
    # When actual_env is a full os.environ, filter is wrong — require exact.
    if actual_env is None:
        # Production: environment must already be the sanitized map only.
        filtered = {k: env_actual[k] for k in ENV_KEY_ORDER if k in env_actual}
        extras = set(env_actual) - set(ENV_KEY_ORDER)
        # Allow only the minimal implementation keys that Python itself may keep
        # if launch failed to isolate — still fail closed: any extra is reject.
        if extras or set(filtered) != set(ENV_KEY_ORDER):
            raise ContractError(
                "env_mismatch",
                f"process environment not exactly RC1.2 table; extras={sorted(extras)}",
            )
        verify_environment_exact(cin["environment"], filtered)
        env_for_digest = filtered
    else:
        verify_environment_exact(cin["environment"], actual_env)
        env_for_digest = dict(actual_env)

    verify_bound_and_sequence_digests(
        cin,
        bound_inputs_digest=revalidate_bound_digest,
        sequence_input_digest=revalidate_sequence_digest,
    )

    # Path / build-dir sanity.
    if cin["environment"]["SACCADE_BUILD_PATH"] != f"{root}/{BUILD_DIR_REL}":
        raise ContractError("env_mismatch", "SACCADE_BUILD_PATH")

    lifecycle = build_trace_lifecycle(run_id)
    if eval_hook is not None:
        eval_hook({**cin, "trace_lifecycle": lifecycle})
        exit_class = "completed"
        failure = "none"
        stdout = b""
        stderr = b""
    else:
        # No sealed authority / no hook: refuse real Phase A (substrate boundary).
        raise ContractError(
            "unclassified",
            "child refuses Phase-A evaluation without parent-sealed hermetic hook "
            "or future v3 execution authority; no best-effort path",
        )

    return build_child_result(
        run_id=run_id,
        exit_class=exit_class,
        command_vector=cin["command_vector"],
        evaluator_argv_vec=cin["evaluator_argv"],
        environment_digest_hex=environment_digest(env_for_digest),
        bound_inputs_digest=cin["bound_inputs_digest"],
        sequence_input_digest=cin["sequence_input_digest"],
        capture_run_uuid=cin["capture_run_uuid"],
        stdout_sha256=sha256_bytes(stdout),
        stderr_sha256=sha256_bytes(stderr),
        failure_reason_code=failure,
    )


_FAILURE_REASON_CODES = frozenset(
    {
        "none",
        "env_mismatch",
        "argv_reject",
        "bound_input_drift",
        "sequence_digest_mismatch",
        "path_not_canonical",
        "symlink_or_traversal",
        "unexpected_file_access",
        "unexpected_exit",
        "malformed_output",
        "unclassified",
    }
)

_EXIT_CLASS_FOR_CODE = {
    "env_mismatch": "provenance_invalid",
    "argv_reject": "provenance_invalid",
    "bound_input_drift": "provenance_invalid",
    "sequence_digest_mismatch": "provenance_invalid",
    "symlink_or_traversal": "provenance_invalid",
    "path_not_canonical": "provenance_invalid",
    "unexpected_file_access": "runner_nonzero",
    "unexpected_exit": "runner_nonzero",
    "malformed_output": "malformed",
    "unclassified": "unclassified_execution_failure",
}


def execute_child_contract_safe(
    child_input: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Like ``execute_child_contract`` but maps ContractError to child_result exit_class."""
    try:
        return execute_child_contract(child_input, **kwargs)
    except ContractError as exc:
        cin = child_input if isinstance(child_input, Mapping) else {}
        run_id = cin.get("run_id", "00_capture_off")
        if run_id not in RUN_IDS:
            run_id = "00_capture_off"
        code = exc.code
        exit_class = _EXIT_CLASS_FOR_CODE.get(code, "unclassified_execution_failure")
        empty = sha256_bytes(b"")
        bound = cin.get("bound_inputs_digest")
        seq = cin.get("sequence_input_digest")
        return build_child_result(
            run_id=run_id,
            exit_class=exit_class,
            command_vector=list(cin.get("command_vector") or []),
            evaluator_argv_vec=list(cin.get("evaluator_argv") or []),
            environment_digest_hex=sha256_bytes(b"invalid"),
            bound_inputs_digest=bound
            if isinstance(bound, str) and len(bound) == 64
            else empty,
            sequence_input_digest=seq
            if isinstance(seq, str) and len(seq) == 64
            else empty,
            capture_run_uuid=str(cin.get("capture_run_uuid") or "invalid"),
            stdout_sha256=empty,
            stderr_sha256=empty,
            failure_reason_code=code
            if code in _FAILURE_REASON_CODES
            else "unclassified",
        )


def child_main(argv: Sequence[str] | None = None) -> int:
    """Process entry. Fail-closed; writes child_result JSON to stdout on contract completion."""
    try:
        args = list(sys.argv[1:] if argv is None else argv)
        run_id = parse_child_argv(args)
        # Derive repository root from this file's location (physical path).
        root = str(TOOLS_DIR.parents[1])
        root = require_physical_abs_path(
            os.path.realpath(root), field="repository_root"
        )
        # instrumentation_head and context come from parent-written child_input.json.
        # Without parent context the child cannot proceed (fail closed).
        # Locate incomplete evidence by scanning for the sole run context is forbidden
        # free choice; parent must write context at the frozen relative path.
        # Child discovers head only from child_input next to its run dir — but needs
        # head to find the path. Parent therefore also writes a single session pointer
        # is not in RC1. RC1: child knows RUN via incomplete root from HEAD.
        # Derive head from cwd/git is operator concern; child reads fixed relative
        # discovery file written by parent:
        #   <ROOT>/.h0_phase_a_active_session.json  — NOT in RC1.
        #
        # Spec: incomplete root = h0_phase_a_<H>.incomplete where H = git HEAD.
        # Using git is allowed as identity verification (A7.3), not free choice.
        head = _read_git_head(root)
        ctx_path = child_context_path(root, head, run_id)
        cin = load_child_input_file(ctx_path)
        if cin["run_id"] != run_id:
            raise ContractError("argv_reject", "run_id mismatch vs child_input")
        if cin["instrumentation_head"] != head:
            raise ContractError(
                "bound_input_drift", "HEAD != child_input instrumentation_head"
            )
        assert_cwd_is_repository_root(cin["repository_root"])
        # Production path refuses real Phase A until sealed authority exists.
        result = execute_child_contract(cin, actual_env=None)
        sys.stdout.buffer.write(
            (
                json.dumps(
                    result, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                )
                + "\n"
            ).encode("utf-8")
        )
        return 0
    except ContractError as exc:
        sys.stderr.write(f"ContractError: {exc.code}: {exc.message}\n")
        return 2
    except Exception as exc:  # noqa: BLE001 — fail closed on any unclassified error
        sys.stderr.write(f"ContractError: unclassified: {exc}\n")
        return 2


def _read_git_head(root: str) -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(
            "bound_input_drift", f"git rev-parse HEAD failed: {exc}"
        ) from exc
    if len(out) != 40 or any(c not in "0123456789abcdef" for c in out):
        # Accept also full hex if git returns lowercase; normalize.
        out_l = out.lower()
        if len(out_l) != 40 or any(c not in "0123456789abcdef" for c in out_l):
            raise ContractError("bound_input_drift", f"bad HEAD {out!r}")
        out = out_l
    return out


def main() -> None:
    raise SystemExit(child_main())


if __name__ == "__main__":
    main()
