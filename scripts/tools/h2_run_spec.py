#!/usr/bin/env python3
"""Issue, validate, and project the sole-authority H2 Phase-A RunSpec.

The authoring input is an owner-adjudicated, byte-frozen, complete 454-key
profile.  This resolver validates that profile and its separate owner decision,
then copies the namespace into a RunSpec.  It never resolves a normal preset or
fills a profile from live parser defaults.  Runtime parser defaults, argv, and
the four repository-owned environment values are projections of the RunSpec.

Object digests use canonical JSON bytes without a trailing newline.  Serialized
artifact files add exactly one trailing newline; their file digest is therefore
intentionally different from an object digest.
"""
# status: stable

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL = REPO_ROOT / "scripts" / "eval"
if _EVAL.as_posix() not in sys.path:
    sys.path.insert(0, _EVAL.as_posix())

from h2_runtime_inputs import canonical_json_bytes, digest  # noqa: E402

RUN_SPEC_SCHEMA = "h2_phase_a_run_spec_v1"
NAMESPACE_SCHEMA = "mot17_args_resolved_namespace_v1"
PROJECTION_SCHEMA = "h2_execution_semantics_projection_v1"
OBJECT_CANONICALIZATION = "utf8_lexicographic_keys_compact_finite_no_trailing_lf_v1"
ARTIFACT_SERIALIZATION = "utf8_lexicographic_keys_compact_finite_single_trailing_lf_v1"

AUTHORING_PROFILE_REL = "docs/research/contracts/h2_phase_a_authoring_profile_v1.json"
AUTHORING_PROFILE_SCHEMA_REL = (
    "docs/research/contracts/h2_phase_a_authoring_profile_v1.schema.json"
)
AUTHORING_DECISION_REL = (
    "docs/research/contracts/h2_phase_a_run_spec_authoring_decision_v1.json"
)
AUTHORING_PROFILE_SCHEMA = "h2_phase_a_authoring_profile_v1"
AUTHORING_DECISION_SCHEMA = "h2_phase_a_run_spec_authoring_decision_v1"
AUTHORING_BINDING_SCHEMA = "h2_run_spec_authoring_binding_v1"
PROFILE_SERIALIZATION = "utf8_lexicographic_keys_indent_2_finite_trailing_lf_v1"
RUN_DIR_OUTPUT_TOKEN = "${H2_RUN_DIR}/_runtime"

# The schema declares this exact content set.  It is repeated here only as the
# resolver implementation and is contract-tested against the schema; it never
# consults h2_path_partition.  Both this resolver and the schema are members, so
# changing either changes the projection digest.
EXECUTION_SEMANTICS_PATHS: tuple[str, ...] = (
    AUTHORING_PROFILE_REL,
    AUTHORING_PROFILE_SCHEMA_REL,
    AUTHORING_DECISION_REL,
    "docs/research/contracts/h2_phase_a_run_spec_v1.json",
    "scripts/eval/mot17_args.py",
    "scripts/tools/check_h2_measure_archives.py",
    "scripts/tools/h0_bridge_decision_trace_schema_v2.json",
    "scripts/tools/h2_measurement_evidence.py",
    "scripts/tools/h2_run_spec.py",
    "scripts/tools/run_h2_measurement.py",
    "scripts/tools/run_h2_measurement_child.py",
    "scripts/tools/verify_h0_phase_a.py",
    "scripts/tools/verify_h2_measurement.py",
    "scripts/tools/verify_headline_bridge_decision_trace.py",
)

CONFIG_ENV_KEYS = frozenset(
    {
        "SACCADE_DETECT_BARRIER",
        "SACCADE_DOUBLE_BUFFER",
        "SACCADE_GPU_DECODE",
        "SACCADE_MAIN_NMS_GRAPHED",
    }
)
REPOSITORY_OWNED_ENV_KEYS = frozenset({*CONFIG_ENV_KEYS, "SACCADE_STREAM_MODE"})


class RunSpecError(RuntimeError):
    """The H2 RunSpec or one of its runtime projections is invalid."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"non-finite JSON token: {token}")


def _load_pretty_document(
    relative: str, *, require_canonical: bool = True
) -> tuple[dict[str, Any], bytes]:
    path = REPO_ROOT / relative
    if path.is_symlink() or not path.is_file():
        raise RunSpecError(f"authoring input is not a physical file: {relative}")
    try:
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise RunSpecError(f"authoring input is unreadable JSON: {relative}") from exc
    if not isinstance(payload, dict):
        raise RunSpecError(f"authoring input is not an object: {relative}")
    expected = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if require_canonical and raw != expected:
        raise RunSpecError(f"authoring input is not canonical pretty JSON: {relative}")
    return payload, raw


def load_authoring_profile() -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the frozen profile and its append-only owner decision."""
    import jsonschema

    profile, profile_raw = _load_pretty_document(AUTHORING_PROFILE_REL)
    schema, schema_raw = _load_pretty_document(
        AUTHORING_PROFILE_SCHEMA_REL, require_canonical=False
    )
    decision, decision_raw = _load_pretty_document(AUTHORING_DECISION_REL)
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(profile)
    except (
        jsonschema.exceptions.SchemaError,
        jsonschema.exceptions.ValidationError,
    ) as exc:
        raise RunSpecError(f"frozen authoring profile schema failure: {exc}") from exc
    if decision.get("schema") != AUTHORING_DECISION_SCHEMA:
        raise RunSpecError("authoring decision schema mismatch")
    _require_exact_members(
        decision,
        {
            "authority",
            "decision_date",
            "explicit_adjudications",
            "key_count",
            "profile",
            "profile_sha256",
            "runtime_interpretation",
            "schema",
        },
        label="authoring decision",
    )
    if (
        decision.get("authority") != "research_owner"
        or decision.get("profile") != AUTHORING_PROFILE_REL
        or decision.get("profile_sha256") != _sha256_bytes(profile_raw)
        or decision.get("key_count") != profile.get("key_count")
    ):
        raise RunSpecError("authoring decision does not bind the frozen profile")
    interpretation = decision.get("runtime_interpretation")
    if interpretation != {
        "preset_loader": False,
        "profile_kind": "frozen_authoring_profile_not_runtime_preset",
    }:
        raise RunSpecError("authoring decision runtime interpretation mismatch")
    resolved = profile.get("resolved_namespace")
    adjudications = decision.get("explicit_adjudications")
    if not isinstance(resolved, dict) or not isinstance(adjudications, dict):
        raise RunSpecError("authoring profile namespace or adjudication is absent")
    if set(adjudications) != {"detector", "max_frames", "preset", "warmup_frames"}:
        raise RunSpecError("authoring decision adjudication inventory mismatch")
    if any(resolved.get(key) != value for key, value in adjudications.items()):
        raise RunSpecError("authoring profile differs from owner adjudications")
    if (
        profile.get("schema") != AUTHORING_PROFILE_SCHEMA
        or profile.get("serialization") != PROFILE_SERIALIZATION
        or profile.get("namespace_schema") != NAMESPACE_SCHEMA
        or profile.get("key_count") != len(resolved)
        or profile.get("resolved_namespace_digest") != digest(resolved)
    ):
        raise RunSpecError("authoring profile identity or namespace digest mismatch")
    try:
        canonical_json_bytes(resolved)
    except (TypeError, ValueError) as exc:
        raise RunSpecError("authoring namespace is not finite canonical JSON") from exc
    binding = {
        "authoring_lineage": profile["authoring_lineage"],
        "owner_decision": AUTHORING_DECISION_REL,
        "owner_decision_sha256": _sha256_bytes(decision_raw),
        "profile": AUTHORING_PROFILE_REL,
        "profile_schema": AUTHORING_PROFILE_SCHEMA_REL,
        "profile_schema_sha256": _sha256_bytes(schema_raw),
        "profile_sha256": _sha256_bytes(profile_raw),
        "schema": AUTHORING_BINDING_SCHEMA,
    }
    return profile, binding


def execution_semantics_projection() -> dict[str, Any]:
    """Digest the exact schema-declared execution-semantics content set."""
    members: list[dict[str, Any]] = []
    for relative in EXECUTION_SEMANTICS_PATHS:
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise RunSpecError(
                f"execution-semantics member is not a physical file: {relative}"
            )
        payload = path.read_bytes()
        if not payload:
            raise RunSpecError(f"execution-semantics member is empty: {relative}")
        members.append(
            {
                "length": len(payload),
                "path": relative,
                "sha256": _sha256_bytes(payload),
            }
        )
    members.sort(key=lambda member: str(member["path"]))
    projection_digest = digest(members)
    return {
        "algorithm": "sha256_canonical_json_content_members_v1",
        "digest": projection_digest,
        "members": members,
        "schema": PROJECTION_SCHEMA,
    }


def _run_spec_digest_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in document.items()
        if key != "resolved_run_spec_digest"
    }


def build_run_spec() -> dict[str, Any]:
    """Issue one RunSpec from the validated frozen authoring profile."""
    profile, authoring_binding = load_authoring_profile()
    resolved = dict(profile["resolved_namespace"])
    projection = execution_semantics_projection()
    document: dict[str, Any] = {
        "artifact_serialization": ARTIFACT_SERIALIZATION,
        "authoring_profile": authoring_binding,
        "execution_semantics_projection": projection,
        "execution_semantics_projection_digest": projection["digest"],
        "namespace_schema": NAMESPACE_SCHEMA,
        "object_canonicalization": OBJECT_CANONICALIZATION,
        "phase": "phase_a",
        "resolved_namespace": resolved,
        "resolved_namespace_digest": digest(resolved),
        "resolved_namespace_keys": sorted(resolved),
        "schema": RUN_SPEC_SCHEMA,
    }
    document["resolved_run_spec_digest"] = digest(_run_spec_digest_payload(document))
    validate_run_spec(document, verify_projection=True)
    return document


def _require_exact_members(
    value: Mapping[str, Any], expected: set[str], *, label: str
) -> None:
    if set(value) != expected:
        raise RunSpecError(
            f"{label} has missing or unknown members: {sorted(set(value) ^ expected)}"
        )


def validate_run_spec(
    document: Mapping[str, Any], *, verify_projection: bool = True
) -> None:
    """Fail closed on schema, digest, completeness, or content-set drift."""
    expected_members = {
        "artifact_serialization",
        "authoring_profile",
        "execution_semantics_projection",
        "execution_semantics_projection_digest",
        "namespace_schema",
        "object_canonicalization",
        "phase",
        "resolved_namespace",
        "resolved_namespace_digest",
        "resolved_namespace_keys",
        "resolved_run_spec_digest",
        "schema",
    }
    _require_exact_members(document, expected_members, label="RunSpec")
    if (
        document.get("schema") != RUN_SPEC_SCHEMA
        or document.get("phase") != "phase_a"
        or document.get("namespace_schema") != NAMESPACE_SCHEMA
        or document.get("object_canonicalization") != OBJECT_CANONICALIZATION
        or document.get("artifact_serialization") != ARTIFACT_SERIALIZATION
    ):
        raise RunSpecError(
            "RunSpec schema, object canonicalization, or artifact serialization "
            "mismatch"
        )
    resolved = document.get("resolved_namespace")
    keys = document.get("resolved_namespace_keys")
    if not isinstance(resolved, dict) or len(resolved) != 454:
        raise RunSpecError("RunSpec resolved namespace cardinality is not 454")
    if (
        not isinstance(keys, list)
        or any(not isinstance(key, str) for key in keys)
        or keys != sorted(resolved)
    ):
        raise RunSpecError("RunSpec namespace key inventory is not exact and sorted")
    try:
        canonical_json_bytes(resolved)
    except (TypeError, ValueError) as exc:
        raise RunSpecError("RunSpec namespace is not finite canonical JSON") from exc
    if document.get("resolved_namespace_digest") != digest(resolved):
        raise RunSpecError("RunSpec namespace digest mismatch")
    if resolved.get("preset") is not None:
        raise RunSpecError("RunSpec runtime namespace retained a live preset identity")

    authoring = document.get("authoring_profile")
    if not isinstance(authoring, dict):
        raise RunSpecError("RunSpec authoring-profile binding is absent")
    _require_exact_members(
        authoring,
        {
            "authoring_lineage",
            "owner_decision",
            "owner_decision_sha256",
            "profile",
            "profile_schema",
            "profile_schema_sha256",
            "profile_sha256",
            "schema",
        },
        label="authoring-profile binding",
    )
    if (
        authoring.get("schema") != AUTHORING_BINDING_SCHEMA
        or authoring.get("profile") != AUTHORING_PROFILE_REL
        or authoring.get("profile_schema") != AUTHORING_PROFILE_SCHEMA_REL
        or authoring.get("owner_decision") != AUTHORING_DECISION_REL
        or not all(
            _valid_sha256(authoring.get(key))
            for key in (
                "owner_decision_sha256",
                "profile_schema_sha256",
                "profile_sha256",
            )
        )
    ):
        raise RunSpecError("RunSpec authoring-profile binding mismatch")
    lineage = authoring.get("authoring_lineage")
    if (
        not isinstance(lineage, dict)
        or lineage.get("runtime_preset_loader") is not False
    ):
        raise RunSpecError("RunSpec authoring lineage permits a runtime preset")

    projection = document.get("execution_semantics_projection")
    if not isinstance(projection, dict):
        raise RunSpecError("RunSpec execution-semantics projection is absent")
    _require_exact_members(
        projection, {"algorithm", "digest", "members", "schema"}, label="projection"
    )
    if (
        projection.get("schema") != PROJECTION_SCHEMA
        or projection.get("algorithm") != "sha256_canonical_json_content_members_v1"
    ):
        raise RunSpecError("RunSpec projection schema or algorithm mismatch")
    members = projection.get("members")
    if not isinstance(members, list) or len(members) != len(EXECUTION_SEMANTICS_PATHS):
        raise RunSpecError("RunSpec projection member cardinality mismatch")
    member_paths: list[str] = []
    for member in members:
        if not isinstance(member, dict):
            raise RunSpecError("RunSpec projection member is not an object")
        _require_exact_members(
            member, {"length", "path", "sha256"}, label="projection member"
        )
        path = member.get("path")
        if not isinstance(path, str):
            raise RunSpecError("RunSpec projection path is not a string")
        member_paths.append(path)
    if member_paths != sorted(EXECUTION_SEMANTICS_PATHS):
        raise RunSpecError("RunSpec projection does not name the declared content set")
    member_by_path = {str(member["path"]): member for member in members}
    for binding_key, relative in (
        ("profile_sha256", AUTHORING_PROFILE_REL),
        ("profile_schema_sha256", AUTHORING_PROFILE_SCHEMA_REL),
        ("owner_decision_sha256", AUTHORING_DECISION_REL),
    ):
        if member_by_path[relative]["sha256"] != authoring[binding_key]:
            raise RunSpecError(
                f"RunSpec authoring binding differs from projection: {binding_key}"
            )
    if projection.get("digest") != digest(members):
        raise RunSpecError("RunSpec projection digest mismatch")
    if document.get("execution_semantics_projection_digest") != projection.get(
        "digest"
    ):
        raise RunSpecError("RunSpec top-level projection digest mismatch")
    if verify_projection and projection != execution_semantics_projection():
        raise RunSpecError("RunSpec execution-semantics bytes changed")
    if document.get("resolved_run_spec_digest") != digest(
        _run_spec_digest_payload(document)
    ):
        raise RunSpecError("RunSpec object digest mismatch")


def runtime_argv(document: Mapping[str, Any], run_dir: Path) -> tuple[str, ...]:
    """Derive the runtime-only argv transport from a validated RunSpec."""
    validate_run_spec(document, verify_projection=True)
    if not run_dir.is_absolute():
        raise RunSpecError("runtime directory is not absolute")
    resolved = document["resolved_namespace"]
    if (
        not isinstance(resolved.get("sequences"), str)
        or resolved.get("output") != RUN_DIR_OUTPUT_TOKEN
    ):
        raise RunSpecError("RunSpec runtime argv transport is malformed")
    return (
        "--sequences",
        str(resolved["sequences"]),
        "--output",
        (run_dir / "_runtime").as_posix(),
    )


def parse_runtime_namespace(
    document: Mapping[str, Any],
    run_dir: Path,
    *,
    parser: argparse.ArgumentParser | None = None,
) -> argparse.Namespace:
    """Parse runtime argv with every parser default projected from RunSpec."""
    from mot17_args import build_parser

    validate_run_spec(document, verify_projection=True)
    selected = build_parser() if parser is None else parser
    selected.set_defaults(**document["resolved_namespace"])
    return selected.parse_args(list(runtime_argv(document, run_dir)))


def environment_projection(document: Mapping[str, Any]) -> dict[str, str]:
    """Derive the four repository-owned environment values from RunSpec."""
    validate_run_spec(document, verify_projection=True)
    resolved = document["resolved_namespace"]
    double_buffer = resolved.get("double_buffer")
    detect_barrier = resolved.get("detect_barrier")
    no_gpu_decode = resolved.get("no_gpu_decode")
    main_nms_graphed = resolved.get("main_nms_graphed")
    if not isinstance(double_buffer, bool) or not isinstance(no_gpu_decode, bool):
        raise RunSpecError("RunSpec environment booleans are malformed")
    if not isinstance(main_nms_graphed, bool):
        raise RunSpecError("RunSpec main-NMS environment projection is malformed")
    if double_buffer and detect_barrier not in {None, "event"}:
        raise RunSpecError("RunSpec double buffer requires the event barrier")
    barrier = "event" if double_buffer else (detect_barrier or "full")
    if barrier not in {"event", "full", "no_postproc"}:
        raise RunSpecError("RunSpec detect barrier is malformed")
    return {
        "SACCADE_DETECT_BARRIER": str(barrier),
        "SACCADE_DOUBLE_BUFFER": "1" if double_buffer else "0",
        "SACCADE_GPU_DECODE": "0" if no_gpu_decode else "1",
        "SACCADE_MAIN_NMS_GRAPHED": "1" if main_nms_graphed else "0",
    }


def apply_environment_projection(
    document: Mapping[str, Any], environ: MutableMapping[str, str]
) -> None:
    """Apply only the RunSpec-derived environment projection."""
    environ.update(environment_projection(document))
    environ.pop("SACCADE_STREAM_MODE", None)


def _normalized_runtime_namespace(
    args: argparse.Namespace, run_dir: Path
) -> dict[str, Any]:
    actual = dict(vars(args))
    output = actual.get("output")
    if output == (run_dir / "_runtime").as_posix():
        actual["output"] = RUN_DIR_OUTPUT_TOKEN
    return actual


def assert_runtime_matches(
    document: Mapping[str, Any],
    args: argparse.Namespace,
    environ: Mapping[str, str],
    run_dir: Path,
) -> None:
    """Assert the full parser namespace and env projection before/after use."""
    validate_run_spec(document, verify_projection=True)
    expected = document["resolved_namespace"]
    actual = _normalized_runtime_namespace(args, run_dir)
    if actual != expected:
        differing = sorted(
            key
            for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        )
        raise RunSpecError(f"runtime namespace differs from RunSpec: {differing}")
    projected = environment_projection(document)
    mismatches = sorted(
        key for key, value in projected.items() if environ.get(key) != value
    )
    if mismatches:
        raise RunSpecError(f"runtime environment differs from RunSpec: {mismatches}")
    if "SACCADE_STREAM_MODE" in environ:
        raise RunSpecError("runtime environment retained SACCADE_STREAM_MODE")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--emit", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    document = build_run_spec()
    target = args.emit
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json_bytes(document) + b"\n")
    print(document["resolved_run_spec_digest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
