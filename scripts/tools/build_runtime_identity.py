#!/usr/bin/env python3
"""Publish an H2 runtime coordinate plus a bounded behavior probe.

The publication deliberately keeps three concepts separate:

* ``coordinate`` versions source, configuration, environment, identity semantics,
  and content-bound stable runtime inputs;
* ``probe`` records what MOT17-09 observed under the deterministic probe mode;
* ``equivalence`` is unproven. Probe equality never upgrades it.

This v1 publication remains a historical/diagnostic coordinate. Successor
executions bind the physical extension/plugin bytes they actually consume in
``runtime_binding.json``; they do not derive validity from reproducing this
publication on another host.
"""
# status: stable

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = REPO_ROOT / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import h2_behavioral_identity as behavior  # noqa: E402
import h2_path_partition as partition  # noqa: E402
import h2_runtime_inputs as runtime_inputs  # noqa: E402

IDENTITY_SCHEMA = "h2_runtime_coordinate_probe_v1"
ENVIRONMENT_FILES = ("CMakeLists.txt", "pyproject.toml", "uv.lock")
STATIC_COORDINATE_AXES = (
    "decision_surface",
    "environment",
    "implementation",
    "identity_semantics",
)
ALL_COORDINATE_AXES = (*STATIC_COORDINATE_AXES, "runtime_inputs")


class IdentityError(RuntimeError):
    pass


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def decision_surface_axis() -> dict[str, Any]:
    for extra in (
        REPO_ROOT / "scripts" / "eval",
        REPO_ROOT / "scripts" / "eval" / "config",
        REPO_ROOT / "src",
    ):
        if extra.as_posix() not in sys.path:
            sys.path.insert(0, extra.as_posix())
    from gen_golden_snapshot import _resolve_config
    from resolved_bridge_policy_config import fingerprint

    resolved = _resolve_config(behavior.POLICY_PRESET_STEM)
    resolved_fingerprint = fingerprint(behavior.POLICY_PRESET_STEM)
    return {
        "digest": behavior.digest(
            {
                "preset": behavior.POLICY_PRESET_REL,
                "resolved_bridge_policy_config_v1": resolved_fingerprint,
                "resolved_parameters": resolved,
            }
        ),
        "parameter_count": len(resolved),
        "preset": behavior.POLICY_PRESET_REL,
        "resolved_bridge_policy_config_v1": resolved_fingerprint,
    }


def tracked_files_for_class(path_class: partition.PathClass) -> tuple[str, ...]:
    tracked = [
        line
        for line in _git(
            "ls-files", "--cached", "--others", "--exclude-standard"
        ).splitlines()
        if line
    ]
    return tuple(
        sorted(
            path
            for path in tracked
            if partition.classify(path) == path_class
            and not path.endswith((".md", ".rst", ".txt"))
        )
    )


def _content_axis(path_class: partition.PathClass) -> dict[str, Any]:
    files = tracked_files_for_class(path_class)
    if not files:
        raise IdentityError(f"{path_class} file set is empty")
    members = []
    for path in files:
        blob = _git("hash-object", "--", path)
        if len(blob) != 40:
            raise IdentityError(f"unexpected blob id for {path}: {blob!r}")
        members.append({"blob": blob, "path": path})
    return {
        "digest": behavior.digest(members),
        "file_count": len(members),
        "files": members,
    }


def decision_relevant_files() -> tuple[str, ...]:
    return tracked_files_for_class("decision_relevant")


def implementation_axis() -> dict[str, Any]:
    return _content_axis("decision_relevant")


def identity_semantics_axis() -> dict[str, Any]:
    return _content_axis("identity_semantics")


def plumbing_axis() -> dict[str, Any]:
    return _content_axis("plumbing_only")


def _torch_environment() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment probe
        return {"available": False, "error": type(exc).__name__}
    info: dict[str, Any] = {
        "available": True,
        "cuda_compiled": getattr(torch.version, "cuda", None),
        "cudnn": None,
        "torch": torch.__version__,
    }
    try:
        info["cudnn"] = torch.backends.cudnn.version()
    except Exception:  # pragma: no cover
        pass
    try:
        if torch.cuda.is_available():
            info["driver_device_count"] = torch.cuda.device_count()
            info["device_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception:  # pragma: no cover
        pass
    try:
        import tensorrt

        info["tensorrt"] = tensorrt.__version__
    except Exception:
        info["tensorrt"] = None
    return info


def environment_axis() -> dict[str, Any]:
    recipe = []
    for name in ENVIRONMENT_FILES:
        path = REPO_ROOT / name
        if not path.is_file():
            raise IdentityError(f"environment input is absent: {name}")
        recipe.append({"blob": _git("hash-object", "--", name), "path": name})
    toolchain = _torch_environment()
    return {
        "digest": behavior.digest({"recipe": recipe, "toolchain": toolchain}),
        "recipe": recipe,
        "toolchain": toolchain,
    }


def _validate_build_witness(witness: Any, *, verify_files: bool) -> dict[str, Any]:
    if not isinstance(witness, Mapping):
        raise IdentityError("probe build_witness is missing")
    if witness.get("schema") != behavior.BUILD_WITNESS_SCHEMA:
        raise IdentityError("probe build_witness schema mismatch")
    artifacts = witness.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 2:
        raise IdentityError("probe build_witness must bind extension and plugin")
    roles = {item.get("role") for item in artifacts if isinstance(item, Mapping)}
    if roles != {"tracking_extension", "tensorrt_scan_plugin"}:
        raise IdentityError(f"probe build_witness roles are invalid: {roles}")
    projection = []
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise IdentityError("probe build_witness artifact is malformed")
        if not _valid_sha256(item.get("sha256")):
            raise IdentityError("probe build_witness artifact sha256 is invalid")
        if not isinstance(item.get("length"), int) or item["length"] <= 0:
            raise IdentityError("probe build_witness artifact length is invalid")
        projection.append(
            {
                "coordinate": item.get("coordinate"),
                "length": item["length"],
                "role": item["role"],
                "sha256": item["sha256"],
            }
        )
        if verify_files:
            path = Path(str(item.get("path", "")))
            if not path.is_file():
                raise IdentityError(f"probe build_witness artifact is absent: {path}")
            if (
                path.stat().st_size != item["length"]
                or sha256_file(path) != item["sha256"]
            ):
                raise IdentityError(f"probe build_witness artifact moved: {path}")
    if witness.get("digest") != behavior.digest(projection):
        raise IdentityError("probe build_witness digest mismatch")
    return dict(witness)


def _validate_probe_payload(
    payload: Any,
    *,
    source: Path,
    expected_mode: str,
    expected_sequence: str,
    expected_pinning: bool,
    minimum_repeats: int,
    verify_witness_files: bool,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise IdentityError(f"{source}: behavior probe payload is not a mapping")
    if payload.get("schema") != behavior.RESULT_SCHEMA:
        raise IdentityError(f"{source}: not a {behavior.RESULT_SCHEMA} payload")
    if payload.get("mode") != expected_mode:
        raise IdentityError(f"{source}: expected mode {expected_mode!r}")
    if payload.get("sequence") != expected_sequence:
        raise IdentityError(f"{source}: expected sequence {expected_sequence!r}")
    if payload.get("preset") != behavior.POLICY_PRESET_REL:
        raise IdentityError(f"{source}: probe preset mismatch")
    if payload.get("determinism_pinned") is not expected_pinning:
        raise IdentityError(f"{source}: determinism pinning mismatch")
    repeats = payload.get("repeats")
    if not isinstance(repeats, int) or repeats < minimum_repeats:
        raise IdentityError(f"{source}: repeats must be >= {minimum_repeats}")
    digests = payload.get("digests")
    if not isinstance(digests, list) or len(digests) != repeats:
        raise IdentityError(f"{source}: digests/repeats cardinality mismatch")
    if payload.get("identical") is not True or len(set(digests)) != 1:
        raise IdentityError(f"{source}: repeats disagreed")
    probe_digest = payload.get("digest")
    if not _valid_sha256(probe_digest) or any(
        not _valid_sha256(item) for item in digests
    ):
        raise IdentityError(f"{source}: probe digest is not a valid SHA-256")
    if probe_digest != digests[0]:
        raise IdentityError(f"{source}: digest does not match repeat digests")
    recorder_digest = payload.get("recorder_sha256")
    current_recorder = sha256_file(Path(behavior.__file__))
    if recorder_digest != current_recorder:
        raise IdentityError(
            f"{source}: behavior-probe recorder does not match this tree"
        )
    expected_fingerprint = decision_surface_axis()["resolved_bridge_policy_config_v1"]
    if payload.get("resolved_fingerprint") != expected_fingerprint:
        raise IdentityError(f"{source}: resolved_fingerprint does not match this tree")
    witness = _validate_build_witness(
        payload.get("build_witness"), verify_files=verify_witness_files
    )
    return {
        "build_witness": witness,
        "determinism_pinned": expected_pinning,
        "digest": probe_digest,
        "mode": expected_mode,
        "preset": behavior.POLICY_PRESET_REL,
        "repeats": repeats,
        "recorder_sha256": current_recorder,
        "resolved_fingerprint": expected_fingerprint,
        "schema": behavior.RESULT_SCHEMA,
        "sequence": expected_sequence,
        "state": "computed",
    }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IdentityError(f"{path}: unreadable JSON: {exc}") from exc


def load_identity_behavior_probe(
    path: Path, *, verify_witness_files: bool = True
) -> dict[str, Any]:
    return _validate_probe_payload(
        _read_json(path),
        source=path,
        expected_mode="identity",
        expected_sequence=behavior.IDENTITY_SEQUENCE,
        expected_pinning=True,
        minimum_repeats=1,
        verify_witness_files=verify_witness_files,
    )


def load_production_repeat_probe(
    path: Path, *, verify_witness_files: bool = True
) -> dict[str, Any]:
    """Load G2 evidence only; this result can never populate the identity probe."""
    return _validate_probe_payload(
        _read_json(path),
        source=path,
        expected_mode="production",
        expected_sequence=behavior.MEASUREMENT_SEQUENCE,
        expected_pinning=False,
        minimum_repeats=2,
        verify_witness_files=verify_witness_files,
    )


def witness() -> dict[str, Any]:
    return {
        "head": _git("rev-parse", "HEAD"),
        "note": (
            "HEAD/tree are navigation witness. The coordinate digests working-tree "
            "content; the Layer-P certificate additionally requires a selected base."
        ),
        "tree": _git("rev-parse", "HEAD^{tree}"),
        "worktree_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }


def build_publication(
    *,
    probe: dict[str, Any] | None,
    runtime_input_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    axes: dict[str, dict[str, Any]] = {
        "decision_surface": decision_surface_axis(),
        "environment": environment_axis(),
        "implementation": implementation_axis(),
        "identity_semantics": identity_semantics_axis(),
    }
    if runtime_input_manifest is None:
        axes["runtime_inputs"] = {"digest": None, "state": "not_computed"}
    else:
        axes["runtime_inputs"] = runtime_inputs.publication_axis(runtime_input_manifest)
    coordinate = {name: axes[name].get("digest") for name in ALL_COORDINATE_AXES}
    if probe is None:
        probe_axis = {
            "digest": None,
            "kind": "identity_probe",
            "state": "not_computed",
            "sufficiency": "none",
        }
    else:
        probe_axis = {
            **probe,
            "kind": "identity_probe",
            "sufficiency": "fixture_change_detector_only",
        }
    complete = all(coordinate.values()) and probe_axis["digest"] is not None
    return {
        "axes": axes,
        "coordinate": coordinate,
        "equivalence": {
            "proof": None,
            "state": "unproven",
            "note": (
                "Probe equality is not measurement-domain equivalence. A future "
                "accepted proof requires a versioned verifier and full declared scope."
            ),
        },
        "probe": probe_axis,
        "publication_complete": bool(complete),
        "schema": IDENTITY_SCHEMA,
        "witness": witness(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", type=Path, default=None)
    parser.add_argument("--probe-from", type=Path, default=None)
    parser.add_argument("--runtime-inputs-from", type=Path, default=None)
    parser.add_argument("--run-probe", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=None)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.probe_from and args.run_probe:
            parser.error("--probe-from and --run-probe are mutually exclusive")
        probe: dict[str, Any] | None = None
        manifest: dict[str, Any] | None = None
        if args.probe_from:
            probe = load_identity_behavior_probe(args.probe_from)
        elif args.run_probe:
            if args.build_dir is None:
                parser.error("--run-probe requires --build-dir")
            previous_build_path = os.environ.get("SACCADE_BUILD_PATH")
            os.environ["SACCADE_BUILD_PATH"] = args.build_dir.resolve().as_posix()
            try:
                result = behavior.run_behavior_inventory(
                    sequence=behavior.IDENTITY_SEQUENCE,
                    identity_mode=True,
                    output_dir=REPO_ROOT / "out" / "h2_behavior" / "identity",
                )
            finally:
                if previous_build_path is None:
                    os.environ.pop("SACCADE_BUILD_PATH", None)
                else:
                    os.environ["SACCADE_BUILD_PATH"] = previous_build_path
            payload = {
                "build_witness": result["build_witness"],
                "determinism_pinned": True,
                "digest": result["digest"],
                "digests": [result["digest"]],
                "identical": True,
                "mode": result["mode"],
                "preset": result["preset"],
                "repeats": 1,
                "recorder_sha256": sha256_file(Path(behavior.__file__)),
                "resolved_fingerprint": result["resolved_fingerprint"],
                "schema": behavior.RESULT_SCHEMA,
                "sequence": result["sequence"],
            }
            probe = _validate_probe_payload(
                payload,
                source=Path("<in-process>"),
                expected_mode="identity",
                expected_sequence=behavior.IDENTITY_SEQUENCE,
                expected_pinning=True,
                minimum_repeats=1,
                verify_witness_files=True,
            )
        if args.runtime_inputs_from:
            manifest = runtime_inputs.load_manifest(
                args.runtime_inputs_from, verify_files=True
            )
        elif args.build_dir is not None:
            manifest = runtime_inputs.build_manifest(build_dir=args.build_dir)
        publication = build_publication(probe=probe, runtime_input_manifest=manifest)
    except (IdentityError, runtime_inputs.RuntimeInputError, OSError) as exc:
        print(f"runtime coordinate publication failed: {exc}", file=sys.stderr)
        return 1

    output = behavior.canonical_json_bytes(publication) + b"\n"
    if args.emit:
        args.emit.parent.mkdir(parents=True, exist_ok=True)
        args.emit.write_bytes(output)
        print(f"wrote {args.emit}")
    else:
        sys.stdout.write(output.decode("utf-8"))
    for name, value in publication["coordinate"].items():
        print(f"  coordinate.{name:18} {value}")
    print(f"  probe.behavior       {publication['probe']['digest']}")
    print("  equivalence          unproven")
    if args.require_complete and not publication["publication_complete"]:
        print("coordinate/probe publication is incomplete", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
