#!/usr/bin/env python3
"""Produce the canonical Phase-A ``h2_measurement_freeze_v1`` record.

This is a producer, never an authority source.  It derives every binding from
the exact Git head and already-produced Layer-P/runtime-identity artifacts.  It
does not issue ``I``, ``F``, ``S`` or an exactly-once authorization.
"""
# status: stable

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = REPO_ROOT / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import build_runtime_identity as identity  # noqa: E402
import h2_behavioral_identity as behavior  # noqa: E402
import h2_measurement_evidence as evidence  # noqa: E402
import h2_runtime_inputs as runtime_inputs  # noqa: E402
from run_h2_layer_p import CERTIFICATE_SCHEMA  # noqa: E402


class FreezeError(RuntimeError):
    """The requested freeze cannot be reconstructed exactly."""


def _hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _git(*args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
        ).stdout
    except subprocess.SubprocessError as exc:
        raise FreezeError(f"git {' '.join(args)} failed: {exc}") from exc


def _git_text(*args: str) -> str:
    return _git(*args).decode("utf-8").strip()


def _load_canonical(path: Path, *, schema: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FreezeError(f"input is not a physical regular file: {path}")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FreezeError(f"input is unreadable JSON: {path} ({exc})") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema") != schema
        or raw != evidence.canonical_json_bytes(value) + b"\n"
    ):
        raise FreezeError(f"input is not canonical {schema}: {path}")
    return value


def git_surface_digests(head: str) -> dict[str, str]:
    if not _hex(head, 40):
        raise FreezeError("instrumentation head is not full lowercase 40-hex")
    return {
        path: hashlib.sha256(_git("show", f"{head}:{path}")).hexdigest()
        for path in evidence.PHASE_A_EXECUTED_SURFACE_PATHS
    }


def build_freeze(
    *,
    certificate: Mapping[str, Any],
    certificate_digest: str,
    reference_probe: Mapping[str, Any],
    reference_probe_file_digest: str,
    runtime_manifest: Mapping[str, Any],
    runtime_manifest_file_digest: str,
    published_identity: Mapping[str, Any],
    published_identity_file_digest: str,
    executed_surfaces: Mapping[str, str],
    capture_abi_digest: str,
) -> dict[str, Any]:
    head = certificate.get("source_head")
    selected_base = certificate.get("selected_base")
    changed_path_verdict = certificate.get("changed_path_verdict")
    coordinate = certificate.get("published_coordinate")
    probe = certificate.get("behavior_probe")
    build_artifacts = runtime_manifest.get("build_artifacts")
    if not _hex(head, 40):
        raise FreezeError("certificate source_head is not full lowercase 40-hex")
    if not _hex(selected_base, 40):
        raise FreezeError("certificate selected_base is not full lowercase 40-hex")
    if (
        not isinstance(changed_path_verdict, Mapping)
        or changed_path_verdict.get("admissible") is not True
        or changed_path_verdict.get("base") != selected_base
    ):
        raise FreezeError(
            "certificate changed-path verdict base/admissibility differs from "
            "selected_base"
        )
    if set(executed_surfaces) != set(evidence.PHASE_A_EXECUTED_SURFACE_PATHS) or any(
        not _hex(value, 64) for value in executed_surfaces.values()
    ):
        raise FreezeError("executed surface binding is incomplete or malformed")
    if not _hex(capture_abi_digest, 64):
        raise FreezeError("capture ABI digest is not full lowercase 64-hex")
    if (
        certificate.get("schema") != CERTIFICATE_SCHEMA
        or not _hex(certificate_digest, 64)
        or not isinstance(coordinate, Mapping)
        or set(coordinate) != set(identity.ALL_COORDINATE_AXES)
        or any(
            not _hex(coordinate.get(axis), 64) for axis in identity.ALL_COORDINATE_AXES
        )
        or not _hex(probe, 64)
        or reference_probe.get("schema") != behavior.RESULT_SCHEMA
        or reference_probe.get("digest") != probe
        or reference_probe.get("identical") is not True
        or reference_probe.get("mode") != "identity"
        or reference_probe.get("sequence") != behavior.IDENTITY_SEQUENCE
        or published_identity.get("schema") != identity.IDENTITY_SCHEMA
        or published_identity.get("coordinate") != coordinate
        or not isinstance(published_identity.get("probe"), Mapping)
        or published_identity["probe"].get("digest") != probe
        or not isinstance(published_identity.get("equivalence"), Mapping)
        or published_identity["equivalence"].get("state") != "unproven"
        or published_identity.get("publication_complete") is not True
        or runtime_manifest.get("schema") != runtime_inputs.SCHEMA
        or not isinstance(build_artifacts, Mapping)
        or not (
            certificate.get("runtime_input_coordinate_digest")
            == runtime_manifest.get("coordinate_digest")
            == coordinate.get("runtime_inputs")
        )
        or certificate.get("runtime_input_full_digest")
        != runtime_manifest.get("full_digest")
        or certificate.get("build_artifact_digest") != build_artifacts.get("digest")
        or certificate.get("probe_result_file_digest") != reference_probe_file_digest
        or certificate.get("runtime_input_manifest_file_digest")
        != runtime_manifest_file_digest
        or certificate.get("published_identity_file_digest")
        != published_identity_file_digest
        or certificate.get("published_probe") != probe
        or certificate.get("equivalence") != "unproven"
    ):
        raise FreezeError("primary certificate/runtime identity bindings disagree")
    for value, label in (
        (reference_probe_file_digest, "reference probe file"),
        (runtime_manifest_file_digest, "runtime-input file"),
        (published_identity_file_digest, "published-identity file"),
        (runtime_manifest.get("coordinate_digest"), "runtime coordinate"),
        (runtime_manifest.get("full_digest"), "runtime full"),
        (build_artifacts.get("digest"), "build artifact"),
    ):
        if not _hex(value, 64):
            raise FreezeError(f"{label} digest is malformed")

    freeze = {
        "schema": evidence.FREEZE_SCHEMA,
        "capture_phase": evidence.CAPTURE_PHASE["a"],
        "instrumentation_head": head,
        "selected_base": selected_base,
        "coordinate": dict(coordinate),
        "probe": probe,
        "equivalence": "unproven",
        "layer_p_certificate": {
            "schema": CERTIFICATE_SCHEMA,
            "digest": certificate_digest,
        },
        "reference_probe": {
            "schema": behavior.RESULT_SCHEMA,
            "file_digest": reference_probe_file_digest,
        },
        "runtime_inputs": {
            "schema": runtime_inputs.SCHEMA,
            "file_digest": runtime_manifest_file_digest,
            "coordinate_digest": runtime_manifest["coordinate_digest"],
            "full_digest": runtime_manifest["full_digest"],
            "build_artifact_digest": build_artifacts["digest"],
        },
        "published_identity": {
            "schema": identity.IDENTITY_SCHEMA,
            "file_digest": published_identity_file_digest,
        },
        "capture_abi": {
            "path": evidence.PHASE_A_CAPTURE_ABI_PATH,
            "sha256": capture_abi_digest,
        },
        "executed_surfaces": dict(sorted(executed_surfaces.items())),
        "run_plan": {
            "sequence": evidence.expected_sequences("a")[0],
            "run_ids": list(evidence.RUN_IDS),
        },
    }
    if set(freeze) != evidence.PHASE_A_FREEZE_MEMBERS:
        raise FreezeError("internal Phase-A freeze member drift")
    return freeze


def produce(
    *,
    certificate_path: Path,
    reference_probe_path: Path,
    runtime_manifest_path: Path,
    published_identity_path: Path,
) -> dict[str, Any]:
    certificate = _load_canonical(certificate_path, schema=CERTIFICATE_SCHEMA)
    reference = _load_canonical(reference_probe_path, schema=behavior.RESULT_SCHEMA)
    published = _load_canonical(
        published_identity_path, schema=identity.IDENTITY_SCHEMA
    )
    try:
        manifest = runtime_inputs.load_manifest(
            runtime_manifest_path, verify_files=True
        )
    except (runtime_inputs.RuntimeInputError, OSError) as exc:
        raise FreezeError(f"runtime-input manifest rejected: {exc}") from exc
    head = certificate.get("source_head")
    if _git_text("rev-parse", "HEAD") != head:
        raise FreezeError("certificate source_head differs from checkout HEAD")
    if _git_text("status", "--porcelain", "--untracked-files=normal"):
        raise FreezeError("freeze production requires a clean checkout")
    selected_base = certificate.get("selected_base")
    changed_path_verdict = certificate.get("changed_path_verdict")
    if (
        not _hex(selected_base, 40)
        or _git_text("rev-parse", f"{selected_base}^{{commit}}") != selected_base
        or not isinstance(changed_path_verdict, Mapping)
        or changed_path_verdict.get("admissible") is not True
        or changed_path_verdict.get("base") != selected_base
    ):
        raise FreezeError(
            "selected_base is not an exact available commit or the changed-path "
            "verdict is not admissible for that base"
        )
    return build_freeze(
        certificate=certificate,
        certificate_digest=evidence.digest(certificate),
        reference_probe=reference,
        reference_probe_file_digest=evidence.sha256_file(reference_probe_path),
        runtime_manifest=manifest,
        runtime_manifest_file_digest=evidence.sha256_file(runtime_manifest_path),
        published_identity=published,
        published_identity_file_digest=evidence.sha256_file(published_identity_path),
        executed_surfaces=git_surface_digests(str(head)),
        capture_abi_digest=hashlib.sha256(
            _git("show", f"{head}:{evidence.PHASE_A_CAPTURE_ABI_PATH}")
        ).hexdigest(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--layer-p-certificate", type=Path, required=True)
    parser.add_argument("--reference-probe", type=Path, required=True)
    parser.add_argument("--runtime-inputs", type=Path, required=True)
    parser.add_argument("--published-identity", type=Path, required=True)
    parser.add_argument("--emit", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        freeze = produce(
            certificate_path=args.layer_p_certificate,
            reference_probe_path=args.reference_probe,
            runtime_manifest_path=args.runtime_inputs,
            published_identity_path=args.published_identity,
        )
        if args.emit.exists() or args.emit.is_symlink():
            raise FreezeError(f"refusing to overwrite freeze: {args.emit}")
        evidence.write_document(args.emit.parent, args.emit.name, freeze)
    except (
        FreezeError,
        evidence.EvidenceError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"H2 freeze rejected: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.emit}")
    print(f"F64 {evidence.freeze_digest(freeze)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
