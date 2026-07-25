#!/usr/bin/env python3
"""Bind the fixtures and runtime assets consumed by H2.

Git identity cannot cover ignored datasets, weights, engines, or build outputs.
This module hashes their content into two coordinates:

* ``coordinate_digest`` binds stable measurement inputs and runtime assets;
* ``full_digest`` additionally binds the physical extension/plugin used by one
  Layer-P pass and is what the eventual freeze must consume.

The build-artifact portion is deliberately not an equivalence claim. Different
builds may have different bytes; the certificate records which bytes Layer M is
allowed to consume.
"""
# status: stable

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "h2_runtime_input_manifest_v1"
POLICY_PRESET_REL = "configs/presets/mamba_whole_graph_m.yaml"
IDENTITY_SEQUENCE = "MOT17-09-SDP"
MEASUREMENT_SEQUENCE = "MOT17-04-SDP"
DEFAULT_DATA_ROOT = "datasets/MOT17"

# These are passed directly to build_mamba_gated_detector by the probe runner.
RUNTIME_ASSET_FIELDS = (
    "mamba_ckpt",
    "mamba_teacher_ckpt",
    "mamba_yolo_weights",
    "fpn_backbone_engine",
    "mamba_head_engine",
)


class RuntimeInputError(RuntimeError):
    pass


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _shown(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return resolved.relative_to(REPO_ROOT).as_posix()
    return resolved.as_posix()


def _record(*, path: Path, role: str, coordinate: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeInputError(f"runtime input is not a regular file: {resolved}")
    return {
        "coordinate": coordinate,
        "length": resolved.stat().st_size,
        "resolved_path": resolved.as_posix(),
        "role": role,
        "sha256": sha256_file(resolved),
    }


def _digest_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "coordinate": record["coordinate"],
        "length": record["length"],
        "role": record["role"],
        "sha256": record["sha256"],
    }


def _section(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    members = sorted(records, key=lambda item: (item["role"], item["coordinate"]))
    if not members:
        raise RuntimeInputError("runtime-input section is empty")
    projection = [_digest_projection(item) for item in members]
    return {"digest": digest(projection), "file_count": len(members), "files": members}


def _fixture_section(*, data_root: Path, sequence: str, role: str) -> dict[str, Any]:
    sequence_root = (data_root / "train" / sequence).resolve(strict=True)
    if not (sequence_root / "seqinfo.ini").is_file():
        raise RuntimeInputError(f"{sequence}: seqinfo.ini is absent")
    if not (sequence_root / "img1").is_dir():
        raise RuntimeInputError(f"{sequence}: img1 is absent")
    records = [
        _record(
            path=path,
            role=role,
            coordinate=f"{sequence}/{path.relative_to(sequence_root).as_posix()}",
        )
        for path in sequence_root.rglob("*")
        if path.is_file()
    ]
    section = _section(records)
    section["sequence"] = sequence
    section["root"] = _shown(sequence_root)
    return section


def _runtime_assets_section(preset_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(preset_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise RuntimeInputError(f"{preset_path}: preset is not a mapping")
    records: list[dict[str, Any]] = []
    for field in RUNTIME_ASSET_FIELDS:
        configured = payload.get(field)
        if not isinstance(configured, str) or not configured:
            raise RuntimeInputError(
                f"{preset_path}: required runtime asset {field} absent"
            )
        path = Path(configured)
        if not path.is_absolute():
            path = REPO_ROOT / path
        records.append(_record(path=path, role=field, coordinate=configured))
    return _section(records)


def _third_party_section() -> dict[str, Any]:
    root = REPO_ROOT / "third_party" / "TrackEval" / "trackeval"
    records = [
        _record(
            path=path,
            role="third_party_runtime_component",
            coordinate=path.relative_to(REPO_ROOT).as_posix(),
        )
        for path in root.rglob("*.py")
        if path.is_file()
    ]
    return _section(records)


def _build_artifacts_section(build_dir: Path) -> dict[str, Any]:
    resolved = build_dir.resolve(strict=True)
    extension = sorted(resolved.glob("saccade_tracking_ext*.so"))
    plugins = sorted(resolved.glob("libsaccade_scan_plugin.so"))
    if len(extension) != 1:
        raise RuntimeInputError(
            f"{resolved}: expected one saccade_tracking_ext*.so, found {len(extension)}"
        )
    if len(plugins) != 1:
        raise RuntimeInputError(
            f"{resolved}: expected one libsaccade_scan_plugin.so, found {len(plugins)}"
        )
    records = [
        _record(
            path=extension[0],
            role="tracking_extension",
            coordinate=extension[0].name,
        ),
        _record(
            path=plugins[0],
            role="tensorrt_scan_plugin",
            coordinate=plugins[0].name,
        ),
    ]
    section = _section(records)
    section["build_dir"] = resolved.as_posix()
    return section


def build_manifest(
    *,
    build_dir: Path,
    data_root: Path | None = None,
    identity_sequence: str = IDENTITY_SEQUENCE,
    measurement_sequence: str = MEASUREMENT_SEQUENCE,
) -> dict[str, Any]:
    """Hash every declared fixture, asset, third-party component and build output."""
    if identity_sequence != IDENTITY_SEQUENCE:
        raise RuntimeInputError(
            f"identity fixture must be {IDENTITY_SEQUENCE}, got {identity_sequence}"
        )
    if measurement_sequence != MEASUREMENT_SEQUENCE:
        raise RuntimeInputError(
            f"measurement fixture must be {MEASUREMENT_SEQUENCE}, got {measurement_sequence}"
        )
    root = data_root or (REPO_ROOT / DEFAULT_DATA_ROOT)
    sections = {
        "identity_fixture": _fixture_section(
            data_root=root, sequence=identity_sequence, role="identity_fixture_input"
        ),
        "measurement_fixture": _fixture_section(
            data_root=root,
            sequence=measurement_sequence,
            role="measurement_fixture_input",
        ),
        "runtime_assets": _runtime_assets_section(REPO_ROOT / POLICY_PRESET_REL),
        "third_party_runtime": _third_party_section(),
        "build_artifacts": _build_artifacts_section(build_dir),
    }
    coordinate_digest = digest(
        {
            name: sections[name]["digest"]
            for name in (
                "identity_fixture",
                "measurement_fixture",
                "runtime_assets",
                "third_party_runtime",
            )
        }
    )
    full_digest = digest(
        {
            "coordinate_digest": coordinate_digest,
            "build_artifacts": sections["build_artifacts"]["digest"],
        }
    )
    return {
        **sections,
        "coordinate_digest": coordinate_digest,
        "full_digest": full_digest,
        "policy_preset": POLICY_PRESET_REL,
        "schema": SCHEMA,
    }


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_manifest(
    payload: Mapping[str, Any], *, verify_files: bool = False
) -> dict[str, Any]:
    if payload.get("schema") != SCHEMA:
        raise RuntimeInputError(f"not a {SCHEMA} payload")
    if payload.get("policy_preset") != POLICY_PRESET_REL:
        raise RuntimeInputError("runtime-input manifest preset mismatch")
    section_names = (
        "identity_fixture",
        "measurement_fixture",
        "runtime_assets",
        "third_party_runtime",
        "build_artifacts",
    )
    sections: dict[str, Mapping[str, Any]] = {}
    for name in section_names:
        section = payload.get(name)
        if not isinstance(section, Mapping):
            raise RuntimeInputError(f"runtime-input manifest missing {name}")
        files = section.get("files")
        if not isinstance(files, list) or not files:
            raise RuntimeInputError(f"runtime-input section {name} is empty")
        coordinates: set[tuple[str, str]] = set()
        for record in files:
            if not isinstance(record, Mapping):
                raise RuntimeInputError(f"{name}: malformed file record")
            key = (str(record.get("role")), str(record.get("coordinate")))
            if key in coordinates:
                raise RuntimeInputError(f"{name}: duplicate runtime input {key}")
            coordinates.add(key)
            if not _valid_sha256(record.get("sha256")):
                raise RuntimeInputError(f"{name}: invalid sha256 for {key}")
            if not isinstance(record.get("length"), int) or record["length"] < 0:
                raise RuntimeInputError(f"{name}: invalid length for {key}")
            if verify_files:
                resolved = Path(str(record.get("resolved_path", "")))
                if not resolved.is_file():
                    raise RuntimeInputError(f"{name}: bound file absent: {resolved}")
                if resolved.stat().st_size != record["length"]:
                    raise RuntimeInputError(f"{name}: length moved: {resolved}")
                if sha256_file(resolved) != record["sha256"]:
                    raise RuntimeInputError(f"{name}: content moved: {resolved}")
        expected = digest([_digest_projection(item) for item in files])
        if section.get("digest") != expected:
            raise RuntimeInputError(f"{name}: section digest mismatch")
        if section.get("file_count") != len(files):
            raise RuntimeInputError(f"{name}: file_count mismatch")
        sections[name] = section
    if sections["identity_fixture"].get("sequence") != IDENTITY_SEQUENCE:
        raise RuntimeInputError("identity fixture sequence mismatch")
    if sections["measurement_fixture"].get("sequence") != MEASUREMENT_SEQUENCE:
        raise RuntimeInputError("measurement fixture sequence mismatch")
    coordinate = digest(
        {
            name: sections[name]["digest"]
            for name in (
                "identity_fixture",
                "measurement_fixture",
                "runtime_assets",
                "third_party_runtime",
            )
        }
    )
    if payload.get("coordinate_digest") != coordinate:
        raise RuntimeInputError("runtime-input coordinate digest mismatch")
    full = digest(
        {
            "coordinate_digest": coordinate,
            "build_artifacts": sections["build_artifacts"]["digest"],
        }
    )
    if payload.get("full_digest") != full:
        raise RuntimeInputError("runtime-input full digest mismatch")
    return dict(payload)


def load_manifest(path: Path, *, verify_files: bool = False) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeInputError(
            f"{path}: unreadable runtime-input manifest: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RuntimeInputError(f"{path}: runtime-input manifest is not a mapping")
    return validate_manifest(payload, verify_files=verify_files)


def bound_paths(payload: Mapping[str, Any]) -> tuple[Path, ...]:
    validated = validate_manifest(payload)
    paths: set[Path] = set()
    for name in (
        "identity_fixture",
        "measurement_fixture",
        "runtime_assets",
        "third_party_runtime",
        "build_artifacts",
    ):
        for record in validated[name]["files"]:
            paths.add(Path(record["resolved_path"]))
    return tuple(sorted(paths, key=lambda item: item.as_posix()))


def publication_axis(payload: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_manifest(payload)
    return {
        "digest": validated["coordinate_digest"],
        "identity_fixture": {
            "digest": validated["identity_fixture"]["digest"],
            "file_count": validated["identity_fixture"]["file_count"],
            "sequence": IDENTITY_SEQUENCE,
        },
        "measurement_fixture": {
            "digest": validated["measurement_fixture"]["digest"],
            "file_count": validated["measurement_fixture"]["file_count"],
            "sequence": MEASUREMENT_SEQUENCE,
        },
        "runtime_assets": {
            "digest": validated["runtime_assets"]["digest"],
            "file_count": validated["runtime_assets"]["file_count"],
        },
        "third_party_runtime": {
            "digest": validated["third_party_runtime"]["digest"],
            "file_count": validated["third_party_runtime"]["file_count"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--emit", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = build_manifest(build_dir=args.build_dir, data_root=args.data_root)
    except (RuntimeInputError, OSError) as exc:
        print(f"runtime-input manifest failed: {exc}", file=sys.stderr)
        return 1
    args.emit.parent.mkdir(parents=True, exist_ok=True)
    args.emit.write_bytes(canonical_json_bytes(payload) + b"\n")
    print(f"wrote {args.emit}")
    print(f"  coordinate_digest {payload['coordinate_digest']}")
    print(f"  full_digest       {payload['full_digest']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
