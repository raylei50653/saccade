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

**All seven measurement sequences are bound in both phases** (declaration
§ C3.2 item 4 / § C3.9 item 2). Phase A executes only ``MOT17-04-SDP``, but
§ C3.1(b) requires the five published axes to be *equal* across Phase A and
Phase B, and the `runtime_inputs` axis is one of them. A Phase-A manifest that
bound one measurement sequence would have to move at the Phase-B freeze —
inside the window in which `identity_semantics` is frozen — so the seven-sequence
member set is established before the Phase-A seal, not at the point of use.
``MOT17-09-SDP`` therefore appears twice, in the two roles § C3.2 forbids
conflating: once as the Layer-P identity fixture, once as a measurement
sequence.

The two digests are computed over fixed section member sets, named once in
``COORDINATE_SECTIONS`` / ``FULL_DIGEST_SECTIONS``:

* ``coordinate_digest`` — the four sections above; ``build_artifacts`` is
  excluded by construction so one published coordinate can span builds;
* ``full_digest`` — the coordinate together with ``build_artifacts``, which is
  what pins the exact executed extension and TensorRT-plugin bytes.

``phase_a_evidence`` (§ C3.8) is an ``F_B``-only section and a member of
*neither* digest: it is absent until Phase A has passed, so admitting it to
either would make an axis undefined until after the seal that freezes it. It is
bound by ``F_B`` and watched by ``BoundInputMonitor``, and a contract test pins
that adding it moves neither digest.
"""
# status: stable

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "h2_runtime_input_manifest_v1"
# The same four build-independent sections, recorded by an execution that never
# reached build artifacts to hash. `coordinate_digest` is bit-identical to the
# full manifest's at the same coordinate — that is the point, and a test pins it.
# There is no `full_digest`, because the quantity it names does not exist yet.
COORDINATE_SCHEMA = "h2_runtime_input_coordinate_v1"
POLICY_PRESET_REL = "configs/presets/mamba_whole_graph_m.yaml"
IDENTITY_SEQUENCE = "MOT17-09-SDP"

# The § C3.2 item 10 run plan, in the lexicographic order it declares. This is a
# member set, not a schedule: the manifest binds all seven in both phases.
MEASUREMENT_SEQUENCES: tuple[str, ...] = (
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
)
DEFAULT_DATA_ROOT = "datasets/MOT17"

IDENTITY_FIXTURE_ROLE = "identity_fixture_input"
MEASUREMENT_FIXTURE_ROLE = "measurement_fixture_input"
PHASE_A_EVIDENCE_ROLE = "phase_a_evidence_input"
PHASE_A_EVIDENCE_SECTION = "phase_a_evidence"

# The exact member set of each digest. Stated once so the two can never drift
# apart by an edit that adds a section and forgets which digest it belongs to.
COORDINATE_SECTIONS: tuple[str, ...] = (
    "identity_fixture",
    "measurement_fixture",
    "runtime_assets",
    "third_party_runtime",
)
FULL_DIGEST_SECTIONS: tuple[str, ...] = (*COORDINATE_SECTIONS, "build_artifacts")

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


def _absolute_lexical(path: Path) -> Path:
    """Return the consumer-visible absolute path without resolving symlinks."""
    return Path(os.path.abspath(path))


def _symlink_chain(path: Path) -> list[dict[str, str]]:
    """Record every symlink component exactly as the consumer traverses it.

    Walks the full resolution chain, including intermediate links introduced by
    earlier targets (multi-hop). Matches the H0 verifier traversal: when a
    component is a symlink, its target is expanded and traversal continues from
    that target with loop detection. Recording only configured-path components
    would miss retargets of intermediate links such as::

        configured -> intermediate -> final
    """
    lexical = _absolute_lexical(path)
    chain: list[dict[str, str]] = []
    current = Path(lexical.anchor)
    pending = list(lexical.parts[1:])
    seen: set[Path] = set()
    while pending:
        current /= pending.pop(0)
        if not current.is_symlink():
            continue
        if current in seen:
            raise RuntimeInputError(f"symlink loop in runtime input: {path}")
        seen.add(current)
        target = os.readlink(current)
        chain.append({"path": current.as_posix(), "target": target})
        replacement = (
            Path(target) if Path(target).is_absolute() else current.parent / target
        )
        replacement = _absolute_lexical(replacement)
        current = Path(replacement.anchor)
        pending = list(replacement.parts[1:]) + pending
    return chain


def _path_binding(path: Path) -> tuple[Path, Path, list[dict[str, str]]]:
    configured = _absolute_lexical(path)
    resolved = configured.resolve(strict=True)
    return configured, resolved, _symlink_chain(configured)


def watch_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    """Expand consumer paths to the full set BoundInputMonitor must watch.

    Includes the configured lexical path, the final resolved target, and every
    multi-hop symlink component on the resolution chain. Membership discovery
    stays consumer-path-only; this expansion is only for continuous watching.
    """
    watched: set[Path] = set()
    for path in paths:
        configured, resolved, chain = _path_binding(path)
        watched.add(configured)
        watched.add(resolved)
        watched.update(Path(item["path"]) for item in chain)
    return tuple(sorted(watched, key=lambda item: item.as_posix()))


def _shown(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return resolved.relative_to(REPO_ROOT).as_posix()
    return resolved.as_posix()


def _record(*, path: Path, role: str, coordinate: str) -> dict[str, Any]:
    configured, resolved, symlink_chain = _path_binding(path)
    if not resolved.is_file():
        raise RuntimeInputError(f"runtime input is not a regular file: {resolved}")
    return {
        "configured_path": configured.as_posix(),
        "coordinate": coordinate,
        "length": resolved.stat().st_size,
        "resolved_path": resolved.as_posix(),
        "role": role,
        "sha256": sha256_file(resolved),
        "symlink_chain": symlink_chain,
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


def _files_below(root: Path, pattern: str = "*") -> tuple[Path, ...]:
    lexical_root = _absolute_lexical(root)
    lexical_root.resolve(strict=True)
    entries = tuple(lexical_root.rglob(pattern))
    directory_symlinks = sorted(
        (path for path in entries if path.is_symlink() and path.is_dir()),
        key=lambda item: item.as_posix(),
    )
    if directory_symlinks:
        raise RuntimeInputError(
            "runtime-input directory symlinks are unsupported: "
            + ", ".join(path.as_posix() for path in directory_symlinks)
        )
    return tuple(
        sorted(
            (path for path in entries if path.is_file()),
            key=lambda item: item.as_posix(),
        )
    )


def _fixture_paths(*, data_root: Path, sequence: str) -> tuple[Path, ...]:
    sequence_root = _absolute_lexical(data_root / "train" / sequence)
    if not (sequence_root / "seqinfo.ini").is_file():
        raise RuntimeInputError(f"{sequence}: seqinfo.ini is absent")
    if not (sequence_root / "img1").is_dir():
        raise RuntimeInputError(f"{sequence}: img1 is absent")
    return _files_below(sequence_root)


def _fixture_records(
    *, data_root: Path, sequence: str, role: str
) -> list[dict[str, Any]]:
    sequence_root = _absolute_lexical(data_root / "train" / sequence)
    return [
        _record(
            path=path,
            role=role,
            coordinate=f"{sequence}/{path.relative_to(sequence_root).as_posix()}",
        )
        for path in _fixture_paths(data_root=data_root, sequence=sequence)
    ]


def _identity_fixture_section(*, data_root: Path, sequence: str) -> dict[str, Any]:
    sequence_root = _absolute_lexical(data_root / "train" / sequence)
    section = _section(
        _fixture_records(
            data_root=data_root, sequence=sequence, role=IDENTITY_FIXTURE_ROLE
        )
    )
    section["configured_root"] = sequence_root.as_posix()
    section["sequence"] = sequence
    section["root"] = _shown(sequence_root)
    return section


def _measurement_fixture_section(
    *, data_root: Path, sequences: Iterable[str]
) -> dict[str, Any]:
    """One section over all seven sequences: one digest, one axis member."""
    records: list[dict[str, Any]] = []
    declared: list[dict[str, Any]] = []
    for sequence in sequences:
        sequence_root = _absolute_lexical(data_root / "train" / sequence)
        per_sequence = _fixture_records(
            data_root=data_root, sequence=sequence, role=MEASUREMENT_FIXTURE_ROLE
        )
        records.extend(per_sequence)
        declared.append(
            {
                "configured_root": sequence_root.as_posix(),
                "file_count": len(per_sequence),
                "root": _shown(sequence_root),
                "sequence": sequence,
            }
        )
    section = _section(records)
    section["sequences"] = declared
    return section


def _phase_a_evidence_section(evidence_root: Path) -> dict[str, Any]:
    """The § C3.8 `F_B`-only section. In neither digest; bound and watched."""
    configured = _absolute_lexical(evidence_root)
    records = [
        _record(
            path=path,
            role=PHASE_A_EVIDENCE_ROLE,
            coordinate=path.relative_to(configured).as_posix(),
        )
        for path in _files_below(configured)
    ]
    section = _section(records)
    section["configured_root"] = configured.as_posix()
    section["root"] = _shown(configured)
    return section


def _runtime_asset_paths(preset_path: Path) -> tuple[tuple[str, str, Path], ...]:
    payload = yaml.safe_load(preset_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise RuntimeInputError(f"{preset_path}: preset is not a mapping")
    paths: list[tuple[str, str, Path]] = []
    for field in RUNTIME_ASSET_FIELDS:
        configured = payload.get(field)
        if not isinstance(configured, str) or not configured:
            raise RuntimeInputError(
                f"{preset_path}: required runtime asset {field} absent"
            )
        path = Path(configured)
        if not path.is_absolute():
            path = REPO_ROOT / path
        paths.append((field, configured, _absolute_lexical(path)))
    return tuple(paths)


def _runtime_assets_section(preset_path: Path) -> dict[str, Any]:
    records = [
        _record(path=path, role=field, coordinate=configured)
        for field, configured, path in _runtime_asset_paths(preset_path)
    ]
    return _section(records)


def _third_party_paths() -> tuple[Path, ...]:
    root = _absolute_lexical(REPO_ROOT / "third_party" / "TrackEval" / "trackeval")
    return _files_below(root, "*.py")


def _third_party_section() -> dict[str, Any]:
    root = _absolute_lexical(REPO_ROOT / "third_party" / "TrackEval" / "trackeval")
    records = [
        _record(
            path=path,
            role="third_party_runtime_component",
            coordinate=path.relative_to(REPO_ROOT).as_posix(),
        )
        for path in _third_party_paths()
    ]
    section = _section(records)
    section["configured_root"] = root.as_posix()
    return section


def _build_artifact_paths(build_dir: Path) -> tuple[Path, Path]:
    configured = _absolute_lexical(build_dir)
    configured.resolve(strict=True)
    extension = sorted(configured.glob("saccade_tracking_ext*.so"))
    plugins = sorted(configured.glob("libsaccade_scan_plugin.so"))
    if len(extension) != 1:
        raise RuntimeInputError(
            f"{configured}: expected one saccade_tracking_ext*.so, "
            f"found {len(extension)}"
        )
    if len(plugins) != 1:
        raise RuntimeInputError(
            f"{configured}: expected one libsaccade_scan_plugin.so, "
            f"found {len(plugins)}"
        )
    return extension[0], plugins[0]


def _build_artifacts_section(build_dir: Path) -> dict[str, Any]:
    configured = _absolute_lexical(build_dir)
    extension, plugin = _build_artifact_paths(configured)
    records = [
        _record(
            path=extension,
            role="tracking_extension",
            coordinate=extension.name,
        ),
        _record(
            path=plugin,
            role="tensorrt_scan_plugin",
            coordinate=plugin.name,
        ),
    ]
    section = _section(records)
    section["build_dir"] = configured.as_posix()
    return section


def _checked_sequences(
    identity_sequence: str, measurement_sequences: Iterable[str]
) -> tuple[str, ...]:
    if identity_sequence != IDENTITY_SEQUENCE:
        raise RuntimeInputError(
            f"identity fixture must be {IDENTITY_SEQUENCE}, got {identity_sequence}"
        )
    declared = tuple(measurement_sequences)
    if declared != MEASUREMENT_SEQUENCES:
        raise RuntimeInputError(
            f"measurement fixtures must be {list(MEASUREMENT_SEQUENCES)}, "
            f"got {list(declared)}"
        )
    return declared


def discover_bound_paths(
    *,
    build_dir: Path,
    data_root: Path | None = None,
    identity_sequence: str = IDENTITY_SEQUENCE,
    measurement_sequences: Iterable[str] = MEASUREMENT_SEQUENCES,
    phase_a_evidence_root: Path | None = None,
) -> tuple[Path, ...]:
    """Discover lexical consumer paths before any content digest is computed."""
    sequences = _checked_sequences(identity_sequence, measurement_sequences)
    root = _absolute_lexical(data_root or (REPO_ROOT / DEFAULT_DATA_ROOT))
    paths = {
        *_fixture_paths(data_root=root, sequence=identity_sequence),
        *(
            path
            for sequence in sequences
            for path in _fixture_paths(data_root=root, sequence=sequence)
        ),
        *(
            path
            for _field, _configured, path in _runtime_asset_paths(
                REPO_ROOT / POLICY_PRESET_REL
            )
        ),
        *_third_party_paths(),
        *_build_artifact_paths(build_dir),
    }
    if phase_a_evidence_root is not None:
        paths.update(_files_below(_absolute_lexical(phase_a_evidence_root)))
    return tuple(sorted(paths, key=lambda item: item.as_posix()))


def build_manifest(
    *,
    build_dir: Path | None,
    data_root: Path | None = None,
    identity_sequence: str = IDENTITY_SEQUENCE,
    measurement_sequences: Iterable[str] = MEASUREMENT_SEQUENCES,
    phase_a_evidence_root: Path | None = None,
) -> dict[str, Any]:
    """Hash every declared fixture, asset, third-party component and build output.

    `phase_a_evidence_root` is supplied only when an `F_B` is being constructed;
    it adds the § C3.8 section and moves neither digest.

    `build_dir=None` records the **coordinate form**: the same four sections, the
    same `coordinate_digest`, no build artifacts and no `full_digest`. It exists
    for an execution that stopped at or before `build` and so has no artifacts to
    hash — an execution whose bound fixtures, assets and third-party components
    were nonetheless real and are recorded rather than omitted. Passing a build
    directory is not optional in the other direction: an execution that did build
    must hash what it built.
    """
    sequences = _checked_sequences(identity_sequence, measurement_sequences)
    root = _absolute_lexical(data_root or (REPO_ROOT / DEFAULT_DATA_ROOT))
    sections = {
        "identity_fixture": _identity_fixture_section(
            data_root=root, sequence=identity_sequence
        ),
        "measurement_fixture": _measurement_fixture_section(
            data_root=root, sequences=sequences
        ),
        "runtime_assets": _runtime_assets_section(REPO_ROOT / POLICY_PRESET_REL),
        "third_party_runtime": _third_party_section(),
    }
    if build_dir is not None:
        sections["build_artifacts"] = _build_artifacts_section(build_dir)
    if phase_a_evidence_root is not None:
        sections[PHASE_A_EVIDENCE_SECTION] = _phase_a_evidence_section(
            phase_a_evidence_root
        )
    coordinate_digest = digest(
        {name: sections[name]["digest"] for name in COORDINATE_SECTIONS}
    )
    manifest = {
        **sections,
        "coordinate_digest": coordinate_digest,
        "data_root": root.as_posix(),
        "policy_preset": POLICY_PRESET_REL,
        "schema": SCHEMA if build_dir is not None else COORDINATE_SCHEMA,
    }
    if build_dir is not None:
        manifest["full_digest"] = digest(
            {
                "coordinate_digest": coordinate_digest,
                "build_artifacts": sections["build_artifacts"]["digest"],
            }
        )
    return manifest


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_measurement_sequences(section: Mapping[str, Any]) -> None:
    """The declared seven, in order, each actually carrying files."""
    declared = section.get("sequences")
    if not isinstance(declared, list):
        raise RuntimeInputError("measurement fixture sequences are absent")
    names = tuple(
        item.get("sequence") if isinstance(item, Mapping) else None for item in declared
    )
    if names != MEASUREMENT_SEQUENCES:
        raise RuntimeInputError(
            f"measurement fixture sequences must be {list(MEASUREMENT_SEQUENCES)}, "
            f"got {list(names)}"
        )
    observed: dict[str, int] = {name: 0 for name in MEASUREMENT_SEQUENCES}
    for record in section["files"]:
        sequence = str(record.get("coordinate", "")).split("/", 1)[0]
        if sequence not in observed:
            raise RuntimeInputError(
                f"measurement fixture file outside the declared sequences: "
                f"{record.get('coordinate')}"
            )
        observed[sequence] += 1
    for item in declared:
        sequence = item["sequence"]
        if item.get("file_count") != observed[sequence]:
            raise RuntimeInputError(
                f"measurement fixture file_count mismatch for {sequence}: "
                f"declared {item.get('file_count')}, observed {observed[sequence]}"
            )
        if not observed[sequence]:
            raise RuntimeInputError(f"measurement fixture {sequence} is empty")
        root = Path(str(item.get("configured_root", "")))
        if not root.is_absolute():
            raise RuntimeInputError(
                f"measurement fixture {sequence} configured_root is not absolute"
            )


def validate_manifest(
    payload: Mapping[str, Any], *, verify_files: bool = False
) -> dict[str, Any]:
    schema = payload.get("schema")
    if schema not in (SCHEMA, COORDINATE_SCHEMA):
        raise RuntimeInputError(f"not a {SCHEMA} or {COORDINATE_SCHEMA} payload")
    binds_build = schema == SCHEMA
    if payload.get("policy_preset") != POLICY_PRESET_REL:
        raise RuntimeInputError("runtime-input manifest preset mismatch")
    data_root = Path(str(payload.get("data_root", "")))
    if not data_root.is_absolute():
        raise RuntimeInputError("runtime-input manifest data_root is not absolute")
    # The coordinate form is a different claim, not a shorter one: it must not
    # carry the two members that assert an execution reached its build outputs.
    if not binds_build and ("build_artifacts" in payload or "full_digest" in payload):
        raise RuntimeInputError(
            f"a {COORDINATE_SCHEMA} payload binds no build artifacts, so it may "
            "carry neither build_artifacts nor full_digest"
        )
    present = list(FULL_DIGEST_SECTIONS if binds_build else COORDINATE_SECTIONS)
    if PHASE_A_EVIDENCE_SECTION in payload:
        present.append(PHASE_A_EVIDENCE_SECTION)
    sections: dict[str, Mapping[str, Any]] = {}
    for name in present:
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
            configured = Path(str(record.get("configured_path", "")))
            resolved = Path(str(record.get("resolved_path", "")))
            symlink_chain = record.get("symlink_chain")
            if not configured.is_absolute() or not resolved.is_absolute():
                raise RuntimeInputError(f"{name}: non-absolute path binding for {key}")
            if not isinstance(symlink_chain, list) or any(
                not isinstance(item, Mapping)
                or not Path(str(item.get("path", ""))).is_absolute()
                or not isinstance(item.get("target"), str)
                for item in symlink_chain
            ):
                raise RuntimeInputError(f"{name}: malformed symlink chain for {key}")
            if verify_files:
                try:
                    current_configured, current_resolved, current_chain = _path_binding(
                        configured
                    )
                except (OSError, RuntimeError) as exc:
                    raise RuntimeInputError(
                        f"{name}: bound path unusable for {key}: {exc}"
                    ) from exc
                if (
                    current_configured != configured
                    or current_resolved != resolved
                    or current_chain != symlink_chain
                ):
                    raise RuntimeInputError(
                        f"{name}: path binding moved for {key}: "
                        f"{configured} -> {current_resolved}"
                    )
                if not current_resolved.is_file():
                    raise RuntimeInputError(
                        f"{name}: bound file absent: {current_resolved}"
                    )
                if current_resolved.stat().st_size != record["length"]:
                    raise RuntimeInputError(f"{name}: length moved: {current_resolved}")
                if sha256_file(current_resolved) != record["sha256"]:
                    raise RuntimeInputError(
                        f"{name}: content moved: {current_resolved}"
                    )
        expected = digest([_digest_projection(item) for item in files])
        if section.get("digest") != expected:
            raise RuntimeInputError(f"{name}: section digest mismatch")
        if section.get("file_count") != len(files):
            raise RuntimeInputError(f"{name}: file_count mismatch")
        sections[name] = section
    if sections["identity_fixture"].get("sequence") != IDENTITY_SEQUENCE:
        raise RuntimeInputError("identity fixture sequence mismatch")
    _validate_measurement_sequences(sections["measurement_fixture"])
    coordinate = digest(
        {name: sections[name]["digest"] for name in COORDINATE_SECTIONS}
    )
    if payload.get("coordinate_digest") != coordinate:
        raise RuntimeInputError("runtime-input coordinate digest mismatch")
    if binds_build:
        full = digest(
            {
                "coordinate_digest": coordinate,
                "build_artifacts": sections["build_artifacts"]["digest"],
            }
        )
        if payload.get("full_digest") != full:
            raise RuntimeInputError("runtime-input full digest mismatch")
    # Membership reconciliation is build-rooted — `discover_bound_paths` walks a
    # build directory — so the coordinate form verifies its own files without it.
    # It claims no build, so there is no membership across a build to reconcile.
    if verify_files and binds_build:
        build_dir = Path(str(sections["build_artifacts"].get("build_dir", "")))
        if not build_dir.is_absolute():
            raise RuntimeInputError(
                "runtime-input manifest build_artifacts.build_dir is not absolute"
            )
        evidence = sections.get(PHASE_A_EVIDENCE_SECTION)
        evidence_root = (
            Path(str(evidence.get("configured_root", ""))) if evidence else None
        )
        if evidence is not None and not (evidence_root and evidence_root.is_absolute()):
            raise RuntimeInputError(
                "runtime-input manifest phase_a_evidence.configured_root is not absolute"
            )
        current_paths = discover_bound_paths(
            build_dir=build_dir,
            data_root=data_root,
            phase_a_evidence_root=evidence_root,
        )
        recorded_paths = _consumer_paths_from_sections(sections)
        if current_paths != recorded_paths:
            added = sorted(set(current_paths) - set(recorded_paths))
            removed = sorted(set(recorded_paths) - set(current_paths))
            raise RuntimeInputError(
                "runtime-input membership moved: "
                f"added={[path.as_posix() for path in added]}, "
                f"removed={[path.as_posix() for path in removed]}"
            )
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


def _bound_section_names(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Every section a consumer opens — digest membership is a separate question.

    `phase_a_evidence` is in neither digest (§ C3.8) and is nonetheless bound and
    watched (§ C3.2 item 7), which is exactly why membership is enumerated here
    rather than reused from `FULL_DIGEST_SECTIONS`.
    """
    if PHASE_A_EVIDENCE_SECTION in payload:
        return (*FULL_DIGEST_SECTIONS, PHASE_A_EVIDENCE_SECTION)
    return FULL_DIGEST_SECTIONS


def _consumer_paths_from_sections(
    sections: Mapping[str, Mapping[str, Any]],
) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for name in _bound_section_names(sections):
        for record in sections[name]["files"]:
            paths.add(Path(record["configured_path"]))
    return tuple(sorted(paths, key=lambda item: item.as_posix()))


def consumer_paths(payload: Mapping[str, Any]) -> tuple[Path, ...]:
    validated = validate_manifest(payload)
    sections = {name: validated[name] for name in _bound_section_names(validated)}
    return _consumer_paths_from_sections(sections)


def bound_paths(payload: Mapping[str, Any]) -> tuple[Path, ...]:
    validated = validate_manifest(payload)
    paths: set[Path] = set()
    for name in _bound_section_names(validated):
        for record in validated[name]["files"]:
            paths.add(Path(record["configured_path"]))
            paths.add(Path(record["resolved_path"]))
            paths.update(Path(item["path"]) for item in record["symlink_chain"])
    return tuple(sorted(paths, key=lambda item: item.as_posix()))


def publication_axis(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The published `runtime_inputs` axis: the four coordinate sections only.

    `build_artifacts` is excluded because the coordinate spans builds, and
    `phase_a_evidence` because it does not exist until after the seal that
    freezes this axis (§ C3.8).
    """
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
            "sequences": [
                {
                    "file_count": item["file_count"],
                    "sequence": item["sequence"],
                }
                for item in validated["measurement_fixture"]["sequences"]
            ],
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
    parser.add_argument(
        "--phase-a-evidence-root",
        type=Path,
        default=None,
        help=(
            "bind the Phase-A evidence root as an F_B-only section (§ C3.8); "
            "it moves neither digest"
        ),
    )
    args = parser.parse_args(argv)
    try:
        payload = build_manifest(
            build_dir=args.build_dir,
            data_root=args.data_root,
            phase_a_evidence_root=args.phase_a_evidence_root,
        )
    except (RuntimeInputError, OSError) as exc:
        print(f"runtime-input manifest failed: {exc}", file=sys.stderr)
        return 1
    args.emit.parent.mkdir(parents=True, exist_ok=True)
    args.emit.write_bytes(canonical_json_bytes(payload) + b"\n")
    print(f"wrote {args.emit}")
    print(f"  coordinate_digest {payload['coordinate_digest']}")
    print(f"  full_digest       {payload['full_digest']}")
    if PHASE_A_EVIDENCE_SECTION in payload:
        section = payload[PHASE_A_EVIDENCE_SECTION]
        print(
            f"  phase_a_evidence  {section['digest']} "
            f"({section['file_count']} files, in neither digest)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
