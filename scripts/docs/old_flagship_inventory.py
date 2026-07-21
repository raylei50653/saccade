"""Validate and summarize the old-flagship per-study lifecycle inventory.

This is deliberately a recovery-slice tool, not a generic disposal mechanism.
It connects the frozen ``old-flagship`` resolved-file set to the terminal or
live owner of each underlying study.  A shared-support file is explicit, but
cannot supply terminal coverage or authorize disposal.
"""
# status: diagnostic

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

try:  # Supports both ``python -m`` and direct script execution.
    from .migration_manifest import (
        MigrationManifest,
        MigrationManifestError,
        parse_migration_manifest,
    )
    from .strict_yaml import StrictYamlError, strict_safe_load
    from .terminal_slot_schema import (
        TerminalSlotValidationError,
        WorkedExampleValidationError,
        extract_yaml_slots_from_markdown,
        validate_terminal_slot,
    )
except ImportError:  # pragma: no cover - exercised by direct CLI use
    from migration_manifest import (
        MigrationManifest,
        MigrationManifestError,
        parse_migration_manifest,
    )
    from strict_yaml import StrictYamlError, strict_safe_load
    from terminal_slot_schema import (
        TerminalSlotValidationError,
        WorkedExampleValidationError,
        extract_yaml_slots_from_markdown,
        validate_terminal_slot,
    )


OLD_FLAGSHIP = "old-flagship"
INVENTORY_SCHEMA_VERSION = 1
FILE_KINDS = frozenset({"process", "shared_support"})

_TOP_LEVEL_FIELDS = frozenset({"schema_version", "cluster", "studies", "file_roles"})
_STUDY_FIELDS = frozenset({"terminal_ref", "live_owner"})
_PROCESS_FILE_FIELDS = frozenset({"path", "kind", "study_id"})
_SHARED_SUPPORT_FILE_FIELDS = frozenset({"path", "kind"})


class OldFlagshipInventoryError(ValueError):
    """A fail-closed inventory validation error with a stable error class."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


@dataclass(frozen=True)
class StudyInventory:
    """One study's terminal or live owner, never both.

    The inventory routes to the current source of truth; it deliberately does
    not duplicate live scheduling state such as ``proposed`` or ``parked``.
    """

    study_id: str
    terminal_ref: str | None
    live_owner: str | None

    @property
    def is_terminal_backed(self) -> bool:
        return self.terminal_ref is not None


@dataclass(frozen=True)
class FileRole:
    """The only two allowed roles for an old-flagship resolved file."""

    path: str
    kind: str
    study_id: str | None


@dataclass(frozen=True)
class OldFlagshipInventory:
    """Validated roles plus the lifecycle view derived from them."""

    source: Path
    cluster: str
    migration_state: str
    studies: Mapping[str, StudyInventory]
    file_roles: Mapping[str, FileRole]
    terminal_backed_files: frozenset[str]
    live_owned_files: frozenset[str]
    shared_support_files: frozenset[str]
    unmapped_files: frozenset[str]
    classification: str
    disposal_authorized: bool

    def summary(self) -> Mapping[str, object]:
        """Return the deterministic machine-facing lifecycle projection."""

        return {
            "cluster": self.cluster,
            "migration_state": self.migration_state,
            "terminal_backed_files": sorted(self.terminal_backed_files),
            "live_owned_files": sorted(self.live_owned_files),
            "shared_support_files": sorted(self.shared_support_files),
            "unmapped_files": sorted(self.unmapped_files),
            "classification": self.classification,
            "disposal_authorized": self.disposal_authorized,
        }


def _fail(error_class: str, message: str) -> None:
    raise OldFlagshipInventoryError(error_class, message)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail("invalid_field_type", f"{field} must be a mapping with string keys")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("invalid_field_type", f"{field} must be a non-empty string")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail("invalid_field_type", f"{field} must be a list")
    return value


def _repo_relative_path(value: object, *, field: str) -> str:
    path_text = _string(value, field=field)
    path = PurePosixPath(path_text)
    if path.is_absolute() or ".." in path.parts or path_text != path.as_posix():
        _fail("invalid_path", f"{field} must be a normalized repo-relative path")
    return path_text


def _load_inventory_document(path: Path) -> Mapping[str, object]:
    try:
        content: Any = strict_safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        _fail("inventory_unreadable", f"cannot read {path}: {error}")
    except StrictYamlError as error:
        _fail(error.error_class, f"cannot parse {path}: {error}")
    return _mapping(content, field="inventory")


def _require_exact_fields(
    value: Mapping[str, object], *, field: str, permitted: frozenset[str]
) -> None:
    unknown = sorted(set(value) - permitted)
    if unknown:
        _fail("unknown_field", f"{field} has unknown field {unknown[0]!r}")


def _read_terminal_owner(
    terminal_ref: str,
    *,
    study_id: str,
    repo_root: Path,
) -> None:
    """Require one schema-valid slot in the cited per-study owner document."""

    owner_path = repo_root / terminal_ref
    if not owner_path.is_file():
        _fail(
            "terminal_ref_missing",
            f"terminal_ref {terminal_ref!r} for {study_id!r} does not exist",
        )
    try:
        slots = extract_yaml_slots_from_markdown(owner_path)
    except WorkedExampleValidationError as error:
        _fail(error.error_class, f"terminal_ref {terminal_ref!r}: {error}")

    matching_slots = [slot for slot in slots if slot.get("study_id") == study_id]
    if len(matching_slots) != 1:
        _fail(
            "terminal_ref_slot_count",
            f"terminal_ref {terminal_ref!r} for {study_id!r} must contain exactly one matching terminal slot",
        )
    try:
        validate_terminal_slot(matching_slots[0])
    except TerminalSlotValidationError as error:
        _fail(
            error.error_class,
            f"terminal_ref {terminal_ref!r} for {study_id!r} is invalid: {error}",
        )


def _parse_studies(value: object, *, repo_root: Path) -> Mapping[str, StudyInventory]:
    studies_data = _mapping(value, field="studies")
    if not studies_data:
        _fail("invalid_value", "studies must not be empty")

    studies: dict[str, StudyInventory] = {}
    for raw_study_id, raw_study in studies_data.items():
        study_id = _string(raw_study_id, field="study id")
        study = _mapping(raw_study, field=f"studies.{study_id}")
        _require_exact_fields(
            study, field=f"studies.{study_id}", permitted=_STUDY_FIELDS
        )
        has_terminal_ref = "terminal_ref" in study
        has_live_owner = "live_owner" in study
        if has_terminal_ref == has_live_owner:
            _fail(
                "invalid_owner_form",
                f"study {study_id!r} must declare exactly one of terminal_ref or live_owner",
            )

        if has_terminal_ref:
            terminal_ref = _repo_relative_path(
                study["terminal_ref"], field=f"studies.{study_id}.terminal_ref"
            )
            _read_terminal_owner(terminal_ref, study_id=study_id, repo_root=repo_root)
            live_owner = None
        else:
            live_owner = _repo_relative_path(
                study["live_owner"], field=f"studies.{study_id}.live_owner"
            )
            if not (repo_root / live_owner).is_file():
                _fail(
                    "live_owner_missing",
                    f"live_owner {live_owner!r} for {study_id!r} does not exist",
                )
            terminal_ref = None

        studies[study_id] = StudyInventory(
            study_id=study_id,
            terminal_ref=terminal_ref,
            live_owner=live_owner,
        )
    return MappingProxyType(studies)


def _parse_file_roles(
    value: object, *, studies: Mapping[str, StudyInventory]
) -> Mapping[str, FileRole]:
    raw_roles = _sequence(value, field="file_roles")
    if not raw_roles:
        _fail("invalid_value", "file_roles must not be empty")

    file_roles: dict[str, FileRole] = {}
    for index, raw_role in enumerate(raw_roles):
        field = f"file_roles[{index}]"
        role = _mapping(raw_role, field=field)
        if "kind" not in role:
            _fail("missing_required_field", f"{field} is missing 'kind'")
        kind = _string(role["kind"], field=f"{field}.kind")
        if kind not in FILE_KINDS:
            _fail("unknown_enum", f"{field}.kind has unknown value {kind!r}")

        permitted = (
            _PROCESS_FILE_FIELDS if kind == "process" else _SHARED_SUPPORT_FILE_FIELDS
        )
        _require_exact_fields(role, field=field, permitted=permitted)
        if "path" not in role:
            _fail("missing_required_field", f"{field} is missing 'path'")
        path = _repo_relative_path(role["path"], field=f"{field}.path")
        if path in file_roles:
            _fail("duplicate_file_role", f"file_roles repeats {path!r}")

        study_id: str | None = None
        if kind == "process":
            if "study_id" not in role:
                _fail("missing_required_field", f"{field} is missing 'study_id'")
            study_id = _string(role["study_id"], field=f"{field}.study_id")
            if study_id not in studies:
                _fail(
                    "unknown_study_id",
                    f"{field}.study_id {study_id!r} is not declared in studies",
                )
        file_roles[path] = FileRole(path=path, kind=kind, study_id=study_id)
    return MappingProxyType(file_roles)


def parse_old_flagship_inventory(
    path: str | Path,
    *,
    manifest: MigrationManifest,
    repo_root: str | Path,
) -> OldFlagshipInventory:
    """Validate old-flagship roles and derive its non-disposal lifecycle view."""

    source = Path(path).resolve()
    root = Path(repo_root).resolve()
    document = _load_inventory_document(source)
    _require_exact_fields(document, field="inventory", permitted=_TOP_LEVEL_FIELDS)
    for required in _TOP_LEVEL_FIELDS:
        if required not in document:
            _fail("missing_required_field", f"inventory is missing {required!r}")
    if document["schema_version"] != INVENTORY_SCHEMA_VERSION:
        _fail(
            "unsupported_schema_version",
            f"schema_version must be {INVENTORY_SCHEMA_VERSION}",
        )
    cluster = _string(document["cluster"], field="cluster")
    if cluster != OLD_FLAGSHIP:
        _fail("unsupported_cluster", f"inventory only supports {OLD_FLAGSHIP!r}")
    manifest_cluster = manifest.clusters.get(cluster)
    if manifest_cluster is None:
        _fail("missing_manifest_cluster", f"manifest is missing {cluster!r}")
    if manifest_cluster.migration_state != "quarantined":
        _fail(
            "old_flagship_not_quarantined",
            "old-flagship must remain migration_state='quarantined' in this slice",
        )

    studies = _parse_studies(document["studies"], repo_root=root)
    file_roles = _parse_file_roles(document["file_roles"], studies=studies)
    resolved_files = frozenset(manifest_cluster.resolved_files)
    role_paths = frozenset(file_roles)
    extraneous_files = sorted(role_paths - resolved_files)
    if extraneous_files:
        _fail(
            "file_role_not_resolved",
            f"file role {extraneous_files[0]!r} is not in old-flagship resolved_files",
        )
    unmapped_files = resolved_files - role_paths
    if unmapped_files:
        _fail(
            "unmapped_resolved_file",
            f"old-flagship resolved_file {sorted(unmapped_files)[0]!r} has no role",
        )

    referenced_studies = {
        role.study_id for role in file_roles.values() if role.study_id is not None
    }
    unreferenced_studies = sorted(set(studies) - referenced_studies)
    if unreferenced_studies:
        _fail(
            "study_without_process_file",
            f"study {unreferenced_studies[0]!r} has no process file",
        )

    terminal_backed_files = frozenset(
        role.path
        for role in file_roles.values()
        if role.kind == "process"
        and role.study_id is not None
        and studies[role.study_id].is_terminal_backed
    )
    live_owned_files = (
        frozenset(role.path for role in file_roles.values() if role.kind == "process")
        - terminal_backed_files
    )
    shared_support_files = frozenset(
        role.path for role in file_roles.values() if role.kind == "shared_support"
    )

    # A requires every resolved file to be terminal-backed.  In particular a
    # shared-support entry is intentionally not a terminal and cannot carry a
    # cluster over the disposal boundary.
    classification = "A" if terminal_backed_files == resolved_files else "B"
    disposal_authorized = (
        classification == "A" and manifest_cluster.migration_state == "disposable"
    )
    if classification != "B" or disposal_authorized:
        _fail(
            "old_flagship_disposal_forbidden",
            "old-flagship must remain classification B with disposal_authorized=false in this slice",
        )

    return OldFlagshipInventory(
        source=source,
        cluster=cluster,
        migration_state=manifest_cluster.migration_state,
        studies=studies,
        file_roles=file_roles,
        terminal_backed_files=terminal_backed_files,
        live_owned_files=live_owned_files,
        shared_support_files=shared_support_files,
        unmapped_files=frozenset(),
        classification=classification,
        disposal_authorized=disposal_authorized,
    )


def _default_path(relative: str) -> Path:
    return Path.cwd() / relative


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate old-flagship per-study terminal inventory"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_default_path("docs/ownership/doc_migration_manifest.yaml"),
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=_default_path("docs/ownership/old_flagship_per_study_inventory.yaml"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true", help="print derived status")
    arguments = parser.parse_args()
    try:
        manifest = parse_migration_manifest(
            arguments.manifest, repo_root=arguments.repo_root
        )
        inventory = parse_old_flagship_inventory(
            arguments.inventory, manifest=manifest, repo_root=arguments.repo_root
        )
    except (MigrationManifestError, OldFlagshipInventoryError) as error:
        error_class = getattr(error, "error_class", "validation_error")
        print(f"old-flagship inventory validation failed [{error_class}]: {error}")
        return 1

    if arguments.json:
        print(json.dumps(inventory.summary(), indent=2, sort_keys=True))
    else:
        print(
            "old-flagship inventory green: "
            f"{len(inventory.terminal_backed_files)} terminal-backed, "
            f"{len(inventory.live_owned_files)} live-owned, "
            f"{len(inventory.shared_support_files)} shared-support, "
            f"classification {inventory.classification}, "
            f"disposal_authorized={str(inventory.disposal_authorized).lower()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
