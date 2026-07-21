"""Parser for the migration manifest's frozen, resolved-file sets.

Only ``resolved_files`` determines a cluster's set.  ``process_globs`` and
``snapshot.frozen_at_commit`` remain parsed provenance metadata, but this
module never expands a glob or invokes Git to make normal execution work.
"""
# status: stable

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

try:  # Supports both ``python -m`` and direct script execution.
    from .strict_yaml import StrictYamlError, strict_safe_load
except ImportError:  # pragma: no cover - exercised by direct CLI use
    from strict_yaml import StrictYamlError, strict_safe_load


MIGRATION_STATES = frozenset({"active", "quarantined", "disposable"})
_TOP_LEVEL_FIELDS = frozenset({"refuted_claims_ref", "snapshot", "clusters"})
_CLUSTER_FIELDS = frozenset(
    {
        "migration_state",
        "terminal_owner",
        "navigation_ref",
        "premise_refs",
        "process_globs",
        "resolved_files",
    }
)
_REQUIRED_CLUSTER_FIELDS = frozenset(
    {
        "migration_state",
        "terminal_owner",
        "premise_refs",
        "process_globs",
        "resolved_files",
    }
)


class MigrationManifestError(ValueError):
    """A fail-closed manifest parse error with a concise error class."""

    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


@dataclass(frozen=True)
class MigrationCluster:
    name: str
    migration_state: str
    resolved_files: tuple[str, ...]
    process_globs: tuple[str, ...]
    premise_refs: tuple[str, ...]
    terminal_owner: None
    navigation_ref: str | None


@dataclass(frozen=True)
class MigrationManifest:
    source: Path
    frozen_at_commit: str | None
    resolved_at: str | None
    clusters: Mapping[str, MigrationCluster]

    @property
    def resolved_file_sets(self) -> Mapping[str, frozenset[str]]:
        """The authoritative resolved-file set for every cluster."""

        return MappingProxyType(
            {
                name: frozenset(cluster.resolved_files)
                for name, cluster in self.clusters.items()
            }
        )

    @property
    def frozen_sets(self) -> Mapping[str, frozenset[str]]:
        """Resolved-file sets that are frozen by ``migration_state=quarantined``."""

        return MappingProxyType(
            {
                name: frozenset(cluster.resolved_files)
                for name, cluster in self.clusters.items()
                if cluster.migration_state == "quarantined"
            }
        )

    @property
    def frozen_files(self) -> frozenset[str]:
        return frozenset(path for paths in self.frozen_sets.values() for path in paths)


def _fail(error_class: str, message: str) -> None:
    raise MigrationManifestError(error_class, message)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail("invalid_field_type", f"{field} must be a mapping with string keys")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("invalid_field_type", f"{field} must be a non-empty string")
    return value


def _provenance_text(value: object, *, field: str) -> str:
    """Keep YAML's date coercion from changing the manifest contract."""

    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return _string(value, field=field)


def _strings(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail("invalid_field_type", f"{field} must be a list of strings")
    result = tuple(_string(item, field=f"{field}[]") for item in value)
    return result


def _validate_repo_relative_path(value: str, *, field: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value != path.as_posix():
        _fail(
            "invalid_resolved_file_path",
            f"{field} must be a normalized repo-relative path",
        )
    return value


def _load_document(path: Path) -> Mapping[str, object]:
    try:
        content: Any = strict_safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise MigrationManifestError(
            "manifest_unreadable", f"cannot read {path}: {error}"
        ) from error
    except StrictYamlError as error:
        raise MigrationManifestError(
            error.error_class, f"cannot parse {path}: {error}"
        ) from error
    return _mapping(content, field="manifest")


def parse_migration_manifest(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> MigrationManifest:
    """Parse and validate manifest frozen sets without using Git or globs."""

    source = Path(path).resolve()
    root = Path.cwd().resolve() if repo_root is None else Path(repo_root).resolve()
    document = _load_document(source)

    unknown_top_level = sorted(set(document) - _TOP_LEVEL_FIELDS)
    if unknown_top_level:
        _fail("unknown_field", f"manifest has unknown field {unknown_top_level[0]!r}")
    for required in ("snapshot", "clusters"):
        if required not in document:
            _fail("missing_required_field", f"manifest is missing {required!r}")

    snapshot = _mapping(document["snapshot"], field="snapshot")
    unknown_snapshot = sorted(set(snapshot) - {"frozen_at_commit", "resolved_at"})
    if unknown_snapshot:
        _fail("unknown_field", f"snapshot has unknown field {unknown_snapshot[0]!r}")
    for required in ("frozen_at_commit", "resolved_at"):
        if required not in snapshot:
            _fail("missing_required_field", f"snapshot is missing {required!r}")
    # Provenance only: validating it is text is intentionally the final action.
    # Do not resolve it with Git or use it to reconstruct resolved_files.
    frozen_at_commit = _string(
        snapshot["frozen_at_commit"], field="snapshot.frozen_at_commit"
    )
    resolved_at = _provenance_text(
        snapshot["resolved_at"], field="snapshot.resolved_at"
    )

    clusters_data = _mapping(document["clusters"], field="clusters")
    if not clusters_data:
        _fail("invalid_value", "clusters must not be empty")

    clusters: dict[str, MigrationCluster] = {}
    file_owners: dict[str, str] = {}
    for name, raw_cluster in clusters_data.items():
        cluster_name = _string(name, field="cluster name")
        cluster = _mapping(raw_cluster, field=f"clusters.{cluster_name}")
        unknown = sorted(set(cluster) - _CLUSTER_FIELDS)
        if unknown:
            _fail(
                "unknown_field",
                f"clusters.{cluster_name} has unknown field {unknown[0]!r}",
            )
        missing = sorted(_REQUIRED_CLUSTER_FIELDS - set(cluster))
        if missing:
            _fail(
                "missing_required_field",
                f"clusters.{cluster_name} is missing {missing[0]!r}",
            )

        state = _string(
            cluster["migration_state"], field=f"clusters.{cluster_name}.migration_state"
        )
        if state not in MIGRATION_STATES:
            _fail(
                "unknown_enum",
                f"clusters.{cluster_name} has unknown migration_state {state!r}",
            )
        if cluster["terminal_owner"] is not None:
            _fail(
                "terminal_owner_must_be_null",
                f"clusters.{cluster_name}.terminal_owner must be null at cluster scope",
            )

        premise_refs = _strings(
            cluster["premise_refs"], field=f"clusters.{cluster_name}.premise_refs"
        )
        # This is deliberately only provenance; never expand it to create a set.
        process_globs = _strings(
            cluster["process_globs"], field=f"clusters.{cluster_name}.process_globs"
        )
        raw_resolved_files = _strings(
            cluster["resolved_files"], field=f"clusters.{cluster_name}.resolved_files"
        )
        if not raw_resolved_files:
            _fail(
                "invalid_value",
                f"clusters.{cluster_name}.resolved_files must not be empty",
            )

        resolved_files: list[str] = []
        seen_in_cluster: set[str] = set()
        for raw_path in raw_resolved_files:
            resolved_path = _validate_repo_relative_path(
                raw_path, field=f"clusters.{cluster_name}.resolved_files"
            )
            if resolved_path in seen_in_cluster:
                _fail(
                    "duplicate_resolved_file",
                    f"{resolved_path!r} is repeated in cluster {cluster_name!r}",
                )
            seen_in_cluster.add(resolved_path)
            owner = file_owners.get(resolved_path)
            if owner is not None:
                _fail(
                    "resolved_file_cluster_overlap",
                    f"{resolved_path!r} belongs to both {owner!r} and {cluster_name!r}",
                )
            full_path = root / resolved_path
            if not full_path.is_file():
                _fail(
                    "resolved_file_missing",
                    f"{resolved_path!r} from cluster {cluster_name!r} does not exist",
                )
            file_owners[resolved_path] = cluster_name
            resolved_files.append(resolved_path)

        navigation_ref = cluster.get("navigation_ref")
        if navigation_ref is not None:
            navigation_ref = _string(
                navigation_ref, field=f"clusters.{cluster_name}.navigation_ref"
            )
        clusters[cluster_name] = MigrationCluster(
            name=cluster_name,
            migration_state=state,
            resolved_files=tuple(resolved_files),
            process_globs=process_globs,
            premise_refs=premise_refs,
            terminal_owner=None,
            navigation_ref=navigation_ref,
        )

    return MigrationManifest(
        source=source,
        frozen_at_commit=frozen_at_commit,
        resolved_at=resolved_at,
        clusters=MappingProxyType(clusters),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse manifest resolved-file frozen sets"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path.cwd() / "docs/ownership/doc_migration_manifest.yaml",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true", help="print frozen sets as JSON")
    arguments = parser.parse_args()
    try:
        manifest = parse_migration_manifest(
            arguments.manifest, repo_root=arguments.repo_root
        )
    except MigrationManifestError as error:
        print(f"migration manifest validation failed [{error.error_class}]: {error}")
        return 1

    if arguments.json:
        print(
            json.dumps(
                {name: sorted(paths) for name, paths in manifest.frozen_sets.items()},
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(
            "migration manifest green: "
            f"{len(manifest.clusters)} clusters, {len(manifest.frozen_files)} frozen files"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
