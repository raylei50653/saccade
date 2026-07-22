"""Contract for the doc migration manifest parser and master-map generator."""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.docs.build_master_map import (
    build_master_map,
    master_map_is_current,
    render_master_map,
    write_master_map,
)
from scripts.docs.migration_manifest import (
    MigrationManifestError,
    parse_migration_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs/ownership/doc_migration_manifest.yaml"
MASTER_MAP = ROOT / "docs/ownership/master_map.generated.md"


def _write_manifest(root: Path, clusters: str) -> Path:
    (root / "docs/research").mkdir(parents=True)
    (root / "docs/research/frozen.md").write_text("frozen", encoding="utf-8")
    (root / "docs/research/active.md").write_text("active", encoding="utf-8")
    manifest = root / "docs/ownership/doc_migration_manifest.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        """snapshot:
  frozen_at_commit: not-a-real-git-commit
  resolved_at: 2026-07-16
clusters:
"""
        + clusters,
        encoding="utf-8",
    )
    return manifest


def test_manifest_produces_current_frozen_sets() -> None:
    manifest = parse_migration_manifest(MANIFEST, repo_root=ROOT)

    assert len(manifest.clusters) == 8
    assert len(manifest.frozen_files) == 55
    assert set(manifest.frozen_sets) == set(manifest.clusters)
    assert all(paths for paths in manifest.resolved_file_sets.values())


def test_manifest_uses_resolved_files_without_git_or_glob_expansion(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        """  isolated:
    migration_state: quarantined
    terminal_owner: null
    premise_refs: []
    process_globs: [docs/research/not-expanded-*]
    resolved_files: [docs/research/frozen.md]
""",
    )

    manifest = parse_migration_manifest(manifest_path, repo_root=tmp_path)

    assert manifest.frozen_at_commit == "not-a-real-git-commit"
    assert manifest.frozen_sets["isolated"] == frozenset({"docs/research/frozen.md"})


def test_manifest_rejects_a_file_repeated_in_one_cluster(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        """  isolated:
    migration_state: quarantined
    terminal_owner: null
    premise_refs: []
    process_globs: []
    resolved_files:
      - docs/research/frozen.md
      - docs/research/frozen.md
""",
    )

    with pytest.raises(MigrationManifestError) as raised:
        parse_migration_manifest(manifest_path, repo_root=tmp_path)

    assert raised.value.error_class == "duplicate_resolved_file"


def test_manifest_rejects_resolved_file_cross_cluster_overlap(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        """  first:
    migration_state: quarantined
    terminal_owner: null
    premise_refs: []
    process_globs: []
    resolved_files: [docs/research/frozen.md]
  second:
    migration_state: quarantined
    terminal_owner: null
    premise_refs: []
    process_globs: []
    resolved_files: [docs/research/frozen.md]
""",
    )

    with pytest.raises(MigrationManifestError) as raised:
        parse_migration_manifest(manifest_path, repo_root=tmp_path)

    assert raised.value.error_class == "resolved_file_cluster_overlap"


@pytest.mark.parametrize(
    "clusters",
    [
        """  duplicate: {}
  duplicate: {}
""",
        """  duplicate_field:
    resolved_files: []
    resolved_files: []
""",
    ],
)
def test_manifest_rejects_duplicate_cluster_or_field_key(
    tmp_path: Path, clusters: str
) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        """snapshot:
  frozen_at_commit: not-a-real-git-commit
  resolved_at: 2026-07-16
clusters:
"""
        + clusters,
        encoding="utf-8",
    )

    with pytest.raises(MigrationManifestError) as raised:
        parse_migration_manifest(manifest_path, repo_root=tmp_path)

    assert raised.value.error_class == "duplicate_yaml_key"


def test_master_map_grays_out_quarantined_files_from_active_views(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        """  isolated:
    migration_state: quarantined
    terminal_owner: null
    premise_refs: []
    process_globs: []
    resolved_files: [docs/research/frozen.md]
""",
    )
    manifest = parse_migration_manifest(manifest_path, repo_root=tmp_path)

    master_map = build_master_map(manifest, repo_root=tmp_path)

    assert "docs/research/frozen.md" in master_map.grayed_out_files
    assert "docs/research/frozen.md" not in master_map.active_index
    assert "docs/research/frozen.md" not in master_map.active_search_view
    assert "docs/research/active.md" in master_map.active_index


def test_generated_master_map_is_deterministic_and_detects_staleness(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        """  isolated:
    migration_state: quarantined
    terminal_owner: null
    premise_refs: []
    process_globs: []
    resolved_files: [docs/research/frozen.md]
""",
    )
    manifest = parse_migration_manifest(manifest_path, repo_root=tmp_path)
    master_map = build_master_map(manifest, repo_root=tmp_path)
    output = tmp_path / "docs/ownership/master_map.generated.md"

    first_render = render_master_map(
        master_map, manifest_path="docs/ownership/doc_migration_manifest.yaml"
    )
    second_render = render_master_map(
        master_map, manifest_path="docs/ownership/doc_migration_manifest.yaml"
    )
    assert first_render == second_render
    assert first_render.startswith(
        "<!-- Generated by scripts/docs/build_master_map.py; do not edit manually. -->"
    )

    write_master_map(
        output,
        master_map,
        manifest_path="docs/ownership/doc_migration_manifest.yaml",
    )
    assert master_map_is_current(
        output,
        master_map,
        manifest_path="docs/ownership/doc_migration_manifest.yaml",
    )

    output.write_text("stale", encoding="utf-8")
    assert not master_map_is_current(
        output,
        master_map,
        manifest_path="docs/ownership/doc_migration_manifest.yaml",
    )


def test_checked_in_master_map_is_current() -> None:
    manifest = parse_migration_manifest(MANIFEST, repo_root=ROOT)
    master_map = build_master_map(manifest, repo_root=ROOT)

    assert master_map_is_current(
        MASTER_MAP,
        master_map,
        manifest_path="docs/ownership/doc_migration_manifest.yaml",
    )
