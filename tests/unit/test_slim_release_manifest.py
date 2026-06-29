"""Integrity checks for the slim release manifest.

Guards against editing mistakes in scripts/release/slim_manifest.yaml:
duplicate entries, entries made redundant by a kept directory, and missing
top-level files. Reproduction against a git ref is exercised via the
build_slim_release.py --check flow, not here.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "scripts" / "release" / "slim_manifest.yaml"


def _load() -> dict[str, list[str]]:
    return yaml.safe_load(MANIFEST.read_text())


def test_manifest_keys_are_lists() -> None:
    data = _load()
    for key in ("keep_dirs", "keep_root_files", "keep_files"):
        assert isinstance(data.get(key), list), f"{key} must be a list"
        assert data[key], f"{key} must not be empty"


def test_no_duplicate_entries() -> None:
    data = _load()
    for key in ("keep_dirs", "keep_root_files", "keep_files"):
        entries = data[key]
        assert len(entries) == len(set(entries)), f"duplicate entries in {key}"


def test_keep_files_not_under_keep_dirs() -> None:
    """Explicit files under a wholly-kept dir are redundant (and a footgun)."""
    data = _load()
    kept_dirs = [d.rstrip("/") + "/" for d in data["keep_dirs"]]
    for path in data["keep_files"]:
        for kept in kept_dirs:
            assert not path.startswith(kept), (
                f"{path} is redundant: {kept} is already kept wholesale"
            )


def test_curated_subset_paths_exist() -> None:
    """Every explicitly listed file must exist in the working tree."""
    data = _load()
    for path in (*data["keep_root_files"], *data["keep_files"]):
        assert (REPO_ROOT / path).exists(), f"manifest path missing: {path}"
