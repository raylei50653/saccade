"""Runtime fixtures and assets are content-bound, never called non-execution."""

# scope: tracking, system
# function: contract
# lifecycle: active

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import h2_runtime_inputs as inputs  # noqa: E402


def _write(path: Path, value: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _fixture_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.setattr(inputs, "REPO_ROOT", tmp_path)
    data_root = tmp_path / "datasets" / "MOT17"
    for sequence in (inputs.IDENTITY_SEQUENCE, inputs.MEASUREMENT_SEQUENCE):
        root = data_root / "train" / sequence
        _write(root / "seqinfo.ini", sequence.encode())
        _write(root / "img1" / "000001.jpg", b"image-" + sequence.encode())
        _write(root / "gt" / "gt.txt", b"gt-" + sequence.encode())

    preset = {}
    for index, field in enumerate(inputs.RUNTIME_ASSET_FIELDS):
        relative = f"assets/{field}.bin"
        _write(tmp_path / relative, f"asset-{index}".encode())
        preset[field] = relative
    _write(
        tmp_path / inputs.POLICY_PRESET_REL,
        yaml.safe_dump(preset).encode(),
    )
    _write(
        tmp_path / "third_party" / "TrackEval" / "trackeval" / "eval.py",
        b"runtime = True\n",
    )
    build_dir = tmp_path / "build" / "h2"
    _write(build_dir / "saccade_tracking_ext.test.so", b"extension")
    _write(build_dir / "libsaccade_scan_plugin.so", b"plugin")
    return data_root, build_dir


def test_manifest_binds_both_fixtures_assets_third_party_and_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, build_dir = _fixture_tree(tmp_path, monkeypatch)
    manifest = inputs.build_manifest(build_dir=build_dir, data_root=data_root)
    inputs.validate_manifest(manifest, verify_files=True)
    assert manifest["identity_fixture"]["sequence"] == inputs.IDENTITY_SEQUENCE
    assert manifest["measurement_fixture"]["sequence"] == inputs.MEASUREMENT_SEQUENCE
    assert manifest["runtime_assets"]["file_count"] == len(inputs.RUNTIME_ASSET_FIELDS)
    assert manifest["third_party_runtime"]["file_count"] == 1
    assert {item["role"] for item in manifest["build_artifacts"]["files"]} == {
        "tracking_extension",
        "tensorrt_scan_plugin",
    }


def test_asset_mutation_moves_coordinate_and_fails_old_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, build_dir = _fixture_tree(tmp_path, monkeypatch)
    before = inputs.build_manifest(build_dir=build_dir, data_root=data_root)
    asset = tmp_path / "assets" / f"{inputs.RUNTIME_ASSET_FIELDS[0]}.bin"
    asset.write_bytes(b"changed")
    with pytest.raises(inputs.RuntimeInputError, match="content moved"):
        inputs.validate_manifest(before, verify_files=True)
    after = inputs.build_manifest(build_dir=build_dir, data_root=data_root)
    assert before["coordinate_digest"] != after["coordinate_digest"]


def test_build_artifact_changes_full_digest_but_not_stable_coordinate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, build_dir = _fixture_tree(tmp_path, monkeypatch)
    before = inputs.build_manifest(build_dir=build_dir, data_root=data_root)
    (build_dir / "libsaccade_scan_plugin.so").write_bytes(b"rebuilt-plugin")
    after = inputs.build_manifest(build_dir=build_dir, data_root=data_root)
    assert before["coordinate_digest"] == after["coordinate_digest"]
    assert before["full_digest"] != after["full_digest"]


def test_discovery_returns_the_lexical_paths_that_consumers_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, build_dir = _fixture_tree(tmp_path, monkeypatch)
    target = _write(tmp_path / "assets" / "alternate.bin", b"asset-0")
    configured = tmp_path / "assets" / f"{inputs.RUNTIME_ASSET_FIELDS[0]}.bin"
    configured.unlink()
    configured.symlink_to(target)

    discovered = set(
        inputs.discover_bound_paths(build_dir=build_dir, data_root=data_root)
    )

    assert Path(os.path.abspath(configured)) in discovered
    assert target.resolve() not in discovered


def test_manifest_rejects_an_equal_content_asset_symlink_retarget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, build_dir = _fixture_tree(tmp_path, monkeypatch)
    first = _write(tmp_path / "assets" / "first.bin", b"same-bytes")
    second = _write(tmp_path / "assets" / "second.bin", b"same-bytes")
    configured = tmp_path / "assets" / f"{inputs.RUNTIME_ASSET_FIELDS[0]}.bin"
    configured.unlink()
    configured.symlink_to(first)
    manifest = inputs.build_manifest(build_dir=build_dir, data_root=data_root)

    configured.unlink()
    configured.symlink_to(second)

    with pytest.raises(inputs.RuntimeInputError, match="path binding moved"):
        inputs.validate_manifest(manifest, verify_files=True)


def test_manifest_revalidation_rejects_fixture_membership_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, build_dir = _fixture_tree(tmp_path, monkeypatch)
    manifest = inputs.build_manifest(build_dir=build_dir, data_root=data_root)
    _write(
        data_root / "train" / inputs.MEASUREMENT_SEQUENCE / "img1" / "000002.jpg",
        b"late-frame",
    )

    with pytest.raises(inputs.RuntimeInputError, match="membership moved"):
        inputs.validate_manifest(manifest, verify_files=True)
