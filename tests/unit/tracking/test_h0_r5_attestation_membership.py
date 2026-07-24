"""H0-R5 extension/plugin runtime-attestation membership predicates."""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "scripts/tools"
sys.path.insert(0, TOOLS.as_posix())

import h0_runtime_confinement as confinement  # noqa: E402
import run_h0_phase_a as controller  # noqa: E402


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _record(
    path: Path,
    *,
    data: bytes,
    operations: list[str] | None = None,
    bindings: list[str] | None = None,
) -> dict[str, object]:
    return {
        "bindings": list(bindings) if bindings is not None else ["build_artifact"],
        "length": len(data),
        "operations": list(operations)
        if operations is not None
        else ["openat", "mmap_exec"],
        "realpath": path.resolve(strict=True).as_posix(),
        "sha256": _sha(data),
    }


def _attestation(records: list[dict[str, object]], *, state: str = "complete") -> dict:
    return {
        "regular_files": records,
        "state": state,
        "violations": [],
    }


def test_confined_extension_and_plugin_membership_positive(tmp_path: Path) -> None:
    extension = tmp_path / "ext.so"
    plugin = tmp_path / "plugin.so"
    extension.write_bytes(b"extension-bytes")
    plugin.write_bytes(b"plugin-bytes")
    att = _attestation(
        [
            _record(extension, data=b"extension-bytes"),
            _record(plugin, data=b"plugin-bytes"),
        ]
    )
    result = confinement.assert_extension_plugin_membership(
        att,
        extension=extension,
        plugin=plugin,
    )
    assert result["extension_artifact_observed"] is True
    assert result["plugin_artifact_observed"] is True
    assert result["extension_identity_equal"] is True
    assert result["plugin_identity_equal"] is True
    assert "build_artifact" in result["extension_record"]["bindings"]
    assert "build_artifact" in result["plugin_record"]["bindings"]


@pytest.mark.parametrize(
    "mutator,match",
    [
        (
            lambda att, extension, plugin: att["regular_files"].pop(0),
            "absent from runtime attestation",
        ),
        (
            lambda att, extension, plugin: att["regular_files"].pop(1),
            "absent from runtime attestation",
        ),
        (
            lambda att, extension, plugin: att["regular_files"].__setitem__(
                0, {**att["regular_files"][0], "bindings": ["build_runtime_closure"]}
            ),
            "missing build_artifact binding",
        ),
        (
            lambda att, extension, plugin: att["regular_files"].__setitem__(
                1, {**att["regular_files"][1], "sha256": "0" * 64}
            ),
            "hash drift",
        ),
        (
            lambda att, extension, plugin: att["regular_files"].__setitem__(
                0, {**att["regular_files"][0], "operations": ["declared_loaded"]}
            ),
            "do not prove runtime consumption",
        ),
        (
            lambda att, extension, plugin: att["regular_files"].__setitem__(
                0,
                {
                    **att["regular_files"][0],
                    "realpath": (extension.parent / "wrong.so").as_posix(),
                },
            ),
            "absent from runtime attestation",
        ),
    ],
)
def test_membership_negative_cases(tmp_path: Path, mutator, match: str) -> None:
    extension = tmp_path / "ext.so"
    plugin = tmp_path / "plugin.so"
    extension.write_bytes(b"extension-bytes")
    plugin.write_bytes(b"plugin-bytes")
    att = _attestation(
        [
            _record(extension, data=b"extension-bytes"),
            _record(plugin, data=b"plugin-bytes"),
        ]
    )
    mutator(att, extension, plugin)
    with pytest.raises(confinement.ConfinementError, match=match):
        confinement.assert_extension_plugin_membership(
            att, extension=extension, plugin=plugin
        )


def test_only_dependency_closure_observed_rejects(tmp_path: Path) -> None:
    extension = tmp_path / "ext.so"
    plugin = tmp_path / "plugin.so"
    dep = tmp_path / "dep.so"
    extension.write_bytes(b"extension-bytes")
    plugin.write_bytes(b"plugin-bytes")
    dep.write_bytes(b"dependency")
    att = _attestation(
        [
            _record(
                dep,
                data=b"dependency",
                bindings=["build_runtime_closure"],
                operations=["openat", "mmap_read"],
            )
        ]
    )
    with pytest.raises(confinement.ConfinementError, match="absent"):
        confinement.assert_extension_plugin_membership(
            att, extension=extension, plugin=plugin
        )


def test_attestation_complete_but_one_top_level_absent_rejects(tmp_path: Path) -> None:
    extension = tmp_path / "ext.so"
    plugin = tmp_path / "plugin.so"
    extension.write_bytes(b"extension-bytes")
    plugin.write_bytes(b"plugin-bytes")
    att = _attestation([_record(extension, data=b"extension-bytes")], state="complete")
    with pytest.raises(confinement.ConfinementError, match="absent"):
        confinement.assert_extension_plugin_membership(
            att, extension=extension, plugin=plugin
        )


def test_synthetically_inserted_record_without_operation_rejects(
    tmp_path: Path,
) -> None:
    extension = tmp_path / "ext.so"
    plugin = tmp_path / "plugin.so"
    extension.write_bytes(b"extension-bytes")
    plugin.write_bytes(b"plugin-bytes")
    att = _attestation(
        [
            _record(extension, data=b"extension-bytes", operations=[]),
            _record(plugin, data=b"plugin-bytes"),
        ]
    )
    with pytest.raises(confinement.ConfinementError, match="consuming operations"):
        confinement.assert_extension_plugin_membership(
            att, extension=extension, plugin=plugin
        )


def test_python_interpreter_runtime_discovery_includes_base_prefix() -> None:
    python = ROOT / ".venv/bin/python"
    if not python.is_file():
        pytest.skip("venv python unavailable")
    paths = controller.discover_python_interpreter_runtime_paths(python)
    assert paths
    assert any("python3.12" in path or "encodings" in path for path in paths)
    assert any(path.endswith("ld.so.cache") for path in paths)


def test_r4_archive_remains_provenance_invalid_and_valid() -> None:
    """Historical R4 packet must stay immutable (result + digests unchanged)."""
    packet = (
        ROOT
        / "docs/modules/semantic/research/evidence"
        / "h0_phase_a_2a233387a6a321dd43570e2e30dc718571b3b4f4"
    )
    assert packet.is_dir()
    result = packet / "result.json"
    text = result.read_text(encoding="utf-8")
    assert "provenance_invalid" in text
    # Known immutable digests from the R4 witness / claim-state registry.
    assert (packet / "checksums.sha256").read_text(encoding="utf-8")  # non-empty
    manifest = (packet / "manifest.json").read_bytes()
    assert _sha(manifest) == (
        "c3cf4bd8bdfbf0fc2dc500b982ceca8136913d7381b185a4b9506e724e903cf0"
    )
    result_bytes = result.read_bytes()
    assert _sha(result_bytes) == (
        "2c1cfa17c977ad02c6c1dee335810b9ee7ff37f1cbba1382d41a00f06b96529a"
    )
