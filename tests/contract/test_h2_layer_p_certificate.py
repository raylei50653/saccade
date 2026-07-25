"""Layer-P cannot skip admissibility and its certificate binds the full coordinate."""

# scope: tracking, system
# function: contract
# lifecycle: active

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import h2_behavioral_identity as behavior  # noqa: E402
import run_h2_layer_p as layer_p  # noqa: E402


def _write(path: Path, value: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def test_base_is_required_by_the_cli() -> None:
    with pytest.raises(SystemExit):
        layer_p.main([])


def test_certificate_binds_every_review_required_coordinate(tmp_path: Path) -> None:
    controller = layer_p.LayerP.__new__(layer_p.LayerP)
    controller.base = "selected-I"
    controller.build_dir_rel = "build/h2"
    controller.retry_verdict = {
        "admissible": True,
        "base": "selected-I",
        "changed_count": 0,
        "decision_relevant": [],
        "identity_semantics": [],
        "identity_fixture_input": [],
        "measurement_input": [],
        "runtime_asset": [],
        "unclassified": [],
        "plumbing_only": [],
        "non_execution": [],
    }
    probe_path = tmp_path / "probe.json"
    manifest_path = tmp_path / "runtime-inputs.json"
    probe_path.write_bytes(b"probe\n")
    manifest_path.write_bytes(b"manifest\n")
    published = {
        "coordinate": {
            "decision_surface": "d" * 64,
            "environment": "e" * 64,
            "implementation": "i" * 64,
            "identity_semantics": "s" * 64,
            "runtime_inputs": "r" * 64,
        },
        "probe": {"digest": "p" * 64},
    }
    probe = {
        "build_witness": {"digest": "b" * 64},
        "digest": "p" * 64,
        "mode": "identity",
        "schema": behavior.RESULT_SCHEMA,
        "sequence": behavior.IDENTITY_SEQUENCE,
    }
    manifest = {
        "build_artifacts": {"digest": "a" * 64},
        "coordinate_digest": "r" * 64,
        "full_digest": "f" * 64,
    }
    certificate = controller.build_certificate(
        published=published,
        probe=probe,
        manifest=manifest,
        probe_path=probe_path,
        manifest_path=manifest_path,
        extension_witness={"extension_sha256": "x" * 64},
    )
    required = {
        "source_head",
        "source_tree",
        "selected_base",
        "changed_path_verdict",
        "decision_relevant_digest",
        "identity_semantics_digest",
        "plumbing_set_digest",
        "published_identity_file_digest",
        "probe_result_file_digest",
        "runtime_input_manifest_file_digest",
        "runtime_input_coordinate_digest",
        "runtime_input_full_digest",
        "build_artifact_digest",
        "behavior_probe",
        "probe_schema",
        "fixture",
        "mode",
        "equivalence",
    }
    assert required <= set(certificate)
    assert certificate["selected_base"] == "selected-I"
    assert certificate["equivalence"] == "unproven"
    assert certificate["changed_path_verdict"]["admissible"] is True


def test_identity_run_monitors_hashing_and_revalidates_before_final_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    bound_input = _write(tmp_path / "measurement.jpg")
    controller = layer_p.LayerP.__new__(layer_p.LayerP)
    controller.build_dir = tmp_path
    controller.fixture = behavior.IDENTITY_SEQUENCE
    controller.work_dir = tmp_path / "work"
    controller.work_dir.mkdir()
    controller.record = {"stages": {}}
    monkeypatch.setattr(controller, "_stage", lambda *_args, **_kwargs: None)

    manifest = {
        "build_artifacts": {"digest": "b" * 64},
        "coordinate_digest": "r" * 64,
        "full_digest": "f" * 64,
    }
    probe = {
        "build_witness": {"digest": "b" * 64},
        "digest": "p" * 64,
    }

    def discover_bound_paths(**_kwargs: object) -> tuple[Path, ...]:
        events.append("discover")
        return (bound_input,)

    class Monitor:
        def __init__(self, paths: object) -> None:
            assert bound_input in paths  # type: ignore[operator]
            events.append("monitor_start")

        def drain(self) -> list[object]:
            events.append("final_drain")
            return []

        def close(self) -> None:
            events.append("monitor_close")

    def build_manifest(**_kwargs: object) -> dict[str, object]:
        events.append("build_manifest")
        return manifest

    def run_probe(*_args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        events.append("probe")
        command = kwargs.get("args")
        if command is None and _args:
            command = _args[0]
        assert isinstance(command, list)
        result_path = Path(command[command.index("--emit") + 1])
        result_path.write_text("{}\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    def validate_manifest(
        _manifest: object, *, verify_files: bool = False
    ) -> dict[str, object]:
        assert verify_files is True
        events.append("post_run_validate")
        return manifest

    monkeypatch.setattr(
        layer_p.runtime_inputs, "discover_bound_paths", discover_bound_paths
    )
    monkeypatch.setattr(layer_p.h0_controller, "BoundInputMonitor", Monitor)
    monkeypatch.setattr(layer_p.runtime_inputs, "build_manifest", build_manifest)
    monkeypatch.setattr(
        layer_p.runtime_inputs,
        "consumer_paths",
        lambda _manifest: (bound_input,),
    )
    monkeypatch.setattr(layer_p.runtime_inputs, "validate_manifest", validate_manifest)
    monkeypatch.setattr(
        layer_p.identity, "tracked_files_for_class", lambda _path_class: ()
    )
    monkeypatch.setattr(layer_p.subprocess, "run", run_probe)
    monkeypatch.setattr(
        layer_p.identity, "load_identity_behavior_probe", lambda _path: probe
    )

    controller.identity_run(
        {
            "coordinate": {"runtime_inputs": "r" * 64},
            "probe": {"digest": "p" * 64},
        }
    )

    assert events == [
        "discover",
        "monitor_start",
        "build_manifest",
        "probe",
        "post_run_validate",
        "final_drain",
        "monitor_close",
    ]


def test_identity_run_blocks_when_post_run_manifest_revalidation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bound_input = _write(tmp_path / "measurement.jpg")
    controller = layer_p.LayerP.__new__(layer_p.LayerP)
    controller.build_dir = tmp_path
    controller.fixture = behavior.IDENTITY_SEQUENCE
    controller.work_dir = tmp_path / "work"
    controller.work_dir.mkdir()
    controller.record = {"stages": {}, "result": "in_progress"}
    monkeypatch.setattr(controller, "_persist", lambda: None)

    manifest = {
        "build_artifacts": {"digest": "b" * 64},
        "coordinate_digest": "r" * 64,
        "full_digest": "f" * 64,
    }

    class Monitor:
        def __init__(self, _paths: object) -> None:
            pass

        def drain(self) -> list[object]:
            return []

        def close(self) -> None:
            pass

    def run_probe(*args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        command = args[0]
        assert isinstance(command, list)
        Path(command[command.index("--emit") + 1]).write_text("{}\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        layer_p.runtime_inputs,
        "discover_bound_paths",
        lambda **_kwargs: (bound_input,),
    )
    monkeypatch.setattr(layer_p.h0_controller, "BoundInputMonitor", Monitor)
    monkeypatch.setattr(
        layer_p.runtime_inputs, "build_manifest", lambda **_kwargs: manifest
    )
    monkeypatch.setattr(
        layer_p.runtime_inputs,
        "consumer_paths",
        lambda _manifest: (bound_input,),
    )
    monkeypatch.setattr(layer_p.subprocess, "run", run_probe)
    monkeypatch.setattr(
        layer_p.runtime_inputs,
        "validate_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            layer_p.runtime_inputs.RuntimeInputError("content moved")
        ),
    )
    monkeypatch.setattr(
        layer_p.identity, "tracked_files_for_class", lambda _path_class: ()
    )

    with pytest.raises(layer_p.Blocked, match="post-run validation failed"):
        controller.identity_run(
            {
                "coordinate": {"runtime_inputs": "r" * 64},
                "probe": {"digest": "p" * 64},
            }
        )
