"""The behavior probe is bounded and its publication intake fails closed.

Three properties carry the probe, and each has a way of being quietly wrong:

  * **member set** — the digest covers exactly the four § 4.0 members. Digesting
    the whole inventory dict would fold in mode, fixture, and any future key, so
    an identity-mode digest and a production digest of the same behavior would
    differ and the probe would stop being about observed behavior.
  * **sensitivity** — every member must actually move the digest. A member that
    is collected but not digested is worse than absent: it looks like coverage.
  * **fail-closed intake** — a probe may not be built from a run whose repeats
    disagreed. A digest that averages over a non-reproducible run is a fiction.

No GPU here: these exercise the pure digest/intake logic. The runs themselves are
gates G1/G2, recorded in the declaration § 5.1.2–5.1.3.
"""

# scope: tracking, system
# function: contract
# lifecycle: active

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import build_runtime_identity as identity  # noqa: E402
import h2_behavioral_identity as behavior  # noqa: E402
import h2_runtime_inputs as runtime_inputs  # noqa: E402


def _inventory() -> dict:
    return {
        "active_tid_slot_pairs": [{"frame": 1, "pairs": [[7, 0], [3, 1]]}],
        "final_track_rows": [
            {
                "binary32_bits": [1065353216, 0, 1073741824, 0, 1056964608],
                "class": 0,
                "frame": 1,
                "row_index": 0,
                "track_id": 7,
            }
        ],
        "mot_output": {"length": 42, "sha256": "ab" * 32},
        "relink_debug_raw": list(range(13)),
        "schema": behavior.BEHAVIOR_SCHEMA,
    }


def test_the_declared_member_set_is_exactly_four() -> None:
    assert behavior.BEHAVIOR_MEMBERS == (
        "active_tid_slot_pairs",
        "final_track_rows",
        "mot_output",
        "relink_debug_raw",
    )


def test_trace_only_members_are_not_in_the_probe() -> None:
    """§ 4.0: they cannot exist capture-off, so including them would make the
    axis uncomputable in identity mode."""
    for excluded in (
        "proposal_projection",
        "winner_commit_projection",
        "overflow_vector",
    ):
        assert excluded not in behavior.BEHAVIOR_MEMBERS


def test_the_digest_ignores_provenance_keys() -> None:
    """Mode, fixture and future keys must not enter the digest."""
    base = _inventory()
    decorated = {**base, "mode": "identity", "sequence": "MOT17-09-SDP", "future": 1}
    assert behavior.behavior_digest(decorated) == behavior.behavior_digest(base)


@pytest.mark.parametrize("member", behavior.BEHAVIOR_MEMBERS)
def test_every_member_moves_the_digest(member: str) -> None:
    base = _inventory()
    before = behavior.behavior_digest(base)
    mutated = copy.deepcopy(base)
    if member == "mot_output":
        mutated[member] = {"length": 43, "sha256": "cd" * 32}
    elif member == "relink_debug_raw":
        mutated[member] = [99] + list(range(1, 13))
    elif member == "active_tid_slot_pairs":
        mutated[member] = [{"frame": 1, "pairs": [[7, 0], [4, 1]]}]
    else:
        mutated[member][0]["track_id"] = 8
    assert behavior.behavior_digest(mutated) != before, (
        f"{member} is collected but does not affect the digest — coverage in name only"
    )


def test_slot_order_is_normalized_not_asserted() -> None:
    """The recorder owes A7.6's canonical order; the native call does not provide it.

    `get_active_tid_slot_pairs()` iterates `std::unordered_map<int,int>` in
    track-id bucket order (tracker_gpu.cu:5084). Two runs that observe the same
    active set in different bucket orders must produce the same digest.
    """
    ordered = _inventory()
    ordered["active_tid_slot_pairs"] = [{"frame": 1, "pairs": [[7, 0], [3, 1]]}]
    same_set_other_order = _inventory()
    same_set_other_order["active_tid_slot_pairs"] = [
        {"frame": 1, "pairs": sorted([[3, 1], [7, 0]], key=lambda p: p[1])}
    ]
    assert behavior.behavior_digest(ordered) == behavior.behavior_digest(
        same_set_other_order
    )


def test_an_incomplete_inventory_fails_closed() -> None:
    for member in behavior.BEHAVIOR_MEMBERS:
        broken = _inventory()
        del broken[member]
        with pytest.raises(behavior.BehavioralIdentityError):
            behavior.behavior_digest(broken)


def test_an_empty_inventory_is_not_an_identity() -> None:
    empty = _inventory()
    empty["final_track_rows"] = []
    with pytest.raises(behavior.BehavioralIdentityError):
        behavior.behavior_digest(empty)


def test_a_wrong_schema_fails_closed() -> None:
    wrong = _inventory()
    wrong["schema"] = "something_else_v1"
    with pytest.raises(behavior.BehavioralIdentityError):
        behavior.behavior_digest(wrong)


def test_canonical_json_is_h0s_convention() -> None:
    raw = behavior.canonical_json_bytes({"b": 1, "a": [1, 2]})
    assert raw == b'{"a":[1,2],"b":1}'
    with pytest.raises(ValueError):
        behavior.canonical_json_bytes({"nan": float("nan")})


# --------------------------------------------------------------------------- #
# Identity intake                                                              #
# --------------------------------------------------------------------------- #
def _behavior_payload(tmp_path: Path, **overrides) -> dict:
    build_dir = tmp_path / "build"
    build_dir.mkdir(exist_ok=True)
    extension = build_dir / "saccade_tracking_ext.test.so"
    plugin = build_dir / "libsaccade_scan_plugin.so"
    extension.write_bytes(b"extension")
    plugin.write_bytes(b"plugin")
    artifacts = []
    for role, path in sorted(
        (
            ("tracking_extension", extension),
            ("tensorrt_scan_plugin", plugin),
        )
    ):
        artifacts.append(
            {
                "coordinate": path.name,
                "length": path.stat().st_size,
                "path": path.as_posix(),
                "role": role,
                "sha256": identity.sha256_file(path),
            }
        )
    projection = [
        {
            "coordinate": item["coordinate"],
            "length": item["length"],
            "role": item["role"],
            "sha256": item["sha256"],
        }
        for item in artifacts
    ]
    payload = {
        "build_witness": {
            "artifacts": artifacts,
            "build_dir": build_dir.as_posix(),
            "digest": behavior.digest(projection),
            "schema": behavior.BUILD_WITNESS_SCHEMA,
        },
        "determinism_pinned": True,
        "digest": "ab" * 32,
        "digests": ["ab" * 32],
        "identical": True,
        "mode": "identity",
        "preset": behavior.POLICY_PRESET_REL,
        "repeats": 1,
        "recorder_sha256": identity.sha256_file(Path(behavior.__file__)),
        "resolved_fingerprint": identity.decision_surface_axis()[
            "resolved_bridge_policy_config_v1"
        ],
        "schema": behavior.RESULT_SCHEMA,
        "sequence": behavior.IDENTITY_SEQUENCE,
    }
    payload.update(overrides)
    return payload


def test_a_non_reproducible_run_cannot_define_a_probe(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            _behavior_payload(
                tmp_path,
                identical=False,
                repeats=3,
                digest=None,
                digests=["ab" * 32, "cd" * 32, "ef" * 32],
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(identity.IdentityError, match="repeats disagreed"):
        identity.load_identity_behavior_probe(path)


def test_a_foreign_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(_behavior_payload(tmp_path, schema="other_v1")), encoding="utf-8"
    )
    with pytest.raises(identity.IdentityError):
        identity.load_identity_behavior_probe(path)


def test_a_valid_payload_loads(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_behavior_payload(tmp_path)), encoding="utf-8")
    loaded = identity.load_identity_behavior_probe(path)
    assert loaded["digest"] == "ab" * 32
    assert loaded["state"] == "computed"


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("mode", "production", "expected mode"),
        ("sequence", behavior.MEASUREMENT_SEQUENCE, "expected sequence"),
        ("preset", "other.yaml", "preset mismatch"),
        ("determinism_pinned", False, "pinning mismatch"),
        ("resolved_fingerprint", "00" * 32, "does not match"),
        ("recorder_sha256", "00" * 32, "recorder does not match"),
        ("digest", "not-a-sha256", "valid SHA-256"),
        ("build_witness", None, "build_witness"),
    ],
)
def test_identity_probe_intake_rejects_wrong_provenance(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    path = tmp_path / f"{field}.json"
    path.write_text(
        json.dumps(_behavior_payload(tmp_path, **{field: value})), encoding="utf-8"
    )
    with pytest.raises(identity.IdentityError, match=match):
        identity.load_identity_behavior_probe(path)


def test_production_repeat_probe_is_separate_from_identity_intake(
    tmp_path: Path,
) -> None:
    path = tmp_path / "production.json"
    path.write_text(
        json.dumps(
            _behavior_payload(
                tmp_path,
                determinism_pinned=False,
                digests=["ab" * 32, "ab" * 32],
                mode="production",
                repeats=2,
                sequence=behavior.MEASUREMENT_SEQUENCE,
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(identity.IdentityError, match="expected mode"):
        identity.load_identity_behavior_probe(path)
    loaded = identity.load_production_repeat_probe(path)
    assert loaded["mode"] == "production"


def test_cli_defaults_production_probe_to_measurement_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[tuple[str, bool]] = []

    def fake_run(*, sequence: str, identity_mode: bool, output_dir: Path) -> dict:
        observed.append((sequence, identity_mode))
        return {
            "build_witness": {},
            "digest": "ab" * 32,
            "mode": "identity" if identity_mode else "production",
            "preset": behavior.POLICY_PRESET_REL,
            "resolved_fingerprint": "cd" * 32,
            "sequence": sequence,
        }

    monkeypatch.setattr(behavior, "run_behavior_inventory", fake_run)
    output = tmp_path / "probe.json"
    assert behavior.main(["--emit", output.as_posix()]) == 0
    assert observed == [(behavior.MEASUREMENT_SEQUENCE, False)]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["determinism_pinned"] is False


# --------------------------------------------------------------------------- #
# Axis assembly                                                                #
# --------------------------------------------------------------------------- #
def test_a_publication_without_probe_or_runtime_inputs_is_incomplete() -> None:
    built = identity.build_publication(probe=None, runtime_input_manifest=None)
    assert built["publication_complete"] is False
    assert built["probe"]["digest"] is None
    assert built["coordinate"]["runtime_inputs"] is None
    assert built["equivalence"]["state"] == "unproven"


def test_the_three_static_axes_are_reproducible() -> None:
    first = identity.build_publication(probe=None, runtime_input_manifest=None)
    second = identity.build_publication(probe=None, runtime_input_manifest=None)
    for axis in identity.STATIC_COORDINATE_AXES:
        assert first["coordinate"][axis] == second["coordinate"][axis]
        assert first["coordinate"][axis] is not None


def test_the_implementation_axis_covers_the_kernel_and_the_preset() -> None:
    files = set(identity.decision_relevant_files())
    assert "src/tracking/tracker_gpu.cu" in files
    assert "include/tracking/tracker_gpu.hpp" in files
    assert "configs/presets/mamba_whole_graph_m.yaml" in files
    assert "src/saccade/perception/eval/relink.py" in files
    assert "src/saccade/perception/temporal_yolo/mamba_gated_detector.py" in files
    # Prose is classified decision-relevant but kept out of the axis, so a README
    # edit cannot bump `implementation`.
    assert not any(path.endswith(".md") for path in files)


def test_identity_semantics_axis_binds_the_ruler_itself() -> None:
    files = set(identity.tracked_files_for_class("identity_semantics"))
    assert {
        ".github/workflows/runtime_identity.yml",
        "scripts/tools/h2_behavioral_identity.py",
        "scripts/tools/h2_path_partition.py",
        "scripts/tools/h2_runtime_inputs.py",
        "scripts/tools/h2_terminal_partition.py",
        "scripts/tools/build_runtime_identity.py",
        "scripts/tools/check_runtime_identity_staleness.py",
        "scripts/tools/run_h2_layer_p.py",
        *identity.partition.EXECUTION_ARTIFACT_SCHEMA_PATHS,
    } <= files


def test_gpu_reattestation_triggers_when_the_ruler_changes() -> None:
    workflow = (_REPO / ".github/workflows/runtime_identity.yml").read_text(
        encoding="utf-8"
    )
    for path in sorted(identity.partition.IDENTITY_SEMANTICS_PATHS):
        assert f'- "{path}"' in workflow, (
            f"runtime re-attestation does not trigger when {path} changes"
        )


def test_gpu_reattestation_runs_for_same_repository_pull_requests_only() -> None:
    workflow = (_REPO / ".github/workflows/runtime_identity.yml").read_text(
        encoding="utf-8"
    )
    assert "pull_request:" in workflow
    assert (
        "github.event.pull_request.head.repo.full_name == github.repository" in workflow
    )
    assert "github.event.pull_request.head.sha || github.sha" in workflow


def test_gpu_reattestation_binds_controlled_host_runtime_inputs_lexically() -> None:
    workflow = (_REPO / ".github/workflows/runtime_identity.yml").read_text(
        encoding="utf-8"
    )
    assert "SACCADE_CONTROLLED_RESOURCE_ROOT" in workflow
    # Relative to GITHUB_WORKSPACE — no absolute /home/... path in the workflow.
    assert "/home/ray/" not in workflow
    assert "../../../../developer/ai/saccade" in workflow
    assert "Bind controlled host-local runtime inputs into the checkout" in workflow
    assert "uv sync --frozen --extra dali" in workflow
    assert 'ln -s --relative "${source}" "${target}"' in workflow
    # Every sequence the manifest binds, not a sample of them: the manifest step
    # fails closed on a missing sequence, so a dropped bind is a red CI job on the
    # controlled host and nothing quieter.
    for sequence in {
        runtime_inputs.IDENTITY_SEQUENCE,
        *runtime_inputs.MEASUREMENT_SEQUENCES,
    }:
        assert f'bind_runtime_input "datasets/MOT17/train/{sequence}"' in workflow
    for path in (
        "runs/mamba_gt_yolo26m_v14replica_t3_t1/best.ckpt",
        "runs/gated_det_yolo26m_v14replica/epoch_0012.ckpt",
        "models/yolo/yolo26m.pt",
        "models/yolo/yolo26m_backbone_640_best.engine",
        "models/yolo/mamba_head_26m.engine",
    ):
        assert f'bind_runtime_input "{path}"' in workflow


def test_witness_fields_are_marked_as_carrying_no_authority() -> None:
    built = identity.build_publication(probe=None, runtime_input_manifest=None)
    note = built["witness"]["note"]
    assert "navigation witness" in note
