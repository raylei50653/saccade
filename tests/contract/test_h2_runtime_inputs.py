"""Runtime fixtures and assets are content-bound, never called non-execution.

Two things beyond content binding are pinned here, both about *membership*:

  * the measurement fixture is the § C3.2 item 10 seven-sequence member set, in
    both phases, because § C3.1(b) requires the published axes to be equal across
    them — a Phase-A manifest binding one sequence would have to move at the
    Phase-B freeze, inside the window that freezes `identity_semantics`;
  * `phase_a_evidence` (§ C3.8) is bound and watched while belonging to *neither*
    digest. It cannot move the published axis, which is undefined until Phase A
    has run, and it cannot move `full_digest`, which the Layer-P v2 certificate
    records — a section that moved it would invalidate certificates for reasons
    having nothing to do with what was built or run.
"""

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
    for sequence in {inputs.IDENTITY_SEQUENCE, *inputs.MEASUREMENT_SEQUENCES}:
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
    assert [
        item["sequence"] for item in manifest["measurement_fixture"]["sequences"]
    ] == list(inputs.MEASUREMENT_SEQUENCES)
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


def test_symlink_chain_records_multi_hop_intermediate_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """configured -> intermediate -> final must record the intermediate hop."""
    data_root, build_dir = _fixture_tree(tmp_path, monkeypatch)
    final = _write(tmp_path / "controlled" / "version-a.bin", b"same-bytes")
    intermediate = tmp_path / "controlled" / "current"
    intermediate.parent.mkdir(parents=True, exist_ok=True)
    intermediate.symlink_to(final)
    configured = tmp_path / "assets" / f"{inputs.RUNTIME_ASSET_FIELDS[0]}.bin"
    configured.unlink()
    configured.symlink_to(intermediate)

    manifest = inputs.build_manifest(build_dir=build_dir, data_root=data_root)
    record = next(
        item
        for item in manifest["runtime_assets"]["files"]
        if item["role"] == inputs.RUNTIME_ASSET_FIELDS[0]
    )
    chain_paths = {item["path"] for item in record["symlink_chain"]}
    configured_abs = Path(os.path.abspath(configured)).as_posix()
    intermediate_abs = Path(os.path.abspath(intermediate)).as_posix()
    final_abs = Path(os.path.abspath(final)).as_posix()

    assert configured_abs in chain_paths
    assert intermediate_abs in chain_paths
    assert record["resolved_path"] == final_abs

    watched = set(inputs.watch_paths((configured,)))
    assert Path(configured_abs) in watched
    assert Path(intermediate_abs) in watched
    assert Path(final_abs) in watched

    bound = set(inputs.bound_paths(manifest))
    assert Path(intermediate_abs) in bound


def test_manifest_rejects_multi_hop_intermediate_equal_content_retarget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Transient-capable intermediate retarget must fail post-run revalidation.

    Chain: configured link → intermediate link → final file.
    Retargeting only the intermediate (equal content) must move the binding
    even when the configured path and final content digest look unchanged.
    """
    data_root, build_dir = _fixture_tree(tmp_path, monkeypatch)
    first = _write(tmp_path / "controlled" / "version-a.bin", b"same-bytes")
    second = _write(tmp_path / "controlled" / "version-b.bin", b"same-bytes")
    intermediate = tmp_path / "controlled" / "current"
    intermediate.parent.mkdir(parents=True, exist_ok=True)
    intermediate.symlink_to(first)
    configured = tmp_path / "assets" / f"{inputs.RUNTIME_ASSET_FIELDS[0]}.bin"
    configured.unlink()
    configured.symlink_to(intermediate)
    manifest = inputs.build_manifest(build_dir=build_dir, data_root=data_root)

    intermediate.unlink()
    intermediate.symlink_to(second)

    with pytest.raises(inputs.RuntimeInputError, match="path binding moved"):
        inputs.validate_manifest(manifest, verify_files=True)


def test_manifest_revalidation_rejects_fixture_membership_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, build_dir = _fixture_tree(tmp_path, monkeypatch)
    manifest = inputs.build_manifest(build_dir=build_dir, data_root=data_root)
    _write(
        data_root / "train" / inputs.MEASUREMENT_SEQUENCES[0] / "img1" / "000002.jpg",
        b"late-frame",
    )

    with pytest.raises(inputs.RuntimeInputError, match="membership moved"):
        inputs.validate_manifest(manifest, verify_files=True)


# --------------------------------------------------------------------------- #
# The seven-sequence member set (§ C3.2 item 10 / § C3.9 item 2)               #
# --------------------------------------------------------------------------- #
def test_the_measurement_fixture_binds_all_seven_sequences_in_lexicographic_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, build_dir = _fixture_tree(tmp_path, monkeypatch)
    manifest = inputs.build_manifest(build_dir=build_dir, data_root=data_root)
    section = manifest["measurement_fixture"]

    assert len(inputs.MEASUREMENT_SEQUENCES) == 7
    assert list(inputs.MEASUREMENT_SEQUENCES) == sorted(inputs.MEASUREMENT_SEQUENCES)
    assert [item["sequence"] for item in section["sequences"]] == list(
        inputs.MEASUREMENT_SEQUENCES
    )
    assert (
        sum(item["file_count"] for item in section["sequences"])
        == section["file_count"]
    )
    prefixes = {record["coordinate"].split("/", 1)[0] for record in section["files"]}
    assert prefixes == set(inputs.MEASUREMENT_SEQUENCES)


def test_the_identity_fixture_is_bound_separately_from_its_measurement_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§ C3.2: MOT17-09-SDP has two roles and they may not be conflated."""
    data_root, build_dir = _fixture_tree(tmp_path, monkeypatch)
    manifest = inputs.build_manifest(build_dir=build_dir, data_root=data_root)

    assert inputs.IDENTITY_SEQUENCE in inputs.MEASUREMENT_SEQUENCES
    assert {record["role"] for record in manifest["identity_fixture"]["files"]} == {
        inputs.IDENTITY_FIXTURE_ROLE
    }
    assert {record["role"] for record in manifest["measurement_fixture"]["files"]} == {
        inputs.MEASUREMENT_FIXTURE_ROLE
    }
    identity_coordinates = {
        record["coordinate"] for record in manifest["identity_fixture"]["files"]
    }
    measured = {
        record["coordinate"]
        for record in manifest["measurement_fixture"]["files"]
        if record["coordinate"].startswith(f"{inputs.IDENTITY_SEQUENCE}/")
    }
    assert identity_coordinates == measured


def test_a_short_measurement_sequence_list_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, build_dir = _fixture_tree(tmp_path, monkeypatch)
    with pytest.raises(inputs.RuntimeInputError, match="measurement fixtures must be"):
        inputs.build_manifest(
            build_dir=build_dir,
            data_root=data_root,
            measurement_sequences=inputs.MEASUREMENT_SEQUENCES[:1],
        )


def test_a_dropped_sequence_is_caught_by_revalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, build_dir = _fixture_tree(tmp_path, monkeypatch)
    manifest = inputs.build_manifest(build_dir=build_dir, data_root=data_root)
    dropped = dict(manifest["measurement_fixture"])
    dropped["sequences"] = dropped["sequences"][:-1]
    with pytest.raises(inputs.RuntimeInputError, match="sequences must be"):
        inputs.validate_manifest({**manifest, "measurement_fixture": dropped})


def test_every_sequence_moves_the_published_coordinate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No sequence is decorative: each one's content is inside the axis."""
    data_root, build_dir = _fixture_tree(tmp_path, monkeypatch)
    for index, sequence in enumerate(inputs.MEASUREMENT_SEQUENCES):
        before = inputs.build_manifest(build_dir=build_dir, data_root=data_root)
        _write(
            data_root / "train" / sequence / "img1" / "000001.jpg",
            f"moved-{index}".encode(),
        )
        after = inputs.build_manifest(build_dir=build_dir, data_root=data_root)
        assert before["coordinate_digest"] != after["coordinate_digest"], sequence


# --------------------------------------------------------------------------- #
# phase_a_evidence is bound, watched, and in neither digest (§ C3.8)           #
# --------------------------------------------------------------------------- #
def _evidence_root(tmp_path: Path) -> Path:
    root = tmp_path / "evidence" / "h2_measure_a_I40A_F64"
    _write(root / "freeze_record.json", b'{"freeze": true}')
    _write(root / "observation.json", b'{"result": "measurement_pass"}')
    return root


def test_phase_a_evidence_moves_neither_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The published axis would otherwise be undefined until Phase A had run, and
    `full_digest` is what the Layer-P v2 certificate records."""
    data_root, build_dir = _fixture_tree(tmp_path, monkeypatch)
    without = inputs.build_manifest(build_dir=build_dir, data_root=data_root)
    with_evidence = inputs.build_manifest(
        build_dir=build_dir,
        data_root=data_root,
        phase_a_evidence_root=_evidence_root(tmp_path),
    )

    assert inputs.PHASE_A_EVIDENCE_SECTION not in without
    assert with_evidence[inputs.PHASE_A_EVIDENCE_SECTION]["file_count"] == 2
    assert without["coordinate_digest"] == with_evidence["coordinate_digest"]
    assert without["full_digest"] == with_evidence["full_digest"]


def test_phase_a_evidence_is_absent_from_the_published_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, build_dir = _fixture_tree(tmp_path, monkeypatch)
    manifest = inputs.build_manifest(
        build_dir=build_dir,
        data_root=data_root,
        phase_a_evidence_root=_evidence_root(tmp_path),
    )
    axis = inputs.publication_axis(manifest)

    assert set(axis) == {"digest", *inputs.COORDINATE_SECTIONS}
    assert inputs.PHASE_A_EVIDENCE_SECTION not in axis
    assert "build_artifacts" not in axis


def test_phase_a_evidence_is_bound_and_watched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§ C3.2 item 7: it is an `F_B` member and a `BoundInputMonitor` watch-set
    member — terminal 1 fires on a write to it."""
    data_root, build_dir = _fixture_tree(tmp_path, monkeypatch)
    evidence = _evidence_root(tmp_path)
    manifest = inputs.build_manifest(
        build_dir=build_dir, data_root=data_root, phase_a_evidence_root=evidence
    )
    observation = Path(os.path.abspath(evidence / "observation.json"))

    assert observation in set(inputs.consumer_paths(manifest))
    assert observation in set(inputs.bound_paths(manifest))
    inputs.validate_manifest(manifest, verify_files=True)


def test_a_write_to_the_bound_phase_a_evidence_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, build_dir = _fixture_tree(tmp_path, monkeypatch)
    evidence = _evidence_root(tmp_path)
    manifest = inputs.build_manifest(
        build_dir=build_dir, data_root=data_root, phase_a_evidence_root=evidence
    )
    # Equal length, different bytes: the content check must be what catches it,
    # not the cheaper size check.
    (evidence / "observation.json").write_bytes(b'{"result": "measurement_FAIL"}')

    with pytest.raises(inputs.RuntimeInputError, match="content moved"):
        inputs.validate_manifest(manifest, verify_files=True)


def test_an_added_phase_a_evidence_artifact_is_membership_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, build_dir = _fixture_tree(tmp_path, monkeypatch)
    evidence = _evidence_root(tmp_path)
    manifest = inputs.build_manifest(
        build_dir=build_dir, data_root=data_root, phase_a_evidence_root=evidence
    )
    _write(evidence / "late_artifact.json", b"{}")

    with pytest.raises(inputs.RuntimeInputError, match="membership moved"):
        inputs.validate_manifest(manifest, verify_files=True)


def test_the_digest_member_sets_are_exactly_the_declared_ones() -> None:
    """§ C3.8's split, stated once so an added section cannot land in the wrong
    digest by omission."""
    assert inputs.COORDINATE_SECTIONS == (
        "identity_fixture",
        "measurement_fixture",
        "runtime_assets",
        "third_party_runtime",
    )
    assert inputs.FULL_DIGEST_SECTIONS == (
        *inputs.COORDINATE_SECTIONS,
        "build_artifacts",
    )
    assert inputs.PHASE_A_EVIDENCE_SECTION not in inputs.FULL_DIGEST_SECTIONS
