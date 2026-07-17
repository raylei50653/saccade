"""Generic schema contract for sealed research evidence packets.

These tests are the permanent replacement for per-packet pytest files: once a
study is sealed, its packet is preserved by artifacts plus this generic
validation, not by packet-specific regression tests (see
tests/research/README.md).
"""

from __future__ import annotations

import re

import pytest

from tests.contract.packet_inventory import (
    EVIDENCE_ROOT,
    GENERIC_RESEARCH_PACKET,
    H0_PRESEAL_FREEZE_V3_ARTIFACT,
    H0_PRESEAL_FREEZE_V3_FILENAME,
    checksum_inventory,
    evidence_kind,
    h0_preseal_freeze_v3_dirs,
    h0_preseal_freeze_v3_layout_errors,
    load_manifest,
    packet_dirs,
    packet_ids,
    unclassified_evidence_dirs,
)


def test_evidence_root_exists() -> None:
    # Deliberately NOT a skip: losing or renaming the evidence root must
    # fail CI, otherwise the whole packet contract is fail-open.
    assert EVIDENCE_ROOT.is_dir(), f"evidence root missing: {EVIDENCE_ROOT}"


def test_packet_set_nonempty() -> None:
    assert packet_dirs(), (
        f"no sealed packets found under {EVIDENCE_ROOT}; an empty packet set "
        "would silently skip every parameterized contract check"
    )


def test_all_evidence_directories_are_classified() -> None:
    unknown = unclassified_evidence_dirs()
    assert not unknown, (
        "evidence directories must be explicitly classified as a dated generic "
        f"packet or a governance artifact; unknown: {[path.name for path in unknown]}"
    )


def test_exact_h0_preseal_freeze_v3_dirs_are_governance_artifacts() -> None:
    violations: dict[str, list[str]] = {}
    for evidence_dir in h0_preseal_freeze_v3_dirs():
        errors = h0_preseal_freeze_v3_layout_errors(evidence_dir)
        if errors:
            violations[evidence_dir.name] = errors
    assert not violations, f"non-canonical H0 v3 governance artifacts: {violations}"


@pytest.fixture(params=packet_dirs(), ids=packet_ids())
def packet(request):
    return request.param


def test_packet_dir_is_dated(packet) -> None:
    assert evidence_kind(packet) == GENERIC_RESEARCH_PACKET, (
        f"packet dir {packet.name!r} must be a dated generic research packet"
    )


def test_manifest_exists_and_parses(packet) -> None:
    manifest_path = packet / "manifest.json"
    assert manifest_path.is_file(), f"{packet.name}: manifest.json missing"
    manifest = load_manifest(packet)
    assert isinstance(manifest, dict) and manifest, (
        f"{packet.name}: manifest.json must be a non-empty JSON object"
    )


def test_packet_has_checksum_inventory(packet) -> None:
    inventory = checksum_inventory(packet)
    assert inventory, (
        f"{packet.name}: no checksum inventory found "
        "(expected SHA256SUMS.json or a files/artifacts/artifact_sha256 "
        "mapping in manifest.json)"
    )
    hex_re = re.compile(r"^[0-9a-f]{64}$")
    for name, sha in inventory.items():
        assert hex_re.match(sha), f"{packet.name}: bad sha256 for {name!r}: {sha!r}"


@pytest.mark.parametrize(
    ("name", "expected_kind"),
    [
        ("study_20260717", GENERIC_RESEARCH_PACKET),
        ("h0_preseal_freeze_20260716", GENERIC_RESEARCH_PACKET),
        ("h0_preseal_freeze_" + "a" * 40, H0_PRESEAL_FREEZE_V3_ARTIFACT),
        ("h0_preseal_freeze_" + "a" * 39, None),
        ("h0_preseal_freeze_" + "a" * 41, None),
        ("h0_preseal_freeze_" + "A" * 40, None),
        ("unclassified_evidence", None),
    ],
)
def test_evidence_kind_is_explicit_and_fail_closed(
    tmp_path, name: str, expected_kind: str | None
) -> None:
    assert evidence_kind(tmp_path / name) == expected_kind


def test_h0_preseal_freeze_v3_layout_accepts_only_canonical_artifact(tmp_path) -> None:
    evidence_dir = tmp_path / ("h0_preseal_freeze_" + "a" * 40)
    evidence_dir.mkdir()
    (evidence_dir / H0_PRESEAL_FREEZE_V3_FILENAME).write_text("{}")

    assert h0_preseal_freeze_v3_layout_errors(evidence_dir) == []


@pytest.mark.parametrize(
    "names",
    [
        ["wrong_artifact.json"],
        [H0_PRESEAL_FREEZE_V3_FILENAME, "extra.json"],
    ],
    ids=["wrong-artifact-name", "extra-file"],
)
def test_h0_preseal_freeze_v3_layout_rejects_noncanonical_entries(
    tmp_path, names
) -> None:
    evidence_dir = tmp_path / ("h0_preseal_freeze_" + "a" * 40)
    evidence_dir.mkdir()
    for name in names:
        (evidence_dir / name).write_text("{}")

    assert h0_preseal_freeze_v3_layout_errors(evidence_dir)
