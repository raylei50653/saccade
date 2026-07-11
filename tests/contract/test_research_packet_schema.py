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
    checksum_inventory,
    load_manifest,
    packet_dirs,
    packet_ids,
)

pytestmark = pytest.mark.skipif(
    not EVIDENCE_ROOT.is_dir(), reason="evidence root not present"
)

_DATED_NAME = re.compile(r".+_\d{8}(T\d{6}Z)?$")


@pytest.fixture(params=packet_dirs(), ids=packet_ids())
def packet(request):
    return request.param


def test_packet_dir_is_dated(packet) -> None:
    assert _DATED_NAME.match(packet.name), (
        f"packet dir {packet.name!r} must carry a _YYYYMMDD seal date suffix"
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
