"""Generic manifest/integrity contract for sealed research evidence packets.

Verifies that every checksum-inventory entry resolves to a real file inside
the packet and that its sha256 still matches — i.e. the sealed evidence has
not silently rotted or been edited without re-sealing.
"""

from __future__ import annotations

import hashlib

import pytest

from tests.contract.packet_inventory import (
    EXTERNAL_ARTIFACT_HASH_EXCEPTIONS,
    checksum_inventory,
    load_manifest,
    packet_dirs,
    packet_ids,
    resolve_inventory_path,
)

# No skip guard here: test_research_packet_schema.py asserts the evidence
# root exists and the packet set is non-empty, so a missing root fails CI
# instead of empty-skipping this file's parameterization.


@pytest.fixture(params=packet_dirs(), ids=packet_ids())
def packet(request):
    return request.param


def test_inventory_paths_stay_inside_packet(packet) -> None:
    escaping = [
        name
        for name in checksum_inventory(packet)
        if resolve_inventory_path(packet, name) is None
    ]
    assert not escaping, (
        f"{packet.name}: inventory entries are absolute or resolve outside "
        f"the packet root: {escaping}"
    )


def test_inventory_paths_resolve(packet) -> None:
    missing = []
    for name in checksum_inventory(packet):
        resolved = resolve_inventory_path(packet, name)
        if resolved is not None and not resolved.is_file():
            missing.append(name)
    assert not missing, f"{packet.name}: inventory files missing on disk: {missing}"


def test_unverified_hash_fields_are_declared_exceptions(packet) -> None:
    """Hash fields the checker skips must be declared, not silently ignored."""
    manifest = load_manifest(packet)
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict):
        return
    unresolvable = [
        name
        for name in hashes
        if (p := resolve_inventory_path(packet, name)) is None or not p.is_file()
    ]
    if unresolvable:
        assert packet.name in EXTERNAL_ARTIFACT_HASH_EXCEPTIONS, (
            f"{packet.name}: artifact_hashes entries {unresolvable[:5]} do not "
            "resolve inside the packet and the packet is not listed in "
            "EXTERNAL_ARTIFACT_HASH_EXCEPTIONS (tests/contract/packet_inventory.py)"
        )


def test_exception_allowlist_has_no_stale_entries() -> None:
    known = set(packet_ids())
    stale = set(EXTERNAL_ARTIFACT_HASH_EXCEPTIONS) - known
    assert not stale, (
        f"EXTERNAL_ARTIFACT_HASH_EXCEPTIONS lists unknown packets: {stale}"
    )


def test_inventory_checksums_match(packet) -> None:
    mismatched: list[str] = []
    for name, expected in checksum_inventory(packet).items():
        path = resolve_inventory_path(packet, name)
        if path is None or not path.is_file():
            continue  # covered by the containment / resolve tests above
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            mismatched.append(f"{name}: expected {expected[:12]}…, got {digest[:12]}…")
    assert not mismatched, f"{packet.name}: sealed artifacts changed: {mismatched}"
