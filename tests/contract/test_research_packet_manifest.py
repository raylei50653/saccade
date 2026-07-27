"""Generic manifest/integrity contract for sealed research evidence packets.

Verifies that every checksum-inventory entry resolves to a real file inside
the packet and that its sha256 still matches — i.e. the sealed evidence has
not silently rotted or been edited without re-sealing.

The check is total in both directions. Verifying only the listed files would
be fail-open against *additive* contamination: an unlisted file dropped into a
packet leaves every listed checksum intact. So the inventory must also cover
every file physically present, and packets may contain no symlinks.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
import json

import pytest

from tests.contract.packet_inventory import (
    checksum_inventory,
    checksum_inventory_source,
    EXTERNAL_ARTIFACT_HASH_EXCEPTIONS,
    inventory_completeness_errors,
    load_manifest,
    packet_dirs,
    packet_ids,
    packet_physical_files,
    packet_symlinks,
    resolve_inventory_path,
    UNINVENTORIED_PACKET_FILES,
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


def test_inventory_is_total_over_packet_contents(packet) -> None:
    """Reverse direction: every file that exists must be inventoried.

    Verifying only the listed files is fail-open against additive
    contamination — an unlisted file can be dropped anywhere in the packet and
    every checksum still passes. This is the gate that catches it.
    """
    uncovered = inventory_completeness_errors(packet)
    assert not uncovered, (
        f"{packet.name}: files present in the packet but absent from "
        f"{checksum_inventory_source(packet)}: {uncovered}. Add them to the "
        "inventory; do not widen UNINVENTORIED_PACKET_FILES for new content."
    )


def test_packet_contains_no_symlinks(packet) -> None:
    """A symlink has no bytes of its own, so no inventory entry can bind it."""
    links = packet_symlinks(packet)
    assert not links, (
        f"{packet.name}: sealed packets must not contain symlinks: {links}"
    )


def test_uninventoried_exception_allowlist_has_no_stale_entries() -> None:
    """A declared gap must still be a real gap, and must not have grown."""
    known = {p.name: p for p in packet_dirs()}
    stale: dict[str, str] = {}
    for name, declared in UNINVENTORIED_PACKET_FILES.items():
        packet = known.get(name)
        if packet is None:
            stale[name] = "packet no longer exists"
            continue
        uncovered = packet_physical_files(packet) - set(checksum_inventory(packet))
        if resolved := set(declared) - uncovered:
            stale[name] = f"now inventoried or absent, drop from allowlist: {resolved}"
    assert not stale, f"stale UNINVENTORIED_PACKET_FILES entries: {stale}"


def _synthetic_packet(root, files: dict[str, str]):
    """Build a packet whose SHA256SUMS.json covers exactly `files`."""
    packet = root / "synthetic_20260728"
    packet.mkdir()
    rows = []
    for name, text in files.items():
        path = packet / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        rows.append({"file": name, "sha256": hashlib.sha256(text.encode()).hexdigest()})
    (packet / "SHA256SUMS.json").write_text(json.dumps({"files": rows}))
    return packet


def test_completeness_gate_accepts_a_fully_inventoried_packet(tmp_path) -> None:
    packet = _synthetic_packet(tmp_path, {"manifest.json": "{}", "note.md": "x"})

    assert inventory_completeness_errors(packet) == []


@pytest.mark.parametrize(
    "extra",
    ["extra.json", "nested/extra.json"],
    ids=["top-level", "nested"],
)
def test_completeness_gate_rejects_added_files_at_any_depth(tmp_path, extra) -> None:
    """The 2026-07-27 chain-of-custody incident, as a regression test."""
    packet = _synthetic_packet(tmp_path, {"manifest.json": "{}"})
    added = packet / extra
    added.parent.mkdir(parents=True, exist_ok=True)
    added.write_text("contamination")

    assert inventory_completeness_errors(packet) == [extra]


def test_completeness_gate_ignores_only_the_inventory_source(tmp_path) -> None:
    packet = _synthetic_packet(tmp_path, {"manifest.json": "{}"})

    assert checksum_inventory_source(packet) == "SHA256SUMS.json"
    assert "SHA256SUMS.json" not in packet_physical_files(packet)
    # A manifest-only packet cannot self-list either, but nothing else escapes.
    (packet / "SHA256SUMS.json").unlink()
    assert checksum_inventory_source(packet) == "manifest.json"
    assert "manifest.json" not in packet_physical_files(packet)


def test_symlinks_are_reported_and_never_counted_as_inventoried(tmp_path) -> None:
    packet = _synthetic_packet(tmp_path, {"manifest.json": "{}"})
    target = tmp_path / "outside.txt"
    target.write_text("smuggled")
    (packet / "link.txt").symlink_to(target)

    assert packet_symlinks(packet) == ["link.txt"]
    assert "link.txt" not in packet_physical_files(packet)


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
