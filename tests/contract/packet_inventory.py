"""Shared discovery/inventory helpers for research-packet contract tests.

A "packet" is a dated directory under docs/modules/semantic/research/evidence/
holding the sealed artifacts of one study. These helpers expose the subset of
structure that is common to every packet so the contract tests stay generic:
per-packet regression logic belongs to the study, not here.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = REPO / "docs" / "modules" / "semantic" / "research" / "evidence"

# manifest.json keys observed to map filename -> sha256 hex digest.
# `artifact_hashes` is deliberately absent: its keys are logical artifact
# names (possibly stored outside the packet) and are not resolvable here.
_INVENTORY_KEYS = ("files", "artifacts", "artifact_sha256")

# Audited exceptions: packets whose manifest carries hash fields the generic
# checker intentionally does NOT verify. A packet with unresolvable
# `artifact_hashes` entries must be listed here with a reason, so an
# unverified hash is a declared decision, never a checker blind spot.
EXTERNAL_ARTIFACT_HASH_EXCEPTIONS: dict[str, str] = {
    "m_b1_5_stage2_q45_20260710": (
        "artifact_hashes uses logical names for artifacts stored outside the "
        "repository (out/…); packet files are verified via SHA256SUMS.json"
    ),
}


def packet_dirs() -> list[Path]:
    if not EVIDENCE_ROOT.is_dir():
        return []
    return sorted(p for p in EVIDENCE_ROOT.iterdir() if p.is_dir())


def packet_ids() -> list[str]:
    return [p.name for p in packet_dirs()]


def load_manifest(packet: Path) -> dict:
    return json.loads((packet / "manifest.json").read_text())


def checksum_inventory(packet: Path) -> dict[str, str]:
    """Return {relative filename: expected sha256} for a packet.

    Prefers SHA256SUMS.json (authoritative full-directory listing) and falls
    back to the manifest's filename->sha mapping fields.
    """
    sums = packet / "SHA256SUMS.json"
    if sums.is_file():
        data = json.loads(sums.read_text())
        return {row["file"]: row["sha256"] for row in data["files"]}
    manifest = load_manifest(packet)
    inventory: dict[str, str] = {}
    for key in _INVENTORY_KEYS:
        value = manifest.get(key)
        if isinstance(value, dict):
            inventory.update(
                {name: sha for name, sha in value.items() if isinstance(sha, str)}
            )
    return inventory
