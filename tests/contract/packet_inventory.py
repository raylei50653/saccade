"""Shared discovery/inventory helpers for research-packet contract tests.

A "packet" is a dated directory under docs/modules/semantic/research/evidence/
holding the sealed artifacts of one study. These helpers expose the subset of
structure that is common to every packet so the contract tests stay generic:
per-packet regression logic belongs to the study, not here.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = REPO / "docs" / "modules" / "semantic" / "research" / "evidence"

GENERIC_RESEARCH_PACKET = "generic_research_packet"
H0_PRESEAL_FREEZE_V3_ARTIFACT = "h0_preseal_freeze_v3_artifact"
H0_PRESEAL_FREEZE_V3_FILENAME = "h0_preseal_freeze_v3.json"

_DATED_PACKET_NAME = re.compile(r".+_\d{8}(T\d{6}Z)?$")
_H0_PRESEAL_FREEZE_V3_DIR_NAME = re.compile(r"^h0_preseal_freeze_[0-9a-f]{40}$")

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


def evidence_entries(evidence_root: Path = EVIDENCE_ROOT) -> list[Path]:
    """Return every top-level evidence-root entry without filtering by type."""
    if not evidence_root.is_dir() or evidence_root.is_symlink():
        return []
    return sorted(evidence_root.iterdir())


def evidence_kind(evidence_dir: Path) -> str | None:
    """Classify one evidence directory, returning None for an unknown kind.

    The exact H0 v3 governance-artifact family is intentionally separate from
    dated research packets.  Every other directory must either be a dated
    packet or be rejected by the schema contract; it must never disappear from
    generic validation merely because it lacks a manifest.
    """
    if _H0_PRESEAL_FREEZE_V3_DIR_NAME.fullmatch(evidence_dir.name):
        return H0_PRESEAL_FREEZE_V3_ARTIFACT
    if _DATED_PACKET_NAME.fullmatch(evidence_dir.name):
        return GENERIC_RESEARCH_PACKET
    return None


def _is_physical_directory(entry: Path) -> bool:
    return entry.is_dir() and not entry.is_symlink()


def evidence_entry_errors(entry: Path) -> list[str]:
    """Return fail-closed classification/type errors for an evidence entry."""
    kind = evidence_kind(entry)
    if kind is None:
        return [f"{entry.name}: unknown evidence entry kind"]
    if not _is_physical_directory(entry):
        return [f"{entry.name}: {kind} container must be a physical directory"]
    return []


def _physical_dirs_of_kind(kind: str) -> list[Path]:
    return [
        entry
        for entry in evidence_entries()
        if evidence_kind(entry) == kind and _is_physical_directory(entry)
    ]


def generic_packet_dirs() -> list[Path]:
    return _physical_dirs_of_kind(GENERIC_RESEARCH_PACKET)


def h0_preseal_freeze_v3_dirs() -> list[Path]:
    return _physical_dirs_of_kind(H0_PRESEAL_FREEZE_V3_ARTIFACT)


def h0_preseal_freeze_v3_layout_errors(evidence_dir: Path) -> list[str]:
    """Return structural errors for one H0 v3 governance artifact directory.

    This is deliberately a layout check only.  The dedicated H0 v3 verifier
    remains the authority for artifact contents, identity binding, and v3
    canonicality.
    """
    if not _is_physical_directory(evidence_dir):
        return [
            f"{evidence_dir.name}: governance artifact container must be a "
            "physical directory"
        ]

    names = {entry.name for entry in evidence_dir.iterdir()}
    expected = {H0_PRESEAL_FREEZE_V3_FILENAME}
    errors: list[str] = []
    if names != expected:
        errors.append(
            f"{evidence_dir.name}: expected only {sorted(expected)}, found {sorted(names)}"
        )

    artifact = evidence_dir / H0_PRESEAL_FREEZE_V3_FILENAME
    if not artifact.is_file() or artifact.is_symlink():
        errors.append(
            f"{evidence_dir.name}: {H0_PRESEAL_FREEZE_V3_FILENAME} "
            "must be a physical regular file"
        )
    return errors


def packet_dirs() -> list[Path]:
    """Return only dated generic research packets for manifest validation."""
    return generic_packet_dirs()


def resolve_inventory_path(packet: Path, name: str) -> Path | None:
    """Resolve an inventory entry safely, or None if it escapes the packet.

    Rejects absolute paths and any entry whose resolved location falls
    outside the packet root (e.g. `../`), so a manifest can never point the
    integrity check at a file it does not own.
    """
    candidate = Path(name)
    if candidate.is_absolute():
        return None
    resolved = (packet / candidate).resolve()
    if not resolved.is_relative_to(packet.resolve()):
        return None
    return resolved


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
