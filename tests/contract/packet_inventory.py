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
H0_PHASE_A_EXECUTION_PACKET = "h0_phase_a_execution_packet"

# ASCII digits only: `\d` would also accept Unicode digits (e.g. ٠١٢٣),
# which must stay unclassified and be rejected fail-closed.
_DATED_PACKET_NAME = re.compile(r".+_[0-9]{8}(T[0-9]{6}Z)?$")
_H0_PRESEAL_FREEZE_V3_DIR_NAME = re.compile(r"^h0_preseal_freeze_[0-9a-f]{40}$")
_H0_PHASE_A_EXECUTION_DIR_NAME = re.compile(r"^h0_phase_a_[0-9a-f]{40}$")

# manifest.json keys observed to map filename -> sha256 hex digest.
# `artifact_hashes` is deliberately absent: its keys are logical artifact
# names (possibly stored outside the packet) and are not resolvable here.
_INVENTORY_KEYS = ("files", "artifacts", "artifact_sha256")

# Compiled bytecode: gitignored interpreter output, never packet content.
# Matched by suffix so that a non-bytecode file cannot hide inside a
# `__pycache__/` directory.
_INTERPRETER_OUTPUT_SUFFIXES = frozenset({".pyc", ".pyo"})

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


# Disclosed legacy integrity gaps: files that physically exist in a sealed
# packet but are not named by its checksum inventory. Each entry is a frozen,
# exhaustive list, so anything *newly* added to these packets still fails — a
# legacy gap never becomes a standing licence to add files. Sealed packets are
# not re-hashed to close these; new packets must list everything.
#
# Scope, stated plainly: this table freezes the *set of paths* that escape the
# inventory. It stores no digests, so the bytes of these seven files are NOT
# bound by the generic corpus contract — an in-place edit to one of them passes
# CI. Closing that needs a separate path->digest post-hoc audit table, which
# can be added without touching the sealed packets.
UNINVENTORIED_PACKET_FILES: dict[str, tuple[str, ...]] = {
    # The runner's digest is declared under the logical manifest key
    # `runner_sha256`, which is not in _INVENTORY_KEYS and so is not checked
    # generically; verified equal to the file on disk by hand when this
    # exception was recorded (2026-07-28).
    "gap_conditioned_motion_e0_20260711": ("run_e0_audit.py",),
    "gap_conditioned_motion_e1_m0_20260711": ("run_e1_m0.py",),
    "gap_conditioned_motion_e2_family_20260711": ("run_e2_family_freeze.py",),
    "gap_conditioned_motion_e3_signals_20260711": ("run_e3_signals.py",),
    # Sidecar holding the sha256 of the inventoried fixture_pack.json; it
    # repeats a covered digest and adds no uncovered content.
    "gctm_d1_substrate_agnostic_ranking_20260723": ("fixture_pack.json.sha256",),
    # Sealed 2026-07-10 with a metadata sidecar outside SHA256SUMS.json.
    "m_b1_5_stage2_q45_20260710": ("README.json",),
    "m_b1_5_t0_region_interpretation_20260710": ("README.json",),
}


def evidence_entries(evidence_root: Path = EVIDENCE_ROOT) -> list[Path]:
    """Return every top-level evidence-root entry without filtering by type."""
    if not evidence_root.is_dir() or evidence_root.is_symlink():
        return []
    return sorted(evidence_root.iterdir())


def is_h0_preseal_freeze_v3_name(name: str) -> bool:
    return _H0_PRESEAL_FREEZE_V3_DIR_NAME.fullmatch(name) is not None


def is_h0_phase_a_execution_name(name: str) -> bool:
    return _H0_PHASE_A_EXECUTION_DIR_NAME.fullmatch(name) is not None


def is_generic_dated_packet_name(name: str) -> bool:
    return _DATED_PACKET_NAME.fullmatch(name) is not None


def evidence_kind(evidence_dir: Path) -> str | None:
    """Classify one evidence directory, returning None for an unknown kind.

    Exact H0 governance and execution-evidence families are intentionally
    separate from dated research packets.  Their final 40-hex component has no
    underscore, so neither can end in the dated ``_[0-9]{8}`` grammar and
    classification does not depend on check order.  Every other directory must
    either be a dated packet or be rejected by the schema contract; it must
    never disappear from generic validation merely because it lacks a manifest.
    """
    if is_h0_preseal_freeze_v3_name(evidence_dir.name):
        return H0_PRESEAL_FREEZE_V3_ARTIFACT
    if is_h0_phase_a_execution_name(evidence_dir.name):
        return H0_PHASE_A_EXECUTION_PACKET
    if is_generic_dated_packet_name(evidence_dir.name):
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


def h0_phase_a_execution_dirs() -> list[Path]:
    return _physical_dirs_of_kind(H0_PHASE_A_EXECUTION_PACKET)


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


def checksum_inventory_source(packet: Path) -> str:
    """Return the filename the packet's inventory is read from.

    A file cannot list its own digest, so this one name is excluded from the
    completeness comparison below; every other file must be inventoried.
    """
    if (packet / "SHA256SUMS.json").is_file():
        return "SHA256SUMS.json"
    return "manifest.json"


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


def packet_symlinks(packet: Path) -> list[str]:
    """Return packet-relative paths of symlinks anywhere inside a packet.

    A symlink has no content of its own to hash, so it can carry bytes into a
    sealed packet that the inventory never covers.
    """
    return sorted(
        entry.relative_to(packet).as_posix()
        for entry in packet.rglob("*")
        if entry.is_symlink()
    )


def packet_physical_files(packet: Path) -> set[str]:
    """Return packet-relative paths of every real file inside a packet.

    Recursive, so a file added in a subdirectory is as visible as one added at
    the top level. Excluded: the inventory source itself (it cannot list its
    own digest), symlinks (reported separately by `packet_symlinks`), and
    compiled bytecode, which is gitignored interpreter output.

    The bytecode exclusion is by file *suffix*, not by directory: excluding
    everything under `__pycache__/` would leave an additive blind spot, since
    an arbitrary file such as `__pycache__/extra.json` would then never be
    enumerated.
    """
    source = checksum_inventory_source(packet)
    return {
        entry.relative_to(packet).as_posix()
        for entry in packet.rglob("*")
        if entry.is_file()
        and not entry.is_symlink()
        and entry.suffix not in _INTERPRETER_OUTPUT_SUFFIXES
        and entry.relative_to(packet).as_posix() != source
    }


def inventory_completeness_errors(packet: Path) -> list[str]:
    """Return files present in a packet that its inventory does not cover.

    Checksums alone are fail-open against *additive* contamination: every
    listed file can still verify while an unlisted file sits beside it. This
    is the reverse direction — enumerate what is physically there and require
    the inventory to be total over it.
    """
    uncovered = packet_physical_files(packet) - set(checksum_inventory(packet))
    declared = UNINVENTORIED_PACKET_FILES.get(packet.name, ())
    return sorted(uncovered - set(declared))
