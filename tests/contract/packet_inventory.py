"""Shared discovery/inventory helpers for research-packet contract tests.

A "packet" is a dated directory under docs/modules/semantic/research/evidence/
holding the sealed artifacts of one study. These helpers expose the subset of
structure that is common to every packet so the contract tests stay generic:
per-packet regression logic belongs to the study, not here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

REPO = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = REPO / "docs" / "modules" / "semantic" / "research" / "evidence"

GENERIC_RESEARCH_PACKET = "generic_research_packet"
H0_PRESEAL_FREEZE_V3_ARTIFACT = "h0_preseal_freeze_v3_artifact"
H0_PRESEAL_FREEZE_V3_FILENAME = "h0_preseal_freeze_v3.json"
H0_PHASE_A_EXECUTION_PACKET = "h0_phase_a_execution_packet"
H2_MEASUREMENT_EXECUTION_PACKET = "h2_measurement_execution_packet"
H2_MEASUREMENT_ENVELOPE_PACKET = "h2_measurement_envelope_packet"

# ASCII digits only: `\d` would also accept Unicode digits (e.g. ٠١٢٣),
# which must stay unclassified and be rejected fail-closed.
_DATED_PACKET_NAME = re.compile(r".+_[0-9]{8}(T[0-9]{6}Z)?$")
_H0_PRESEAL_FREEZE_V3_DIR_NAME = re.compile(r"^h0_preseal_freeze_[0-9a-f]{40}$")
_H0_PHASE_A_EXECUTION_DIR_NAME = re.compile(r"^h0_phase_a_[0-9a-f]{40}$")
# Both H2 phases, so a Phase-B root can never be silently unclassified. The
# grammar is owned by h2_measurement_evidence.parse_root_name; the schema
# contract asserts this pattern agrees with it rather than trusting the copy.
_H2_MEASUREMENT_EXECUTION_DIR_NAME = re.compile(
    r"^h2_measure_(b_[0-9a-f]{40}_[0-9a-f]{64}|[0-9a-f]{40})$"
)
# The v2 successor envelope is a *packet*: `archive/` and `runs/` beside the
# authorization records, which the flat-archive family above rejects outright.
# The canonical corpus discovers it by artifact family and treats root names as
# audit metadata (Correction 5), so no producer owns this grammar; it exists
# only so the repository-side taxonomy has a class for it. It ends in the
# instrumentation head's 40 hex characters, so like every other exact family it
# cannot end in the dated `_[0-9]{8}` grammar and classification stays
# order-independent. Membership buys stricter validation, not less: these roots
# are held to their own inventory *and* to the dedicated envelope verifier.
_H2_MEASUREMENT_ENVELOPE_DIR_NAME = re.compile(r"^h2_measure_envelope_[0-9a-f]{40}$")

# An H2 archive carries its own inventory in H0's format: exactly
# `<64 lowercase hex><two spaces><relative posix path>`, which is what
# `h2_measurement_evidence` writes (`:496`). Parsing is fail-closed on the whole
# line rather than lenient on the digest, because a line this checker cannot
# read is a line it cannot enforce.
H2_ARCHIVE_INVENTORY_NAME = "checksums.sha256"
_SHA256SUM_LINE = re.compile(r"^(?P<digest>[0-9a-f]{64})  (?P<path>[^\s].*)$")

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


def is_h2_measurement_execution_name(name: str) -> bool:
    return _H2_MEASUREMENT_EXECUTION_DIR_NAME.fullmatch(name) is not None


def is_h2_measurement_envelope_name(name: str) -> bool:
    return _H2_MEASUREMENT_ENVELOPE_DIR_NAME.fullmatch(name) is not None


def is_generic_dated_packet_name(name: str) -> bool:
    return _DATED_PACKET_NAME.fullmatch(name) is not None


def evidence_kind(evidence_dir: Path) -> str | None:
    """Classify one evidence directory, returning None for an unknown kind.

    Exact H0 governance and execution-evidence families, and both H2 families
    (the Layer-M flat archive and the v2 successor envelope packet), are
    intentionally separate from dated research packets.  Their final 40-hex
    component has no underscore, so none of them can end in the dated
    ``_[0-9]{8}`` grammar and classification does not depend on check order.
    Every other directory must either be a dated packet or be rejected by the
    schema contract; it must never disappear from generic validation merely
    because it lacks a manifest.

    A family is a *stricter* class, never an exemption: each one is held to its
    own inventory and to its own dedicated verifier, both of which check more
    than the generic manifest contract does.  Adding a family to escape a
    failing generic check, rather than to route evidence to the verifier that
    actually owns it, would invert that and is the failure this docstring has
    always warned about.
    """
    if is_h0_preseal_freeze_v3_name(evidence_dir.name):
        return H0_PRESEAL_FREEZE_V3_ARTIFACT
    if is_h0_phase_a_execution_name(evidence_dir.name):
        return H0_PHASE_A_EXECUTION_PACKET
    if is_h2_measurement_execution_name(evidence_dir.name):
        return H2_MEASUREMENT_EXECUTION_PACKET
    if is_h2_measurement_envelope_name(evidence_dir.name):
        return H2_MEASUREMENT_ENVELOPE_PACKET
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


def h2_measurement_execution_dirs() -> list[Path]:
    return _physical_dirs_of_kind(H2_MEASUREMENT_EXECUTION_PACKET)


def h2_measurement_envelope_dirs() -> list[Path]:
    return _physical_dirs_of_kind(H2_MEASUREMENT_ENVELOPE_PACKET)


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


def parse_h2_archive_inventory(text: str) -> tuple[dict[str, str], list[str]]:
    """Parse an H2 archive's `checksums.sha256` fail-closed.

    Returns `(inventory, errors)`. Everything that could make the mapping
    ambiguous is an error rather than a silent resolution:

    * a **duplicate path** would otherwise be overwritten by whichever line came
      last, so a second entry for the same file could carry any digest at all
      and the inventory would still "verify";
    * a **malformed line** cannot be enforced, so it must not be skipped;
    * an **absolute path, a `..` component, a `./` prefix or a backslash** would
      let an entry name something other than the file it appears to name;
    * the inventory **cannot list itself** — a self-entry can never hold its own
      digest, so accepting one only creates a permanently unsatisfiable row.
    """
    inventory: dict[str, str] = {}
    errors: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            errors.append(f"line {number}: blank line in the inventory")
            continue
        match = _SHA256SUM_LINE.fullmatch(line)
        if match is None:
            errors.append(f"line {number}: malformed inventory line: {line!r}")
            continue
        path = match.group("path")
        parts = PurePosixPath(path).parts
        if (
            path != path.strip()
            or path.startswith("/")
            or path.startswith("./")
            or "\\" in path
            or ".." in parts
            or "." in parts
        ):
            errors.append(f"line {number}: non-canonical inventory path: {path!r}")
            continue
        if path == H2_ARCHIVE_INVENTORY_NAME:
            errors.append(f"line {number}: the inventory may not list itself")
            continue
        if path in inventory:
            errors.append(f"line {number}: duplicate inventory entry: {path!r}")
            continue
        inventory[path] = match.group("digest")
    return inventory, errors


def h2_archive_integrity_errors(archive: Path) -> list[str]:
    """Return every integrity failure of one H2 measurement archive.

    Host- and history-independent by construction: it reads only the archive's
    own bytes. Bidirectional — an inventoried file that vanished and an
    uninventoried file that appeared are both failures — and exact, since every
    digest is recomputed rather than compared to a stored summary.
    """
    inventory_path = archive / H2_ARCHIVE_INVENTORY_NAME
    if not inventory_path.is_file() or inventory_path.is_symlink():
        return [f"{H2_ARCHIVE_INVENTORY_NAME} is missing or not a regular file"]

    inventory, errors = parse_h2_archive_inventory(
        inventory_path.read_text(encoding="utf-8")
    )
    present: set[str] = set()
    for entry in sorted(archive.rglob("*")):
        relative = entry.relative_to(archive).as_posix()
        if entry.is_symlink():
            errors.append(f"archive contains a symlink: {relative}")
        elif entry.is_file() and relative != H2_ARCHIVE_INVENTORY_NAME:
            present.add(relative)

    errors += [
        f"present but uninventoried: {name}"
        for name in sorted(present - set(inventory))
    ]
    errors += [
        f"inventoried but absent: {name}" for name in sorted(set(inventory) - present)
    ]
    for name in sorted(set(inventory) & present):
        actual = hashlib.sha256((archive / name).read_bytes()).hexdigest()
        if actual != inventory[name]:
            errors.append(f"digest changed: {name}")
    return errors
