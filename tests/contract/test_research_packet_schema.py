"""Generic schema contract for sealed research evidence packets.

These tests are the permanent replacement for per-packet pytest files: once a
study is sealed, its packet is preserved by artifacts plus this generic
validation, not by packet-specific regression tests (see
tests/research/README.md).
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
import pathlib
import re
import sys

import pytest

from tests.contract.packet_inventory import (
    EVIDENCE_ROOT,
    GENERIC_RESEARCH_PACKET,
    H0_PHASE_A_EXECUTION_PACKET,
    H0_PRESEAL_FREEZE_V3_ARTIFACT,
    H0_PRESEAL_FREEZE_V3_FILENAME,
    H2_MEASUREMENT_EXECUTION_PACKET,
    checksum_inventory,
    evidence_entries,
    evidence_entry_errors,
    evidence_kind,
    h0_phase_a_execution_dirs,
    h0_preseal_freeze_v3_dirs,
    h0_preseal_freeze_v3_layout_errors,
    h2_archive_integrity_errors,
    h2_measurement_execution_dirs,
    is_generic_dated_packet_name,
    is_h0_phase_a_execution_name,
    is_h0_preseal_freeze_v3_name,
    is_h2_measurement_execution_name,
    load_manifest,
    packet_dirs,
    packet_ids,
    REPO,
)


TOOLS = REPO / "scripts" / "tools"
sys.path.insert(0, TOOLS.as_posix())

import h2_measurement_evidence as h2_evidence  # noqa: E402
import verify_h0_phase_a as phase_a_verifier  # noqa: E402
import verify_h2_measurement as measurement_verifier  # noqa: E402


def test_evidence_root_exists() -> None:
    # Deliberately NOT a skip: losing or renaming the evidence root must
    # fail CI, otherwise the whole packet contract is fail-open.
    assert EVIDENCE_ROOT.is_dir() and not EVIDENCE_ROOT.is_symlink(), (
        f"evidence root missing or not a physical directory: {EVIDENCE_ROOT}"
    )


def test_packet_set_nonempty() -> None:
    assert packet_dirs(), (
        f"no sealed packets found under {EVIDENCE_ROOT}; an empty packet set "
        "would silently skip every parameterized contract check"
    )


def test_all_evidence_root_entries_are_classified_and_physical() -> None:
    violations: dict[str, list[str]] = {}
    for entry in evidence_entries():
        errors = evidence_entry_errors(entry)
        if errors:
            violations[entry.name] = errors
    assert not violations, f"invalid evidence-root entries: {violations}"


def test_exact_h0_preseal_freeze_v3_dirs_are_governance_artifacts() -> None:
    violations: dict[str, list[str]] = {}
    for evidence_dir in h0_preseal_freeze_v3_dirs():
        errors = h0_preseal_freeze_v3_layout_errors(evidence_dir)
        if errors:
            violations[evidence_dir.name] = errors
    assert not violations, f"non-canonical H0 v3 governance artifacts: {violations}"


def test_h0_phase_a_execution_dirs_are_verified_by_the_dedicated_verifier() -> None:
    evidence_dirs = h0_phase_a_execution_dirs()
    assert evidence_dirs, "no H0 Phase-A execution evidence directories were found"
    reports: dict[str, object] = {}
    failures: dict[str, str] = {}
    for evidence_dir in evidence_dirs:
        try:
            reports[evidence_dir.name] = phase_a_verifier.verify_evidence_root(
                evidence_dir
            )
        except phase_a_verifier.VerificationError as exc:
            failures[evidence_dir.name] = str(exc)
    assert not failures, f"invalid H0 Phase-A execution evidence: {failures}"
    assert all(
        isinstance(report, dict) and report.get("valid") is True
        for report in reports.values()
    )


def test_h0_phase_a_archive_verification_is_execution_host_independent(
    monkeypatch,
) -> None:
    """Archive verification must never read the execution host's payloads.

    CI runners have no /opt/cuda, no build tree, and no dataset/model/venv
    payloads, so every byte the verifier consumes must come from the packet
    itself or repository code.
    """
    evidence_dirs = h0_phase_a_execution_dirs()
    assert evidence_dirs, "no H0 Phase-A execution evidence directories were found"
    repository = REPO.resolve()
    absent_on_ci = tuple(
        repository / name for name in ("build", ".venv", "datasets", "models", "runs")
    )
    real_read_bytes = pathlib.Path.read_bytes

    def guarded_read_bytes(self: pathlib.Path) -> bytes:
        absolute = pathlib.Path(self).absolute()
        if not absolute.is_relative_to(repository) or any(
            absolute.is_relative_to(prefix) for prefix in absent_on_ci
        ):
            raise FileNotFoundError(
                f"host-coupled read during archive verification: {self}"
            )
        return real_read_bytes(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", guarded_read_bytes)
    for evidence_dir in evidence_dirs:
        report = phase_a_verifier.verify_evidence_root(evidence_dir)
        assert isinstance(report, dict) and report.get("valid") is True


@pytest.mark.parametrize(
    "name",
    [
        "h2_measure_" + "a" * 40,
        "h2_measure_b_" + "a" * 40 + "_" + "b" * 64,
        "h2_measure_" + "a" * 39,
        "h2_measure_" + "A" * 40,
        "h2_measure_b_" + "a" * 40,
        "h2_measure_b_" + "a" * 40 + "_" + "b" * 63,
        "h2_measure_",
        "h2_phase_a_" + "a" * 40,
        "study_20260728",
    ],
)
def test_h2_family_grammar_agrees_with_the_producer(name: str) -> None:
    """The classifier's copy of the root-name grammar is bound to the ruler.

    `packet_inventory` re-types the H2 root-name pattern so the contract layer
    stays importable on its own. C3.9's trap applies: a re-typed grammar can
    drift while the producer stands still, so agreement with
    `h2_measurement_evidence.parse_root_name` is asserted, not remembered.
    """
    try:
        h2_evidence.parse_root_name(name)
    except h2_evidence.EvidenceError:
        producer_accepts = False
    else:
        producer_accepts = True

    assert is_h2_measurement_execution_name(name) == producer_accepts


def test_h2_measurement_archives_are_inventory_complete_and_unrotted() -> None:
    """Committed H2 Layer-M archives keep their own inventory total and exact.

    This is the host-independent half of archive validation: every file present
    is named by `checksums.sha256`, every named file exists, and every digest
    still matches. It catches rot, silent edits and additive contamination
    anywhere a reviewer or CI runs.

    It is deliberately *not* the full verifier. `verify_h2_measurement`
    recomputes the authorization execution domain from the verifying host's
    `/etc/machine-id` and `os.getuid()` and requires equality with the archived
    record, so a Phase-A archive currently verifies only on the machine that
    produced it — see the 2026-07-28 registration packet, §4.2. Binding the
    *grant* to one host is intended and correct at launch; carrying that live
    recomputation into archive verification is not, and until it is repaired a
    cross-host gate here could only be satisfied by weakening it.
    """
    evidence_dirs = h2_measurement_execution_dirs()
    assert evidence_dirs, "no H2 Layer-M measurement evidence directories were found"
    failures = {
        evidence_dir.name: errors
        for evidence_dir in evidence_dirs
        if (errors := h2_archive_integrity_errors(evidence_dir))
    }
    assert not failures, f"H2 Layer-M archive integrity failures: {failures}"


def _h2_archive(root, files: dict[str, bytes], *, extra_lines: tuple[str, ...] = ()):
    """Build an H2 archive whose inventory covers exactly `files`, plus lines."""
    archive = root / ("h2_measure_" + "a" * 40)
    archive.mkdir()
    rows = []
    for name, payload in files.items():
        path = archive / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        rows.append(f"{hashlib.sha256(payload).hexdigest()}  {name}")
    (archive / "checksums.sha256").write_text(
        "\n".join(rows + list(extra_lines)) + "\n", encoding="utf-8"
    )
    return archive


def test_h2_archive_integrity_accepts_an_exactly_inventoried_archive(tmp_path) -> None:
    archive = _h2_archive(tmp_path, {"terminal.json": b"{}", "runs/a/stderr.log": b"x"})

    assert h2_archive_integrity_errors(archive) == []


@pytest.mark.parametrize(
    ("extra_line", "expected"),
    [
        pytest.param(
            f"{'b' * 64}  terminal.json",
            "duplicate inventory entry",
            id="duplicate-path",
        ),
        pytest.param("not an inventory line", "malformed inventory line", id="garbage"),
        pytest.param(f"{'B' * 64}  other.json", "malformed inventory line", id="upper"),
        pytest.param(f"{'b' * 63}  other.json", "malformed inventory line", id="short"),
        pytest.param(
            f"{'b' * 64} other.json", "malformed inventory line", id="1-space"
        ),
        pytest.param("", "blank line in the inventory", id="blank"),
        pytest.param(
            f"{'b' * 64}  /etc/passwd", "non-canonical inventory path", id="absolute"
        ),
        pytest.param(
            f"{'b' * 64}  ../outside.json", "non-canonical inventory path", id="parent"
        ),
        pytest.param(
            f"{'b' * 64}  ./terminal.json", "non-canonical inventory path", id="dot"
        ),
        pytest.param(
            f"{'b' * 64}  runs\\a.json", "non-canonical inventory path", id="backslash"
        ),
        pytest.param(
            f"{'b' * 64}  checksums.sha256",
            "the inventory may not list itself",
            id="self",
        ),
    ],
)
def test_h2_archive_integrity_rejects_ambiguous_inventory_lines(
    tmp_path, extra_line: str, expected: str
) -> None:
    """An inventory line this checker cannot read is one it cannot enforce.

    The duplicate case is the load-bearing one: a dict assignment would let a
    second entry for the same path overwrite the first, so any digest at all
    could be smuggled in behind a correct-looking final line.
    """
    archive = _h2_archive(tmp_path, {"terminal.json": b"{}"}, extra_lines=(extra_line,))

    errors = h2_archive_integrity_errors(archive)
    assert any(expected in error for error in errors), errors


def test_h2_archive_integrity_catches_rot_addition_and_removal(tmp_path) -> None:
    archive = _h2_archive(tmp_path, {"terminal.json": b"{}", "observation.json": b"{}"})
    (archive / "terminal.json").write_bytes(b"{tampered}")
    (archive / "observation.json").unlink()
    (archive / "smuggled.json").write_bytes(b"{}")

    assert sorted(h2_archive_integrity_errors(archive)) == [
        "digest changed: terminal.json",
        "inventoried but absent: observation.json",
        "present but uninventoried: smuggled.json",
    ]


def test_h2_archive_integrity_reports_symlinks(tmp_path) -> None:
    archive = _h2_archive(tmp_path, {"terminal.json": b"{}"})
    target = tmp_path / "outside.json"
    target.write_text("{}")
    (archive / "link.json").symlink_to(target)

    assert h2_archive_integrity_errors(archive) == [
        "archive contains a symlink: link.json"
    ]


@pytest.mark.parametrize(
    "entry_name", [None, "unexpected.json"], ids=["empty", "extra"]
)
def test_h2_measurement_execution_layout_is_rejected_by_dedicated_verifier(
    tmp_path, entry_name: str | None
) -> None:
    evidence_dir = tmp_path / ("h2_measure_" + "a" * 40)
    evidence_dir.mkdir()
    if entry_name is not None:
        (evidence_dir / entry_name).write_text("not an archive")

    assert evidence_kind(evidence_dir) == H2_MEASUREMENT_EXECUTION_PACKET
    with pytest.raises(measurement_verifier.VerificationError):
        measurement_verifier.verify_evidence_root(evidence_dir)


@pytest.mark.parametrize(
    "entry_name", [None, "unexpected.json"], ids=["empty", "extra"]
)
def test_h0_phase_a_execution_layout_is_rejected_by_dedicated_verifier(
    tmp_path, entry_name: str | None
) -> None:
    evidence_dir = tmp_path / ("h0_phase_a_" + "a" * 40)
    evidence_dir.mkdir()
    if entry_name is not None:
        (evidence_dir / entry_name).write_text("not a packet")

    assert evidence_kind(evidence_dir) == H0_PHASE_A_EXECUTION_PACKET
    with pytest.raises(phase_a_verifier.VerificationError):
        phase_a_verifier.verify_evidence_root(evidence_dir)


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
        ("h0_phase_a_" + "a" * 40, H0_PHASE_A_EXECUTION_PACKET),
        ("h2_measure_" + "a" * 40, H2_MEASUREMENT_EXECUTION_PACKET),
        ("h0_preseal_freeze_" + "a" * 39, None),
        ("h0_preseal_freeze_" + "a" * 41, None),
        ("h0_preseal_freeze_" + "A" * 40, None),
        ("h0_phase_a_" + "a" * 39, None),
        ("h0_phase_a_" + "a" * 41, None),
        ("h0_phase_a_" + "A" * 40, None),
        (
            "h2_measure_b_" + "a" * 40 + "_" + "b" * 64,
            H2_MEASUREMENT_EXECUTION_PACKET,
        ),
        ("h2_measure_" + "a" * 39, None),
        ("h2_measure_" + "a" * 41, None),
        ("h2_measure_" + "A" * 40, None),
        ("h2_measure_b_" + "a" * 40, None),
        ("h2_measure_b_" + "a" * 40 + "_" + "b" * 63, None),
        ("unclassified_evidence", None),
        ("study_٠١٢٣٤٥٦٧", None),
    ],
)
def test_evidence_kind_is_explicit_and_fail_closed(
    tmp_path, name: str, expected_kind: str | None
) -> None:
    assert evidence_kind(tmp_path / name) == expected_kind


@pytest.mark.parametrize(
    ("prefix", "is_h0_name", "expected_kind"),
    [
        (
            "h0_preseal_freeze_",
            is_h0_preseal_freeze_v3_name,
            H0_PRESEAL_FREEZE_V3_ARTIFACT,
        ),
        ("h0_phase_a_", is_h0_phase_a_execution_name, H0_PHASE_A_EXECUTION_PACKET),
    ],
)
def test_h0_and_dated_name_grammars_are_disjoint(
    tmp_path, prefix, is_h0_name, expected_kind
) -> None:
    # The hardest case for disjointness: an H0 sha whose last 8 hex chars are
    # all decimal digits still has no `_` before them, so the dated grammar
    # (`.+_[0-9]{8}$`) must not match and classification cannot depend on
    # check order in evidence_kind.
    name = prefix + "a" * 32 + "12345678"

    assert is_h0_name(name)
    assert not is_generic_dated_packet_name(name)
    assert evidence_kind(tmp_path / name) == expected_kind


def test_evidence_entries_include_all_top_level_entry_types(tmp_path) -> None:
    (tmp_path / "study_20260717").mkdir()
    (tmp_path / "not_a_directory").write_text("not a directory")

    assert {entry.name for entry in evidence_entries(tmp_path)} == {
        "not_a_directory",
        "study_20260717",
    }


@pytest.mark.parametrize(
    "prefix",
    ["h0_preseal_freeze_", "h0_phase_a_"],
    ids=["preseal", "phase-a"],
)
@pytest.mark.parametrize(
    "target_kind",
    ["directory", "file", "missing"],
    ids=["directory-symlink", "file-symlink", "broken-symlink"],
)
def test_exact_h0_entry_symlinks_are_not_dropped_and_are_rejected(
    tmp_path, prefix: str, target_kind: str
) -> None:
    entry = tmp_path / (prefix + "a" * 40)
    target = tmp_path / "target"
    if target_kind == "directory":
        target.mkdir()
    elif target_kind == "file":
        target.write_text("not a directory")
    else:
        target = tmp_path / "missing"
    entry.symlink_to(target, target_is_directory=target_kind == "directory")

    assert entry in evidence_entries(tmp_path)
    assert evidence_entry_errors(entry)


@pytest.mark.parametrize(
    "prefix",
    ["h0_preseal_freeze_", "h0_phase_a_"],
    ids=["preseal", "phase-a"],
)
def test_exact_h0_regular_file_is_not_dropped_and_is_rejected(tmp_path, prefix) -> None:
    entry = tmp_path / (prefix + "a" * 40)
    entry.write_text("not a directory")

    assert entry in evidence_entries(tmp_path)
    assert evidence_entry_errors(entry)


def test_generic_packet_container_must_be_a_physical_directory(tmp_path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    entry = tmp_path / "study_20260717"
    entry.symlink_to(target, target_is_directory=True)

    assert evidence_kind(entry) == GENERIC_RESEARCH_PACKET
    assert evidence_entry_errors(entry)


def test_h0_preseal_freeze_v3_layout_accepts_only_canonical_artifact(tmp_path) -> None:
    evidence_dir = tmp_path / ("h0_preseal_freeze_" + "a" * 40)
    evidence_dir.mkdir()
    (evidence_dir / H0_PRESEAL_FREEZE_V3_FILENAME).write_text("{}")

    assert h0_preseal_freeze_v3_layout_errors(evidence_dir) == []


def test_h0_preseal_freeze_v3_layout_rejects_artifact_file_symlink(tmp_path) -> None:
    evidence_dir = tmp_path / ("h0_preseal_freeze_" + "a" * 40)
    evidence_dir.mkdir()
    target = tmp_path / "artifact-target.json"
    target.write_text("{}")
    (evidence_dir / H0_PRESEAL_FREEZE_V3_FILENAME).symlink_to(target)

    assert h0_preseal_freeze_v3_layout_errors(evidence_dir)


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
