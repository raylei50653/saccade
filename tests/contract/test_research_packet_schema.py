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
import subprocess
import sys

import pytest

from tests.contract.packet_inventory import (
    EVIDENCE_ROOT,
    GENERIC_RESEARCH_PACKET,
    H0_PHASE_A_EXECUTION_PACKET,
    H0_PRESEAL_FREEZE_V3_ARTIFACT,
    H0_PRESEAL_FREEZE_V3_FILENAME,
    H2_MEASUREMENT_ENVELOPE_PACKET,
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
    is_h2_measurement_envelope_name,
    is_h2_measurement_execution_name,
    load_manifest,
    packet_dirs,
    packet_ids,
    REPO,
)


TOOLS = REPO / "scripts" / "tools"
sys.path.insert(0, TOOLS.as_posix())

import check_h2_measure_archives as measurement_corpus  # noqa: E402
import h2_measurement_evidence as h2_evidence  # noqa: E402
import verify_h0_phase_a as phase_a_verifier  # noqa: E402
import verify_h2_measurement as measurement_verifier  # noqa: E402
import verify_h2_measurement_envelope as envelope_verifier  # noqa: E402


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

    It is deliberately *not* the full verifier, which additionally needs the
    archived head's git history. The host coupling that once kept the full
    verifier out of reach here — recomputing the authorization execution domain
    from the verifying host's `/etc/machine-id` and `os.getuid()`, registered in
    the 2026-07-28 packet §4.2 — is repaired, and
    `test_h2_phase_a_archive_verification_is_execution_host_independent` holds
    it repaired.
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
def test_h2_measurement_envelope_layout_is_rejected_by_dedicated_verifier(
    tmp_path, entry_name: str | None
) -> None:
    """Envelope membership must route to a stricter owner, not out of checking.

    An empty directory and a directory holding one stray file both carry the
    family name, so nothing but the dedicated verifier stands between them and
    the corpus.  Asserting it refuses them is what makes this class stricter
    than the generic manifest contract rather than an exemption from it.
    """
    evidence_dir = tmp_path / ("h2_measure_envelope_" + "a" * 40)
    evidence_dir.mkdir()
    if entry_name is not None:
        (evidence_dir / entry_name).write_text("not a packet")

    assert evidence_kind(evidence_dir) == H2_MEASUREMENT_ENVELOPE_PACKET
    with pytest.raises(envelope_verifier.EnvelopeVerificationError):
        envelope_verifier.verify_packet(evidence_dir)


def test_h2_measurement_envelope_names_never_collide_with_another_family() -> None:
    """The envelope grammar is disjoint, so classification order cannot matter.

    Every exact family ends in bare hex, and this one is checked after the flat
    archive family.  Pinning disjointness directly means a later edit to either
    pattern fails here instead of silently re-routing sealed evidence to the
    wrong verifier.
    """
    name = "h2_measure_envelope_" + "a" * 40
    assert is_h2_measurement_envelope_name(name)
    assert not is_h2_measurement_execution_name(name)
    assert not is_generic_dated_packet_name(name)
    assert not is_h0_phase_a_execution_name(name)
    assert not is_h0_preseal_freeze_v3_name(name)


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
        ("h2_measure_envelope_" + "a" * 40, H2_MEASUREMENT_ENVELOPE_PACKET),
        ("h2_measure_envelope_" + "a" * 39, None),
        ("h2_measure_envelope_" + "a" * 41, None),
        ("h2_measure_envelope_" + "A" * 40, None),
        ("h2_measure_envelope_20260804", GENERIC_RESEARCH_PACKET),
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


# -- H2 archive verification is host-independent ---------------------------- #


def _archived_authorization_fixture(
    destination: pathlib.Path, source: pathlib.Path
) -> tuple[dict, object]:
    """Copy the three authorization documents an archive binds together."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        h2_evidence.AUTHORIZATION_NAME,
        h2_evidence.AUTHORIZATION_GRANT_NAME,
        h2_evidence.AUTHORIZATION_DOMAIN_NAME,
    ):
        (destination / name).write_bytes((source / name).read_bytes())
    freeze = h2_evidence.load_document(source, h2_evidence.FREEZE_NAME)
    return freeze, h2_evidence.parse_root_name(source.name)


def _rebind_execution_domain(root: pathlib.Path, domain: dict) -> None:
    """Rewrite the whole digest chain so a fixture is internally consistent.

    domain -> digest(domain) -> grant["execution_domain"] -> digest(grant)
           -> receipt["execution_domain"], receipt["authorization_digest"]

    Without this, a semantic mutation dies at the digest binding and proves
    nothing about the predicates that judge the record's own shape.
    """
    grant = h2_evidence.load_document(root, h2_evidence.AUTHORIZATION_GRANT_NAME)
    receipt = h2_evidence.load_document(root, h2_evidence.AUTHORIZATION_NAME)
    digest = h2_evidence.digest(domain)
    grant["execution_domain"] = digest
    receipt["execution_domain"] = digest
    receipt["authorization_digest"] = h2_evidence.digest(grant)
    h2_evidence.write_document(root, h2_evidence.AUTHORIZATION_DOMAIN_NAME, domain)
    h2_evidence.write_document(root, h2_evidence.AUTHORIZATION_GRANT_NAME, grant)
    h2_evidence.write_document(root, h2_evidence.AUTHORIZATION_NAME, receipt)


def test_h2_phase_a_archive_verification_is_execution_host_independent(
    monkeypatch,
) -> None:
    """A committed attempt must verify on any host, not only its producer.

    Until 2026-07-29 `_authorization` recomputed the authorization execution
    domain from the *verifying* host's `/etc/machine-id` and `os.getuid()` and
    required equality with the archived record, so a Phase-A archive verified
    only on the machine that produced it — neither an independent reviewer nor
    CI could read it. Binding a grant to one host at launch is intended and
    still enforced by `run_h2_measurement`; carrying that live recomputation
    into archive verification was not.

    The guards below are the negative control: the live-host derivation is made
    to raise, the uid is moved, and the machine-identity files are made
    unreadable. Any reintroduction of a live-host read fails here.
    """
    import os

    archives = h2_measurement_execution_dirs()
    assert archives, "no H2 Layer-M measurement evidence directories were found"

    def refuse(*_args: object, **_kwargs: object) -> dict:
        raise AssertionError("archive verification consulted live host identity")

    monkeypatch.setattr(h2_evidence, "authorization_execution_domain", refuse)
    foreign_uid = os.getuid() + 1
    monkeypatch.setattr(os, "getuid", lambda: foreign_uid)
    repository = REPO.resolve()
    real_read_bytes = pathlib.Path.read_bytes

    def guarded_read_bytes(self: pathlib.Path) -> bytes:
        absolute = pathlib.Path(self).absolute()
        if absolute.as_posix() in (
            "/etc/machine-id",
            "/var/lib/dbus/machine-id",
        ) or not absolute.is_relative_to(repository):
            raise FileNotFoundError(f"host-coupled read during verification: {self}")
        return real_read_bytes(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", guarded_read_bytes)
    for archive in archives:
        report = measurement_verifier.verify_evidence_root(archive)
        assert isinstance(report, dict) and report.get("valid") is True
        assert report.get("verify_class") == "complete"


def test_h2_launch_time_host_binding_is_not_relaxed(monkeypatch) -> None:
    """The repair frees archive verification only, never the grant.

    `run_h2_measurement` still derives the execution domain live when an
    authorization is admitted and consumed, and still fail-closes when the
    controlled host has no machine identity.
    """
    monkeypatch.setattr(
        pathlib.Path,
        "read_bytes",
        lambda self: (_ for _ in ()).throw(OSError("absent")),
    )
    with pytest.raises(h2_evidence.EvidenceError):
        h2_evidence.authorization_execution_domain(pathlib.Path("/var/lib/h2"))


def test_h2_archived_execution_domain_integrity_is_still_fail_closed(
    tmp_path,
) -> None:
    """Class 1: mutate the domain alone and the digest binding must refuse."""
    source = h2_measurement_execution_dirs()[0]
    for mutation in (
        {"host_identity": "f" * 64},
        {"operator_uid": 4242},
        {"ledger_root": "/var/lib/other"},
    ):
        root = tmp_path / f"integrity_{sorted(mutation)[0]}"
        freeze, name = _archived_authorization_fixture(root, source)
        domain = h2_evidence.load_document(root, h2_evidence.AUTHORIZATION_DOMAIN_NAME)
        h2_evidence.write_document(
            root, h2_evidence.AUTHORIZATION_DOMAIN_NAME, {**domain, **mutation}
        )
        with pytest.raises(measurement_verifier.VerificationError) as excinfo:
            measurement_verifier._authorization(root, "a", freeze=freeze, name=name)
        assert "digest-unreconstructable" in str(excinfo.value)

    root = tmp_path / "integrity_member"
    freeze, name = _archived_authorization_fixture(root, source)
    domain = h2_evidence.load_document(root, h2_evidence.AUTHORIZATION_DOMAIN_NAME)
    domain.pop("operator_uid")
    h2_evidence.write_document(root, h2_evidence.AUTHORIZATION_DOMAIN_NAME, domain)
    with pytest.raises(measurement_verifier.VerificationError):
        measurement_verifier._authorization(root, "a", freeze=freeze, name=name)


def test_h2_archived_authorization_fixture_is_accepted_unmutated(tmp_path) -> None:
    """The base fixture must pass, or every negative below proves nothing."""
    source = h2_measurement_execution_dirs()[0]
    freeze, name = _archived_authorization_fixture(tmp_path / "base", source)
    receipt = measurement_verifier._authorization(
        tmp_path / "base", "a", freeze=freeze, name=name
    )
    assert receipt["state"] == "consumed"

    # A rebind that changes nothing must also survive, so a later failure is
    # attributable to the mutation and not to the rebinding machinery.
    root = tmp_path / "rebound"
    freeze, name = _archived_authorization_fixture(root, source)
    domain = h2_evidence.load_document(root, h2_evidence.AUTHORIZATION_DOMAIN_NAME)
    _rebind_execution_domain(root, domain)
    assert (
        measurement_verifier._authorization(root, "a", freeze=freeze, name=name)[
            "state"
        ]
        == "consumed"
    )


@pytest.mark.parametrize(
    "member,value",
    [
        ("host_identity", "a" * 63),
        ("host_identity", "a" * 65),
        ("host_identity", "A" * 64),
        ("host_identity", "g" * 64),
        ("host_identity", None),
        ("host_identity", 1),
        ("operator_uid", True),
        ("operator_uid", "1000"),
        ("operator_uid", -1),
        ("operator_uid", 1.0),
        ("operator_uid", None),
        ("ledger_root", "relative/ledger"),
        ("ledger_root", "/var/../etc/ledger"),
        ("ledger_root", "/var//lib/ledger"),
        ("ledger_root", "/var/./lib/ledger"),
        ("ledger_root", "/var/lib/ledger/"),
        ("ledger_root", "//var/lib/ledger"),
        ("ledger_root", ""),
        ("ledger_root", "/var/lib/led\0ger"),
        ("ledger_root", 17),
    ],
)
def test_h2_archived_execution_domain_shape_is_judged_after_the_digest_chain(
    tmp_path, member: str, value: object
) -> None:
    """Class 2: internally consistent, semantically illegal.

    The whole digest chain is recomputed so the record passes the binding
    clauses and dies precisely on the shape predicate it violates.
    """
    source = h2_measurement_execution_dirs()[0]
    root = tmp_path / "shape"
    freeze, name = _archived_authorization_fixture(root, source)
    domain = h2_evidence.load_document(root, h2_evidence.AUTHORIZATION_DOMAIN_NAME)
    _rebind_execution_domain(root, {**domain, member: value})
    with pytest.raises(measurement_verifier.VerificationError) as excinfo:
        measurement_verifier._authorization(root, "a", freeze=freeze, name=name)
    assert "archived authorization execution domain is malformed" in str(excinfo.value)


def test_h2_controlled_host_domain_anchor_agrees_with_the_committed_corpus() -> None:
    """The anchor must be the domain attempts were actually consumed under.

    A tracked anchor that matched no archive would be an authority invented by
    the file that declares it. Every committed Phase-A attempt is checked, so
    the anchor cannot be moved to admit a root the corpus never contained.
    """
    anchor = measurement_corpus.controlled_host_execution_domain()
    assert set(anchor) == set(h2_evidence.AUTHORIZATION_DOMAIN_MEMBERS)

    # An untracked file could define the authority without ever being reviewed.
    subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            measurement_corpus.CONTROLLED_HOST_DOMAIN_PATH.relative_to(REPO).as_posix(),
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
    )

    archives = [
        root
        for root in h2_measurement_execution_dirs()
        if h2_evidence.parse_root_name(root.name).phase == "a"
    ]
    assert archives, "no committed Phase-A attempt to anchor against"
    for root in archives:
        assert (
            h2_evidence.load_document(root, h2_evidence.AUTHORIZATION_DOMAIN_NAME)
            == anchor
        ), root.name


def test_h2_committed_archives_are_admitted_by_the_controlled_host_domain() -> None:
    """Archive validity alone is not canonical acceptance.

    `verify_evidence_root` answers whether a root is internally consistent. The
    corpus additionally requires that it was consumed under the controlled
    host's ledger, so this test asserts the conjunction rather than treating
    `valid is True` as admission on its own.
    """
    for root in h2_measurement_execution_dirs():
        verify_class = measurement_verifier.classify(root)
        phase = h2_evidence.parse_root_name(root.name).phase
        assert measurement_verifier.VERIFIERS[verify_class](root)["valid"] is True
        assert (
            measurement_corpus.execution_domain_admission_reasons(
                root, verify_class, phase
            )
            == ()
        ), root.name


def test_h2_rehearsal_shaped_archive_is_archive_valid_and_corpus_refused() -> None:
    """The hazard the guard exists for, built from a real committed attempt.

    A rehearsal runs the production controller against a disposable ledger, so
    its evidence root has the canonical shape and every internal binding holds.
    The surrogate here is that archive: the whole digest chain is recomputed
    from the archived producers, not patched by hand, so the only thing that
    distinguishes it is the ledger it was consumed against.

    Both halves matter. If the archive verifier refused it, the corpus rule
    would be redundant; if the corpus admitted it, a rehearsal could be
    registered as a measurement.
    """
    import shutil
    import tempfile

    source = h2_measurement_execution_dirs()[0]
    with tempfile.TemporaryDirectory() as workspace:
        root = pathlib.Path(workspace) / source.name
        shutil.copytree(source, root)
        before = {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

        domain = h2_evidence.load_document(root, h2_evidence.AUTHORIZATION_DOMAIN_NAME)
        _rebind_execution_domain(
            root, {**domain, "ledger_root": "/tmp/h2-rehearsal-ledger"}
        )
        h2_evidence.write_checksum_inventory(root)

        after = {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
        assert set(after) == set(before), "the surrogate added or dropped a file"
        assert {name for name in after if after[name] != before[name]} == {
            h2_evidence.AUTHORIZATION_DOMAIN_NAME,
            h2_evidence.AUTHORIZATION_GRANT_NAME,
            h2_evidence.AUTHORIZATION_NAME,
            "checksums.sha256",
        }

        assert measurement_verifier.verify_evidence_root(root)["valid"] is True

        with pytest.raises(measurement_corpus.CorpusError) as excinfo:
            measurement_corpus.check_corpus([root])
        assert "controlled host" in str(excinfo.value)


def _rebind_authorization_grant(root: pathlib.Path, grant: dict) -> None:
    """Rewrite the grant and the receipt digest that binds it.

    Same reason as `_rebind_execution_domain`: a semantic mutation that dies at
    `authorization_digest` proves nothing about the predicate under test.
    """
    receipt = h2_evidence.load_document(root, h2_evidence.AUTHORIZATION_NAME)
    receipt["authorization_digest"] = h2_evidence.digest(grant)
    h2_evidence.write_document(root, h2_evidence.AUTHORIZATION_GRANT_NAME, grant)
    h2_evidence.write_document(root, h2_evidence.AUTHORIZATION_NAME, receipt)


def test_h2_archive_verifier_reads_the_issuer_from_the_authority_constant(
    tmp_path, monkeypatch
) -> None:
    """The verifier must consult `AUTHORIZATION_ISSUER` at call time.

    Both directions are asserted from one fixture: with the authority moved, the
    archived grant's `research_owner` is no longer an issuer, and a grant naming
    the moved authority is. A verifier holding its own literal would answer the
    same in both halves.
    """
    source = h2_measurement_execution_dirs()[0]
    moved = "successor_research_owner"
    assert moved != h2_evidence.AUTHORIZATION_ISSUER

    root = tmp_path / "stale-issuer"
    freeze, name = _archived_authorization_fixture(root, source)
    monkeypatch.setattr(h2_evidence, "AUTHORIZATION_ISSUER", moved)
    with pytest.raises(measurement_verifier.VerificationError):
        measurement_verifier._authorization(root, "a", freeze=freeze, name=name)

    followed = tmp_path / "moved-issuer"
    freeze, name = _archived_authorization_fixture(followed, source)
    grant = h2_evidence.load_document(followed, h2_evidence.AUTHORIZATION_GRANT_NAME)
    _rebind_authorization_grant(followed, {**grant, "issued_by": moved})
    assert (
        measurement_verifier._authorization(followed, "a", freeze=freeze, name=name)[
            "state"
        ]
        == "consumed"
    )


def test_h2_authorization_issuer_value_is_unchanged() -> None:
    """Normalizing the literal into a constant must not move the authority."""
    assert h2_evidence.AUTHORIZATION_ISSUER == "research_owner"


@pytest.mark.parametrize("ledger_root", ["/", "/var/lib/h2", "/a/b/c"])
def test_h2_archived_execution_domain_accepts_canonical_absolute_paths(
    tmp_path, ledger_root: str
) -> None:
    """The shape rule must not be so strict it refuses legal ledger roots."""
    source = h2_measurement_execution_dirs()[0]
    root = tmp_path / "canonical"
    freeze, name = _archived_authorization_fixture(root, source)
    domain = h2_evidence.load_document(root, h2_evidence.AUTHORIZATION_DOMAIN_NAME)
    _rebind_execution_domain(root, {**domain, "ledger_root": ledger_root})
    assert (
        measurement_verifier._authorization(root, "a", freeze=freeze, name=name)[
            "state"
        ]
        == "consumed"
    )
