"""ADR 026: frozen inputs of CLOSED packets -- historical immutability vs HEAD.

Live-tree assertions are the always-on gate: no frozen input of any CLOSED
packet may drift from its digest without a supersession-ledger entry, and the
ledger itself must validate.  The scenario tests run in a detached worktree at
HEAD so they can mutate ``tracker_gpu.hpp`` (the #434 collision file) and the
ledger without touching the checkout:

* drift with no entry is ``unrecorded_drift`` on both arms;
* a valid historicization entry makes the packet ``historical`` -- development
  passes with a warning, attested fails, and the packet's pinned targeted tests
  are the ones scoped out of the development arm;
* an entry that lists the wrong packets, points at a ref whose blob does not
  hash to the frozen digest, or names a packet artifact, is refused, and the
  drift it was meant to cover falls back to ``unrecorded_drift``;
* supersession needs a successor packet that re-freezes the evolved bytes and
  declares ``supersedes`` under the same owner acceptance id.
"""

# scope: cross-module
# function: contract
# lifecycle: active

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import frozen_source_status as frozen  # noqa: E402

LEDGER = _REPO / frozen.LEDGER_REL
SCHEMA = _REPO / frozen.SCHEMA_REL
POLICY = _REPO / frozen.POLICY_REL
HPP = "include/tracking/tracker_gpu.hpp"
CU = "src/tracking/tracker_gpu.cu"
STATIC_PACKET = "h0_gctm_interface_static_feasibility_20260723"
UNIVERSE_PACKET = "gctm_runtime_native_candidate_universe_20260724"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", root.as_posix(), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


# --------------------------------------------------------------- live tree gate


def test_policy_ledger_and_schema_exist_and_validate() -> None:
    assert POLICY.is_file(), "ADR 026 must exist; the ledger points at it"
    ledger = frozen.load_ledger(LEDGER)
    assert ledger["schema"] == frozen.LEDGER_SCHEMA_ID
    assert ledger["policy"] == frozen.POLICY_REL
    assert frozen.validate_ledger_schema(ledger, SCHEMA) == []


def test_live_tree_has_no_unrecorded_drift_and_ledger_is_valid() -> None:
    report = frozen.evaluate(_REPO, mode="development")
    assert report.errors == [], "\n".join(report.errors)
    statuses = {item.status for item in report.bindings}
    assert statuses <= {frozen.STATUS_CURRENT, frozen.STATUS_HISTORICAL}


def test_the_two_tracker_binding_packets_are_discovered_with_pinned_tooling() -> None:
    bindings = frozen.discover_bindings(_REPO)
    tracker = {(b.packet_id, b.path) for b in bindings if b.path in (HPP, CU)}
    assert tracker == {
        (STATIC_PACKET, HPP),
        (STATIC_PACKET, CU),
        (UNIVERSE_PACKET, HPP),
        (UNIVERSE_PACKET, CU),
    }
    targeted = frozen.packet_targeted_tests(_REPO)
    assert (
        targeted[STATIC_PACKET]
        == "tests/contract/test_h0_gctm_static_feasibility_v1.py"
    )
    assert (
        targeted[UNIVERSE_PACKET] == "tests/contract/test_gctm_runtime_universe_v1.py"
    )


def test_cli_development_arm_passes_on_live_tree() -> None:
    proc = subprocess.run(
        [sys.executable, (_TOOLS / "frozen_source_status.py").as_posix(), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["mode"] == "development"
    assert payload["packet"] is None
    assert payload["policy"] == frozen.POLICY_REL


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        (HPP, frozen.KIND_SOURCE),
        (CU, frozen.KIND_SOURCE),
        ("scripts/tools/validate_gctm_runtime_universe.py", frozen.KIND_TOOLING),
        ("tests/contract/test_gctm_runtime_universe_v1.py", frozen.KIND_TOOLING),
        (
            "docs/research/contracts/score_ranking_evidence_contract.md",
            frozen.KIND_DOCUMENT,
        ),
        (
            f"{frozen.EVIDENCE_REL}/{STATIC_PACKET}/terminal_report.json",
            frozen.KIND_PACKET_ARTIFACT,
        ),
        (frozen.decl_id.H0_CAPTURE_DECLARATION_RELPATH, frozen.KIND_OWNER_DECLARATION),
    ],
)
def test_classify_path(path: str, kind: str) -> None:
    assert frozen.classify_path(path) == kind


def test_attested_consumer_switch_parsing() -> None:
    assert frozen.attested_consumer_requested({}) is False
    assert frozen.attested_consumer_requested({frozen.ATTESTED_ENV: "0"}) is False
    assert frozen.attested_consumer_requested({frozen.ATTESTED_ENV: "1"}) is True
    assert frozen.attested_consumer_requested({frozen.ATTESTED_ENV: "true"}) is True


# ------------------------------------------------------------------ scenarios


@pytest.fixture(scope="module")
def worktree(tmp_path_factory: pytest.TempPathFactory):
    """Detached worktree at HEAD with the live ledger/schema copied in.

    Scenario tests mutate files here; ``scenario`` resets them between tests.
    """
    root = tmp_path_factory.mktemp("frozen-worktree")
    target = root / "wt"
    subprocess.run(
        [
            "git",
            "-C",
            _REPO.as_posix(),
            "worktree",
            "add",
            "--detach",
            target.as_posix(),
            "HEAD",
        ],
        check=True,
        capture_output=True,
    )
    try:
        (target / frozen.LEDGER_REL).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LEDGER, target / frozen.LEDGER_REL)
        shutil.copyfile(SCHEMA, target / frozen.SCHEMA_REL)
        yield target
    finally:
        subprocess.run(
            [
                "git",
                "-C",
                _REPO.as_posix(),
                "worktree",
                "remove",
                "--force",
                target.as_posix(),
            ],
            check=False,
            capture_output=True,
        )
        shutil.rmtree(target, ignore_errors=True)


class Scenario:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.head = _git(root, "rev-parse", "HEAD")
        self.bindings = frozen.discover_bindings(root)
        self.hpp_frozen = next(b.sha256 for b in self.bindings if b.path == HPP)
        self.hpp_bound_by = sorted(
            (
                {"packet_id": b.packet_id, "binding": b.binding}
                for b in self.bindings
                if b.path == HPP
            ),
            key=lambda d: (d["packet_id"], d["binding"]),
        )

    # -- mutations
    def drift(
        self, path: str = HPP, suffix: bytes = b"\n// ADR 026 scenario drift\n"
    ) -> str:
        target = self.root / path
        target.write_bytes(target.read_bytes() + suffix)
        return _sha(target)

    def write_ledger(self, entries: list[dict[str, Any]]) -> None:
        ledger = frozen.load_ledger(self.root / frozen.LEDGER_REL)
        payload = dict(ledger)
        payload["entries"] = entries
        (self.root / frozen.LEDGER_REL).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    def entry(self, **overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "entry_id": "tracker_gpu_hpp_scenario",
            "kind": frozen.ENTRY_HISTORICIZATION,
            "path": HPP,
            "frozen_sha256": self.hpp_frozen,
            "bound_by": deepcopy(self.hpp_bound_by),
            "last_current_ref": self.head,
            "recorded_on": "2026-09-20",
            "recorded_by_pr": 450,
            "rationale": "scenario: tracker header evolves, no research claim inherits",
            "claims_inherited": False,
            "successor_packet_id": None,
            "owner_authorization": None,
        }
        base.update(overrides)
        return base

    def evaluate(self, mode: str = "development") -> frozen.Report:
        return frozen.evaluate(self.root, mode=mode)

    def statuses(self, report: frozen.Report, path: str = HPP) -> set[str]:
        return {i.status for i in report.bindings if i.binding.path == path}

    def write_successor(
        self,
        packet_id: str,
        new_sha: str,
        *,
        supersedes: list[dict[str, str]] | None,
        pin_sha: str | None = None,
    ) -> None:
        packet = self.root / frozen.EVIDENCE_REL / packet_id
        packet.mkdir(parents=True, exist_ok=True)
        (packet / "manifest.json").write_text(
            json.dumps({"schema": "scenario_manifest", "packet_id": packet_id}) + "\n"
        )
        record: dict[str, Any] = {
            "schema": "scenario_frozen_inputs",
            "identity_policy": "path_plus_sha256_only_no_mutable_branch_tip",
            "inputs": [
                {
                    "role": "h0_capture_record_types",
                    "path": HPP,
                    "sha256": pin_sha or new_sha,
                }
            ],
        }
        if supersedes is not None:
            record["supersedes"] = supersedes
        (packet / "frozen_input_identities.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )

    # -- reset
    def reset(self) -> None:
        subprocess.run(
            [
                "git",
                "-C",
                self.root.as_posix(),
                "checkout",
                "--",
                HPP,
                CU,
                frozen.EVIDENCE_REL,
            ],
            check=True,
            capture_output=True,
        )
        for extra in (self.root / frozen.EVIDENCE_REL).glob("scenario_*"):
            shutil.rmtree(extra, ignore_errors=True)
        self.write_ledger([])


@pytest.fixture
def scenario(worktree: Path):
    sc = Scenario(worktree)
    sc.reset()
    yield sc
    sc.reset()


def test_worktree_baseline_is_all_current(scenario: Scenario) -> None:
    report = scenario.evaluate()
    assert report.ok
    assert {i.status for i in report.bindings} == {frozen.STATUS_CURRENT}


def test_drift_without_entry_is_unrecorded_on_both_arms(scenario: Scenario) -> None:
    scenario.drift()
    for mode in ("development", "attested"):
        report = scenario.evaluate(mode)
        assert not report.ok
        assert scenario.statuses(report) == {frozen.STATUS_UNRECORDED}
        assert sum("unrecorded drift" in e for e in report.errors) == 2, report.errors
        assert all(frozen.POLICY_REL in e for e in report.errors if "unrecorded" in e)
        assert frozen.targeted_tests_to_skip(report, scenario.root) == {}


def test_historicization_entry_makes_packet_historical_not_broken(
    scenario: Scenario,
) -> None:
    scenario.drift()
    scenario.write_ledger([scenario.entry()])

    dev = scenario.evaluate("development")
    assert dev.ok, dev.errors
    assert scenario.statuses(dev) == {frozen.STATUS_HISTORICAL}
    assert sorted(dev.historical_packets) == [UNIVERSE_PACKET, STATIC_PACKET]
    assert len([w for w in dev.warnings if "is historical" in w]) == 2
    # The untouched sibling input keeps its own status: only hpp moved.
    assert scenario.statuses(dev, CU) == {frozen.STATUS_CURRENT}
    # The packets' own currency suites move to the attested arm.
    skip = frozen.targeted_tests_to_skip(dev, scenario.root)
    assert set(skip) == {
        "tests/contract/test_h0_gctm_static_feasibility_v1.py",
        "tests/contract/test_gctm_runtime_universe_v1.py",
    }
    assert all(frozen.ATTESTED_ENV in reason for reason in skip.values())

    attested = scenario.evaluate("attested")
    assert not attested.ok
    assert all("is historical" in e for e in attested.errors)


def test_historical_digest_stays_byte_verifiable_from_git(scenario: Scenario) -> None:
    scenario.drift()
    scenario.write_ledger([scenario.entry()])
    assert (
        frozen.git_blob_sha256(scenario.root, scenario.head, HPP) == scenario.hpp_frozen
    )
    assert _sha(scenario.root / HPP) != scenario.hpp_frozen


def test_dormant_entry_when_bytes_return_to_frozen_is_a_warning(
    scenario: Scenario,
) -> None:
    scenario.write_ledger([scenario.entry()])
    report = scenario.evaluate()
    assert report.ok
    assert scenario.statuses(report) == {frozen.STATUS_CURRENT}
    assert any("dormant" in w for w in report.warnings)


def test_entry_must_list_every_packet_pinning_the_digest(scenario: Scenario) -> None:
    scenario.drift()
    scenario.write_ledger([scenario.entry(bound_by=scenario.hpp_bound_by[:1])])
    report = scenario.evaluate()
    assert not report.ok
    assert any("bound_by must list exactly" in e for e in report.errors)
    assert scenario.statuses(report) == {frozen.STATUS_UNRECORDED}


def test_entry_ref_must_carry_the_frozen_bytes(scenario: Scenario) -> None:
    scenario.drift()
    stale_ref = _git(scenario.root, "rev-parse", "746609f1~1")
    scenario.write_ledger([scenario.entry(last_current_ref=stale_ref)])
    report = scenario.evaluate()
    assert not report.ok
    assert any("not the frozen" in e for e in report.errors), report.errors
    assert scenario.statuses(report) == {frozen.STATUS_UNRECORDED}


def test_entry_ref_must_be_an_ancestor_of_head(scenario: Scenario) -> None:
    scenario.drift()
    scenario.write_ledger([scenario.entry(last_current_ref="0" * 40)])
    report = scenario.evaluate()
    assert not report.ok
    assert any("not a commit reachable" in e for e in report.errors), report.errors


def test_packet_artifacts_cannot_be_historicized(scenario: Scenario) -> None:
    artifact = f"{frozen.EVIDENCE_REL}/{STATIC_PACKET}/terminal_report.json"
    artifact_sha = next(b.sha256 for b in scenario.bindings if b.path == artifact)
    bound = sorted(
        (
            {"packet_id": b.packet_id, "binding": b.binding}
            for b in scenario.bindings
            if b.path == artifact
        ),
        key=lambda d: (d["packet_id"], d["binding"]),
    )
    scenario.drift(artifact, b"\n")
    scenario.write_ledger(
        [
            scenario.entry(
                entry_id="artifact_attempt",
                path=artifact,
                frozen_sha256=artifact_sha,
                bound_by=bound,
            )
        ]
    )
    report = scenario.evaluate()
    assert not report.ok
    assert any("cannot be historicized" in e for e in report.errors), report.errors
    assert any("immutable" in e for e in report.errors)


def test_entry_for_an_unbound_digest_is_refused(scenario: Scenario) -> None:
    scenario.write_ledger([scenario.entry(frozen_sha256="f" * 64)])
    report = scenario.evaluate()
    assert not report.ok
    assert any("no packet binds" in e for e in report.errors)


def test_claims_inherited_is_never_true_in_the_ledger(scenario: Scenario) -> None:
    scenario.drift()
    scenario.write_ledger([scenario.entry(claims_inherited=True)])
    report = scenario.evaluate()
    assert not report.ok
    assert any("claims_inherited must be false" in e for e in report.errors)


def test_historicization_must_not_name_a_successor(scenario: Scenario) -> None:
    scenario.drift()
    scenario.write_ledger([scenario.entry(successor_packet_id="scenario_successor")])
    report = scenario.evaluate()
    assert not report.ok
    assert any("must not name a successor" in e for e in report.errors)


def test_supersession_requires_successor_and_owner_authorization(
    scenario: Scenario,
) -> None:
    scenario.drift()
    scenario.write_ledger([scenario.entry(kind=frozen.ENTRY_SUPERSESSION)])
    report = scenario.evaluate()
    assert any("requires successor_packet_id" in e for e in report.errors)

    scenario.write_ledger(
        [
            scenario.entry(
                kind=frozen.ENTRY_SUPERSESSION, successor_packet_id="scenario_successor"
            )
        ]
    )
    report = scenario.evaluate()
    assert any("requires owner_authorization" in e for e in report.errors)


def test_supersession_successor_must_refreeze_and_declare_supersedes(
    scenario: Scenario,
) -> None:
    new_sha = scenario.drift()
    auth = {
        "owner_acceptance_id": "scenario_owner_acceptance_20260920",
        "date": "2026-09-20",
    }
    entry = scenario.entry(
        kind=frozen.ENTRY_SUPERSESSION,
        successor_packet_id="scenario_successor",
        owner_authorization=auth,
    )
    scenario.write_ledger([entry])

    # no successor packet on disk
    report = scenario.evaluate()
    assert any("must exist with" in e for e in report.errors)

    # successor pins the superseded digest instead of the evolved bytes
    scenario.write_successor(
        "scenario_successor", new_sha, supersedes=[], pin_sha=scenario.hpp_frozen
    )
    report = scenario.evaluate()
    assert any("superseded digest" in e for e in report.errors), report.errors

    # successor re-freezes but does not declare supersedes
    scenario.write_successor("scenario_successor", new_sha, supersedes=None)
    report = scenario.evaluate()
    assert any("must carry a `supersedes` list" in e for e in report.errors)

    # supersedes lists only one of the two bound packets
    scenario.write_successor(
        "scenario_successor",
        new_sha,
        supersedes=[
            {
                "packet_id": STATIC_PACKET,
                "owner_acceptance_id": auth["owner_acceptance_id"],
            }
        ],
    )
    report = scenario.evaluate()
    assert any(UNIVERSE_PACKET in e and "must declare" in e for e in report.errors)

    # complete successor: historical on development, still not current on attested
    scenario.write_successor(
        "scenario_successor",
        new_sha,
        supersedes=[
            {"packet_id": p, "owner_acceptance_id": auth["owner_acceptance_id"]}
            for p in (STATIC_PACKET, UNIVERSE_PACKET)
        ],
    )
    dev = scenario.evaluate()
    assert dev.ok, dev.errors
    assert scenario.statuses(dev) == {frozen.STATUS_HISTORICAL, frozen.STATUS_CURRENT}
    # the successor's own pin of the new bytes is current; the old packets are historical
    by_packet = {
        (i.binding.packet_id, i.status) for i in dev.bindings if i.binding.path == HPP
    }
    assert ("scenario_successor", frozen.STATUS_CURRENT) in by_packet
    assert (STATIC_PACKET, frozen.STATUS_HISTORICAL) in by_packet
    assert not scenario.evaluate("attested").ok


def _complete_successor(scenario: Scenario) -> str:
    """Historicize hpp for both packets and stand up a valid successor packet."""
    new_sha = scenario.drift()
    auth = {
        "owner_acceptance_id": "scenario_owner_acceptance_20260920",
        "date": "2026-09-20",
    }
    scenario.write_ledger(
        [
            scenario.entry(
                kind=frozen.ENTRY_SUPERSESSION,
                successor_packet_id="scenario_successor",
                owner_authorization=auth,
            )
        ]
    )
    scenario.write_successor(
        "scenario_successor",
        new_sha,
        supersedes=[
            {"packet_id": p, "owner_acceptance_id": auth["owner_acceptance_id"]}
            for p in (STATIC_PACKET, UNIVERSE_PACKET)
        ],
    )
    return "scenario_successor"


def test_scoped_attestation_passes_for_a_current_successor(scenario: Scenario) -> None:
    successor = _complete_successor(scenario)
    report = frozen.evaluate(scenario.root, mode="attested", packet=successor)
    assert report.ok, report.errors
    assert report.packet == successor
    # predecessors are still historical, reported but not fatal to the successor's claim
    assert sorted(report.historical_packets) == [UNIVERSE_PACKET, STATIC_PACKET]
    assert len([w for w in report.warnings if "is historical" in w]) == 2
    # the global claim is still refused: the repo as a whole is not at the old coordinate
    assert not frozen.evaluate(scenario.root, mode="attested").ok


def test_scoped_attestation_of_a_historical_packet_fails(scenario: Scenario) -> None:
    _complete_successor(scenario)
    report = frozen.evaluate(scenario.root, mode="attested", packet=STATIC_PACKET)
    assert not report.ok
    assert all("is historical" in e for e in report.errors), report.errors
    assert all(STATIC_PACKET in e for e in report.errors)


def test_scoped_attestation_of_an_unknown_packet_fails_closed(
    scenario: Scenario,
) -> None:
    report = frozen.evaluate(scenario.root, mode="attested", packet="no_such_packet")
    assert not report.ok
    assert any("unknown packet cannot be attested" in e for e in report.errors)


def test_scoped_attestation_still_fails_on_unrecorded_drift(scenario: Scenario) -> None:
    successor = _complete_successor(scenario)
    scenario.drift(CU)  # second frozen path moves without an entry
    report = frozen.evaluate(scenario.root, mode="attested", packet=successor)
    assert not report.ok
    assert any("unrecorded drift" in e for e in report.errors)


def test_packet_scoping_requires_attested_mode() -> None:
    with pytest.raises(ValueError, match="attested"):
        frozen.evaluate(_REPO, mode="development", packet=STATIC_PACKET)


# ----------------------------------------------------------------- append-only


def _ledger(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": frozen.LEDGER_SCHEMA_ID,
        "policy": frozen.POLICY_REL,
        "entries": entries,
    }


def test_append_only_accepts_appends_and_an_absent_base() -> None:
    a = {"entry_id": "a", "kind": "historicization"}
    b = {"entry_id": "b", "kind": "historicization"}
    assert frozen.append_only_violations(_ledger([a, b]), None) == []
    assert frozen.append_only_violations(_ledger([a, b]), _ledger([a])) == []
    assert frozen.append_only_violations(_ledger([a]), _ledger([a])) == []


def test_append_only_refuses_removed_or_modified_entries() -> None:
    a = {"entry_id": "a", "kind": "historicization", "rationale": "x"}
    a_edit = {"entry_id": "a", "kind": "historicization", "rationale": "y"}
    removed = frozen.append_only_violations(_ledger([]), _ledger([a]))
    assert removed == ["ledger is append-only: entry a was removed"]
    modified = frozen.append_only_violations(_ledger([a_edit]), _ledger([a]))
    assert modified == ["ledger is append-only: entry a was modified"]


def test_append_only_base_resolution(scenario: Scenario) -> None:
    # HEAD itself as base: the committed ledger is the base, scenario entries are appends
    scenario.drift()
    scenario.write_ledger([scenario.entry()])
    report = frozen.evaluate(scenario.root, base="HEAD", base_required=True)
    assert report.ok, report.errors
    # an unresolvable base is a warning by default and an error when required
    lenient = frozen.evaluate(scenario.root, base="refs/no/such/ref")
    assert lenient.ok
    assert any("append-only check skipped" in w for w in lenient.warnings)
    strict = frozen.evaluate(scenario.root, base="refs/no/such/ref", base_required=True)
    assert not strict.ok
    assert any("cannot resolve merge-base" in e for e in strict.errors)


def test_missing_ledger_is_a_deleted_guard(scenario: Scenario) -> None:
    (scenario.root / frozen.LEDGER_REL).unlink()
    report = scenario.evaluate()
    assert not report.ok
    assert any("deleted guard" in e for e in report.errors)
    shutil.copyfile(LEDGER, scenario.root / frozen.LEDGER_REL)


def test_replay_runs_a_packet_suite_at_a_frozen_coordinate() -> None:
    """The strongest historical check: the packet still passes its own pinned tests
    at the coordinate where it was current.  HEAD is such a coordinate today."""
    head = _git(_REPO, "rev-parse", "HEAD")
    env = dict(os.environ)
    env.pop(frozen.ATTESTED_ENV, None)
    proc = subprocess.run(
        [
            sys.executable,
            (_TOOLS / "frozen_source_status.py").as_posix(),
            "--replay",
            UNIVERSE_PACKET,
            "--at",
            head,
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
