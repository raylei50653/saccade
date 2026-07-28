"""The H2 S4 controller must produce exactly what the independent verifier reads."""

# scope: tracking, system
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "scripts" / "tools"
if _TOOLS.as_posix() not in sys.path:
    sys.path.insert(0, _TOOLS.as_posix())

import build_runtime_identity as identity  # noqa: E402
import h2_behavioral_identity as behavior  # noqa: E402
import h2_measurement_evidence as evidence  # noqa: E402
import run_h2_measurement as controller  # noqa: E402
import run_h2_measurement_child as child  # noqa: E402
import verify_h2_measurement as verifier  # noqa: E402
from export_headline_bridge_decision_trace import (  # noqa: E402
    canonical_semantic_packet,
)
from run_h2_layer_p import CERTIFICATE_SCHEMA  # noqa: E402
from verify_headline_bridge_decision_trace import verify_capture  # noqa: E402


def _packet_builder():
    path = _REPO / "tests/unit/tracking/test_headline_bridge_decision_trace.py"
    spec = importlib.util.spec_from_file_location("_h2_controller_packet", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._packet


_packet = _packet_builder()


class NullMonitor:
    def __init__(self, *_: object, **__: object) -> None:
        self.history: list[Any] = []
        self.closed = False

    def drain(self) -> list[Any]:
        return []

    def close(self) -> None:
        self.closed = True


def _projection(capture: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    packet = canonical_semantic_packet(capture)
    streams = packet["streams"]
    candidates = [
        row for row in streams["candidate_records"] if int(row["proposal_emitted"]) == 1
    ]
    claims = streams["claim_records"]
    commits = streams["commit_records"]
    proposal = {"candidates": candidates, "claims": claims}
    winner = {
        "commits": commits,
        "winning_claims": [row for row in claims if int(row["claim_won"]) == 1],
    }
    return (
        {
            "count": len(candidates),
            "digest": evidence.digest(proposal),
            "records": proposal,
        },
        {
            "count": len(commits),
            "digest": evidence.digest(winner),
            "records": winner,
        },
    )


def _products(
    invocation: Mapping[str, Any],
    *,
    perturbed_run: str | None = None,
) -> child.RunProducts:
    run_id = str(invocation["run_id"])
    mot_bytes = (
        b"1,9,0,0,1,1,0.9,-1,-1,-1\n"
        if run_id != perturbed_run
        else b"1,9,0,0,2,1,0.9,-1,-1,-1\n"
    )
    packet = None
    packet_verification = None
    proposal = None
    winner = None
    if run_id != evidence.CAPTURE_OFF_RUN:
        capture = _packet(run_uuid=f"{run_id}-uuid")
        proposal, winner = _projection(capture)
        packet = capture
        packet_verification = {"report": verify_capture(capture), "state": "pass"}
    inventory = {
        behavior.BEHAVIOR_MEMBERS[0]: [{"frame": 1, "pairs": [[9, 3], [7, 4]]}],
        behavior.BEHAVIOR_MEMBERS[1]: [
            {
                "binary32_bits": [1, 2, 3, 4, 5],
                "class": 1,
                "frame": 1,
                "row_index": 0,
                "track_id": 9,
            }
        ],
        behavior.BEHAVIOR_MEMBERS[2]: {
            "length": len(mot_bytes),
            "sha256": hashlib.sha256(mot_bytes).hexdigest(),
        },
        behavior.A76_OVERFLOW_MEMBER: list(behavior.A76_OVERFLOW_ZERO_VECTOR),
        behavior.A76_PROJECTION_MEMBERS[0]: proposal,
        behavior.BEHAVIOR_MEMBERS[3]: list(range(13)),
        "schema": evidence.POLICY_INVENTORY_SCHEMA,
        behavior.A76_PROJECTION_MEMBERS[1]: winner,
    }
    base_inventory = {
        **{member: inventory[member] for member in behavior.A76_EQUALITY_MEMBERS},
        "schema": evidence.BASE_POLICY_INVENTORY_SCHEMA,
    }
    return child.RunProducts(
        mot_bytes=mot_bytes,
        policy_base_inventory=base_inventory,
        policy_inventory=inventory,
        packet=packet,
        packet_verification=packet_verification,
    )


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    evidence.write_document(path.parent, path.name, payload)


def _bundle(tmp_path: Path) -> controller.LaunchBundle:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=_REPO, text=True
    ).strip()
    coordinate = dict(
        zip(
            identity.ALL_COORDINATE_AXES,
            (character * 64 for character in "12345"),
            strict=True,
        )
    )
    commit_axes = verifier._commit_content_axes(head)
    coordinate["implementation"] = commit_axes["decision_relevant"]["digest"]
    coordinate["identity_semantics"] = commit_axes["identity_semantics"]["digest"]
    probe_digest = "a" * 64
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    manifest = {
        "schema": "h2_runtime_input_manifest_v1",
        "coordinate_digest": coordinate["runtime_inputs"],
        "full_digest": "b" * 64,
        "build_artifacts": {
            "build_dir": build_dir.as_posix(),
            "digest": "c" * 64,
        },
    }
    reference = {
        "schema": behavior.RESULT_SCHEMA,
        "build_witness": {"digest": manifest["build_artifacts"]["digest"]},
        "digest": probe_digest,
        "digests": [probe_digest],
        "identical": True,
        "mode": "identity",
        "sequence": behavior.IDENTITY_SEQUENCE,
    }
    published = {
        "schema": identity.IDENTITY_SCHEMA,
        "coordinate": coordinate,
        "equivalence": {"state": "unproven"},
        "probe": {"digest": probe_digest},
        "publication_complete": True,
    }
    reference_path = tmp_path / "reference.json"
    runtime_path = tmp_path / "runtime.json"
    published_path = tmp_path / "published.json"
    _write(reference_path, reference)
    _write(runtime_path, manifest)
    _write(published_path, published)
    certificate = {
        "schema": CERTIFICATE_SCHEMA,
        "source_head": head,
        "source_tree": tree,
        "selected_base": head,
        "changed_path_verdict": {"admissible": True, "base": head},
        "decision_relevant_digest": coordinate["implementation"],
        "equivalence": "unproven",
        "identity_semantics_digest": coordinate["identity_semantics"],
        "plumbing_set_digest": commit_axes["plumbing_only"]["digest"],
        "published_coordinate": coordinate,
        "behavior_probe": probe_digest,
        "build_witness": reference["build_witness"],
        "fixture": behavior.IDENTITY_SEQUENCE,
        "mode": "identity",
        "probe_schema": behavior.RESULT_SCHEMA,
        "published_probe": probe_digest,
        "runtime_input_coordinate_digest": manifest["coordinate_digest"],
        "runtime_input_full_digest": manifest["full_digest"],
        "build_artifact_digest": manifest["build_artifacts"]["digest"],
        "runtime_input_manifest_file_digest": evidence.sha256_file(runtime_path),
        "probe_result_file_digest": evidence.sha256_file(reference_path),
        "published_identity_file_digest": evidence.sha256_file(published_path),
        "build_dir": build_dir.as_posix(),
    }
    certificate_path = tmp_path / "certificate.json"
    _write(certificate_path, certificate)
    freeze = {
        "schema": evidence.FREEZE_SCHEMA,
        "capture_phase": evidence.CAPTURE_PHASE["a"],
        "instrumentation_head": head,
        "selected_base": head,
        "coordinate": coordinate,
        "probe": probe_digest,
        "equivalence": "unproven",
        "layer_p_certificate": {
            "schema": CERTIFICATE_SCHEMA,
            "digest": evidence.digest(certificate),
        },
        "reference_probe": {
            "schema": behavior.RESULT_SCHEMA,
            "file_digest": evidence.sha256_file(reference_path),
        },
        "runtime_inputs": {
            "schema": manifest["schema"],
            "file_digest": evidence.sha256_file(runtime_path),
            "coordinate_digest": manifest["coordinate_digest"],
            "full_digest": manifest["full_digest"],
            "build_artifact_digest": manifest["build_artifacts"]["digest"],
        },
        "published_identity": {
            "schema": identity.IDENTITY_SCHEMA,
            "file_digest": evidence.sha256_file(published_path),
        },
        "capture_abi": {
            "path": evidence.PHASE_A_CAPTURE_ABI_PATH,
            "sha256": hashlib.sha256(
                subprocess.check_output(
                    [
                        "git",
                        "show",
                        f"{head}:{evidence.PHASE_A_CAPTURE_ABI_PATH}",
                    ],
                    cwd=_REPO,
                )
            ).hexdigest(),
        },
        "executed_surfaces": {
            path: hashlib.sha256(
                subprocess.check_output(
                    ["git", "show", f"{head}:{path}"],
                    cwd=_REPO,
                )
            ).hexdigest()
            for path in evidence.PHASE_A_EXECUTED_SURFACE_PATHS
        },
        "run_plan": {
            "sequence": controller.SEQUENCE,
            "run_ids": list(evidence.RUN_IDS),
        },
    }
    freeze_path = tmp_path / "freeze.json"
    _write(freeze_path, freeze)
    return controller.LaunchBundle(
        freeze=freeze,
        certificate=certificate,
        reference_probe=reference,
        runtime_manifest=manifest,
        published_identity=published,
        freeze_path=freeze_path,
        certificate_path=certificate_path,
        reference_probe_path=reference_path,
        runtime_manifest_path=runtime_path,
        published_identity_path=published_path,
    )


def _authorization(
    tmp_path: Path,
    bundle: controller.LaunchBundle,
    *,
    identity_suffix: str = "",
) -> controller.AuthorizationGrant:
    seed = evidence.digest(
        {
            "path": tmp_path.as_posix(),
            "suffix": identity_suffix,
        }
    )
    grant = {
        "schema": evidence.AUTHORIZATION_GRANT_SCHEMA,
        "authorization_id": seed,
        "capture_phase": evidence.CAPTURE_PHASE["a"],
        "controller_digest": bundle.freeze["executed_surfaces"][
            "scripts/tools/run_h2_measurement.py"
        ],
        "freeze_digest": evidence.freeze_digest(bundle.freeze),
        "instrumentation_head": bundle.head,
        "invocation_id": evidence.digest({"invocation": seed}),
        "issued_by": "research_owner",
    }
    path = tmp_path / f"authorization{identity_suffix}.json"
    _write(path, grant)
    return controller.load_authorization(
        path,
        bundle,
        invocation_id=grant["invocation_id"],
    )


def _checkout_witness(
    bundle: controller.LaunchBundle,
    *,
    current_head: str,
    current_tree: str,
) -> dict[str, Any]:
    return {
        "schema": evidence.CHECKOUT_WITNESS_SCHEMA,
        "axes": verifier._commit_content_axes(current_head),
        "build_dir": bundle.build_dir.resolve(strict=True).as_posix(),
        "repository_root": _REPO.resolve(strict=True).as_posix(),
        "source_head": current_head,
        "source_tree": current_tree,
    }


def _checkout_state() -> tuple[str, str, bool]:
    return (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip(),
        subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=_REPO, text=True
        ).strip(),
        True,
    )


def _probe(root: Path, **_: object) -> dict[str, Any]:
    freeze = evidence.load_document(root, evidence.FREEZE_NAME)
    manifest = evidence.load_document(root, evidence.RUNTIME_INPUTS_NAME)
    payload = {
        "schema": behavior.RESULT_SCHEMA,
        "build_witness": {"digest": manifest["build_artifacts"]["digest"]},
        "digest": freeze["probe"],
        "digests": [freeze["probe"]],
        "identical": True,
        "mode": "identity",
        "sequence": behavior.IDENTITY_SEQUENCE,
    }
    evidence.write_document(root, evidence.LAUNCH_PROBE_NAME, payload)
    return payload


def _no_revalidation(_bundle: controller.LaunchBundle) -> tuple[str, ...]:
    """Synthetic fixtures exercise controller flow, not host runtime inputs."""
    return ()


def _launcher(
    *,
    perturbed_run: str | None = None,
    invalid_run: str | None = None,
    projection_mismatch_run: str | None = None,
):
    def launch(
        invocation_path: Path,
        environment: Mapping[str, str],
        **_: object,
    ) -> int:
        def products(invocation: Mapping[str, Any]) -> child.RunProducts:
            produced = _products(invocation, perturbed_run=perturbed_run)
            if (
                str(invocation["run_id"]) == projection_mismatch_run
                and produced.policy_inventory is not None
            ):
                records = {"candidates": [], "claims": []}
                produced.policy_inventory[behavior.A76_PROJECTION_MEMBERS[0]] = {
                    "count": 0,
                    "digest": evidence.digest(records),
                    "records": records,
                }
            if str(invocation["run_id"]) != invalid_run:
                return produced
            capture = {"capture_schema_version": "invalid"}
            processed = child.persist_and_process_capture(
                Path(str(invocation["run_dir"])), capture
            )
            assert processed.valid is False
            return child.RunProducts(
                mot_bytes=produced.mot_bytes,
                policy_base_inventory=produced.policy_base_inventory,
                policy_inventory=None,
                packet=capture,
                packet_verification=processed.verification,
            )

        return child.execute_child(
            invocation_path,
            environment=environment,
            runner=products,
        )

    return launch


def _execute(
    tmp_path: Path,
    *,
    perturbed_run: str | None = None,
    invalid_run: str | None = None,
    projection_mismatch_run: str | None = None,
):
    bound = tmp_path / "bound"
    bound.write_text("frozen\n", encoding="utf-8")
    bundle = _bundle(tmp_path)
    return controller.execute_controller(
        bundle,
        authorization=_authorization(tmp_path, bundle),
        evidence_parent=tmp_path / "evidence",
        bound_paths=(bound,),
        require_clean_checkout=False,
        launch_probe=_probe,
        launch_child=_launcher(
            perturbed_run=perturbed_run,
            invalid_run=invalid_run,
            projection_mismatch_run=projection_mismatch_run,
        ),
        monitor_factory=NullMonitor,
        bundle_revalidator=_no_revalidation,
        checkout_witness_builder=_checkout_witness,
        checkout_state_reader=_checkout_state,
        inherited_environment={},
    )


def test_authorization_is_required_and_bound_to_the_exact_invocation(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    with pytest.raises(controller.ControllerError, match="authorization is absent"):
        controller.execute_controller(
            bundle,
            authorization=None,  # type: ignore[arg-type]
            evidence_parent=tmp_path / "absent",
            require_clean_checkout=False,
        )

    grant = _authorization(tmp_path, bundle)
    with pytest.raises(controller.ControllerError, match="another"):
        controller.load_authorization(
            grant.path,
            bundle,
            invocation_id="9" * 64,
        )


def test_controller_rejects_changed_path_verdict_for_another_base(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    bundle.certificate["changed_path_verdict"]["base"] = "9" * 40
    bundle.freeze["layer_p_certificate"]["digest"] = evidence.digest(bundle.certificate)
    witness = _checkout_witness(
        bundle,
        current_head=bundle.head,
        current_tree=str(bundle.certificate["source_tree"]),
    )
    reasons = controller.certificate_match_reasons(
        bundle,
        current_head=bundle.head,
        current_tree=str(bundle.certificate["source_tree"]),
        checkout_witness=witness,
    )
    assert reasons == ("certificate changed-path verdict is not clean",)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("instrumentation_head", "9" * 40),
        ("freeze_digest", "9" * 64),
        ("controller_digest", "9" * 64),
        ("capture_phase", "phase_b"),
    ),
)
def test_authorization_bound_to_another_surface_cannot_launch(
    tmp_path: Path, field: str, value: str
) -> None:
    bundle = _bundle(tmp_path)
    grant = _authorization(tmp_path, bundle)
    record = evidence.load_document(grant.path.parent, grant.path.name)
    record[field] = value
    _write(grant.path, record)
    with pytest.raises(controller.ControllerError, match="another"):
        controller.load_authorization(
            grant.path,
            bundle,
            invocation_id=str(record["invocation_id"]),
        )


def test_authorization_consumption_is_exclusive_across_evidence_roots(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    grant = _authorization(tmp_path, bundle)
    ledger = tmp_path / "ledger"
    first, selection = controller.execute_controller(
        bundle,
        authorization=grant,
        evidence_parent=tmp_path / "first",
        authorization_ledger=ledger,
        bound_paths=(bundle.runtime_manifest_path,),
        require_clean_checkout=False,
        launch_probe=_probe,
        launch_child=_launcher(),
        monitor_factory=NullMonitor,
        bundle_revalidator=_no_revalidation,
        checkout_witness_builder=_checkout_witness,
        checkout_state_reader=_checkout_state,
        inherited_environment={},
    )
    assert selection.terminal is None
    archived_grant = evidence.load_document(
        first,
        evidence.AUTHORIZATION_GRANT_NAME,
        schema=evidence.AUTHORIZATION_GRANT_SCHEMA,
    )
    receipt = evidence.load_document(
        first,
        evidence.AUTHORIZATION_NAME,
        schema=evidence.AUTHORIZATION_SCHEMA,
    )
    assert archived_grant == grant.record
    assert receipt["authorization_digest"] == evidence.digest(archived_grant)
    assert (first / evidence.AUTHORIZATION_NAME).is_file()
    with pytest.raises(controller.ControllerError, match="already consumed"):
        controller.execute_controller(
            bundle,
            authorization=grant,
            evidence_parent=tmp_path / "second",
            authorization_ledger=ledger,
            require_clean_checkout=False,
        )


def test_prior_default_ledger_marker_does_not_block_a_new_authorization(
    tmp_path: Path,
) -> None:
    assert controller.checkout_hygiene_reasons() == ()
    bundle = _bundle(tmp_path)
    grant_a = _authorization(tmp_path, bundle, identity_suffix="-ledger-a")
    grant_b = _authorization(tmp_path, bundle, identity_suffix="-ledger-b")
    ledger = controller.default_authorization_ledger()
    marker_a = ledger / f"{grant_a.record['authorization_id']}.json"
    marker_b = ledger / f"{grant_b.record['authorization_id']}.json"
    assert not marker_a.exists()
    assert not marker_b.exists()
    try:
        controller._consume_authorization(grant_a, bundle=bundle, ledger=ledger)
        assert marker_a.is_file()
        root, selection = controller.execute_controller(
            bundle,
            authorization=grant_b,
            evidence_parent=tmp_path / "second-authorization",
            bound_paths=(bundle.runtime_manifest_path,),
            require_clean_checkout=True,
            launch_probe=_probe,
            launch_child=_launcher(),
            monitor_factory=NullMonitor,
            bundle_revalidator=_no_revalidation,
            checkout_witness_builder=_checkout_witness,
            checkout_state_reader=_checkout_state,
            inherited_environment={},
        )
        assert marker_a.is_file()
        assert marker_b.is_file()
        assert selection.terminal is None
        assert verifier.verify_evidence_root(root)["result"] == "measurement_pass"
        with pytest.raises(controller.ControllerError, match="already consumed"):
            controller._consume_authorization(
                grant_a,
                bundle=bundle,
                ledger=ledger,
            )
    finally:
        for marker in (marker_a, marker_b):
            if marker.is_file():
                marker.unlink()


def test_crash_after_consumption_before_child_still_prevents_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path)
    grant = _authorization(tmp_path, bundle)
    ledger = tmp_path / "ledger"
    real_consume = controller._consume_authorization

    def consume_then_crash(*args: object, **kwargs: object) -> dict[str, Any]:
        real_consume(*args, **kwargs)  # type: ignore[arg-type]
        raise KeyboardInterrupt("injected crash after durable consumption")

    monkeypatch.setattr(controller, "_consume_authorization", consume_then_crash)
    with pytest.raises(KeyboardInterrupt, match="durable consumption"):
        controller.execute_controller(
            bundle,
            authorization=grant,
            evidence_parent=tmp_path / "crashed",
            authorization_ledger=ledger,
            require_clean_checkout=False,
        )
    monkeypatch.setattr(controller, "_consume_authorization", real_consume)
    with pytest.raises(controller.ControllerError, match="already consumed"):
        controller.execute_controller(
            bundle,
            authorization=grant,
            evidence_parent=tmp_path / "retry",
            authorization_ledger=ledger,
            require_clean_checkout=False,
        )


def test_checkout_hygiene_excludes_only_bounded_controller_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "H2 Contract"],
        cwd=tmp_path,
        check=True,
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    own_root = tmp_path / "evidence" / "h2_measure_test.incomplete"
    own_root.mkdir(parents=True)
    (own_root / "record.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(controller, "REPO_ROOT", tmp_path)
    assert controller.checkout_hygiene_reasons(excluded_roots=(own_root,)) == ()

    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("dirty\n", encoding="utf-8")
    assert controller.checkout_hygiene_reasons(excluded_roots=(own_root,))
    unrelated.unlink()
    tracked.write_text("dirty\n", encoding="utf-8")
    assert controller.checkout_hygiene_reasons(excluded_roots=(own_root,))


def test_checkout_dirtiness_does_not_contaminate_certificate_or_monitor_predicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path)
    calls = 0

    def hygiene(**_kwargs: object) -> tuple[str, ...]:
        nonlocal calls
        calls += 1
        return () if calls == 1 else ("??:unrelated.txt",)

    monkeypatch.setattr(controller, "checkout_hygiene_reasons", hygiene)
    root, selection = controller.execute_controller(
        bundle,
        authorization=_authorization(tmp_path, bundle),
        evidence_parent=tmp_path / "evidence",
        bound_paths=(bundle.runtime_manifest_path,),
        require_clean_checkout=True,
        launch_probe=lambda *_args, **_kwargs: pytest.fail("probe must not run"),
        launch_child=lambda *_args, **_kwargs: pytest.fail("child must not run"),
        monitor_factory=NullMonitor,
        bundle_revalidator=_no_revalidation,
        checkout_witness_builder=_checkout_witness,
        checkout_state_reader=_checkout_state,
    )
    observation = evidence.load_document(root, evidence.OBSERVATION_NAME)
    record = evidence.load_document(root, evidence.CONTROLLER_NAME)
    assert selection.terminal == verifier.partition.EXECUTION_INVALID_TERMINAL
    assert observation["layer_p_certificate_matches_freeze"] is True
    assert observation["bound_input_mutated"] is False
    assert record["certificate_mismatch_reasons"] == []
    assert (
        record["predicate_ownership"]["execution_checkout_hygiene"]["passed"] is False
    )
    assert verifier.verify_evidence_root(root)["terminal"] == selection.terminal


def test_slot_normalization_sorts_by_slot_and_rejects_duplicates() -> None:
    assert child.normalize_active_pairs([[5, 8], [9, 2]], frame_id=7) == [
        [9, 2],
        [5, 8],
    ]
    with pytest.raises(child.ChildError, match="duplicate slot"):
        child.normalize_active_pairs([[5, 2], [9, 2]], frame_id=7)


def test_child_vector_is_h2_specific_and_binds_one_invocation(tmp_path: Path) -> None:
    invocation = (tmp_path / "invocation.json").resolve()
    vector = controller.child_argv(invocation)
    assert vector[-2:] == ("--invocation", invocation.as_posix())
    assert vector[3].endswith("run_h2_measurement_child.py")
    assert "run_h0_phase_a_child.py" not in vector


def test_clean_controller_archive_verifies_as_phase_a_progression(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    bound = tmp_path / "bound"
    bound.write_text("frozen\n", encoding="utf-8")
    launched: list[str] = []

    def launch(
        invocation_path: Path,
        environment: Mapping[str, str],
        **_: object,
    ) -> int:
        invocation = evidence.load_document(
            invocation_path.parent, invocation_path.name
        )
        launched.append(str(invocation["run_id"]))
        return child.execute_child(
            invocation_path,
            environment=environment,
            runner=_products,
        )

    root, selection = controller.execute_controller(
        bundle,
        authorization=_authorization(tmp_path, bundle),
        evidence_parent=tmp_path / "evidence",
        bound_paths=(bound,),
        require_clean_checkout=False,
        launch_probe=_probe,
        launch_child=launch,
        monitor_factory=NullMonitor,
        bundle_revalidator=_no_revalidation,
        checkout_witness_builder=_checkout_witness,
        checkout_state_reader=_checkout_state,
        inherited_environment={},
    )
    assert selection.terminal is None
    assert verifier.verify_evidence_root(root)["result"] == selection.result
    assert launched == list(evidence.RUN_IDS)
    assert sorted(
        path.name
        for path in (root / evidence.RUNS_DIR / controller.SEQUENCE).iterdir()
        if path.is_dir()
    ) == list(evidence.RUN_IDS)


def test_repo_owned_archive_is_reachable_with_production_hygiene_enabled(
    tmp_path: Path,
) -> None:
    assert controller.checkout_hygiene_reasons() == ()
    bundle = _bundle(tmp_path)
    authorization = _authorization(tmp_path, bundle, identity_suffix="-repo")
    expected_root = (
        _REPO / evidence.EVIDENCE_REL / evidence.phase_a_root_name(bundle.head)
    )
    marker = (
        controller.default_authorization_ledger()
        / f"{authorization.record['authorization_id']}.json"
    )
    assert not expected_root.exists()
    assert not marker.exists()
    try:
        root, selection = controller.execute_controller(
            bundle,
            authorization=authorization,
            bound_paths=(bundle.runtime_manifest_path,),
            launch_probe=_probe,
            launch_child=_launcher(),
            monitor_factory=NullMonitor,
            bundle_revalidator=_no_revalidation,
            checkout_witness_builder=_checkout_witness,
            checkout_state_reader=_checkout_state,
            inherited_environment={},
        )
        report = verifier.verify_evidence_root(root)
        lifecycle = (root / evidence.LIFECYCLE_NAME).read_text(encoding="utf-8")
        assert selection.terminal is None
        assert report["result"] == "measurement_pass"
        assert lifecycle.count('"event":"child_launch"') == len(evidence.RUN_IDS)
        assert '"event":"stop_boundary_recorded"' in lifecycle
    finally:
        if expected_root.is_dir():
            shutil.rmtree(expected_root)
        if marker.is_file():
            marker.unlink()


def test_capture_inequality_selects_terminal_2(tmp_path: Path) -> None:
    root, selection = _execute(tmp_path, perturbed_run=evidence.CAPTURE_ON_RUNS[0])
    assert selection.terminal == verifier.partition.TERMINALS[1].name
    assert verifier.verify_evidence_root(root)["terminal"] == selection.terminal


def test_invalid_packet_selects_terminal_3(tmp_path: Path) -> None:
    root, selection = _execute(tmp_path, invalid_run=evidence.CAPTURE_ON_RUNS[0])
    observation = evidence.load_document(
        root, evidence.OBSERVATION_NAME, schema=evidence.OBSERVATION_SCHEMA
    )
    assert observation["capture_off_on_equal"] is True
    assert observation["packets_valid"] is False
    assert observation["execution_complete"] is True
    assert selection.terminal == verifier.partition.TERMINALS[2].name
    assert verifier.verify_evidence_root(root)["terminal"] == selection.terminal


def test_policy_inequality_has_priority_over_invalid_packet(tmp_path: Path) -> None:
    run_id = evidence.CAPTURE_ON_RUNS[0]
    root, selection = _execute(
        tmp_path,
        perturbed_run=run_id,
        invalid_run=run_id,
    )
    assert selection.terminal == verifier.partition.TERMINALS[1].name
    assert verifier.verify_evidence_root(root)["terminal"] == selection.terminal


def test_packet_derived_projection_failure_selects_terminal_3_not_terminal_2(
    tmp_path: Path,
) -> None:
    root, selection = _execute(
        tmp_path,
        projection_mismatch_run=evidence.CAPTURE_ON_RUNS[0],
    )
    observation = evidence.load_document(
        root, evidence.OBSERVATION_NAME, schema=evidence.OBSERVATION_SCHEMA
    )
    assert observation["capture_off_on_equal"] is True
    assert observation["packets_valid"] is False
    assert selection.terminal == verifier.partition.TERMINALS[2].name
    assert verifier.verify_evidence_root(root)["terminal"] == selection.terminal


@pytest.mark.parametrize(
    "failure",
    (
        OverflowError("overflow"),
        struct.error("binary layout"),
        KeyError("missing"),
    ),
)
def test_capture_processing_persists_raw_packet_and_classifies_structural_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    capture = {"capture_schema_version": "malformed"}

    def reject(_capture: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        raise failure

    monkeypatch.setattr(child, "_projection_records", reject)
    outcome = child.persist_and_process_capture(tmp_path, capture)
    assert outcome.verification == {"failure": "packet_invalid", "state": "fail"}
    assert evidence.load_document(tmp_path, evidence.PACKET_NAME) == capture
    assert (
        evidence.load_document(tmp_path, evidence.PACKET_VERIFICATION_NAME)
        == outcome.verification
    )


def test_capture_processing_does_not_classify_implementation_errors_as_packet_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = {"capture_schema_version": "malformed"}

    def crash(_capture: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        raise RuntimeError("implementation defect")

    monkeypatch.setattr(child, "_projection_records", crash)
    with pytest.raises(RuntimeError, match="implementation defect"):
        child.persist_and_process_capture(tmp_path, capture)
    assert evidence.load_document(tmp_path, evidence.PACKET_NAME) == capture
    assert not (tmp_path / evidence.PACKET_VERIFICATION_NAME).exists()


@pytest.mark.parametrize(
    ("perturbed", "terminal_index"),
    ((False, 2), (True, 1)),
)
def test_child_nonzero_replays_surviving_base_and_invalid_packet(
    tmp_path: Path, perturbed: bool, terminal_index: int
) -> None:
    bundle = _bundle(tmp_path)
    failed_run = evidence.CAPTURE_ON_RUNS[0]

    def launch(
        invocation_path: Path,
        environment: Mapping[str, str],
        **_: object,
    ) -> int:
        invocation = evidence.load_document(
            invocation_path.parent, invocation_path.name
        )

        def fail_after_packet(current: Mapping[str, Any]) -> child.RunProducts:
            produced = _products(
                current,
                perturbed_run=failed_run if perturbed else None,
            )
            child._persist_base_products(
                Path(str(current["run_dir"])),
                str(current["sequence"]),
                produced.mot_bytes,
                produced.policy_base_inventory,
            )
            child.persist_and_process_capture(
                Path(str(current["run_dir"])),
                {"capture_schema_version": "invalid"},
            )
            raise RuntimeError("failure after durable packet evidence")

        try:
            return child.execute_child(
                invocation_path,
                environment=environment,
                runner=(
                    fail_after_packet
                    if invocation["run_id"] == failed_run
                    else _products
                ),
            )
        except RuntimeError:
            return 2

    root, selection = controller.execute_controller(
        bundle,
        authorization=_authorization(tmp_path, bundle),
        evidence_parent=tmp_path / "evidence",
        bound_paths=(bundle.runtime_manifest_path,),
        require_clean_checkout=False,
        launch_probe=_probe,
        launch_child=launch,
        monitor_factory=NullMonitor,
        bundle_revalidator=_no_revalidation,
        checkout_witness_builder=_checkout_witness,
        checkout_state_reader=_checkout_state,
        inherited_environment={},
    )
    assert selection.terminal == verifier.partition.TERMINALS[terminal_index].name
    assert verifier.verify_evidence_root(root)["terminal"] == selection.terminal


def test_mot_only_survivor_selects_terminal_2_when_base_json_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path)
    failed_run = evidence.CAPTURE_ON_RUNS[0]
    real_write_document = evidence.write_document

    def launch(
        invocation_path: Path,
        environment: Mapping[str, str],
        **_: object,
    ) -> int:
        invocation = evidence.load_document(
            invocation_path.parent, invocation_path.name
        )
        if invocation["run_id"] != failed_run:
            return child.execute_child(
                invocation_path,
                environment=environment,
                runner=_products,
            )

        def fail_mid_base_write(
            root: Path, name: str, payload: Mapping[str, Any]
        ) -> Path:
            if name != evidence.BASE_POLICY_INVENTORY_NAME:
                return real_write_document(root, name, payload)
            real_os_write = os.write
            calls = 0

            def partial_then_fail(descriptor: int, data: Any) -> int:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return real_os_write(descriptor, data[: max(1, len(data) // 2)])
                raise OSError("injected base inventory write failure")

            monkeypatch.setattr(os, "write", partial_then_fail)
            try:
                return real_write_document(root, name, payload)
            finally:
                monkeypatch.setattr(os, "write", real_os_write)

        monkeypatch.setattr(evidence, "write_document", fail_mid_base_write)
        try:
            return child.execute_child(
                invocation_path,
                environment=environment,
                runner=lambda current: _products(
                    current,
                    perturbed_run=failed_run,
                ),
            )
        except OSError:
            return 2
        finally:
            monkeypatch.setattr(evidence, "write_document", real_write_document)

    root, selection = controller.execute_controller(
        bundle,
        authorization=_authorization(tmp_path, bundle),
        evidence_parent=tmp_path / "evidence",
        bound_paths=(bundle.runtime_manifest_path,),
        require_clean_checkout=False,
        launch_probe=_probe,
        launch_child=launch,
        monitor_factory=NullMonitor,
        bundle_revalidator=_no_revalidation,
        checkout_witness_builder=_checkout_witness,
        checkout_state_reader=_checkout_state,
        inherited_environment={},
    )
    run_dir = evidence.run_dir(root, controller.SEQUENCE, failed_run)
    observation = evidence.load_document(
        root, evidence.OBSERVATION_NAME, schema=evidence.OBSERVATION_SCHEMA
    )
    assert (run_dir / f"{controller.SEQUENCE}.txt").is_file()
    assert not (run_dir / evidence.BASE_POLICY_INVENTORY_NAME).exists()
    assert not tuple(run_dir.glob(f".{evidence.BASE_POLICY_INVENTORY_NAME}.*.tmp"))
    assert observation["capture_off_on_equal"] is False
    assert observation["execution_complete"] is False
    assert selection.terminal == verifier.partition.TERMINALS[1].name
    assert verifier.verify_evidence_root(root)["terminal"] == selection.terminal


def test_child_nonzero_after_base_without_packet_remains_terminal_4(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    failed_run = evidence.CAPTURE_ON_RUNS[0]

    def launch(
        invocation_path: Path,
        environment: Mapping[str, str],
        **_: object,
    ) -> int:
        invocation = evidence.load_document(
            invocation_path.parent, invocation_path.name
        )

        def fail_after_base(current: Mapping[str, Any]) -> child.RunProducts:
            produced = _products(current)
            child._persist_base_products(
                Path(str(current["run_dir"])),
                str(current["sequence"]),
                produced.mot_bytes,
                produced.policy_base_inventory,
            )
            raise RuntimeError("failure before packet evidence")

        try:
            return child.execute_child(
                invocation_path,
                environment=environment,
                runner=(
                    fail_after_base if invocation["run_id"] == failed_run else _products
                ),
            )
        except RuntimeError:
            return 2

    root, selection = controller.execute_controller(
        bundle,
        authorization=_authorization(tmp_path, bundle),
        evidence_parent=tmp_path / "evidence",
        bound_paths=(bundle.runtime_manifest_path,),
        require_clean_checkout=False,
        launch_probe=_probe,
        launch_child=launch,
        monitor_factory=NullMonitor,
        bundle_revalidator=_no_revalidation,
        checkout_witness_builder=_checkout_witness,
        checkout_state_reader=_checkout_state,
        inherited_environment={},
    )
    observation = evidence.load_document(
        root, evidence.OBSERVATION_NAME, schema=evidence.OBSERVATION_SCHEMA
    )
    assert observation["packets_valid"] is True
    assert observation["execution_complete"] is False
    assert selection.terminal == verifier.partition.EXECUTION_INVALID_TERMINAL
    assert verifier.verify_evidence_root(root)["terminal"] == selection.terminal


def test_certificate_mismatch_stops_before_any_measurement_run(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    bundle.freeze["layer_p_certificate"]["digest"] = "f" * 64
    bound = tmp_path / "bound"
    bound.write_text("frozen\n", encoding="utf-8")
    root, selection = controller.execute_controller(
        bundle,
        authorization=_authorization(tmp_path, bundle),
        evidence_parent=tmp_path / "evidence",
        bound_paths=(bound,),
        require_clean_checkout=False,
        launch_probe=lambda *_args, **_kwargs: pytest.fail("probe must not run"),
        launch_child=lambda *_args, **_kwargs: pytest.fail("child must not run"),
        monitor_factory=NullMonitor,
        bundle_revalidator=_no_revalidation,
        checkout_witness_builder=_checkout_witness,
        checkout_state_reader=_checkout_state,
    )
    assert selection.terminal == verifier.partition.TERMINALS[0].name
    assert not (root / evidence.RUNS_DIR).exists()
    assert verifier.verify_evidence_root(root)["terminal"] == selection.terminal


def test_launch_probe_mismatch_stops_before_measurement_runs(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bound = tmp_path / "bound"
    bound.write_text("frozen\n", encoding="utf-8")

    def moved_probe(root: Path, **_: object) -> dict[str, Any]:
        payload = {
            "schema": behavior.RESULT_SCHEMA,
            "digest": "e" * 64,
            "digests": ["e" * 64],
            "identical": True,
            "mode": "identity",
            "sequence": behavior.IDENTITY_SEQUENCE,
        }
        evidence.write_document(root, evidence.LAUNCH_PROBE_NAME, payload)
        return payload

    root, selection = controller.execute_controller(
        bundle,
        authorization=_authorization(tmp_path, bundle),
        evidence_parent=tmp_path / "evidence",
        bound_paths=(bound,),
        require_clean_checkout=False,
        launch_probe=moved_probe,
        launch_child=lambda *_args, **_kwargs: pytest.fail("child must not run"),
        monitor_factory=NullMonitor,
        bundle_revalidator=_no_revalidation,
        checkout_witness_builder=_checkout_witness,
        checkout_state_reader=_checkout_state,
    )
    assert selection.terminal == verifier.partition.TERMINALS[0].name
    assert verifier.verify_evidence_root(root)["terminal"] == selection.terminal


def test_child_failure_is_a_verifiable_terminal_4_envelope(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bound = tmp_path / "bound"
    bound.write_text("frozen\n", encoding="utf-8")

    def failed_child(*_args: object, **_kwargs: object) -> int:
        return 2

    root, selection = controller.execute_controller(
        bundle,
        authorization=_authorization(tmp_path, bundle),
        evidence_parent=tmp_path / "evidence",
        bound_paths=(bound,),
        require_clean_checkout=False,
        launch_probe=_probe,
        launch_child=failed_child,
        monitor_factory=NullMonitor,
        bundle_revalidator=_no_revalidation,
        checkout_witness_builder=_checkout_witness,
        checkout_state_reader=_checkout_state,
    )
    assert selection.terminal == verifier.partition.EXECUTION_INVALID_TERMINAL
    assert verifier.verify_evidence_root(root)["terminal"] == selection.terminal


def test_bound_input_drift_has_priority_over_execution_failure(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    bound = tmp_path / "bound"
    bound.write_text("frozen\n", encoding="utf-8")

    class DriftMonitor(NullMonitor):
        def __init__(self, *_: object, **__: object) -> None:
            super().__init__()
            self.history = [
                type(
                    "Event",
                    (),
                    {
                        "classification": "bound_mutation",
                        "mask": 2,
                        "path": bound.as_posix(),
                    },
                )()
            ]

    def drift(*_args: object, **_kwargs: object) -> int:
        raise controller.h0_controller.DriftError("mutated")

    root, selection = controller.execute_controller(
        bundle,
        authorization=_authorization(tmp_path, bundle),
        evidence_parent=tmp_path / "evidence",
        bound_paths=(bound,),
        require_clean_checkout=False,
        launch_probe=_probe,
        launch_child=drift,
        monitor_factory=DriftMonitor,
        bundle_revalidator=_no_revalidation,
        checkout_witness_builder=_checkout_witness,
        checkout_state_reader=_checkout_state,
    )
    assert selection.terminal == verifier.partition.TERMINALS[0].name
    assert verifier.verify_evidence_root(root)["terminal"] == selection.terminal


def test_monitor_precedes_revalidation_and_inputs_are_revalidated_post_run(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    bound = tmp_path / "bound"
    bound.write_text("frozen\n", encoding="utf-8")
    events: list[str] = []

    class OrderedMonitor(NullMonitor):
        def __init__(self, *_: object, **__: object) -> None:
            super().__init__()
            events.append("monitor")

        def close(self) -> None:
            events.append("close")
            super().close()

    def revalidate(_bundle: controller.LaunchBundle) -> tuple[str, ...]:
        events.append("revalidate")
        return ()

    def probe(root: Path, **kwargs: object) -> dict[str, Any]:
        events.append("probe")
        return _probe(root, **kwargs)

    def launch(
        invocation_path: Path,
        environment: Mapping[str, str],
        **_: object,
    ) -> int:
        events.append(f"run:{invocation_path.parent.name}")
        return child.execute_child(
            invocation_path,
            environment=environment,
            runner=_products,
        )

    _root, selection = controller.execute_controller(
        bundle,
        authorization=_authorization(tmp_path, bundle),
        evidence_parent=tmp_path / "evidence",
        bound_paths=(bound,),
        require_clean_checkout=False,
        launch_probe=probe,
        launch_child=launch,
        monitor_factory=OrderedMonitor,
        bundle_revalidator=revalidate,
        checkout_witness_builder=_checkout_witness,
        checkout_state_reader=_checkout_state,
        inherited_environment={},
    )
    assert selection.terminal is None
    assert events == [
        "monitor",
        "revalidate",
        "probe",
        *(f"run:{run_id}" for run_id in evidence.RUN_IDS),
        "revalidate",
        "close",
    ]


def test_post_run_revalidation_failure_has_terminal_1_priority(
    tmp_path: Path,
) -> None:
    calls = 0

    def moved_after_runs(_bundle: controller.LaunchBundle) -> tuple[str, ...]:
        nonlocal calls
        calls += 1
        return () if calls == 1 else ("runtime input moved",)

    bundle = _bundle(tmp_path)
    bound = tmp_path / "bound"
    bound.write_text("frozen\n", encoding="utf-8")
    root, selection = controller.execute_controller(
        bundle,
        authorization=_authorization(tmp_path, bundle),
        evidence_parent=tmp_path / "evidence",
        bound_paths=(bound,),
        require_clean_checkout=False,
        launch_probe=_probe,
        launch_child=_launcher(),
        monitor_factory=NullMonitor,
        bundle_revalidator=moved_after_runs,
        checkout_witness_builder=_checkout_witness,
        checkout_state_reader=_checkout_state,
        inherited_environment={},
    )
    assert calls == 2
    assert selection.terminal == verifier.partition.TERMINALS[0].name
    assert verifier.verify_evidence_root(root)["terminal"] == selection.terminal


def test_final_drain_mutation_beats_a_nonzero_child(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bound = tmp_path / "bound"
    bound.write_text("frozen\n", encoding="utf-8")

    class LateMutationMonitor(NullMonitor):
        def drain(self) -> list[Any]:
            if self.history:
                return []
            event = type(
                "Event",
                (),
                {
                    "classification": "bound_mutation",
                    "mask": 2,
                    "path": bound.as_posix(),
                },
            )()
            self.history.append(event)
            return [event]

    root, selection = controller.execute_controller(
        bundle,
        authorization=_authorization(tmp_path, bundle),
        evidence_parent=tmp_path / "evidence",
        bound_paths=(bound,),
        require_clean_checkout=False,
        launch_probe=_probe,
        launch_child=lambda *_args, **_kwargs: 2,
        monitor_factory=LateMutationMonitor,
        bundle_revalidator=_no_revalidation,
        checkout_witness_builder=_checkout_witness,
        checkout_state_reader=_checkout_state,
    )
    assert selection.terminal == verifier.partition.TERMINALS[0].name
    assert verifier.verify_evidence_root(root)["terminal"] == selection.terminal


def test_mutation_after_clean_final_drain_is_outside_the_invocation(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    original = bundle.runtime_manifest_path.read_bytes()
    calls = 0

    def revalidate(current: controller.LaunchBundle) -> tuple[str, ...]:
        nonlocal calls
        calls += 1
        return (
            ()
            if current.runtime_manifest_path.read_bytes() == original
            else ("runtime-input manifest changed",)
        )

    class MutatingCloseMonitor(NullMonitor):
        def close(self) -> None:
            bundle.runtime_manifest_path.write_bytes(original + b" ")
            super().close()

    root, selection = controller.execute_controller(
        bundle,
        authorization=_authorization(tmp_path, bundle),
        evidence_parent=tmp_path / "evidence",
        bound_paths=(bundle.runtime_manifest_path,),
        require_clean_checkout=False,
        launch_probe=_probe,
        launch_child=_launcher(),
        monitor_factory=MutatingCloseMonitor,
        bundle_revalidator=revalidate,
        checkout_witness_builder=_checkout_witness,
        checkout_state_reader=_checkout_state,
        inherited_environment={},
    )
    stop = evidence.load_document(
        root, evidence.STOP_BOUNDARY_NAME, schema=evidence.STOP_BOUNDARY_SCHEMA
    )
    assert calls == 2
    assert stop["monitor_closed"] is True
    assert stop["linearization"] == "clean_final_drain"
    assert stop["revalidation_reasons"] == []
    assert bundle.runtime_manifest_path.read_bytes() != original
    assert selection.terminal is None
    assert verifier.verify_evidence_root(root)["terminal"] is None


def test_mutation_during_sequential_revalidation_is_caught_by_final_drain(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    original = bundle.runtime_manifest_path.read_bytes()
    calls = 0

    class RevalidationMutationMonitor(NullMonitor):
        def __init__(self, *_: object, **__: object) -> None:
            super().__init__()
            self.returned = False

        def drain(self) -> list[Any]:
            if self.returned:
                return []
            self.returned = True
            return list(self.history)

    monitors: list[RevalidationMutationMonitor] = []

    def revalidate(_current: controller.LaunchBundle) -> tuple[str, ...]:
        nonlocal calls
        calls += 1
        if calls == 2:
            bundle.runtime_manifest_path.write_bytes(original + b" ")
            bundle.runtime_manifest_path.write_bytes(original)
            monitors[0].history.append(
                type(
                    "Event",
                    (),
                    {
                        "classification": "bound_mutation",
                        "mask": 2,
                        "path": bundle.runtime_manifest_path.as_posix(),
                    },
                )()
            )
        return ()

    def factory(*args: object, **kwargs: object) -> RevalidationMutationMonitor:
        current = RevalidationMutationMonitor(*args, **kwargs)
        monitors.append(current)
        return current

    root, selection = controller.execute_controller(
        bundle,
        authorization=_authorization(tmp_path, bundle),
        evidence_parent=tmp_path / "evidence",
        bound_paths=(bundle.runtime_manifest_path,),
        require_clean_checkout=False,
        launch_probe=_probe,
        launch_child=_launcher(),
        monitor_factory=factory,
        bundle_revalidator=revalidate,
        checkout_witness_builder=_checkout_witness,
        checkout_state_reader=_checkout_state,
        inherited_environment={},
    )
    stop = evidence.load_document(
        root, evidence.STOP_BOUNDARY_NAME, schema=evidence.STOP_BOUNDARY_SCHEMA
    )
    assert calls == 2
    assert stop["revalidation_completed_while_monitored"] is True
    assert stop["final_drain_completed"] is True
    assert stop["linearization"] is None
    assert selection.terminal == verifier.partition.TERMINALS[0].name
    assert verifier.verify_evidence_root(root)["terminal"] == selection.terminal


def test_controller_files_remain_plumbing_only() -> None:
    import h2_path_partition as path_partition

    assert (
        path_partition.classify("scripts/tools/run_h2_measurement.py")
        == "plumbing_only"
    )
    assert (
        path_partition.classify("scripts/tools/run_h2_measurement_child.py")
        == "plumbing_only"
    )
