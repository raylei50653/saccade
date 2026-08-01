"""The H2 S4 controller must produce exactly what the independent verifier reads."""

# scope: tracking, system
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
import importlib.util
import json
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
# `mot17_args` is otherwise importable only after something in the session has
# run `behavior._import_eval_stack()`, which made the tests that read the real
# configuration producer pass or fail depending on selection order.
_EVAL = _REPO / "scripts" / "eval"
if _EVAL.as_posix() not in sys.path:
    sys.path.insert(0, _EVAL.as_posix())

import build_runtime_identity as identity  # noqa: E402
import h2_behavioral_identity as behavior  # noqa: E402
import h2_measurement_evidence as evidence  # noqa: E402
import h2_run_spec as run_spec  # noqa: E402
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
    authorization_ledger: Path | None = None,
) -> controller.AuthorizationGrant:
    ledger = authorization_ledger or (tmp_path / "ledger")
    execution_domain = evidence.digest(evidence.authorization_execution_domain(ledger))
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
        "execution_domain": execution_domain,
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
        authorization_ledger=ledger,
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
        **kwargs: object,
    ) -> int:
        on_started = kwargs["on_started"]
        assert callable(on_started)
        on_started()

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


def _returncode_launcher(returncode: int):
    def launch(*_args: object, **kwargs: object) -> int:
        on_started = kwargs["on_started"]
        assert callable(on_started)
        on_started()
        return returncode

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
        authorization_ledger=tmp_path / "ledger",
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
            authorization_ledger=tmp_path / "ledger",
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
            authorization_ledger=tmp_path / "ledger",
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


def test_default_authorization_ledger_rejects_replay_across_clones(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_home = (tmp_path / "controlled-host-state").resolve()
    clone_a = tmp_path / "clone-a"
    clone_b = tmp_path / "clone-b"
    clone_a.mkdir()
    clone_b.mkdir()
    monkeypatch.setenv("XDG_STATE_HOME", state_home.as_posix())
    monkeypatch.setattr(controller, "REPO_ROOT", clone_a)
    ledger_a = controller.default_authorization_ledger()
    bundle = _bundle(tmp_path)
    grant = _authorization(
        tmp_path,
        bundle,
        identity_suffix="-cross-clone",
        authorization_ledger=ledger_a,
    )
    receipt = controller._consume_authorization(
        grant,
        bundle=bundle,
        ledger=ledger_a,
    )

    monkeypatch.setattr(controller, "REPO_ROOT", clone_b)
    ledger_b = controller.default_authorization_ledger()
    assert ledger_b == ledger_a
    assert grant.record["execution_domain"] == receipt["execution_domain"]
    with pytest.raises(controller.ControllerError, match="already consumed"):
        controller._consume_authorization(
            grant,
            bundle=bundle,
            ledger=ledger_b,
        )
    with pytest.raises(controller.ControllerError, match="another"):
        controller.load_authorization(
            grant.path,
            bundle,
            invocation_id=str(grant.record["invocation_id"]),
            authorization_ledger=tmp_path / "different-domain-ledger",
        )


def test_prior_default_ledger_marker_does_not_block_a_new_authorization(
    tmp_path: Path,
) -> None:
    assert controller.checkout_hygiene_reasons() == ()
    bundle = _bundle(tmp_path)
    ledger = controller.default_authorization_ledger()
    grant_a = _authorization(
        tmp_path,
        bundle,
        identity_suffix="-ledger-a",
        authorization_ledger=ledger,
    )
    grant_b = _authorization(
        tmp_path,
        bundle,
        identity_suffix="-ledger-b",
        authorization_ledger=ledger,
    )
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
        authorization_ledger=tmp_path / "ledger",
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


# -- the two observation surfaces are separate evidence ---------------------- #


class _FakeTensor:
    """The two attributes the recorder uses: slicing, then `.numpy()`."""

    def __init__(self, values: list[list[float]] | list[float]) -> None:
        self._values = values

    def __getitem__(self, item: slice) -> "_FakeTensor":
        return _FakeTensor(self._values[item])

    def numpy(self) -> list[Any]:
        return self._values


def _track_results(count: int) -> dict[str, Any]:
    return {
        "boxes": _FakeTensor([[1.0, 2.0, 5.0, 9.0] for _ in range(count)]),
        "classes": None,
        "count": count,
        "scores": _FakeTensor([0.9 for _ in range(count)]),
    }


def _emitted_line(frame: int, track_id: int) -> str:
    return f"{frame},{track_id},1.0,2.0,4.0,7.0,0.9,-1,-1,-1"


def _recorded(frames: dict[int, list[int]]) -> list[dict[str, Any]]:
    """Record the raw evidence for one emission per frame, as the child does."""
    rows: list[dict[str, Any]] = []
    for frame, track_ids in frames.items():
        lines = [_emitted_line(frame, track_id) for track_id in track_ids]
        child.record_raw_emission(
            rows,
            lines=lines,
            track_results=_track_results(len(lines)),
            frame=frame,
            person_class=0,
        )
    return rows


def test_interpolated_rows_do_not_invalidate_the_raw_emission_evidence() -> None:
    """The sequence callback may deliver rows no emission ever produced.

    Interpolation fills gaps after the last emission, so the callback carries rows
    with frames the raw evidence never saw. Both records stand: A7.6 compares
    `final_track_rows` and `mot_output` capture-off to capture-on, each on its own,
    and asks for no projection between them.
    """
    rows = _recorded({1: [7], 3: [7]})
    interpolated = (
        _emitted_line(1, 7),
        _emitted_line(2, 7),  # the interpolated row
        _emitted_line(3, 7),
    )
    assert len(child.canonical_callback_bytes(interpolated).splitlines()) == 3
    assert len(rows) == 2


def test_filtered_rows_do_not_invalidate_the_raw_emission_evidence() -> None:
    """The quality filter removes whole tracklets after they were emitted."""
    rows = _recorded({1: [7, 8], 2: [7, 8]})
    filtered = (_emitted_line(1, 7), _emitted_line(2, 7))
    assert child.canonical_callback_bytes(filtered)
    assert len(rows) == 4


def test_remapped_ids_do_not_invalidate_the_raw_emission_evidence() -> None:
    """Deferred alias resolution renames ids the raw rows recorded under."""
    rows = _recorded({1: [7], 2: [7]})
    remapped = (_emitted_line(1, 42), _emitted_line(2, 42))
    assert child.canonical_callback_bytes(remapped)
    assert {row["track_id"] for row in rows} == {7}


def test_an_emission_that_disagrees_with_its_own_track_results_still_fails() -> None:
    """Retiring the cross-boundary equality did not retire the local one.

    At the moment of emission the boxes still in `track_results` are the ones that
    produced exactly those lines, so a disagreement there is a broken recorder and
    not a later transformation.
    """
    rows: list[dict[str, Any]] = []
    with pytest.raises(child.ChildError, match="cardinality"):
        child.record_raw_emission(
            rows,
            lines=[_emitted_line(1, 7)],
            track_results=_track_results(2),
            frame=1,
            person_class=0,
        )
    with pytest.raises(child.ChildError, match="disagree"):
        child.record_raw_emission(
            rows,
            lines=[_emitted_line(9, 7)],
            track_results=_track_results(1),
            frame=1,
            person_class=0,
        )
    assert rows == []


def test_the_callback_still_refuses_a_non_canonical_row() -> None:
    for lines in ((("1,7,1.0,2.0,4.0,7.0,0.9,-1,-1"),), (("x,7,1,2,3,4,5,-1,-1,-1"),)):
        with pytest.raises(child.ChildError, match="non-canonical MOT result row"):
            child.canonical_callback_bytes(lines)


def test_the_child_holds_no_equality_between_the_two_surfaces() -> None:
    """The retired assertion, pinned as retired.

    Reads the source rather than the behaviour, because the defect was a
    comparison that existed at all: any reintroduction under another name would
    reproduce it.
    """
    source = (_TOOLS / "run_h2_measurement_child.py").read_text(encoding="utf-8")
    assert "differ from callback order" not in source
    assert "callback_rows" not in source


def test_child_vector_is_h2_specific_and_binds_one_invocation(tmp_path: Path) -> None:
    invocation = (tmp_path / "invocation.json").resolve()
    vector = controller.child_argv(invocation)
    assert vector[-2:] == ("--invocation", invocation.as_posix())
    # The bootstrap, not the child directly: the import recorder has to be
    # running before the child module's own top-level imports resolve, and by
    # the time any statement in that file executes they already have.
    assert vector[3].endswith("h2_child_bootstrap.py")
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
        **kwargs: object,
    ) -> int:
        on_started = kwargs["on_started"]
        assert callable(on_started)
        on_started()
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
        authorization_ledger=tmp_path / "ledger",
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
    ledger = controller.default_authorization_ledger()
    authorization = _authorization(
        tmp_path,
        bundle,
        identity_suffix="-repo",
        authorization_ledger=ledger,
    )
    expected_root = (
        _REPO / evidence.EVIDENCE_REL / evidence.phase_a_root_name(bundle.head)
    )
    marker = ledger / f"{authorization.record['authorization_id']}.json"
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
        **kwargs: object,
    ) -> int:
        on_started = kwargs["on_started"]
        assert callable(on_started)
        on_started()
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
        authorization_ledger=tmp_path / "ledger",
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
        **kwargs: object,
    ) -> int:
        on_started = kwargs["on_started"]
        assert callable(on_started)
        on_started()
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
        authorization_ledger=tmp_path / "ledger",
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
        **kwargs: object,
    ) -> int:
        on_started = kwargs["on_started"]
        assert callable(on_started)
        on_started()
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
        authorization_ledger=tmp_path / "ledger",
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
        authorization_ledger=tmp_path / "ledger",
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
        authorization_ledger=tmp_path / "ledger",
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
        on_started = _kwargs["on_started"]
        assert callable(on_started)
        on_started()
        return 2

    root, selection = controller.execute_controller(
        bundle,
        authorization=_authorization(tmp_path, bundle),
        evidence_parent=tmp_path / "evidence",
        authorization_ledger=tmp_path / "ledger",
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


def test_prepare_failure_records_no_child_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)

    def fail_prepare(*_args: object, **_kwargs: object) -> tuple[Path, dict[str, str]]:
        raise controller.ControllerError("injected prepare failure")

    monkeypatch.setattr(controller, "_prepare_run", fail_prepare)
    root, selection = controller.execute_controller(
        bundle,
        authorization=_authorization(tmp_path, bundle),
        evidence_parent=tmp_path / "evidence",
        authorization_ledger=tmp_path / "ledger",
        bound_paths=(bundle.runtime_manifest_path,),
        require_clean_checkout=False,
        launch_probe=_probe,
        launch_child=lambda *_args, **_kwargs: pytest.fail(
            "launcher must not be called"
        ),
        monitor_factory=NullMonitor,
        bundle_revalidator=_no_revalidation,
        checkout_witness_builder=_checkout_witness,
        checkout_state_reader=_checkout_state,
        inherited_environment={},
    )
    lifecycle = (root / evidence.LIFECYCLE_NAME).read_text(encoding="utf-8")
    assert selection.terminal == verifier.partition.EXECUTION_INVALID_TERMINAL
    assert '"event":"child_launch"' not in lifecycle
    assert '"event":"child_completed"' not in lifecycle
    assert verifier.verify_evidence_root(root)["terminal"] == selection.terminal


def test_popen_failure_records_no_child_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    popen_calls = 0
    real_popen = controller.subprocess.Popen

    def fail_popen(*args: object, **kwargs: object) -> Any:
        nonlocal popen_calls
        vector = args[0] if args else kwargs.get("args")
        if (
            isinstance(vector, (list, tuple))
            and vector
            and Path(str(vector[0])).name.startswith("python")
            and any(str(member).endswith("h2_child_bootstrap.py") for member in vector)
        ):
            popen_calls += 1
            raise OSError("injected Popen failure")
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(controller.subprocess, "Popen", fail_popen)
    root, selection = controller.execute_controller(
        bundle,
        authorization=_authorization(tmp_path, bundle),
        evidence_parent=tmp_path / "evidence",
        authorization_ledger=tmp_path / "ledger",
        bound_paths=(bundle.runtime_manifest_path,),
        require_clean_checkout=False,
        launch_probe=_probe,
        launch_child=controller.default_child_launcher,
        monitor_factory=NullMonitor,
        bundle_revalidator=_no_revalidation,
        checkout_witness_builder=_checkout_witness,
        checkout_state_reader=_checkout_state,
        inherited_environment={},
    )
    monkeypatch.setattr(controller.subprocess, "Popen", real_popen)
    lifecycle = (root / evidence.LIFECYCLE_NAME).read_text(encoding="utf-8")
    assert popen_calls == 1
    assert selection.terminal == verifier.partition.EXECUTION_INVALID_TERMINAL
    assert '"event":"child_launch"' not in lifecycle
    assert '"event":"child_completed"' not in lifecycle
    assert verifier.verify_evidence_root(root)["terminal"] == selection.terminal


def test_started_child_without_completion_is_a_legal_partial_lifecycle(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)

    def started_then_failed(*_args: object, **kwargs: object) -> int:
        on_started = kwargs["on_started"]
        assert callable(on_started)
        on_started()
        raise OSError("injected failure after process start")

    root, selection = controller.execute_controller(
        bundle,
        authorization=_authorization(tmp_path, bundle),
        evidence_parent=tmp_path / "evidence",
        authorization_ledger=tmp_path / "ledger",
        bound_paths=(bundle.runtime_manifest_path,),
        require_clean_checkout=False,
        launch_probe=_probe,
        launch_child=started_then_failed,
        monitor_factory=NullMonitor,
        bundle_revalidator=_no_revalidation,
        checkout_witness_builder=_checkout_witness,
        checkout_state_reader=_checkout_state,
        inherited_environment={},
    )
    lifecycle = (root / evidence.LIFECYCLE_NAME).read_text(encoding="utf-8")
    assert selection.terminal == verifier.partition.EXECUTION_INVALID_TERMINAL
    assert lifecycle.count('"event":"child_launch"') == 1
    assert '"event":"child_completed"' not in lifecycle
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
        on_started = _kwargs["on_started"]
        assert callable(on_started)
        on_started()
        raise controller.h0_controller.DriftError("mutated")

    root, selection = controller.execute_controller(
        bundle,
        authorization=_authorization(tmp_path, bundle),
        evidence_parent=tmp_path / "evidence",
        authorization_ledger=tmp_path / "ledger",
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
        **kwargs: object,
    ) -> int:
        on_started = kwargs["on_started"]
        assert callable(on_started)
        on_started()
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
        authorization_ledger=tmp_path / "ledger",
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
        authorization_ledger=tmp_path / "ledger",
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
        authorization_ledger=tmp_path / "ledger",
        bound_paths=(bound,),
        require_clean_checkout=False,
        launch_probe=_probe,
        launch_child=_returncode_launcher(2),
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
        authorization_ledger=tmp_path / "ledger",
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
        authorization_ledger=tmp_path / "ledger",
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


# -- ingress authorization authority (declaration Review Correction 4) ------- #
#
# The 2026-07-28 Phase-A attempt died at terminal 4 because `repository_runner`
# re-derived the ingress predicate from the live environment *after* importing
# the eval stack, and cv2 4.11.0 mutates the environment as an import side
# effect.  The registered delta, reproduced verbatim from the registration
# packet's `root_cause_probe.json`.
_IMPORT_SIDE_EFFECT = {
    "QT_QPA_FONTDIR": "/nonexistent/fonts",
    "QT_QPA_PLATFORM_PLUGIN_PATH": "/nonexistent/plugins/platforms",
}


def _launch_fixture(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    """One canonical child invocation, without running the controller."""
    run_dir = tmp_path / "runs" / evidence.MEASUREMENT_SEQUENCE / evidence.RUN_IDS[0]
    for leaf in ("home", "tmp", "xdg-cache"):
        (run_dir / "_env" / leaf).mkdir(parents=True)
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    document = run_spec.build_run_spec()
    environment = controller.child_environment(
        run_dir, build_dir=build_dir, document=document, inherited={}
    )
    invocation = {
        "schema": child.INVOCATION_SCHEMA,
        "build_dir": build_dir.as_posix(),
        "capture_phase": evidence.CAPTURE_PHASE["a"],
        "capture_run_uuid": "fixture-uuid",
        "environment_digest": child._environment_digest(environment),
        "instrumentation_head": "0" * 40,
        "run_spec": document,
        "run_dir": run_dir.as_posix(),
        "run_id": evidence.RUN_IDS[0],
        "sequence": evidence.MEASUREMENT_SEQUENCE,
        "state": "running",
    }
    path = evidence.write_document(run_dir, "invocation.json", invocation)
    return path, environment, run_dir


def test_repository_owned_env_keys_agree_with_the_producer() -> None:
    """The declared set is exactly what `configure_runtime_env` may touch.

    A subset assertion alone would only stop the producer from growing new
    undeclared behaviour; it would let the allowlist itself be widened, and a
    widened allowlist is a gate that waves through the next key.  So every
    configuration must stay inside the set, and their union must exhaust it.
    """
    import argparse

    from mot17_args import configure_runtime_env

    def mutated(environ: dict[str, str], **flags: object) -> set[str]:
        before = dict(environ)
        configure_runtime_env(argparse.Namespace(**flags), environ)
        return {
            key
            for key in set(before) | set(environ)
            if before.get(key) != environ.get(key)
        }

    # `SACCADE_STREAM_MODE` must be present up front or its removal is
    # unobservable.
    configurations = [
        {
            "double_buffer": True,
            "detect_barrier": "event",
            "no_gpu_decode": False,
            "main_nms_graphed": True,
        },
        {
            "double_buffer": False,
            "detect_barrier": None,
            "no_gpu_decode": True,
            "main_nms_graphed": False,
        },
        {
            "double_buffer": False,
            "detect_barrier": "full",
            "no_gpu_decode": False,
            "main_nms_graphed": True,
        },
    ]
    observed: set[str] = set()
    for flags in configurations:
        keys = mutated({"SACCADE_STREAM_MODE": "ptds_probe"}, **flags)
        assert keys <= child.REPOSITORY_OWNED_ENV_KEYS, sorted(
            keys - child.REPOSITORY_OWNED_ENV_KEYS
        )
        observed |= keys
    assert observed == child.REPOSITORY_OWNED_ENV_KEYS


def test_ingress_authorization_is_decided_once_against_the_launch_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The durable statement of the repair, through the real `repository_runner`.

    An injected runner would prove only that `execute_child` judges the snapshot
    before calling it, and the AST test would prove only that today's source has
    no *direct* call — a helper or a rename would slip past both. So the
    production runner is what executes here: only `_import_eval_stack` and the
    detector builder are stubbed, and the stub mutates the live environment the
    way cv2 4.11.0 does. The run must reach the post-configuration sentinel,
    which it cannot do if anything between the import and that point rebuilds
    the ingress predicate from live state.
    """
    import yaml as real_yaml

    invocation_path, environment, run_dir = _launch_fixture(tmp_path)
    build_dir = Path(
        str(evidence.load_document(run_dir, "invocation.json")["build_dir"])
    )

    # The process environment is the sanitized one, so the production runner's
    # own contracts are exercised rather than stepped around — and the injected
    # snapshot still stops being the live environment the moment cv2 lands.
    monkeypatch.setattr(os, "environ", dict(environment))

    judged: list[dict[str, str]] = []
    real_validate = child.validate_environment

    def recording(env: Mapping[str, str], invocation: Mapping[str, Any]) -> None:
        judged.append(dict(env))
        real_validate(env, invocation)

    monkeypatch.setattr(child, "validate_environment", recording)

    class _ReachedTheDetector(Exception):
        """Raised where a real run would start using the GPU."""

    def import_eval_stack() -> tuple[Any, ...]:
        # cv2 4.11.0's registered side effect, plus a leaked shell export for
        # the configuration stage to clear.
        os.environ.update(_IMPORT_SIDE_EFFECT)
        os.environ["LD_LIBRARY_PATH"] = f"{os.environ['LD_LIBRARY_PATH']}:/opt/cv2/lib"
        os.environ["SACCADE_STREAM_MODE"] = "ptds_probe"

        # The real parser, so this stub cannot drift out of step with the
        # fixed execution vector the child now sends through it.
        from mot17_args import build_parser

        def configure_runtime_env(_args: Any, env: Any) -> None:
            env["SACCADE_GPU_DECODE"] = "1"
            env.pop("SACCADE_STREAM_MODE", None)

        def build_detector(**_: object) -> Any:
            raise _ReachedTheDetector()

        return (
            real_yaml,
            build_parser,
            configure_runtime_env,
            lambda _stem: "1" * 64,
            None,
            None,
            None,
            build_detector,
            None,
        )

    monkeypatch.setattr(behavior, "_import_eval_stack", import_eval_stack)
    monkeypatch.setattr(behavior, "resolve_build_dir", lambda: build_dir)
    monkeypatch.setattr(behavior, "_assert_extension_consumed", lambda _dir: "witness")

    with pytest.raises(_ReachedTheDetector):
        child.execute_child(
            invocation_path,
            environment=environment,
            runner=child.repository_runner,
        )

    # Judged exactly once, and on the snapshot — never on the live environment,
    # which by then no longer satisfies the predicate at all.
    assert judged == [environment]
    with pytest.raises(child.ChildError):
        real_validate(
            dict(os.environ), evidence.load_document(run_dir, "invocation.json")
        )
    assert evidence.load_document(run_dir, child.ENVIRONMENT_DELTA_NAME) == {
        "schema": child.ENVIRONMENT_DELTA_SCHEMA,
        "authority": "diagnostic_only",
        "added": [
            "QT_QPA_FONTDIR",
            "QT_QPA_PLATFORM_PLUGIN_PATH",
            "SACCADE_STREAM_MODE",
        ],
        "removed": [],
        "changed": ["LD_LIBRARY_PATH"],
    }


def test_import_delta_document_is_exact_and_carries_no_environment_values(
    tmp_path: Path,
) -> None:
    """Diagnostic only, so it records key names and nothing else."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # Every value is a distinctive sentinel: a low-entropy fixture ("0", "1")
    # would collide with the document's own schema string and make the
    # value-absence assertion below meaningless rather than strict.
    secret = "b3d1f0e2c4a5"
    before = {
        "KEPT": f"kept-{secret}",
        "LD_LIBRARY_PATH": f"build-{secret}",
        "REMOVED": f"removed-{secret}",
    }
    after = {
        "KEPT": f"kept-{secret}",
        "LD_LIBRARY_PATH": f"build-{secret}:cv2-{secret}",
        "QT_QPA_FONTDIR": f"fonts-{secret}",
    }
    document = child.record_import_delta(run_dir, before, after)
    assert document == {
        "schema": "h2_child_environment_import_delta_v1",
        "authority": "diagnostic_only",
        "added": ["QT_QPA_FONTDIR"],
        "removed": ["REMOVED"],
        "changed": ["LD_LIBRARY_PATH"],
    }
    written = (run_dir / child.ENVIRONMENT_DELTA_NAME).read_text(encoding="utf-8")
    assert evidence.load_document(run_dir, child.ENVIRONMENT_DELTA_NAME) == document
    assert secret not in written
    for value in (*before.values(), *after.values()):
        assert value not in written


def test_repository_owned_mutation_gate_rejects_foreign_keys() -> None:
    document = run_spec.build_run_spec()
    baseline = {
        **child.HYGIENE_ENV,
        **run_spec.environment_projection(document),
    }
    applied = {**baseline, "QT_QPA_FONTDIR": "/nonexistent/fonts"}
    with pytest.raises(child.ChildError) as excinfo:
        child.validate_repository_owned_mutation(baseline, applied, document)
    message = str(excinfo.value)
    assert "outside its declared set" in message
    assert "QT_QPA_FONTDIR" in message
    # Never the ingress vocabulary: this contract has its own subject matter.
    assert "frozen A5 execution environment" not in message

    with pytest.raises(child.ChildError):
        child.validate_repository_owned_mutation(
            {**baseline, "PATH": "/usr/bin"}, baseline, document
        )


def test_repository_owned_mutation_gate_is_blind_to_the_import_delta() -> None:
    """The baseline is post-import, so the import's delta is on neither side.

    Taking the launch snapshot as this baseline instead would charge cv2's
    injection to the repository and reproduce the registered failure under a
    new name; the same delta applied *after* the baseline is a violation,
    because by then nothing but `configure_runtime_env` may write.
    """
    document = run_spec.build_run_spec()
    snapshot = {
        **child.HYGIENE_ENV,
        **run_spec.environment_projection(document),
    }
    post_import = {**snapshot, **_IMPORT_SIDE_EFFECT}

    baseline = dict(post_import)
    applied = dict(post_import)
    baseline["SACCADE_STREAM_MODE"] = "ptds_probe"
    child.validate_repository_owned_mutation(baseline, applied, document)

    with pytest.raises(child.ChildError):
        child.validate_repository_owned_mutation(snapshot, post_import, document)


def test_repository_owned_mutation_gate_pins_the_run_spec_values() -> None:
    document = run_spec.build_run_spec()
    baseline = {
        **child.HYGIENE_ENV,
        **run_spec.environment_projection(document),
    }
    with pytest.raises(child.ChildError) as excinfo:
        child.validate_repository_owned_mutation(
            baseline, {**baseline, "SACCADE_GPU_DECODE": "0"}, document
        )
    assert "RunSpec values" in str(excinfo.value)

    with pytest.raises(child.ChildError) as excinfo:
        child.validate_repository_owned_mutation(
            baseline,
            {**baseline, "SACCADE_STREAM_MODE": "ptds_probe"},
            document,
        )
    assert "SACCADE_STREAM_MODE" in str(excinfo.value)


def test_repository_runner_does_not_re_derive_the_ingress_predicate() -> None:
    """The 2026-07-28 defect, stated as source structure.

    Standalone by construction — no fixture, no helper added by this repair —
    so it can be run unmodified against the pre-repair head, where it fails on
    the call at `repository_runner`'s configuration stage.
    """
    import ast

    source = (_TOOLS / "run_h2_measurement_child.py").read_text(encoding="utf-8")
    functions = {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }
    called = {
        node.func.id
        for node in ast.walk(functions["repository_runner"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "validate_environment" not in called
    assert "validate_repository_owned_mutation" in called


def _issued_grant(
    tmp_path: Path,
    bundle: controller.LaunchBundle,
    *,
    issuer: str,
    ledger: Path,
) -> Path:
    """One grant that is legal except for whatever the caller varies."""
    seed = evidence.digest({"issuer": issuer, "path": tmp_path.as_posix()})
    grant = {
        "schema": evidence.AUTHORIZATION_GRANT_SCHEMA,
        "authorization_id": seed,
        "capture_phase": evidence.CAPTURE_PHASE["a"],
        "controller_digest": bundle.freeze["executed_surfaces"][
            "scripts/tools/run_h2_measurement.py"
        ],
        "execution_domain": evidence.digest(
            evidence.authorization_execution_domain(ledger)
        ),
        "freeze_digest": evidence.freeze_digest(bundle.freeze),
        "instrumentation_head": bundle.head,
        "invocation_id": evidence.digest({"invocation": seed}),
        "issued_by": issuer,
    }
    path = tmp_path / f"grant-{seed[:8]}.json"
    _write(path, grant)
    return path


def test_controller_admission_reads_the_issuer_from_the_authority_constant(
    tmp_path, monkeypatch
) -> None:
    """Admission must consult `AUTHORIZATION_ISSUER` at call time.

    The two halves move together only if the controller holds no literal of its
    own: with the authority moved, `research_owner` stops being an issuer and
    the moved value starts being one.
    """
    bundle = _bundle(tmp_path)
    ledger = tmp_path / "ledger"
    moved = "successor_research_owner"
    assert moved != evidence.AUTHORIZATION_ISSUER

    stale = _issued_grant(tmp_path, bundle, issuer="research_owner", ledger=ledger)
    followed = _issued_grant(tmp_path, bundle, issuer=moved, ledger=ledger)

    def _load(path: Path) -> controller.AuthorizationGrant:
        record = evidence.load_document(path.parent, path.name)
        return controller.load_authorization(
            path,
            bundle,
            invocation_id=record["invocation_id"],
            authorization_ledger=ledger,
        )

    assert _load(stale).record["issued_by"] == "research_owner"
    with pytest.raises(controller.ControllerError):
        _load(followed)

    monkeypatch.setattr(evidence, "AUTHORIZATION_ISSUER", moved)
    assert _load(followed).record["issued_by"] == moved
    with pytest.raises(controller.ControllerError):
        _load(stale)


def test_no_production_surface_restates_the_authorization_issuer() -> None:
    """The issuer literal may exist in exactly one place (§ C3.9).

    A second copy in a producer or a validator is a second, silently drifting
    answer to who may authorize a measurement — which is what this constant was
    extracted to prevent. Test fixtures may still spell it out: a fixture is an
    external contract sample, not an authority.
    """
    owner = _TOOLS / "h2_measurement_evidence.py"
    literal = '"research_owner"'
    assert owner.read_text(encoding="utf-8").count(literal) == 1

    for name in evidence.PHASE_A_EXECUTED_SURFACE_PATHS:
        surface = _REPO / name
        if surface == owner:
            continue
        assert literal not in surface.read_text(encoding="utf-8"), name


# -- the sole-authority resolved RunSpec ------------------------------------- #

# The rehearsal at `ba40b3f8` reached `H2_MEASUREMENT_EXECUTION_INVALID` because
# the child took every knob but the sequence and the output from the A5 preset,
# and the preset declares neither `double_buffer` nor `detect_barrier`. The
# parser resolved them to `False`/`None`, and `configure_runtime_env` — which
# documents the parsed arguments as its authority — rewrote the frozen A5 pair
# to `full`/`0`. H0 had always sent its fixed choices through that same surface.


def test_run_spec_issues_the_complete_owner_declared_namespace() -> None:
    profile, _binding = run_spec.load_authoring_profile()
    document = run_spec.build_run_spec()
    resolved = document["resolved_namespace"]
    assert len(resolved) == 454
    assert document["resolved_namespace_keys"] == sorted(resolved)
    assert resolved == profile["resolved_namespace"]
    assert resolved["detector"] is None
    assert resolved["max_frames"] is None
    assert resolved["preset"] is None
    assert resolved["warmup_frames"] == 50
    assert resolved["sequences"] == evidence.MEASUREMENT_SEQUENCE
    assert resolved["output"] == run_spec.RUN_DIR_OUTPUT_TOKEN


def test_run_spec_separates_object_and_artifact_byte_domains(tmp_path: Path) -> None:
    target = tmp_path / "run_spec.json"
    assert run_spec.main(["--emit", target.as_posix()]) == 0

    raw = target.read_bytes()
    document = json.loads(raw)
    object_bytes = run_spec.canonical_json_bytes(document)
    digest_payload = run_spec._run_spec_digest_payload(document)
    digest_bytes = run_spec.canonical_json_bytes(digest_payload)

    assert document["object_canonicalization"] == (
        "utf8_lexicographic_keys_compact_finite_no_trailing_lf_v1"
    )
    assert document["artifact_serialization"] == (
        "utf8_lexicographic_keys_compact_finite_single_trailing_lf_v1"
    )
    assert "canonicalization" not in document
    assert not object_bytes.endswith(b"\n")
    assert raw == object_bytes + b"\n"
    assert (
        document["resolved_run_spec_digest"] == hashlib.sha256(digest_bytes).hexdigest()
    )
    assert (
        document["resolved_run_spec_digest"]
        != hashlib.sha256(digest_bytes + b"\n").hexdigest()
    )


def test_run_spec_authoring_does_not_consult_live_parser_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mot17_args

    expected = run_spec.build_run_spec()

    def forbidden_parser() -> Any:
        raise AssertionError("frozen-profile authoring consulted the live parser")

    monkeypatch.setattr(mot17_args, "build_parser", forbidden_parser)
    assert run_spec.build_run_spec() == expected


def test_runtime_parser_and_environment_are_only_run_spec_projections(
    tmp_path: Path,
) -> None:
    from mot17_args import configure_runtime_env

    document = run_spec.build_run_spec()
    run_dir = tmp_path.resolve()
    args = run_spec.parse_runtime_namespace(document, run_dir)
    baseline = {
        **child.HYGIENE_ENV,
        **run_spec.environment_projection(document),
    }
    environment = dict(baseline)
    configure_runtime_env(args, environment)

    assert environment == baseline
    assert vars(args)["output"] == (run_dir / "_runtime").as_posix()
    assert args.latency_only is True
    run_spec.assert_runtime_matches(document, args, environment, run_dir)


def test_run_spec_gates_full_namespace_and_environment_drift(tmp_path: Path) -> None:
    document = run_spec.build_run_spec()
    run_dir = tmp_path.resolve()
    args = run_spec.parse_runtime_namespace(document, run_dir)
    environment = run_spec.environment_projection(document)

    args.warmup_frames = 0
    with pytest.raises(run_spec.RunSpecError, match="warmup_frames"):
        run_spec.assert_runtime_matches(document, args, environment, run_dir)

    args.warmup_frames = 50
    drifted = {**environment, "SACCADE_DETECT_BARRIER": "full"}
    with pytest.raises(run_spec.RunSpecError, match="SACCADE_DETECT_BARRIER"):
        run_spec.assert_runtime_matches(document, args, drifted, run_dir)


def test_child_has_no_fixed_argv_or_runtime_preset_reload() -> None:
    source = (_TOOLS / "run_h2_measurement_child.py").read_text(encoding="utf-8")
    assert "FIXED_EXECUTION_ARGV" not in source
    repository_runner = source[
        source.index("def repository_runner") : source.index(
            "\ndef validate_products", source.index("def repository_runner")
        )
    ]
    assert "POLICY_PRESET_REL" not in repository_runner
    assert "safe_load" not in repository_runner
    assert repository_runner.count("run_spec.assert_runtime_matches") == 2
    resolver = (_TOOLS / "h2_run_spec.py").read_text(encoding="utf-8")
    assert "resolve_namespace" not in resolver
    assert "_load_preset" not in resolver
    assert "import yaml" not in resolver


def test_the_production_runner_reaches_run_eval_with_the_frozen_choices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The repair, through the real parser and configuration function.

    Here `_import_eval_stack` returns the real `build_parser` and real
    `configure_runtime_env`; the complete defaults come from the invocation's
    RunSpec, and no preset bytes are read at runtime. Only GPU-bearing objects
    are stubs.

    The sentinel sits *after* configuration, at `run_eval`, and refuses anything
    but `latency_only=True` — otherwise this test would prove the environment
    gate was cleared while saying nothing about the boundary immediately behind
    it, which is the next contradiction the same launch would have hit.
    """
    import types

    import yaml as real_yaml

    from mot17_args import build_parser, configure_runtime_env

    invocation_path, environment, run_dir = _launch_fixture(tmp_path)
    build_dir = Path(
        str(evidence.load_document(run_dir, child.INVOCATION_NAME)["build_dir"])
    )
    monkeypatch.setattr(os, "environ", dict(environment))

    seen_args: list[Any] = []
    baselines: list[dict[str, str]] = []

    def recording_configure(args: Any, env: Any) -> None:
        seen_args.append(args)
        baselines.append(dict(env))
        configure_runtime_env(args, env)

    class _ReachedRunEval(Exception):
        """Raised where a real run would start consuming frames."""

    eval_kwargs: dict[str, Any] = {}

    def run_eval(**keywords: Any) -> dict[str, Any]:
        eval_kwargs.update(keywords)
        if keywords.get("latency_only") is not True:
            raise AssertionError(
                "run_eval was reached without latency_only: this run would have "
                "written MOT output and read ground truth"
            )
        raise _ReachedRunEval()

    class _EvalPipeline:
        def __init__(self, *_: Any, **__: Any) -> None:  # pragma: no cover - stub
            raise AssertionError("the pipeline is never constructed here")

    evaluator_module = types.SimpleNamespace(
        run_eval=run_eval,
        _run_frame=lambda *a, **k: None,
        _fast_emit_mot_lines=lambda *a, **k: None,
        EvalPipeline=_EvalPipeline,
    )
    stages_module = types.SimpleNamespace(_fast_emit_mot_lines=lambda *a, **k: None)
    head = types.SimpleNamespace(
        set_head_compile=lambda _value: None, set_block_compile=lambda _value: None
    )
    detector = types.SimpleNamespace(mamba_head=head)

    def import_eval_stack() -> tuple[Any, ...]:
        os.environ.update(_IMPORT_SIDE_EFFECT)
        os.environ["SACCADE_STREAM_MODE"] = "ptds_probe"
        return (
            real_yaml,
            build_parser,
            recording_configure,
            lambda _stem: "1" * 64,
            evaluator_module,
            stages_module,
            _EvalPipeline,
            lambda **_: detector,
            lambda _value: None,
        )

    monkeypatch.setattr(behavior, "_import_eval_stack", import_eval_stack)
    monkeypatch.setattr(behavior, "resolve_build_dir", lambda: build_dir)
    monkeypatch.setattr(behavior, "_assert_extension_consumed", lambda _dir: "witness")

    with pytest.raises(_ReachedRunEval):
        child.execute_child(
            invocation_path,
            environment=environment,
            runner=child.repository_runner,
        )

    document = evidence.load_document(run_dir, child.INVOCATION_NAME)["run_spec"]

    # All 454 choices arrived through the RunSpec-projected parser namespace.
    (args,) = seen_args
    actual = dict(vars(args))
    actual["output"] = run_spec.RUN_DIR_OUTPUT_TOKEN
    assert actual == document["resolved_namespace"]
    assert eval_kwargs["latency_only"] is True

    # Repository configuration preserves all RunSpec-owned environment values.
    for key, value in run_spec.environment_projection(document).items():
        assert os.environ.get(key) == value, key
    (baseline,) = baselines
    mutated = {
        key
        for key in set(baseline) | set(os.environ)
        if baseline.get(key) != os.environ.get(key)
    }
    # The leaked shell export is the one thing configuration is expected to move.
    assert mutated == {"SACCADE_STREAM_MODE"}
    assert "SACCADE_STREAM_MODE" not in os.environ
