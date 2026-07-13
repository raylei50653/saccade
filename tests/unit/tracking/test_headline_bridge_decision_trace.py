"""Outcome-blind contracts for H0's full bridge-decision trace packet."""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "scripts/tools"
sys.path.insert(0, str(TOOLS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPORT = _load("export_headline_bridge_decision_trace")
VERIFY = _load("verify_headline_bridge_decision_trace")
CHECK = _load("check_h0_bridge_decision_trace_contract")


def _scalar(value: float, status: int = 1) -> dict[str, int]:
    return {"bits": struct.unpack("!I", struct.pack("!f", value))[0], "status": status}


def _not_computed() -> dict[str, int]:
    return {"bits": 0, "status": 0}


def _packet(*, run_uuid: str = "raw-run-a") -> dict[str, object]:
    sentinel = _scalar(1e30)
    pair = {
        "schema_version": 1,
        "seq": "MOT17-04-SDP",
        "frame": 12,
        "cand_slot": 4,
        "lost_slot": 3,
        "cand_precommit_track_id": 9,
        "lost_precommit_track_id": 7,
        "cand_instance_uid": (1 << 32) | 4,
        "lost_instance_uid": (1 << 32) | 3,
        "la": 7,
        "bridge_at": 4,
        "cand_ring_length": 4,
        "lost_ring_length": 4,
        "ema_lost": _scalar(80.0),
        "ema_cand": _scalar(80.0),
        "height_ratio": _scalar(1.0),
        "height_verdict": 3,
        "speed": _not_computed(),
        "speed_verdict": 3,
        "spatial_distance": _not_computed(),
        "spatial_verdict": 3,
        "lost_anchor_x": _scalar(100.0),
        "lost_anchor_y": _scalar(200.0),
        "cand_anchor_x": _scalar(116.0),
        "cand_anchor_y": _scalar(200.0),
        "lost_velocity_x": _scalar(0.0),
        "lost_velocity_y": _scalar(0.0),
        "cand_velocity_x": _scalar(0.0),
        "cand_velocity_y": _scalar(0.0),
        "h_ref": _scalar(80.0),
        "fwd_r": _scalar(0.2),
        "bwd_r": _scalar(0.2),
        "dist_h": _scalar(0.2),
        "s_lost": _scalar(0.0),
        "w": _scalar(0.0),
        "direction_cosine": _not_computed(),
        "directional_alpha": _scalar(0.0),
        "directional_cross_bdist": _not_computed(),
        "bdist_before_direction": _scalar(0.2),
        "bdist_after_direction": _scalar(0.2),
        "cutoff_verdict": 1,
        "occupancy_verdict": 3,
        "occupancy_coverage": _not_computed(),
        "appearance_verdict": 3,
        "appearance_cosine": _not_computed(),
        "portable_tail_verdict": 3,
        "portable_tail_mask": 0,
        "final_pair_eligible": 1,
        "reject_reason": 0,
    }
    candidate = {
        "schema_version": 1,
        "seq": pair["seq"],
        "frame": pair["frame"],
        "cand_slot": pair["cand_slot"],
        "cand_precommit_track_id": pair["cand_precommit_track_id"],
        "cand_instance_uid": pair["cand_instance_uid"],
        "structural_competitors": 1,
        "pre_score_passes": 1,
        "final_pair_eligible_count": 1,
        "best_lost_slot": pair["lost_slot"],
        "second_lost_slot": -1,
        "best_lost_precommit_track_id": pair["lost_precommit_track_id"],
        "second_lost_precommit_track_id": -1,
        "best_lost_instance_uid": pair["lost_instance_uid"],
        "second_lost_instance_uid": 0,
        "best_bdist": _scalar(0.2),
        "second_best_bdist": sentinel,
        "margin": sentinel,
        "no_second_competitor": 1,
        "margin_verdict": 3,
        "proposal_emitted": 1,
        "proposal_reject_reason": 0,
        "candidate_status": 4,
    }
    score = _scalar(0.9)
    score_value = struct.unpack("!f", int(score["bits"]).to_bytes(4, "big"))[0]
    sq = int(min(max(score_value, 0.0), 1.0) * 32767.0)
    claim = {
        "schema_version": 1,
        "seq": pair["seq"],
        "frame": pair["frame"],
        "proposing_cand_slot": pair["cand_slot"],
        "proposed_lost_slot": pair["lost_slot"],
        "proposing_cand_precommit_track_id": pair["cand_precommit_track_id"],
        "proposed_lost_precommit_track_id": pair["lost_precommit_track_id"],
        "proposing_cand_instance_uid": pair["cand_instance_uid"],
        "proposed_lost_instance_uid": pair["lost_instance_uid"],
        "detection_score": score,
        "sq": sq,
        "packed_atomic_key": (sq << 16) | pair["cand_slot"],
        "candidate_index_component": pair["cand_slot"],
        "winning_cand_slot": pair["cand_slot"],
        "winning_cand_precommit_track_id": pair["cand_precommit_track_id"],
        "winning_cand_instance_uid": pair["cand_instance_uid"],
        "claim_won": 1,
    }
    commit = {
        "schema_version": 1,
        "seq": pair["seq"],
        "frame": pair["frame"],
        "cand_slot": pair["cand_slot"],
        "lost_slot": pair["lost_slot"],
        "cand_precommit_track_id": pair["cand_precommit_track_id"],
        "lost_precommit_track_id": pair["lost_precommit_track_id"],
        "cand_postcommit_track_id": pair["lost_precommit_track_id"],
        "lost_postcommit_track_id": pair["lost_precommit_track_id"],
        "cand_instance_uid": pair["cand_instance_uid"],
        "lost_instance_uid": pair["lost_instance_uid"],
        "cand_active_before": 1,
        "cand_active_after": 1,
        "lost_active_before": 1,
        "lost_active_after": 0,
        "commit_executed": 1,
        "lost_slot_deactivated": 1,
    }
    candidate_key = {
        name: candidate[name]
        for name in EXPORT.UNIVERSE_STREAMS["native_candidate_keys"]
    }
    pair_key = {
        name: pair[name] for name in EXPORT.UNIVERSE_STREAMS["native_pair_keys"]
    }
    claim_key = {
        name: claim[name] for name in EXPORT.UNIVERSE_STREAMS["native_proposal_keys"]
    }
    commit_key = {
        name: commit[name] for name in EXPORT.UNIVERSE_STREAMS["native_commit_keys"]
    }
    return {
        "capture_schema_version": EXPORT.SCHEMA_VERSION,
        "capture_run_uuid": run_uuid,
        "trace_armed": True,
        "processed_frame_count": 12,
        "bridge_attempt_count": 1,
        "bridge_commit_count": 1,
        "capture_phase": "phase_a",
        "require_candidate_exposure": True,
        "require_commit_exposure": True,
        "pair_records": [pair],
        "candidate_records": [candidate],
        "claim_records": [claim],
        "commit_records": [commit],
        "native_candidate_keys": [candidate_key],
        "native_pair_keys": [pair_key],
        "native_proposal_keys": [claim_key],
        "native_claim_winner_keys": [claim_key],
        "native_commit_keys": [commit_key],
        "total_pair_records": 1,
        "total_candidate_records": 1,
        "total_claim_records": 1,
        "total_commit_records": 1,
        "overflow_pair_records": 0,
        "overflow_candidate_records": 0,
        "overflow_claim_records": 0,
        "overflow_commit_records": 0,
        "total_native_candidate_keys": 1,
        "total_native_pair_keys": 1,
        "total_native_proposal_keys": 1,
        "total_native_claim_winner_keys": 1,
        "total_native_commit_keys": 1,
        "overflow_native_candidate_keys": 0,
        "overflow_native_pair_keys": 0,
        "overflow_native_proposal_keys": 0,
        "overflow_native_claim_winner_keys": 0,
        "overflow_native_commit_keys": 0,
        "identity_uid_wrap_events": 0,
    }


def test_h0_semantic_digest_excludes_run_uuid_and_raw_provenance() -> None:
    first = _packet(run_uuid="raw-run-a")
    second = _packet(run_uuid="raw-run-b")
    second["processed_frame_count"] = 99

    assert EXPORT.semantic_digest(first) == EXPORT.semantic_digest(second)


def test_h0_verifier_replays_full_single_claim_commit() -> None:
    result = VERIFY.verify_capture(_packet())

    assert result["replay"] == "full_commit_decision_trace_v2"
    assert result["stream_totals"] == {
        "pair_records": 1,
        "candidate_records": 1,
        "claim_records": 1,
        "commit_records": 1,
    }
    assert result["native_universe_totals"] == {
        "native_candidate_keys": 1,
        "native_pair_keys": 1,
        "native_proposal_keys": 1,
        "native_claim_winner_keys": 1,
        "native_commit_keys": 1,
    }


def test_h0_packet_rejects_duplicate_stable_key() -> None:
    packet = _packet()
    packet["pair_records"] = [packet["pair_records"][0], packet["pair_records"][0]]
    packet["total_pair_records"] = 2

    with pytest.raises(ValueError, match="duplicate stable keys"):
        EXPORT.canonical_semantic_packet(packet)


def test_h0_packet_rejects_missing_evaluation_frame_identity() -> None:
    packet = _packet()
    packet["pair_records"][0]["frame"] = 0

    with pytest.raises(ValueError, match="non-positive evaluator frame identity"):
        EXPORT.canonical_semantic_packet(packet)


def test_h0_packet_rejects_incomplete_frozen_field_schema() -> None:
    packet = _packet()
    del packet["candidate_records"][0]["margin"]

    with pytest.raises(ValueError, match="field schema mismatch"):
        EXPORT.canonical_semantic_packet(packet)


def test_h0_packet_rejects_unarmed_or_incomplete_envelope() -> None:
    packet = _packet()
    packet["trace_armed"] = False

    with pytest.raises(ValueError, match="not trace_armed"):
        EXPORT.canonical_semantic_packet(packet)

    packet = _packet()
    del packet["overflow_claim_records"]

    with pytest.raises(ValueError, match="envelope missing required fields"):
        EXPORT.canonical_semantic_packet(packet)


def test_h0_packet_rejects_empty_required_exposure() -> None:
    packet = _packet()
    for stream in (
        "pair_records",
        "candidate_records",
        "claim_records",
        "commit_records",
        "native_candidate_keys",
        "native_pair_keys",
        "native_proposal_keys",
        "native_claim_winner_keys",
        "native_commit_keys",
    ):
        packet[stream] = []
    for field in (
        "total_pair_records",
        "total_candidate_records",
        "total_claim_records",
        "total_commit_records",
        "total_native_candidate_keys",
        "total_native_pair_keys",
        "total_native_proposal_keys",
        "total_native_claim_winner_keys",
        "total_native_commit_keys",
    ):
        packet[field] = 0
    packet["bridge_attempt_count"] = 0
    packet["bridge_commit_count"] = 0

    with pytest.raises(ValueError, match="required candidate exposure"):
        EXPORT.canonical_semantic_packet(packet)


def test_h0_packet_requires_phase_b_commit_exposure() -> None:
    packet = _packet()
    packet["capture_phase"] = "phase_b"
    packet["require_commit_exposure"] = False

    with pytest.raises(ValueError, match="Phase B capture must require"):
        EXPORT.canonical_semantic_packet(packet)

    packet = _packet()
    for stream in (
        "claim_records",
        "commit_records",
        "native_proposal_keys",
        "native_claim_winner_keys",
        "native_commit_keys",
    ):
        packet[stream] = []
    for field in (
        "total_claim_records",
        "total_commit_records",
        "total_native_proposal_keys",
        "total_native_claim_winner_keys",
        "total_native_commit_keys",
    ):
        packet[field] = 0
    packet["bridge_commit_count"] = 0

    with pytest.raises(ValueError, match="required commit exposure"):
        EXPORT.canonical_semantic_packet(packet)


def test_h0_packet_rejects_missing_record_even_when_packet_remains_internal() -> None:
    packet = _packet()
    packet["pair_records"] = []
    packet["candidate_records"] = []
    packet["total_pair_records"] = 0
    packet["total_candidate_records"] = 0

    with pytest.raises(ValueError, match="native_candidate_keys does not equal"):
        EXPORT.canonical_semantic_packet(packet)


def test_h0_native_source_keeps_observer_state_separate_from_policy_state() -> None:
    source = (ROOT / "src/tracking/tracker_gpu.cu").read_text(encoding="utf-8")

    assert "struct H0TraceDeviceBuffers" in source
    assert "h0_append_record" in source
    assert "native_candidate_keys" in source
    assert "native_claim_winner_keys" in source
    assert "track_ids[cand] = track_ids[lost];" in source
    assert "H0 bridge trace observes the real commit path" in source
    assert "research_h0_bridge_trace_ ? d_h0_slot_generation_ : nullptr" in source
    assert "relink_bidir_propose_kernel<false>" in source
    assert "relink_bidir_propose_kernel<true>" in source
    assert "frame_input = d_h0_trace_frame_input_" in source


def test_h0_static_coverage_is_a_replayable_contract_artifact() -> None:
    report, failures = CHECK.coverage_report()

    assert failures == []
    assert report["coverage_schema_version"] == "h0_coverage_v2"
    assert report["all_components_true"] is True
    assert report["coverage_components"] == {
        "track_instance_uid_v1": True,
        "pair_record": True,
        "candidate_record": True,
        "claim_record": True,
        "commit_record": True,
        "native_universe_v2": True,
        "capture_envelope_v2": True,
    }


def test_h0_comment_mask_preserves_offsets_but_not_comment_evidence() -> None:
    source = (
        'const char* literal = "// not a comment";\n'
        "/* h0_append_record(h0.claim_records, cap, cursor, overflow, record);\n"
        "   key.cand_slot = cand; */\n"
        "// h0_append_record(h0.claim_records, cap, cursor, overflow, record);\n"
        'const char* raw = R"tag(// not a comment)tag";\n'
    )

    masked = CHECK.strip_cpp_comments(source)

    assert len(masked) == len(source)
    assert masked.count("\n") == source.count("\n")
    assert "h0_append_record" not in masked
    assert "key.cand_slot = cand" not in masked
    assert '"// not a comment"' in masked
    assert 'R"tag(// not a comment)tag"' in masked


def test_h0_static_checker_rejects_writer_wiring_mutations() -> None:
    cuda_path = ROOT / "src/tracking/tracker_gpu.cu"
    source = cuda_path.read_text(encoding="utf-8")

    missing_claim_append = source.replace(
        "const int claim_record_index = h0_append_record(\n"
        "            h0.claim_records, h0.claim_capacity, h0.claim_cursor, h0.claim_overflow, h0_claim);",
        "const int claim_record_index = -1;",
        1,
    )
    report, failures = CHECK.coverage_report({cuda_path: missing_claim_append})
    assert report["coverage_components"]["claim_record"] is False
    assert any(
        "claim_records has no h0_append_record" in failure for failure in failures
    )

    commented_claim_append = source.replace(
        "const int claim_record_index = h0_append_record(\n"
        "            h0.claim_records, h0.claim_capacity, h0.claim_cursor, h0.claim_overflow, h0_claim);",
        "const int claim_record_index = -1;\n"
        "        /* h0_append_record(\n"
        "            h0.claim_records, h0.claim_capacity, h0.claim_cursor, h0.claim_overflow, h0_claim); */",
        1,
    )
    report, failures = CHECK.coverage_report({cuda_path: commented_claim_append})
    assert report["coverage_components"]["claim_record"] is False
    assert any(
        "claim_records has no h0_append_record" in failure for failure in failures
    )

    record_cursor_for_native_cursor = source.replace(
        "h0.native_candidate_cursor, h0.native_candidate_overflow, key",
        "h0.candidate_cursor, h0.native_candidate_overflow, key",
        1,
    )
    report, failures = CHECK.coverage_report(
        {cuda_path: record_cursor_for_native_cursor}
    )
    assert report["coverage_components"]["native_universe_v2"] is False
    assert any("native_candidate_keys append wiring" in failure for failure in failures)

    before = "        key.cand_slot = cand;\n        key.cand_instance_uid"
    after = "        key.cand_instance_uid"
    moved_field_after_append = source.replace(before, after, 1).replace(
        "                         h0.native_candidate_cursor, h0.native_candidate_overflow, key);",
        "                         h0.native_candidate_cursor, h0.native_candidate_overflow, key);\n"
        "        key.cand_slot = cand;",
        1,
    )
    report, failures = CHECK.coverage_report({cuda_path: moved_field_after_append})
    assert report["coverage_components"]["native_universe_v2"] is False
    assert any(
        "native_candidate_keys fields must be assigned before" in failure
        for failure in failures
    )

    commented_field_before_append = source.replace(
        before,
        "        // key.cand_slot = cand;\n        key.cand_instance_uid",
        1,
    ).replace(
        "                         h0.native_candidate_cursor, h0.native_candidate_overflow, key);",
        "                         h0.native_candidate_cursor, h0.native_candidate_overflow, key);\n"
        "        key.cand_slot = cand;",
        1,
    )
    report, failures = CHECK.coverage_report({cuda_path: commented_field_before_append})
    assert report["coverage_components"]["native_universe_v2"] is False
    assert any(
        "native_candidate_keys fields must be assigned before" in failure
        for failure in failures
    )
