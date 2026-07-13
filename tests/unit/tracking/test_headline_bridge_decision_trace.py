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
    return {
        "capture_schema_version": "h0_bridge_decision_trace_v1",
        "capture_run_uuid": run_uuid,
        "pair_records": [pair],
        "candidate_records": [candidate],
        "claim_records": [claim],
        "commit_records": [commit],
        "total_pair_records": 1,
        "total_candidate_records": 1,
        "total_claim_records": 1,
        "total_commit_records": 1,
        "overflow_pair_records": 0,
        "overflow_candidate_records": 0,
        "overflow_claim_records": 0,
        "overflow_commit_records": 0,
        "identity_uid_wrap_events": 0,
    }


def test_h0_semantic_digest_excludes_run_uuid_and_raw_stream_order() -> None:
    first = _packet(run_uuid="raw-run-a")
    second = _packet(run_uuid="raw-run-b")

    assert EXPORT.semantic_digest(first) == EXPORT.semantic_digest(second)


def test_h0_verifier_replays_full_single_claim_commit() -> None:
    result = VERIFY.verify_capture(_packet())

    assert result["replay"] == "full_commit_decision_trace_v1"
    assert result["stream_totals"] == {
        "pair_records": 1,
        "candidate_records": 1,
        "claim_records": 1,
        "commit_records": 1,
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


def test_h0_native_source_keeps_observer_state_separate_from_policy_state() -> None:
    source = (ROOT / "src/tracking/tracker_gpu.cu").read_text(encoding="utf-8")

    assert "struct H0TraceDeviceBuffers" in source
    assert "h0_append_record" in source
    assert "track_ids[cand] = track_ids[lost];" in source
    assert "H0 bridge trace observes the real commit path" in source
    assert "research_h0_bridge_trace_ ? d_h0_slot_generation_ : nullptr" in source
    assert "relink_bidir_propose_kernel<false>" in source
    assert "relink_bidir_propose_kernel<true>" in source
    assert "frame_input = d_h0_trace_frame_input_" in source
