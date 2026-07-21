"""Unit tests for the GPU tracker bidirectional bridge (perception.tracking.tracker_gpu)."""

# scope: tracking
# function: behavior
# lifecycle: active

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "build"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts/tools"))

pytest.importorskip("saccade_tracking_ext", exc_type=ImportError)
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

from saccade.perception.tracking.tracker_gpu import (  # noqa: E402
    GPUByteTracker,
    GraphedTrackerUpdate,
)
from export_headline_bridge_decision_trace import semantic_digest  # noqa: E402
from verify_headline_bridge_decision_trace import verify_capture  # noqa: E402


def _tracker(*, bidirectional: bool, bridge_px: float = 0.25) -> GPUByteTracker:
    tracker = GPUByteTracker(2048, 768)
    tracker.set_frame_size(640, 480)
    tracker.set_quality_params(False, 0.5, 0.3, 0.2)
    tracker.set_params(
        0.05,
        0.45,
        0.5,
        30,
        0.10,
        3,
        0.50,
        False,
        0.28,
        False,
        2.8,
        0.0,
        0.0,
        0.5,
        0.0,
        0.0,
    )
    tracker.set_relink_params(
        True,
        256,
        0.6,
        2.5,
        4.0,
        300,
        bidirectional,
        bridge_px,
        4,
        2,
        120,
        0.0,
        1.65,
        30.0,
        0.0,
        0.0,
    )
    return tracker


def _box(
    cx: float, cy: float = 200.0, w: float = 50.0, h: float = 100.0
) -> list[float]:
    return [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5]


def _run_sequence(
    *,
    bidirectional: bool,
    competing_candidates: bool = False,
    fidelity_audit: bool = False,
    h0_trace: bool = False,
    shadow: bool = False,
) -> tuple[dict[int, list[int]], list[int], dict[str, object] | None]:
    tracker = _tracker(
        bidirectional=bidirectional,
        bridge_px=0.7 if competing_candidates else 0.25,
    )
    if fidelity_audit:
        tracker.set_research_bridge_fidelity_audit(True, capacity=64)
    if h0_trace:
        tracker.set_research_h0_bridge_trace(
            True,
            pair_capacity=64,
            candidate_capacity=32,
            claim_capacity=32,
            commit_capacity=32,
        )
    if shadow:
        tracker.set_research_bridge_shadow(True)
    buffers = tracker.allocate_result_buffers(device="cuda")
    gmc = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device="cuda")
    detections = (
        {
            1: [(100, 200, 0.90)],
            2: [(110, 200, 0.90)],
            3: [(120, 200, 0.90)],
            4: [(130, 200, 0.90)],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [(210, 200, 0.70), (210, 260, 0.95)],
            10: [(220, 200, 0.70), (220, 260, 0.95)],
            11: [(230, 200, 0.70), (230, 260, 0.95)],
            12: [(240, 200, 0.70), (240, 260, 0.95)],
        }
        if competing_candidates
        else {
            1: [(100, 200, 0.90)],
            2: [(110, 200, 0.90)],
            3: [(120, 200, 0.90)],
            4: [(130, 200, 0.90)],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [(210, 200, 0.90)],
            10: [(220, 200, 0.90)],
            11: [(230, 200, 0.90)],
            12: [(240, 200, 0.90)],
        }
    )
    outputs: dict[int, list[int]] = {}
    for frame_id, dets in detections.items():
        if dets:
            boxes = torch.tensor(
                [_box(cx, cy) for cx, cy, _score in dets],
                dtype=torch.float32,
                device="cuda",
            )
            scores = torch.tensor(
                [_score for _cx, _cy, _score in dets],
                dtype=torch.float32,
                device="cuda",
            )
            classes = torch.zeros((len(dets),), dtype=torch.int32, device="cuda")
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32, device="cuda")
            scores = torch.zeros((0,), dtype=torch.float32, device="cuda")
            classes = torch.zeros((0,), dtype=torch.int32, device="cuda")

        tracker.update_into(
            boxes,
            scores,
            classes,
            buffers,
            gmc=gmc,
            h0_frame_id=frame_id if h0_trace else None,
        )
        torch.cuda.synchronize()
        count = int(buffers["count"].item())
        outputs[frame_id] = buffers["ids"][:count].detach().cpu().tolist()
    capture = None
    if fidelity_audit:
        capture = tracker.drain_research_bridge_fidelity_events(seq="bridge_fixture")
    if h0_trace:
        capture = tracker.drain_research_h0_bridge_trace(
            seq="bridge_fixture",
            capture_phase="phase_a",
            require_candidate_exposure=True,
            require_commit_exposure=True,
        )
    return outputs, tracker.get_relink_debug(), capture


def test_bidirectional_bridge_revives_live_lost_identity() -> None:
    without_bridge, without_debug, _ = _run_sequence(bidirectional=False)
    with_bridge, with_debug, _ = _run_sequence(bidirectional=True)

    assert without_bridge[11] == [2]
    assert without_bridge[12] == [2]
    assert with_bridge[11] == [2]
    assert with_bridge[12] == [1]
    # get_relink_debug layout: [0]=cursor, [1]=births, [2]=revived,
    # [3]=bridge_attempts, [4]=bridge_accepts, [5..12]=appearance-relink
    # diagnostics. Without the bidirectional bridge, no bridge attempt or
    # accept fires (indices 3/4); the appended appearance-relink counters
    # may be nonzero and are not asserted here.
    assert without_debug[3] == 0
    assert without_debug[4] == 0
    assert with_debug[3] >= with_debug[4]
    assert with_debug[4] == 1


def test_shadow_bridge_proposes_and_captures_without_changing_identity() -> None:
    """Issue #112: shadow mode must be output-equivalent to a bridge-off run.

    The fidelity capture only runs inside the propose kernel, which is gated on
    the bridge being enabled -- but a committing bridge rewrites track identity,
    so captured events could not be joined against a bridge-off pair cohort.
    Shadow mode skips only the commit kernel, so the bridge still attempts (and
    captures) while output identity stays bit-identical to bridge-off.
    """
    without_bridge, without_debug, _ = _run_sequence(bidirectional=False)
    shadow, shadow_debug, capture = _run_sequence(
        bidirectional=True, fidelity_audit=True, shadow=True
    )

    assert shadow == without_bridge
    assert without_debug[3] == 0
    assert shadow_debug[3] > 0  # bridge attempted
    assert shadow_debug[4] == 0  # but never committed

    assert capture is not None
    events = capture["events"]
    assert isinstance(events, list) and events
    assert capture["overflow_events"] == 0


def test_bidirectional_bridge_prefers_higher_score_candidate() -> None:
    outputs, debug, _ = _run_sequence(
        bidirectional=True,
        competing_candidates=True,
    )

    assert outputs[11] == [2, 3]
    assert outputs[12] == [2, 1]
    assert debug[3] >= debug[4]
    assert debug[4] == 1


def test_cost_mode_settings_do_not_leak_between_trackers() -> None:
    other = _tracker(bidirectional=False)
    other.set_multiplicative_cost(True)
    other.set_stability_cost_w(10.0)
    other.set_sinkhorn_lambda(7.0)

    outputs, _debug, _ = _run_sequence(bidirectional=False)

    assert outputs[11] == [2]
    assert outputs[12] == [2]


def test_bridge_fidelity_audit_is_observational_and_exports_native_values() -> None:
    baseline, _debug, _ = _run_sequence(bidirectional=True)
    captured, _debug, capture = _run_sequence(bidirectional=True, fidelity_audit=True)

    assert captured == baseline
    assert capture is not None
    assert capture["complete"] is True
    assert capture["overflow_events"] == 0
    events = capture["events"]
    assert isinstance(events, list)
    assert events
    event = events[0]
    assert event["capture_mode"] == "runtime_cuda_event_ring"
    assert event["production_threshold"] == pytest.approx(0.25)
    assert event["h_ref"] >= 1.0
    assert event["la"] == event["gap"] + event["bridge_at"] - 1


def test_h0_trace_observes_real_commit_without_changing_bridge_output() -> None:
    baseline, baseline_debug, _ = _run_sequence(bidirectional=True)
    captured, captured_debug, trace = _run_sequence(bidirectional=True, h0_trace=True)

    assert captured == baseline
    assert captured_debug == baseline_debug
    assert trace is not None
    assert trace["complete"] is True
    assert trace["trace_armed"] is True
    assert int(trace["processed_frame_count"]) > 0
    assert int(trace["bridge_attempt_count"]) == len(trace["native_candidate_keys"])
    assert int(trace["bridge_commit_count"]) == len(trace["native_commit_keys"])
    assert trace["stream_overflow"] == {
        "pair_records": 0,
        "candidate_records": 0,
        "claim_records": 0,
        "commit_records": 0,
    }
    assert trace["claim_records"]
    assert trace["commit_records"]
    commit = trace["commit_records"][0]
    assert commit["cand_postcommit_track_id"] == commit["lost_precommit_track_id"]
    assert (commit["lost_active_before"], commit["lost_active_after"]) == (1, 0)
    assert verify_capture(trace)["replay"] == "full_commit_decision_trace_v2"

    repeated, repeated_debug, repeated_trace = _run_sequence(
        bidirectional=True, h0_trace=True
    )
    assert repeated == baseline
    assert repeated_debug == baseline_debug
    assert repeated_trace is not None
    assert semantic_digest(repeated_trace) == semantic_digest(trace)


def test_h0_unarmed_native_drain_is_rejected_before_envelope_creation() -> None:
    tracker = _tracker(bidirectional=True)

    with pytest.raises(RuntimeError, match="not armed"):
        tracker.drain_research_h0_bridge_trace(
            seq="bridge_fixture",
            capture_phase="phase_a",
            require_candidate_exposure=True,
            require_commit_exposure=False,
        )


def _run_h0_graph_sequence(
    *, graphed: bool, h0_trace: bool = True
) -> tuple[dict[int, list[int]], list[int], dict[str, object] | None]:
    """Two real bridge commits with evaluator-owned MOT frame IDs."""
    tracker = _tracker(bidirectional=True)
    if h0_trace:
        tracker.set_research_h0_bridge_trace(
            True,
            pair_capacity=128,
            candidate_capacity=64,
            claim_capacity=64,
            commit_capacity=64,
        )
    buffers = tracker.allocate_result_buffers(device="cuda")
    graph = GraphedTrackerUpdate(tracker, max_assoc=4) if graphed else None
    gmc = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device="cuda")
    detections = {
        1: [(100, 200, 0.90)],
        2: [(110, 200, 0.90)],
        3: [(120, 200, 0.90)],
        4: [(130, 200, 0.90)],
        5: [],
        6: [],
        7: [],
        8: [],
        9: [(210, 200, 0.90)],
        10: [(220, 200, 0.90)],
        11: [(230, 200, 0.90)],
        12: [(240, 200, 0.90)],
        13: [],
        14: [],
        15: [],
        16: [],
        17: [(320, 200, 0.90)],
        18: [(330, 200, 0.90)],
        19: [(340, 200, 0.90)],
        20: [(350, 200, 0.90)],
    }
    outputs: dict[int, list[int]] = {}
    for frame_id, dets in detections.items():
        boxes = (
            torch.tensor(
                [_box(cx, cy) for cx, cy, _score in dets],
                dtype=torch.float32,
                device="cuda",
            )
            if dets
            else torch.zeros((0, 4), dtype=torch.float32, device="cuda")
        )
        scores = (
            torch.tensor(
                [_score for _cx, _cy, _score in dets],
                dtype=torch.float32,
                device="cuda",
            )
            if dets
            else torch.zeros((0,), dtype=torch.float32, device="cuda")
        )
        classes = torch.zeros((len(dets),), dtype=torch.int32, device="cuda")
        if graph is None:
            tracker.update_into(
                boxes,
                scores,
                classes,
                buffers,
                gmc=gmc,
                h0_frame_id=frame_id if h0_trace else None,
            )
            result = buffers
        else:
            graph.copy_inputs(
                boxes,
                scores,
                classes,
                gmc,
                frame_id=frame_id if h0_trace else None,
            )
            result = graph.replay()
        torch.cuda.synchronize()
        count = int(result["count"].item())
        outputs[frame_id] = result["ids"][:count].detach().cpu().tolist()
    trace = (
        tracker.drain_research_h0_bridge_trace(
            seq="bridge_graph_fixture",
            capture_phase="phase_a",
            require_candidate_exposure=True,
            require_commit_exposure=True,
        )
        if h0_trace
        else None
    )
    return outputs, tracker.get_relink_debug(), trace


def test_h0_trace_graph_replay_uses_evaluator_frame_input() -> None:
    eager_outputs, _eager_debug, eager_trace = _run_h0_graph_sequence(graphed=False)
    graph_outputs, _graph_debug, graph_trace = _run_h0_graph_sequence(graphed=True)

    assert eager_trace is not None
    assert graph_trace is not None
    assert graph_outputs == eager_outputs
    assert verify_capture(graph_trace)["replay"] == "full_commit_decision_trace_v2"
    assert semantic_digest(graph_trace) == semantic_digest(eager_trace)

    # The two bridge events occur on distinct, evaluator-supplied MOT frames.
    # This would be [capture_frame, capture_frame] with the former host counter.
    frames = sorted({int(row["frame"]) for row in graph_trace["pair_records"]})
    assert frames == [12, 20]
    # Frame 4 has the first candidate decision but no structural lost track;
    # later decision, claim, and commit records retain their own evaluation
    # frames rather than graph-capture's frozen frame.
    assert {int(row["frame"]) for row in graph_trace["candidate_records"]} == {
        4,
        12,
        20,
    }
    for stream in ("claim_records", "commit_records"):
        assert {int(row["frame"]) for row in graph_trace[stream]} == {12, 20}


def test_h0_trace_is_graph_output_and_counter_neutral() -> None:
    baseline_outputs, baseline_debug, _ = _run_h0_graph_sequence(
        graphed=True, h0_trace=False
    )
    captured_outputs, captured_debug, trace = _run_h0_graph_sequence(
        graphed=True, h0_trace=True
    )

    assert captured_outputs == baseline_outputs
    assert captured_debug == baseline_debug
    assert trace is not None
    assert trace["complete"] is True
