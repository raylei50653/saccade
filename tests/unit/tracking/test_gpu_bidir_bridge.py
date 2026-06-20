from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "build"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

pytest.importorskip("saccade_tracking_ext", exc_type=ImportError)
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

from saccade.perception.tracking.tracker_gpu import GPUByteTracker  # noqa: E402


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
    *, bidirectional: bool, competing_candidates: bool = False
) -> tuple[dict[int, list[int]], list[int]]:
    tracker = _tracker(
        bidirectional=bidirectional,
        bridge_px=0.7 if competing_candidates else 0.25,
    )
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

        tracker.update_into(boxes, scores, classes, buffers, gmc=gmc)
        torch.cuda.synchronize()
        count = int(buffers["count"].item())
        outputs[frame_id] = buffers["ids"][:count].detach().cpu().tolist()
    return outputs, tracker.get_relink_debug()


def test_bidirectional_bridge_revives_live_lost_identity() -> None:
    without_bridge, without_debug = _run_sequence(bidirectional=False)
    with_bridge, with_debug = _run_sequence(bidirectional=True)

    assert without_bridge[11] == [2]
    assert without_bridge[12] == [2]
    assert with_bridge[11] == [2]
    assert with_bridge[12] == [1]
    assert without_debug[3:] == [0, 0]
    assert with_debug[3] >= with_debug[4]
    assert with_debug[4] == 1


def test_bidirectional_bridge_prefers_higher_score_candidate() -> None:
    outputs, debug = _run_sequence(
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

    outputs, _debug = _run_sequence(bidirectional=False)

    assert outputs[11] == [2]
    assert outputs[12] == [2]
