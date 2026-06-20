import os
import pytest
import torch
from contextlib import nullcontext
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import saccade.perception.eval.evaluator as evaluator_mod

from saccade.perception.eval.evaluator import (
    _record_profile_scope,
    _apply_stage2_quality_gate,
    _consecutive_birth_check,
    _suppress_duplicate_detections,
    _apply_detection_cap,
    _compute_fp_filter_ranking,
    _apply_fp_hard_filter,
    _get_softmax3_torch_params,
    _predict_softmax3_probs_torch,
    _apply_external_fp_filter,
    _compute_adaptive_cap,
    _build_active_track_priors,
    _env_flag_enabled,
    _build_cpp_seq_config,
)
from saccade.perception.eval.external_fp_model import (
    RuleBaselineConfig,
    LogisticModel,
    SoftmaxLinearModel,
)


def test_nms_graph_retains_captured_count_buffer(monkeypatch):
    """The native graph keeps this raw pointer after ``_run_nms`` returns."""

    class _FakeGraph:
        def __init__(self):
            self.replay_calls = 0

        def replay(self):
            self.replay_calls += 1

    class _FakePipeline:
        def __init__(self):
            self.out_count_ptrs: list[int] = []

        def process_detections_graph(self, *args):
            # out_count is argument 11 in the native binding contract.
            self.out_count_ptrs.append(args[11])

    class _FakeStream:
        cuda_stream = 7

    graph = _FakeGraph()
    pipe = _FakePipeline()
    original_zeros = torch.zeros

    def _cpu_zeros(*args, **kwargs):
        kwargs.pop("device", None)
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(evaluator_mod, "copy_pad_detections", lambda *args: None)
    monkeypatch.setattr(evaluator_mod.torch, "zeros", _cpu_zeros)
    monkeypatch.setattr(evaluator_mod.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        evaluator_mod.torch.cuda, "current_stream", lambda: _FakeStream()
    )
    monkeypatch.setattr(
        evaluator_mod.torch.cuda,
        "CUDAGraph",
        lambda: graph,
    )
    monkeypatch.setattr(
        evaluator_mod.torch.cuda,
        "graph",
        lambda _graph: nullcontext(),
    )

    state = SimpleNamespace(
        perception_pipeline=pipe,
        nms_in={
            "boxes": torch.empty((4, 4)),
            "scores": torch.empty(4),
            "classes": torch.empty(4, dtype=torch.int32),
        },
        post_bufs={
            "boxes": torch.empty((4, 4)),
            "scores": torch.empty(4),
            "classes": torch.empty(4, dtype=torch.int32),
            "suspect": torch.empty(4, dtype=torch.bool),
        },
        nms_fixed_n=4,
        w_orig=1920,
        h_orig=1080,
        nms_graph_out_count=None,
    )
    boxes = torch.empty((2, 4))
    scores = torch.empty(2)
    classes = torch.empty(2, dtype=torch.int32)

    _, captured_graph = evaluator_mod._run_nms(
        state,
        raw_boxes_contig=boxes,
        raw_scores_contig=scores,
        raw_classes_contig=classes,
        raw_box_count=2,
        num_priors=0,
        is_tiled=False,
        nms_graph=None,
    )

    assert captured_graph is graph
    assert state.nms_graph_out_count is not None
    assert pipe.out_count_ptrs == [state.nms_graph_out_count.data_ptr()] * 2

    evaluator_mod._run_nms(
        state,
        raw_boxes_contig=boxes,
        raw_scores_contig=scores,
        raw_classes_contig=classes,
        raw_box_count=2,
        num_priors=0,
        is_tiled=False,
        nms_graph=captured_graph,
    )
    assert graph.replay_calls == 1


# Test _record_profile_scope
def test_record_profile_scope():
    with _record_profile_scope("test_scope"):
        pass


# Test _apply_stage2_quality_gate
def test_apply_stage2_quality_gate_empty():
    fused_boxes = torch.zeros((0, 4), dtype=torch.float32)
    fused_scores = torch.zeros((0,), dtype=torch.float32)
    fused_classes = torch.zeros((0,), dtype=torch.int32)
    geometry_suspect_mask = torch.zeros((0,), dtype=torch.bool)

    b, s, c, m, k = _apply_stage2_quality_gate(
        fused_boxes,
        fused_scores,
        fused_classes,
        geometry_suspect_mask,
        None,
        track_thresh=0.1,
        mid_thresh=0.4,
        quality_min=0.5,
        quality=torch.zeros((0,)),
    )
    assert b.shape[0] == 0


def test_apply_stage2_quality_gate_no_mid_score():
    fused_boxes = torch.tensor(
        [[10, 10, 20, 20], [30, 30, 40, 40]], dtype=torch.float32
    )
    fused_scores = torch.tensor([0.5, 0.05], dtype=torch.float32)
    fused_classes = torch.tensor([0, 0], dtype=torch.int32)
    geometry_suspect_mask = torch.tensor([False, False], dtype=torch.bool)
    quality = torch.tensor([0.9, 0.1], dtype=torch.float32)

    b, s, c, m, k = _apply_stage2_quality_gate(
        fused_boxes,
        fused_scores,
        fused_classes,
        geometry_suspect_mask,
        None,
        track_thresh=0.1,
        mid_thresh=0.4,
        quality_min=0.5,
        quality=quality,
    )
    assert b.shape[0] == 2


def test_apply_stage2_quality_gate_filtering():
    fused_boxes = torch.tensor(
        [[10, 10, 20, 20], [30, 30, 40, 40], [50, 50, 60, 60]], dtype=torch.float32
    )
    fused_scores = torch.tensor([0.2, 0.25, 0.5], dtype=torch.float32)
    fused_classes = torch.tensor([0, 0, 0], dtype=torch.int32)
    geometry_suspect_mask = torch.tensor([False, False, False], dtype=torch.bool)
    quality = torch.tensor([0.9, 0.1, 0.1], dtype=torch.float32)
    aligned_kps = torch.zeros((3, 17, 3), dtype=torch.float32)

    # 0.2 is mid-score, quality is 0.9 (>= 0.5) -> keep
    # 0.25 is mid-score, quality is 0.1 (< 0.5) -> remove
    # 0.5 is high-score (>= mid_thresh) -> keep regardless of quality
    b, s, c, m, k = _apply_stage2_quality_gate(
        fused_boxes,
        fused_scores,
        fused_classes,
        geometry_suspect_mask,
        aligned_kps,
        track_thresh=0.1,
        mid_thresh=0.4,
        quality_min=0.5,
        quality=quality,
    )
    assert b.shape[0] == 2
    assert torch.equal(s, torch.tensor([0.2, 0.5]))
    assert k is not None
    assert k.shape[0] == 2


# Test _consecutive_birth_check
def test_consecutive_birth_check_empty_window():
    sub_boxes = torch.tensor([[10, 10, 20, 20]], dtype=torch.float32)
    res = _consecutive_birth_check(sub_boxes, [], iou_thresh=0.5)
    assert res.shape == (1,)
    assert res[0].item() is True


def test_consecutive_birth_check_with_matches():
    sub_boxes = torch.tensor(
        [[10, 10, 20, 20], [100, 100, 110, 110]], dtype=torch.float32
    )
    window = [
        torch.tensor([[12, 12, 22, 22], [50, 50, 60, 60]], dtype=torch.float32),
        torch.tensor([[9, 9, 19, 19], [200, 200, 210, 210]], dtype=torch.float32),
    ]
    # Box 1 matches (10, 10, 20, 20) is close to (12, 12, 22, 22) [IoU ~0.47] and (9, 9, 19, 19) [IoU ~0.65]
    # If we set iou_thresh=0.4, it matches both window frames.
    # Box 2 (100, 100) doesn't match both window frames.
    res = _consecutive_birth_check(sub_boxes, window, iou_thresh=0.4)
    assert res[0].item() is True
    assert res[1].item() is False


def test_consecutive_birth_check_motion_gating():
    sub_boxes = torch.tensor([[10, 10, 100, 100]], dtype=torch.float32)
    # oldest frame box is identical -> motion = 0
    window1 = [torch.tensor([[10, 10, 100, 100]], dtype=torch.float32)]
    res1 = _consecutive_birth_check(
        sub_boxes, window1, iou_thresh=0.5, min_motion_px=5.0
    )
    assert res1[0].item() is False

    # oldest frame box moved -> motion = sqrt(10^2 + 10^2) ~ 14.1 > 5.0, IoU ~ 0.65
    window2 = [torch.tensor([[0, 0, 90, 90]], dtype=torch.float32)]
    res2 = _consecutive_birth_check(
        sub_boxes, window2, iou_thresh=0.5, min_motion_px=5.0
    )
    assert res2[0].item() is True


# Test _suppress_duplicate_detections
def test_suppress_duplicate_detections_few_boxes():
    fused_boxes = torch.tensor([[10, 10, 20, 20]], dtype=torch.float32)
    fused_scores = torch.tensor([0.9], dtype=torch.float32)
    keep = _suppress_duplicate_detections(fused_boxes, fused_scores)
    assert keep[0].item() is True


def test_suppress_duplicate_detections_overlapping():
    fused_boxes = torch.tensor(
        [
            [10, 10, 20, 20],
            [11, 11, 21, 21],  # duplicate, lower score
            [50, 50, 60, 60],  # distinct
        ],
        dtype=torch.float32,
    )
    fused_scores = torch.tensor([0.9, 0.7, 0.8], dtype=torch.float32)
    # IoU is ~0.68. If we set iou_threshold=0.6:
    keep = _suppress_duplicate_detections(
        fused_boxes, fused_scores, iou_threshold=0.6, min_score_ratio=1.1
    )
    assert keep[0].item() is True
    assert keep[1].item() is False
    assert keep[2].item() is True


def test_suppress_duplicate_detections_overlapping_low_ratio():
    fused_boxes = torch.tensor(
        [[10, 10, 20, 20], [11, 11, 21, 21]], dtype=torch.float32
    )
    fused_scores = torch.tensor([0.9, 0.88], dtype=torch.float32)
    # ratio is 0.9/0.88 ~ 1.02. If min_score_ratio=1.1, it shouldn't suppress
    keep = _suppress_duplicate_detections(
        fused_boxes, fused_scores, iou_threshold=0.6, min_score_ratio=1.1
    )
    assert keep[0].item() is True
    assert keep[1].item() is True


# Test _apply_detection_cap
def test_apply_detection_cap_no_op():
    boxes = torch.tensor([[10, 10, 20, 20]], dtype=torch.float32)
    scores = torch.tensor([0.8], dtype=torch.float32)
    classes = torch.tensor([0], dtype=torch.int32)

    # n <= max_detections
    b, s, c = _apply_detection_cap(boxes, scores, classes, max_detections=5)
    assert b.shape[0] == 1

    # max_detections <= 0
    b, s, c = _apply_detection_cap(boxes, scores, classes, max_detections=0)
    assert b.shape[0] == 1


def test_apply_detection_cap_score():
    boxes = torch.tensor(
        [[10, 10, 20, 20], [30, 30, 40, 40], [50, 50, 60, 60]], dtype=torch.float32
    )
    scores = torch.tensor([0.5, 0.9, 0.7], dtype=torch.float32)
    classes = torch.tensor([0, 1, 2], dtype=torch.int32)

    b, s, c = _apply_detection_cap(
        boxes, scores, classes, max_detections=2, rank_method="score"
    )
    assert b.shape[0] == 2
    # Should keep highest 2 scores: 0.9 and 0.7.
    assert 0.9 in s
    assert 0.7 in s
    assert 0.5 not in s


def test_apply_detection_cap_quality():
    boxes = torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]], dtype=torch.float32)
    scores = torch.tensor([0.8, 0.9], dtype=torch.float32)
    classes = torch.tensor([0, 0], dtype=torch.int32)
    quality_factors = torch.tensor([1.0, 0.1], dtype=torch.float32)

    # rank by score * quality_factor -> [0.8*1.0 = 0.8, 0.9*0.1 = 0.09]
    b, s, c = _apply_detection_cap(
        boxes,
        scores,
        classes,
        max_detections=1,
        quality_factors=quality_factors,
        rank_method="quality",
    )
    assert b.shape[0] == 1
    assert s[0].item() == pytest.approx(0.8)


def test_apply_detection_cap_fp_filter():
    boxes = torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]], dtype=torch.float32)
    scores = torch.tensor([0.8, 0.9], dtype=torch.float32)
    classes = torch.tensor([0, 0], dtype=torch.int32)

    b, s, c = _apply_detection_cap(
        boxes, scores, classes, max_detections=1, rank_method="fp_filter"
    )
    assert b.shape[0] == 1


def test_apply_detection_cap_fp_filter_quality():
    boxes = torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]], dtype=torch.float32)
    scores = torch.tensor([0.8, 0.9], dtype=torch.float32)
    classes = torch.tensor([0, 0], dtype=torch.int32)
    quality_factors = torch.tensor([1.0, 0.1], dtype=torch.float32)

    b, s, c = _apply_detection_cap(
        boxes,
        scores,
        classes,
        max_detections=1,
        quality_factors=quality_factors,
        rank_method="fp_filter_quality",
    )
    assert b.shape[0] == 1


# Test _compute_fp_filter_ranking
def test_compute_fp_filter_ranking_empty():
    boxes = torch.zeros((0, 4), dtype=torch.float32)
    scores = torch.zeros((0,), dtype=torch.float32)
    ranking = _compute_fp_filter_ranking(boxes, scores)
    assert ranking.shape[0] == 0


def test_compute_fp_filter_ranking_logic():
    # Large area box (>4000) with moderate score (0.4-0.6) should be penalized
    boxes = torch.tensor(
        [
            [0, 0, 100, 100],  # area = 10000, aspect = 1.0
            [0, 0, 10, 10],  # area = 100, aspect = 1.0
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.5, 0.5], dtype=torch.float32)
    ranking = _compute_fp_filter_ranking(boxes, scores)
    assert ranking[0] < ranking[1]  # Large area box penalized more


# Test _apply_fp_hard_filter
def test_apply_fp_hard_filter():
    boxes = torch.zeros((0, 4), dtype=torch.float32)
    scores = torch.zeros((0,), dtype=torch.float32)
    classes = torch.zeros((0,), dtype=torch.int32)
    b, s, c = _apply_fp_hard_filter(boxes, scores, classes)
    assert b.shape[0] == 0

    # Detections:
    # 1. Very low score (< 0.25) -> remove
    # 2. Score < 0.45 and area > 10000 -> remove
    # 3. High score -> keep
    boxes = torch.tensor(
        [
            [0, 0, 10, 10],  # area = 100
            [0, 0, 200, 200],  # area = 40000
            [0, 0, 10, 10],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.1, 0.35, 0.8], dtype=torch.float32)
    classes = torch.tensor([0, 0, 0], dtype=torch.int32)

    b, s, c = _apply_fp_hard_filter(
        boxes,
        scores,
        classes,
        min_score=0.25,
        max_suspicious_area=10000,
        max_suspicious_score=0.45,
    )
    assert b.shape[0] == 1
    assert s[0].item() == pytest.approx(0.8)


# Test _get_softmax3_torch_params
def test_get_softmax3_torch_params():
    model = SoftmaxLinearModel(
        feature_names=("score", "width"),
        class_names=("tp", "fp", "np"),
        weights=((0.1, 0.2, 0.7), (0.3, 0.4, 0.3)),
        bias=(0.1, 0.2, 0.3),
        mean=(0.5, 50.0),
        std=(0.2, 10.0),
    )
    device = torch.device("cpu")
    w, b, m, s = _get_softmax3_torch_params(model, device, torch.float32)
    assert w.shape == (2, 3)
    assert b.shape == (3,)
    assert m.shape == (2,)
    assert s.shape == (2,)


# Test _predict_softmax3_probs_torch
def test_predict_softmax3_probs_torch():
    model = SoftmaxLinearModel(
        feature_names=("score", "width", "edge_margin_norm"),
        class_names=("tp", "fp", "np"),
        weights=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        bias=(0.0, 0.0, 0.0),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )
    subset_scores = torch.tensor([0.5, 0.8], dtype=torch.float32)
    subset_widths = torch.tensor([10.0, 20.0], dtype=torch.float32)
    subset_heights = torch.tensor([30.0, 40.0], dtype=torch.float32)
    center_x = torch.tensor([5.0, 10.0], dtype=torch.float32)
    center_y = torch.tensor([15.0, 20.0], dtype=torch.float32)
    edge_margin = torch.tensor([2.0, 4.0], dtype=torch.float32)
    touches_edge = torch.tensor([0.0, 0.0], dtype=torch.float32)

    probs = _predict_softmax3_probs_torch(
        model,
        subset_scores=subset_scores,
        subset_widths=subset_widths,
        subset_heights=subset_heights,
        center_x=center_x,
        center_y=center_y,
        edge_margin=edge_margin,
        touches_edge=touches_edge,
        image_width=100,
        image_height=100,
    )
    assert probs.shape == (2, 3)
    assert torch.allclose(probs.sum(dim=1), torch.tensor(1.0))


# Test _apply_external_fp_filter
def test_apply_external_fp_filter_off_or_empty():
    boxes = torch.zeros((0, 4), dtype=torch.float32)
    scores = torch.zeros((0,), dtype=torch.float32)
    classes = torch.zeros((0,), dtype=torch.int32)

    # empty boxes
    b, s, c = _apply_external_fp_filter(
        boxes,
        scores,
        classes,
        image_width=100,
        image_height=100,
        mode="rule",
        rule_config=RuleBaselineConfig(),
        logistic_model=None,
        logistic_threshold=0.5,
        max_score=0.9,
        penalty=1.0,
        min_score=0.0,
        softmax_min_scale=0.1,
    )
    assert b.shape[0] == 0

    # mode="off"
    boxes = torch.tensor([[10, 10, 20, 20]], dtype=torch.float32)
    scores = torch.tensor([0.8], dtype=torch.float32)
    classes = torch.tensor([0], dtype=torch.int32)
    b, s, c = _apply_external_fp_filter(
        boxes,
        scores,
        classes,
        image_width=100,
        image_height=100,
        mode="off",
        rule_config=RuleBaselineConfig(),
        logistic_model=None,
        logistic_threshold=0.5,
        max_score=0.9,
        penalty=1.0,
        min_score=0.0,
        softmax_min_scale=0.1,
    )
    assert b.shape[0] == 1


def test_apply_external_fp_filter_rule_mode():
    boxes = torch.tensor(
        [
            [10, 10, 20, 20],  # w=10, h=10 -> height < min_height (72)
            [10, 10, 20, 90],  # w=10, h=80 -> height >= min_height
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.08, 0.08], dtype=torch.float32)
    classes = torch.tensor([0, 0], dtype=torch.int32)

    rule_cfg = RuleBaselineConfig(min_score=0.05, low_score=0.10, min_height=72.0)
    b, s, c = _apply_external_fp_filter(
        boxes,
        scores,
        classes,
        image_width=100,
        image_height=100,
        mode="rule",
        rule_config=rule_cfg,
        logistic_model=None,
        logistic_threshold=0.5,
        max_score=0.9,
        penalty=0.5,
        min_score=0.05,
        softmax_min_scale=0.1,
    )
    # Box 1: low score (0.08 < 0.10) & height (10 < 72) -> filtered (since penalty=0.5 brings score to 0.04 < min_score)
    # Box 2: kept
    assert b.shape[0] == 1
    assert torch.allclose(s, torch.tensor([0.08]))


def test_apply_external_fp_filter_logistic_mode():
    boxes = torch.tensor([[10, 10, 20, 30]], dtype=torch.float32)
    scores = torch.tensor([0.4], dtype=torch.float32)
    classes = torch.tensor([0], dtype=torch.int32)

    # Create simple LogisticModel
    feature_names = (
        "score",
        "width",
        "height",
        "area",
        "aspect_ratio",
        "center_x_norm",
        "center_y_norm",
        "edge_margin_norm",
        "touches_edge",
    )
    model = LogisticModel(
        feature_names=feature_names,
        weights=tuple([0.1] * 9),
        bias=0.0,
        mean=tuple([0.0] * 9),
        std=tuple([1.0] * 9),
    )

    b, s, c = _apply_external_fp_filter(
        boxes,
        scores,
        classes,
        image_width=100,
        image_height=100,
        mode="logistic",
        rule_config=RuleBaselineConfig(),
        logistic_model=model,
        logistic_threshold=0.1,
        max_score=0.9,
        penalty=1.0,
        min_score=0.0,
        softmax_min_scale=0.1,
    )
    assert b.shape[0] == 1


def test_apply_external_fp_filter_softmax3_mode():
    boxes = torch.tensor([[10, 10, 20, 30]], dtype=torch.float32)
    scores = torch.tensor([0.4], dtype=torch.float32)
    classes = torch.tensor([0], dtype=torch.int32)

    model = SoftmaxLinearModel(
        feature_names=("score", "width"),
        class_names=("tp", "fp", "np"),
        weights=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        bias=(0.0, 0.0, 0.0),
        mean=(0.0, 0.0),
        std=(1.0, 1.0),
    )

    b, s, c = _apply_external_fp_filter(
        boxes,
        scores,
        classes,
        image_width=100,
        image_height=100,
        mode="softmax3",
        rule_config=RuleBaselineConfig(),
        logistic_model=model,
        logistic_threshold=0.5,
        max_score=0.9,
        penalty=1.0,
        min_score=0.0,
        softmax_min_scale=0.1,
    )
    assert b.shape[0] == 1


# Test _compute_adaptive_cap
def test_compute_adaptive_cap():
    # Empty boxes
    assert _compute_adaptive_cap(torch.zeros((0, 4)), torch.zeros((0,))) == 40

    # Sparse scene
    boxes_sparse = torch.tensor([[10, 10, 20, 20]], dtype=torch.float32)
    scores_sparse = torch.tensor([0.3], dtype=torch.float32)
    assert _compute_adaptive_cap(boxes_sparse, scores_sparse) == 40

    # Crowd scene
    # avg_score > 0.55 (+0.3), n > 50 (+0.3), median_area > 8000 (+0.2)
    # crowd_factor = 0.8
    boxes_crowd = torch.tensor([[0, 0, 100, 100]] * 60, dtype=torch.float32)
    scores_crowd = torch.tensor([0.9] * 60, dtype=torch.float32)
    cap = _compute_adaptive_cap(
        boxes_crowd, scores_crowd, base_cap=40, min_cap=15, max_cap=60
    )
    assert cap == 24  # 40 * (1 - 0.8*0.5) = 24


# Test _build_active_track_priors
def test_build_active_track_priors():
    tracker = MagicMock()
    # Empty snapshots
    tracker.get_state_snapshots.return_value = []
    pb, pc = _build_active_track_priors(tracker, torch.device("cpu"))
    assert pb is None
    assert pc is None

    # With snapshots
    snap1 = MagicMock()
    snap1.age = 5
    snap1.score = 0.8
    snap1.class_id = 0
    snap1.state = [
        10.0,
        20.0,
        2.0,
        10.0,
    ]  # cx, cy, a, h -> w = a*h = 20. cx - w/2 = 0, cy - h/2 = 15, cx + w/2 = 20, cy + h/2 = 25

    snap2 = MagicMock()
    snap2.age = 1
    snap2.score = 0.3
    snap2.class_id = 1
    snap2.state = [100.0, 100.0, 1.0, 10.0]

    tracker.get_state_snapshots.return_value = [snap1, snap2]

    # age threshold filter out snap2
    pb, pc = _build_active_track_priors(
        tracker, torch.device("cpu"), min_track_age=3, min_track_score=0.5
    )
    assert pb is not None
    assert pc is not None
    assert pb.shape[0] == 1
    assert pc.shape[0] == 1
    assert torch.allclose(pb[0], torch.tensor([0.0, 15.0, 20.0, 25.0]))
    assert pc[0].item() == 0


# Test _env_flag_enabled
def test_env_flag_enabled():
    with patch.dict(os.environ, {"TEST_FLAG": "1"}):
        assert _env_flag_enabled("TEST_FLAG") is True
    with patch.dict(os.environ, {"TEST_FLAG": "0"}):
        assert _env_flag_enabled("TEST_FLAG") is False
    with patch.dict(os.environ, {"TEST_FLAG": "true"}):
        assert _env_flag_enabled("TEST_FLAG") is True
    with patch.dict(os.environ, {"TEST_FLAG": "false"}):
        assert _env_flag_enabled("TEST_FLAG") is False


# Test _build_cpp_seq_config
def test_build_cpp_seq_config(tmp_path):
    # Setup directories
    seq_dir = tmp_path / "train" / "SEQ1" / "img1"
    seq_dir.mkdir(parents=True)
    # create dummy image and seqinfo.ini
    (seq_dir / "000001.jpg").touch()

    seqinfo = tmp_path / "train" / "SEQ1" / "seqinfo.ini"
    with open(seqinfo, "w") as f:
        f.write("[Sequence]\nimWidth=640\nimHeight=480\n")

    cfg = SimpleNamespace(
        detection_quality_w_aspect=0.4,
        reid_enabled=True,
        reid_engine="dummy.engine",
        reid_model="dinov2",
        crop_hw=(112, 112),
        kwargs={"confirm_streak": 2, "confirm_score_thresh": 0.1},
    )

    try:
        from saccade_eval_ext import CppSequenceConfig  # noqa: F401
    except ImportError:
        pytest.skip("saccade_eval_ext is required to test _build_cpp_seq_config")

    c = _build_cpp_seq_config(cfg, "SEQ1", str(tmp_path), split="train")
    assert c.name == "SEQ1"
    assert c.width == 640
    assert c.height == 480
    assert c.w_aspect == pytest.approx(0.4)
    assert c.confirm_streak == 2
    assert c.reid_engine_path == "dummy.engine"
    assert c.reid_model_type == 1  # dinov2
    assert c.reid_crop_h == 112
    assert c.reid_crop_w == 112
