import pytest
import torch
from scripts.eval.mot17_args import build_parser
from saccade.perception.eval.config import parse_eval_config

from saccade.perception.eval.utils import (
    apply_narrow_person_score_bonus,
    box_iou_tuple,
    count_tile_seam_boxes,
    debug_frame_selected,
    mot_result_line,
    parse_debug_frame_ranges,
    safe_cpp_ptr,
    shift_box,
    tile_seam_mask,
)


# --- parse_debug_frame_ranges ---


def test_parse_debug_frame_ranges_single():
    assert parse_debug_frame_ranges("5") == [(5, 5)]


def test_parse_debug_frame_ranges_range():
    assert parse_debug_frame_ranges("10-20") == [(10, 20)]


def test_parse_debug_frame_ranges_mixed():
    assert parse_debug_frame_ranges("1-3, 7, 10-15") == [(1, 3), (7, 7), (10, 15)]


def test_parse_debug_frame_ranges_inverted_swapped():
    assert parse_debug_frame_ranges("20-10") == [(10, 20)]


def test_parse_debug_frame_ranges_empty():
    assert parse_debug_frame_ranges("") == []


# --- debug_frame_selected ---


def test_debug_frame_selected_in_range():
    assert debug_frame_selected("SEQ1", 5, "SEQ1", [(1, 10)]) is True


def test_debug_frame_selected_out_of_range():
    assert debug_frame_selected("SEQ1", 50, "SEQ1", [(1, 10)]) is False


def test_debug_frame_selected_wrong_seq():
    assert debug_frame_selected("SEQ1", 5, "SEQ2", [(1, 10)]) is False


def test_debug_frame_selected_no_ranges_matches_all():
    assert debug_frame_selected("SEQ1", 999, "SEQ1", []) is True


def test_debug_frame_selected_empty_target_seq():
    assert debug_frame_selected("SEQ1", 5, "", []) is False


# --- box_iou_tuple ---


def test_box_iou_tuple_overlapping():
    iou = box_iou_tuple((0, 0, 10, 10), (5, 5, 15, 15))
    assert abs(iou - 25.0 / 175.0) < 1e-4


def test_box_iou_tuple_no_overlap():
    assert box_iou_tuple((0, 0, 5, 5), (10, 10, 20, 20)) == pytest.approx(0.0)


def test_box_iou_tuple_identical():
    assert box_iou_tuple((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


# --- shift_box ---


def test_shift_box_translates():
    result = shift_box((0, 0, 10, 10), (3, 4), 2.0)
    assert result == pytest.approx((6.0, 8.0, 16.0, 18.0))


def test_shift_box_zero_velocity():
    box = (1.0, 2.0, 5.0, 6.0)
    assert shift_box(box, (0, 0), 100.0) == pytest.approx(box)


# --- mot_result_line ---


def test_mot_result_line_format():
    line = mot_result_line(1, 42, (10.0, 20.0, 110.0, 120.0), 0.9, 1920, 1080)
    parts = line.split(",")
    assert parts[0] == "1"
    assert parts[1] == "42"
    assert line.endswith("-1,-1,-1")


def test_mot_result_line_clips_negative():
    line = mot_result_line(1, 1, (-5.0, -5.0, 10.0, 10.0), 0.5, 100, 100)
    parts = line.split(",")
    assert float(parts[2]) == pytest.approx(0.0)
    assert float(parts[3]) == pytest.approx(0.0)


# --- safe_cpp_ptr ---


def test_safe_cpp_ptr_no_attr():
    assert safe_cpp_ptr(object()) == 0


def test_safe_cpp_ptr_with_attr():
    class Obj:
        cpp_ptr = 42

    assert safe_cpp_ptr(Obj()) == 42


# --- apply_narrow_person_score_bonus ---


def test_narrow_person_bonus_zero_returns_same():
    scores = torch.tensor([0.5, 0.6])
    boxes = torch.tensor([[0.0, 0.0, 10.0, 50.0], [0.0, 0.0, 20.0, 60.0]])
    classes = torch.tensor([0, 0])
    result = apply_narrow_person_score_bonus(
        boxes,
        scores,
        classes,
        frame_w=1000,
        frame_h=1000,
        person_class=0,
        bonus=0.0,
        max_width_ratio=1.0,
        min_height_ratio=0.0,
        min_aspect=0.0,
        max_aspect=100.0,
    )
    assert result is scores


def test_narrow_person_bonus_clips_at_one():
    scores = torch.tensor([0.95])
    boxes = torch.tensor([[0.0, 0.0, 5.0, 50.0]])
    classes = torch.tensor([0])
    result = apply_narrow_person_score_bonus(
        boxes,
        scores,
        classes,
        frame_w=1000,
        frame_h=1000,
        person_class=0,
        bonus=0.10,
        max_width_ratio=1.0,
        min_height_ratio=0.0,
        min_aspect=0.0,
        max_aspect=100.0,
    )
    assert float(result[0]) == pytest.approx(1.0)


def test_narrow_person_bonus_empty_inputs():
    scores = torch.zeros((0,))
    boxes = torch.zeros((0, 4))
    classes = torch.zeros((0,), dtype=torch.long)
    result = apply_narrow_person_score_bonus(
        boxes,
        scores,
        classes,
        frame_w=1000,
        frame_h=1000,
        person_class=0,
        bonus=0.1,
        max_width_ratio=1.0,
        min_height_ratio=0.0,
        min_aspect=0.0,
        max_aspect=100.0,
    )
    assert result is scores


# --- tile_seam_mask / count_tile_seam_boxes ---


def test_tile_seam_mask_empty_boxes():
    boxes = torch.zeros((0, 4))
    mask = tile_seam_mask(boxes, tiling="960p_2x2", h_orig=720, w_orig=1280)
    assert mask.shape == (0,)


def test_tile_seam_mask_unknown_tiling():
    boxes = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
    mask = tile_seam_mask(boxes, tiling="native_960", h_orig=720, w_orig=1280)
    assert bool(mask[0]) is False


def test_tile_seam_mask_sahi_matches_2x2():
    boxes = torch.tensor([[300.0, 100.0, 350.0, 200.0]])
    native_mask = tile_seam_mask(boxes, tiling="960p_2x2", h_orig=720, w_orig=1280)
    sahi_mask = tile_seam_mask(boxes, tiling="sahi_960p_2x2", h_orig=720, w_orig=1280)
    assert torch.equal(native_mask, sahi_mask)


def test_count_tile_seam_boxes_returns_int():
    boxes = torch.tensor([[300.0, 100.0, 350.0, 200.0]])
    result = count_tile_seam_boxes(boxes, tiling="960p_2x2", h_orig=720, w_orig=1280)
    assert isinstance(result, int)


def test_mot17_parser_defaults_to_fixed_interval_reid():
    args = build_parser().parse_args([])

    assert args.need_reid is False
    assert args.reid_interval == 20
    assert args.birth_quality_gate is False
    assert args.birth_min_quality == 0.0
    assert args.birth_quality_score_bias == 0.15


def test_mot17_parser_accepts_sahi_tiling():
    args = build_parser().parse_args(["--tiling", "sahi_960p_2x2"])

    assert args.tiling == "sahi_960p_2x2"


def test_parse_eval_config_defaults_to_fixed_interval_reid(tmp_path):
    cfg = parse_eval_config(
        output=str(tmp_path),
        data_root="datasets/MOT17",
        split="train",
        sequences="MOT17-04-SDP",
        conf_threshold=0.05,
        reid_mode="semantic",
        reid_model="siglip2",
        profile_stages=False,
        kwargs={},
    )

    assert cfg.need_reid_enabled is False
    assert cfg.reid_interval == 20
    assert cfg.birth_quality_gate is False
    assert cfg.birth_min_quality == 0.0
    assert cfg.birth_quality_score_bias == 0.15


def test_parse_eval_config_preserves_sahi_tiling(tmp_path):
    cfg = parse_eval_config(
        output=str(tmp_path),
        data_root="datasets/MOT17",
        split="train",
        sequences="MOT17-04-SDP",
        conf_threshold=0.05,
        reid_mode="semantic",
        reid_model="siglip2",
        profile_stages=False,
        kwargs={"tiling": "sahi_960p_2x2"},
    )

    assert cfg.tiling == "sahi_960p_2x2"
    assert cfg.nms_iou_threshold == pytest.approx(0.5)
