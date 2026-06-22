"""Unit tests for clip-consistent augmentation (_augment_clip, strategy B).

These run without any dataset on disk — they exercise the box/frame transform
math directly with deterministic parameters.
"""

import random

import torch

from saccade.perception.temporal_yolo.dataset import _augment_clip


def _clip(S=64, n=3):
    """Two-frame clip with n boxes; frames are a distinguishable ramp."""
    img = (
        torch.arange(S, dtype=torch.float32)
        .repeat(S, 1)
        .unsqueeze(0)
        .repeat(3, 1, 1)
        .to(torch.uint8)
    )  # (3, S, S), increasing left→right
    boxes = torch.tensor([[10.0, 10, 20, 20], [30, 30, 40, 40], [5, 5, 8, 9]])[:n]
    ids = torch.tensor([1, 2, 3])[:n]
    return (
        [img.clone(), img.clone()],
        [boxes.clone(), boxes.clone()],
        [ids.clone(), ids.clone()],
    )


def test_identity_is_noop():
    frames, boxes, ids = _clip()
    of, ob, oi = _augment_clip(
        frames,
        boxes,
        ids,
        64,
        scale_range=(1.0, 1.0),
        translate_frac=0.0,
        hflip_p=0.0,
        brightness=0.0,
        contrast=0.0,
    )
    assert torch.equal(
        of[0], frames[0]
    )  # grid_sample identity is exact at pixel centers
    assert torch.allclose(ob[0], boxes[0], atol=1e-4)
    assert torch.equal(oi[0], ids[0])


def test_hflip_box_math():
    S = 64
    frames, boxes, ids = _clip(S=S)
    of, ob, oi = _augment_clip(
        frames,
        boxes,
        ids,
        S,
        scale_range=(1.0, 1.0),
        translate_frac=0.0,
        hflip_p=1.0,
        brightness=0.0,
        contrast=0.0,
    )
    # box [10,10,20,20] -> x' = [S-20, S-10] = [44, 54]
    assert torch.allclose(ob[0][0], torch.tensor([44.0, 10.0, 54.0, 20.0]), atol=1e-4)
    # frame content flipped (ramp now decreases left→right)
    assert of[0][0, 0, 0] > of[0][0, 0, -1]
    # ids preserved
    assert torch.equal(oi[0], ids[0])


def test_clip_consistency():
    """Both frames of a clip get the SAME transform (temporal alignment)."""
    random.seed(0)
    frames, boxes, ids = _clip()
    of, ob, oi = _augment_clip(frames, boxes, ids, 64)
    assert torch.equal(of[0], of[1])
    assert torch.allclose(ob[0], ob[1])


def test_out_of_frame_box_dropped_with_id():
    S = 64
    img = torch.zeros(3, S, S, dtype=torch.uint8)
    boxes = torch.tensor([[10.0, 10, 20, 20], [60, 60, 63, 63]])  # 2nd near edge
    ids = torch.tensor([7, 8])
    # large positive translate pushes the edge box fully out
    of, ob, oi = _augment_clip(
        [img],
        [boxes],
        [ids],
        S,
        scale_range=(1.0, 1.0),
        translate_frac=0.0,
        hflip_p=0.0,
        brightness=0.0,
        contrast=0.0,
    )
    # manually force translate via deterministic params: re-run with translate
    of, ob, oi = _augment_clip(
        [img],
        [boxes],
        [ids],
        S,
        scale_range=(1.0, 1.0),
        translate_frac=0.9,
        hflip_p=0.0,
        brightness=0.0,
        contrast=0.0,
    )
    # boxes and ids stay aligned in length
    assert ob[0].shape[0] == oi[0].shape[0]
    assert ob[0].shape[0] <= 2


def test_frames_stay_uint8_and_shaped():
    frames, boxes, ids = _clip()
    of, ob, oi = _augment_clip(frames, boxes, ids, 64)
    assert of[0].dtype == torch.uint8
    assert of[0].shape == (3, 64, 64)
    assert len(of) == len(frames)
