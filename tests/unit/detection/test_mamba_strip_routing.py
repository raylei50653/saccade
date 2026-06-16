import torch

from saccade.perception.temporal_yolo.mamba_head import (
    build_strip_oracle_positions,
)


def test_strip_oracle_routes_small_gt_and_keeps_fixed_budget():
    torch.manual_seed(3)
    boxes = [
        torch.tensor(
            [
                [80.0, 80.0, 87.0, 87.0],
                [320.0, 320.0, 360.0, 380.0],
            ]
        ),
        torch.zeros(0, 4),
    ]

    positions, mask = build_strip_oracle_positions(
        boxes,
        p3_hw=(80, 80),
        img_size=640,
        budget=16,
        small_threshold=8.0,
        device=torch.device("cpu"),
    )

    assert positions.shape == (2, 16, 2)
    assert mask.shape == (2, 16)
    assert mask.all()
    assert ((positions[0, :, 0] == 10) & (positions[0, :, 1] == 10)).any()
    assert (positions[..., 0] >= 0).all() and (positions[..., 0] < 80).all()
    assert (positions[..., 1] >= 0).all() and (positions[..., 1] < 80).all()


def test_strip_oracle_marks_padding_when_budget_exceeds_grid():
    positions, mask = build_strip_oracle_positions(
        [torch.zeros(0, 4)],
        p3_hw=(2, 2),
        img_size=16,
        budget=6,
        small_threshold=8.0,
        device=torch.device("cpu"),
    )

    assert positions.shape == (1, 6, 2)
    assert mask.tolist() == [[True, True, True, True, False, False]]
