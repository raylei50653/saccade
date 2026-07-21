"""Unit tests for temporal-YOLO training utilities (perception.temporal_yolo.training_utils)."""

# scope: detection
# function: behavior
# lifecycle: active

import pytest
import torch

from saccade.perception.temporal_yolo.training_utils import (
    build_warmup_cosine_scheduler,
    capture_rng_state,
    resolve_training_sequences,
    restore_rng_state,
    sha256_file,
)


def test_warmup_cosine_scheduler_has_one_continuous_schedule():
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_epochs=6,
        warmup_epochs=2,
        min_lr_ratio=0.1,
    )

    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(5):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs[0] == pytest.approx(0.5)
    assert lrs[1] == pytest.approx(1.0)
    assert lrs[-1] == pytest.approx(0.1)
    assert all(left >= right for left, right in zip(lrs[1:], lrs[2:]))


@pytest.mark.parametrize(
    ("total_epochs", "warmup_epochs"),
    [(0, 0), (3, -1), (3, 4)],
)
def test_warmup_cosine_scheduler_rejects_invalid_epoch_ranges(
    total_epochs,
    warmup_epochs,
):
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    with pytest.raises(ValueError):
        build_warmup_cosine_scheduler(
            optimizer,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
        )


def test_resolve_training_sequences_excludes_holdout(tmp_path):
    split = tmp_path / "train"
    for seq in ("MOT17-02-SDP", "MOT17-04-SDP", "MOT17-05-SDP"):
        (split / seq).mkdir(parents=True)

    train, holdout = resolve_training_sequences(
        tmp_path,
        "",
        "MOT17-02-SDP",
    )

    assert train == ["MOT17-04-SDP", "MOT17-05-SDP"]
    assert holdout == ["MOT17-02-SDP"]


def test_sha256_file(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"saccade")
    assert (
        sha256_file(path)
        == "3941d4453740985f0c363433c74070c2c4aa649d8a73fe52a573831c0f87aa7e"
    )


def test_rng_state_restores_torch_and_dataloader_generator():
    torch.manual_seed(7)
    generator = torch.Generator().manual_seed(11)
    state = capture_rng_state(generator)
    expected_torch = torch.rand(3)
    expected_data = torch.rand(3, generator=generator)

    torch.manual_seed(99)
    generator.manual_seed(101)
    restore_rng_state(state, generator)

    assert torch.equal(torch.rand(3), expected_torch)
    assert torch.equal(torch.rand(3, generator=generator), expected_data)


def test_warmup_cosine_scheduler_preserves_param_group_lr_ratio():
    first = torch.nn.Parameter(torch.zeros(()))
    second = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW(
        [
            {"params": [first], "lr": 1e-3},
            {"params": [second], "lr": 1e-5},
        ]
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_epochs=4,
        warmup_epochs=1,
    )

    for _ in range(4):
        gate_lr, yolo_lr = (group["lr"] for group in optimizer.param_groups)
        assert gate_lr / yolo_lr == pytest.approx(100.0)
        optimizer.step()
        scheduler.step()
