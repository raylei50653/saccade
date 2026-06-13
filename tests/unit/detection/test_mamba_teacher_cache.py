import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from scripts.train.temporal_yolo import train_mamba_head


def _complete_cache() -> dict[str, torch.Tensor]:
    cached = {}
    for scale in ("p3", "p4", "p5"):
        cached[scale] = torch.ones(2)
        cached[f"cls_{scale}"] = torch.ones(3)
        cached[f"reg_{scale}"] = torch.ones(4)
    return cached


def test_split_cached_teacher_frame_returns_features_and_targets():
    features, teacher_cls, teacher_reg = train_mamba_head._split_cached_teacher_frame(
        _complete_cache()
    )

    assert len(features) == 3
    assert len(teacher_cls) == 3
    assert len(teacher_reg) == 3


def test_split_cached_teacher_frame_rejects_legacy_feature_only_cache():
    cached = {scale: torch.ones(2) for scale in ("p3", "p4", "p5")}

    with pytest.raises(ValueError, match="Regenerate it with --precompute-dir"):
        train_mamba_head._split_cached_teacher_frame(cached)


class _MustNotRun(nn.Module):
    def forward(self, *_args, **_kwargs):
        raise AssertionError("frozen YOLO path ran during cached Mamba training")


def test_cached_training_does_not_run_teacher_or_detect_head():
    train_mamba_head._feature_ram_cache = {"seq": {1: _complete_cache()}}
    batch = {
        "frames": torch.zeros(1, 1, 3, 2, 2),
        "seq": ["seq"],
        "frame_ids": [[1]],
    }
    try:
        features, teacher_cls, teacher_reg = train_mamba_head._load_or_forward_teacher(
            batch,
            0,
            cache_dir=Path("unused"),
            device=torch.device("cpu"),
            teacher=_MustNotRun(),
            detect_head=_MustNotRun(),
        )
    finally:
        train_mamba_head._feature_ram_cache = {}

    assert len(features) == len(teacher_cls) == len(teacher_reg) == 3


def test_cache_manifest_requires_matching_complete_lineage(tmp_path):
    manifest = {
        "schema": train_mamba_head.CACHE_SCHEMA,
        "status": "complete",
        "img_size": 640,
        "sequences": ["MOT17-04-SDP"],
        "total_frames": 1,
    }
    (tmp_path / train_mamba_head.CACHE_MANIFEST).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    loaded = train_mamba_head._validate_cache_manifest(
        tmp_path,
        {
            "schema": train_mamba_head.CACHE_SCHEMA,
            "img_size": 640,
            "sequences": ["MOT17-04-SDP"],
        },
        require_complete=True,
    )

    assert loaded["total_frames"] == 1


def test_cache_manifest_rejects_stale_teacher_lineage(tmp_path):
    manifest = {
        "schema": train_mamba_head.CACHE_SCHEMA,
        "status": "complete",
        "teacher_checkpoint_sha256": "old",
    }
    (tmp_path / train_mamba_head.CACHE_MANIFEST).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Cache manifest mismatch"):
        train_mamba_head._validate_cache_manifest(
            tmp_path,
            {
                "schema": train_mamba_head.CACHE_SCHEMA,
                "teacher_checkpoint_sha256": "new",
            },
            require_complete=True,
        )
