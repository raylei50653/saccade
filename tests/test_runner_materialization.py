from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "build"))

from saccade.perception.eval.helpers import materialize_gpu_track_results as _materialize_gpu_track_results  # noqa: E402


def test_materialize_gpu_track_results_empty() -> None:
    result = _materialize_gpu_track_results(
        {
            "boxes": torch.empty((4, 4), dtype=torch.float32),
            "scores": torch.empty((4,), dtype=torch.float32),
            "ids": torch.empty((4,), dtype=torch.int32),
            "classes": torch.empty((4,), dtype=torch.int32),
            "det_idx": torch.empty((4,), dtype=torch.int32),
            "count": torch.tensor(0, dtype=torch.int32),
        }
    )

    assert result["count"] == 0
    assert tuple(result["boxes"].shape) == (0, 4)
    assert tuple(result["scores"].shape) == (0,)
    assert tuple(result["ids"].shape) == (0,)
    assert tuple(result["classes"].shape) == (0,)
    assert tuple(result["det_idx"].shape) == (0,)


def test_materialize_gpu_track_results_preserves_fields() -> None:
    result = _materialize_gpu_track_results(
        {
            "boxes": torch.tensor(
                [
                    [1.0, 2.0, 30.0, 40.0],
                    [5.0, 6.0, 50.0, 60.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            "scores": torch.tensor([0.9, 0.8, 0.0], dtype=torch.float32),
            "ids": torch.tensor([11, 22, 0], dtype=torch.int32),
            "classes": torch.tensor([1, 3, 0], dtype=torch.int32),
            "det_idx": torch.tensor([7, 9, -1], dtype=torch.int32),
            "count": torch.tensor(2, dtype=torch.int32),
        }
    )

    assert result["count"] == 2
    assert result["boxes"].tolist() == [[1.0, 2.0, 30.0, 40.0], [5.0, 6.0, 50.0, 60.0]]
    assert torch.equal(result["scores"], torch.tensor([0.9, 0.8], dtype=torch.float32))
    assert result["ids"].tolist() == [11, 22]
    assert result["classes"] is not None
    assert result["classes"].tolist() == [1, 3]
    assert result["det_idx"] is not None
    assert result["det_idx"].tolist() == [7, 9]


def test_materialize_gpu_track_results_overrides_default_class_and_det_idx() -> None:
    result = _materialize_gpu_track_results(
        {
            "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32),
            "scores": torch.tensor([0.95], dtype=torch.float32),
            "ids": torch.tensor([42], dtype=torch.int32),
            "classes": torch.tensor([99], dtype=torch.int32),
            "det_idx": torch.tensor([5], dtype=torch.int32),
            "count": torch.tensor(1, dtype=torch.int32),
        },
        default_class_id=1,
        include_det_idx=False,
    )

    assert result["count"] == 1
    assert result["classes"] is not None
    assert result["classes"].tolist() == [1]
    assert result["det_idx"] is None
