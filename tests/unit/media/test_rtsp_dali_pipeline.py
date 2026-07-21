"""Unit tests for the DALI RTSP pipeline optimizer (saccade.media.rtsp_dali_pipeline)."""

# scope: media
# function: behavior
# lifecycle: active

from __future__ import annotations

from typing import Any

import pytest
import torch

pytest.importorskip(
    "nvidia.dali",
    reason="DALI optional extra; install with: uv sync --extra dali",
)

from saccade.media import rtsp_dali_pipeline
from saccade.media.rtsp_dali_pipeline import DALIRTSPOptimizer


class _FakeDaliTensor:
    def shape(self) -> tuple[int, int, int, int]:
        return (2, 3, 640, 640)

    def data_ptr(self) -> int:
        return 1234


class _FakeDaliOutput:
    def as_tensor(self) -> _FakeDaliTensor:
        return _FakeDaliTensor()


class _FakePipeline:
    instances: list["_FakePipeline"] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.built = False
        self.feed_calls: list[tuple[str, list[torch.Tensor]]] = []
        self.run_calls = 0
        _FakePipeline.instances.append(self)

    def build(self) -> None:
        self.built = True

    def feed_input(self, name: str, tensors: list[torch.Tensor]) -> None:
        self.feed_calls.append((name, tensors))

    def run(self) -> list[_FakeDaliOutput]:
        self.run_calls += 1
        return [_FakeDaliOutput()]


@pytest.fixture(autouse=True)
def clear_fake_pipeline_instances() -> None:
    _FakePipeline.instances.clear()


def test_optimizer_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        DALIRTSPOptimizer(batch_size=0)


def test_process_rejects_empty_tensor_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rtsp_dali_pipeline, "RTSPDALIPipeline", _FakePipeline)
    optimizer = DALIRTSPOptimizer(batch_size=2)

    with pytest.raises(ValueError, match="at least one tensor"):
        optimizer.process([])


def test_process_primes_pipeline_pads_batch_and_returns_clone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_interfaces: list[dict[str, Any]] = []

    def fake_as_tensor(obj: Any, *, device: str) -> torch.Tensor:
        assert device == "cuda"
        captured_interfaces.append(obj.__cuda_array_interface__)
        return torch.ones((2, 3, 640, 640), dtype=torch.float32)

    monkeypatch.setattr(rtsp_dali_pipeline, "RTSPDALIPipeline", _FakePipeline)
    monkeypatch.setattr(rtsp_dali_pipeline.torch, "as_tensor", fake_as_tensor)
    optimizer = DALIRTSPOptimizer(batch_size=3, device_id=7, output_size=320)
    pipeline = _FakePipeline.instances[0]
    tensor = torch.arange(12, dtype=torch.uint8).view(3, 4).t()

    result = optimizer.process([tensor])

    assert pipeline.built
    assert pipeline.kwargs == {"batch_size": 3, "device_id": 7, "output_size": 320}
    assert len(pipeline.feed_calls) == 3
    assert pipeline.run_calls == 1
    for name, batch in pipeline.feed_calls:
        assert name == "rtsp_raw"
        assert len(batch) == 3
        assert all(item.is_contiguous() for item in batch)
    assert captured_interfaces == [
        {
            "shape": (2, 3, 640, 640),
            "typestr": "<f4",
            "data": (1234, False),
            "version": 3,
        }
    ]
    torch.testing.assert_close(result, torch.ones((2, 3, 640, 640)))


def test_process_only_primes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rtsp_dali_pipeline, "RTSPDALIPipeline", _FakePipeline)
    monkeypatch.setattr(
        rtsp_dali_pipeline.torch,
        "as_tensor",
        lambda _obj, *, device: torch.zeros((1, 3, 640, 640), dtype=torch.float32),
    )
    optimizer = DALIRTSPOptimizer(batch_size=1)
    pipeline = _FakePipeline.instances[0]
    tensor = torch.zeros((3, 4, 4), dtype=torch.uint8)

    optimizer.process([tensor])
    optimizer.process([tensor])

    assert len(pipeline.feed_calls) == 4
    assert pipeline.run_calls == 2
