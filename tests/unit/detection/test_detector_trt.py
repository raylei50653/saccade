"""Unit tests for the TRT YOLO detector and batched proxy (perception.detector_trt)."""

# scope: perception
# function: behavior
# lifecycle: active

from __future__ import annotations

from typing import Optional

import pytest
import torch

from saccade.perception.detector_trt import BatchedDetectorProxy, TRTYoloDetector


def _stub_detector() -> TRTYoloDetector:
    detector = TRTYoloDetector.__new__(TRTYoloDetector)
    detector.device = "cpu"
    detector.output_name = "output0"
    return detector


def test_empty_result_uses_int32_classes() -> None:
    detector = _stub_detector()

    boxes, scores, classes, extra = detector._empty_result()

    assert boxes.shape == (0, 4)
    assert scores.shape == (0,)
    assert classes.shape == (0,)
    assert classes.dtype == torch.int32
    assert extra is None


def test_detect_batch_empty_input_returns_without_inference() -> None:
    detector = _stub_detector()

    def fail_infer(input_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        raise AssertionError("empty batch should not enter TensorRT inference")

    detector.infer_raw_batch = fail_infer  # type: ignore[method-assign]
    images = torch.empty((0, 3, 640, 640), dtype=torch.float32)

    assert detector.detect_batch(images) == []


def test_detect_batch_decodes_scores_classes_and_pose_extra() -> None:
    detector = _stub_detector()
    raw = torch.zeros((1, 2, 57), dtype=torch.float32)
    raw[0, 0, :6] = torch.tensor([1.0, 2.0, 3.0, 4.0, 0.9, 5.0])
    raw[0, 0, 6:] = torch.arange(51, dtype=torch.float32)
    raw[0, 1, :6] = torch.tensor([9.0, 9.0, 10.0, 10.0, 0.1, 1.0])

    detector.infer_raw_batch = lambda input_tensor: {"output0": raw}  # type: ignore[method-assign]
    images = torch.zeros((1, 3, 640, 640), dtype=torch.float32)

    [(boxes, scores, classes, keypoints)] = detector.detect_batch(
        images,
        conf_threshold=0.25,
    )

    torch.testing.assert_close(boxes, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    torch.testing.assert_close(scores, torch.tensor([0.9]))
    torch.testing.assert_close(classes, torch.tensor([5], dtype=torch.int32))
    assert keypoints is not None
    assert keypoints.shape == (1, 17, 3)
    torch.testing.assert_close(
        keypoints.flatten(), torch.arange(51, dtype=torch.float32)
    )


def test_detect_batch_uses_embedding_side_output() -> None:
    detector = _stub_detector()
    raw = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0, 0.9, 0.0],
                [5.0, 6.0, 7.0, 8.0, 0.1, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    embeddings = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32)
    detector.infer_raw_batch = lambda input_tensor: {  # type: ignore[method-assign]
        "output0": raw,
        "embeddings": embeddings,
    }

    [(boxes, _scores, _classes, extra)] = detector.detect_batch(
        torch.zeros((1, 3, 640, 640)),
        conf_threshold=0.25,
    )

    assert boxes.shape == (1, 4)
    assert extra is not None
    torch.testing.assert_close(extra, torch.tensor([[0.1, 0.2]]))


def test_batched_detector_proxy_decodes_raw_without_reinference() -> None:
    class _FakeBase:
        output_name = "output0"

        def __init__(self) -> None:
            self.detect_batch_calls = 0

        def detect_batch(
            self,
            input_tensor: torch.Tensor,
            conf_threshold: float = 0.25,
        ) -> list[object]:
            self.detect_batch_calls += 1
            raise AssertionError("raw batched output must not be reinferred")

        def _decode_outputs(
            self,
            outputs: dict[str, torch.Tensor],
            batch_size: int,
            conf_threshold: float,
        ) -> list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
        ]:
            raw = outputs["output0"]
            assert raw.shape == (1, 2, 6)
            assert batch_size == 1
            assert conf_threshold == pytest.approx(0.7)
            return [
                (
                    raw[0, :1, :4],
                    raw[0, :1, 4],
                    raw[0, :1, 5].to(torch.int32),
                    None,
                )
            ]

    class _FakeBatcher:
        def __init__(self) -> None:
            self._base = _FakeBase()

    proxy = BatchedDetectorProxy.__new__(BatchedDetectorProxy)
    proxy._batcher = _FakeBatcher()
    proxy._base = proxy._batcher._base
    raw = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0, 0.8, 2.0], [0.0, 0.0, 0.0, 0.0, 0.1, 0.0]]]
    )
    proxy.detect_raw = lambda input_tensor: raw  # type: ignore[method-assign]

    results = proxy.detect_batch(torch.zeros((1, 3, 640, 640)), conf_threshold=0.7)

    boxes, scores, classes, extra = results[0]
    torch.testing.assert_close(boxes, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    torch.testing.assert_close(scores, torch.tensor([0.8]))
    torch.testing.assert_close(classes, torch.tensor([2], dtype=torch.int32))
    assert extra is None
    assert proxy._base.detect_batch_calls == 0
