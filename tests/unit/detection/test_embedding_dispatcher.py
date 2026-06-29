from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

import numpy as np
import pytest
import torch

from saccade.perception.embedding_dispatcher import AsyncEmbeddingDispatcher


class _FakeExtractor:
    device = "cpu"

    def __init__(self, feature_dim: int = 3) -> None:
        self.feature_dim = feature_dim
        self.calls: list[torch.Tensor] = []

    def extract(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.calls.append(input_tensor.detach().clone())
        values = torch.arange(
            input_tensor.size(0) * self.feature_dim,
            dtype=torch.float32,
            device=input_tensor.device,
        )
        return values.view(input_tensor.size(0), self.feature_dim)


async def _stop_worker(dispatcher: AsyncEmbeddingDispatcher) -> None:
    task = dispatcher._worker_task
    dispatcher.stop()
    if task is not None:
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_submit_empty_uses_extractor_feature_dim() -> None:
    dispatcher = AsyncEmbeddingDispatcher(_FakeExtractor(feature_dim=5))
    crops = torch.empty((0, 3, 224, 224), dtype=torch.float32)

    result = await dispatcher.submit(crops)

    assert result.shape == (0, 5)
    assert result.device == crops.device


@pytest.mark.asyncio
async def test_put_crops_empty_does_not_enqueue() -> None:
    dispatcher = AsyncEmbeddingDispatcher(_FakeExtractor())
    crops = torch.empty((0, 3, 224, 224), dtype=torch.float32)

    await dispatcher.put_crops("cam-1", crops, [])

    assert dispatcher.queue.qsize() == 0


@pytest.mark.asyncio
async def test_start_is_idempotent() -> None:
    dispatcher = AsyncEmbeddingDispatcher(_FakeExtractor())

    dispatcher.start()
    first_task = dispatcher._worker_task
    dispatcher.start()

    assert dispatcher._worker_task is first_task
    await _stop_worker(dispatcher)


@pytest.mark.asyncio
async def test_submit_runs_extractor_and_returns_features() -> None:
    extractor = _FakeExtractor(feature_dim=2)
    dispatcher = AsyncEmbeddingDispatcher(extractor, max_batch=4)
    crops = torch.ones((2, 3, 8, 8), dtype=torch.float32)
    dispatcher.start()

    try:
        result = await asyncio.wait_for(dispatcher.submit(crops), timeout=1.0)
    finally:
        await _stop_worker(dispatcher)

    assert len(extractor.calls) == 1
    torch.testing.assert_close(extractor.calls[0], crops)
    torch.testing.assert_close(result, torch.tensor([[0.0, 1.0], [2.0, 3.0]]))


@pytest.mark.asyncio
async def test_put_crops_invokes_callback_with_numpy_embeddings() -> None:
    payload: dict[str, Any] = {}
    ready = asyncio.Event()

    async def on_ready(
        stream_id: str,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> None:
        payload["stream_id"] = stream_id
        payload["embeddings"] = embeddings
        payload["metadata"] = metadata
        ready.set()

    dispatcher = AsyncEmbeddingDispatcher(
        _FakeExtractor(feature_dim=2),
        max_batch=4,
        on_embeddings_ready=on_ready,
    )
    crops = torch.ones((1, 3, 8, 8), dtype=torch.float32)
    metadata = [{"track_id": 7}]
    dispatcher.start()

    try:
        await dispatcher.put_crops("cam-1", crops, metadata)
        await asyncio.wait_for(ready.wait(), timeout=1.0)
    finally:
        await _stop_worker(dispatcher)

    assert payload["stream_id"] == "cam-1"
    np.testing.assert_array_equal(payload["embeddings"], np.array([[0.0, 1.0]]))
    assert payload["metadata"] == metadata
