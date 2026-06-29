from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saccade.pipeline.health import check_redis


@pytest.mark.asyncio
async def test_check_redis_measures_stream_depth() -> None:
    client = MagicMock()
    client.ping = AsyncMock()
    client.xlen = AsyncMock(return_value=7)
    client.aclose = AsyncMock()

    with patch("saccade.pipeline.health.aioredis.from_url", return_value=client):
        status, depth = await check_redis()

    assert status.ok is True
    assert depth == 7
    client.xlen.assert_called_once_with("saccade:stream")


@pytest.mark.asyncio
async def test_check_redis_closes_connection_after_xlen() -> None:
    client = MagicMock()
    client.ping = AsyncMock()
    client.xlen = AsyncMock(return_value=0)
    client.aclose = AsyncMock()

    with patch("saccade.pipeline.health.aioredis.from_url", return_value=client):
        await check_redis()

    client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_redis_closes_connection_when_xlen_fails() -> None:
    client = MagicMock()
    client.ping = AsyncMock()
    client.xlen = AsyncMock(side_effect=RuntimeError("xlen failed"))
    client.aclose = AsyncMock()

    with patch("saccade.pipeline.health.aioredis.from_url", return_value=client):
        status, depth = await check_redis()

    assert status.ok is False
    assert "xlen failed" in status.detail
    assert depth == 0
    client.aclose.assert_awaited_once()
