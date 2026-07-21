"""Tests for RedisCache and MicroBatcher (storage/redis_cache.py).

Covers:
  - MicroBatcher: add, flush, _flush_locked, timer cancellation
  - RedisCache: __init__, connect, disconnect, stream ops,
    object CRUD, cleanup, publish_event
"""

# scope: storage
# function: behavior
# lifecycle: active

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import ResponseError

from saccade.storage.redis_cache import MicroBatcher, RedisCache

if TYPE_CHECKING:
    pass


# ─── Fixtures / Helpers ──────────────────────────────────────────────────────


def _make_batcher(
    max_size: int = 5, window_ms: int = 100, queue: str = "test:queue"
) -> MicroBatcher:
    """Create a MicroBatcher with a mocked client."""
    mock_client = MagicMock()
    mock_client.rpush = AsyncMock()
    mock_client.expire = AsyncMock()
    batcher = MicroBatcher(mock_client, queue, window_ms=window_ms, max_size=max_size)
    return batcher


# ─── MicroBatcher ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_batcher_add_small_does_not_flush() -> None:
    """Adding fewer than max_size items does not flush."""
    batcher = _make_batcher(max_size=5)
    batcher._buf = []
    await batcher.add({"key": "val1"})
    await batcher.add({"key": "val2"})
    assert len(batcher._buf) == 2
    batcher.client.rpush.assert_not_called()


@pytest.mark.asyncio
async def test_batcher_add_reaches_max_size_flushes() -> None:
    """Adding exactly max_size items triggers _flush_locked."""
    batcher = _make_batcher(max_size=3)
    batcher._buf = []
    await batcher.add({"key": "val1"})
    await batcher.add({"key": "val2"})
    await batcher.add({"key": "val3"})
    assert len(batcher._buf) == 0
    batcher.client.rpush.assert_called_once()


@pytest.mark.asyncio
async def test_batcher_add_exceeds_max_size_flushes() -> None:
    """Adding more than max_size triggers flush."""
    batcher = _make_batcher(max_size=2)
    batcher._buf = []
    await batcher.add({"key": "val1"})
    await batcher.add({"key": "val2"})
    await batcher.add({"key": "val3"})
    assert len(batcher._buf) == 1  # 1 item left over
    assert batcher.client.rpush.call_count == 1


@pytest.mark.asyncio
async def test_batcher_flush_calls_rpush_and_expire() -> None:
    """flush() calls _flush_locked which does rpush + expire."""
    batcher = _make_batcher(max_size=10)
    batcher._buf = [json.dumps({"a": 1}), json.dumps({"a": 2})]
    batcher.client.rpush = AsyncMock()
    batcher.client.expire = AsyncMock()
    batcher._timer = None  # ensure no timer

    await batcher.flush()

    batcher.client.rpush.assert_called_once()
    batcher.client.expire.assert_called_once()
    assert batcher._buf == []


@pytest.mark.asyncio
async def test_batcher_flush_with_timer_cancels_timer() -> None:
    """flush() cancels any pending timer."""
    batcher = _make_batcher(max_size=10)
    batcher._buf = ["some_data"]
    batcher.client.rpush = AsyncMock()
    batcher.client.expire = AsyncMock()
    mock_timer = MagicMock()
    batcher._timer = mock_timer

    await batcher.flush()

    mock_timer.cancel.assert_called_once()
    assert batcher._timer is None


@pytest.mark.asyncio
async def test_batcher_flush_empty_does_nothing() -> None:
    """flush() with empty buffer does not call rpush."""
    batcher = _make_batcher(max_size=10)
    batcher._buf = []
    batcher.client.rpush = AsyncMock()
    batcher.client.expire = AsyncMock()
    batcher._timer = None

    await batcher.flush()

    batcher.client.rpush.assert_not_called()
    batcher.client.expire.assert_not_called()


@pytest.mark.asyncio
async def test_batcher_timer_schedules_flush() -> None:
    """add() with buffer below max_size schedules a timer-based flush."""
    batcher = _make_batcher(max_size=10, window_ms=50)
    batcher._buf = []
    batcher.client.rpush = AsyncMock()
    batcher.client.expire = AsyncMock()

    loop = asyncio.get_running_loop()
    original_call_later = loop.call_later
    timers_scheduled = []

    def mock_call_later(delay, callback):
        handle = MagicMock()
        timers_scheduled.append((delay, callback))
        return handle

    loop.call_later = mock_call_later
    try:
        await batcher.add({"key": "val1"})
    finally:
        loop.call_later = original_call_later

    assert len(timers_scheduled) == 1
    assert timers_scheduled[0][0] == 0.05  # window_ms / 1000.0


@pytest.mark.asyncio
async def test_batcher_timer_cancelled_on_flush() -> None:
    """When buffer reaches max_size and flushes, timer is not scheduled."""
    batcher = _make_batcher(max_size=2, window_ms=50)
    batcher._buf = []
    batcher.client.rpush = AsyncMock()
    batcher.client.expire = AsyncMock()

    loop = asyncio.get_running_loop()
    original_call_later = loop.call_later
    timers_scheduled = []

    def mock_call_later(delay, callback):
        handle = MagicMock()
        timers_scheduled.append((delay, callback))
        return handle

    loop.call_later = mock_call_later
    try:
        await batcher.add({"key": "val1"})
        await batcher.add({"key": "val2"})
    finally:
        loop.call_later = original_call_later

    # Only one timer should be scheduled (for first add); second add flushes immediately
    assert len(timers_scheduled) == 1


@pytest.mark.asyncio
async def test_batcher_add_serializes_json() -> None:
    """add() serializes dict to JSON string."""
    batcher = _make_batcher(max_size=2)
    batcher._buf = []
    batcher.client.rpush = AsyncMock()
    batcher.client.expire = AsyncMock()
    await batcher.add({"data": {"nested": True}})
    # 1 item added, max_size=2, so no flush yet
    assert len(batcher._buf) == 1
    assert json.loads(batcher._buf[0]) == {"data": {"nested": True}}


# ─── RedisCache ──────────────────────────────────────────────────────────────


async def _empty_scan_iter(match=None):
    return
    yield  # type: ignore[unreachable]


def _make_mock_client() -> MagicMock:
    """Create a MagicMock with AsyncMock methods."""
    mock = MagicMock()
    mock.xgroup_create = AsyncMock()
    mock.xadd = AsyncMock(return_value="0000-0")
    mock.xreadgroup = AsyncMock(return_value=[])
    mock.xack = AsyncMock()
    mock.scan_iter = MagicMock(side_effect=_empty_scan_iter)
    mock.delete = AsyncMock()
    mock.info = AsyncMock(return_value={"used_memory": 0})
    mock.set = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.aclose = AsyncMock()
    mock.pipeline = MagicMock(return_value=_make_mock_pipeline())
    return mock


def _make_mock_pipeline() -> MagicMock:
    mock = MagicMock()
    mock.xadd = MagicMock()
    mock.execute = AsyncMock(return_value=[])
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=None)
    return mock


@pytest.mark.asyncio
async def test_redis_cache_init_defaults() -> None:
    """RedisCache init uses default URL when no env var and no url param."""
    with patch.dict("os.environ", {}, clear=True):
        cache = RedisCache()
    assert cache.url == "redis://localhost:6379/0"
    assert cache.client is None
    assert cache.batchers == {}
    assert cache.stream_name == "saccade:stream"
    assert cache.max_len == 10000


@pytest.mark.asyncio
async def test_redis_cache_init_with_url() -> None:
    """RedisCache init uses provided URL over env var."""
    cache = RedisCache(url="redis://custom:6379/1")
    assert cache.url == "redis://custom:6379/1"


@pytest.mark.asyncio
async def test_redis_cache_init_uses_env_url() -> None:
    """RedisCache init uses REDIS_URL env var when no explicit URL."""
    with patch.dict("os.environ", {"REDIS_URL": "redis://envhost:6380/2"}):
        cache = RedisCache()
    assert cache.url == "redis://envhost:6380/2"


@pytest.mark.asyncio
async def test_connect_creates_client() -> None:
    """connect() creates a Redis client if None."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.connect()
        mock_redis.from_url.assert_called_once()
        assert cache.client is not None


@pytest.mark.asyncio
async def test_connect_skips_if_client_exists() -> None:
    """connect() does nothing if client already set."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        cache = RedisCache(url="redis://test:6379/0")
        cache.client = mock_client
        await cache.connect()
        mock_redis.from_url.assert_not_called()


@pytest.mark.asyncio
async def test_connect_creates_stream_group() -> None:
    """connect() creates stream group if it doesn't exist."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.connect()
        mock_client.xgroup_create.assert_called_once()


@pytest.mark.asyncio
async def test_connect_handles_busygroup_error() -> None:
    """connect() ignores BUSYGROUP error (group already exists)."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()

        async def raise_busygroup(*args, **kwargs):
            raise ResponseError("BUSYGROUP Consumer Group name already exists")

        mock_client.xgroup_create = raise_busygroup
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.connect()
        # Should not raise


@pytest.mark.asyncio
async def test_connect_raises_other_response_errors() -> None:
    """connect() raises non-BUSYGROUP ResponseError."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()

        async def raise_other_error(*args, **kwargs):
            raise ResponseError("OTHER ERROR")

        mock_client.xgroup_create = raise_other_error
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        with pytest.raises(ResponseError):
            await cache.connect()


@pytest.mark.asyncio
async def test_add_to_stream_calls_xadd() -> None:
    """add_to_stream() calls xadd with serialized payload."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_client.xadd = AsyncMock(return_value="0000-0")
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.connect()

        result = await cache.add_to_stream({"track_id": 42, "label": "person"})
        assert result == "0000-0"
        mock_client.xadd.assert_called_once()
        call_args = mock_client.xadd.call_args
        assert call_args[0][1] == {
            "data": json.dumps({"track_id": 42, "label": "person"})
        }
        assert call_args[1]["maxlen"] == 10000
        assert call_args[1]["approximate"] is True


@pytest.mark.asyncio
async def test_add_to_stream_auto_connects() -> None:
    """add_to_stream() auto-connects if client is None."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_client.xadd = AsyncMock(return_value="0000-0")
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        # client is None, should auto-connect
        await cache.add_to_stream({"key": "val"})
        mock_redis.from_url.assert_called_once()


@pytest.mark.asyncio
async def test_add_to_stream_batch_calls_pipeline() -> None:
    """add_to_stream_batch() uses pipeline for batch writes."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_pipe = _make_mock_pipeline()
        mock_pipe.execute = AsyncMock(return_value=["id1", "id2", "id3"])
        mock_client.pipeline = MagicMock(return_value=mock_pipe)
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.connect()

        events = [{"a": 1}, {"b": 2}, {"c": 3}]
        results = await cache.add_to_stream_batch(events)
        assert results == ["id1", "id2", "id3"]
        assert mock_client.pipeline.call_count == 1
        assert mock_pipe.execute.call_count == 1


@pytest.mark.asyncio
async def test_add_to_stream_batch_empty_returns_empty_list() -> None:
    """add_to_stream_batch([]) returns []."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.connect()

        results = await cache.add_to_stream_batch([])
        assert results == []


@pytest.mark.asyncio
async def test_read_stream_batch_returns_empty_when_no_streams() -> None:
    """read_stream_batch() returns [] when no streams."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_client.xreadgroup = AsyncMock(return_value=[])
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.connect()

        result = await cache.read_stream_batch()
        assert result == []


@pytest.mark.asyncio
async def test_read_stream_batch_parses_events() -> None:
    """read_stream_batch() parses xreadgroup results."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_client.xreadgroup = AsyncMock(
            return_value=[
                [
                    "saccade:stream",
                    [
                        ("id-1", {"data": json.dumps({"track_id": 1})}),
                        ("id-2", {"data": json.dumps({"track_id": 2})}),
                    ],
                ]
            ]
        )
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.connect()

        result = await cache.read_stream_batch(count=10, timeout_ms=200)
        assert len(result) == 2
        assert result[0] == ("id-1", {"track_id": 1})
        assert result[1] == ("id-2", {"track_id": 2})


@pytest.mark.asyncio
async def test_read_stream_batch_call_parameters() -> None:
    """read_stream_batch() passes correct parameters to xreadgroup."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_client.xreadgroup = AsyncMock(return_value=[])
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.connect()

        await cache.read_stream_batch(count=50, timeout_ms=1000)
        mock_client.xreadgroup.assert_called_once_with(
            "orchestrator_group",
            "worker_1",
            {"saccade:stream": ">"},
            count=50,
            block=1000,
        )


@pytest.mark.asyncio
async def test_acknowledge_calls_xack() -> None:
    """acknowledge() calls xack with message ids."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.connect()

        await cache.acknowledge(["id-1", "id-2", "id-3"])
        mock_client.xack.assert_called_once_with(
            "saccade:stream", "orchestrator_group", "id-1", "id-2", "id-3"
        )


@pytest.mark.asyncio
async def test_acknowledge_empty_list_does_nothing() -> None:
    """acknowledge([]) does not call xack."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.connect()

        await cache.acknowledge([])
        mock_client.xack.assert_not_called()


@pytest.mark.asyncio
async def test_disconnect_flushes_batchers_and_closes_client() -> None:
    """disconnect() flushes all batchers and closes client."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.connect()

        batcher = _make_batcher(max_size=10)
        cache.batchers["test"] = batcher
        batcher.flush = AsyncMock()

        await cache.disconnect()

        batcher.flush.assert_called_once()
        mock_client.aclose.assert_called_once()
        assert cache.client is None


@pytest.mark.asyncio
async def test_cleanup_expired_objects_below_threshold() -> None:
    """cleanup() does nothing when memory is below threshold."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_client.info = AsyncMock(return_value={"used_memory": 100 * 1024 * 1024})
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.connect()

        await cache.cleanup_expired_objects(max_memory_mb=500)
        mock_client.scan_iter.assert_not_called()
        mock_client.delete.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_expired_objects_above_threshold() -> None:
    """cleanup() deletes half of saccade:obj:* keys when memory exceeded."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_client.info = AsyncMock(return_value={"used_memory": 600 * 1024 * 1024})
        scan_keys = [
            b"saccade:obj:1",
            b"saccade:obj:2",
            b"saccade:obj:3",
            b"saccade:obj:4",
        ]

        async def _scan(match=None):
            for k in scan_keys:
                yield k

        mock_client.scan_iter.side_effect = _scan
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.connect()

        await cache.cleanup_expired_objects(max_memory_mb=500)
        mock_client.scan_iter.assert_called_once_with("saccade:obj:*")
        mock_client.delete.assert_called_once()
        # Should delete half: [1, 2] from [1, 2, 3, 4]
        call_args = mock_client.delete.call_args
        assert len(call_args[0]) == 2


@pytest.mark.asyncio
async def test_cleanup_expired_objects_no_keys() -> None:
    """cleanup() handles empty key list gracefully."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_client.info = AsyncMock(return_value={"used_memory": 600 * 1024 * 1024})
        mock_client.scan_iter.side_effect = _empty_scan_iter
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.connect()

        await cache.cleanup_expired_objects(max_memory_mb=500)
        mock_client.delete.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_expired_objects_error_handled() -> None:
    """cleanup() catches and prints exceptions without crashing."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_client.info = AsyncMock(side_effect=Exception("Connection lost"))
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.connect()

        await cache.cleanup_expired_objects()  # should not raise
        mock_client.scan_iter.assert_not_called()


@pytest.mark.asyncio
async def test_update_object_track_sets_key() -> None:
    """update_object_track() sets a key with JSON data and TTL."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.connect()

        await cache.update_object_track(
            42, "person", [10.0, 20.0, 100.0, 200.0], 1234.56
        )
        mock_client.set.assert_called_once()
        call_args = mock_client.set.call_args
        assert call_args[0][0] == "saccade:obj:42"
        expected_data = json.dumps(
            {
                "id": 42,
                "label": "person",
                "box": [10.0, 20.0, 100.0, 200.0],
                "timestamp": 1234.56,
            }
        )
        assert call_args[0][1] == expected_data
        assert call_args[1]["ex"] == 300


@pytest.mark.asyncio
async def test_get_active_objects_returns_ids() -> None:
    """get_active_objects() scans keys and returns valid object IDs."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        scan_keys: list[str] = [
            "saccade:obj:1",
            "saccade:obj:2",
            "saccade:obj:3",
        ]

        async def _scan(match=None):
            for k in scan_keys:
                yield k

        mock_client.scan_iter.side_effect = _scan
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.connect()

        result = await cache.get_active_objects()
        assert result == [1, 2, 3]


@pytest.mark.asyncio
async def test_get_active_objects_skips_invalid_keys() -> None:
    """get_active_objects() skips malformed keys."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        scan_keys = [
            "saccade:obj:42",
            "saccade:obj:bad",  # not an int
            "saccade:obj:",  # empty
        ]

        async def _scan(match=None):
            for k in scan_keys:
                yield k

        mock_client.scan_iter.side_effect = _scan
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.connect()

        result = await cache.get_active_objects()
        assert result == [42]


@pytest.mark.asyncio
async def test_get_active_objects_empty() -> None:
    """get_active_objects() returns [] when no keys match."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_client.scan_iter.side_effect = _empty_scan_iter
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.connect()

        result = await cache.get_active_objects()
        assert result == []


@pytest.mark.asyncio
async def test_get_object_history_returns_data() -> None:
    """get_object_history() returns parsed JSON data for existing key."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_client.get = AsyncMock(
            return_value=json.dumps(
                {
                    "id": 42,
                    "label": "car",
                    "box": [0.0, 0.0, 50.0, 50.0],
                    "timestamp": 100.0,
                }
            )
        )
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.connect()

        result = await cache.get_object_history(42)
        assert result is not None
        assert result["id"] == 42
        assert result["label"] == "car"


@pytest.mark.asyncio
async def test_get_object_history_returns_none_for_missing() -> None:
    """get_object_history() returns None for missing key."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_client.get = AsyncMock(return_value=None)
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.connect()

        result = await cache.get_object_history(999)
        assert result is None


@pytest.mark.asyncio
async def test_publish_event_creates_batcher() -> None:
    """publish_event() creates a batcher for new queue."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.connect()

        await cache.publish_event("my_queue", {"event": "test"})
        assert "my_queue" in cache.batchers
        assert isinstance(cache.batchers["my_queue"], MicroBatcher)


@pytest.mark.asyncio
async def test_publish_event_uses_existing_batcher() -> None:
    """publish_event() reuses existing batcher for same queue."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.connect()

        await cache.publish_event("q", {"a": 1})
        await cache.publish_event("q", {"b": 2})
        assert len(cache.batchers) == 1
        assert "q" in cache.batchers


# ─── Attributes ─────────────────────────────────────────────────────────────


def test_batcher_queue_attribute() -> None:
    """MicroBatcher stores the queue name."""
    batcher = _make_batcher(max_size=5)
    assert batcher.queue == "test:queue"


def test_batcher_max_size_attribute() -> None:
    """MicroBatcher stores max_size correctly."""
    batcher = _make_batcher(max_size=100)
    assert batcher.max_size == 100


def test_batcher_window_ms_attribute() -> None:
    """MicroBatcher stores window_ms correctly."""
    batcher = _make_batcher(window_ms=250)
    assert batcher.window_ms == 250


def test_redis_cache_stream_name_attribute() -> None:
    """RedisCache.stream_name is fixed."""
    cache = RedisCache()
    assert cache.stream_name == "saccade:stream"


def test_redis_cache_max_len_attribute() -> None:
    """RedisCache.max_len is fixed."""
    cache = RedisCache()
    assert cache.max_len == 10000


@pytest.mark.asyncio
async def test_disconnect_when_no_client() -> None:
    """disconnect() handles None client gracefully."""
    cache = RedisCache()
    await cache.disconnect()  # should not raise


@pytest.mark.asyncio
async def test_disconnect_no_batchers() -> None:
    """disconnect() with empty batchers dict does not fail."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.connect()
        cache.batchers = {}

        await cache.disconnect()
        mock_client.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_add_to_stream_batch_auto_connects() -> None:
    """add_to_stream_batch() auto-connects if client is None."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_pipe = _make_mock_pipeline()
        mock_pipe.execute = AsyncMock(return_value=["id1"])
        mock_client.pipeline = MagicMock(return_value=mock_pipe)
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.add_to_stream_batch([{"key": "val"}])
        mock_redis.from_url.assert_called_once()


@pytest.mark.asyncio
async def test_read_stream_batch_auto_connects() -> None:
    """read_stream_batch() auto-connects if client is None."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.read_stream_batch()
        mock_redis.from_url.assert_called_once()


@pytest.mark.asyncio
async def test_acknowledge_auto_connects() -> None:
    """acknowledge() auto-connects if client is None."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.acknowledge(["id-1"])
        mock_redis.from_url.assert_called_once()


@pytest.mark.asyncio
async def test_update_object_track_auto_connects() -> None:
    """update_object_track() auto-connects if client is None."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.update_object_track(1, "test", [0.0, 0.0, 10.0, 10.0], 0.0)
        mock_redis.from_url.assert_called_once()


@pytest.mark.asyncio
async def test_get_active_objects_auto_connects() -> None:
    """get_active_objects() auto-connects if client is None."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.get_active_objects()
        mock_redis.from_url.assert_called_once()


@pytest.mark.asyncio
async def test_get_object_history_auto_connects() -> None:
    """get_object_history() auto-connects if client is None."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.get_object_history(1)
        mock_redis.from_url.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_auto_connects() -> None:
    """cleanup_expired_objects() auto-connects if client is None."""
    with patch("saccade.storage.redis_cache.redis") as mock_redis:
        mock_client = _make_mock_client()
        mock_redis.from_url = MagicMock(return_value=mock_client)
        cache = RedisCache(url="redis://test:6379/0")
        await cache.cleanup_expired_objects()
        mock_redis.from_url.assert_called_once()
