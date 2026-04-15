import pytest
import os
import shutil
import time
from unittest.mock import patch, AsyncMock
from storage.redis_cache import RedisCache
from storage.chroma_store import ChromaStore


@pytest.mark.anyio
async def test_redis_cache_operations():
    # Mock redis client
    mock_client = AsyncMock()

    with patch("redis.asyncio.from_url", return_value=mock_client):
        cache = RedisCache()
        await cache.connect()

        # Test event stream add
        mock_client.xadd.return_value = "123-0"
        msg_id = await cache.add_to_stream({"data": 1})
        assert mock_client.xadd.called
        assert msg_id == "123-0"

        # Test stream read batch
        mock_client.xreadgroup.return_value = [
            (cache.stream_name, [("123-0", {"data": '{"test": 1}'})])
        ]
        events = await cache.read_stream_batch()
        assert len(events) == 1
        assert events[0][0] == "123-0"
        assert events[0][1] == {"test": 1}

        # Test ack
        await cache.acknowledge(["123-0"])
        assert mock_client.xack.called


def test_chroma_store_operations():
    test_db = "./storage/test_pytest_db"
    if os.path.exists(test_db):
        shutil.rmtree(test_db)

    try:
        store = ChromaStore(path=test_db, collection_name="test_coll")

        # Add
        mid = store.add_memory(
            content="Person detected at gate.",
            metadata={"source": "cam1", "timestamp": time.time()},
        )
        assert mid is not None

        # Query
        results = store.hybrid_query("Who is at the gate?")
        assert len(results["ids"]) > 0
        assert "Person" in results["documents"][0][0]

    finally:
        if os.path.exists(test_db):
            shutil.rmtree(test_db)
