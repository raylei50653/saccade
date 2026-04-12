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
        
        # Test event publish
        await cache.publish_event("test_q", {"data": 1})
        assert mock_client.rpush.called
        
        # Test state update
        await cache.update_object_state("123", {"pos": [1, 2]})
        assert mock_client.set.called
        
        # Test state get
        mock_client.get.return_value = '{"pos": [1, 2]}'
        state = await cache.get_object_state("123")
        assert state["pos"] == [1, 2]

def test_chroma_store_operations():
    test_db = "./storage/test_pytest_db"
    if os.path.exists(test_db):
        shutil.rmtree(test_db)
        
    try:
        store = ChromaStore(path=test_db, collection_name="test_coll")
        
        # Add
        mid = store.add_memory(
            content="Person detected at gate.",
            metadata={"source": "cam1", "timestamp": time.time()}
        )
        assert mid is not None
        
        # Query
        results = store.hybrid_query("Who is at the gate?")
        assert len(results["ids"]) > 0
        assert "Person" in results["documents"][0][0]
        
    finally:
        if os.path.exists(test_db):
            shutil.rmtree(test_db)
