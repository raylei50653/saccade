import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock
from perception.entropy import EntropyTrigger
from pipeline.orchestrator import PipelineOrchestrator


@pytest.mark.anyio
async def test_entropy_trigger_emit_event():
    # Mock Redis
    mock_redis = AsyncMock()

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        trigger = EntropyTrigger(threshold=0.5)

        # Test case: 3 detections -> entropy 0.6 >= 0.5 threshold
        success = await trigger.process_frame(
            frame_id=100, detections=["person", "car", "dog"], source_path="test_source"
        )

        assert success is True
        assert mock_redis.rpush.called

        # Verify the event data
        args, kwargs = mock_redis.rpush.call_args
        queue_name = args[0]
        event_json = args[1]

        assert queue_name == "saccade:events"
        event_data = json.loads(event_json)
        assert event_data["type"] == "entropy_trigger"
        assert event_data["metadata"]["entropy_value"] == 0.6
        assert event_data["metadata"]["frame_id"] == 100
        assert "person" in event_data["metadata"]["objects"]


@pytest.mark.anyio
async def test_entropy_trigger_cooldown():
    # Mock Redis
    mock_redis = AsyncMock()

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        trigger = EntropyTrigger(threshold=0.1, cooldown=10.0)

        # First emit should succeed
        success1 = await trigger.process_frame(1, ["p1"], "s")
        assert success1 is True
        assert mock_redis.rpush.call_count == 1

        # Second emit immediately after should fail due to cooldown
        success2 = await trigger.process_frame(2, ["p2"], "s")
        assert success2 is False
        assert mock_redis.rpush.call_count == 1


@pytest.mark.anyio
async def test_orchestrator_process_event_batch():
    mock_redis = AsyncMock()
    mock_chroma = MagicMock()
    mock_collection = MagicMock()
    mock_chroma.collection = mock_collection

    with (
        patch("pipeline.orchestrator.RedisCache", return_value=mock_redis),
        patch("pipeline.orchestrator.ChromaStore", return_value=mock_chroma),
    ):
        orchestrator = PipelineOrchestrator()

        batch = [
            (
                "msg-1",
                {
                    "metadata": {
                        "frame_id": 100,
                        "entropy_value": 0.9,
                        "objects": ["person", "knife"],
                    }
                },
            )
        ]

        await orchestrator.process_event_batch(batch)

        assert mock_collection.add.called
        args, kwargs = mock_collection.add.call_args
        assert "knife" in kwargs["documents"][0]
        assert kwargs["metadatas"][0]["is_anomaly"] == 1
        assert mock_redis.acknowledge.called
