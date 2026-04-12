import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from cognition.llm_engine import LLMEngine

@pytest.mark.anyio
async def test_llm_engine_generate_success():
    engine = LLMEngine(base_url="http://fake-llm:8080")
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"content": "Test analysis result"}
    
    with patch("httpx.AsyncClient.post", return_value=mock_response):
        result = await engine.generate("Analyze this", image_data="base64data")
        assert result == "Test analysis result"

@pytest.mark.anyio
async def test_llm_engine_generate_timeout():
    engine = LLMEngine(base_url="http://fake-llm:8080")
    
    with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Timeout")):
        result = await engine.generate("Analyze this")
        assert "Error: LLM server request timed out" in result

@pytest.mark.anyio
async def test_llm_engine_health_ok():
    engine = LLMEngine(base_url="http://fake-llm:8080")
    
    mock_response = AsyncMock()
    mock_response.status_code = 200
    
    with patch("httpx.AsyncClient.get", return_value=mock_response):
        is_healthy = await engine.get_health()
        assert is_healthy is True

@pytest.mark.anyio
async def test_llm_engine_health_fail():
    engine = LLMEngine(base_url="http://fake-llm:8080")
    
    with patch("httpx.AsyncClient.get", side_effect=Exception("Connection refused")):
        is_healthy = await engine.get_health()
        assert is_healthy is False
