import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from cognition.llm_engine import LLMEngine
from cognition.resource_manager import ResourceManager

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

def test_resource_manager_select_profile():
    # Mock pynvml to simulate specific VRAM conditions
    with patch("pynvml.nvmlInit"), \
         patch("pynvml.nvmlDeviceGetHandleByIndex"), \
         patch("pynvml.nvmlDeviceGetMemoryInfo") as mock_mem, \
         patch("pynvml.nvmlShutdown"):
        
        # Scenario: 10GB free VRAM
        # 10GB - 2GB (reserve) = 8GB available -> should pick "8GB-balanced"
        mock_mem.return_value.free = 10 * 1024 * 1024 * 1024 
        
        manager = ResourceManager()
        profile = manager.select_optimal_profile(reserve_mb=2048)
        
        assert profile["name"] == "8GB-balanced"

def test_resource_manager_low_vram():
    with patch("pynvml.nvmlInit"), \
         patch("pynvml.nvmlDeviceGetHandleByIndex"), \
         patch("pynvml.nvmlDeviceGetMemoryInfo") as mock_mem, \
         patch("pynvml.nvmlShutdown"):
        
        # Scenario: 3GB free VRAM
        # 3GB - 2GB (reserve) = 1GB available -> should fall back to "4GB-low"
        mock_mem.return_value.free = 3 * 1024 * 1024 * 1024 
        
        manager = ResourceManager()
        profile = manager.select_optimal_profile(reserve_mb=2048)
        
        assert profile["name"] == "4GB-low"
