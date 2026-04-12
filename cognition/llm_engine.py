import os
import httpx
import asyncio
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class LLMEngine:
    """
    對接 llama-server (llama.cpp HTTP API) 的推理客戶端
    
    支援 OpenAI 格式與 llama.cpp 原生格式推理。
    """
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.timeout = httpx.Timeout(60.0, connect=5.0)

    async def generate(self, prompt: str, max_tokens: int = 128, temperature: float = 0.7) -> str:
        """
        執行非同步文字生成 (llama.cpp /completion API)
        """
        url = f"{self.base_url}/completion"
        payload = {
            "prompt": f"### Human: {prompt}\n### Assistant: ",
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": ["### Human:", "\n"],
            "stream": False
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return str(data.get("content", ""))
            except httpx.HTTPError as e:
                return f"Error: LLM server communication failed - {str(e)}"

    async def get_health(self) -> bool:
        """確認 llama-server 是否運作中"""
        url = f"{self.base_url}/health"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url)
                return response.status_code == 200
            except Exception:
                return False

if __name__ == "__main__":
    # 簡易測試範例
    async def test():
        engine = LLMEngine()
        if await engine.get_health():
            res = await engine.generate("What is the visual activity in the current frame?")
            print(f"LLM Response: {res}")
        else:
            print("LLM Server is offline.")

    asyncio.run(test())
