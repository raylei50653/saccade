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

    async def generate(self, prompt: str, image_data: Optional[str] = None, max_tokens: int = 128, temperature: float = 0.7) -> str:
        """
        執行非同步文字生成 (支援 VLM 圖片輸入)
        """
        url = f"{self.base_url}/completion"
        
        # 針對 VLM 模型 (如 Qwen2-VL) 使用特定的 Prompt 格式與 image_data 欄位
        full_prompt = f"USER: [img-0] {prompt}\nASSISTANT: " if image_data else f"USER: {prompt}\nASSISTANT: "
        
        payload = {
            "prompt": full_prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": ["USER:", "\n", "</s>"],
            "stream": False
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # 確保 Base64 乾淨且無頭
                if image_data:
                    clean_image = image_data.split(",")[-1] if "," in image_data else image_data
                    payload["image_data"] = [{"data": clean_image, "id": 0}]

                response = await client.post(url, json=payload)
                
                if response.status_code != 200:
                    print(f"❌ [LLMEngine] Server returned error {response.status_code}: {response.text}")
                    return f"Error: LLM server returned {response.status_code}"
                
                data = response.json()
                return str(data.get("content", "")).strip()
                
            except httpx.ConnectError:
                return "Error: LLM server connection refused (is it running on port 8080?)"
            except httpx.TimeoutException:
                return "Error: LLM server request timed out"
            except Exception as e:
                import traceback
                print(f"❌ [LLMEngine] Unexpected error: {str(e)}")
                traceback.print_exc()
                return f"Error: {type(e).__name__} - {str(e)}"

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
