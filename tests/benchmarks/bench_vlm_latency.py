import asyncio
import time
import os
import cv2
import base64
from cognition.llm_engine import LLMEngine
from media.mediamtx_client import MediaMTXClient
from dotenv import load_dotenv

load_dotenv()

async def benchmark_vlm():
    print("🚀 [Benchmark] Starting VLM Performance Test...")
    
    # 1. 初始化引擎與媒體客戶端
    engine = LLMEngine()
    dummy_video = os.getenv("DUMMY_VIDEO_PATH", "assets/videos/demo.mp4")
    media = MediaMTXClient(dummy_video=dummy_video)
    
    if not media.connect():
        print("❌ Failed to connect to video source.")
        return

    # 等待第一影格
    time.sleep(1)
    ret, frame = media.grab_frame()
    if not ret or frame is None:
        print("❌ Failed to grab frame for benchmark.")
        return

    print(f"📸 Image Resolution: {frame.shape[1]}x{frame.shape[0]}")

    # 2. 測試：影格編碼速度 (OpenCV -> Base64)
    start_enc = time.perf_counter()
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buffer).decode("utf-8")
    enc_time = (time.perf_counter() - start_enc) * 1000
    print(f"  ⚡ Encoding (JPEG+Base64): {enc_time:.2f} ms")

    # 3. 測試：VLM 推理速度 (llama-server)
    prompt = "Describe the main objects and their actions in this scene concisely."
    
    print("  🧠 Sending request to VLM (llama-server)...")
    start_inf = time.perf_counter()
    
    response = await engine.generate(prompt, image_data=img_b64, max_tokens=128)
    
    inf_time = (time.perf_counter() - start_inf) * 1000
    
    # 4. 結果分析
    if "Error" in response:
        print(f"  ❌ Inference Failed: {response}")
    else:
        char_count = len(response)
        words = len(response.split())
        print(f"\n💡 [VLM Result]: {response}")
        print("\n📊 Performance Metrics:")
        print(f"  - Total Latency: {inf_time:.2f} ms")
        print(f"  - Characters Generated: {char_count}")
        print(f"  - Words Generated: {words}")
        print(f"  - Est. Generation Speed: {words / (inf_time/1000):.2f} words/sec")

    media.release()

if __name__ == "__main__":
    asyncio.run(benchmark_vlm())
