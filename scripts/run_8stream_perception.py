import asyncio
import time
import torch
from perception.detector_trt import TRTYoloDetector
from perception.dispatcher import AsyncDispatcher
from media.mediamtx_client import MediaMTXClient


async def run_stream_producer(
    stream_id: str, dispatcher: AsyncDispatcher, rtsp_url: str
) -> None:
    """
    單路串流生產者：負責抓取 RTSP 影格並推入分發器。
    """
    # MediaMTXClient 傳入正確的 rtsp_url
    media = MediaMTXClient(rtsp_url=rtsp_url)

    print(f"⏳ Connecting to {rtsp_url}...")
    while not media.connect():
        await asyncio.sleep(1)

    print(f"✅ Stream [{stream_id}] connected.")

    try:
        frame_count = 0
        last_frame_time = time.time()
        while True:
            ret, tensor = media.grab_tensor()
            if ret and tensor is not None:
                # 預處理：[H, W, C] -> [3, 640, 640] GPU Tensor
                input_tensor = tensor.permute(2, 0, 1).float() / 255.0
                yolo_input = torch.nn.functional.interpolate(
                    input_tensor.unsqueeze(0), size=(640, 640)
                ).squeeze(0)

                await dispatcher.put_frame(stream_id, yolo_input, time.time())
                frame_count += 1
                last_frame_time = time.time()
            else:
                if time.time() - last_frame_time > 5.0:
                    print(f"⚠️ Stream [{stream_id}] has not received frames for 5s.")
                    last_frame_time = time.time()

            await asyncio.sleep(0.001)
    except Exception as e:
        print(f"❌ Stream [{stream_id}] error: {e}")
    finally:
        media.release()


async def run_8stream_perception() -> None:
    """啟動 8 路感知管道"""
    print("🚀 Initializing 8-stream Perception Pipeline...")

    detector = TRTYoloDetector()
    # 設置 max_batch 為 8，以發揮 YOLO26 在 GPU 上的 Batching 效能
    from cognition.resource_manager import ResourceManager
    resource_manager = ResourceManager()
    dispatcher = AsyncDispatcher(detector, resource_manager, max_batch=8)
    await dispatcher.start()

    tasks = []
    for i in range(8):
        sid = f"stream_{i}"
        url = f"rtsp://127.0.0.1:8554/{sid}"
        tasks.append(asyncio.create_task(run_stream_producer(sid, dispatcher, url)))

    print("📊 Monitoring 8 streams. Press Ctrl+C to stop.")

    try:
        # 每隔 5 秒打印一次系統統計 (可選)
        while True:
            await asyncio.sleep(5)
            # 這裡可以加入讀取 dispatcher 統計的邏輯
    except asyncio.CancelledError:
        print("🛑 Shutting down...")
    finally:
        dispatcher.stop()


if __name__ == "__main__":
    try:
        asyncio.run(run_8stream_perception())
    except KeyboardInterrupt:
        pass
