import asyncio
import time
import os
import torch
import numpy as np
from perception.detector_trt import TRTYoloDetector
from perception.feature_extractor import TRTFeatureExtractor
from perception.cropper import ZeroCopyCropper
from perception.dispatcher import AsyncDispatcher
from media.mediamtx_client import MediaMTXClient
from dotenv import load_dotenv

load_dotenv()

async def run_multistream_benchmark(num_streams=10, num_frames_per_stream=500):
    print(f"🚀 Starting Multi-Stream Stress Test ({num_streams} streams, {num_frames_per_stream} frames/stream)...")
    
    # 初始化核心組件
    detector = TRTYoloDetector(engine_path="models/yolo/yolo26n_native.engine")
    dispatcher = AsyncDispatcher(detector=detector, max_batch=num_streams)
    dispatcher.start()
    
    # 建立多路 MediaMTXClient
    clients = []
    dummy_video = os.getenv("DUMMY_VIDEO_PATH", "assets/videos/demo.mp4")
    
    for i in range(num_streams):
        client = MediaMTXClient(dummy_video=dummy_video)
        if client.connect():
            clients.append(client)
            print(f"  - Stream {i+1} connected.")
        else:
            print(f"  - Stream {i+1} failed to connect.")

    print(f"✅ Total Active Streams: {len(clients)}")
    
    start_time = time.perf_counter()
    total_frames = 0
    
    async def stream_producer(client_id, client):
        nonlocal total_frames
        frames_done = 0
        while frames_done < num_frames_per_stream:
            ret, tensor = client.grab_tensor()
            if ret and tensor is not None:
                # 模擬分發到 Dispatcher
                # [H, W, 3] -> [3, 640, 640]
                frame_gpu = tensor.float() / 255.0
                frame_chw = frame_gpu.permute(2, 0, 1).unsqueeze(0)
                yolo_input = torch.nn.functional.interpolate(frame_chw, size=(640, 640)).squeeze(0)
                
                await dispatcher.put_frame(f"stream_{client_id}", yolo_input, time.time())
                frames_done += 1
                total_frames += 1
                
                if frames_done % 100 == 0:
                    print(f"    [Stream {client_id}] {frames_done}/{num_frames_per_stream} frames pushed.")
            else:
                await asyncio.sleep(0.001)

    # 啟動所有生產者
    producers = [stream_producer(i, clients[i]) for i in range(len(clients))]
    await asyncio.gather(*producers)
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    print("\n" + "═" * 60)
    print(f"📊 Multi-Stream Stress Test Results")
    print("-" * 60)
    print(f"Total Streams:      {len(clients)}")
    print(f"Total Frames:       {total_frames}")
    print(f"Total Duration:     {duration:.2f} s")
    print(f"Aggregate Throughput: {total_frames / duration:.2f} FPS")
    print(f"Per-Stream Avg:     {(total_frames / duration) / len(clients):.2f} FPS")
    print("═" * 60)
    
    dispatcher.stop()
    for client in clients:
        client.release()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", type=int, default=10)
    parser.add_argument("--frames", type=int, default=500)
    args = parser.parse_args()
    
    asyncio.run(run_multistream_benchmark(num_streams=args.streams, num_frames_per_stream=args.frames))
