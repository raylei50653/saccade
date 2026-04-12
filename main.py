import asyncio
import argparse
import os
import torch
from perception.detector import Detector
from perception.entropy import EntropyTrigger
from media.mediamtx_client import MediaMTXClient
from pipeline.orchestrator import PipelineOrchestrator
from dotenv import load_dotenv

from media.ffmpeg_utils import RTSPStreamer

load_dotenv()

async def run_perception():
    """快路徑：感知偵測循環 (基於 Zero-Copy CUDA Tensor)"""
    use_local = os.getenv("USE_LOCAL_CAMERA", "false").lower() == "true"
    dummy_video = os.getenv("DUMMY_VIDEO_PATH")
    
    detector = Detector()
    trigger = EntropyTrigger(threshold=0.8, cooldown=5.0)
    media = MediaMTXClient(use_local=use_local, dummy_video=dummy_video)
    
    # 視覺化推流器
    streamer = RTSPStreamer(rtsp_url="rtsp://localhost:8554/detected")
    
    # 建立串流連線
    while not media.connect():
        print("⏳ Waiting for media source (RTSP/Camera/Dummy)... retrying in 2s")
        await asyncio.sleep(2)
    
    print(f"✅ Perception Pipeline connected via Zero-Copy GStreamer!")
    
    # 等待第一影格就緒
    print("⏳ Waiting for first frame...")
    for _ in range(50):
        ret, _ = media.grab_tensor()
        if ret:
            break
        await asyncio.sleep(0.1)

    frame_id = 0
    try:
        while True:
            # 優先使用 CUDA Tensor 進行 Zero-Copy 推理
            ret, tensor = media.grab_tensor()
            if not ret or tensor is None:
                await asyncio.sleep(0.01)
                continue
            
            frame_id += 1
            
            # 1. 執行偵測 (YOLO 直接接收 CUDA Tensor)
            # 將 [H, W, C] 轉為 [1, C, H, W] 並歸一化與縮放
            with torch.no_grad():
                # [1080, 1920, 3] -> [1, 3, 1080, 1920]
                input_tensor = tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
                # 縮放至 640x640
                input_tensor = torch.nn.functional.interpolate(
                    input_tensor, size=(640, 640), mode='bilinear', align_corners=False
                )
                # 使用 track 模式以維持物件 ID 一致性
                results = detector.model.track(input_tensor, verbose=False, persist=True)

            
            # 2. 視覺化 (使用 Numpy 格式推流)
            if frame_id % 3 == 0:
                # 獲取對應的 Numpy 影格進行繪製 (MediaMTXClient 內部已同步準備好)
                _, frame_np = media.grab_frame()
                if frame_np is not None:
                    # 這裡將結果套用到 Numpy 影格上
                    annotated_frame = results[0].plot()
                    streamer.push_frame(annotated_frame)
            
            # 3. 評估與觸發事件
            if len(results[0].boxes) > 0:
                labels = detector.get_actionable_labels(results)
                await trigger.process_frame(
                    frame_id=frame_id, 
                    detections=labels, 
                    source_path="local_cam" if use_local else "rtsp"
                )
            
            await asyncio.sleep(0.005) # 降低等待時間，極大化吞吐量
    finally:
        media.release()
        streamer.stop()
        await trigger.close()

def main():
    parser = argparse.ArgumentParser(description="Saccade - Dual-Track Video Perception")
    parser.add_argument("--mode", choices=["perception", "orchestrator", "full"], default="full")
    args = parser.parse_args()

    if args.mode == "perception":
        asyncio.run(run_perception())
    elif args.mode == "orchestrator":
        orchestrator = PipelineOrchestrator()
        asyncio.run(orchestrator.run())
    else:
        print("💡 Running in full mode - starting orchestrator.")
        orchestrator = PipelineOrchestrator()
        asyncio.run(orchestrator.run())

if __name__ == "__main__":
    main()
