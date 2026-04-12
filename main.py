import asyncio
import argparse
import os
from perception.detector import Detector
from perception.entropy import EntropyTrigger
from media.mediamtx_client import MediaMTXClient
from pipeline.orchestrator import PipelineOrchestrator
from dotenv import load_dotenv

load_dotenv()

from media.ffmpeg_utils import RTSPStreamer

async def run_perception():
    """快路徑：感知偵測循環 (帶視覺化輸出)"""
    use_local = os.getenv("USE_LOCAL_CAMERA", "false").lower() == "true"
    dummy_video = os.getenv("DUMMY_VIDEO_PATH")
    
    detector = Detector()
    trigger = EntropyTrigger(threshold=0.6)
    media = MediaMTXClient(use_local=use_local, dummy_video=dummy_video)
    
    # 增加二次推流器 (Visualizer)
    streamer = RTSPStreamer(rtsp_url="rtsp://localhost:8554/detected")
    
    # 建立串流連線 (增加重試機制)
    while not media.connect():
        print("⏳ Waiting for media source (RTSP/Camera/Dummy)... retrying in 2s")
        await asyncio.sleep(2)
    
    source_name = f"Dummy ({dummy_video})" if dummy_video else ("Local" if use_local else "RTSP")
    print(f"✅ Perception Pipeline connected! (Source: {source_name})")
    
    frame_id = 0
    try:
        while True:
            ret, frame = media.grab_frame()
            if not ret or frame is None:
                await asyncio.sleep(0.01)
                continue
            
            frame_id += 1
            
            # 1. 執行偵測
            results = detector.detect(frame)
            
            # 2. 繪製偵測結果 (視覺化輸出)
            if results and len(results) > 0:
                # 使用 YOLO 的內建繪製功能
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
            else:
                # 若無結果，直接推原圖以保持流暢
                streamer.push_frame(frame)
            
            await asyncio.sleep(0.01) 
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
        # 同時啟動雙軌 (目前優先推薦獨立服務啟動)
        print("💡 Running in full mode - orchestration focus.")
        orchestrator = PipelineOrchestrator()
        asyncio.run(orchestrator.run())

if __name__ == "__main__":
    main()
