import asyncio
import argparse
import os
import torch
from perception.detector import Detector
from perception.entropy import EntropyTrigger
from perception.cropper import ZeroCopyCropper
from perception.feature_extractor import TRTFeatureExtractor
from perception.tracker import SmartTracker
from perception.drift_handler import SemanticDriftHandler
from media.mediamtx_client import MediaMTXClient
from pipeline.orchestrator import PipelineOrchestrator
from dotenv import load_dotenv

from media.ffmpeg_utils import RTSPStreamer

load_dotenv()

async def run_perception() -> None:
    """快路徑：感知偵測與特徵提取循環 (Zero-Copy)"""
    use_local = os.getenv("USE_LOCAL_CAMERA", "false").lower() == "true"
    dummy_video = os.getenv("DUMMY_VIDEO_PATH")
    
    print("🚀 Initializing Perception Pipeline components...")
    detector = Detector()
    trigger = EntropyTrigger(threshold=0.8, cooldown=5.0)
    media = MediaMTXClient(use_local=use_local, dummy_video=dummy_video)
    
    # 初始化 Phase 1~4 的零拷貝語義特徵提取組件
    cropper = ZeroCopyCropper(output_size=(224, 224))
    extractor = TRTFeatureExtractor()
    tracker = SmartTracker(iou_threshold=0.7, velocity_angle_threshold=45.0)
    drift_handler = SemanticDriftHandler(similarity_threshold=0.95)
    
    # 視覺化推流器
    streamer = RTSPStreamer(rtsp_url="rtsp://localhost:8554/detected")
    
    # 建立串流連線
    while not media.connect():
        print("⏳ Waiting for media source (RTSP/Camera/Dummy)... retrying in 2s")
        await asyncio.sleep(2)
    
    print("✅ Perception Pipeline connected via Zero-Copy GStreamer!")
    
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
            ret, tensor = media.grab_tensor()
            if not ret or tensor is None:
                await asyncio.sleep(0.01)
                continue
            
            frame_id += 1
            
            with torch.no_grad():
                # [1080, 1920, 3] -> [1, 3, 1080, 1920]
                input_tensor = tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
                
                # 為了 YOLO 進行縮放 (不影響 SigLIP 的高解析裁切)
                yolo_input = torch.nn.functional.interpolate(
                    input_tensor, size=(640, 640), mode='bilinear', align_corners=False
                )
                
                # 1. 執行 YOLO 偵測與追蹤
                results = detector.model.track(yolo_input, verbose=False, persist=True)

            # 確保有追蹤結果
            if results and len(results[0].boxes) > 0 and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy
                ids = results[0].boxes.id
                
                # 將 640x640 的 box 映射回 1080p
                scale_x = tensor.shape[1] / 640.0
                scale_y = tensor.shape[0] / 640.0
                boxes_1080p = boxes.clone()
                boxes_1080p[:, [0, 2]] *= scale_x
                boxes_1080p[:, [1, 3]] *= scale_y

                # 2. 智能追蹤過濾 (Phase 3)
                ext_ids, ext_boxes = tracker.update_and_filter(ids, boxes_1080p)

                # 3. 非同步特徵提取 (Phase 1 & 2)
                features = tracker.async_extract_features(input_tensor, ext_boxes, cropper, extractor)

                if features is not None:
                    # 等待 CUDA Stream 執行完畢以進行語義比對
                    torch.cuda.current_stream().wait_stream(tracker.extraction_stream)
                    
                    # 4. 語義去重與漂移檢測 (Phase 4)
                    novel_ids, novel_features = drift_handler.filter_novel_features(ext_ids, features)
                    
                    if novel_ids.numel() > 0:
                        print(f"✨ [Frame {frame_id}] Extracted novel semantics for Obj IDs: {novel_ids.tolist()}")
                        # 未來可以在此處將 novel_features 寫入 Redis/ChromaDB
                
                # 觸發傳統事件
                labels = detector.get_actionable_labels(results)
                await trigger.process_frame(
                    frame_id=frame_id, 
                    detections=labels, 
                    source_path="local_cam" if use_local else "rtsp"
                )

            # 視覺化輸出
            if frame_id % 3 == 0:
                _, frame_np = media.grab_frame()
                if frame_np is not None and results:
                    annotated_frame = results[0].plot()
                    streamer.push_frame(annotated_frame)
            
            await asyncio.sleep(0.005)
    finally:
        media.release()
        streamer.stop()
        await trigger.close()

def main() -> None:
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
