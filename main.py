"""
Saccade CLI Entrypoint

此模組僅作為應用程式的 CLI 進入點（Entrypoint）。
其主要職責為：
1. 解析命令列參數（CLI Arguments）。
2. 初始化環境變數與全局設定。
3. 根據指定的模式（如 perception, orchestrator）實例化並啟動對應的核心邏輯。

注意：核心的系統調度、非同步事件循環與各層級之間的資料流動，皆由 `pipeline/orchestrator.py` 負責處理。本檔案不應包含任何業務邏輯或感知流程細節。
"""

import asyncio
import argparse
import os
import torch
import time
from perception.detector_trt import TRTYoloDetector
from perception.entropy import EntropyTrigger
from perception.cropper import ZeroCopyCropper
from perception.feature_extractor import TRTFeatureExtractor
from perception.tracker import SmartTracker
from perception.drift_handler import SemanticDriftHandler
from media.mediamtx_client import MediaMTXClient
from storage.redis_cache import RedisCache
from pipeline.orchestrator import PipelineOrchestrator
from dotenv import load_dotenv

from media.ffmpeg_utils import RTSPStreamer

load_dotenv()


async def run_perception() -> None:
    """快路徑：感知偵測與特徵提取循環 (Zero-Copy)"""
    use_local = os.getenv("USE_LOCAL_CAMERA", "false").lower() == "true"
    dummy_video = os.getenv("DUMMY_VIDEO_PATH")

    print("🚀 Initializing Perception Pipeline components...")
    detector = TRTYoloDetector()
    trigger = EntropyTrigger(threshold=0.8, cooldown=5.0)
    media = MediaMTXClient(use_local=use_local, dummy_video=dummy_video)
    redis_cache = RedisCache()  # 實時時空快取

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
            now = time.time()

            with torch.no_grad():
                # [1080, 1920, 3] -> [1, 3, 1080, 1920]
                input_tensor = tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0

                # 為了 YOLO 進行縮放 (不影響 SigLIP 的高解析裁切)
                yolo_input = torch.nn.functional.interpolate(
                    input_tensor, size=(640, 640), mode="bilinear", align_corners=False
                )

                # 1. 執行 YOLO 偵測與追蹤
                results = detector.detect(yolo_input, conf_threshold=0.25)

            # 確保有追蹤結果 (現在強制使用 TRTYoloDetector)
            boxes, scores, cls_ids, ids = results
            if ids is None or ids.numel() == 0:
                await asyncio.sleep(0.005)
                continue
            # TRT 目前不自動回傳 names，提供基礎對應或由外部配置
            names = {
                0: "person",
                1: "bicycle",
                2: "car",
                3: "motorcycle",
                5: "bus",
                7: "truck",
            }

            # --- [實時時空整合] ---
            # 將 YOLO 偵測到的物件即時更新至 Redis 時空快取
            for i, obj_id_tensor in enumerate(ids):
                obj_id = int(obj_id_tensor.item())
                # 修正：names 可能是 list 或 dict，使用安全存取
                cls_idx = int(cls_ids[i].item())
                if isinstance(names, list):
                    label = names[cls_idx] if cls_idx < len(names) else "unknown"
                else:
                    label = names.get(cls_idx, "unknown")

                box = boxes[i].tolist()
                # 這裡是異步非阻塞更新
                asyncio.create_task(
                    redis_cache.update_object_track(obj_id, label, box, now)
                )
            # ----------------------

            # 將 640x640 的 box 映射回 1080p 供裁切使用
            scale_x = tensor.shape[1] / 640.0
            scale_y = tensor.shape[0] / 640.0
            boxes_1080p = boxes.clone()
            boxes_1080p[:, [0, 2]] *= scale_x
            boxes_1080p[:, [1, 3]] *= scale_y

            # 2. 智能追蹤過濾 (Phase 3)
            ext_ids, ext_boxes = tracker.update_and_filter(ids, boxes_1080p)

            # 3. 非同步特徵提取 (Phase 1 & 2)
            features = tracker.async_extract_features(
                input_tensor, ext_boxes, cropper, extractor
            )

            if features is not None:
                # 等待 CUDA Stream 執行完畢以進行語義比對
                torch.cuda.current_stream().wait_stream(tracker.extraction_stream)

                # 4. 語義去重與漂移檢測 (Phase 4)
                novel_ids, novel_features = drift_handler.filter_novel_features(
                    ext_ids, features
                )

                if novel_ids.numel() > 0:
                    print(
                        f"✨ [Frame {frame_id}] Extracted novel semantics for Obj IDs: {novel_ids.tolist()}"
                    )

            # 觸發傳統事件
            labels = [names.get(int(c.item()), "unknown") for c in cls_ids]
            await trigger.process_frame(
                frame_id=frame_id,
                detections=labels,
                source_path="local_cam" if use_local else "rtsp",
            )

            # 視覺化輸出
            if frame_id % 3 == 0:
                _, frame_np = media.grab_frame()
                if frame_np is not None:
                    # 目前 Native 模式不提供自動繪製，直接推流原始影格或由開發者擴充
                    streamer.push_frame(frame_np)

            await asyncio.sleep(0.005)
    finally:
        media.release()
        streamer.stop()
        await redis_cache.disconnect()
        await trigger.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Saccade - Dual-Track Video Perception"
    )
    parser.add_argument(
        "--mode", choices=["perception", "orchestrator", "full"], default="full"
    )
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
