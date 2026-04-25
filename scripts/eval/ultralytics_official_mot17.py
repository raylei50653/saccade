from ultralytics import YOLO
import os
import cv2
import time
from pathlib import Path
import argparse
import configparser

def run_official_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/yolo/yolo11n.pt") # 官方權重
    parser.add_argument("--data-root", default="datasets/MOT17")
    parser.add_argument("--sequences", default="MOT17-05-SDP")
    parser.add_argument("--output", default="results/ultralytics_official")
    args = parser.parse_args()

    # 加載模型
    model = YOLO(args.model)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    seqs = args.sequences.split(",")
    for seq in seqs:
        seq_path = Path(args.data_root) / "train" / seq
        img_dir = seq_path / "img1"
        
        # 讀取序列資訊
        config = configparser.ConfigParser()
        config.read(seq_path / "seqinfo.ini")
        w_orig = config.getint("Sequence", "imWidth")
        h_orig = config.getint("Sequence", "imHeight")
        
        results_lines = []
        img_files = sorted(list(img_dir.glob("*.jpg")))
        
        print(f"🚀 Running Official Ultralytics Tracking on {seq}...")
        start_time = time.time()
        
        # 執行追蹤 (使用官方 BoT-SORT 設定)
        # 注意：官方版本會自動處理預處理與後處理
        results = model.track(
            source=str(img_dir),
            conf=0.25,
            iou=0.45,
            tracker="botsort.yaml",
            persist=True,
            stream=True,
            verbose=False,
            device="cuda:0",
            classes=[0] # 只追蹤行人
        )

        for i, res in enumerate(results):
            frame_id = i + 1
            if res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy().astype(int)
                scores = res.boxes.conf.cpu().numpy()
                
                for j in range(len(ids)):
                    x1, y1, x2, y2 = boxes[j]
                    w, h = x2 - x1, y2 - y1
                    # 格式: frame,id,x1,y1,w,h,conf,-1,-1,-1
                    results_lines.append(f"{frame_id},{ids[j]},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{scores[j]:.4f},-1,-1,-1")
            
            if frame_id % 100 == 0:
                print(f"  🎬 Frame {frame_id}/{len(img_files)}")

        # 儲存結果
        (output_root / f"{seq}.txt").write_text("\n".join(results_lines))
        print(f"✅ Finished {seq} in {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    run_official_eval()
