# mypy: ignore-errors
import argparse
import configparser
import os
import sys
import time
import tempfile
import torch
import numpy as np
import cv2
from pathlib import Path

# 🚀 確保環境路徑
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
build_path = project_root / "build"
if build_path.exists(): sys.path.insert(0, str(build_path))

# MUST IMPORT THIS BEFORE torchvision TO AVOID LIBJPEG CONFLICT
from perception.detector_trt import TRTYoloDetector
from perception.cropper import ZeroCopyCropper
from perception.feature_extractor import TRTFeatureExtractor

from torchvision.ops import batched_nms, roi_align
try:
    from perception.embedding_dispatcher import AsyncEmbeddingDispatcher
except ImportError:
    AsyncEmbeddingDispatcher = None
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

class AdaptiveFramePool:
    def __init__(self, h, w, device='cuda'):
        print(f"🕯️ Allocating Adaptive VRAM Buffers for {w}x{h}...")
        self.frame_buffer = torch.zeros((3, h, w), device=device, dtype=torch.float32)
        self.canvas_960p = torch.zeros((3, 960, 960), device=device, dtype=torch.float32)
        self.tiles_batch4 = torch.zeros((4, 3, 640, 640), device=device, dtype=torch.float32)

def detect_adaptive(detector, pool, conf_threshold, h_orig, w_orig):
    # 如果解析度原本就小於 960，直接縮放處理
    if w_orig <= 960 and h_orig <= 960:
        img_input = torch.nn.functional.interpolate(pool.frame_buffer.unsqueeze(0), size=(640, 640))
        return detector.detect_batch(img_input, conf_threshold=conf_threshold), 640.0/w_orig, 640.0/h_orig, 0, 0, False
    
    # 大解析度使用 Tiled 模式
    r = 960.0 / max(h_orig, w_orig)
    h_new, w_new = int(h_orig * r), int(w_orig * r)
    img_resized = torch.nn.functional.interpolate(pool.frame_buffer.unsqueeze(0), size=(h_new, w_new)).squeeze(0)
    
    pool.canvas_960p.fill_(114.0/255.0)
    y_off = (960 - h_new) // 2
    x_off = (960 - w_new) // 2
    pool.canvas_960p[:, y_off:y_off+h_new, x_off:x_off+w_new].copy_(img_resized)
    
    pool.tiles_batch4[0].copy_(pool.canvas_960p[:, 0:640, 0:640])
    pool.tiles_batch4[1].copy_(pool.canvas_960p[:, 0:640, 320:960])
    pool.tiles_batch4[2].copy_(pool.canvas_960p[:, 320:960, 0:640])
    pool.tiles_batch4[3].copy_(pool.canvas_960p[:, 320:960, 320:960])
    return detector.detect_batch(pool.tiles_batch4, conf_threshold=conf_threshold), r, r, x_off, y_off, True

class DALIStreamerStream:
    def __init__(self, img_dir: Path):
        self.img_files = sorted(list(img_dir.glob("*.jpg")))
        self._setup()

    def _setup(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            for img in self.img_files: f.write(f"{img.absolute()} 0\n")
            self.file_list_path = f.name
        class JpgPipe(Pipeline):
            def __init__(self, flist):
                super().__init__(1, 4, 0, prefetch_queue_depth=2)
                self.input = fn.readers.file(file_list=flist, name="reader")
            def define_graph(self):
                jpegs, _ = self.input
                return fn.decoders.image(jpegs, device="cpu", output_type=types.RGB).gpu()
        self.pipe = JpgPipe(self.file_list_path)
        self.pipe.build()
        self.iterator = DALIGenericIterator([self.pipe], ["data"], auto_reset=True)

    def __iter__(self): return self
    def __next__(self):
        try: return next(self.iterator)[0]["data"][0]
        except StopIteration:
            if os.path.exists(self.file_list_path): os.remove(self.file_list_path)
            raise StopIteration

from ultralytics.trackers.bot_sort import BOTSORT

def get_args_obj():
    class Args:
        tracker_type = 'botsort'
        track_high_thresh = 0.5
        track_low_thresh = 0.1
        new_track_thresh = 0.6
        track_buffer = 30
        match_thresh = 0.8
        gmc_method = 'sparseOptFlow'
        fuse_score = True
        proximity_thresh = 0.5
        appearance_thresh = 0.25
        with_reid = False
    return Args()

def run_eval(engine, output, data_root, split, sequences, max_frames, conf_threshold, no_reid, **kwargs):
    output_root = Path(output); output_root.mkdir(parents=True, exist_ok=True)
    detector = TRTYoloDetector(engine_path=engine)
    reid_enabled = not no_reid and AsyncEmbeddingDispatcher is not None
    if not no_reid and AsyncEmbeddingDispatcher is None:
        print("⚠️ ReID dispatcher is unavailable; falling back to GPUByteTracker only.")
    extractor = TRTFeatureExtractor() if reid_enabled else None
    cropper = ZeroCopyCropper() if reid_enabled else None
    dispatcher = AsyncEmbeddingDispatcher(detector, extractor, cropper) if reid_enabled else None
    if dispatcher: dispatcher.start()
    
    seqs = sequences.split(",") if sequences else [d.name for d in (Path(data_root)/split).iterdir() if d.is_dir()]

    for seq in seqs:
        # 重置 BoT-SORT 追蹤器
        py_tracker = BOTSORT(get_args_obj())
        seq_path = Path(data_root) / split / seq
        if not (seq_path / "seqinfo.ini").exists(): continue
        config = configparser.ConfigParser(); config.read(seq_path / "seqinfo.ini")
        w_orig, h_orig = config.getint("Sequence", "imWidth"), config.getint("Sequence", "imHeight")
        frame_end = min(max_frames or 1e9, config.getint("Sequence", "seqLength"))
        
        # 設置 GPU 追蹤器參數 (關鍵修復：確保追蹤器門檻跟隨參數)
        high_threshold = max(0.5, conf_threshold)
        detector.tracker.set_params(
            track_thresh=conf_threshold,
            high_thresh=high_threshold,
            match_thresh=0.8,
            track_buffer=30,
        )
        
        pool = AdaptiveFramePool(h_orig, w_orig)
        streamer = DALIStreamerStream(seq_path / "img1")
        results_lines, frame_latencies = [], []
        start_time = time.time()
        warmup_frames = 50

        for frame_id, frame_gpu in enumerate(streamer, 1):
            if frame_id > frame_end: break
            t_frame_start = time.perf_counter()
            
            # 1. 注入影格
            pool.frame_buffer.copy_(frame_gpu.permute(2, 0, 1).float() / 255.0)

            # 2. 偵測
            yolo_event = torch.cuda.Event(enable_timing=True)
            batch_dets, rx, ry, x_off, y_off, is_tiled = detect_adaptive(detector, pool, conf_threshold, h_orig, w_orig)
            yolo_event.record()
            
            # 3. 座標還原
            all_b, all_s, all_c = [], [], []
            if is_tiled:
                offsets = [(0, 0), (320, 0), (0, 320), (320, 320)]
                for i, (boxes, scores, classes, _) in enumerate(batch_dets):
                    if boxes.numel() == 0: continue
                    sel_b = boxes.clone()
                    sel_b[:, [0, 2]] = (sel_b[:, [0, 2]] + offsets[i][0] - x_off) / rx
                    sel_b[:, [1, 3]] = (sel_b[:, [1, 3]] + offsets[i][1] - y_off) / ry
                    all_b.append(sel_b); all_s.append(scores); all_c.append(classes)
            else:
                boxes, scores, classes, _ = batch_dets[0]
                if boxes.numel() > 0:
                    sel_b = boxes.clone()
                    sel_b[:, [0, 2]] /= rx
                    sel_b[:, [1, 3]] /= ry
                    all_b.append(sel_b); all_s.append(scores); all_c.append(classes)
            
            fused_boxes, fused_scores, fused_classes = detector._empty_result()[:3]
            if all_b:
                cb, cs, cc = torch.cat(all_b), torch.cat(all_s), torch.cat(all_c)
                k = batched_nms(cb, cs, cc, 0.5)
                fused_boxes, fused_scores, fused_classes = cb[k], cs[k], cc[k]

            # 4. 非同步提交
            if dispatcher:
                reid_int = kwargs.get('reid_interval', 4)
                should_reid = (frame_id % reid_int == 0)
                dispatcher.submit(pool.frame_buffer if should_reid else None, 
                                  fused_boxes if should_reid else torch.empty((0,4), device='cuda'), 
                                  (frame_id, fused_boxes, fused_scores, fused_classes, frame_id), 
                                  yolo_event)
            
            # 5. 獲取結果
            if dispatcher:
                p_res = dispatcher.get_result_non_blocking()
                if p_res:
                    t_b, t_s, t_c, t_i, pf = p_res
                    if t_b.numel() > 0:
                        hb, hs, hc, hi = t_b.cpu().numpy(), t_s.cpu().numpy(), t_c.cpu().numpy(), t_i.cpu().numpy()
                        for i in range(hb.shape[0]):
                            if int(hc[i]) != 0: continue
                            tid, (x1, y1, x2, y2) = int(hi[i]), hb[i]
                            results_lines.append(f"{pf},{tid},{max(0,x1):.2f},{max(0,y1):.2f},{min(w_orig,x2)-max(0,x1):.2f},{min(h_orig,y2)-max(0,y1):.2f},{hs[i]:.4f},-1,-1,-1")
            else:
                # 無分發器：直接同步呼叫 C++ GPUByteTracker
                tracks = detector.tracker.update(
                    fused_boxes,
                    fused_scores,
                    fused_classes.to(torch.int32)
                )
                for t in tracks:
                    if int(t.class_id) != 0: continue
                    tid = t.obj_id
                    x1, y1, x2, y2 = t.x1, t.y1, t.x2, t.y2
                    results_lines.append(f"{frame_id},{tid},{max(0,x1):.2f},{max(0,y1):.2f},{min(w_orig,x2)-max(0,x1):.2f},{min(h_orig,y2)-max(0,y1):.2f},{t.score:.4f},-1,-1,-1")

            if frame_id > warmup_frames:
                frame_latencies.append((time.perf_counter() - t_frame_start) * 1000)
            if frame_id % 100 == 0: print(f"🎬 {seq} [{frame_id}/{frame_end}]")

        # 報告指標
        if frame_latencies:
            lats = np.array(frame_latencies)
            print(f"\n📊 Production Latency Report for {seq}:")
            print(f"  - FPS:  {1000/np.mean(lats):.2f}")

        Path(output_root / f"{seq}.txt").write_text("\n".join(results_lines))
        print(f"✅ Finished {seq} (Total Time: {time.time()-start_time:.2f}s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="models/yolo/yolo26m_batch4.engine")
    parser.add_argument("--output", default="results/MOT17_eval")
    parser.add_argument("--data-root", default="datasets/MOT17")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sequences", default="")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--conf-threshold", type=float, default=0.7)
    parser.add_argument("--no-reid", action="store_true")
    parser.add_argument("--reid-interval", type=int, default=4)
    args = parser.parse_args()
    run_eval(**vars(args))
