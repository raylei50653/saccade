import argparse
import configparser
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np
import cv2
import tempfile
from pathlib import Path

# 🚀 確保環境路徑
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
build_path = project_root / "build"
if build_path.exists():
    sys.path.insert(0, str(build_path))

# MUST IMPORT THIS BEFORE torchvision TO AVOID LIBJPEG CONFLICT
from perception.detector_trt import TRTYoloDetector  # noqa: E402
from perception.cropper import ZeroCopyCropper  # noqa: E402
from perception.feature_extractor import TRTFeatureExtractor  # noqa: E402

from torchvision.ops import batched_nms, nms
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from perception.roi_selector import ROISelector  # noqa: E402
from perception.feature_bank import FeatureBank  # noqa: E402
from perception.embedding_dispatcher import AsyncEmbeddingDispatcher  # noqa: E402

class DALIStreamer:
    def __init__(self, img_dir: Path, batch_size: int = 1):
        self.img_dir = img_dir
        self.img_files = sorted(list(img_dir.glob("*.jpg")))
        self.batch_size = batch_size
        self.iterator = None
        self.file_list_path = None
        self._setup()

    def _setup(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            for img in self.img_files:
                f.write(f"{img.absolute()} 0\n")
            self.file_list_path = f.name
        
        class JpgPipe(Pipeline):
            def __init__(self, flist, bs):
                super().__init__(bs, 4, 0)
                self.input = fn.readers.file(file_list=flist, name="reader")
            def define_graph(self):
                jpegs, labels = self.input
                images = fn.decoders.image(jpegs, device="cpu", output_type=types.RGB)
                return images.gpu()

        pipe = JpgPipe(self.file_list_path, self.batch_size)
        pipe.build()
        self.iterator = DALIGenericIterator([pipe], ["data"], auto_reset=False)

    def __iter__(self): return self
    def __next__(self):
        try: return next(self.iterator)[0]["data"]
        except StopIteration:
            if self.file_list_path and os.path.exists(self.file_list_path):
                os.remove(self.file_list_path)
            raise StopIteration

def load_seq_info(seq_path: Path):
    config = configparser.ConfigParser()
    config.read(seq_path / "seqinfo.ini")
    return {
        "name": config.get("Sequence", "name"),
        "imDir": config.get("Sequence", "imDir"),
        "seqLength": config.getint("Sequence", "seqLength"),
        "imWidth": config.getint("Sequence", "imWidth"),
        "imHeight": config.getint("Sequence", "imHeight"),
    }

def parse_sequences(data_root: Path, split: str, sequences_arg: str):
    if sequences_arg:
        return sequences_arg.split(",")
    split_path = data_root / split
    return [d.name for d in split_path.iterdir() if d.is_dir()]

def sanitize_boxes(boxes, scores, classes, width, height, min_height=0.0, min_aspect=0.0):
    if boxes.numel() == 0: return boxes, scores, classes
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height - 1)
    bw, bh = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    valid = (bw > 2.0) & (bh > 4.0)
    if min_height > 0: valid &= (bh >= min_height)
    if min_aspect > 0: valid &= ((bh / (bw + 1e-6)) >= min_aspect)
    return boxes[valid], scores[valid], classes[valid]

def detect_tiled_generic(detector, img_orig_np, conf_threshold, target_res=960):
    h_orig, w_orig = img_orig_np.shape[:2]
    tile_size = detector.input_shape[2]
    r = float(target_res) / max(h_orig, w_orig)
    nw, nh = int(round(w_orig * r)), int(round(h_orig * r))
    pw, ph = (target_res - nw) / 2, (target_res - nh) / 2
    img_resized = cv2.resize(img_orig_np, (nw, nh))
    canvas = np.full((target_res, target_res, 3), 114, dtype=np.uint8)
    canvas[int(ph):int(ph)+nh, int(pw):int(pw)+nw] = img_resized
    canvas_t = torch.from_numpy(canvas).to("cuda").permute(2, 0, 1).float().div(255.0)
    stride = target_res - tile_size
    offsets = [(x, y) for y in [0, stride] for x in [0, stride]]
    tiles = torch.stack([canvas_t[:, y:y+tile_size, x:x+tile_size] for x, y in offsets])
    batch_res = detector.infer_detections_batch(tiles, conf_threshold=conf_threshold)
    all_b, all_s, all_c = [], [], []
    for i, (boxes, scores, classes) in enumerate(batch_res):
        if boxes.numel() == 0: continue
        off_x, off_y = offsets[i]
        cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0
        dist_x = torch.min(cx, tile_size - cx) / (tile_size / 2.0)
        dist_y = torch.min(cy, tile_size - cy) / (tile_size / 2.0)
        weight = (dist_x * dist_y).pow(2.0).clamp(0.2, 1.0)
        adj_s = scores * weight
        sel_b = boxes.clone()
        sel_b[:, [0, 2]] = (sel_b[:, [0, 2]] + off_x - pw) / r
        sel_b[:, [1, 3]] = (sel_b[:, [1, 3]] + off_y - ph) / r
        all_b.append(sel_b); all_s.append(adj_s); all_c.append(classes)
    if not all_b: return detector._empty_result()[:3]
    cat_b, cat_s, cat_c = torch.cat(all_b), torch.cat(all_s), torch.cat(all_c)
    keep = batched_nms(cat_b, cat_s, cat_c, iou_threshold=0.5)
    return cat_b[keep], cat_s[keep], cat_c[keep]

def get_seq_params(seq_name, base_conf, base_tiled):
    conf, min_h, min_asp, tiled, use_dynamic_roi = base_conf, 0.0, 0.0, base_tiled, False
    track_params = {"track_thresh": 0.4, "high_thresh": 0.5, "match_thresh": 0.7, "track_buffer": 30, "std_pos": 0.05, "std_vel": 0.00625}
    if "MOT20" in seq_name:
        # 🚀 MOT20 極限召回模式
        conf = 0.05 # 極低門檻
        tiled = True 
        track_params.update({
            "track_thresh": 0.1, # 追蹤門檻也同步下調
            "match_thresh": 0.8,
            "track_buffer": 60,  # 增加生存時間
            "std_pos": 0.01,    # 極致穩定小人
        })
    return conf, min_h, min_asp, tiled, track_params, use_dynamic_roi

def run_eval(engine, output, data_root, split, sequences, max_frames, conf_threshold, no_reid, **kwargs):
    output_root = Path(output); output_root.mkdir(parents=True, exist_ok=True)
    # 🚀 MOT20 結構: datasets/MOT20/MOT20/train/...
    data_root_path = Path(data_root) / "MOT20" 
    detector = TRTYoloDetector(engine_path=engine)
    extractor = None if no_reid else TRTFeatureExtractor()
    cropper = None if no_reid else ZeroCopyCropper()
    roi_selector = ROISelector()
    feature_bank = FeatureBank()
    sequences_list = parse_sequences(data_root_path, split, sequences)

    for seq in sequences_list:
        detector.reset_tracker(); feature_bank.reset()
        seq_path = data_root_path / split / seq
        # 💡 若 seqinfo 不存在，嘗試自動修正路徑
        if not (seq_path / "seqinfo.ini").exists():
            print(f"⚠️ Warning: {seq_path / 'seqinfo.ini'} not found, searching...")
            # 可能在更深一層
        
        info = load_seq_info(seq_path)
        w_orig, h_orig = info["imWidth"], info["imHeight"]
        frame_end = info["seqLength"] if max_frames is None else min(max_frames, info["seqLength"])
        print(f"🎬 Processing {seq}: {frame_end} frames")
        result_file = output_root / f"{seq}.txt"
        
        conf, min_h, min_asp, tiled_mode, t_p, dyn_roi = get_seq_params(seq, conf_threshold, kwargs.get('tiled', False))
        if hasattr(detector.tracker, "set_params"):
            detector.tracker.set_params(t_p["track_thresh"], t_p["high_thresh"], t_p["match_thresh"], t_p["track_buffer"], t_p["std_pos"], t_p["std_vel"])
        
        streamer = DALIStreamer(seq_path / "img1")
        dispatcher = AsyncEmbeddingDispatcher(extractor, cropper) if extractor else None
        if dispatcher: dispatcher.start()
        
        last_fused_boxes, results_lines, first_seen = None, [], {}
        
        for frame_id, frame_gpu_batch in enumerate(streamer, 1):
            if frame_id > frame_end: break
            img_hwc_gpu = frame_gpu_batch[0].contiguous()
            img_orig_gpu = img_hwc_gpu.permute(2, 0, 1).float() / 255.0

            # 1. Detection (Frame N)
            yolo_event = torch.cuda.Event()
            if tiled_mode:
                img_np = (img_orig_gpu.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                boxes, scores, classes = detect_tiled_generic(detector, img_np, conf, 960)
            elif dyn_roi:
                th, tw = detector.input_shape[2], detector.input_shape[3]
                img_g = torch.nn.functional.interpolate(img_orig_gpu.unsqueeze(0), size=(th, tw)).squeeze(0).contiguous()
                tensors = [img_g]
                roi = roi_selector.select_best_roi(last_fused_boxes, roi_size=th) if last_fused_boxes is not None else None
                if roi: tensors.append(img_orig_gpu[:, roi[1]:roi[3], roi[0]:roi[2]].contiguous())
                batch_res = detector.infer_detections_batch(torch.stack(tensors), conf_threshold=conf)
                all_b, all_s, all_c = [], [], []
                gb, gs, gc = batch_res[0]
                if gb.numel() > 0:
                    gb[:, [0, 2]] /= (tw/w_orig); gb[:, [1, 3]] /= (th/h_orig)
                    all_b.append(gb); all_s.append(gs); all_c.append(gc)
                if len(batch_res) > 1 and roi:
                    rb, rs, rc = batch_res[1]
                    if rb.numel() > 0:
                        rcx, rcy = (rb[:, 0] + rb[:, 2])/2, (rb[:, 1] + rb[:, 3])/2
                        rw = (torch.min(rcx, tw-rcx)/(tw/2) * torch.min(rcy, th-rcy)/(th/2)).pow(2.0).clamp(0.1, 1.0)
                        rs = rs * rw * 1.05
                        rb[:, [0, 2]] += roi[0]; rb[:, [1, 3]] += roi[1]
                        all_b.append(rb); all_s.append(rs); all_c.append(rc)
                if all_b:
                    cb, cs, cc = torch.cat(all_b), torch.cat(all_s), torch.cat(all_c)
                    k = batched_nms(cb, cs, cc, 0.5)
                    boxes, scores, classes = cb[k], cs[k], cc[k]
                else: boxes, scores, classes = detector._empty_result()[:3]
            else:
                th, tw = detector.input_shape[2], detector.input_shape[3]
                img_g = torch.nn.functional.interpolate(img_orig_gpu.unsqueeze(0), size=(th, tw))
                boxes, scores, classes = detector.infer_detections_batch(img_g, conf)[0]
                boxes[:, [0, 2]] /= (tw/w_orig); boxes[:, [1, 3]] /= (th/h_orig)
            
            yolo_event.record()

            boxes, scores, classes = sanitize_boxes(boxes, scores, classes, w_orig, h_orig, min_h, min_asp)
            person_mask = classes == 0
            boxes, scores, classes = boxes[person_mask], scores[person_mask], classes[person_mask]

            # 2. Pipeline (N -> N-1)
            if dispatcher: dispatcher.submit(img_orig_gpu, boxes, (frame_id, boxes, scores, classes, img_hwc_gpu), yolo_event)
            p_res = dispatcher.get_result_non_blocking() if dispatcher else (None, boxes, (frame_id, boxes, scores, classes, img_hwc_gpu))
            
            if p_res:
                emb, pb, pm = p_res; pf, pb, ps, pc, pih = pm
                t_b, t_s, t_c, t_i = detector.track_detections(pb, ps, pc, emb, image=pih)
                if t_b.numel() > 0:
                    hb, hs, hc, hi = t_b.cpu().numpy(), t_s.cpu().numpy(), t_c.cpu().numpy(), t_i.cpu().numpy()
                    for i in range(hb.shape[0]):
                        if int(hc[i]) != 0: continue
                        tid = int(hi[i])
                        if tid not in first_seen: first_seen[tid] = pf
                        if (pf - first_seen[tid]) < 3 and hs[i] < 0.65: continue
                        x1, y1, x2, y2 = hb[i]
                        results_lines.append(f"{pf},{tid},{max(0,x1):.2f},{max(0,y1):.2f},{min(w_orig,x2)-max(0,x1):.2f},{min(h_orig,y2)-max(0,y1):.2f},{hs[i]:.4f},-1,-1,-1")
                last_fused_boxes = pb
            if frame_id % 100 == 0: print(f"  Frame {frame_id}/{frame_end}")
        
        if dispatcher: dispatcher.stop()
        Path(result_file).write_text("\n".join(results_lines))
        print(f"✅ Saved {seq}.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="models/yolo/yolo26m.engine")
    parser.add_argument("--output", default="results/MOT17_ULTIMATE")
    parser.add_argument("--data-root", default="datasets/MOT20")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sequences", default="")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    parser.add_argument("--no-reid", action="store_true")
    parser.add_argument("--tiled", action="store_true")
    args = parser.parse_args()
    run_eval(**vars(args))
