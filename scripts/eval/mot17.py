# mypy: ignore-errors
import argparse
import configparser
import os
import sys
import time
import tempfile
from collections import OrderedDict
import torch
import torch.nn.functional as F
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
try:
    from saccade_tracking_ext import (
        merge_cross_tile_duplicates as cpp_merge_cross_tile_duplicates,
        merge_cross_tile_duplicates_cuda as cpp_merge_cross_tile_duplicates_cuda,
    )
except ImportError:
    cpp_merge_cross_tile_duplicates = None
    cpp_merge_cross_tile_duplicates_cuda = None

from torchvision.ops import batched_nms, nms
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator


class AdaptiveFramePool:
    def __init__(self, h, w, device='cuda'):
        print(f"🕯️ Allocating VRAM Buffers for adaptive 960 tiled eval ({w}x{h})...")
        self.frame_buffer = torch.zeros((3, h, w), device=device, dtype=torch.float32)
        self.canvas_640p = torch.zeros((3, 640, 640), device=device, dtype=torch.float32)
        self.canvas_960p = torch.zeros((3, 960, 960), device=device, dtype=torch.float32)
        self.tiles_batch4 = torch.zeros((4, 3, 640, 640), device=device, dtype=torch.float32)
        # Pre-allocated tile x/y offsets — avoids per-frame GPU tensor creation.
        self.tile_dx = torch.tensor([0.0, 320.0, 0.0, 320.0], device=device, dtype=torch.float32).view(4, 1, 1)
        self.tile_dy = torch.tensor([0.0, 0.0, 320.0, 320.0], device=device, dtype=torch.float32).view(4, 1, 1)


_duplicate_merge_cuda_workspace: dict[tuple[int, torch.dtype, torch.dtype], dict[str, torch.Tensor | int]] = {}


def _get_duplicate_merge_cuda_workspace(
    num_boxes: int,
    box_device: torch.device,
    box_dtype: torch.dtype,
    score_dtype: torch.dtype,
) -> dict[str, torch.Tensor | int]:
    device_index = box_device.index if box_device.index is not None else torch.cuda.current_device()
    key = (device_index, box_dtype, score_dtype)
    workspace = _duplicate_merge_cuda_workspace.get(key)
    if workspace is not None and int(workspace["capacity"]) >= num_boxes:
        return workspace

    capacity = max(num_boxes, 1)
    workspace = {
        "capacity": capacity,
        "anchor_indices": torch.empty((capacity,), device=box_device, dtype=torch.int32),
        "box_sums": torch.empty((capacity, 4), device=box_device, dtype=torch.float32),
        "score_sums": torch.empty((capacity,), device=box_device, dtype=torch.float32),
        "score_bits_max": torch.empty((capacity,), device=box_device, dtype=torch.int32),
        "cluster_counts": torch.empty((capacity,), device=box_device, dtype=torch.int32),
        "out_boxes": torch.empty((capacity, 4), device=box_device, dtype=box_dtype),
        "out_scores": torch.empty((capacity,), device=box_device, dtype=score_dtype),
        "out_classes": torch.empty((capacity,), device=box_device, dtype=torch.int32),
        "out_count": torch.zeros((), device=box_device, dtype=torch.int32),
    }
    _duplicate_merge_cuda_workspace[key] = workspace
    return workspace


def parse_preprocess(value):
    modes = [mode.strip().lower() for mode in value.split(",") if mode.strip()]
    if not modes or "none" in modes:
        return []
    allowed = {"gamma", "contrast", "letterbox"}
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise ValueError(f"Unsupported preprocess mode(s): {', '.join(unknown)}")
    return modes


def apply_frame_preprocess(frame, modes, gamma, gamma_luma_threshold, contrast):
    if not modes:
        return
    for mode in modes:
        if mode == "gamma":
            if gamma_luma_threshold <= 0.0 or float(frame.mean()) < gamma_luma_threshold:
                frame.copy_(frame.clamp(0.0, 1.0).pow(gamma))
        elif mode == "contrast":
            mean = frame.mean()
            frame.copy_(((frame - mean) * contrast + mean).clamp(0.0, 1.0))


class SemanticRelinker:
    def __init__(
        self,
        sim_threshold=0.985,
        ttl=45,
        ema_beta=0.83,
        spatial_gate=0.11,
        min_lost_frames=2,
        min_iou=0.0,
        mahalanobis_threshold=6.6,
        debug=False,
    ):
        self.sim_threshold = sim_threshold
        self.ttl = ttl
        self.ema_beta = ema_beta
        self.spatial_gate = spatial_gate
        self.min_lost_frames = min_lost_frames
        self.min_iou = min_iou
        self.mahalanobis_threshold = mahalanobis_threshold
        self.debug = debug
        self.alias = {}
        self.features = {}
        self.last_seen = {}
        self.last_boxes = {}
        self.motion = {}
        self.stats = {
            "attempts": 0,
            "accepted": 0,
            "reject_age": 0,
            "reject_assigned": 0,
            "reject_spatial": 0,
            "reject_mahalanobis": 0,
            "reject_similarity": 0,
            "new_ids": 0,
        }
        self.accept_sims = []
        self.accept_ious = []
        self.accept_center_dists = []
        self.accept_mahas = []

    def _normalize(self, embedding):
        return F.normalize(embedding.float(), dim=0)

    def _spatial_metrics(self, box, old_box, w, h):
        cx = (box[0] + box[2]) * 0.5
        cy = (box[1] + box[3]) * 0.5
        ocx = (old_box[0] + old_box[2]) * 0.5
        ocy = (old_box[1] + old_box[3]) * 0.5
        dist = ((cx - ocx) ** 2 + (cy - ocy) ** 2) ** 0.5
        center_norm = dist / max(w, h)

        ix1, iy1 = max(box[0], old_box[0]), max(box[1], old_box[1])
        ix2, iy2 = min(box[2], old_box[2]), min(box[3], old_box[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        old_area = max(0.0, old_box[2] - old_box[0]) * max(0.0, old_box[3] - old_box[1])
        iou = inter / (area + old_area - inter + 1e-6)
        return center_norm, iou

    def update_motion_snapshots(self, snapshots):
        for snap in snapshots:
            canonical = self.alias.get(snap.obj_id, snap.obj_id)
            self.motion[canonical] = snap

    def _measurement(self, box):
        w = max(1e-6, box[2] - box[0])
        h = max(1e-6, box[3] - box[1])
        return np.array(
            [(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5, w / h, h],
            dtype=np.float32,
        )

    def _mahalanobis(self, box, snapshot):
        state = np.asarray(snapshot.state[:4], dtype=np.float32)
        cov = np.asarray(snapshot.covariance, dtype=np.float32).reshape(8, 8)[:4, :4]
        h = max(float(state[3]), 1e-6)
        pos_std = h / 20.0
        r = np.diag([pos_std**2, pos_std**2, 1e-2, pos_std**2]).astype(np.float32)
        s = cov + r
        residual = self._measurement(box) - state
        try:
            solved = np.linalg.solve(s, residual)
        except np.linalg.LinAlgError:
            solved = np.linalg.pinv(s) @ residual
        return float(residual @ solved)

    def resolve(self, raw_id, embedding, box, score, frame_id, w, h, assigned):
        if embedding is None:
            canonical = self.alias.get(raw_id, raw_id)
            self.alias.setdefault(raw_id, canonical)
            return canonical

        emb = self._normalize(embedding)
        if raw_id not in self.alias:
            self.stats["attempts"] += 1
            best_id, best_sim = None, self.sim_threshold
            best_iou, best_center, best_maha = 0.0, 0.0, 0.0
            for cid, old_emb in self.features.items():
                age = frame_id - self.last_seen.get(cid, -(10**9))
                if cid in assigned:
                    self.stats["reject_assigned"] += 1
                    continue
                if age < self.min_lost_frames or age > self.ttl:
                    self.stats["reject_age"] += 1
                    continue
                center_norm, iou = self._spatial_metrics(box, self.last_boxes[cid], w, h)
                if center_norm > self.spatial_gate or iou < self.min_iou:
                    self.stats["reject_spatial"] += 1
                    continue
                maha = 0.0
                if self.mahalanobis_threshold > 0.0:
                    snapshot = self.motion.get(cid)
                    if snapshot is None:
                        self.stats["reject_mahalanobis"] += 1
                        continue
                    maha = self._mahalanobis(box, snapshot)
                    if maha > self.mahalanobis_threshold:
                        self.stats["reject_mahalanobis"] += 1
                        continue
                sim = torch.dot(emb, old_emb).item()
                if sim > best_sim:
                    best_id, best_sim = cid, sim
                    best_iou, best_center, best_maha = iou, center_norm, maha
                else:
                    self.stats["reject_similarity"] += 1
            if best_id is not None:
                self.stats["accepted"] += 1
                self.accept_sims.append(best_sim)
                self.accept_ious.append(best_iou)
                self.accept_center_dists.append(best_center)
                self.accept_mahas.append(best_maha)
                self.alias[raw_id] = best_id
            else:
                self.stats["new_ids"] += 1
                self.alias[raw_id] = raw_id

        canonical = self.alias[raw_id]
        old = self.features.get(canonical)
        updated = emb if old is None else F.normalize(self.ema_beta * old + (1.0 - self.ema_beta) * emb, dim=0)
        self.features[canonical] = updated.detach()
        self.last_seen[canonical] = frame_id
        self.last_boxes[canonical] = box
        assigned.add(canonical)
        return canonical

    def report(self):
        print("🔁 Semantic Relink Report:")
        print(
            "  attempts={attempts} accepted={accepted} new_ids={new_ids} "
            "reject_age={reject_age} reject_assigned={reject_assigned} "
            "reject_spatial={reject_spatial} reject_mahalanobis={reject_mahalanobis} "
            "reject_similarity={reject_similarity}".format(**self.stats)
        )
        if self.accept_sims:
            print(
                f"  accepted mean_sim={np.mean(self.accept_sims):.3f} "
                f"mean_iou={np.mean(self.accept_ious):.3f} "
                f"mean_center_norm={np.mean(self.accept_center_dists):.3f} "
                f"mean_maha={np.mean(self.accept_mahas):.3f}"
            )


class GlobalTrackIdMapper:
    """Map per-sequence local track IDs into run-global unique numeric IDs."""

    def __init__(self) -> None:
        self._next_global_id = 1
        self._mapping: dict[tuple[str, int], int] = {}

    def map(self, sequence: str, local_track_id: int) -> int:
        key = (sequence, int(local_track_id))
        global_id = self._mapping.get(key)
        if global_id is None:
            global_id = self._next_global_id
            self._mapping[key] = global_id
            self._next_global_id += 1
        return global_id

    def dump_lines(self) -> list[str]:
        lines = []
        for (sequence, local_id), global_id in sorted(
            self._mapping.items(), key=lambda item: item[1]
        ):
            lines.append(
                f"{sequence}\tlocal_id={local_id}\tglobal_id={global_id}"
            )
        return lines


def detect_single_patch_640(detector, pool, h_orig, w_orig, preprocess_modes):
    if "letterbox" in preprocess_modes:
        r = 640.0 / max(h_orig, w_orig)
        h_new, w_new = int(h_orig * r), int(w_orig * r)
        img_resized = torch.nn.functional.interpolate(
            pool.frame_buffer.unsqueeze(0), size=(h_new, w_new)
        ).squeeze(0)
        pool.canvas_640p.fill_(114.0 / 255.0)
        y_off = (640 - h_new) // 2
        x_off = (640 - w_new) // 2
        pool.canvas_640p[:, y_off:y_off + h_new, x_off:x_off + w_new].copy_(img_resized)

        raw_dets = detector.detect_raw(pool.canvas_640p.unsqueeze(0))
        boxes = raw_dets[0, :, :4]
        scores = raw_dets[0, :, 4]
        classes = raw_dets[0, :, 5]

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - x_off) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - y_off) / r
        return boxes, scores, classes

    img_input = torch.nn.functional.interpolate(pool.frame_buffer.unsqueeze(0), size=(640, 640))
    raw_dets = detector.detect_raw(img_input)
    boxes = raw_dets[0, :, :4]
    scores = raw_dets[0, :, 4]
    classes = raw_dets[0, :, 5]

    boxes[:, [0, 2]] /= (640.0 / w_orig)
    boxes[:, [1, 3]] /= (640.0 / h_orig)
    return boxes, scores, classes


def detect_adaptive_960_tiled(detector, pool, h_orig, w_orig, preprocess_modes):
    if w_orig <= 960 and h_orig <= 960:
        boxes, scores, classes = detect_single_patch_640(
            detector, pool, h_orig, w_orig, preprocess_modes
        )
        return boxes, scores, classes, False

    r = 960.0 / max(h_orig, w_orig)
    h_new, w_new = int(h_orig * r), int(w_orig * r)
    img_resized = torch.nn.functional.interpolate(
        pool.frame_buffer.unsqueeze(0), size=(h_new, w_new)
    ).squeeze(0)

    pool.canvas_960p.fill_(114.0 / 255.0)
    y_off = (960 - h_new) // 2
    x_off = (960 - w_new) // 2
    pool.canvas_960p[:, y_off:y_off + h_new, x_off:x_off + w_new].copy_(img_resized)

    pool.tiles_batch4[0].copy_(pool.canvas_960p[:, 0:640, 0:640])
    pool.tiles_batch4[1].copy_(pool.canvas_960p[:, 0:640, 320:960])
    pool.tiles_batch4[2].copy_(pool.canvas_960p[:, 320:960, 0:640])
    pool.tiles_batch4[3].copy_(pool.canvas_960p[:, 320:960, 320:960])

    raw_dets = detector.detect_raw(pool.tiles_batch4)
    # One clone for all tiles, two broadcast ops using pre-allocated offsets
    # instead of 4 individual clones + 8 per-tile scalar ops + 3 cat calls.
    all_boxes = raw_dets[:, :, :4].clone()  # [4, 300, 4]
    all_boxes[:, :, [0, 2]] = (all_boxes[:, :, [0, 2]] + pool.tile_dx - x_off) / r
    all_boxes[:, :, [1, 3]] = (all_boxes[:, :, [1, 3]] + pool.tile_dy - y_off) / r
    # Scores and classes are non-contiguous views; reshape() copies them automatically.
    return all_boxes.reshape(-1, 4), raw_dets[:, :, 4].reshape(-1), raw_dets[:, :, 5].reshape(-1), True


def merge_cross_tile_duplicates(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float = 0.45,
    center_threshold: float = 0.18,
    area_ratio_threshold: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0 or boxes.size(0) <= 1:
        return boxes, scores, classes

    order = torch.argsort(scores, descending=True)
    remaining = order.tolist()
    merged_boxes = []
    merged_scores = []
    merged_classes = []

    while remaining:
        anchor_idx = remaining[0]
        anchor_box = boxes[anchor_idx]
        anchor_class = classes[anchor_idx]

        candidate_indices = torch.tensor(remaining, device=boxes.device, dtype=torch.long)
        candidate_boxes = boxes[candidate_indices]
        candidate_scores = scores[candidate_indices]
        candidate_classes = classes[candidate_indices]

        same_class = candidate_classes == anchor_class
        ious = _box_iou_single(anchor_box, candidate_boxes)

        anchor_center = (anchor_box[:2] + anchor_box[2:]) * 0.5
        candidate_centers = (candidate_boxes[:, :2] + candidate_boxes[:, 2:]) * 0.5
        center_dist = torch.linalg.norm(candidate_centers - anchor_center, dim=1)

        anchor_wh = (anchor_box[2:] - anchor_box[:2]).clamp(min=1e-6)
        candidate_wh = (candidate_boxes[:, 2:] - candidate_boxes[:, :2]).clamp(min=1e-6)
        min_wh = torch.minimum(candidate_wh, anchor_wh.unsqueeze(0))
        center_gate = torch.linalg.norm(min_wh, dim=1) * center_threshold

        anchor_area = float(anchor_wh[0] * anchor_wh[1])
        candidate_areas = candidate_wh[:, 0] * candidate_wh[:, 1]
        area_ratio = torch.minimum(
            candidate_areas / max(anchor_area, 1e-6),
            torch.tensor(anchor_area, device=boxes.device) / candidate_areas.clamp(min=1e-6),
        )

        duplicate_mask = same_class & (
            (ious >= iou_threshold)
            | ((center_dist <= center_gate) & (area_ratio >= area_ratio_threshold))
        )

        cluster_indices = candidate_indices[duplicate_mask]
        cluster_boxes = boxes[cluster_indices]
        cluster_scores = scores[cluster_indices]
        weights = cluster_scores / cluster_scores.sum().clamp(min=1e-6)
        fused_box = (cluster_boxes * weights.unsqueeze(1)).sum(dim=0)
        fused_score = cluster_scores.max()

        merged_boxes.append(fused_box)
        merged_scores.append(fused_score)
        merged_classes.append(anchor_class)

        cluster_set = set(int(idx) for idx in cluster_indices.tolist())
        remaining = [idx for idx in remaining if idx not in cluster_set]

    return (
        torch.stack(merged_boxes, dim=0),
        torch.stack(merged_scores, dim=0),
        torch.stack(merged_classes, dim=0),
    )


def merge_cross_tile_duplicates_fast(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float = 0.45,
    center_threshold: float = 0.18,
    area_ratio_threshold: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0 or boxes.size(0) <= 1:
        return merge_cross_tile_duplicates(
            boxes, scores, classes, iou_threshold, center_threshold, area_ratio_threshold
        )

    boxes = boxes.contiguous()
    scores = scores.contiguous()
    classes_i32 = classes.to(torch.int32).contiguous()

    if cpp_merge_cross_tile_duplicates_cuda is not None and boxes.is_cuda:
        workspace = _get_duplicate_merge_cuda_workspace(
            int(boxes.size(0)),
            boxes.device,
            boxes.dtype,
            scores.dtype,
        )
        cpp_merge_cross_tile_duplicates_cuda(
            boxes.data_ptr(),
            scores.data_ptr(),
            classes_i32.data_ptr(),
            int(boxes.size(0)),
            workspace["anchor_indices"].data_ptr(),
            workspace["box_sums"].data_ptr(),
            workspace["score_sums"].data_ptr(),
            workspace["score_bits_max"].data_ptr(),
            workspace["cluster_counts"].data_ptr(),
            workspace["out_boxes"].data_ptr(),
            workspace["out_scores"].data_ptr(),
            workspace["out_classes"].data_ptr(),
            workspace["out_count"].data_ptr(),
            iou_threshold,
            center_threshold,
            area_ratio_threshold,
            torch.cuda.current_stream().cuda_stream,
        )
        merged_count = int(workspace["out_count"].item())
        return (
            workspace["out_boxes"][:merged_count],
            workspace["out_scores"][:merged_count],
            workspace["out_classes"][:merged_count],
        )

    if cpp_merge_cross_tile_duplicates is not None:
        device = boxes.device
        boxes_np = boxes.detach().to("cpu").numpy()
        scores_np = scores.detach().to("cpu").numpy()
        classes_np = classes_i32.detach().to("cpu").numpy()
        merged_boxes_np, merged_scores_np, merged_classes_np = cpp_merge_cross_tile_duplicates(
            boxes_np,
            scores_np,
            classes_np,
            iou_threshold,
            center_threshold,
            area_ratio_threshold,
        )
        merged_boxes = torch.from_numpy(np.asarray(merged_boxes_np)).to(device=device, dtype=boxes.dtype)
        merged_scores = torch.from_numpy(np.asarray(merged_scores_np)).to(device=device, dtype=scores.dtype)
        merged_classes = torch.from_numpy(np.asarray(merged_classes_np)).to(device=device, dtype=classes_i32.dtype)
        return merged_boxes, merged_scores, merged_classes

    return merge_cross_tile_duplicates(
        boxes, scores, classes_i32, iou_threshold, center_threshold, area_ratio_threshold
    )


class DALIStreamerStream:
    def __init__(self, img_dir: Path):
        self.img_files = sorted(list(img_dir.glob("*.jpg")))
        self._setup()

    def _setup(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            for img in self.img_files:
                f.write(f"{img.absolute()} 0\n")
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

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)[0]["data"][0]
        except StopIteration:
            if os.path.exists(self.file_list_path):
                os.remove(self.file_list_path)
            raise


def run_eval(engine, output, data_root, split, sequences, max_frames, conf_threshold, no_reid, **kwargs):
    output_root = Path(output)
    output_root.mkdir(parents=True, exist_ok=True)
    fps_summary_lines = []
    overall_latency_ms = []
    profile_stages = bool(kwargs.get("profile_stages", False))
    stage_summary_lines = []
    global_id_mapper = GlobalTrackIdMapper()

    # --- 初始化偵測器（內建 GPUByteTracker）---
    detector = TRTYoloDetector(engine_path=engine)

    # --- ReID：直接使用 TRTFeatureExtractor + ZeroCopyCropper ---
    reid_enabled = not no_reid
    extractor = TRTFeatureExtractor() if reid_enabled else None
    cropper = ZeroCopyCropper() if reid_enabled else None
    reid_interval = int(kwargs.get("reid_interval", 4))

    # --- 其他參數 ---
    use_semantic_relink = kwargs.get("semantic_relink", False)
    person_class = int(kwargs.get("person_class", 0))
    track_person_only = bool(kwargs.get("track_person_only", True))
    track_thresh = float(kwargs.get("track_thresh", 0.1))
    high_thresh = float(kwargs.get("high_thresh", 0.5))
    match_thresh = float(kwargs.get("match_thresh", 0.8))
    preprocess_modes = parse_preprocess(kwargs.get("preprocess", "letterbox"))
    gamma = float(kwargs.get("gamma", 0.9))
    gamma_luma_threshold = float(kwargs.get("gamma_luma_threshold", 0.35))
    contrast = float(kwargs.get("contrast", 1.08))
    seqs = sequences.split(",") if sequences else [
        d.name for d in (Path(data_root) / split).iterdir() if d.is_dir()
    ]

    def time_stage(stage_totals, stage_name, fn, sync_cuda=False):
        if profile_stages and sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        if profile_stages and sync_cuda:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if profile_stages:
            stage_totals[stage_name] += elapsed_ms
        return result, elapsed_ms

    overall_stage_totals = OrderedDict(
        (name, 0.0)
        for name in (
            "fetch",
            "ingest_preprocess",
            "detect",
            "postprocess",
            "reid",
            "track",
            "relink_write",
            "frame_total",
        )
    )
    overall_profiled_frames = 0

    for seq in seqs:
        # 每個 sequence 重置追蹤器
        detector.reset_tracker()
        detector.tracker.set_params(
            track_thresh=track_thresh,
            high_thresh=high_thresh,
            match_thresh=match_thresh,
            track_buffer=30,
        )

        relinker = SemanticRelinker(
            sim_threshold=kwargs.get("semantic_threshold", 0.985),
            ttl=kwargs.get("semantic_ttl", 45),
            ema_beta=kwargs.get("semantic_ema", 0.83),
            spatial_gate=kwargs.get("semantic_spatial_gate", 0.11),
            min_lost_frames=kwargs.get("semantic_min_lost_frames", 2),
            min_iou=kwargs.get("semantic_min_iou", 0.0),
            mahalanobis_threshold=kwargs.get("semantic_mahalanobis_threshold", 6.6),
            debug=kwargs.get("semantic_debug", False),
        ) if use_semantic_relink else None

        seq_path = Path(data_root) / split / seq
        if not (seq_path / "seqinfo.ini").exists():
            continue
        config = configparser.ConfigParser()
        config.read(seq_path / "seqinfo.ini")
        w_orig = config.getint("Sequence", "imWidth")
        h_orig = config.getint("Sequence", "imHeight")
        frame_end = min(max_frames or int(1e9), config.getint("Sequence", "seqLength"))

        pool = AdaptiveFramePool(h_orig, w_orig)
        streamer = DALIStreamerStream(seq_path / "img1")
        stream_iter = iter(streamer)
        results_lines, frame_latencies = [], []
        start_time = time.time()
        warmup_frames = int(kwargs.get("warmup_frames", 50))
        seq_stage_totals = OrderedDict((name, 0.0) for name in overall_stage_totals.keys())
        seq_profiled_frames = 0

        for frame_id in range(1, frame_end + 1):
            t_e2e_start = time.perf_counter()
            try:
                frame_gpu, _fetch_ms = time_stage(
                    seq_stage_totals, "fetch", lambda: next(stream_iter), sync_cuda=False
                )
            except StopIteration:
                break
            t_frame_start = time.perf_counter()

            # 1. 注入影格
            _, _ = time_stage(
                seq_stage_totals,
                "ingest_preprocess",
                lambda: (
                    pool.frame_buffer.copy_(frame_gpu.permute(2, 0, 1).float() / 255.0),
                    apply_frame_preprocess(pool.frame_buffer, preprocess_modes, gamma, gamma_luma_threshold, contrast),
                ),
                sync_cuda=True,
            )

            # 2. 自適應偵測：高解析度走 960 tiled，低解析度保留單 patch
            (fused_boxes, fused_scores, fused_classes, is_tiled), _ = time_stage(
                seq_stage_totals,
                "detect",
                lambda: detect_adaptive_960_tiled(
                    detector, pool, h_orig, w_orig, preprocess_modes
                ),
                sync_cuda=True,
            )

            if fused_boxes.numel() == 0:
                if frame_id > warmup_frames:
                    frame_latencies.append((time.perf_counter() - t_frame_start) * 1000)
                if profile_stages and frame_id > warmup_frames:
                    seq_stage_totals["frame_total"] += (time.perf_counter() - t_e2e_start) * 1000
                    seq_profiled_frames += 1
                if frame_id % 100 == 0:
                    print(f"🎬 {seq} [{frame_id}/{frame_end}]")
                continue

            # Keep low-score boxes down to track_thresh so ByteTrack's
            # second-stage association can actually use them.
            if profile_stages:
                torch.cuda.synchronize()
                t_post_start = time.perf_counter()
            conf_mask = fused_scores > min(conf_threshold, track_thresh)
            fused_boxes = fused_boxes[conf_mask]
            fused_scores = fused_scores[conf_mask]
            fused_classes = fused_classes[conf_mask]

            if fused_boxes.numel() == 0:
                if frame_id > warmup_frames:
                    frame_latencies.append((time.perf_counter() - t_frame_start) * 1000)
                if profile_stages and frame_id > warmup_frames:
                    seq_stage_totals["frame_total"] += (time.perf_counter() - t_e2e_start) * 1000
                    seq_profiled_frames += 1
                if frame_id % 100 == 0:
                    print(f"🎬 {seq} [{frame_id}/{frame_end}]")
                continue

            if track_person_only:
                mask = fused_classes == person_class
                fused_boxes, fused_scores, fused_classes = fused_boxes[mask], fused_scores[mask], fused_classes[mask]

            if is_tiled and fused_boxes.numel() > 0:
                if track_person_only:
                    keep = nms(fused_boxes, fused_scores, 0.5)
                else:
                    keep = batched_nms(fused_boxes, fused_scores, fused_classes, 0.5)
                fused_boxes = fused_boxes[keep]
                fused_scores = fused_scores[keep]
                fused_classes = fused_classes[keep]

            if is_tiled and fused_boxes.numel() > 1:
                fused_boxes, fused_scores, fused_classes = merge_cross_tile_duplicates_fast(
                    fused_boxes, fused_scores, fused_classes
                )
            if profile_stages:
                torch.cuda.synchronize()
                seq_stage_totals["postprocess"] += (time.perf_counter() - t_post_start) * 1000

            # 4. ReID：每 reid_interval 幀同步提取 embedding
            embeddings = None
            if reid_enabled and extractor and cropper and fused_boxes.numel() > 0 and frame_id % reid_interval == 0:
                if profile_stages:
                    torch.cuda.synchronize()
                    t_reid_start = time.perf_counter()
                crops = cropper.process(pool.frame_buffer.unsqueeze(0), fused_boxes)
                if crops.numel() > 0:
                    embeddings = extractor.extract(crops)
                if profile_stages:
                    torch.cuda.synchronize()
                    seq_stage_totals["reid"] += (time.perf_counter() - t_reid_start) * 1000

            # 5. 追蹤（GPUByteTracker，內含 ReID 融合 + Sinkhorn）
            tracks, _ = time_stage(
                seq_stage_totals,
                "track",
                lambda: detector.tracker.update(
                    fused_boxes,
                    fused_scores,
                    fused_classes.to(torch.int32),
                    embeddings=embeddings,
                ),
                sync_cuda=True,
            )

            # 6. 可選：SemanticRelinker 後處理 Re-ID
            if profile_stages:
                torch.cuda.synchronize()
                t_relink_write_start = time.perf_counter()
            if relinker:
                relinker.update_motion_snapshots(detector.tracker.get_state_snapshots())

            assigned_ids: set = set()
            for t in tracks:
                if int(t.class_id) != person_class:
                    continue
                tid = t.obj_id
                if relinker:
                    # 取得此 track 對應的 embedding（若本幀有提取）
                    emb = None
                    if embeddings is not None:
                        # 找最近 bbox 匹配
                        tb = torch.tensor([t.x1, t.y1, t.x2, t.y2], device=fused_boxes.device)
                        ious = _box_iou_single(tb, fused_boxes)
                        best = int(ious.argmax())
                        if float(ious[best]) > 0.5:
                            emb = embeddings[best]
                    tid = relinker.resolve(
                        t.obj_id, emb,
                        (t.x1, t.y1, t.x2, t.y2),
                        t.score, frame_id, w_orig, h_orig, assigned_ids,
                    )
                global_tid = global_id_mapper.map(seq, tid)
                x1, y1, x2, y2 = t.x1, t.y1, t.x2, t.y2
                results_lines.append(
                    f"{frame_id},{global_tid},{max(0,x1):.2f},{max(0,y1):.2f},"
                    f"{min(w_orig,x2)-max(0,x1):.2f},{min(h_orig,y2)-max(0,y1):.2f},"
                    f"{t.score:.4f},-1,-1,-1"
                )
            if profile_stages:
                torch.cuda.synchronize()
                seq_stage_totals["relink_write"] += (time.perf_counter() - t_relink_write_start) * 1000

            if frame_id > warmup_frames:
                frame_latencies.append((time.perf_counter() - t_frame_start) * 1000)
            if profile_stages and frame_id > warmup_frames:
                seq_stage_totals["frame_total"] += (time.perf_counter() - t_e2e_start) * 1000
                seq_profiled_frames += 1
            if frame_id % 100 == 0:
                print(f"🎬 {seq} [{frame_id}/{frame_end}]")

        if frame_latencies:
            lats = np.array(frame_latencies)
            mean_ms = float(np.mean(lats))
            fps = 1000.0 / mean_ms
            print(f"\n📊 Production Latency Report for {seq}:")
            print(f"  - FPS:  {fps:.2f}")
            print(f"  - Mean latency: {mean_ms:.2f} ms")
            fps_summary_lines.append(
                f"{seq}\tfps={fps:.2f}\tmean_ms={mean_ms:.2f}\tframes={len(frame_latencies)}"
            )
            overall_latency_ms.extend(frame_latencies)
        else:
            fps_summary_lines.append(f"{seq}\tfps=n/a\tmean_ms=n/a\tframes=0")

        Path(output_root / f"{seq}.txt").write_text("\n".join(results_lines))
        print(f"✅ Finished {seq} (Total Time: {time.time()-start_time:.2f}s)")
        if relinker:
            relinker.report()
        if profile_stages and seq_profiled_frames > 0:
            print(f"\n🧪 Stage Profile for {seq}:")
            for stage_name, total_ms in seq_stage_totals.items():
                mean_ms = total_ms / seq_profiled_frames
                share = total_ms / max(seq_stage_totals["frame_total"], 1e-6) * 100.0
                print(f"  - {stage_name}: {mean_ms:.2f} ms/frame ({share:.1f}%)")
                overall_stage_totals[stage_name] += total_ms
            overall_profiled_frames += seq_profiled_frames
            stage_summary_lines.append(f"[{seq}] frames={seq_profiled_frames}")
            for stage_name, total_ms in seq_stage_totals.items():
                mean_ms = total_ms / seq_profiled_frames
                share = total_ms / max(seq_stage_totals["frame_total"], 1e-6) * 100.0
                stage_summary_lines.append(
                    f"{stage_name}\tmean_ms={mean_ms:.2f}\ttotal_ms={total_ms:.2f}\tshare={share:.1f}%"
                )
            stage_summary_lines.append("")

    if fps_summary_lines:
        if overall_latency_ms:
            overall_mean_ms = float(np.mean(np.array(overall_latency_ms)))
            overall_fps = 1000.0 / overall_mean_ms
            fps_summary_lines.append(
                f"OVERALL\tfps={overall_fps:.2f}\tmean_ms={overall_mean_ms:.2f}\tframes={len(overall_latency_ms)}"
            )
            print(f"\n📈 Overall throughput: {overall_fps:.2f} FPS ({overall_mean_ms:.2f} ms)")
        (output_root / "_fps_summary.txt").write_text("\n".join(fps_summary_lines) + "\n")
    mapping_lines = global_id_mapper.dump_lines()
    if mapping_lines:
        (output_root / "_global_id_map.txt").write_text("\n".join(mapping_lines) + "\n")
    if profile_stages and overall_profiled_frames > 0:
        print(f"\n🧪 Overall Stage Profile ({overall_profiled_frames} frames):")
        stage_summary_lines.append(f"[OVERALL] frames={overall_profiled_frames}")
        for stage_name, total_ms in overall_stage_totals.items():
            mean_ms = total_ms / overall_profiled_frames
            share = total_ms / max(overall_stage_totals["frame_total"], 1e-6) * 100.0
            print(f"  - {stage_name}: {mean_ms:.2f} ms/frame ({share:.1f}%)")
            stage_summary_lines.append(
                f"{stage_name}\tmean_ms={mean_ms:.2f}\ttotal_ms={total_ms:.2f}\tshare={share:.1f}%"
            )
        (output_root / "_stage_profile.txt").write_text("\n".join(stage_summary_lines) + "\n")


def _box_iou_single(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """計算單個 box 與一批 boxes 的 IoU。"""
    lt = torch.maximum(box[:2], boxes[:, :2])
    rb = torch.minimum(box[2:], boxes[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter + 1e-6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="models/yolo/yolo26s_batch4.engine")
    parser.add_argument("--output", default="results/MOT17_eval")
    parser.add_argument("--data-root", default="datasets/MOT17")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sequences", default="")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Detection threshold")
    parser.add_argument("--track-thresh", type=float, default=0.1)
    parser.add_argument("--high-thresh", type=float, default=0.5)
    parser.add_argument("--match-thresh", type=float, default=0.8)
    parser.add_argument("--no-reid", action="store_true")
    parser.add_argument("--reid-interval", type=int, default=4)
    parser.add_argument("--semantic-relink", action="store_true")
    parser.add_argument("--semantic-threshold", type=float, default=0.985)
    parser.add_argument("--semantic-ttl", type=int, default=45)
    parser.add_argument("--semantic-ema", type=float, default=0.83)
    parser.add_argument("--semantic-spatial-gate", type=float, default=0.11)
    parser.add_argument("--semantic-min-lost-frames", type=int, default=2)
    parser.add_argument("--semantic-min-iou", type=float, default=0.0)
    parser.add_argument("--semantic-mahalanobis-threshold", type=float, default=6.6)
    parser.add_argument("--semantic-debug", action="store_true")
    parser.add_argument("--person-class", type=int, default=0)
    parser.add_argument(
        "--track-person-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only keep person-class detections before tracking (default: enabled)",
    )
    parser.add_argument("--preprocess", default="letterbox",
                        help="Comma-separated: none, gamma, contrast, letterbox")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--gamma-luma-threshold", type=float, default=0.35)
    parser.add_argument("--contrast", type=float, default=1.08)
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--warmup-frames", type=int, default=50)
    args = parser.parse_args()
    run_eval(**vars(args))
