#!/usr/bin/env python3
"""
Eval script for MambaGatedDetector (Option F).

Replaces GatedYOLODetector's YOLO Detect head with MambaDetectionHead,
keeping the same backbone + spatial gate + SimpleTracker eval loop.

Inference loop:
  - frame t gate_input = predicted boxes from frame t-1 (self-conditioned)
  - MambaDetectionHead runs on gated FPN features
  - NMS applied per-frame before writing MOT txt

Usage:
    uv run scripts/eval/eval_mamba_head.py \
        --teacher-ckpt runs/gated_det_v1/best.ckpt \
        --mamba-ckpt runs/mamba_distill_v1/best.ckpt \
        --data-root datasets/MOT17 \
        --output /tmp/mamba_head_eval \
        [--sequences MOT17-02-SDP,...] \
        [--score-thr 0.3] [--nms-iou 0.5] [--no-gate] \
        [--conf-thr 0.01] [--max-det 1000]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import saccade_tracking_ext  # noqa: F401  # must load before torchvision

import torch
import torchvision

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from saccade.perception.temporal_yolo.yolo_conditioned import TrackerGateInput  # noqa: E402
from saccade.perception.temporal_yolo.mamba_gated_detector import (  # noqa: E402
    build_mamba_gated_detector,
)

_SDP_SEQS = [
    "MOT17-02-SDP",
    "MOT17-04-SDP",
    "MOT17-05-SDP",
    "MOT17-09-SDP",
    "MOT17-10-SDP",
    "MOT17-11-SDP",
    "MOT17-13-SDP",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _iter_frames(seq_dir: Path, img_size: int):
    import torchvision.io as tv_io

    img_dir = seq_dir / "img1"
    paths = sorted(img_dir.glob("*.jpg")) or sorted(img_dir.glob("*.png"))
    for p in paths:
        fid = int(p.stem)
        img = tv_io.read_image(str(p)).float() / 255.0
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
        yield fid, img


def _orig_hw(seq_dir: Path) -> tuple[int, int]:
    ini = seq_dir / "seqinfo.ini"
    w, h = 1920, 1080
    if ini.exists():
        for line in ini.read_text().splitlines():
            if line.startswith("imWidth"):
                w = int(line.split("=")[1])
            elif line.startswith("imHeight"):
                h = int(line.split("=")[1])
    return h, w


def _nms_filter(
    boxes_xyxy: torch.Tensor,
    scores: torch.Tensor,
    score_thr: float,
    nms_iou: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    keep = scores >= score_thr
    boxes_f, scores_f = boxes_xyxy[keep], scores[keep]
    if boxes_f.numel() == 0:
        return boxes_f.new_zeros((0, 4)), scores_f.new_zeros((0,))
    idx = torchvision.ops.nms(boxes_f, scores_f, nms_iou)
    return boxes_f[idx], scores_f[idx]


def _to_mot_lines(frame_id: int, xyxy: torch.Tensor, track_ids: list[int]) -> list[str]:
    lines = []
    for (x1, y1, x2, y2), tid in zip(xyxy.tolist(), track_ids):
        w, h = x2 - x1, y2 - y1
        lines.append(f"{frame_id},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},-1,-1,-1,-1")
    return lines


# ---------------------------------------------------------------------------
# Simple IoU tracker
# ---------------------------------------------------------------------------
class SimpleTracker:
    def __init__(self, iou_thr: float = 0.4, max_age: int = 1):
        self.next_id = 1
        self.tracks: list[dict] = []
        self.iou_thr = iou_thr
        self.max_age = max_age

    def update(self, xyxy: torch.Tensor) -> list[int]:
        if xyxy.numel() == 0:
            for t in self.tracks:
                t["age"] += 1
            self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]
            return []
        n_det = xyxy.shape[0]
        assigned = [-1] * n_det
        if self.tracks:
            track_boxes = torch.stack([t["xyxy"] for t in self.tracks]).to(xyxy.device)
            iou = torchvision.ops.box_iou(xyxy, track_boxes)
            matched_det: set[int] = set()
            matched_trk: set[int] = set()
            vals, cols = iou.max(dim=1)
            for di in vals.argsort(descending=True).tolist():
                ti = int(cols[di])
                if (
                    float(vals[di]) >= self.iou_thr
                    and di not in matched_det
                    and ti not in matched_trk
                ):
                    assigned[di] = self.tracks[ti]["id"]
                    self.tracks[ti]["xyxy"] = xyxy[di]
                    self.tracks[ti]["age"] = 0
                    matched_det.add(di)
                    matched_trk.add(ti)
            for ti, t in enumerate(self.tracks):
                if ti not in matched_trk:
                    t["age"] += 1
            self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]
        ids: list[int] = []
        for di in range(n_det):
            if assigned[di] == -1:
                tid = self.next_id
                self.next_id += 1
                self.tracks.append({"id": tid, "xyxy": xyxy[di], "age": 0})
                ids.append(tid)
            else:
                ids.append(assigned[di])
        return ids

    def reset(self) -> None:
        self.tracks = []
        self.next_id = 1


class JdeSimpleTracker(SimpleTracker):
    """SimpleTracker with JDE embedding-based re-identification.

    Uses IoU matching first, then cosine-similarity relink for lost tracks.
    Maintains quality-weighted EMA embeddings per track.
    """

    def __init__(
        self,
        iou_thr: float = 0.4,
        max_age: int = 30,
        cos_thr: float = 0.60,
        emb_alpha: float = 0.20,
    ):
        super().__init__(iou_thr=iou_thr, max_age=max_age)
        self.cos_thr = cos_thr
        self.emb_alpha = emb_alpha

    def update(
        self,
        xyxy: torch.Tensor,
        embeddings: torch.Tensor | None = None,
    ) -> list[int]:
        has_emb = embeddings is not None and embeddings.numel() > 0 and xyxy.numel() > 0

        if xyxy.numel() == 0:
            for t in self.tracks:
                t["age"] += 1
            self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]
            return []
        n_det = xyxy.shape[0]
        assigned = [-1] * n_det
        if self.tracks:
            track_boxes = torch.stack([t["xyxy"] for t in self.tracks]).to(xyxy.device)
            iou = torchvision.ops.box_iou(xyxy, track_boxes)
            matched_det: set[int] = set()
            matched_trk: set[int] = set()

            vals, cols = iou.max(dim=1)
            for di in vals.argsort(descending=True).tolist():
                ti = int(cols[di])
                if (
                    float(vals[di]) >= self.iou_thr
                    and di not in matched_det
                    and ti not in matched_trk
                ):
                    assigned[di] = self.tracks[ti]["id"]
                    self.tracks[ti]["xyxy"] = xyxy[di]
                    self.tracks[ti]["age"] = 0
                    if has_emb and "emb" in self.tracks[ti]:
                        self._ema_update(ti, embeddings[di])
                    matched_det.add(di)
                    matched_trk.add(ti)

            if has_emb:
                unmatched_det = [di for di in range(n_det) if assigned[di] == -1]
                lost_trks = [
                    ti
                    for ti, t in enumerate(self.tracks)
                    if t["age"] > 0 and "emb" in t
                ]
                if unmatched_det and lost_trks:
                    det_embs = embeddings[unmatched_det]
                    trk_embs = torch.stack([self.tracks[ti]["emb"] for ti in lost_trks])
                    cos_sim = det_embs @ trk_embs.T
                    cos_vals, cos_idx = cos_sim.max(dim=1)
                    matched_cos: set[int] = set()
                    for di_rank in cos_vals.argsort(descending=True).tolist():
                        ti_rank = int(cos_idx[di_rank])
                        if (
                            float(cos_vals[di_rank]) >= self.cos_thr
                            and di_rank not in matched_cos
                            and ti_rank not in matched_trk
                        ):
                            orig_di = unmatched_det[di_rank]
                            orig_ti = lost_trks[ti_rank]
                            assigned[orig_di] = self.tracks[orig_ti]["id"]
                            self.tracks[orig_ti]["xyxy"] = xyxy[orig_di]
                            self.tracks[orig_ti]["age"] = 0
                            self._ema_update(orig_ti, embeddings[orig_di])
                            matched_cos.add(di_rank)
                            matched_trk.add(orig_ti)

            for ti in range(len(self.tracks)):
                if ti not in matched_trk:
                    self.tracks[ti]["age"] += 1
            self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]

        ids: list[int] = []
        for di in range(n_det):
            if assigned[di] == -1:
                tid = self.next_id
                self.next_id += 1
                trk = {"id": tid, "xyxy": xyxy[di], "age": 0}
                if has_emb:
                    trk["emb"] = embeddings[di]
                self.tracks.append(trk)
                ids.append(tid)
            else:
                ids.append(assigned[di])
        return ids

    def _ema_update(self, ti: int, new_emb: torch.Tensor) -> None:
        t = self.tracks[ti]
        if "emb" in t:
            t["emb"] = self.emb_alpha * new_emb + (1 - self.emb_alpha) * t["emb"]
            t["emb"] = torch.nn.functional.normalize(t["emb"], dim=0)
        else:
            t["emb"] = new_emb

    def reset(self) -> None:
        self.tracks = []
        self.next_id = 1


# ---------------------------------------------------------------------------
# Per-sequence eval
# ---------------------------------------------------------------------------
def eval_sequence(
    model,
    seq_dir: Path,
    output_dir: Path,
    img_size: int,
    score_thr: float,
    nms_iou: float,
    device: torch.device,
    disable_gate: bool = False,
    use_jde: bool = False,
    jde_max_age: int = 30,
    jde_cos_thr: float = 0.60,
) -> int:
    seq_name = seq_dir.name
    orig_h, orig_w = _orig_hw(seq_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_jde:
        tracker = JdeSimpleTracker(
            iou_thr=0.4, max_age=jde_max_age, cos_thr=jde_cos_thr
        )
    else:
        tracker = SimpleTracker()
    prev_boxes_xyxy: torch.Tensor | None = None
    prev_scores: torch.Tensor | None = None
    sx, sy = orig_w / img_size, orig_h / img_size

    mot_lines: list[str] = []
    num_frames = 0
    t0 = time.perf_counter()

    with torch.inference_mode():
        for frame_id, frame in _iter_frames(seq_dir, img_size):
            frame = frame.to(device)

            gate_input = None
            if (
                not disable_gate
                and prev_boxes_xyxy is not None
                and prev_boxes_xyxy.numel() > 0
            ):
                gate_input = TrackerGateInput.from_boxes_scores(
                    prev_boxes_xyxy,
                    prev_scores,
                    (img_size, img_size),
                    assume_absolute=True,
                ).to(device)

            out = model(frame, gate_input=gate_input)
            raw = out[0][0]  # (max_det, 6)
            extra = out[1]

            boxes_xyxy_640, kept_scores = _nms_filter(
                raw[:, :4], raw[:, 4], score_thr, nms_iou
            )
            prev_boxes_xyxy = boxes_xyxy_640
            prev_scores = kept_scores

            xyxy = boxes_xyxy_640.clone()
            if xyxy.numel() > 0:
                xyxy[:, [0, 2]] *= sx
                xyxy[:, [1, 3]] *= sy

            embeddings = None
            if use_jde and "emb_preds" in extra and xyxy.numel() > 0:
                nms_boxes_640 = boxes_xyxy_640.to(device)
                embeddings = model.extract_det_embeddings(
                    extra["emb_preds"], nms_boxes_640
                )

            if use_jde:
                track_ids = tracker.update(xyxy, embeddings)
            else:
                track_ids = tracker.update(xyxy)
            mot_lines.extend(_to_mot_lines(frame_id, xyxy, track_ids))
            num_frames += 1

    elapsed = time.perf_counter() - t0
    fps = num_frames / max(elapsed, 1e-6)
    (output_dir / f"{seq_name}.txt").write_text("\n".join(mot_lines))
    print(f"  {seq_name}: {num_frames} frames, {fps:.1f} FPS, {len(mot_lines)} dets")
    return num_frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-ckpt", default="runs/gated_det_v1/best.ckpt")
    parser.add_argument("--mamba-ckpt", default="runs/mamba_distill_v1/best.ckpt")
    parser.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    parser.add_argument("--data-root", default="datasets/MOT17")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sequences", default="")
    parser.add_argument("--output", default="/tmp/mamba_head_eval")
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--conf-thr", type=float, default=0.01)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--trt-engine", default="")
    parser.add_argument("--emb-dim", type=int, default=0)
    parser.add_argument("--jde-proj-ckpt", default="")
    parser.add_argument("--jde-max-age", type=int, default=30)
    parser.add_argument("--jde-cos-thr", type=float, default=0.60)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.teacher_ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = project_root / ckpt_path
    if not ckpt_path.exists():
        print(f"[Warn] Teacher ckpt not found: {ckpt_path} — using fresh backbone")
        ckpt_path = Path("__missing__")  # sentinel for 'not found'
    mamba_path = Path(args.mamba_ckpt)
    if not mamba_path.is_absolute():
        mamba_path = project_root / mamba_path

    model = build_mamba_gated_detector(
        yolo_pt_path=args.yolo_weights,
        teacher_ckpt=str(ckpt_path),
        mamba_ckpt=str(mamba_path),
        img_size=args.img_size,
        device=device,
        conf_thr=args.conf_thr,
        max_det=args.max_det,
        trt_backbone_engine=args.trt_engine,
        emb_dim=args.emb_dim,
        jde_proj_ckpt=args.jde_proj_ckpt,
    )
    model.eval()
    use_jde = args.emb_dim > 0
    print(f"Loaded teacher: {ckpt_path}")
    print(f"Loaded mamba:   {mamba_path}")
    print(
        f"Gate {'DISABLED' if args.no_gate else 'ENABLED'}  "
        f"score_thr={args.score_thr}  conf_thr={args.conf_thr}  "
        f"max_det={args.max_det}"
    )
    if use_jde:
        print(
            f"JDE emb_dim={args.emb_dim} proj_ckpt={args.jde_proj_ckpt or 'none'} "
            f"max_age={args.jde_max_age} cos_thr={args.jde_cos_thr}"
        )

    seqs = [s.strip() for s in args.sequences.split(",") if s.strip()] or _SDP_SEQS
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = project_root / data_root
    output_dir = Path(args.output)

    print(f"\nEvaluating {len(seqs)} sequences -> {output_dir}")
    for seq in seqs:
        seq_dir = data_root / args.split / seq
        if not seq_dir.exists():
            print(f"  [SKIP] {seq_dir}")
            continue
        eval_sequence(
            model,
            seq_dir,
            output_dir,
            args.img_size,
            args.score_thr,
            args.nms_iou,
            device,
            disable_gate=args.no_gate,
            use_jde=use_jde,
            jde_max_age=args.jde_max_age,
            jde_cos_thr=args.jde_cos_thr,
        )

    try:
        from saccade.perception.eval.metrics import run_motmetrics_evaluation

        metrics = run_motmetrics_evaluation(
            data_root=str(data_root),
            split=args.split,
            output=str(output_dir),
            sequences=",".join(seqs),
            detector="SDP",
        )
        if metrics:
            print("\n=== OVERALL METRICS ===")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"\n[Warn] motmetrics failed: {e}")

    model.remove_hooks()


if __name__ == "__main__":
    main()
