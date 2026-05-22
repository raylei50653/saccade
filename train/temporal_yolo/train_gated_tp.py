"""
GatedYOLODetector training with Track-guided TP Recall Loss.

Loss = v8DetectionLoss + tp_weight * TPRecallLoss

TPRecallLoss:
  - GPUByteTracker runs on model's own predictions (self-conditioned)
  - Kalman-predicted positions for frame t (with velocity correction)
  - Gaussian soft-assignment to anchor grid → weighted sigmoid score
  - -log(weighted_score): focal recall loss at each predicted track position
  - Reward when detector fires near existing tracks

Gate input = same Kalman predictions (tracker → gate + TP loss simultaneously)

Usage:
    uv run train/temporal_yolo/train_gated_tp.py \\
        --data-root datasets/MOT17 \\
        --yolo-weights models/yolo/yolo26s.pt \\
        --resume runs/gated_det_v1/best.ckpt \\
        --epochs 30 --tp-weight 0.5 --gt-ratio 0.3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from saccade.perception.temporal_yolo.dataset import build_mot17_dataloader  # noqa: E402
from saccade.perception.temporal_yolo.yolo_conditioned import TrackerGateInput  # noqa: E402
from saccade.perception.temporal_yolo.yolo_gated_detector import (  # noqa: E402
    GatedDetConfig,
    GatedYOLODetector,
    build_gated_yolo_detector,
)
from saccade.perception.tracking.tracker_gpu import GPUByteTracker  # noqa: E402
from ultralytics.utils.tal import make_anchors  # noqa: E402


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(
    state: dict[str, Any], run_dir: Path, epoch: int, is_best: bool = False
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, run_dir / "latest.ckpt")
    torch.save(state, run_dir / f"epoch_{epoch:04d}.ckpt")
    if is_best:
        torch.save(state, run_dir / "best.ckpt")
    tag = " [BEST]" if is_best else ""
    print(f"  Saved epoch_{epoch:04d}.ckpt{tag}")


def _strip_compiled_keys(sd: dict[str, Any]) -> dict[str, Any]:
    return {k.replace("._orig_mod.", "."): v for k, v in sd.items()}


def load_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer
) -> int:
    print(f"[Resume] {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    sd = _strip_compiled_keys(state["model"])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Missing ({len(missing)}): {missing[:3]}")
    if unexpected:
        print(f"  Unexpected ({len(unexpected)})")
    try:
        optimizer.load_state_dict(state["optimizer"])
    except Exception:
        print("  [Warn] Optimizer state not loaded")
    return state.get("epoch", 0) + 1  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# GT helpers
# ---------------------------------------------------------------------------
def _xyxy_to_cxcywh_norm(boxes: torch.Tensor, img_size: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2 / img_size
    cy = (y1 + y2) / 2 / img_size
    w = (x2 - x1) / img_size
    h = (y2 - y1) / img_size
    return torch.stack([cx, cy, w, h], dim=1)


def _make_yolo_batch(
    gt_boxes_list: list[torch.Tensor],
    img_size: int,
    device: torch.device,
) -> dict[str, Any]:
    batch_idxs, clss, bboxes = [], [], []
    for b, boxes in enumerate(gt_boxes_list):
        if boxes.numel() == 0:
            continue
        n = boxes.shape[0]
        batch_idxs.append(torch.full((n,), float(b)))
        clss.append(torch.zeros(n))
        bboxes.append(_xyxy_to_cxcywh_norm(boxes, img_size))
    if not batch_idxs:
        return {
            "batch_idx": torch.zeros(0, device=device),
            "cls": torch.zeros(0, device=device),
            "bboxes": torch.zeros(0, 4, device=device),
        }
    return {
        "batch_idx": torch.cat(batch_idxs).to(device),
        "cls": torch.cat(clss).to(device),
        "bboxes": torch.cat(bboxes).to(device),
    }


# ---------------------------------------------------------------------------
# Anchor grid (precomputed once)
# ---------------------------------------------------------------------------
def build_anchor_grid(
    model: GatedYOLODetector, img_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute anchor center pixel coordinates for TP recall loss."""
    detect_head = model.yolo_model.model[-1]
    with torch.no_grad():
        dummy = torch.zeros(1, 3, img_size, img_size, device=device)
        model.eval()
        out = model(dummy)
        model.train()
        feats = out[1]["one2many"]["feats"]
    anchors, strides = make_anchors(feats, detect_head.stride, 0.5)  # type: ignore[no-untyped-call]
    anchor_cx_px = (anchors[:, 0] * strides[:, 0]).to(device)  # (8400,)
    anchor_cy_px = (anchors[:, 1] * strides[:, 0]).to(device)  # (8400,)
    return anchor_cx_px, anchor_cy_px


# ---------------------------------------------------------------------------
# TP Recall Loss
# ---------------------------------------------------------------------------
def compute_tp_recall_loss(
    scores_raw: torch.Tensor,  # (B, 80, 8400) raw logits from one2many
    anchor_cx_px: torch.Tensor,  # (8400,)
    anchor_cy_px: torch.Tensor,  # (8400,)
    track_pred_list: list[Any],  # B items, each (N, 4) xyxy img_size px or None
    sigma_scale: float = 0.5,
    person_cls: int = 0,
) -> torch.Tensor:
    """
    For each Kalman-predicted track position, penalise low detector score.
    Gaussian soft-assignment avoids hard anchor matching.
    """
    person_scores = scores_raw[:, person_cls, :]  # (B, 8400)
    total_loss = scores_raw.new_zeros(())
    n_tracks = 0

    for b, boxes in enumerate(track_pred_list):
        if boxes is None or boxes.numel() == 0:
            continue
        boxes = boxes.to(scores_raw.device)

        cx = (boxes[:, 0] + boxes[:, 2]) / 2  # (N,)
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        w = (boxes[:, 2] - boxes[:, 0]).clamp(min=4)
        h = (boxes[:, 3] - boxes[:, 1]).clamp(min=4)
        sigma = sigma_scale * (w * h).sqrt()  # (N,) adaptive

        # (N, 8400) distance²
        dx = anchor_cx_px.unsqueeze(0) - cx.unsqueeze(1)
        dy = anchor_cy_px.unsqueeze(0) - cy.unsqueeze(1)
        dist2 = dx * dx + dy * dy

        weights = torch.exp(-dist2 / (2 * sigma.unsqueeze(1) ** 2 + 1e-6))  # (N, 8400)
        sig_scores = torch.sigmoid(person_scores[b])  # (8400,)

        denom = weights.sum(dim=1) + 1e-6
        weighted_score = (weights * sig_scores.unsqueeze(0)).sum(dim=1) / denom  # (N,)

        total_loss = total_loss + (-torch.log(weighted_score.clamp(min=1e-6))).sum()
        n_tracks += len(boxes)

    return total_loss / max(n_tracks, 1)


# ---------------------------------------------------------------------------
# Per-sample tracker: runs on model predictions, provides Kalman predictions
# ---------------------------------------------------------------------------
class TrainTracker:
    """Maintains ByteTrack state on model predictions during training."""

    def __init__(self, img_size: int, device: torch.device, conf_thr: float = 0.25):
        self.img_size = img_size
        self.device = device
        self.conf_thr = conf_thr
        self.tracker = GPUByteTracker(max_objects=512)
        self.tracker.set_params(
            track_thresh=0.05,
            high_thresh=0.45,
            match_thresh=0.66,
            track_buffer=30,
            new_track_thresh=0.28,
            confirm_streak=1,
            fuse_score_weight=0.4,
        )
        self.tracker.set_frame_size(img_size, img_size)
        self._next_pred: torch.Tensor | None = None  # Kalman predictions for next frame

    def reset(self) -> None:
        self.tracker = GPUByteTracker(max_objects=512)
        self.tracker.set_params(
            track_thresh=0.05,
            high_thresh=0.45,
            match_thresh=0.66,
            track_buffer=30,
            new_track_thresh=0.28,
            confirm_streak=1,
            fuse_score_weight=0.4,
        )
        self.tracker.set_frame_size(self.img_size, self.img_size)
        self._next_pred = None

    @torch.no_grad()
    def update_gt(self, gt_boxes: torch.Tensor, scores: torch.Tensor) -> None:
        """Update tracker with GT boxes and compute Kalman predictions for next frame."""
        if gt_boxes.numel() > 0:
            boxes = gt_boxes.to(self.device).contiguous()
            scores_ = scores.to(self.device).contiguous()
            classes = torch.zeros(len(boxes), device=self.device, dtype=torch.int32)
        else:
            boxes = torch.zeros((0, 4), device=self.device)
            scores_ = torch.zeros((0,), device=self.device)
            classes = torch.zeros((0,), device=self.device, dtype=torch.int32)

        self.tracker.update(boxes, scores_, classes)

        # Kalman-predict next frame positions (with velocity correction)
        snaps = self.tracker.get_state_snapshots()
        if not snaps:
            self._next_pred = None
            return
        preds = []
        for s in snaps:
            cx, cy, ar, h = s.state[0], s.state[1], s.state[2], s.state[3]
            vcx, vcy, var, vh = s.state[4], s.state[5], s.state[6], s.state[7]
            cx_p = cx + vcx
            cy_p = cy + vcy
            h_p = max(h + vh, 4.0)
            w_p = max((ar + var) * h_p, 4.0)
            preds.append(
                [cx_p - w_p / 2, cy_p - h_p / 2, cx_p + w_p / 2, cy_p + h_p / 2]
            )
        self._next_pred = torch.tensor(preds, dtype=torch.float32, device=self.device)

        # Kalman-predict next frame positions (velocity correction)
        snaps = self.tracker.get_state_snapshots()
        if not snaps:
            self._next_pred = None
            return

        preds = []
        for s in snaps:
            cx, cy, ar, h = s.state[0], s.state[1], s.state[2], s.state[3]
            vcx, vcy, var, vh = s.state[4], s.state[5], s.state[6], s.state[7]
            cx_p = cx + vcx
            cy_p = cy + vcy
            h_p = max(h + vh, 4.0)
            w_p = max((ar + var) * h_p, 4.0)
            preds.append(
                [cx_p - w_p / 2, cy_p - h_p / 2, cx_p + w_p / 2, cy_p + h_p / 2]
            )

        self._next_pred = torch.tensor(preds, dtype=torch.float32, device=self.device)

    def get_predictions(self) -> torch.Tensor | None:
        return self._next_pred


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(  # type: ignore[no-untyped-def]
    model: GatedYOLODetector,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion,
    scaler: Any,  # GradScaler
    device: torch.device,
    img_size: int,
    gt_ratio: float,
    tp_weight: float,
    tp_sigma_scale: float,
    anchor_cx_px: torch.Tensor,
    anchor_cy_px: torch.Tensor,
    clip_grad: float = 10.0,
) -> tuple[float, float]:
    """
    Training loop with GT-oracle tracker for Kalman velocity predictions.

    Tracker is updated with GT boxes each frame (oracle).
    Kalman predictions for frame t include velocity correction from GT motion.
    TP recall loss: detector must score high at Kalman-predicted positions.
    Gate input: same Kalman predictions.
    """
    model.train()
    total_det_loss = 0.0
    total_tp_loss = 0.0
    n_steps = 0

    for batch in loader:
        frames: torch.Tensor = batch["frames"].to(device, dtype=torch.float32) / 255.0
        gt_boxes_batch: list[list[torch.Tensor]] = batch["gt_boxes"]
        B, T = frames.shape[:2]

        # One GT-oracle tracker per sample; reset per clip
        train_trackers = [TrainTracker(img_size, device) for _ in range(B)]

        optimizer.zero_grad()
        batch_loss = frames.new_zeros(())
        batch_tp_loss = 0.0

        for t in range(T):
            frame_t = frames[:, t]
            gt_t = [gt_boxes_batch[b][t] for b in range(B)]

            # ── Gate input + TP targets: Kalman predictions from GT-oracle tracker ──
            track_preds = [tr.get_predictions() for tr in train_trackers]
            has_preds = any(p is not None and p.numel() > 0 for p in track_preds)

            gate_inputs = None
            if t > 0 and has_preds:
                gate_inputs = [
                    TrackerGateInput.from_boxes_scores(
                        track_preds[b],  # type: ignore[arg-type]
                        None,
                        (img_size, img_size),
                        assume_absolute=True,
                    ).to(device)
                    if track_preds[b] is not None and track_preds[b].numel() > 0  # type: ignore[union-attr]
                    else None
                    for b in range(B)
                ]

            # ── Forward (train mode) ──
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):  # type: ignore[attr-defined]
                out = model(frame_t, gate_input=gate_inputs)
                preds = out["one2many"]

                # Standard detection loss
                yolo_batch = _make_yolo_batch(gt_t, img_size, device)
                det_loss_vec, _ = criterion(preds, yolo_batch)
                step_loss = det_loss_vec.sum() / T

                # TP recall loss at Kalman-predicted track positions
                if t > 0 and tp_weight > 0 and has_preds:
                    tp_loss = compute_tp_recall_loss(
                        preds["scores"].float(),
                        anchor_cx_px,
                        anchor_cy_px,
                        track_preds,
                        sigma_scale=tp_sigma_scale,
                    )
                    step_loss = step_loss + tp_weight * tp_loss / T
                    batch_tp_loss += float(tp_loss.detach())

            batch_loss = batch_loss + step_loss

            # ── Update GT-oracle tracker for next frame (no_grad) ──
            with torch.no_grad():
                for b in range(B):
                    gt_boxes = gt_t[b].to(device)
                    scores = torch.ones(len(gt_boxes), device=device) * 0.9
                    train_trackers[b].update_gt(gt_boxes, scores)

        scaler.scale(batch_loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], clip_grad
        )
        scaler.step(optimizer)
        scaler.update()

        if torch.isfinite(batch_loss):
            total_det_loss += batch_loss.detach().item()
            total_tp_loss += batch_tp_loss
            n_steps += 1

    denom = max(n_steps, 1)
    return total_det_loss / denom, total_tp_loss / denom


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="GatedYOLODetector + TP Recall Loss")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--yolo-weights", default="models/yolo/yolo26s.pt")
    parser.add_argument("--run-dir", default="runs/gated_tp")
    parser.add_argument("--resume", default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--clip-len", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--lr-gate", type=float, default=1e-3)
    parser.add_argument("--lr-yolo", type=float, default=1e-5)
    parser.add_argument(
        "--gt-ratio",
        type=float,
        default=0.3,
        help="Fallback GT oracle gate ratio when tracker has no tracks",
    )
    parser.add_argument(
        "--tp-weight",
        type=float,
        default=0.5,
        help="TP recall loss weight (0 = disable)",
    )
    parser.add_argument(
        "--tp-sigma-scale",
        type=float,
        default=0.5,
        help="Gaussian sigma = sigma_scale * sqrt(w*h)",
    )
    parser.add_argument("--seqs", default="")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--scales", default="p3,p4,p5")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    seqs = args.seqs.split(",") if args.seqs else None
    scales = tuple(s.strip() for s in args.scales.split(","))

    # ── Model ──
    cfg = GatedDetConfig(
        scales=scales,
        gate_sigma_scale=0.5,
        gate_min_score=0.5,
        freeze_backbone=(args.lr_yolo == 0.0),
    )
    yolo_weights = project_root / args.yolo_weights
    model = build_gated_yolo_detector(str(yolo_weights), cfg, device)

    # ── Loss ──
    from ultralytics.utils.loss import v8DetectionLoss
    from types import SimpleNamespace

    base_args = (
        dict(model.yolo_model.args) if isinstance(model.yolo_model.args, dict) else {}
    )
    base_args.setdefault("box", 7.5)
    base_args.setdefault("cls", 0.5)
    base_args.setdefault("dfl", 1.5)
    model.yolo_model.args = SimpleNamespace(**base_args)
    criterion = v8DetectionLoss(model.yolo_model)

    # ── Precompute anchor grid ──
    anchor_cx_px, anchor_cy_px = build_anchor_grid(model, args.img_size, device)
    print(
        f"[TP] Anchor grid: {anchor_cx_px.shape[0]} anchors  "
        f"sigma_scale={args.tp_sigma_scale}  tp_weight={args.tp_weight}"
    )

    # ── Optimizer ──
    param_groups = model.parameter_groups(lr_gate=args.lr_gate, lr_yolo=args.lr_yolo)
    n_gate = sum(
        p.numel() for g in param_groups if g["name"] == "gate" for p in g["params"]
    )
    print(f"[GatedDet] gate params: {n_gate}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_gate * 0.01
    )
    scaler = torch.amp.GradScaler("cuda")  # type: ignore[attr-defined]

    # ── Resume ──
    start_epoch = 1
    best_loss = float("inf")
    if args.resume:
        resumed_epoch = load_checkpoint(Path(args.resume), model, optimizer)
        start_epoch = resumed_epoch
        # If resuming from epoch N, train for args.epochs MORE epochs
        args.epochs = start_epoch - 1 + args.epochs

    # ── DataLoader ──
    loader = build_mot17_dataloader(
        data_root=args.data_root,
        clip_len=args.clip_len,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seqs=seqs,
        preload_to_ram=True,
    )
    print(f"[GatedDet] {len(loader)} batches/epoch  gt_ratio={args.gt_ratio}")
    print(f"[GatedDet] Initial alphas — {model.alpha_summary()}")

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()
        det_loss, tp_loss = train_one_epoch(
            model,
            loader,
            optimizer,
            criterion,
            scaler,
            device,
            args.img_size,
            args.gt_ratio,
            args.tp_weight,
            args.tp_sigma_scale,
            anchor_cx_px,
            anchor_cy_px,
        )
        scheduler.step()
        elapsed = time.perf_counter() - t0
        total_loss = det_loss + args.tp_weight * tp_loss
        is_best = total_loss < best_loss
        best_loss = min(best_loss, total_loss)
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"det={det_loss:.4f}  tp={tp_loss:.4f}  "
            f"alphas=[{model.alpha_summary()}]  "
            f"{elapsed / 60:.1f}min" + (" [BEST]" if is_best else "")
        )

        if epoch % args.save_every == 0 or epoch == args.epochs or is_best:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "args": vars(args),
                    "cfg": cfg,
                },
                run_dir,
                epoch,
                is_best=is_best,
            )

    print(f"\n[Done] best_loss={best_loss:.4f}  ckpt → {run_dir}/best.ckpt")


if __name__ == "__main__":
    main()
