"""Probe: is the spatial gate per-scale separable?

The proposed heatmap-cache fast variant for Stage 2 assumes:

    live  teacher(frame, gate_input)[scale]
      ==  cached_identity_fpn[scale] * (1 + alpha[scale] * heatmap[scale])

This is exact ONLY if gating one scale does not change the pre-gate features of
the others. Gate layers sit at PAN indices p3=16, p4=19, p5=22; layer 16's
output feeds the bottom-up path into 19/22, so P3 gating may propagate. Measure
the per-scale residual between the live gated feature and the cache-reconstructed
feature. ~0 => separable (fast variant is bit-equivalent); large => NO-GO.
"""
# status: experiment

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from saccade.perception.temporal_yolo.dataset import build_mot17_dataloader
from saccade.perception.temporal_yolo.yolo_conditioned import TrackerGateInput
from saccade.perception.temporal_yolo.yolo_gated_detector import (
    _GATE_LAYER_IDX,
    GatedDetConfig,
    build_gated_yolo_detector,
)

DEV = torch.device("cuda")
IMG = 640
TEACHER_CKPT = ROOT / "runs/gated_det_v14_frozenyolo/epoch_0030.ckpt"


def main() -> None:
    cfg = GatedDetConfig(scales=("p3", "p4", "p5"), freeze_backbone=True, img_size=IMG)
    teacher = build_gated_yolo_detector(
        str(ROOT / "models/yolo/yolo26s.pt"),
        cfg=cfg,
        device=DEV,
        weights_path=str(TEACHER_CKPT),
    )
    teacher.eval()
    assert teacher.fusion is None, "probe assumes fusion disabled"
    alphas = {s: float(teacher.gate.alphas[s]) for s in ("p3", "p4", "p5")}
    print("trained alphas:", {k: round(v, 5) for k, v in alphas.items()})

    feats: dict[str, torch.Tensor] = {}
    for scale in ("p3", "p4", "p5"):
        idx = _GATE_LAYER_IDX[scale]

        def _cap(_m, _i, _o, s=scale):  # type: ignore[no-untyped-def]
            feats[s] = _o.detach()

        teacher.yolo_model.model[idx].register_forward_hook(_cap)

    loader = build_mot17_dataloader(
        data_root=str(ROOT / "datasets/MOT17"),
        clip_len=2,
        img_size=IMG,
        batch_size=4,
        stride=8,
        shuffle=True,
        seqs=["MOT17-02-SDP"],
        preload_to_ram=True,
        load_images=True,
        seed=42,
    )
    batch = next(iter(loader))
    frames = batch["frames"].to(DEV, dtype=torch.float32) / 255.0  # (B,T,3,H,W)
    gt = batch["gt_boxes"]
    B = frames.shape[0]
    frame1 = frames[:, 1]
    prev_gt = [gt[b][0] for b in range(B)]
    gate_inputs = [
        TrackerGateInput.from_boxes_scores(
            boxes.to(DEV), None, (IMG, IMG), assume_absolute=True
        ).to(DEV)
        for boxes in prev_gt
    ]

    with torch.no_grad():
        # 1) live: gate active on all scales
        feats.clear()
        teacher(frame1, gate_input=gate_inputs)
        live = {s: feats[s].clone() for s in ("p3", "p4", "p5")}
        # 2) identity: gate off -> exactly what Stage-0 cache stores
        feats.clear()
        teacher(frame1, gate_input=None)
        ident = {s: feats[s].clone() for s in ("p3", "p4", "p5")}

        # 3) cache reconstruction: ident * (1 + alpha * heatmap)
        renderer = teacher.gate.renderer
        print(
            f"\n{'scale':6} {'gate Δ (live vs ident)':24} "
            f"{'recon residual (live vs cache)':32} {'rel':8}"
        )
        for s in ("p3", "p4", "p5"):
            hw = (live[s].shape[2], live[s].shape[3])
            maps = renderer.batch_forward(gate_inputs, hw)
            gate_map = 1.0 + teacher.gate.alphas[s] * maps.unsqueeze(1)
            recon = ident[s] * gate_map
            gate_delta = (live[s] - ident[s]).abs().max().item()
            resid = (live[s] - recon).abs().max().item()
            scale_mag = live[s].abs().max().item()
            rel = resid / max(scale_mag, 1e-9)
            print(f"{s:6} {gate_delta:<24.6e} {resid:<32.6e} {rel:<8.2e}")


if __name__ == "__main__":
    main()
