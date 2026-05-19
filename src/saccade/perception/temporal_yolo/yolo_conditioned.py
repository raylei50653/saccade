"""
Option D: TemporalYOLOConditioned — Track-Conditioned YOLO FPN Gate.

Extends TemporalYOLOJoint (Option C) with a per-scale Gaussian heatmap gate:
  gate_s = 1 + alpha_s * heatmap_s

where heatmap_s is built from tracker confirmed-track positions (optionally with
Kalman velocity prediction). alpha_s is initialized to 0 so training starts
identical to the unfused Option C baseline — no gradient disruption.

Usage (inference, full tracker output):
    gate_input = TrackerGateInput.from_tracker_results(
        track_results, state_snapshots, candidates, img_hw=(H, W)
    )
    out = model(frame, gate_input=gate_input)

Usage (training Phase 1, GT oracle boxes):
    out = model(frame, prev_boxes=gt_boxes, prev_scores=None)

Training phases:
  Phase 1  freeze backbone+decoder, train gate.alphas only with GT oracle boxes
  Phase 2  unfreeze all, differential LR, gradually replace GT with predicted boxes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .model import TemporalYOLOConfig, TrackQueryDecoderLayer, TrackQueryLifecycle
from .yolo_joint import FPNSequenceProjection, YOLOFeaturePyramid


# ---------------------------------------------------------------------------
# TrackerGateInput
# ---------------------------------------------------------------------------
@dataclass
class TrackerGateInput:
    """
    Unified tracker-to-gate interface. All boxes in absolute pixel coords [x1,y1,x2,y2].

    confirmed_occluded: per-track bool mask, True where det_idx == -1 (pure Kalman
    prediction, no detection match this frame). These tracks get wider sigma (×1.5)
    and halved intensity to signal positional uncertainty.
    """

    confirmed_boxes: Tensor  # (N, 4) [x1,y1,x2,y2] absolute pixels
    confirmed_scores: Tensor  # (N,) confidence in [0, 1]
    velocities: Tensor | None  # (N, 2) [vx, vy] px/frame from Kalman state[4:6]
    tentative_boxes: Tensor | None  # (M, 4) unconfirmed tracks
    tentative_ratios: (
        Tensor | None
    )  # (M,) hit_streak / required_confirm_streak ∈ [0, 1]
    img_hw: tuple[int, int]
    confirmed_occluded: Tensor | None = None  # (N,) bool; True ↔ det_idx == -1

    @classmethod
    def from_tracker_results(
        cls,
        track_results: list[Any],  # list[TrackResult]
        state_snapshots: list[Any],  # list[TrackStateSnapshot]
        candidates: list[Any],  # list[TrackCandidateSnapshot]
        img_hw: tuple[int, int],
    ) -> "TrackerGateInput":
        """
        Convert raw tracker outputs into TrackerGateInput.

        state_snapshots is indexed by obj_id to extract Kalman velocity (state[4:6]).
        candidates are tentative (unconfirmed) tracks from get_tentative_candidates().
        """
        vel_lookup: dict[int, tuple[float, float]] = {
            s.obj_id: (float(s.state[4]), float(s.state[5])) for s in state_snapshots
        }

        boxes, scores, vels, occluded = [], [], [], []
        for tr in track_results:
            boxes.append([tr.x1, tr.y1, tr.x2, tr.y2])
            scores.append(float(tr.score))
            vx, vy = vel_lookup.get(tr.obj_id, (0.0, 0.0))
            vels.append([vx, vy])
            occluded.append(tr.det_idx == -1)

        t_boxes, t_ratios = [], []
        for c in candidates:
            t_boxes.append([c.x1, c.y1, c.x2, c.y2])
            t_ratios.append(c.hit_streak / max(c.required_confirm_streak, 1))

        confirmed_boxes = (
            torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4)
        )
        confirmed_scores = (
            torch.tensor(scores, dtype=torch.float32) if scores else torch.zeros(0)
        )
        velocities = torch.tensor(vels, dtype=torch.float32) if vels else None
        confirmed_occluded = (
            torch.tensor(occluded, dtype=torch.bool) if occluded else None
        )
        tentative_boxes = (
            torch.tensor(t_boxes, dtype=torch.float32) if t_boxes else None
        )
        tentative_ratios = (
            torch.tensor(t_ratios, dtype=torch.float32) if t_ratios else None
        )

        return cls(
            confirmed_boxes=confirmed_boxes,
            confirmed_scores=confirmed_scores,
            velocities=velocities,
            tentative_boxes=tentative_boxes,
            tentative_ratios=tentative_ratios,
            img_hw=img_hw,
            confirmed_occluded=confirmed_occluded,
        )

    @classmethod
    def from_boxes_scores(
        cls,
        boxes: Tensor,  # (N, 4) or (B, N, 4) — absolute xyxy or normalized cxcywh
        scores: Tensor | None,  # (N,) or (B, N)
        img_hw: tuple[int, int],
        assume_absolute: bool = False,
    ) -> "TrackerGateInput":
        """
        Convenience constructor for training shortcuts (no velocity / tentative info).

        Format auto-detection: if all box values ≤ 2.0 → normalized [cx,cy,w,h];
        otherwise absolute [x1,y1,x2,y2]. The threshold 2.0 works because real pixel
        coords for MOT17 at 640px are always >> 2.

        assume_absolute=True skips the float(boxes.max()) D2H sync — use when boxes
        are known to be in absolute pixel coords (e.g. model predictions on GPU).
        """
        if boxes.dim() == 3:
            boxes = boxes.reshape(-1, 4)
        if scores is not None and scores.dim() == 2:
            scores = scores.reshape(-1)

        # Detect normalized cxcywh → convert to absolute xyxy
        if not assume_absolute and boxes.numel() > 0 and float(boxes.max()) <= 2.0:
            H, W = img_hw
            cx, cy, w, h = boxes.unbind(-1)
            x1 = (cx - w * 0.5) * W
            y1 = (cy - h * 0.5) * H
            x2 = (cx + w * 0.5) * W
            y2 = (cy + h * 0.5) * H
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        # Ignore scores if count doesn't match boxes (happens in Phase 1 when GT boxes
        # are mixed with model query scores from the previous frame).
        if scores is not None and scores.shape[0] != boxes.shape[0]:
            scores = None
        if scores is None:
            scores = boxes.new_ones(boxes.shape[0])

        return cls(
            confirmed_boxes=boxes,
            confirmed_scores=scores,
            velocities=None,
            tentative_boxes=None,
            tentative_ratios=None,
            img_hw=img_hw,
            confirmed_occluded=None,
        )

    def to(self, device: torch.device | str) -> "TrackerGateInput":
        """Move all tensors to device (e.g. after dataclass construction on CPU)."""
        return TrackerGateInput(
            confirmed_boxes=self.confirmed_boxes.to(device),
            confirmed_scores=self.confirmed_scores.to(device),
            velocities=(
                self.velocities.to(device) if self.velocities is not None else None
            ),
            tentative_boxes=(
                self.tentative_boxes.to(device)
                if self.tentative_boxes is not None
                else None
            ),
            tentative_ratios=(
                self.tentative_ratios.to(device)
                if self.tentative_ratios is not None
                else None
            ),
            img_hw=self.img_hw,
            confirmed_occluded=(
                self.confirmed_occluded.to(device)
                if self.confirmed_occluded is not None
                else None
            ),
        )


# ---------------------------------------------------------------------------
# GaussianHeatmapRenderer
# ---------------------------------------------------------------------------
class GaussianHeatmapRenderer(nn.Module):
    """
    Renders tracker boxes as Gaussian heatmaps on a feature-map grid.

    Per confirmed track:
      1. Velocity prediction: center += velocity  (dt = 1 frame; None → no shift)
      2. sigma = max(box_w, box_h) * sigma_scale  (in pixels)
         occluded (det_idx=-1): sigma *= 1.5, intensity *= 0.5
      3. Gaussian rendered on the FPN grid (pixel-stride-normalized)
    All per-track heatmaps are max-merged → (1, H_s, W_s).

    Tentative tracks rendered separately with intensity = ratio * 0.3,
    then max-merged with the confirmed heatmap.

    Running under @torch.no_grad() because heatmap values are treated as constants
    during backprop; only the learnable alpha in TrackSpatialGate accumulates grad.
    """

    def __init__(self, sigma_scale: float = 0.5, min_score: float = 0.0):
        super().__init__()
        self.sigma_scale = sigma_scale
        self.min_score = min_score

    def _render_gaussians(
        self,
        boxes: Tensor,  # (N, 4) [x1,y1,x2,y2] absolute pixels
        intensities: Tensor,  # (N,) pre-computed ∈ (0, 1]
        img_hw: tuple[int, int],
        feat_hw: tuple[int, int],
        velocities: Tensor | None = None,  # (N, 2) [vx, vy]
        sigma_multiplier: Tensor | None = None,  # (N,) per-track sigma scale
    ) -> Tensor:  # (1, H_s, W_s)
        """Vectorized Gaussian rendering. Returns zero map when N == 0."""
        H_img, W_img = img_hw
        H_s, W_s = feat_hw
        device = boxes.device
        N = boxes.shape[0]

        if N == 0:
            return boxes.new_zeros(1, H_s, W_s)

        # Predicted box centers in pixels
        cx = (boxes[:, 0] + boxes[:, 2]) * 0.5  # (N,)
        cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
        if velocities is not None:
            cx = cx + velocities[:, 0]
            cy = cy + velocities[:, 1]

        # Sigma in pixels (at least 4px to avoid degenerate Gaussians)
        bw = boxes[:, 2] - boxes[:, 0]
        bh = boxes[:, 3] - boxes[:, 1]
        sigma_px = torch.maximum(bw, bh).clamp(min=4.0) * self.sigma_scale  # (N,)
        if sigma_multiplier is not None:
            sigma_px = sigma_px * sigma_multiplier

        # Feature-map cell size in pixels
        stride_y = H_img / H_s
        stride_x = W_img / W_s

        # Grid cell centers in pixels: (H_s,), (W_s,)
        gy = (
            torch.arange(H_s, device=device, dtype=torch.float32) * stride_y
            + stride_y * 0.5
        )
        gx = (
            torch.arange(W_s, device=device, dtype=torch.float32) * stride_x
            + stride_x * 0.5
        )

        # Sigma in pixels (grid centers gy/gx are also in pixels → units match)
        sg_y = sigma_px.clamp(min=1e-4).view(N, 1, 1)
        sg_x = sigma_px.clamp(min=1e-4).view(N, 1, 1)

        # (N, H_s, W_s)
        dy = (gy.view(1, H_s, 1) - cy.view(N, 1, 1)) / sg_y
        dx = (gx.view(1, 1, W_s) - cx.view(N, 1, 1)) / sg_x
        gauss = torch.exp(-0.5 * (dy**2 + dx**2)) * intensities.view(N, 1, 1)

        return gauss.max(dim=0, keepdim=True).values  # (1, H_s, W_s)

    @torch.no_grad()
    def forward(
        self,
        gate_input: TrackerGateInput,
        feat_hw: tuple[int, int],
    ) -> Tensor:  # (1, H_s, W_s) ∈ [0, 1]
        device = gate_input.confirmed_boxes.device

        boxes = gate_input.confirmed_boxes
        scores = gate_input.confirmed_scores
        vels = gate_input.velocities
        occ = gate_input.confirmed_occluded

        # Apply min_score filter
        if self.min_score > 0.0 and boxes.shape[0] > 0:
            keep = scores >= self.min_score
            boxes = boxes[keep]
            scores = scores[keep]
            vels = vels[keep] if vels is not None else None
            occ = occ[keep] if occ is not None else None

        # intensity = sigmoid(score), halved for occluded tracks
        intensity = torch.sigmoid(scores)
        sigma_mult = None
        if occ is not None and occ.any():
            occ_f = occ.float()
            intensity = intensity * (1.0 - 0.5 * occ_f)
            sigma_mult = 1.0 + 0.5 * occ_f  # 1.0 normal, 1.5 occluded

        heatmap = self._render_gaussians(
            boxes, intensity, gate_input.img_hw, feat_hw, vels, sigma_mult
        )

        # Tentative tracks: weaker signal, no velocity, no occlusion handling
        if (
            gate_input.tentative_boxes is not None
            and gate_input.tentative_ratios is not None
        ):
            t_intensity = gate_input.tentative_ratios.to(device) * 0.3
            t_heatmap = self._render_gaussians(
                gate_input.tentative_boxes.to(device),
                t_intensity,
                gate_input.img_hw,
                feat_hw,
            )
            heatmap = torch.maximum(heatmap, t_heatmap)

        return heatmap  # (1, H_s, W_s)

    @torch.no_grad()
    def batch_forward(
        self,
        gate_inputs: list[TrackerGateInput],
        feat_hw: tuple[int, int],
    ) -> Tensor:  # (B, H_s, W_s)
        """
        Batched renderer: all B samples in one vectorized GPU pass.
        Replaces B sequential forward() calls — ~B× fewer kernel launches.
        Falls back to forward() per-sample only when tentative tracks are present.
        """
        B = len(gate_inputs)
        H_s, W_s = feat_hw
        device = gate_inputs[0].confirmed_boxes.device
        H_img, W_img = gate_inputs[0].img_hw

        seg_boxes: list[Tensor] = []
        seg_intens: list[Tensor] = []
        seg_bidx: list[Tensor] = []
        seg_smult: list[Tensor] = []
        seg_vels: list[Tensor] = []
        has_vels = False

        for b, gi in enumerate(gate_inputs):
            boxes, scores = gi.confirmed_boxes, gi.confirmed_scores
            occ, vels = gi.confirmed_occluded, gi.velocities

            if self.min_score > 0.0 and boxes.shape[0] > 0:
                keep = scores >= self.min_score
                boxes, scores = boxes[keep], scores[keep]
                occ = occ[keep] if occ is not None else None
                vels = vels[keep] if vels is not None else None

            if boxes.shape[0] == 0:
                continue

            N_b = boxes.shape[0]
            intensity = torch.sigmoid(scores)
            smult = boxes.new_ones(N_b)
            if occ is not None and occ.any():
                occ_f = occ.float()
                intensity = intensity * (1.0 - 0.5 * occ_f)
                smult = 1.0 + 0.5 * occ_f

            seg_boxes.append(boxes)
            seg_intens.append(intensity)
            seg_bidx.append(torch.full((N_b,), b, device=device, dtype=torch.long))
            seg_smult.append(smult)
            seg_vels.append(vels if vels is not None else boxes.new_zeros(N_b, 2))
            if vels is not None:
                has_vels = True

        out = torch.zeros(B, H_s, W_s, device=device)

        if seg_boxes:
            boxes_c = torch.cat(seg_boxes)  # (sum_N, 4)
            intens_c = torch.cat(seg_intens)  # (sum_N,)
            bidx_c = torch.cat(seg_bidx)  # (sum_N,)
            smult_c = torch.cat(seg_smult)  # (sum_N,)
            vels_c = torch.cat(seg_vels)  # (sum_N, 2)
            sum_N = boxes_c.shape[0]

            cx = (boxes_c[:, 0] + boxes_c[:, 2]) * 0.5
            cy = (boxes_c[:, 1] + boxes_c[:, 3]) * 0.5
            if has_vels:
                cx = cx + vels_c[:, 0]
                cy = cy + vels_c[:, 1]

            bw = boxes_c[:, 2] - boxes_c[:, 0]
            bh = boxes_c[:, 3] - boxes_c[:, 1]
            sigma_px = torch.maximum(bw, bh).clamp(min=4.0) * self.sigma_scale * smult_c

            stride_y = H_img / H_s
            stride_x = W_img / W_s
            gy = (
                torch.arange(H_s, device=device, dtype=torch.float32) * stride_y
                + stride_y * 0.5
            )
            gx = (
                torch.arange(W_s, device=device, dtype=torch.float32) * stride_x
                + stride_x * 0.5
            )

            sg = sigma_px.clamp(min=1e-4).view(sum_N, 1, 1)
            dy = (gy.view(1, H_s, 1) - cy.view(sum_N, 1, 1)) / sg  # (sum_N, H_s, W_s)
            dx = (gx.view(1, 1, W_s) - cx.view(sum_N, 1, 1)) / sg
            gauss = torch.exp(-0.5 * (dy**2 + dx**2)) * intens_c.view(sum_N, 1, 1)

            bidx_exp = bidx_c.view(sum_N, 1, 1).expand(sum_N, H_s, W_s)
            out.scatter_reduce_(0, bidx_exp, gauss, reduce="amax", include_self=True)

        # Tentative tracks: second batched pass, max-merged into out
        t_seg_boxes: list[Tensor] = []
        t_seg_intens: list[Tensor] = []
        t_seg_bidx: list[Tensor] = []
        for b, gi in enumerate(gate_inputs):
            if gi.tentative_boxes is None or gi.tentative_ratios is None:
                continue
            tb = gi.tentative_boxes.to(device)
            ti = gi.tentative_ratios.to(device) * 0.3
            if tb.shape[0] == 0:
                continue
            t_seg_boxes.append(tb)
            t_seg_intens.append(ti)
            t_seg_bidx.append(
                torch.full((tb.shape[0],), b, device=device, dtype=torch.long)
            )

        if t_seg_boxes:
            tb_c = torch.cat(t_seg_boxes)
            ti_c = torch.cat(t_seg_intens)
            tb_bidx = torch.cat(t_seg_bidx)
            sum_T = tb_c.shape[0]

            cx_t = (tb_c[:, 0] + tb_c[:, 2]) * 0.5
            cy_t = (tb_c[:, 1] + tb_c[:, 3]) * 0.5
            bw_t = tb_c[:, 2] - tb_c[:, 0]
            bh_t = tb_c[:, 3] - tb_c[:, 1]
            sp_t = torch.maximum(bw_t, bh_t).clamp(min=4.0) * self.sigma_scale

            gy = (
                torch.arange(H_s, device=device, dtype=torch.float32) * (H_img / H_s)
                + (H_img / H_s) * 0.5
            )
            gx = (
                torch.arange(W_s, device=device, dtype=torch.float32) * (W_img / W_s)
                + (W_img / W_s) * 0.5
            )
            sg_t = sp_t.clamp(min=1e-4).view(sum_T, 1, 1)
            dy_t = (gy.view(1, H_s, 1) - cy_t.view(sum_T, 1, 1)) / sg_t
            dx_t = (gx.view(1, 1, W_s) - cx_t.view(sum_T, 1, 1)) / sg_t
            g_t = torch.exp(-0.5 * (dy_t**2 + dx_t**2)) * ti_c.view(sum_T, 1, 1)

            t_bidx_exp = tb_bidx.view(sum_T, 1, 1).expand(sum_T, H_s, W_s)
            t_out = torch.zeros(B, H_s, W_s, device=device)
            t_out.scatter_reduce_(0, t_bidx_exp, g_t, reduce="amax", include_self=True)
            out = torch.maximum(out, t_out)

        return out  # (B, H_s, W_s)


# ---------------------------------------------------------------------------
# TrackSpatialGate
# ---------------------------------------------------------------------------
class TrackSpatialGate(nn.Module):
    """
    Per-scale multiplicative gate: gate_s = 1 + alpha_s * heatmap_s.

    alpha initialized to 0 → gate = 1 everywhere → exact Option C behavior at epoch 0.
    As training proceeds, alpha learns to amplify FPN features in tracked regions.
    """

    def __init__(
        self,
        scales: tuple[str, ...],
        sigma_scale: float = 0.5,
        min_score: float = 0.0,
    ):
        super().__init__()
        self.scales = list(scales)
        self.renderer = GaussianHeatmapRenderer(sigma_scale, min_score)
        # ParameterDict preserves scale names; all initialized to 0
        self.alphas = nn.ParameterDict(
            {s: nn.Parameter(torch.zeros(1)) for s in scales}
        )

    def forward(
        self,
        gate_inputs: TrackerGateInput | list["TrackerGateInput"],
        feat_hws: dict[str, tuple[int, int]],  # {scale: (H_s, W_s)}
    ) -> dict[str, Tensor]:
        """
        Single TrackerGateInput  → (1, 1, H_s, W_s) — broadcasts across batch (inference).
        list[TrackerGateInput]   → (B, 1, H_s, W_s) — per-item gates (training).
        """
        if isinstance(gate_inputs, list):
            result: dict[str, Tensor] = {}
            for scale in self.scales:
                hw = feat_hws[scale]
                maps = self.renderer.batch_forward(gate_inputs, hw)  # (B, H_s, W_s)
                result[scale] = 1.0 + self.alphas[scale] * maps.unsqueeze(
                    1
                )  # (B, 1, H_s, W_s)
            return result
        else:
            result = {}
            for scale in self.scales:
                heatmap = self.renderer(gate_inputs, feat_hws[scale])  # (1, H_s, W_s)
                gate = 1.0 + self.alphas[scale] * heatmap
                result[scale] = gate.unsqueeze(0)  # (1, 1, H_s, W_s)
            return result


# ---------------------------------------------------------------------------
# ConditionedConfig
# ---------------------------------------------------------------------------
@dataclass
class ConditionedConfig:
    """Configuration for TemporalYOLOConditioned (Option D)."""

    # Decoder hyperparams (same defaults as JointConfig)
    embed_dim: int = 256
    num_heads: int = 8
    num_queries: int = 100
    ffn_dim: int = 1024
    dropout: float = 0.1
    num_decoder_layers: int = 3
    score_threshold: float = 0.3
    # FPN scales to hook: any subset of ("p3", "p4", "p5")
    scales: tuple[str, ...] = ("p5",)
    # Gate hyperparams
    gate_sigma_scale: float = 0.5  # sigma = max(box_w, box_h) * gate_sigma_scale
    gate_min_score: float = 0.5  # skip tracks below this score
    # Freeze flags for Phase 1 training
    freeze_backbone: bool = False
    freeze_decoder: bool = False

    def to_temporal_config(self) -> TemporalYOLOConfig:
        return TemporalYOLOConfig(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_queries=self.num_queries,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout,
            num_decoder_layers=self.num_decoder_layers,
            score_threshold=self.score_threshold,
        )


# ---------------------------------------------------------------------------
# TemporalYOLOConditioned
# ---------------------------------------------------------------------------
class TemporalYOLOConditioned(nn.Module):
    """
    Option D: TemporalYOLOJoint + TrackSpatialGate on FPN outputs.

    Diagram:
        frame_t ──→ YOLOFeaturePyramid ──→ P3/P4/P5
                                               │
              TrackerGateInput ──→ TrackSpatialGate (gate = 1 + alpha*heatmap)
              (prev-frame tracks)              │
                                   P_s_gated ──→ FPNSequenceProjection
                                                         │
                                                 TrackQueryDecoder
                                                         │
                                                 box / score heads
    """

    def __init__(self, cfg: ConditionedConfig, yolo_pt_path: str):
        super().__init__()
        self.cfg = cfg
        tc = cfg.to_temporal_config()

        # YOLO backbone + FPN hooks
        self.pyramid = YOLOFeaturePyramid(
            yolo_pt_path, scales=cfg.scales, freeze=cfg.freeze_backbone
        )
        self._probe_channels()

        # Spatial gate (new in Option D; starts as identity because alpha = 0)
        self.gate = TrackSpatialGate(
            scales=cfg.scales,
            sigma_scale=cfg.gate_sigma_scale,
            min_score=cfg.gate_min_score,
        )

        # FPN → flat token sequence
        self.fpn_proj = FPNSequenceProjection(
            feat_channels=self.pyramid.feat_channels,
            embed_dim=cfg.embed_dim,
            scales=cfg.scales,
        )

        # Learnable query positional encoding
        self.query_pos_embed = nn.Embedding(cfg.num_queries, cfg.embed_dim)

        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList(
            [TrackQueryDecoderLayer(tc) for _ in range(cfg.num_decoder_layers)]
        )

        # Detection heads
        D = cfg.embed_dim
        self.box_head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 4))
        self.score_head = nn.Linear(D, 1)

        # Track query lifecycle (defined before freeze so freeze can reach it)
        self.newborn_embed = nn.Embedding(cfg.num_queries, cfg.embed_dim)
        nn.init.normal_(self.newborn_embed.weight, std=0.02)
        self.__lifecycle: TrackQueryLifecycle | None = None

        if cfg.freeze_decoder:
            for p in (
                *self.fpn_proj.parameters(),
                *self.decoder_layers.parameters(),
                *self.box_head.parameters(),
                *self.score_head.parameters(),
                *self.query_pos_embed.parameters(),
                *self.newborn_embed.parameters(),
            ):
                p.requires_grad_(False)
            print("[TemporalYOLOConditioned] Decoder+FPN FROZEN")

    # ------------------------------------------------------------------
    def _probe_channels(self) -> None:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 640, 640)
            self.pyramid(dummy)

    def _get_lifecycle(self) -> TrackQueryLifecycle:
        if self.__lifecycle is None:
            self.__lifecycle = TrackQueryLifecycle(
                self.cfg.to_temporal_config(), self.newborn_embed
            )
        return self.__lifecycle

    def _init_queries(self, B: int, device: torch.device) -> Tensor:
        slots = torch.arange(self.cfg.num_queries, device=device)
        return self.newborn_embed(slots).unsqueeze(0).expand(B, -1, -1).clone()

    # ------------------------------------------------------------------
    def reset_sequence(self) -> None:
        """Call at the start of each new video sequence to clear lifecycle state."""
        if self.__lifecycle is not None:
            self.__lifecycle.reset()
        self.__lifecycle = None

    def forward_gate_only(
        self,
        frame: Tensor,  # (B, 3, H, W)
        gate_inputs: list[TrackerGateInput],  # len = B
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, tuple[int, int]]]:
        """
        Phase 1 proxy forward: run backbone (no_grad) + gate only, skip decoder.

        Returns (gated_feats, gate_maps, feat_hws) where gated_feats[s] =
        detached_fpn[s] * gate_maps[s]. Gradient flows only through gate_maps → alpha,
        making backward ~3× cheaper than full forward+backward.
        """
        with torch.no_grad():
            fpn_feats = self.pyramid(frame)
        feat_hws = {
            s: (fpn_feats[s].shape[2], fpn_feats[s].shape[3]) for s in self.cfg.scales
        }
        gate_maps = self.gate(gate_inputs, feat_hws)  # grad flows to alpha
        gated = {s: fpn_feats[s].detach() * gate_maps[s] for s in self.cfg.scales}
        return gated, gate_maps, feat_hws

    def parameter_groups(
        self,
        lr_backbone: float,
        lr_gate: float,
        lr_decoder: float,
    ) -> list[dict]:
        """
        Three differential-LR parameter groups.
        Only groups with at least one trainable parameter are returned.
        Phase 1 callers filter to {"name": "gate"} before building the optimizer.
        """
        backbone_ids = {id(p) for p in self.pyramid.parameters()}
        gate_ids = {id(p) for p in self.gate.parameters()}

        backbone = [p for p in self.pyramid.parameters() if p.requires_grad]
        gate = [p for p in self.gate.parameters() if p.requires_grad]
        decoder = [
            p
            for p in self.parameters()
            if p.requires_grad and id(p) not in backbone_ids and id(p) not in gate_ids
        ]

        groups = []
        if backbone:
            n = sum(p.numel() for p in backbone)
            print(f"[ConditionedModel] backbone  {n:,}  lr={lr_backbone}")
            groups.append({"params": backbone, "lr": lr_backbone, "name": "backbone"})
        if gate:
            n = sum(p.numel() for p in gate)
            print(f"[ConditionedModel] gate      {n:,}  lr={lr_gate}")
            groups.append({"params": gate, "lr": lr_gate, "name": "gate"})
        if decoder:
            n = sum(p.numel() for p in decoder)
            print(f"[ConditionedModel] decoder   {n:,}  lr={lr_decoder}")
            groups.append({"params": decoder, "lr": lr_decoder, "name": "decoder"})
        return groups

    def forward(
        self,
        frame: Tensor,  # (B, 3, H, W)
        prev_queries: Tensor | None = None,  # (B, N_q, D) or None
        gate_input: TrackerGateInput
        | None = None,  # inference: single gate (broadcasts)
        gate_inputs: list[TrackerGateInput]
        | None = None,  # training: per-item list (len=B)
        prev_boxes: Tensor | None = None,  # legacy shortcut: (N,4) or (B,N,4)
        prev_scores: Tensor | None = None,
    ) -> dict:
        """
        Priority of gate sources: gate_inputs > gate_input > prev_boxes.

        gate_inputs (list, training): per-item TrackerGateInput → (B,1,H,W) gates.
        gate_input  (single, infer) : one TrackerGateInput      → (1,1,H,W) broadcast.
        prev_boxes  (legacy)        : simplified shortcut, kept for compatibility.

        Returns: boxes (B,N_q,4 normalized cxcywh), scores (B,N_q),
                 updated_queries (B,N_q,D), query_ids.
        """
        B, _, H, W = frame.shape
        lc = self._get_lifecycle()

        if prev_queries is None:
            prev_queries = self._init_queries(B, frame.device)

        # 1. YOLO FPN features
        fpn_feats = self.pyramid(frame)  # {scale: (B, C_s, H_s, W_s)}

        # 2. Resolve gate source (priority: list > single > prev_boxes legacy)
        effective_gate: TrackerGateInput | list[TrackerGateInput] | None = (
            gate_inputs if gate_inputs is not None else gate_input
        )
        if effective_gate is None and prev_boxes is not None and prev_boxes.numel() > 0:
            effective_gate = TrackerGateInput.from_boxes_scores(
                prev_boxes, prev_scores, (H, W)
            ).to(frame.device)

        # 3. Apply spatial gate (no-op when alpha=0 or no gate input)
        if effective_gate is not None:
            feat_hws = {
                s: (fpn_feats[s].shape[2], fpn_feats[s].shape[3])
                for s in self.cfg.scales
            }
            gate_maps = self.gate(effective_gate, feat_hws)  # {scale:(1/B,1,H_s,W_s)}
            for s in self.cfg.scales:
                fpn_feats[s] = fpn_feats[s] * gate_maps[s]

        # 4. FPN → flat token sequence
        feat_seq, feat_pos = self.fpn_proj(fpn_feats)  # (B, S, D)

        # 5. Query positional encoding
        slots = torch.arange(self.cfg.num_queries, device=frame.device)
        query_pos = self.query_pos_embed(slots).unsqueeze(0).expand(B, -1, -1)

        # 6. Transformer decoder
        queries = prev_queries
        for layer in self.decoder_layers:
            queries = layer(queries, feat_seq, query_pos, feat_pos)

        # 7. Detection heads
        boxes = torch.sigmoid(self.box_head(queries))  # (B, N_q, 4) normalized cxcywh
        scores = self.score_head(queries).squeeze(-1)  # (B, N_q)

        # 8. Lifecycle
        updated_queries, query_ids = lc.update(queries, scores)

        return {
            "boxes": boxes,
            "scores": scores,
            "updated_queries": updated_queries,
            "query_ids": query_ids,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_temporal_yolo_conditioned(
    yolo_pt_path: str,
    cfg: ConditionedConfig | None = None,
    device: str | torch.device = "cuda",
    weights_path: str = "",
) -> TemporalYOLOConditioned:
    """
    Build TemporalYOLOConditioned, optionally loading a checkpoint.

    For Phase 1 (resume from Option C JointConfig checkpoint), pass the
    checkpoint path and leave strict=False in load_state_dict — backbone+decoder
    weights load, gate.alphas stay at 0.
    """
    if cfg is None:
        cfg = ConditionedConfig()
    model = TemporalYOLOConditioned(cfg, yolo_pt_path=yolo_pt_path)
    if weights_path:
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        n_gate = len([k for k in missing if "gate" in k])
        if missing:
            print(
                f"[build_conditioned] Missing {len(missing)} keys "
                f"({n_gate} gate.alphas — expected for Phase 1 warm start)"
            )
        if unexpected:
            print(f"[build_conditioned] Unexpected {len(unexpected)} keys")
    return model.to(device)
