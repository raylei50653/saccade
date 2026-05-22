"""
GatedYOLODetector — Option D revised: TrackSpatialGate injected into YOLO via hooks.

Architecture:
    YOLO26s backbone (layers 0-22)
        ↓  hooks modify layer 16 (P3), 19 (P4), 22 (P5) outputs
    TrackSpatialGate  (gate = 1 + alpha * heatmap; alpha init=0)
        ↓  gated FPN features passed to Detect head
    TemporalFeatureFusion  (optional, Option E-v2)
        ↓  P_fused = P_gated + α × Q × warp(P_prev)
    YOLO Detect Head (layer 23, pre-trained)
        ↓
    Standard YOLO detection output
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any

from .yolo_conditioned import TrackerGateInput, TrackSpatialGate
from .temporal_fusion import TemporalFeatureFusion, _compute_alpha_tier, AlphaTierConfig


_GATE_LAYER_IDX: dict[str, int] = {"p3": 16, "p4": 19, "p5": 22}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class GatedDetConfig:
    scales: tuple[str, ...] = ("p3", "p4", "p5")
    gate_sigma_scale: float = 0.5
    gate_min_score: float = 0.5
    freeze_backbone: bool = False
    img_size: int = 640

    # Option E-v2 temporal fusion
    enable_temporal_fusion: bool = False
    fusion_alpha: float = 0.0
    fusion_fixed_alpha: bool = True

    # Phase 3 α_tier
    alpha_tier: AlphaTierConfig | None = None

    # Phase 3 detector heatmap
    enable_detector_heatmap: bool = False


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class GatedYOLODetector(nn.Module):
    """YOLO26s with TrackSpatialGate injected at P3/P4/P5 via forward hooks.

    Option E-v2: enables TemporalFeatureFusion for cross-frame feature reuse.
    """

    def __init__(
        self,
        yolo_pt_path: str,
        cfg: GatedDetConfig | None = None,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        if cfg is None:
            cfg = GatedDetConfig()
        self.cfg = cfg

        from ultralytics import YOLO as _YOLO  # type: ignore[attr-defined]

        yolo = _YOLO(yolo_pt_path)
        self.yolo_model: Any = yolo.model.to(device)  # type: ignore[union-attr]

        for p in self.yolo_model.parameters():
            p.requires_grad_(not cfg.freeze_backbone)

        _ = self._probe_feat_channels(device)

        self.gate = TrackSpatialGate(
            scales=tuple(cfg.scales),
            sigma_scale=cfg.gate_sigma_scale,
            min_score=cfg.gate_min_score,
        )
        for p in self.gate.parameters():
            p.requires_grad_(True)

        # Temporal fusion (Option E-v2)
        self.fusion: TemporalFeatureFusion | None = None
        if cfg.enable_temporal_fusion:
            tier_cfg = cfg.alpha_tier or AlphaTierConfig()
            self.fusion = TemporalFeatureFusion(
                scales=tuple(cfg.scales),
                img_size=cfg.img_size,
                tier_cfg=tier_cfg,
            )
            if cfg.fusion_fixed_alpha:
                self.fusion.set_fixed_alpha(cfg.fusion_alpha)
            for p in self.fusion.parameters():
                p.requires_grad_(True)

        self._current_gate: TrackerGateInput | list[TrackerGateInput] | None = None
        self._feat_cache: dict[str, torch.Tensor] = {}
        self.cache_feats: bool = False

        # Phase 3: prev-frame raw detections for detector heatmap
        self._det_boxes_prev: torch.Tensor | None = None
        self._det_scores_prev: torch.Tensor | None = None

        for scale in cfg.scales:
            idx = _GATE_LAYER_IDX[scale]
            self.yolo_model.model[idx].register_forward_hook(self._make_hook(scale))

    # ------------------------------------------------------------------
    # Detector heatmap support
    # ------------------------------------------------------------------
    def set_prev_detections(
        self, boxes: torch.Tensor | None, scores: torch.Tensor | None
    ) -> None:
        """Set prev-frame raw detections for detector score heatmap (Phase 3)."""
        self._det_boxes_prev = boxes
        self._det_scores_prev = scores

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _probe_feat_channels(self, device: str | torch.device) -> dict[str, int]:
        channels: dict[str, int] = {}
        tmp_hooks = []
        for scale in self.cfg.scales:
            idx = _GATE_LAYER_IDX[scale]

            def _h(m: nn.Module, i: Any, o: torch.Tensor, s: str = scale) -> None:
                channels[s] = o.shape[1]

            tmp_hooks.append(self.yolo_model.model[idx].register_forward_hook(_h))
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 640, 640, device=device)
            self.yolo_model(dummy)
        for h in tmp_hooks:
            h.remove()
        return channels

    def _render_detector_heatmap(
        self, scale: str, hw: tuple[int, int]
    ) -> torch.Tensor | None:
        if not self.cfg.enable_detector_heatmap:
            return None
        if self._det_boxes_prev is None or self._det_boxes_prev.numel() == 0:
            return None

        renderer = self.gate.renderer
        heatmap = renderer._render_gaussians(
            self._det_boxes_prev,
            torch.sigmoid(self._det_scores_prev)
            if self._det_scores_prev is not None
            else torch.ones(
                self._det_boxes_prev.shape[0], device=self._det_boxes_prev.device
            ),
            (self.cfg.img_size, self.cfg.img_size),
            hw,
            velocities=None,
            sigma_multiplier=None,
            per_track_weights=None,
        )
        return heatmap  # (1, H_s, W_s)

    def _make_hook(self, scale: str):  # type: ignore[no-untyped-def]
        def _hook(module: nn.Module, inp: Any, output: torch.Tensor) -> torch.Tensor:
            if self.cache_feats or self.fusion is not None:
                self._feat_cache[scale] = output

            if self._current_gate is None:
                return output

            hw = (output.shape[2], output.shape[3])
            renderer = self.gate.renderer
            alpha = self.gate.alphas[scale]

            if isinstance(self._current_gate, list):
                maps = renderer.batch_forward(self._current_gate, hw)
                gate_map = 1.0 + alpha * maps.unsqueeze(1)
            else:
                heatmap = renderer(self._current_gate, hw)
                gate_map = (1.0 + alpha * heatmap).unsqueeze(0)

            gated = output * gate_map

            if self.fusion is not None:
                gi = self._current_gate
                if isinstance(gi, list):
                    q = maps.unsqueeze(1)
                else:
                    if gi.confirmed_ages is not None:
                        confirmed_w, _ = _compute_alpha_tier(
                            gi.confirmed_ages,
                            gi.confirmed_occluded,
                            tier_cfg=self.fusion.tier_cfg,
                        )
                        q_heatmap = renderer(
                            gi,
                            hw,
                            per_track_weights=confirmed_w,
                        )
                    else:
                        q_heatmap = heatmap

                    # Phase 3: merge detector heatmap (max union)
                    det_hmap = self._render_detector_heatmap(scale, hw)
                    if det_hmap is not None:
                        q_heatmap = torch.maximum(q_heatmap, det_hmap)

                    q = q_heatmap.unsqueeze(0)

                gated = self.fusion.fuse(scale, gated, q)

            return gated  # type: ignore[no-any-return]

        return _hook

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        frame: torch.Tensor,
        gate_input: TrackerGateInput | list[TrackerGateInput] | None = None,
    ) -> dict[str, Any] | tuple[Any, ...]:
        self._current_gate = gate_input
        result = self.yolo_model(frame)
        self._current_gate = None

        if self.fusion is not None and self._feat_cache:
            self.fusion.update_prev(self._feat_cache)

        return result  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Fusion control
    # ------------------------------------------------------------------
    def set_fusion_alpha(self, alpha: float | None) -> None:
        if self.fusion is not None:
            self.fusion.set_fixed_alpha(alpha)

    def set_gmc(self, matrix: torch.Tensor | None) -> None:
        if self.fusion is not None:
            self.fusion.set_gmc(matrix)

    def reset_fusion(self) -> None:
        if self.fusion is not None:
            self.fusion.reset()

    # ------------------------------------------------------------------
    # Optimizer helper
    # ------------------------------------------------------------------
    def parameter_groups(
        self,
        lr_gate: float,
        lr_yolo: float = 0.0,
    ) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = [
            {"params": list(self.gate.parameters()), "name": "gate", "lr": lr_gate},
        ]
        if self.fusion is not None:
            groups.append(
                {
                    "params": list(self.fusion.parameters()),
                    "name": "fusion",
                    "lr": lr_gate,
                }
            )
        if lr_yolo > 0.0:
            trainable = [p for p in self.yolo_model.parameters() if p.requires_grad]
            if trainable:
                groups.append({"params": trainable, "name": "yolo", "lr": lr_yolo})
        return groups

    def alpha_summary(self) -> str:
        parts = [f"{s}={self.gate.alphas[s].item():.4f}" for s in self.cfg.scales]
        if self.fusion is not None:
            parts.append(f"fusion=[{self.fusion.alpha_summary()}]")
        return "  ".join(parts)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_gated_yolo_detector(
    yolo_pt_path: str,
    cfg: GatedDetConfig | None = None,
    device: str | torch.device = "cuda",
    weights_path: str = "",
) -> GatedYOLODetector:
    model = GatedYOLODetector(yolo_pt_path, cfg=cfg, device=device)
    if weights_path:
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        sd = state["model"] if "model" in state else state
        sd = {k.replace("._orig_mod.", "."): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[GatedDet] Missing {len(missing)} keys")
        if unexpected:
            print(f"[GatedDet] Unexpected {len(unexpected)} keys")
    return model.to(device)
