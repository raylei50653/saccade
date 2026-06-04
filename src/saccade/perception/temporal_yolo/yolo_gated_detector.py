"""
GatedYOLODetector — TrackSpatialGate + TemporalFeatureFusion via inline forward patch.

Architecture:
    YOLO26s backbone (layers 0-22)
        ↓  patched forward at layer 16 (P3), 19 (P4), 22 (P5)
    TrackSpatialGate  (gate = 1 + alpha * heatmap; alpha init=0)
        ↓  gated FPN features passed to Detect head
    TemporalFeatureFusion  (optional, Option E-v2)
        ↓  P_fused = P_gated + alpha * Q * warp(P_prev)
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

    enable_temporal_fusion: bool = False
    fusion_alpha: float = 0.0
    fusion_fixed_alpha: bool = True
    alpha_tier: AlphaTierConfig | None = None
    enable_detector_heatmap: bool = False


# ---------------------------------------------------------------------------
# Gate layer (replaces forward hook, runs inline with zero roundtrip)
# ---------------------------------------------------------------------------
class GateLayer(nn.Module):
    """Holds gate/fusion state and applies gate after a YOLO layer's output."""

    gate_module: TrackSpatialGate
    fusion_module: TemporalFeatureFusion | None

    def __init__(
        self,
        scale: str,
        gate: TrackSpatialGate,
        fusion: TemporalFeatureFusion | None,
        img_size: int,
    ):
        super().__init__()
        self.scale = scale
        self.img_size = img_size
        object.__setattr__(self, "gate_module", gate)
        object.__setattr__(self, "fusion_module", fusion)

        self._gate_input: TrackerGateInput | list[TrackerGateInput] | None = None
        self._feat_cache: dict[str, torch.Tensor] = {}
        self._cache_feats: bool = False
        self._det_boxes_prev: torch.Tensor | None = None
        self._det_scores_prev: torch.Tensor | None = None
        self._enable_detector_heatmap: bool = False

    def _render_detector_heatmap(self, hw: tuple[int, int]) -> torch.Tensor | None:
        if not self._enable_detector_heatmap:
            return None
        if self._det_boxes_prev is None or self._det_boxes_prev.numel() == 0:
            return None
        renderer = self.gate_module.renderer
        heatmap = renderer._render_gaussians(
            self._det_boxes_prev,
            torch.sigmoid(self._det_scores_prev)
            if self._det_scores_prev is not None
            else torch.ones(
                self._det_boxes_prev.shape[0], device=self._det_boxes_prev.device
            ),
            (self.img_size, self.img_size),
            hw,
            velocities=None,
            sigma_multiplier=None,
            per_track_weights=None,
        )
        return heatmap

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        if self._cache_feats or self.fusion_module is not None:
            self._feat_cache[self.scale] = output

        gate_input = self._gate_input
        if gate_input is None:
            return output

        hw = (output.shape[2], output.shape[3])
        renderer = self.gate_module.renderer
        alpha = self.gate_module.alphas[self.scale]

        if isinstance(gate_input, list):
            maps = renderer.batch_forward(gate_input, hw)
            gate_map = 1.0 + alpha * maps.unsqueeze(1)
        else:
            heatmap = renderer(gate_input, hw)
            gate_map = (1.0 + alpha * heatmap).unsqueeze(0)

        gated = output * gate_map

        if self.fusion_module is not None:
            gi = gate_input
            if isinstance(gi, list):
                q = maps.unsqueeze(1)
            else:
                if gi.confirmed_ages is not None:
                    confirmed_w, _ = _compute_alpha_tier(
                        gi.confirmed_ages,
                        gi.confirmed_occluded,
                        tier_cfg=self.fusion_module.tier_cfg,
                    )
                    q_heatmap = renderer(gi, hw, per_track_weights=confirmed_w)
                else:
                    q_heatmap = heatmap

                det_hmap = self._render_detector_heatmap(hw)
                if det_hmap is not None:
                    q_heatmap = torch.maximum(q_heatmap, det_hmap)
                q = q_heatmap.unsqueeze(0)

            gated = self.fusion_module.fuse(self.scale, gated, q)

        return gated  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class GatedYOLODetector(nn.Module):
    """YOLO26s with TrackSpatialGate injected inline via forward patching.

    Patches the forward method of YOLO layers 16/19/22 to append gate logic,
    avoiding PyTorch's forward-hook Python callback overhead (~5 ms/frame).
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

        self._gate_layers: dict[str, GateLayer] = {}
        for scale in cfg.scales:
            idx = _GATE_LAYER_IDX[scale]
            gl = GateLayer(scale, self.gate, self.fusion, cfg.img_size)
            gl._enable_detector_heatmap = cfg.enable_detector_heatmap
            self._gate_layers[scale] = gl
            _inject_gate(self.yolo_model.model[idx], gl)

    # ------------------------------------------------------------------
    # Feature cache access (for ROI ReID)
    # ------------------------------------------------------------------
    @property
    def _feat_cache(self) -> dict[str, torch.Tensor]:
        return self._gate_layers[list(self._gate_layers)[0]]._feat_cache

    @property
    def cache_feats(self) -> bool:
        return self._gate_layers[list(self._gate_layers)[0]]._cache_feats

    @cache_feats.setter
    def cache_feats(self, v: bool) -> None:
        for gl in self._gate_layers.values():
            gl._cache_feats = v

    # ------------------------------------------------------------------
    # Detector heatmap support
    # ------------------------------------------------------------------
    def set_prev_detections(
        self, boxes: torch.Tensor | None, scores: torch.Tensor | None
    ) -> None:
        for gl in self._gate_layers.values():
            gl._det_boxes_prev = boxes
            gl._det_scores_prev = scores

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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        frame: torch.Tensor,
        gate_input: TrackerGateInput | list[TrackerGateInput] | None = None,
    ) -> dict[str, Any] | tuple[Any, ...]:
        for gl in self._gate_layers.values():
            gl._gate_input = gate_input

        result = self.yolo_model(frame)

        for gl in self._gate_layers.values():
            gl._gate_input = None

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
# Inline gate injection (patches forward, zero hook roundtrip)
# ---------------------------------------------------------------------------
def _inject_gate(module: nn.Module, gate_layer: GateLayer) -> None:
    _orig = module.forward

    def _patched(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return gate_layer(_orig(x))  # type: ignore[no-any-return]

    module.forward = _patched.__get__(module, type(module))


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
