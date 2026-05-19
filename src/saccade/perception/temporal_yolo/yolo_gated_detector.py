"""
GatedYOLODetector — Option D revised: TrackSpatialGate injected into YOLO via hooks.

Architecture:
    YOLO26s backbone (layers 0-22)
        ↓  hooks modify layer 16 (P3), 19 (P4), 22 (P5) outputs
    TrackSpatialGate  (gate = 1 + alpha * heatmap; alpha init=0)
        ↓  gated FPN features passed to Detect head
    YOLO Detect Head (layer 23, pre-trained)
        ↓
    Standard YOLO detection output

Key difference from TemporalYOLOConditioned:
- No track queries, no Transformer decoder
- Detect head is a shallow CNN → cannot bypass gate
- Standard detection loss (v8DetectionLoss) provides direct gradient to alpha
- Inference: gate_input from ByteTrack previous frame; training: GT oracle (gt_ratio)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass

from .yolo_conditioned import TrackerGateInput, TrackSpatialGate


# Layer indices for P3/P4/P5 in yolo26s (verified by layer dump)
_GATE_LAYER_IDX: dict[str, int] = {"p3": 16, "p4": 19, "p5": 22}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class GatedDetConfig:
    scales: tuple[str, ...] = ("p3", "p4", "p5")
    gate_sigma_scale: float = 0.5
    gate_min_score: float = 0.5
    freeze_backbone: bool = False  # unfreeze for joint fine-tuning


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class GatedYOLODetector(nn.Module):
    """YOLO26s with TrackSpatialGate injected at P3/P4/P5 via forward hooks."""

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

        from ultralytics import YOLO as _YOLO

        yolo = _YOLO(yolo_pt_path)
        self.yolo_model = yolo.model.to(device)

        # Enable / disable backbone gradients
        for p in self.yolo_model.parameters():
            p.requires_grad_(not cfg.freeze_backbone)

        # (feat_channels not needed by TrackSpatialGate; kept for reference only)
        _ = self._probe_feat_channels(device)

        # Gate (always trainable; feat_channels not needed — gate is feature-agnostic)
        self.gate = TrackSpatialGate(
            scales=tuple(cfg.scales),
            sigma_scale=cfg.gate_sigma_scale,
            min_score=cfg.gate_min_score,
        )
        for p in self.gate.parameters():
            p.requires_grad_(True)

        # Hook state: set before forward, cleared after
        self._current_gate: TrackerGateInput | list[TrackerGateInput] | None = None

        # FPN feature cache: populated during forward when cache_feats=True
        self._feat_cache: dict[str, torch.Tensor] = {}
        self.cache_feats: bool = False

        # Register gate-injection hooks at P3/P4/P5 output layers
        for scale in cfg.scales:
            idx = _GATE_LAYER_IDX[scale]
            self.yolo_model.model[idx].register_forward_hook(self._make_hook(scale))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _probe_feat_channels(self, device: str | torch.device) -> dict[str, int]:
        channels: dict[str, int] = {}
        tmp_hooks = []
        for scale in self.cfg.scales:
            idx = _GATE_LAYER_IDX[scale]

            def _h(m, i, o, s=scale):
                channels[s] = o.shape[1]

            tmp_hooks.append(self.yolo_model.model[idx].register_forward_hook(_h))
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 640, 640, device=device)
            self.yolo_model(dummy)
        for h in tmp_hooks:
            h.remove()
        return channels

    def _make_hook(self, scale: str):
        def _hook(module, inp, output: torch.Tensor) -> torch.Tensor:
            if self.cache_feats:
                self._feat_cache[scale] = output  # (B, C, H, W)
            if self._current_gate is None:
                return output
            hw = (output.shape[2], output.shape[3])
            renderer = self.gate.renderer
            alpha = self.gate.alphas[scale]
            if isinstance(self._current_gate, list):
                maps = renderer.batch_forward(self._current_gate, hw)  # (B, H, W)
                gate_map = 1.0 + alpha * maps.unsqueeze(1)  # (B, 1, H, W)
            else:
                heatmap = renderer(self._current_gate, hw)  # (1, H, W)
                gate_map = (1.0 + alpha * heatmap).unsqueeze(0)  # (1, 1, H, W)
            return output * gate_map

        return _hook

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        frame: torch.Tensor,
        gate_input: TrackerGateInput | list[TrackerGateInput] | None = None,
    ) -> dict | tuple:
        """
        Args:
            frame:      (B, 3, H, W)
            gate_input: TrackerGateInput (inference, broadcasts) or
                        list[TrackerGateInput] (training, per-sample)
        Returns:
            train mode: dict with 'one2one' / 'one2many' dicts
            eval mode:  tuple (boxes_tensor (B,300,6), meta_dict)
        """
        self._current_gate = gate_input
        result = self.yolo_model(frame)
        self._current_gate = None
        return result

    # ------------------------------------------------------------------
    # Optimizer helper
    # ------------------------------------------------------------------
    def parameter_groups(
        self,
        lr_gate: float,
        lr_yolo: float = 0.0,
    ) -> list[dict]:
        groups: list[dict] = [
            {"params": list(self.gate.parameters()), "name": "gate", "lr": lr_gate},
        ]
        if lr_yolo > 0.0:
            trainable = [p for p in self.yolo_model.parameters() if p.requires_grad]
            if trainable:
                groups.append({"params": trainable, "name": "yolo", "lr": lr_yolo})
        return groups

    def alpha_summary(self) -> str:
        parts = [f"{s}={self.gate.alphas[s].item():.4f}" for s in self.cfg.scales]
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
    """Build GatedYOLODetector, optionally loading a checkpoint."""
    model = GatedYOLODetector(yolo_pt_path, cfg=cfg, device=device)
    if weights_path:
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        sd = state["model"] if "model" in state else state
        # strip compile prefix if present
        sd = {k.replace("._orig_mod.", "."): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[GatedDet] Missing {len(missing)} keys")
        if unexpected:
            print(f"[GatedDet] Unexpected {len(unexpected)} keys")
    return model.to(device)
