"""
Option C: TemporalYOLOJoint — PyTorch YOLO + Cross-Attention 聯合訓練。

與 Option B (frozen backbone) 的差異：
  1. YOLO backbone 所有參數參與訓練（修正 Ultralytics requires_grad=False 的 bug）
  2. 支援 P3/P4/P5 多尺度 FPN Cross-Attention（預設 P5-only，可升到全尺度）
  3. parameter_groups() 回傳差異學習率：backbone LR 遠低於 decoder LR
  4. 乾淨的 TemporalYOLOJoint.forward() 不依賴 TemporalYOLOHybrid internals

架構圖：
  Frame ─→ YOLOFeaturePyramid ─→ P3/P4/P5 feats
                                        ↓
                               FPNSequenceProjection
                                        ↓   ← Track Queries (prev frame)
                               decoder_layers (Cross-Attn)
                                        ↓
                               boxes / scores / updated_queries
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn

from .model import (
    SinusoidalPositionEncoding2D,
    TemporalYOLOConfig,
    TrackQueryDecoderLayer,
    TrackQueryLifecycle,
)

# FPN hook target layers (by index in model.model Sequential)
_FPN_HOOK_LAYERS: dict[str, int] = {"p3": 16, "p4": 19, "p5": 22}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class JointConfig:
    # Decoder hyperparams
    embed_dim: int = 256
    num_heads: int = 8
    num_queries: int = 100
    ffn_dim: int = 1024
    dropout: float = 0.1
    num_decoder_layers: int = 3
    score_threshold: float = 0.3
    # Joint-specific
    scales: tuple[str, ...] = ("p5",)
    freeze_backbone: bool = False
    use_checkpointing: bool = False
    use_self_attention: bool = True  # Option D core: query-query interaction

    def to_temporal_config(self) -> TemporalYOLOConfig:
        return TemporalYOLOConfig(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_queries=self.num_queries,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout,
            num_decoder_layers=self.num_decoder_layers,
            score_threshold=self.score_threshold,
            use_self_attn=self.use_self_attention,
        )


# ---------------------------------------------------------------------------
# 1. YOLO Feature Pyramid
# ---------------------------------------------------------------------------
class YOLOFeaturePyramid(nn.Module):
    """
    Wraps Ultralytics YOLO26* and exposes P3/P4/P5 FPN feature maps via hooks.

    BUG FIX (vs Option B YOLONeckExtractor):
      Ultralytics loads ALL weights with requires_grad=False by default.
      Option B's freeze=False branch silently skipped enabling grads,
      making --no-freeze-yolo a no-op. We fix that here.
    """

    def __init__(
        self,
        yolo_pt_path: str,
        scales: tuple[str, ...] = ("p5",),
        freeze: bool = False,
    ):
        super().__init__()
        from ultralytics import YOLO as UltralyticsYOLO  # type: ignore[attr-defined]

        yolo = UltralyticsYOLO(yolo_pt_path)
        self.yolo_model: Any = yolo.model
        self.scales = list(scales)
        self._feats: dict[str, torch.Tensor] = {}
        self._hooks: list[Any] = []
        self.feat_channels: dict[str, int] = {}

        for scale in scales:
            idx = _FPN_HOOK_LAYERS[scale]
            hook = self.yolo_model.model[idx].register_forward_hook(  # type: ignore[union-attr]
                self._make_hook(scale)
            )
            self._hooks.append(hook)

        # Ultralytics sets requires_grad=False for all params by default —
        # must explicitly enable for joint training.
        for p in self.yolo_model.parameters():  # type: ignore[union-attr]
            p.requires_grad_(not freeze)

        if freeze:
            self.yolo_model.eval()  # type: ignore[union-attr]
            print(f"[YOLOFeaturePyramid] Backbone FROZEN  scales={scales}")
        else:
            n = sum(p.numel() for p in self.yolo_model.parameters())  # type: ignore[union-attr]
            print(
                f"[YOLOFeaturePyramid] Backbone TRAINABLE  {n:,} params  scales={scales}"
            )

    def _make_hook(self, scale: str) -> Callable[[nn.Module, Any, torch.Tensor], None]:
        def _h(mod: nn.Module, inp: Any, out: torch.Tensor) -> None:
            if isinstance(out, torch.Tensor):
                self._feats[scale] = out
                if scale not in self.feat_channels:
                    self.feat_channels[scale] = out.shape[1]

        return _h

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns {scale: (B, C, H, W)} with gradient flow when backbone is trainable."""
        self._feats.clear()
        _ = self.yolo_model(x)
        return {s: self._feats[s] for s in self.scales}

    def __del__(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 2. FPN → flat token sequence
# ---------------------------------------------------------------------------
class FPNSequenceProjection(nn.Module):
    """
    Projects multi-scale FPN features to embed_dim, adds 2D sinusoidal pos enc,
    flattens each scale and concatenates into a single token sequence.

    Input:  {scale: (B, C_scale, H, W)}
    Output: feat_seq (B, S, D), feat_pos (B, S, D)
            where S = Σ H_i * W_i over selected scales
    """

    def __init__(
        self,
        feat_channels: dict[str, int],
        embed_dim: int,
        scales: tuple[str, ...],
    ):
        super().__init__()
        self.scales = list(scales)
        self.embed_dim = embed_dim
        self.projs = nn.ModuleDict(
            {s: nn.Linear(feat_channels[s], embed_dim) for s in scales}
        )
        self.pos_enc = SinusoidalPositionEncoding2D(embed_dim)

    def forward(
        self, fpn_feats: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seqs, poses = [], []
        for scale in self.scales:
            feat = fpn_feats[scale]  # (B, C, H, W)
            B, _, H, W = feat.shape

            seq = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
            seq = self.projs[scale](seq)  # (B, H*W, D)
            seqs.append(seq)

            # pos enc at embed_dim — dummy zero tensor: only shape matters
            dummy = feat.new_zeros(B, self.embed_dim, H, W)
            pos_seq = self.pos_enc(dummy).flatten(2).transpose(1, 2)  # (B, H*W, D)
            poses.append(pos_seq)

        return torch.cat(seqs, dim=1), torch.cat(poses, dim=1)


# ---------------------------------------------------------------------------
# 3. TemporalYOLOJoint — full joint model
# ---------------------------------------------------------------------------
class TemporalYOLOJoint(nn.Module):
    """
    Option C: YOLO backbone (trainable) + Transformer Track Query Decoder.

    Forward:
        model(frame, prev_queries=None) -> dict with keys:
            boxes           (B, N_q, 4) [cx,cy,w,h] normalized
            scores          (B, N_q)    existence logits
            updated_queries (B, N_q, D) pass to next frame
            query_ids       list[int]   -1 = empty slot
    """

    def __init__(self, cfg: JointConfig, yolo_pt_path: str):
        super().__init__()
        self.cfg = cfg
        tc = cfg.to_temporal_config()

        # YOLO backbone + neck
        self.pyramid = YOLOFeaturePyramid(
            yolo_pt_path, scales=cfg.scales, freeze=cfg.freeze_backbone
        )

        # Discover channel widths via one CPU dummy forward
        self._probe_channels()

        # FPN → token sequence projection
        self.fpn_proj = FPNSequenceProjection(
            feat_channels=self.pyramid.feat_channels,
            embed_dim=cfg.embed_dim,
            scales=cfg.scales,
        )

        # Learnable query positional encoding
        self.query_pos_embed = nn.Embedding(cfg.num_queries, cfg.embed_dim)

        # Decoder stack (reuses TrackQueryDecoderLayer from model.py)
        self.decoder_layers = nn.ModuleList(
            [TrackQueryDecoderLayer(tc) for _ in range(cfg.num_decoder_layers)]
        )

        # Detection head
        D = cfg.embed_dim
        self.box_head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 4))
        self.score_head = nn.Linear(D, 1)

        # Track Query lifecycle
        self.newborn_embed = nn.Embedding(cfg.num_queries, cfg.embed_dim)
        nn.init.normal_(self.newborn_embed.weight, std=0.02)
        self.__lifecycle: TrackQueryLifecycle | None = None

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------
    def _probe_channels(self) -> None:
        """Run a zero forward to populate YOLOFeaturePyramid.feat_channels."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 640, 640)
            self.pyramid(dummy)

    def _get_lifecycle(self) -> TrackQueryLifecycle:
        if self.__lifecycle is None:
            self.__lifecycle = TrackQueryLifecycle(
                self.cfg.to_temporal_config(), self.newborn_embed
            )
        return self.__lifecycle

    def _make_init_queries(self, B: int, device: torch.device) -> torch.Tensor:
        slots = torch.arange(self.cfg.num_queries, device=device)
        return self.newborn_embed(slots).unsqueeze(0).expand(B, -1, -1).clone()  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset_sequence(self) -> None:
        """Call at the start of each new video sequence."""
        if self.__lifecycle is not None:
            self.__lifecycle.reset()
        self.__lifecycle = None

    def parameter_groups(
        self, lr_backbone: float, lr_decoder: float
    ) -> list[dict[str, Any]]:
        """
        Two param groups for differential learning rates:
          - backbone: lower LR (backbone benefits from gentle fine-tuning)
          - decoder:  full LR (cross-attention head trained from scratch)
        """
        backbone_ids = {id(p) for p in self.pyramid.parameters()}
        backbone = [
            p for p in self.parameters() if id(p) in backbone_ids and p.requires_grad
        ]
        decoder = [
            p
            for p in self.parameters()
            if id(p) not in backbone_ids and p.requires_grad
        ]
        n_bb = sum(p.numel() for p in backbone)
        n_dec = sum(p.numel() for p in decoder)
        print(f"[TemporalYOLOJoint] backbone params: {n_bb:,} @ lr={lr_backbone}")
        print(f"[TemporalYOLOJoint] decoder  params: {n_dec:,} @ lr={lr_decoder}")
        return [
            {"params": backbone, "lr": lr_backbone, "name": "backbone"},
            {"params": decoder, "lr": lr_decoder, "name": "decoder"},
        ]

    def forward(
        self,
        frame: torch.Tensor,  # (B, 3, H, W)
        prev_queries: torch.Tensor | None = None,  # (B, N_q, D) or None
    ) -> dict[str, Any]:
        B = frame.shape[0]
        lc = self._get_lifecycle()

        if prev_queries is None:
            prev_queries = self._make_init_queries(B, frame.device)

        # 1. YOLO FPN features
        fpn_feats = self.pyramid(frame)

        # 2. Multi-scale → flat sequence
        feat_seq, feat_pos = self.fpn_proj(fpn_feats)  # (B, S, D)

        # 3. Query positional encoding
        slots = torch.arange(self.cfg.num_queries, device=frame.device)
        query_pos = self.query_pos_embed(slots).unsqueeze(0).expand(B, -1, -1)

        # 4. Decoder layers
        queries = prev_queries
        if self.cfg.use_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint

            def run_layers(q, f, qp, fp):  # type: ignore[no-untyped-def]
                for layer in self.decoder_layers:
                    q = layer(q, f, qp, fp)
                return q

            queries = checkpoint(
                run_layers, queries, feat_seq, query_pos, feat_pos, use_reentrant=False
            )
        else:
            for layer in self.decoder_layers:
                queries = layer(queries, feat_seq, query_pos, feat_pos)

        # 5. Detection head
        boxes = torch.sigmoid(self.box_head(queries))  # (B, N_q, 4)
        scores = self.score_head(queries).squeeze(-1)  # (B, N_q)

        # 6. Lifecycle
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
def build_temporal_yolo_joint(
    yolo_pt_path: str,
    cfg: JointConfig | None = None,
    device: str | torch.device = "cuda",
    weights_path: str = "",
) -> TemporalYOLOJoint:
    """
    Build TemporalYOLOJoint, optionally loading a checkpoint from prior training.

    Args:
        yolo_pt_path:  Path to yolo26s.pt (or other variant)
        cfg:           JointConfig; defaults to scales=("p5",), freeze_backbone=False
        weights_path:  Optional checkpoint path (joint model state_dict)
        device:        Target device
    """
    if cfg is None:
        cfg = JointConfig()
    model = TemporalYOLOJoint(cfg, yolo_pt_path=yolo_pt_path)
    if weights_path:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    return model.to(device)
