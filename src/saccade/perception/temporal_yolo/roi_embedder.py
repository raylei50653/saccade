"""
YOLO ROI Embedder — extracts L2-normalised appearance embeddings from FPN features.

Uses ROI-average-pool at P3/P4/P5 (128+256+512 = 896 dim).
No separate model required; piggybacks on GatedYOLODetector's existing FPN computation.

Quality-weighted EMA bank for temporal denoising:
  bank[id] = normalize(α_eff * e_new + (1-α_eff) * bank[id])
  α_eff = α_base * quality  (quality = conf * sqrt(area/img_area))
  Outlier gate: skip if cosine_sim < consistency_thr and count > warmup

Optional: pass a ReIDHead to project 896-dim → 128-dim discriminative embeddings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

if TYPE_CHECKING:
    from saccade.perception.temporal_yolo.reid_head import ReIDHead

_SPATIAL_SCALES = {"p3": 1 / 8, "p4": 1 / 16, "p5": 1 / 32}
EMB_DIM = 896  # 128 + 256 + 512


# ---------------------------------------------------------------------------
# ROI extraction
# ---------------------------------------------------------------------------
def extract_roi_embeddings(
    feat_cache: dict[str, torch.Tensor],  # populated by GatedYOLODetector hooks
    boxes_xyxy: torch.Tensor,  # (N, 4) absolute px in img_size space
    reid_head: "ReIDHead | None" = None,
) -> torch.Tensor:
    """
    Return L2-normalised embeddings. Returns zeros if no boxes.

    Without reid_head: (N, 896) raw FPN pool.
    With    reid_head: (N, out_dim) projected discriminative embedding.
    """
    if boxes_xyxy.numel() == 0:
        dev = next(iter(feat_cache.values())).device
        out_dim = reid_head.out_dim if reid_head is not None else EMB_DIM
        return torch.zeros((0, out_dim), device=dev)

    # roi_align expects [batch_idx, x1, y1, x2, y2]
    batch_boxes = torch.cat(
        [
            torch.zeros(len(boxes_xyxy), 1, device=boxes_xyxy.device),
            boxes_xyxy.float(),
        ],
        dim=1,
    )  # (N, 5)

    parts = []
    for scale in ("p3", "p4", "p5"):
        feat = feat_cache.get(scale)
        if feat is None:
            continue
        # pool_size=1 → global average over ROI region
        pooled = roi_align(
            feat.float(),
            batch_boxes,
            output_size=1,
            spatial_scale=_SPATIAL_SCALES[scale],
            aligned=True,
        )
        parts.append(pooled.flatten(1))  # (N, C)

    emb = torch.cat(parts, dim=1)  # (N, 896)
    raw = F.normalize(emb, dim=1)

    if reid_head is not None:
        with torch.no_grad():
            return reid_head(raw)  # type: ignore[no-any-return]
    return raw


# ---------------------------------------------------------------------------
# Single-scale FPN crop embedder
# ---------------------------------------------------------------------------
class FPNCropEmbedder:
    """
    Single-scale FPN crop embedder: ROI-align → spatial avg-pool → L2-normalise.

    Default: P4 (256-dim), output_size=7.  Compared to the 3-scale 896-dim
    extract_roi_embeddings(), this is lighter and easier to train a projection
    head on top of.
    """

    _SF = {"p3": 1 / 8, "p4": 1 / 16, "p5": 1 / 32}
    _CH = {"p3": 128, "p4": 256, "p5": 512}

    def __init__(self, scale: str = "p4", output_size: int = 7):
        scale = scale.lower()
        assert scale in self._SF, f"scale must be one of {list(self._SF)}"
        self.scale = scale
        self.output_size = output_size
        self.emb_dim: int = self._CH[scale]

    def extract(
        self,
        feat_cache: dict[str, torch.Tensor],
        boxes_xyxy: torch.Tensor,  # (N, 4) absolute px in img_size space
    ) -> torch.Tensor:
        """Return (N, C) L2-normalised embeddings.  Returns zeros if no boxes."""
        N = len(boxes_xyxy)
        feat = feat_cache.get(self.scale)
        dev = boxes_xyxy.device if N > 0 else next(iter(feat_cache.values())).device
        if feat is None or N == 0:
            return torch.zeros((N, self.emb_dim), device=dev)
        rois = torch.cat(
            [torch.zeros(N, 1, device=feat.device), boxes_xyxy.float().to(feat.device)],
            dim=1,
        )  # (N, 5)
        crops = roi_align(
            feat.float(),
            rois,
            self.output_size,
            spatial_scale=self._SF[self.scale],
            aligned=True,
        )  # (N, C, output_size, output_size)
        embs = crops.flatten(2).mean(2)  # (N, C)
        return F.normalize(embs, dim=1)


# ---------------------------------------------------------------------------
# Per-track quality-weighted EMA bank
# ---------------------------------------------------------------------------
class ROIEmbeddingBank:
    """
    Maintains a denoised embedding per track ID via quality-weighted EMA.

    Update rule (per confirmed match):
        quality = conf * sqrt(w*h / img_size²)   ∈ [0, 1]
        if cosine_sim(new, bank) < consistency_thr and count > warmup:
            skip  (outlier / likely misdetection)
        α_eff = α_base * quality
        bank[id] = normalize(α_eff * new_emb + (1 - α_eff) * bank[id])
    """

    def __init__(
        self,
        alpha: float = 0.15,
        consistency_thr: float = 0.3,
        warmup_frames: int = 3,
        img_size: int = 640,
    ):
        self.alpha = alpha
        self.consistency_thr = consistency_thr
        self.warmup = warmup_frames
        self.img_size = img_size

        self._bank: dict[int, torch.Tensor] = {}  # {track_id: (896,)}
        self._count: dict[int, int] = {}

    def update(
        self,
        track_ids: list[int],
        embeddings: torch.Tensor,  # (N, 896)
        scores: torch.Tensor,  # (N,) detection confidence
        boxes_xyxy: torch.Tensor,  # (N, 4) for area quality
    ) -> None:
        """Update bank for matched tracks."""
        if embeddings.numel() == 0:
            return

        areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=1) * (
            boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        ).clamp(min=1)
        quality = scores.float() * (areas / self.img_size**2).sqrt().clamp(max=1.0)

        for i, tid in enumerate(track_ids):
            e_new = embeddings[i]
            q = float(quality[i])

            if tid in self._bank:
                existing = self._bank[tid]
                sim = float(e_new @ existing)
                if sim < self.consistency_thr and self._count[tid] > self.warmup:
                    continue  # outlier gate
                alpha_eff = self.alpha * q
                blended = alpha_eff * e_new + (1 - alpha_eff) * existing
                self._bank[tid] = F.normalize(blended, dim=0)
            else:
                self._bank[tid] = F.normalize(e_new, dim=0)

            self._count[tid] = self._count.get(tid, 0) + 1

    def get(self, track_id: int) -> torch.Tensor | None:
        return self._bank.get(track_id)

    def get_batch(self, track_ids: list[int]) -> tuple[list[int], torch.Tensor] | None:
        """Return (valid_ids, stacked_embeddings) for ids present in bank."""
        valid_ids = [tid for tid in track_ids if tid in self._bank]
        if not valid_ids:
            return None
        embs = torch.stack([self._bank[tid] for tid in valid_ids])
        return valid_ids, embs

    def remove(self, track_id: int) -> None:
        self._bank.pop(track_id, None)
        self._count.pop(track_id, None)

    def reset(self) -> None:
        self._bank.clear()
        self._count.clear()

    def __len__(self) -> int:
        return len(self._bank)
