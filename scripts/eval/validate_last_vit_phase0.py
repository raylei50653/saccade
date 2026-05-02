#!/usr/bin/env python3
# mypy: ignore-errors
"""
Phase 0: LaSt-ViT Python Prototype Validation

Validates whether the LaSt-ViT channel-wise FFT stability pipeline can:
  1. Discriminate foreground (person) patches from background patches
     via stability scores derived from last_hidden_state.
  2. Improve same-person / different-person cosine similarity gap
     compared to the current image_embeds baseline.

Uses the SigLIP2 TRT engine's Python path to read last_hidden_state
directly from output_buffers — no C++ changes needed.

Usage:
  uv run python scripts/eval/validate_last_vit_phase0.py
  uv run python scripts/eval/validate_last_vit_phase0.py --seq MOT17-02-SDP --n-frames 15
  uv run python scripts/eval/validate_last_vit_phase0.py --sigma-sweep
  uv run python scripts/eval/validate_last_vit_phase0.py --frames 50 100 200 400 600
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, NamedTuple

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
_src = _project_root / "src"
if _src.exists():
    sys.path.insert(0, str(_src))
_build = _project_root / "build"
if _build.exists():
    sys.path.insert(0, str(_build))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Force Python TRT path to access output_buffers["last_hidden_state"].
# The C++ path only writes image_embeds and discards last_hidden_state.
import saccade.perception.feature_extractor as _fe_mod

_fe_mod.HAS_CPP_EXT = False
from saccade.perception.feature_extractor import TRTFeatureExtractor  # noqa: E402


# ---------------------------------------------------------------------------
# GT loader
# ---------------------------------------------------------------------------

def load_gt(gt_path: Path) -> dict[int, list[tuple[int, int, int, int, int, float]]]:
    """Return {frame_id: [(track_id, x, y, w, h, visibility), ...]} for class=pedestrian."""
    result: dict[int, list] = {}
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            fid, tid = int(parts[0]), int(parts[1])
            x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
            is_active = int(parts[6]) if len(parts) > 6 else 1
            class_id = int(float(parts[7])) if len(parts) > 7 else 1
            visibility = float(parts[8]) if len(parts) > 8 else 1.0
            if is_active == 0 or class_id != 1:
                continue
            result.setdefault(fid, []).append((tid, x, y, w, h, visibility))
    return result


# ---------------------------------------------------------------------------
# LaSt-ViT pipeline
# ---------------------------------------------------------------------------

def _gauss_filter(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply 1D Gaussian low-pass along the last dimension via RFFT. Returns filtered tensor."""
    C = x.shape[-1]
    X = torch.fft.rfft(x, dim=-1)
    freqs = torch.arange(C // 2 + 1, device=x.device, dtype=torch.float32) / C
    gauss_w = torch.exp(-freqs**2 / (2.0 * sigma**2))
    return torch.fft.irfft(X * gauss_w, n=C, dim=-1)


def _stability_scores(x: torch.Tensor, x_filt: torch.Tensor) -> torch.Tensor:
    """Per-patch stability: S_i = 1 - ||x_i - x_filt_i||^2 / ||x_i||^2. Shape [B, N]."""
    diff_sq = (x - x_filt).pow(2).sum(dim=-1)
    norm_sq = x.pow(2).sum(dim=-1).clamp(min=1e-8)
    return (1.0 - diff_sq / norm_sq).clamp(0.0, 1.0)


def last_vit_pipeline(
    last_hidden_state: torch.Tensor,
    sigma: float = 0.3,
    top_k_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    LaSt-ViT channel-wise FFT stability pipeline (arXiv:2602.22394).
    Single-sigma variant: same sigma for embedding and gating.

    Returns:
        embedding:        [B, C] L2-normalized via Top-K pooling.
        stability_scores: [B, N] per-patch stability (0=unstable, 1=stable).
    """
    x = last_hidden_state.float()
    x_filt = _gauss_filter(x, sigma)
    scores = _stability_scores(x, x_filt)
    embedding = _topk_pool(x, scores, top_k_ratio)
    return embedding, scores


def last_vit_pipeline_dual(
    last_hidden_state: torch.Tensor,
    sigma_embed: float = 0.015,
    sigma_gate: float = 0.030,
    top_k_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    LaSt-ViT dual-sigma variant.

    Separates embedding quality from stability gating:
      - sigma_embed (small, e.g. 0.015): aggressive low-pass → best cosine gap.
      - sigma_gate  (larger, e.g. 0.030): moderate low-pass → stability scores
        with positive FG-vs-BG discriminability.

    Returns:
        embedding:        [B, C] L2-normalized via Top-K pooling (using sigma_embed).
        stability_scores: [B, N] per-patch stability (using sigma_gate).
    """
    x = last_hidden_state.float()
    x_filt_embed = _gauss_filter(x, sigma_embed)
    x_filt_gate  = _gauss_filter(x, sigma_gate)
    scores_gate  = _stability_scores(x, x_filt_gate)   # used for gating
    scores_embed = _stability_scores(x, x_filt_embed)  # used for Top-K selection
    embedding    = _topk_pool(x, scores_embed, top_k_ratio)
    return embedding, scores_gate


def _topk_pool(x: torch.Tensor, scores: torch.Tensor, top_k_ratio: float) -> torch.Tensor:
    """Select Top-K patches by score, average, and L2-normalize. [B, N, C] → [B, C]."""
    B, N, C = x.shape
    K = max(1, int(N * top_k_ratio))
    topk_idx = scores.topk(K, dim=-1).indices              # [B, K]
    topk_feats = torch.gather(x, 1, topk_idx.unsqueeze(-1).expand(-1, -1, C))
    return F.normalize(topk_feats.mean(dim=1), dim=-1)


def _per_channel_stab(x: torch.Tensor, x_filt: torch.Tensor) -> torch.Tensor:
    """Per-(patch, channel) stability: s[b,n,c] = clamp(1 - (x-xf)^2/(x^2+ε), 0, 1)."""
    diff_sq = (x - x_filt).pow(2)
    norm_sq = x.pow(2).clamp(min=1e-8)
    return (1.0 - diff_sq / norm_sq).clamp(0.0, 1.0)


def last_vit_v2_channel_voting(
    last_hidden_state: torch.Tensor,
    sigma_embed: float = 0.015,
    sigma_gate: float = 0.040,
    top_k_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    V2 — Paper-strict channel-wise Top-1 voting aggregation.

    For each channel c, the patch with highest per-channel stability casts one vote.
    Top-K patches by total vote count are averaged → L2-normalized embedding.
    Stability output: mean of per-(n,c) gate stability.
    """
    x = last_hidden_state.float()
    B, N, C = x.shape
    K = max(1, int(N * top_k_ratio))

    x_filt_embed = _gauss_filter(x, sigma_embed)   # [B, N, C]
    x_filt_gate  = _gauss_filter(x, sigma_gate)    # [B, N, C]

    stab_embed_nc = _per_channel_stab(x, x_filt_embed)  # [B, N, C]
    stab_gate_nc  = _per_channel_stab(x, x_filt_gate)   # [B, N, C]

    # Channel-wise Top-1 vote: each channel votes for its highest-stability patch
    votes_idx = stab_embed_nc.argmax(dim=1)                         # [B, C]
    vote_count = torch.zeros(B, N, device=x.device, dtype=x.dtype)
    vote_count.scatter_add_(1, votes_idx,
                            torch.ones(B, C, device=x.device, dtype=x.dtype))  # [B, N]

    topk_idx   = vote_count.topk(K, dim=-1).indices                 # [B, K]
    topk_feats = torch.gather(x, 1, topk_idx.unsqueeze(-1).expand(-1, -1, C))
    embedding  = F.normalize(topk_feats.mean(dim=1), dim=-1)        # [B, C]

    stab_mean = stab_gate_nc.mean(dim=(1, 2))                       # [B]
    return embedding, stab_mean


def last_vit_v3_weighted_mean(
    last_hidden_state: torch.Tensor,
    sigma_embed: float = 0.015,
    sigma_gate: float = 0.040,
    top_k_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    V3 — Soft score-weighted mean (no hard Top-K threshold).

    embed[c] = Σ_n(s_n · x[n,c]) / Σ_n(s_n)  then L2-normalize.
    top_k_ratio is ignored (kept for API parity).
    """
    x = last_hidden_state.float()

    x_filt_embed = _gauss_filter(x, sigma_embed)
    x_filt_gate  = _gauss_filter(x, sigma_gate)

    scores_gate  = _stability_scores(x, x_filt_gate)   # [B, N]
    scores_embed = _stability_scores(x, x_filt_embed)  # [B, N]

    weights   = scores_embed.unsqueeze(-1)              # [B, N, 1]
    w_sum     = (x * weights).sum(dim=1)                # [B, C]
    w_total   = weights.sum(dim=1).clamp(min=1e-8)      # [B, 1]
    embedding = F.normalize(w_sum / w_total, dim=-1)    # [B, C]

    stab_mean = scores_gate.mean(dim=-1)                # [B]
    return embedding, stab_mean


def last_vit_v4_per_channel_topk(
    last_hidden_state: torch.Tensor,
    sigma_embed: float = 0.015,
    sigma_gate: float = 0.040,
    top_k_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    V4 — Per-channel independent Top-K selection (no cross-channel voting).

    For each channel c, independently select the Top-K patches by per-(n,c) stability.
    embed[c] = mean over those K patches' c-th feature values → L2-normalize.
    """
    x = last_hidden_state.float()
    B, N, C = x.shape
    K = max(1, int(N * top_k_ratio))

    x_filt_embed = _gauss_filter(x, sigma_embed)
    x_filt_gate  = _gauss_filter(x, sigma_gate)

    stab_embed_nc = _per_channel_stab(x, x_filt_embed)  # [B, N, C]
    stab_gate_nc  = _per_channel_stab(x, x_filt_gate)   # [B, N, C]

    # Per-channel Top-K: topk over N for each channel independently
    # stab: [B, N, C] → permute → [B, C, N] → topk → [B, C, K]
    stab_T   = stab_embed_nc.permute(0, 2, 1)           # [B, C, N]
    topk_idx = stab_T.topk(K, dim=-1).indices            # [B, C, K]
    x_T      = x.permute(0, 2, 1)                        # [B, C, N]
    topk_x   = torch.gather(x_T, 2, topk_idx)            # [B, C, K]
    embedding = F.normalize(topk_x.mean(dim=-1), dim=-1) # [B, C]

    stab_mean = stab_gate_nc.mean(dim=(1, 2))            # [B]
    return embedding, stab_mean


# ---------------------------------------------------------------------------
# Phase 2B: variant comparison
# ---------------------------------------------------------------------------

_VARIANTS: dict[str, Any] = {
    "V1 patch-topk    (current)": lambda lhs, se, sg, tk: last_vit_pipeline_dual(lhs, se, sg, tk),
    "V2 channel-vote  (paper)  ": lambda lhs, se, sg, tk: last_vit_v2_channel_voting(lhs, se, sg, tk),
    "V3 weighted-mean (soft)   ": lambda lhs, se, sg, tk: last_vit_v3_weighted_mean(lhs, se, sg, tk),
    "V4 per-ch-topk   (indep)  ": lambda lhs, se, sg, tk: last_vit_v4_per_channel_topk(lhs, se, sg, tk),
}


def variant_compare(
    samples: list["CropSample"],
    sigma_embed: float = 0.015,
    sigma_gate: float = 0.040,
    top_k_ratio: float = 0.5,
) -> None:
    """Run all 4 algorithm variants and print a comparison table."""
    try:
        from scipy import stats as _stats
    except ImportError:
        _stats = None  # type: ignore[assignment]

    clean = [s for s in samples if s.category == "clean_fg" and s.track_id > 0]
    bg    = [s for s in samples if s.category == "background"]
    if not clean:
        print("  Not enough clean_fg samples for variant comparison.")
        return

    all_lhs = torch.stack([s.lhs       for s in clean])
    bg_lhs  = torch.stack([s.lhs       for s in bg]) if bg else None
    img_emb = torch.stack([s.img_embed for s in clean])

    by_track: dict[int, list[int]] = {}
    for i, s in enumerate(clean):
        by_track.setdefault(s.track_id, []).append(i)

    rng = np.random.default_rng(0)

    def _gap(embeds: torch.Tensor) -> tuple[float, float, float]:
        same, diff = [], []
        for idxs in by_track.values():
            for a in range(len(idxs)):
                for b_idx in range(a + 1, len(idxs)):
                    same.append(float(F.cosine_similarity(
                        embeds[idxs[a]].unsqueeze(0), embeds[idxs[b_idx]].unsqueeze(0))))
        n_target = min(len(same) * 3, 500)
        for _ in range(n_target * 8):
            if len(diff) >= n_target:
                break
            ia, ib = rng.integers(0, len(clean), size=2)
            if clean[ia].track_id != clean[ib].track_id:
                diff.append(float(F.cosine_similarity(
                    embeds[ia].unsqueeze(0), embeds[ib].unsqueeze(0))))
        sa = np.array(same) if same else np.array([0.0])
        da = np.array(diff) if diff else np.array([0.0])
        return float(sa.mean()), float(da.mean()), float(sa.mean() - da.mean())

    def _mw_p(fg_scores: np.ndarray, bg_scores: np.ndarray) -> str:
        if _stats is None or len(fg_scores) < 3 or len(bg_scores) < 3:
            return "   N/A"
        _, p = _stats.mannwhitneyu(fg_scores, bg_scores, alternative="greater")
        return f"{p:.4f}"

    # Baseline
    same_m, diff_m, gap_base = _gap(img_emb)
    print(f"\n{'='*74}")
    print("PHASE 2B — ALGORITHM VARIANT COMPARISON")
    print(f"  σ_embed={sigma_embed}  σ_gate={sigma_gate}  top_k={top_k_ratio}")
    print(f"  clean_fg={len(clean)}  background={len(bg)}")
    print(f"{'='*74}")
    print(f"  {'Variant':<36}  {'same':>7}  {'diff':>7}  {'gap':>8}  {'Δgap':>8}  {'p(FG>BG)':>9}")
    print(f"  {'-'*36}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*9}")
    print(f"  {'baseline image_embeds':<36}  {same_m:>7.4f}  {diff_m:>7.4f}  {gap_base:>+8.4f}  {'—':>8}  {'—':>9}")

    best_name, best_gap = "", gap_base
    for name, fn in _VARIANTS.items():
        embeds, stab_fg_patches = fn(all_lhs, sigma_embed, sigma_gate, top_k_ratio)
        s_m, d_m, gap = _gap(embeds)
        delta = gap - gap_base

        # FG vs BG stability discriminability
        stab_fg_arr = stab_fg_patches.mean(dim=-1).cpu().numpy() if stab_fg_patches.ndim == 2 else stab_fg_patches.cpu().numpy()
        if bg_lhs is not None:
            _, stab_bg = fn(bg_lhs, sigma_embed, sigma_gate, top_k_ratio)
            stab_bg_arr = stab_bg.mean(dim=-1).cpu().numpy() if stab_bg.ndim == 2 else stab_bg.cpu().numpy()
        else:
            stab_bg_arr = np.array([])
        pval_str = _mw_p(stab_fg_arr, stab_bg_arr)

        marker = " ◀" if gap > best_gap else ""
        print(f"  {name:<36}  {s_m:>7.4f}  {d_m:>7.4f}  {gap:>+8.4f}  {delta:>+8.4f}  {pval_str:>9}{marker}")
        if gap > best_gap:
            best_gap, best_name = gap, name

    print(f"{'='*74}")
    if best_name:
        print(f"  Best variant: {best_name.strip()}  (gap={best_gap:+.4f}, Δ={best_gap-gap_base:+.4f})")
    else:
        print("  No variant beat the baseline.")


# ---------------------------------------------------------------------------
# Sample container
# ---------------------------------------------------------------------------

class CropSample(NamedTuple):
    track_id: int
    frame_id: int
    bbox: tuple[int, int, int, int]
    visibility: float
    category: str          # "clean_fg" | "small_fg" | "background"
    img_embed: torch.Tensor    # [C] L2-normalized
    lhs: torch.Tensor          # [N, C] last_hidden_state
    stab_scores: torch.Tensor  # [N]


# ---------------------------------------------------------------------------
# Frame collector
# ---------------------------------------------------------------------------

def collect_frame(
    img_rgb: np.ndarray,
    detections: list[tuple[int, int, int, int, int, float]],
    target_hw: tuple[int, int],
    frame_id: int,
) -> list[tuple]:
    """
    Return list of (track_id, x, y, w, h, visibility, category, crop_rgb)
    for one frame.  Categories: clean_fg / small_fg / background.
    """
    H, W = img_rgb.shape[:2]
    crops_info = []

    for tid, x, y, w, h, vis in detections:
        area = w * h
        near_edge = x < 20 or y < 20 or (x + w) > (W - 20) or (y + h) > (H - 20)
        if vis >= 0.8 and area >= 3000 and not near_edge:
            cat = "clean_fg"
        elif vis < 0.4 or area < 800:
            cat = "small_fg"
        else:
            continue  # ambiguous: skip
        crop = _crop_resize(img_rgb, x, y, w, h, target_hw)
        crops_info.append((tid, x, y, w, h, vis, cat, crop))

    # Background: random patches with no GT overlap
    gt_boxes = [(x, y, x + w, y + h) for (_, x, y, w, h, _) in detections]
    rng = np.random.default_rng(frame_id * 7919)
    bg_added = 0
    for _ in range(200):
        if bg_added >= 2:
            break
        bx = int(rng.integers(0, max(1, W - target_hw[1])))
        by = int(rng.integers(0, max(1, H - target_hw[0])))
        bx2, by2 = bx + target_hw[1], by + target_hw[0]
        if any(bx < x2 and bx2 > x1 and by < y2 and by2 > y1 for x1, y1, x2, y2 in gt_boxes):
            continue
        crop = img_rgb[by:by2, bx:bx2]
        if crop.shape[0] < 4 or crop.shape[1] < 4:
            continue
        crop = cv2.resize(crop, (target_hw[1], target_hw[0]))
        crops_info.append((0, bx, by, target_hw[1], target_hw[0], 0.0, "background", crop))
        bg_added += 1

    return crops_info


def _crop_resize(
    img: np.ndarray, x: int, y: int, w: int, h: int, target_hw: tuple[int, int]
) -> np.ndarray:
    H, W = img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((target_hw[0], target_hw[1], 3), dtype=np.uint8)
    return cv2.resize(img[y1:y2, x1:x2], (target_hw[1], target_hw[0]))


# ---------------------------------------------------------------------------
# Background mask utilities
# ---------------------------------------------------------------------------

def _make_gaussian_mask(
    h: int, w: int,
    sigma_y: float,
    sigma_x: float,
    device: str = "cuda",
) -> torch.Tensor:
    """2D Gaussian window [1, 1, H, W] in [0,1], centred on crop."""
    y = (torch.arange(h, device=device).float() - h / 2.0) / (sigma_y * h)
    x = (torch.arange(w, device=device).float() - w / 2.0) / (sigma_x * w)
    mask = torch.exp(-(y[:, None] ** 2 + x[None, :] ** 2) / 2.0)
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


def apply_bg_mask(
    batch: torch.Tensor,   # [B, 3, H, W] float32 in [0, 1]
    mode: str,             # "none" | "gauss_0.30" | "gauss_0.40" | ...
) -> torch.Tensor:
    """Apply background suppression mask to a batch of crops in-place-friendly."""
    if mode == "none":
        return batch

    B, C, H, W = batch.shape

    if mode.startswith("gauss_"):
        parts = mode.split("_")
        sigma = float(parts[1])
        # Person crops are taller than wide; use elliptical mask (wider vertically)
        sigma_y = min(sigma * 1.25, 0.70)
        sigma_x = sigma
        mask = _make_gaussian_mask(H, W, sigma_y, sigma_x, device=batch.device)
        return batch * mask

    if mode == "mean_fill":
        # Replace background (low-mask region) with the per-image mean colour
        mask = _make_gaussian_mask(H, W, 0.38, 0.30, device=batch.device)
        # threshold at 0.5
        hard = (mask >= 0.5).float()
        mean_color = (batch * hard).sum(dim=(-2, -1), keepdim=True) / hard.sum(dim=(-2, -1), keepdim=True).clamp(min=1)
        return batch * hard + mean_color * (1.0 - hard)

    if mode == "vstrip":
        # Keep centre 60 % of width, fade sides
        cx = W / 2.0
        x = (torch.arange(W, device=batch.device).float() - cx) / (0.30 * W)
        strip = torch.exp(-x ** 2 / 2.0).view(1, 1, 1, W)
        return batch * strip

    raise ValueError(f"Unknown bg_mask mode: {mode}")


_BG_MASK_MODES = [
    "none",
    "gauss_0.25",
    "gauss_0.30",
    "gauss_0.38",
    "gauss_0.50",
    "mean_fill",
    "vstrip",
]


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(
    seq_dir: Path,
    extractor: TRTFeatureExtractor,
    frame_ids: list[int],
    gt: dict,
    sigma: float,
    top_k_ratio: float,
    bg_mask: str = "none",
) -> list[CropSample]:
    img_dir = seq_dir / "img1"
    target_hw = extractor.input_hw  # (H, W)
    samples: list[CropSample] = []

    for frame_id in frame_ids:
        if frame_id not in gt:
            continue
        img_path = img_dir / f"{frame_id:06d}.jpg"
        if not img_path.exists():
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        crops_info = collect_frame(img_rgb, gt[frame_id], target_hw, frame_id)
        if not crops_info:
            continue

        # Build GPU batch
        crop_tensors = [
            torch.from_numpy(c.copy()).float().div(255.0).permute(2, 0, 1)
            for *_, c in crops_info
        ]
        batch = torch.stack(crop_tensors).cuda()
        if bg_mask != "none":
            batch = apply_bg_mask(batch, bg_mask)

        # Chunk inference to stay within engine max_batch=16
        all_embeds = []
        all_lhs    = []
        max_b = extractor.max_batch
        with torch.no_grad():
            for start in range(0, len(crops_info), max_b):
                chunk = batch[start : start + max_b]
                extractor._extract_chunk(chunk)
                torch.cuda.synchronize()
                bs = chunk.shape[0]
                all_embeds.append(
                    extractor.output_buffers["image_embeds"][:bs].float().clone()
                )
                all_lhs.append(
                    extractor.output_buffers["last_hidden_state"][:bs].float().clone()
                )

        raw_embeds = torch.cat(all_embeds, dim=0)   # [N, 768]
        lhs_raw    = torch.cat(all_lhs,    dim=0)   # [N, 196, 768]

        # L2-normalize image_embeds (TRT output is pre-norm)
        img_embeds_norm = F.normalize(raw_embeds, dim=-1)

        # Drop CLS token if present (shape [B, 197, C] → [B, 196, C])
        N_tok = lhs_raw.shape[1]
        lhs_patches = lhs_raw[:, 1:, :] if N_tok == 197 else lhs_raw  # [B, 196, C]

        # LaSt-ViT pipeline
        _, stab_scores = last_vit_pipeline(lhs_patches, sigma=sigma, top_k_ratio=top_k_ratio)

        for i, (tid, x, y, w, h, vis, cat, _) in enumerate(crops_info):
            samples.append(CropSample(
                track_id=tid,
                frame_id=frame_id,
                bbox=(x, y, w, h),
                visibility=vis,
                category=cat,
                img_embed=img_embeds_norm[i].cpu(),
                lhs=lhs_patches[i].cpu(),
                stab_scores=stab_scores[i].cpu(),
            ))

        n_clean = sum(1 for c in crops_info if c[6] == "clean_fg")
        n_small = sum(1 for c in crops_info if c[6] == "small_fg")
        n_bg    = sum(1 for c in crops_info if c[6] == "background")
        avg_stab = f"{stab_scores[:n_clean].mean().item():.3f}" if n_clean else "n/a"
        print(f"  frame {frame_id:5d}: {len(crops_info):2d} crops  "
              f"clean_fg={n_clean}  small_fg={n_small}  bg={n_bg}  "
              f"avg_stab(clean)={avg_stab}")

    return samples


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def stability_report(samples: list[CropSample]) -> dict[str, np.ndarray]:
    by_cat: dict[str, list[float]] = {"clean_fg": [], "small_fg": [], "background": []}
    for s in samples:
        by_cat[s.category].append(float(s.stab_scores.mean()))

    print(f"\n{'='*62}")
    print("STABILITY SCORE ANALYSIS  (mean per-patch score per crop)")
    print(f"{'='*62}")
    for cat, vals in by_cat.items():
        if not vals:
            print(f"  [{cat:12s}]  no samples")
            continue
        a = np.array(vals)
        print(f"  [{cat:12s}]  n={len(a):3d}  "
              f"mean={a.mean():.4f}  std={a.std():.4f}  "
              f"p25={np.percentile(a,25):.4f}  p75={np.percentile(a,75):.4f}")

    clean = np.array(by_cat["clean_fg"])
    bg    = np.array(by_cat["background"])
    pval  = 1.0
    if len(clean) >= 5 and len(bg) >= 5:
        try:
            from scipy import stats
            _, pval = stats.mannwhitneyu(clean, bg, alternative="greater")
            print(f"\n  Mann-Whitney U (clean_fg > background): p={pval:.4f} "
                  f"{'✓ significant' if pval < 0.05 else '✗ NOT significant'}")
        except ImportError:
            delta = clean.mean() - bg.mean()
            print(f"\n  clean_fg - background mean delta = {delta:+.4f}  "
                  f"(install scipy for significance test)")
    return {k: np.array(v) for k, v in by_cat.items()}


def _cosine_pairs(
    embeds: torch.Tensor,
    by_track: dict[int, list[int]],
    all_samples: list,
    rng_seed: int = 42,
) -> tuple[list[float], list[float]]:
    """Return (same_person_sims, diff_person_sims) lists."""
    same, diff = [], []
    for idxs in by_track.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                same.append(float(F.cosine_similarity(
                    embeds[idxs[a]].unsqueeze(0), embeds[idxs[b]].unsqueeze(0))))
    rng = np.random.default_rng(rng_seed)
    n_target = min(len(same) * 3, 500)
    for _ in range(n_target * 8):
        if len(diff) >= n_target:
            break
        ia, ib = rng.integers(0, len(all_samples), size=2)
        if all_samples[ia].track_id != all_samples[ib].track_id:
            diff.append(float(F.cosine_similarity(
                embeds[ia].unsqueeze(0), embeds[ib].unsqueeze(0))))
    return same, diff


def _print_sim_row(label: str, same: list[float], diff: list[float]) -> float:
    sa = np.array(same) if same else np.zeros(1)
    da = np.array(diff) if diff else np.zeros(1)
    gap = sa.mean() - da.mean()
    print(f"  [{label}]")
    print(f"    same-person  n={len(sa):3d}  mean={sa.mean():.4f}  std={sa.std():.4f}")
    print(f"    diff-person  n={len(da):3d}  mean={da.mean():.4f}  std={da.std():.4f}")
    print(f"    gap (same-diff): {gap:+.4f}")
    return gap


def similarity_report(
    samples: list[CropSample],
    sigma: float,
    top_k_ratio: float,
    sigma_embed: float | None = None,
    sigma_gate: float | None = None,
) -> tuple[float, float]:
    """
    Report cosine similarity for image_embeds vs LaSt-ViT (single or dual sigma).

    If sigma_embed + sigma_gate are both set, reports three rows:
      baseline | single(sigma) | dual(sigma_embed, sigma_gate)
    and returns (baseline_gap, dual_gap).
    """
    clean = [s for s in samples if s.category == "clean_fg" and s.track_id > 0]
    if not clean:
        print("  No clean_fg samples — skipping similarity analysis.")
        return 0.0, 0.0

    all_lhs    = torch.stack([s.lhs       for s in clean])
    img_embeds = torch.stack([s.img_embed for s in clean])

    by_track: dict[int, list[int]] = {}
    for i, s in enumerate(clean):
        by_track.setdefault(s.track_id, []).append(i)

    # Single-sigma embedding
    lv_embeds_single, _ = last_vit_pipeline(all_lhs, sigma=sigma, top_k_ratio=top_k_ratio)

    print(f"\n{'='*62}")
    print("COSINE SIMILARITY ANALYSIS")
    print(f"{'='*62}")

    same_base, diff_base = _cosine_pairs(img_embeds, by_track, clean)
    gap_base = _print_sim_row("image_embeds (baseline)", same_base, diff_base)
    print()

    same_sv, diff_sv = _cosine_pairs(lv_embeds_single, by_track, clean)
    gap_sv = _print_sim_row(f"LaSt-ViT single  sigma={sigma}", same_sv, diff_sv)

    gap_dual = gap_sv  # default return value when no dual mode
    if sigma_embed is not None and sigma_gate is not None:
        print()
        lv_embeds_dual, _ = last_vit_pipeline_dual(
            all_lhs, sigma_embed=sigma_embed, sigma_gate=sigma_gate,
            top_k_ratio=top_k_ratio,
        )
        same_dv, diff_dv = _cosine_pairs(lv_embeds_dual, by_track, clean)
        gap_dual = _print_sim_row(
            f"LaSt-ViT dual    σ_embed={sigma_embed}  σ_gate={sigma_gate}",
            same_dv, diff_dv,
        )

    return gap_base, gap_dual


def sigma_sweep(
    samples: list[CropSample],
    sigmas: list[float] = (0.1, 0.2, 0.3, 0.5, 0.8),
    top_k_ratio: float = 0.5,
) -> None:
    """Sweep sigma values and report discriminability metric."""
    clean = [s for s in samples if s.category == "clean_fg" and s.track_id > 0]
    bg    = [s for s in samples if s.category == "background"]
    if not clean or not bg:
        print("  Not enough samples for sigma sweep.")
        return

    print(f"\n{'='*62}")
    print("SIGMA SWEEP  (Gaussian low-pass width × Nyquist)")
    print(f"{'='*62}")
    results: list[tuple] = []
    print(f"  {'sigma':>7}  {'stab_fg':>8}  {'stab_bg':>8}  "
          f"{'delta':>8}  {'lv_gap':>10}  {'gap_delta':>12}")

    all_lhs = torch.stack([s.lhs for s in clean])
    bg_lhs  = torch.stack([s.lhs for s in bg])

    # Baseline image_embeds gap
    img_embeds = torch.stack([s.img_embed for s in clean])
    by_track: dict[int, list[int]] = {}
    for i, s in enumerate(clean):
        by_track.setdefault(s.track_id, []).append(i)
    same_sig, diff_sig = [], []
    rng = np.random.default_rng(0)
    for tid, idxs in by_track.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                ia, ib = idxs[a], idxs[b]
                same_sig.append(float(F.cosine_similarity(
                    img_embeds[ia].unsqueeze(0), img_embeds[ib].unsqueeze(0))))
    for _ in range(len(same_sig) * 3):
        if len(diff_sig) >= len(same_sig) * 2:
            break
        ia, ib = rng.integers(0, len(clean), size=2)
        if clean[ia].track_id != clean[ib].track_id:
            diff_sig.append(float(F.cosine_similarity(
                img_embeds[ia].unsqueeze(0), img_embeds[ib].unsqueeze(0))))
    baseline_gap = np.array(same_sig).mean() - np.array(diff_sig).mean() if same_sig and diff_sig else 0.0

    for sigma in sigmas:
        _, stab_fg = last_vit_pipeline(all_lhs, sigma=sigma, top_k_ratio=top_k_ratio)
        _, stab_bg = last_vit_pipeline(bg_lhs,  sigma=sigma, top_k_ratio=top_k_ratio)
        sfg = float(stab_fg.mean())
        sbg = float(stab_bg.mean())

        # Cosine gap
        lv_embeds, _ = last_vit_pipeline(all_lhs, sigma=sigma, top_k_ratio=top_k_ratio)
        same_lv, diff_lv = [], []
        for tid, idxs in by_track.items():
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    ia, ib = idxs[a], idxs[b]
                    same_lv.append(float(F.cosine_similarity(
                        lv_embeds[ia].unsqueeze(0), lv_embeds[ib].unsqueeze(0))))
        rng2 = np.random.default_rng(0)
        for _ in range(len(same_lv) * 5):
            if len(diff_lv) >= len(same_lv) * 2:
                break
            ia, ib = rng2.integers(0, len(clean), size=2)
            if clean[ia].track_id != clean[ib].track_id:
                diff_lv.append(float(F.cosine_similarity(
                    lv_embeds[ia].unsqueeze(0), lv_embeds[ib].unsqueeze(0))))
        lv_gap = (np.array(same_lv).mean() - np.array(diff_lv).mean()
                  if same_lv and diff_lv else 0.0)
        improvement = lv_gap - baseline_gap

        results.append((sigma, sfg, sbg, sfg - sbg, lv_gap, improvement))
        print(f"  {sigma:>7.4f}  {sfg:>8.4f}  {sbg:>8.4f}  "
              f"{sfg-sbg:>+8.4f}  {lv_gap:>+10.4f}  {improvement:>+12.4f}")

    print(f"\n  baseline image_embeds gap = {baseline_gap:+.4f}")
    if results:
        best = max(results, key=lambda r: r[5])
        print(f"  best sigma = {best[0]:.4f}  (gap_delta={best[5]:+.4f})")


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def verdict(gap_baseline: float, gap_lastvit: float, stab_pval: float | None) -> None:
    print(f"\n{'='*62}")
    print("VERDICT")
    print(f"{'='*62}")

    stab_ok = stab_pval is not None and stab_pval < 0.05
    gap_ok  = gap_lastvit > gap_baseline

    if stab_ok:
        print("  [PASS] Stability scores significantly discriminate FG vs BG.")
    else:
        print("  [WARN] Stability scores do NOT significantly discriminate FG vs BG.")
        print("         → Consider adjusting sigma or verifying last_hidden_state shape.")

    if gap_ok:
        delta = gap_lastvit - gap_baseline
        print(f"  [PASS] LaSt-ViT embedding gap ({gap_lastvit:+.4f}) > "
              f"baseline ({gap_baseline:+.4f})  (+{delta:.4f}).")
        print("         → Prototype validates concept. Proceed to Phase 1 CUDA.")
    else:
        print(f"  [WARN] LaSt-ViT gap ({gap_lastvit:+.4f}) <= "
              f"baseline ({gap_baseline:+.4f}).")
        print("         → Run --sigma-sweep to find a better sigma before Phase 1.")

    if stab_ok and gap_ok:
        print("\n  ✓ Phase 0 validation PASSED. LaSt-ViT integration is warranted.")
    else:
        print("\n  ✗ Phase 0 validation INCONCLUSIVE. Review parameters.")


# ---------------------------------------------------------------------------
# Background mask sweep (Phase 2C extension)
# ---------------------------------------------------------------------------

def bg_mask_sweep(
    seq_dir: Path,
    extractor: TRTFeatureExtractor,
    frame_ids: list[int],
    gt: dict,
    sigma_embed: float = 0.015,
    sigma_gate: float = 0.040,
    top_k_ratio: float = 0.5,
    modes: list[str] | None = None,
) -> None:
    """
    Compare background mask strategies on:
      A) baseline image_embeds cosine gap (regression check)
      B) LaSt-ViT stability discriminability (FG > BG, Mann-Whitney p)
      C) LaSt-ViT V1 cosine gap (embedding quality)
    """
    try:
        from scipy import stats as _stats
    except ImportError:
        _stats = None  # type: ignore[assignment]

    if modes is None:
        modes = _BG_MASK_MODES

    print(f"\n{'='*88}")
    print("BACKGROUND MASK SWEEP")
    print(f"  σ_embed={sigma_embed}  σ_gate={sigma_gate}  top_k={top_k_ratio}")
    print(f"  frames={frame_ids[:3]}...  modes={modes}")
    print(f"{'='*88}")
    print(f"  {'Mode':<14}  {'base_gap':>9}  {'lv_gap':>9}  {'Δgap':>8}  "
          f"{'stab_fg':>8}  {'stab_bg':>8}  {'p(FG>BG)':>9}  note")
    print(f"  {'-'*14}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}  ----")

    best_mode, best_delta = "none", -999.0

    for mode in modes:
        samples = run_inference(
            seq_dir, extractor, frame_ids, gt,
            sigma=sigma_gate, top_k_ratio=top_k_ratio,
            bg_mask=mode,
        )
        clean = [s for s in samples if s.category == "clean_fg" and s.track_id > 0]
        bg    = [s for s in samples if s.category == "background"]
        if not clean:
            print(f"  {mode:<14}  (no clean_fg)"); continue

        all_lhs   = torch.stack([s.lhs       for s in clean])
        bg_lhs    = torch.stack([s.lhs for s in bg]) if bg else None
        img_emb   = torch.stack([s.img_embed for s in clean])

        by_track: dict[int, list[int]] = {}
        for i, s in enumerate(clean):
            by_track.setdefault(s.track_id, []).append(i)

        rng = np.random.default_rng(42)

        def _gap_quick(embeds: torch.Tensor) -> float:
            same, diff = [], []
            for idxs in by_track.values():
                for a in range(len(idxs)):
                    for b_idx in range(a + 1, len(idxs)):
                        same.append(float(F.cosine_similarity(
                            embeds[idxs[a]].unsqueeze(0), embeds[idxs[b_idx]].unsqueeze(0))))
            n_t = min(len(same) * 3, 500)
            for _ in range(n_t * 8):
                if len(diff) >= n_t: break
                ia, ib = rng.integers(0, len(clean), size=2)
                if clean[ia].track_id != clean[ib].track_id:
                    diff.append(float(F.cosine_similarity(
                        embeds[ia].unsqueeze(0), embeds[ib].unsqueeze(0))))
            sa = np.array(same) if same else np.zeros(1)
            da = np.array(diff) if diff else np.zeros(1)
            return float(sa.mean() - da.mean())

        base_gap = _gap_quick(img_emb)

        lv_embeds, scores_gate = last_vit_pipeline_dual(
            all_lhs, sigma_embed=sigma_embed, sigma_gate=sigma_gate, top_k_ratio=top_k_ratio)
        lv_gap = _gap_quick(lv_embeds)
        delta   = lv_gap - base_gap

        stab_fg = float(scores_gate.mean(dim=-1).mean())
        stab_bg_str = "  n/a  "
        p_str = "   N/A "
        if bg_lhs is not None:
            _, sg_bg = last_vit_pipeline_dual(
                bg_lhs, sigma_embed=sigma_embed, sigma_gate=sigma_gate, top_k_ratio=top_k_ratio)
            stab_bg = float(sg_bg.mean(dim=-1).mean())
            stab_bg_str = f"{stab_bg:.4f}"
            if _stats is not None:
                fg_arr = scores_gate.mean(dim=-1).cpu().numpy()
                bg_arr = sg_bg.mean(dim=-1).cpu().numpy()
                if len(fg_arr) >= 3 and len(bg_arr) >= 3:
                    _, p = _stats.mannwhitneyu(fg_arr, bg_arr, alternative="greater")
                    sig = " ✓" if p < 0.05 else ""
                    p_str = f"{p:.4f}{sig}"

        note = " ← best" if delta > best_delta else ""
        if delta > best_delta:
            best_delta, best_mode = delta, mode

        print(f"  {mode:<14}  {base_gap:>+9.4f}  {lv_gap:>+9.4f}  {delta:>+8.4f}  "
              f"{stab_fg:>8.4f}  {stab_bg_str:>8}  {p_str:>9}{note}")

    print(f"{'='*88}")
    print(f"  Best mode: {best_mode}  (Δgap={best_delta:+.4f})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 0 LaSt-ViT Prototype Validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seq", default="MOT17-04-SDP",
                        help="MOT17 sequence name.")
    parser.add_argument("--data-root", default="datasets/MOT17",
                        help="MOT17 dataset root.")
    parser.add_argument("--engine",
                        default="models/embedding/google_siglip2-base-patch16-224.engine",
                        help="SigLIP2 TRT engine path.")
    parser.add_argument("--frames", nargs="+", type=int, default=None,
                        help="Specific frame IDs. Overrides --n-frames.")
    parser.add_argument("--n-frames", type=int, default=12,
                        help="Number of frames to sample evenly when --frames not given.")
    parser.add_argument("--sigma", type=float, default=0.3,
                        help="Gaussian low-pass sigma for single-sigma mode.")
    parser.add_argument("--sigma-embed", type=float, default=None,
                        help="Dual-sigma: sigma for embedding Top-K (e.g. 0.015).")
    parser.add_argument("--sigma-gate", type=float, default=None,
                        help="Dual-sigma: sigma for stability gating (e.g. 0.030).")
    parser.add_argument("--top-k-ratio", type=float, default=0.5,
                        help="Fraction of patches to include in Top-K pooling.")
    parser.add_argument("--sigma-sweep", action="store_true",
                        help="Sweep sigma values and report discriminability.")
    parser.add_argument("--sigmas", nargs="+", type=float, default=None,
                        help="Custom sigma list for sweep (overrides built-in range).")
    parser.add_argument("--variant-compare", action="store_true",
                        help="Phase 2B: compare V1/V2/V3/V4 aggregation variants.")
    parser.add_argument("--bg-mask-sweep", action="store_true",
                        help="Phase 2C: sweep background mask strategies and compare.")
    parser.add_argument("--bg-masks", nargs="+", default=None,
                        help=f"Subset of mask modes to test. Available: {_BG_MASK_MODES}")
    parser.add_argument("--bg-mask", default="none",
                        help="Single background mask mode to apply during normal inference.")
    args = parser.parse_args()

    seq_dir  = Path(args.data_root) / "train" / args.seq
    gt_path  = seq_dir / "gt" / "gt.txt"

    if not gt_path.exists():
        print(f"ERROR: GT not found at {gt_path}", file=sys.stderr)
        sys.exit(1)

    dual_mode = args.sigma_embed is not None and args.sigma_gate is not None

    print(f"Sequence : {args.seq}")
    print(f"GT       : {gt_path}")
    print(f"Engine   : {args.engine}")
    if dual_mode:
        print(f"mode     : dual-sigma  σ_embed={args.sigma_embed}  σ_gate={args.sigma_gate}  topk={args.top_k_ratio}")
    else:
        print(f"mode     : single-sigma  σ={args.sigma}  topk={args.top_k_ratio}")

    print("\nLoading GT...")
    gt = load_gt(gt_path)
    available = sorted(gt.keys())
    print(f"  {len(available)} frames with pedestrian annotations.")

    if args.frames:
        frame_ids = args.frames
    else:
        step = max(1, len(available) // args.n_frames)
        frame_ids = available[::step][: args.n_frames]
    print(f"  Using frames: {frame_ids}")

    print("\nLoading TRT engine (Python path)...")
    extractor = TRTFeatureExtractor(engine_path=args.engine, device="cuda:0", model_type="siglip2")

    if "last_hidden_state" not in extractor.output_buffers:
        print(
            "ERROR: last_hidden_state not found in engine output buffers.\n"
            "The SigLIP2 engine must be exported with last_hidden_state as an output.\n"
            "Re-export using scripts/model/export_siglip.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    lhs_buf = extractor.output_buffers["last_hidden_state"]
    print(f"  last_hidden_state buffer shape : {tuple(lhs_buf.shape)}")
    print(f"  image_embeds buffer shape      : {tuple(extractor.output_buffers['image_embeds'].shape)}")

    # In dual mode, store gate-sigma stability scores so stability_report reflects gating.
    # Single-sigma mode uses args.sigma for both.
    sigma_for_stab = args.sigma_gate if dual_mode else args.sigma

    # --bg-mask-sweep: run all mask modes and exit
    if args.bg_mask_sweep:
        bg_mask_sweep(
            seq_dir, extractor, frame_ids, gt,
            sigma_embed=args.sigma_embed if dual_mode else args.sigma,
            sigma_gate=args.sigma_gate if dual_mode else args.sigma,
            top_k_ratio=args.top_k_ratio,
            modes=args.bg_masks,
        )
        return

    print("\nRunning inference...")
    samples = run_inference(
        seq_dir, extractor, frame_ids, gt,
        sigma_for_stab, args.top_k_ratio,
        bg_mask=getattr(args, "bg_mask", "none"),
    )
    print(f"\nCollected {len(samples)} crop samples total.")

    if not samples:
        print("ERROR: No samples collected. Check dataset path and frame IDs.", file=sys.stderr)
        sys.exit(1)

    # --- Analysis ---
    stab_arrays = stability_report(samples)
    if dual_mode:
        print(f"  (stability scores computed with σ_gate={args.sigma_gate})")

    pval: float | None = None
    clean_arr = stab_arrays.get("clean_fg", np.array([]))
    bg_arr    = stab_arrays.get("background", np.array([]))
    if len(clean_arr) >= 5 and len(bg_arr) >= 5:
        try:
            from scipy import stats
            _, pval = stats.mannwhitneyu(clean_arr, bg_arr, alternative="greater")
        except ImportError:
            pass

    if args.variant_compare:
        variant_compare(
            samples,
            sigma_embed=args.sigma_embed if dual_mode else args.sigma,
            sigma_gate=args.sigma_gate if dual_mode else args.sigma,
            top_k_ratio=args.top_k_ratio,
        )

    if args.sigma_sweep or args.sigmas:
        sweep_sigmas = args.sigmas if args.sigmas else [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]
        sigma_sweep(samples, sigmas=sweep_sigmas, top_k_ratio=args.top_k_ratio)

    gap_baseline, gap_lastvit = similarity_report(
        samples, args.sigma, args.top_k_ratio,
        sigma_embed=args.sigma_embed if dual_mode else None,
        sigma_gate=args.sigma_gate if dual_mode else None,
    )
    verdict(gap_baseline, gap_lastvit, pval)


if __name__ == "__main__":
    main()
