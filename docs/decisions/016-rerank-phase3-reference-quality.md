# ADR 016: Rerank Phase 3 - Reference Quality and False-Accept Filtering

## Status
Implemented (2026-05-01) — pending A2 ablation to confirm optimal threshold values

## Context
Following the completion of Rerank Phase 1 and Phase 2, we determined that the main bottleneck for further MOTA/IDF1 improvement is not the multi-sample appearance scoring mechanism (Phase 1) nor the reciprocal margin thresholding alone (Phase 2). Rather, the bottleneck is **reference quality** and **false-accept filtering**. 

When a track is occluded, truncated, or severely malformed, injecting its embedding into the semantic memory bank corrupts the identity representation. Later, when the track reappears, a false accept is more likely due to this corrupted or ambiguous reference.

## Objective
Implement Phase 3 to proactively filter low-quality references from entering the semantic memory and apply stricter rules to prevent false accepts during identity resolution.

## Technical Plan

### 1. Reference Quality Gate (Bank Injection Filter)
We will introduce a `ReferenceQualityFilter` logic to gate which embeddings are allowed to update the `SemanticRelinker` and `TrackAppearanceBank`.
The criteria for a "clean" reference will be:
- **Confidence Score**: Must be >= a `clean_score_threshold` (e.g., 0.85).
- **Edge Truncation**: Bounding box must not touch the image boundaries (e.g., within 1% of the frame edges).
- **Aspect Ratio**: For person tracking, the aspect ratio (h/w) must be within a realistic human range (e.g., 1.2 <= ratio <= 4.5).

### 2. False-Accept Filtering (Association Gate)
During the `SemanticRelinker::resolve()` phase:
- If a candidate embedding has low "current observation quality" (e.g., score < 0.6), we will require a **higher similarity threshold** or **stricter spatial/motion gate** to accept the match.
- This creates a dynamic threshold: high-quality observations can pass with normal thresholds, while low-quality ones require overwhelming evidence.

## Consequences
- **Positive**: Reduces Identity Switches (IDs) and False Positives (FP) by preventing identity contamination.
- **Negative**: May slightly increase fragmentation (FN) if legitimate but low-quality tracklets are prevented from relinking.

## Future Work: LaSt-ViT Backbone Integration

Phase 3 operates as a **post-hoc heuristic filter** on embedding output. A complementary **pre-hoc** fix exists at the backbone level.

**LaSt-ViT** (arXiv:2602.22394, CVPR 2026) identifies "lazy aggregation" in standard ViT: the CLS token aggregates global semantics by using background patches as shortcuts, contaminating the embedding even when the bounding box is clean. The fix applies 1D FFT along the channel dimension, selects the most frequency-stable (foreground-representative) patches, and aggregates only those into the CLS token.

**Relevance to ADR 016:**
- The current `conf_score + aspect_ratio` gate catches geometrically bad samples but cannot detect semantically contaminated embeddings from background-dominated CLS tokens.
- LaSt-ViT-B/16 outputs 768-dim CLS tokens — **fully compatible with the existing C++ tracker and TrackAppearanceBank** (no recompilation required).
- FFT overhead is negligible (< 0.01% additional FLOPs vs. standard ViT-B/16).
- No direct ReID benchmark data exists yet; the foreground-alignment advantage is validated on 12 dense-prediction benchmarks (ImageNet/COCO/ADE20K), not on Market-1501 or MOT17.

**Recommended A2-L Ablation (cheap → expensive, escalate only on miss):**

| Phase | Action | Cost | Escalation rule |
|---|---|---|---|
| 1 | Attention-guided pooling on existing backbone | 1–2 days | If IDs ↓ > 5% → **adopt and stop** (skip Phase 2/3) |
| 2 | Zero-shot `LaSt-ViT-B/16-DINO` backbone swap | 2–3 days | Only if Phase 1 misses; if IDs ↓ > 5% → proceed to Phase 3 |
| 3 | Market-1501 fine-tune + Phase 3 threshold re-tune | ~1 week | Only if Phase 2 validates |

After any phase succeeds, re-run A1 (association score) and A2 (threshold sweep) on top of the new backbone — optimal threshold values may shift.

**Phase 3 threshold adjustment direction (if backbone yields cleaner embeddings):**

| Parameter | Current | Suggested test range |
|---|---|---|
| `high_quality_min_score` | 0.70 | 0.60–0.65 |
| `consistency_threshold` | 0.82 | 0.75–0.80 |
| `strict_sim_threshold` | 0.0 (disabled) | 0.55–0.65 (start at 0.60) |

Full research analysis: [`docs/research/reid/last_vit_integration_analysis.md`](../research/reid/last_vit_integration_analysis.md)
