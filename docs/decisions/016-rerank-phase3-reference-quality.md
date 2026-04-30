# ADR 016: Rerank Phase 3 - Reference Quality and False-Accept Filtering

## Status
Proposed

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
