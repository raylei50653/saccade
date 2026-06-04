# MOT17 ReID Semantic Relink Experiments

Date: 2026-04-25

## Summary

This round tested whether SigLIP2 crop embeddings should participate in MOT17 tracking as the primary association signal or as a conservative post-association relink signal.

Final conclusion:

- Keep the main tracker motion/IoU driven.
- Use ReID only as a conservative semantic relink stage when IDF1 matters.
- Do not enable tracker fusion or hybrid mode as the default on MOT17 SDP.
- Keep advanced crop variants as experimental flags, not production defaults.

Best SDP-wide setting found:

```bash
--reid-mode semantic \
--reid-crop-layout full \
--reid-crop-mode tight \
--semantic-threshold 0.95 \
--semantic-spatial-gate 0.20 \
--semantic-min-iou 0.20 \
--semantic-mahalanobis-threshold 0
```

This improves SDP overall IDF1 from `44.6%` to `45.9%`, keeps MOTA at `32.8%`, and keeps IDs essentially flat (`604 -> 603`), at a large throughput cost (`154.67 FPS -> 83.50 FPS`).

## Implemented Controls

`scripts/eval/mot17.py`（CLI wrapper）透過 `perception/eval/runner.py` 公開明確的 ReID 模式：

```bash
--reid-mode off|tracker|semantic|hybrid
```

Mode behavior:

- `off`: no ReID engine, no crops, no embeddings.
- `tracker`: pass embeddings into C++ tracker association.
- `semantic`: use embeddings only in `SemanticRelinker` post-association relink.
- `hybrid`: enable both tracker fusion and semantic relink.

ReID fusion parameters were made configurable:

```bash
--reid-cos-threshold 0.90
--reid-iou-low 0.30
--reid-iou-high 0.60
--reid-weight 0.40
```

Crop controls were added:

```bash
--reid-crop-mode tight|square|square_mean
--reid-crop-padding 0.0
--reid-crop-layout full|parts
```

Crop behavior:

- `tight`: direct YOLO box RoIAlign to `224x224`.
- `square`: center-expand box to 1:1 RoI before RoIAlign.
- `square_mean`: square RoI, then fill original-box exterior with exterior mean color.
- `parts`: extract full/upper/lower crops and fuse embeddings as `normalize(0.5 full + 0.3 upper + 0.2 lower)`.

Important semantic relink bug fixed:

- Previously, a new raw ID seen without an embedding was permanently aliased to itself, preventing later relink attempts.
- Now missing-embedding frames return the existing alias if present, but do not create a new alias.

## MOT17-09 Single-Sequence Results

Engine: `models/yolo/yolo26s_batch4.engine`

Sequence: `MOT17-09-SDP`

| Config | IDF1 | MOTA | IDs | FPS |
|---|---:|---:|---:|---:|
| `off` | 50.9 | 49.0 | 69 | 148.78 |
| `tracker` | 49.5 | 49.3 | 71 | 109.87 |
| `semantic` tuned | 53.2 | 49.0 | 68 | 92.47 |
| `hybrid` tuned | 51.8 | 49.4 | 70 | 92.10 |

Observation:

- `tracker` can slightly improve MOTA on this sequence, but hurts IDF1 and IDs.
- `semantic` gives the best IDF1 on this sequence.
- `hybrid` improves both IDF1 and MOTA on this sequence, but this did not generalize to SDP full-set.

## SDP Full-Set Results

Sequences:

```text
MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP
```

Baseline output: `results/reid_sdp_off`

Best semantic output: `results/reid_sdp_tune_semantic_t095_i02`

| Config | IDF1 | MOTA | IDs | FPS |
|---|---:|---:|---:|---:|
| `off` | 44.6 | 32.8 | 604 | 154.67 |
| `tracker_default` | 44.9 | 32.8 | 621 | 102.60 |
| `semantic_t097_g010` | 45.2 | 32.8 | 622 | 84.14 |
| `semantic_t098_g010` | 45.0 | 32.8 | 611 | 83.34 |
| `hybrid_t097_g010` | 45.4 | 32.8 | 635 | 82.31 |
| `hybrid_t098_g010` | 45.1 | 32.8 | 619 | 82.86 |
| `semantic_t097_i01` | 45.3 | 32.8 | 599 | 85.70 |
| `semantic_t095_i02` | 45.9 | 32.8 | 603 | 83.50 |
| `hybrid_t097_i01` | 45.5 | 32.8 | 616 | 82.28 |
| `hybrid_t098_i01` | 45.1 | 32.8 | 619 | 82.86 |

Best SDP-wide tradeoff:

```bash
--reid-mode semantic \
--semantic-threshold 0.95 \
--semantic-spatial-gate 0.20 \
--semantic-min-iou 0.20 \
--semantic-mahalanobis-threshold 0
```

Why this was chosen:

- Highest SDP overall IDF1 in the scan: `45.9%`.
- MOTA unchanged: `32.8%`.
- IDs nearly unchanged: `604 -> 603`.
- Hybrid and tracker variants increased IDs more often.

## Occlusion-Heavy Sequences

The semantic relink value is clearer on `MOT17-02-SDP` and `MOT17-04-SDP`.

| Sequence | Off IDF1 | Semantic Tuned IDF1 | Delta | Off IDs | Tuned IDs |
|---|---:|---:|---:|---:|---:|
| `MOT17-02-SDP` | 31.2 | 33.1 | +1.9 | 87 | 82 |
| `MOT17-04-SDP` | 44.2 | 45.8 | +1.6 | 117 | 112 |

Interpretation:

- ReID is useful when treated as a conservative long-chain repair mechanism.
- It reduces fragmentation on hard occlusion sequences.
- The current embedding is still not reliable enough to dominate per-frame association.

## Crop Experiments

All crop probes used `MOT17-02-SDP` and `MOT17-04-SDP`.

### Square Crop

| Crop | 02 IDF1 | 04 IDF1 | Overall IDF1 | IDs | FPS |
|---|---:|---:|---:|---:|---:|
| `tight` | 33.1 | 45.8 | 42.5 | 194 | 67.66 |
| `square + 0.15` | 33.1 | 45.9 | 42.6 | 195 | 64.68 |
| `square + 0.10` | 32.9 | 45.9 | 42.6 | 194 | 64.66 |

Conclusion:

- Square crop has no meaningful improvement over tight crop.
- It slightly helps `MOT17-04`, but not enough to justify changing the default.

### Parts Layout

| Layout | 02 IDF1 | 04 IDF1 | Overall IDF1 | IDs | FPS |
|---|---:|---:|---:|---:|---:|
| `full` | 33.1 | 45.8 | 42.5 | 194 | 67.66 |
| `parts` | 33.1 | 45.9 | 42.6 | 186 | 39.61 |
| `parts + square` | 33.1 | 46.2 | 42.8 | 185 | 39.05 |

Conclusion:

- Local body-part semantics have real signal.
- `parts + square` gave the best 02/04 crop-probe IDF1 and IDs.
- The cost is high because it performs 3x crops and 3x embedding inference.
- Keep it as a high-precision experiment, not a default.

### Square Mean Fill

| Mode | 02 IDF1 | 04 IDF1 | Overall IDF1 | IDs | FPS |
|---|---:|---:|---:|---:|---:|
| `square_mean 0.10 full` | 32.4 | 45.8 | 42.3 | 193 | 65.49 |
| `square_mean 0.15 full` | 32.4 | 45.9 | 42.4 | 194 | 65.14 |
| `parts + square_mean 0.10` | 33.1 | 45.9 | 42.6 | 188 | 38.53 |

Conclusion:

- Mean-fill did not improve IDF1.
- It may remove useful context along with noisy neighboring-person context.
- Keep as an experimental option only.

## Interpretation

The stable architecture is:

```text
Main tracking: motion / IoU
Long-chain repair: conservative semantic relink
```

Current SigLIP crop embeddings are useful but not calibrated enough for primary association:

- Tracker fusion changes per-frame assignment and can amplify crop jitter or appearance ambiguity.
- Semantic relink only acts after fragmentation and can be gated by similarity, spatial distance, and minimum IoU.
- This makes semantic relink safer for crowded pedestrian tracking.

The crop experiments suggest:

- Aspect-ratio distortion is not the only bottleneck.
- Body-part embeddings provide signal, but the current implementation is too expensive.
- NaFlex remains a good accuracy probe, but not yet a production TensorRT path.

## Recommended Defaults

Speed/default path:

```bash
--reid-mode off
```

IDF1-focused path:

```bash
--reid-mode semantic \
--reid-crop-layout full \
--reid-crop-mode tight \
--semantic-threshold 0.95 \
--semantic-spatial-gate 0.20 \
--semantic-min-iou 0.20 \
--semantic-mahalanobis-threshold 0
```

High-precision experiment:

```bash
--reid-mode semantic \
--reid-crop-layout parts \
--reid-crop-mode square \
--reid-crop-padding 0.10
```

Not recommended as defaults:

- `--reid-mode tracker`
- `--reid-mode hybrid`
- `--reid-crop-mode square_mean`

## Backbone Comparison Refresh (2026-04-29)

This archive refreshes the older 2026-04-25 SigLIP2 conclusions on the current tracking base after GMC, semantic buffer, primary appearance bank, `need_reid`, and conditional appearance matching were already integrated.

The purpose was not to re-open the old `tracker vs semantic vs hybrid` question. It was to check whether a dedicated person-ReID backbone could now beat the current SigLIP2 relink path once each model was given a more appropriate semantic threshold.

### Setup

- Eval script: `scripts/eval/ablation_reid_models.py`
- Detector: `SDP`
- Current tracker/relink base: same defaults as `scripts/eval/mot17.py`
- Key hypothesis: SigLIP2 cosine scores are relatively high, so `transreid` and `osnet` must be evaluated with lower relink thresholds instead of inheriting SigLIP2's `0.90`

### SigLIP2 vs TransReID

| Model@thr | IDF1 | Rcll | Prcn | MOTA | IDs | FP | FN | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `siglip2@0.88` | 45.1 | 50.2 | 77.0 | 34.5 | 852 | 16808 | 55892 | 60.38 |
| `siglip2@0.90` | 45.4 | 50.2 | 77.1 | 34.6 | 846 | 16738 | 55894 | 59.10 |
| `siglip2@0.92` | 45.1 | 50.3 | 77.1 | 34.6 | 848 | 16734 | 55833 | 59.45 |
| `transreid@0.78` | 44.3 | 50.4 | 77.1 | 34.6 | 888 | 16771 | 55743 | 64.03 |
| `transreid@0.82` | 44.3 | 50.5 | 77.1 | 34.7 | 885 | 16836 | 55608 | 63.36 |
| `transreid@0.86` | 44.5 | 50.5 | 77.2 | 34.8 | 829 | 16729 | 55616 | 63.98 |
| `transreid@0.90` | 43.8 | 50.4 | 77.1 | 34.6 | 851 | 16806 | 55753 | 63.04 |

Conclusions:

- The threshold-mismatch hypothesis was correct. `transreid@0.90` was too strict and artificially suppressed relink acceptance.
- `transreid` recovered once the threshold moved down to the `0.78-0.86` range.
- Best `transreid` operating point on this base was `0.86`.
- Even at its own best threshold, `transreid` did not beat `siglip2@0.90` on IDF1 (`44.5 < 45.4`).
- `transreid@0.86` remained interesting as a secondary profile because it reduced IDs (`829`) and slightly improved MOTA (`34.8`), but its accuracy ceiling was still lower.

### SigLIP2 vs OSNet

OSNet required a dedicated TensorRT path before it could be measured on the same base:

- ONNX: `models/embedding/osnet_x1_0_256x128.onnx`
- Engine: `models/embedding/osnet_x1_0_256x128.engine`
- Build script: `uv run python scripts/model/build_osnet.py`

| Model@thr | IDF1 | Rcll | Prcn | MOTA | IDs | FP | FN | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `siglip2@0.88` | 45.5 | 50.3 | 77.0 | 34.4 | 879 | 16915 | 55819 | 60.89 |
| `siglip2@0.90` | 45.3 | 50.2 | 77.1 | 34.5 | 845 | 16774 | 55904 | 61.28 |
| `siglip2@0.92` | 44.2 | 50.2 | 77.0 | 34.5 | 854 | 16842 | 55887 | 60.78 |
| `osnet@0.78` | 44.3 | 50.3 | 77.1 | 34.7 | 845 | 16769 | 55756 | 67.11 |
| `osnet@0.82` | 44.6 | 50.2 | 77.1 | 34.5 | 842 | 16734 | 55963 | 69.98 |
| `osnet@0.86` | 44.2 | 50.4 | 77.1 | 34.6 | 848 | 16830 | 55723 | 69.73 |
| `osnet@0.90` | 44.1 | 50.3 | 77.1 | 34.6 | 840 | 16798 | 55847 | 67.01 |

Conclusions:

- Best `osnet` operating point on this base was around `0.82`.
- `osnet` also failed to beat SigLIP2 on IDF1 (`44.6 < 45.3`).
- Its main value was speed: roughly `6-9 FPS` faster than SigLIP2 in this eval path.
- `osnet` is therefore a viable speed-balanced profile, not a new accuracy ceiling.

### Archived Outcome

For the current base, the backbone ranking is:

- Accuracy profile: `siglip2@0.90`
- Lower-IDs / slightly higher-MOTA alternative: `transreid@0.86`
- Speed-balanced alternative: `osnet@0.82`

The main bottleneck after this refresh is no longer "wrong ReID backbone by default". It is scene-adaptive identity recovery and relink policy design. Replacing SigLIP2 with `transreid` or `osnet` alone did not break the current IDF1 ceiling.

## Follow-Up Work

1. ~~Add candidate-gated parts extraction~~  **→ Implemented as Farewell ReID (2026-04-26)**
   - SigLIP2 now fires only when a track disappears, not on every heartbeat.
   - Takes the last ≤3 frames from a 5-frame buffer, L2-normalized average → FeatureBank.
   - Achieves the same "extract only when it matters" principle without body-part complexity.

2. Add NaFlex accuracy probe:
   - Use PyTorch or ONNXRuntime first.
   - Treat it as an upper-bound experiment before attempting TensorRT.

3. Add model-specific relink diagnostics:
   - Focus on why `02/11` remain the main pressure sequences even after backbone swaps.
   - Record accepted/rejected relink counts by model and threshold, not just final MOT metrics.

4. Add diagnostics:
   - Log accepted relink pairs with similarity, IoU, center distance, old/new ID, and frame.
   - Inspect false relinks on sequences where IDs increase.

5. Re-run full 21-sequence MOT17 train after any candidate-gated crop change.
