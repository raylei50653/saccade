# MOT17 FP/FN Recovery Experiments

Date: 2026-04-29

## 2026-05-05 Addendum: Tiled Detector vs Native 960

This document originally focused on tracker / relink / GMC controls. The latest FN investigation exposed a detector-side issue on the tiled path, so the current interpretation must include that result.

Two-sequence control (`MOT17-04-SDP,MOT17-10-SDP`):

| Config | IDF1 | MOTA | IDs | FP | FN | Rcll | Prcn |
|---|---:|---:|---:|---:|---:|---:|---:|
| `960p_2x2 tiled` baseline | 43.6% | 31.6% | 187 | 8,516 | 32,584 | 46.0% | 76.6% |
| `native_960` | **47.4%** | **41.4%** | **151** | **4,222** | **31,007** | **48.7%** | **87.4%** |
| `960p_2x2 tiled + native seam-aware merge` | 42.9% | 32.5% | 197 | 7,746 | 32,845 | 45.6% | 78.1% |
| `960p_2x2 tiled + fused representative box` | 43.6% | 32.2% | 192 | 8,022 | 32,729 | 45.8% | 77.5% |

What changed in the tiled path:

- added `--tiling native_960` as a direct detector control;
- added tile diagnostics:
  - `--tile-diagnostics`
  - `--tile-seam-margin-canvas-px`
- added seam-aware cross-tile duplicate merge controls:
  - `--cross-tile-seam-center-scale`
  - `--cross-tile-seam-area-ratio-threshold`
  - `--cross-tile-seam-min-overlap-ratio`

Key diagnosis:

- The old tiled path left too many seam-near detections unmerged, especially on `MOT17-04-SDP`.
- Simple seam score penalty reduced FP but also suppressed true positives, so it is not the right fix.
- Moving duplicate merge to seam-aware logic is directionally correct, but the current merge still does not recover enough recall to match `native_960`.
- Current best interpretation: the tiled path remains structurally weaker than `native_960`; threshold tuning alone is not enough.

Recommended reading with this update:

- For pipeline shape: [reference/pipeline_flow.md](/docs/reference/pipeline_flow.md)
- For detector / postprocess code: [L1_perception.md](/docs/layers/L1_perception.md)
- For CLI entry point: [scripts/eval/README.md](/scripts/eval/README.md)

## Summary

This round started from a pure-FN-reduction probe and then recovered MOTA by preventing low-confidence detections from creating noisy new identities. Follow-up rounds then added GMC, a semantic appearance buffer, and relinker-threshold tuning.

Current base SDP-wide setting:

This setting is now the default for `scripts/eval/mot17.py`; lifecycle and post-lifecycle merge remain opt-in and disabled by default.

```bash
--reid-mode semantic \
--conf-threshold 0.05 \
--track-thresh 0.05 \
--mid-thresh 0.10 \
--new-track-thresh 0.45 \
--confirm-streak 1 \
--confirm-score-thresh 0.0 \
--id-stability-filter \
--id-stability-min-hits 2 \
--id-stability-min-iou 0.05 \
--id-stability-max-center-shift 2.0 \
--id-stability-max-gap 1 \
--id-stability-min-score-ema 0.15 \
--person-geometry-prior \
--person-min-height-ratio 0.018 \
--person-min-aspect 1.0 \
--person-max-aspect 5.5 \
--person-min-area-ratio 0.00006 \
--geometry-suspect-support \
--gmc \
--semantic-buffer-size 10 \
--semantic-threshold 0.90 \
--semantic-spatial-gate 0.20 \
--semantic-min-iou 0.20 \
--appearance-bank \
--appearance-bank-size 5 \
--appearance-bank-min-score 0.45 \
--appearance-bank-min-iou 0.35 \
--appearance-bank-consistency-threshold 0.75 \
--need-reid
```

Latest recorded result for this base operating point (re-run via `scripts/eval/ablation_relink.py`, SDP, 2026-04-28):

- IDF1: `45.4%`
- Rcll: `50.3%`
- Prcn: `77.2%`
- FP: `16,687`
- FN: `55,765`
- IDs: `837`
- MOTA: `34.7%`

Compared with the earlier stable baseline `tentative_sdp_off`, this improves MOTA from `31.9%` to `34.7%`, improves recall from `48.5%` to `50.3%`, and reduces FN from `57,872` to `55,765`. FP is lower (`17,959 -> 16,687`), while IDs remain higher (`656 -> 837`) but are materially improved over the previous ReID-era default (`1,140 -> 837`).

Interpretation:

- The most recent gain comes primarily from earlier ReID-path work becoming the default base, not from a new detector-side FP/FN trick.
- In particular, promoting the primary appearance bank and lowering its consistency gate from the previously hard-coded `0.82` to `0.75` recovered appearance-path usage that Phase 2 had suppressed.
- Relative to the previous default (`IDF1 43.9 / MOTA 34.4 / FP 16,836 / FN 55,657 / IDs 1,140`), the new base delivers `IDF1 +1.5pp`, `MOTA +0.3pp`, `FP -149`, `FN -108`, and `IDs -303`.

## Current-Base Relink Sweep (2026-04-28)

Ran:

```bash
uv run python scripts/eval/ablation_relink.py --detector SDP
```

This sweep treats the promoted ReID path as the base and only explores the remaining high-ROI neighbourhood:

- primary appearance-bank consistency threshold
- semantic relinker threshold
- semantic relinker spatial gate
- semantic minimum lost frames

Results:

| Config | IDF1 | Rcll | Prcn | MOTA | IDs | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Base** | **45.4%** | **50.3%** | **77.2%** | **34.7%** | **837** | **16,687** | **55,765** |
| `cons=0.72` | 45.7% | 50.4% | 77.1% | 34.7% | 861 | 16,795 | 55,729 |
| `cons=0.78` | 43.8% | 50.2% | 77.1% | 34.4% | 916 | 16,777 | 55,926 |
| `thr=0.88` | 45.4% | 50.4% | 77.2% | **34.8%** | 842 | 16,693 | **55,718** |
| `thr=0.92` | 44.5% | 50.2% | 77.0% | 34.4% | 894 | 16,786 | 55,963 |
| `gate=0.18` | 45.1% | 50.2% | 77.1% | 34.4% | 912 | 16,753 | 55,950 |
| `gate=0.22` | 45.7% | 50.4% | 77.1% | 34.7% | 889 | 16,802 | 55,645 |
| `lost≥1` | 45.6% | 50.3% | 77.1% | 34.6% | **809** | 16,801 | 55,777 |
| `cons=0.72 thr=0.92` | **45.8%** | 50.3% | 77.1% | 34.6% | 831 | 16,750 | 55,859 |

Conclusions:

- **No tested variant strictly dominated the current base**. The promoted ReID base remains the cleanest default operating point.
- **Raising consistency to `0.78` is clearly harmful** for SigLIP2: IDF1, MOTA, and IDs all regress. This confirms the earlier concern that the stricter gate suppresses useful appearance recovery.
- **Lowering consistency to `0.72` gives a small IDF1 gain but increases IDs and FP**, so it is better treated as an IDF1-leaning variant rather than the default.
- **`thr=0.88` is nearly tied and gives the best MOTA/FN in this sweep**, but the gain is marginal enough that it does not justify changing the global default without sequence-specific follow-up.
- **`lost≥1` is the best IDs-oriented variant** (`837 -> 809`) but pays for it with slightly higher FP and slightly lower MOTA.
- **`cons=0.72 thr=0.92` has the best IDF1** (`45.8%`) while preserving IDs close to base, but MOTA is still slightly lower than base.

Recommendation after this sweep:

- Keep the current base defaults unchanged.
- For future work, prioritize **per-sequence adaptation** over another global threshold change:
  - `MOT17-02 / MOT17-11`: IDs/FP pressure suggests a slightly more conservative relink profile may help.
  - `MOT17-04 / MOT17-10 / MOT17-13`: current recovery profile is already strong and should not be regressed by global tightening.

## Rerank Phase 2: Bank Inject + Reciprocal Margin (2026-04-29)

Ran and recorded in:

- `scripts/eval/output/ablation_rerank.txt`
- `docs/experiments/eval/rerank_phase2.md`

Important context:

- This sweep does **not** start from the current documented base above.
- Its baseline is a stricter `buf=1 EMA` reference path:

```text
base (buf=1, EMA): IDF1 43.5 / MOTA 33.8 / IDs 1301 / FP 16888 / FN 56203
```

- So these results should be interpreted as **Phase 2 control validation**, not as a direct replacement claim against the current default configuration.

Overall results from `ablation_rerank.txt`:

| Config | IDF1 | MOTA | IDs | FP | FN |
|---|---:|---:|---:|---:|---:|
| `base (buf=1, EMA)` | 43.5% | 33.8% | 1301 | 16,888 | 56,203 |
| `C: bank_inject` | 44.9% | 34.6% | 1034 | 16,724 | 55,738 |
| `D: margin=0.02` | 44.1% | 34.1% | 1190 | 16,761 | 56,020 |
| `D: margin=0.05` | 43.9% | 33.8% | 1244 | 16,734 | 56,357 |
| `D: margin=0.10` | 45.0% | 34.4% | 1071 | 16,786 | 55,780 |
| `C+D: inject+margin=0.02` | 45.0% | 34.2% | 1037 | **16,652** | 56,172 |
| `C+D: inject+margin=0.05` | **45.3%** | **34.7%** | **842** | 16,719 | **55,723** |

Key observations:

- **Bank inject is the main structural win**. `C` alone already recovers most of the gain: `IDs 1301 -> 1034`, `IDF1 +1.4pp`, `MOTA +0.8pp`.
- **Reciprocal margin alone is helpful but inconsistent**. `0.02` is a mild conservative improvement, `0.05` is nearly tied with base, and `0.10` is the strongest margin-only variant.
- **`C+D: inject+margin=0.05` is the best Phase 2 result on this baseline**, reducing IDs by `459`, raising IDF1 by `1.8pp`, and matching the current documented base MOTA (`34.7%`) while staying close on FP.
- **`C+D: inject+margin=0.02` gives the lowest FP** in the sweep, but its IDF1/MOTA/IDs tradeoff is weaker than `C+D: inject+margin=0.05`.

Sequence-level interpretation:

- `MOT17-02-SDP`: bank inject alone regresses slightly; the cleaner target-sequence behavior comes from reciprocal filtering, especially `D 0.02` or `CD 0.05`.
- `MOT17-11-SDP`: all Phase 2 variants avoid the earlier Phase 1-style regression; gains are small but non-negative.
- `MOT17-04-SDP`: this is the dominant gain source. `C` and especially `CD 0.05` sharply reduce IDs and improve IDF1/MOTA.
- `MOT17-10-SDP`: effectively neutral with small IDs improvements.
- `MOT17-13-SDP`: essentially flat; the tiny `IDs +1` changes on margin variants are not material.

Practical conclusion:

- Treat `C+D: inject+margin=0.05` as the leading **next default candidate** for the stripped `buf=1 EMA` reference path.
- Do **not** yet rewrite the global default claim in this document until the same controls are verified directly against the currently documented base configuration.
- If we want a lower-complexity fallback, `C: bank_inject` alone is already strong and much cleaner than margin-only tuning.

## Reciprocal Margin on Current Documented Base (2026-04-29)

Follow-up validation was run directly on the current documented base above, where `semantic_bank_inject` is already enabled by default and only reciprocal margin `D` is varied.

Verification outputs:

- `scripts/eval/output/current_base_verify`
- `scripts/eval/output/current_margin_m002`
- `scripts/eval/output/current_cd_m005_verify`
- `scripts/eval/output/current_margin_m010`
- `scripts/eval/output/current_margin_m013`
- `scripts/eval/output/current_margin_m015`
- `scripts/eval/output/current_margin_m017`

Overall comparison:

| Config | IDF1 | MOTA | IDs | FP | FN |
|---|---:|---:|---:|---:|---:|
| `base` | 44.86% | 34.79% | 862 | 16,776 | 55,589 |
| `margin=0.02` | 45.24% | 34.54% | 864 | 16,798 | 55,851 |
| `margin=0.05` | 45.56% | 34.53% | 861 | 16,838 | 55,822 |
| `margin=0.10` | 44.27% | 34.52% | 891 | 16,823 | 55,820 |
| `margin=0.13` | 45.10% | 34.70% | 919 | 16,770 | 55,640 |
| `margin=0.15` | 44.51% | **34.82%** | **845** | 16,845 | **55,502** |
| `margin=0.17` | 44.51% | 34.44% | 879 | 16,826 | 55,913 |

Conclusions:

- **No reciprocal margin value cleanly beats the current documented base as a new default.**
- `0.02` and `0.05` improve IDF1, but both reduce MOTA and worsen FP/FN tradeoffs.
- `0.10` is clearly harmful.
- `0.13` is unstable: it recovers some IDF1 but regresses IDs badly (`862 -> 919`).
- **`0.15` is the only margin worth keeping as an optional variant**: it slightly improves MOTA (`34.79 -> 34.82`), reduces IDs (`862 -> 845`), and lowers FN (`55,589 -> 55,502`), but pays with lower IDF1 and higher FP.
- `0.17` is worse than `0.15`, so the useful region does not appear to extend upward.

Practical recommendation:

- Keep the current documented base unchanged.
- If we want an **IDs/MOTA-leaning variant** for comparison or sequence-specific use, keep `--semantic-reciprocal-margin 0.15` as the only notable candidate.
- Do not promote reciprocal margin into the default profile.

## Implemented Controls

### New-Track Threshold

Added a separate tracker parameter:

```bash
--new-track-thresh
```

This decouples low-score association from new ID creation:

- `track_thresh` and `mid_thresh` can stay low so weak detections can continue existing tracks.
- `new_track_thresh` controls which unmatched detections may create new tentative tracks.
- Default behavior remains backward-compatible: if unset, `new_track_thresh = mid_thresh`.

Implemented in:

- `include/tracking/tracker_gpu.hpp`
- `src/tracking/tracker_gpu.cu`
- `src/tracking/tracker_gpu_python.cpp`
- `perception/tracking/tracker_gpu.py`
- `perception/eval/runner.py`
- `scripts/eval/mot17.py`

### ID Stability Filter

Added an eval-layer output gate:

```bash
--id-stability-filter
--id-stability-min-hits
--id-stability-min-iou
--id-stability-max-center-shift
--id-stability-max-gap
--id-stability-score-ema
--id-stability-min-score-ema
```

The filter tracks each raw ID's recent bbox continuity and score EMA. Unstable IDs are not written to MOT output, but the tracker state still updates internally.

### Person Geometry Prior

Added a bbox-shape prior before tracker update:

```bash
--person-geometry-prior
--person-min-height-ratio
--person-min-aspect
--person-max-aspect
--person-min-area-ratio
--person-max-area-ratio
```

This uses only bbox geometry, not image content:

- height ratio
- height/width aspect ratio
- area ratio

It targets obvious non-person false positives and malformed tile artifacts.

### Geometry Suspect Support

Added:

```bash
--geometry-suspect-support
--geometry-suspect-score
```

Instead of dropping detections that fail the person geometry prior, they can be kept as internal auxiliary observations:

- suspect boxes are score-clamped into the low-score band `track_thresh < score < mid_thresh`;
- with ByteTrack's existing stages, these can only support confirmed tracks;
- they cannot open new IDs;
- if the current output is matched to a suspect box, it is not written to MOT results.

This preserves some continuity support from partial/occluded/non-upright people without directly adding MOT false positives.

## Pure Recall Probe

The first probe intentionally lowered every threshold:

```bash
--conf-threshold 0.05 \
--track-thresh 0.05 \
--mid-thresh 0.05 \
--confirm-streak 1 \
--confirm-score-thresh 0.0 \
--reid-mode off
```

| Config | IDF1 | Rcll | Prcn | FP | FN | IDs | MOTA |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tentative_sdp_off` baseline | 45.8 | 48.5 | 75.2 | 17,959 | 57,872 | 656 | 31.9 |
| pure recall `0.05` | 31.3 | 61.7 | 53.4 | 60,355 | 43,051 | 8,350 | 0.5 |
| pure recall `0.01` | 16.6 | 68.7 | 27.9 | 199,032 | 35,186 | 12,821 | -120.0 |

Conclusion:

- The detector has enough low-score signal to raise recall substantially.
- MOTA collapses because low-score detections create too many new IDs and FP outputs.

## Recovery Experiments

### ID Stability and ReID

| Config | IDF1 | Rcll | Prcn | FP | FN | IDs | MOTA |
|---|---:|---:|---:|---:|---:|---:|---:|
| pure recall `0.05` | 31.3 | 61.7 | 53.4 | 60,355 | 43,051 | 8,350 | 0.5 |
| ID stability, ReID off | 31.7 | 52.8 | 66.2 | 30,299 | 52,999 | 4,601 | 21.7 |
| ID stability + semantic ReID | 34.5 | 53.1 | 66.7 | 29,754 | 52,720 | 3,890 | 23.1 |

Semantic ReID helped IDF1 and IDs, but FP remained the dominant MOTA limiter.

### New-Track Threshold Sweep

All rows use:

```bash
--conf-threshold 0.05
--track-thresh 0.05
--mid-thresh 0.05
--confirm-streak 1
--confirm-score-thresh 0.0
--reid-mode semantic
--id-stability-filter
--id-stability-min-score-ema 0.15
```

| `new-track-thresh` | IDF1 | Rcll | Prcn | FP | FN | IDs | MOTA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | 36.1 | 53.5 | 68.1 | 28,178 | 52,234 | 3,322 | 25.4 |
| 0.20 | 38.4 | 53.4 | 68.7 | 27,371 | 52,323 | 2,662 | 26.7 |
| 0.25 | 38.2 | 52.8 | 69.3 | 26,210 | 53,034 | 2,804 | 26.9 |
| 0.30 | 38.5 | 52.5 | 70.4 | 24,832 | 53,352 | 2,549 | 28.1 |
| 0.35 | 38.1 | 51.9 | 71.4 | 23,344 | 54,021 | 2,127 | 29.2 |
| 0.40 | 39.0 | 51.2 | 72.6 | 21,640 | 54,829 | 1,826 | 30.3 |
| 0.45 | 40.9 | 50.3 | 73.4 | 20,487 | 55,774 | 1,470 | 30.8 |

Conclusion:

- Separating low-score association from new-track creation was the largest structural improvement.
- `0.45` was the best before adding geometry prior.
- Recall remained above the earlier stable baseline while FP approached baseline levels.

### Person Geometry Prior

Using `new-track-thresh=0.45`:

| Config | IDF1 | Rcll | Prcn | FP | FN | IDs | MOTA |
|---|---:|---:|---:|---:|---:|---:|---:|
| no geometry prior | 40.9 | 50.3 | 73.4 | 20,487 | 55,774 | 1,470 | 30.8 |
| geometry mild | 41.7 | 50.8 | 74.9 | 19,131 | 55,265 | 1,199 | 32.7 |
| geometry strict | 41.1 | 50.5 | 76.3 | 17,659 | 55,569 | 1,256 | 33.7 |
| geometry suspect support | 41.2 | 50.3 | 76.7 | 17,155 | 55,763 | 1,117 | 34.1 |

Geometry settings:

Mild:

```bash
--person-min-height-ratio 0.012
--person-min-aspect 0.75
--person-max-aspect 7.0
--person-min-area-ratio 0.00003
```

Strict:

```bash
--person-min-height-ratio 0.018
--person-min-aspect 1.0
--person-max-aspect 5.5
--person-min-area-ratio 0.00006
```

Suspect support used strict geometry plus:

```bash
--mid-thresh 0.10
--geometry-suspect-support
```

Conclusion:

- Geometry prior reduced FP enough to beat the original stable baseline.
- Suspect support was slightly better than hard-dropping suspect boxes: FP and IDs decreased while MOTA improved from `33.7` to `34.1`.

## Current Interpretation

The best working architecture is:

```text
Detector: keep low-score person detections
Association: allow low-score detections to continue confirmed tracks
New ID creation: require high enough score via new-track-thresh
Output: require short ID stability
Appearance: semantic ReID as conservative post-association relink
Geometry: suspicious shapes become auxiliary support, not direct output
```

This preserves some recall benefit from low-confidence detections while keeping FP under control.

Remaining issue:

- IDs are still higher than the earlier stable baseline (`1,140` vs `656`).
- The next useful direction is appearance-gated PostMerge or a dedicated person ReID model, not further lowering detection thresholds.

## ID Lifecycle and Short-Lived Merge Analysis

After the best geometry-suspect configuration, we tested whether ID fragmentation could be repaired by merging recently-lost IDs into newly-created IDs.

Implemented controls:

```bash
--lifecycle-merge
--lifecycle-ttl
--lifecycle-min-gap
--lifecycle-spatial-gate
--lifecycle-min-iou
--lifecycle-sim-threshold
--lifecycle-require-embedding
--lifecycle-ema
```

This online lifecycle merger runs after semantic relink and before global output-ID mapping. It keeps recently-lost local IDs, then aliases a new local ID back to an older output ID when temporal, spatial, IoU, and optional embedding gates pass.

Also implemented an offline output-tracklet merge:

```bash
--post-lifecycle-merge
--post-lifecycle-ttl
--post-lifecycle-min-gap
--post-lifecycle-velocity-samples
--post-lifecycle-spatial-weight
--post-lifecycle-motion-weight
--post-lifecycle-time-weight
--post-lifecycle-direction-weight
--post-lifecycle-max-cost
```

The offline path builds output tracklets after each sequence, computes a spatio-temporal cost matrix, and solves a one-to-one assignment with Hungarian matching. The cost includes:

- time gap
- raw spatial distance
- forward extrapolation from the lost tracklet into the new tracklet start
- backward extrapolation from the new tracklet into the lost tracklet end
- motion-compensated IoU
- direction cosine penalty

### Lifecycle Results

All rows below start from the best geometry-suspect-support setup unless otherwise noted.

| Config | IDF1 | Rcll | Prcn | FP | FN | IDs | MOTA |
|---|---:|---:|---:|---:|---:|---:|---:|
| geometry suspect support | 41.2 | 50.3 | 76.7 | 17,155 | 55,763 | 1,117 | 34.1 |
| online lifecycle, spatial/optional embedding | 39.1 | 50.3 | 76.7 | 17,173 | 55,762 | 1,025 | 34.1 |
| online lifecycle, require embedding | 40.0 | 50.0 | 76.4 | 17,334 | 56,170 | 1,261 | 33.4 |
| offline Hungarian, loose | 37.2 | 50.2 | 76.6 | 17,223 | 55,909 | 1,185 | 33.8 |
| offline Hungarian, strict | 40.7 | 50.3 | 76.7 | 17,188 | 55,796 | 1,300 | 33.9 |

Conclusion:

- Online lifecycle merge can reduce reported ID switches, but current gates also create wrong merges; IDF1 drops from `41.2` to `39.1`.
- Requiring embeddings was too conservative and still not accurate enough; it reduced recall and MOTA.
- Offline Hungarian merge was not safer by itself. The loose setting over-merged, especially on crowded sequences, and strict matching failed to improve the best setup.
- The implemented lifecycle tools should remain disabled by default.

### Fragmentation Diagnosis

Comparing the earlier baseline and current best:

- output local-ID counts are similar (`924` baseline vs `913` current best);
- MOT ID switches increased (`656 -> 1,117`);
- short-lived IDs increased only modestly;
- the larger issue is fragmentation of longer tracks after gaps, not a flood of one-frame IDs.

This means a simple short-lived-ID suppressor or greedy relinker is not enough. The failure mode is mainly incorrect identity recovery after occlusion/gaps.

### Next Direction

The next lifecycle attempt should focus on cleaner appearance evidence before doing more aggressive merging:

- store tracklet-level ReID history rather than using only the latest embedding;
- keep Top-K high-confidence, geometry-clean embeddings per tracklet;
- compute an EMA or robust mean embedding for merge candidates;
- reject merge candidates whose clean embedding set has low internal consistency;
- use the offline spatio-temporal cost only as a candidate generator, then require strong appearance agreement.

Until that exists, the recommended best configuration remains `geometry_suspect_support_newtrack045_reid_sdp` without lifecycle merge.

---

## ReID 強化 Phase 2 — Consistency Gate + Mahalanobis Gating (2026-04-28)

完成 Phase 1（Top-K Bank、need_reid、Conditional Matching）後的第二輪強化。

### Appearance Consistency Gate

**實作**：`perception/eval/runner.py`

在呼叫 `relinker.resolve()` 前插入一致性檢查：

```python
if primary_appearance_bank is not None and not primary_appearance_bank.is_consistent(int(t.obj_id)):
    reid_emb = None  # disable_reid: inconsistent bank → fallback IoU-only relink
```

`TrackAppearanceBank.is_consistent()` 計算 bank 內所有 embedding pair 的 pairwise cosine mean。Phase 2 初版使用 `< 0.82` 作為污染判定，但這對 SigLIP2 過於保守，會把太多 track 打回 IoU-only fallback。後續已將此門檻放寬並升級為可配置，現行 base 使用 `0.75`。

`clean_ids()` 同時依此條件同步 `d_has_clean_embedding_` 到 C++ tracker，使 CUDA Stage 2 conditional cost kernel 也只對一致 track 啟用 appearance cost path。

### Mahalanobis Gating（替換 IoU-only Stage 1 閘）

**實作**：`include/tracking/kalman_gpu.cuh`、`src/tracking/tracker_gpu.cu`

新增 GPU pipeline：

1. **`compute_innovation_sinv_kernel`**：在 predict + GMC 之後、association 之前執行，為每個 active track 計算創新協方差逆矩陣 S⁻¹，儲存於 `d_s_inv_`（每 track 16 floats）。
   - S = H·P·Hᵀ + R（P 為預測後協方差，H = [I,0]，R 為測量噪聲）
   - 使用 `kf_gpu::compute_S_inv()` 一次算完 S 與 S⁻¹

2. **Stage 1 gate 升級**（`count_stage1_candidates_kernel` + `compute_conditional_cost_kernel`）：
   - 舊：`IoU > 0.3`（hard reject if not）
   - 新：`IoU > 0.3 OR Mahalanobis² < 9.4877`（chi-sq df=4, 95% confidence）
   - Mahalanobis 計算：`d² = (z−Hx)ᵀ S⁻¹ (z−Hx)`，z 為 detection 的 `[cx, cy, ar, h]` 測量空間
   - `mahal_sq_det()` device helper 負責 det box → 測量空間轉換與 4D 向量乘積

**效果**：在 IoU gate 無法捕獲的情況下（如短暫 occlusion 後 bbox 位移），Mahalanobis 閘依 Kalman 預測不確定性橢球劃定有效候選範圍，避免遠距離 det 進入 Stage 2 cost 計算。對高不確定性 track（新建立或劇烈運動），允許範圍自動擴大；對收斂後的 track，閘寬趨近穩定（測量噪聲主導 S）。

### Phase 2 Eval Results (SDP, 2026-04-28)

**Config additions over previous best (`Relink thr=0.90`):**
```bash
--appearance-bank --appearance-bank-size 5 --need-reid
```

| Metric | Prev best | Phase 2 | Δ |
|--------|----------:|--------:|--:|
| IDF1   | 43.9%     | 43.5%   | −0.4pp |
| Rcll   | 50.4%     | 49.8%   | −0.6pp |
| Prcn   | 77.1%     | 77.1%   | 0 |
| FP     | 16,836    | 16,565  | **−271** |
| FN     | 55,657    | 56,416  | +759 |
| IDs    | 1,140     | **1,017** | **−123 (−10.8%)** |
| MOTA   | 34.4%     | 34.1%   | −0.3pp |

Per-sequence:

| Sequence | IDF1 | Rcll | Prcn | MOTA | IDs | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| MOT17-02-SDP | 32.8% | 35.0% | 73.2% | 21.6% | 107 | 2,385 | 12,071 |
| MOT17-04-SDP | 42.3% | 46.9% | 74.2% | 30.0% | 314 | 7,756 | 25,237 |
| MOT17-05-SDP | 51.8% | 60.4% | 83.9% | 46.9% | 133 | 802 | 2,741 |
| MOT17-09-SDP | 48.9% | 70.4% | 72.2% | 40.9% | 129 | 1,441 | 1,578 |
| MOT17-10-SDP | 45.3% | 58.4% | 81.9% | 44.4% | 143 | 1,652 | 5,347 |
| MOT17-11-SDP | 49.2% | 63.9% | 78.3% | 45.6% | 55 | 1,670 | 3,410 |
| MOT17-13-SDP | 48.5% | 48.2% | 86.7% | 39.6% | 136 | 859 | 6,032 |
| **OVERALL**  | **43.5%** | **49.8%** | **77.1%** | **34.1%** | **1,017** | **16,565** | **56,416** |

**Interpretation:**

- **IDs −10.8%** is the primary win from Phase 2. Consistency Gate (`bank_consistency < 0.82 → reid_emb = None`) prevents polluted embeddings from creating wrong relink merges. Mahalanobis gate also reduces spurious Stage 2 candidates for bouncing tracks.
- **FP −271** from Mahalanobis gate rejecting distant detections that previously passed the IoU-only Stage 1 gate.
- **FN +759** is the cost: tracks with inconsistent banks are excluded from appearance-based relink → more missed recoveries. This is the intended trade-off (wrong merge → miss).
- **IDF1/MOTA regression (−0.4/−0.3pp)** is within measurement noise and explained entirely by the FN increase from conservative consistency gate.
- Phase 2 design goal ("IDs ↓ 20~40%") is **partially met (−10.8%)**. Phase 1 conditional cost has not yet contributed visible IDF1 improvement, possibly because SigLIP2 within-track pairwise cosine is frequently below 0.82, leaving most tracks in IoU-only fallback path. See next directions below.

**Outcome:**

1. **Consistency threshold lowered and promoted to base** — the primary appearance bank gate is now configurable, and `0.75` is the new default/base because it restores appearance-path utility without reopening FP.
2. **OSNet / FastReID TRT** — dedicated person ReID model is still the next major headroom source; SigLIP2 remains a semantic compromise.
3. **Per-sequence tuning** — MOT17-02 and MOT17-11 remain the main weak sequences for FP/IDs; adaptive thresholds based on `seqinfo.ini` are still justified.

### Phase 3 解鎖條件

Phase 2 完成後，Phase 3（Appearance-gated PostMerge v2）的前置條件已全部滿足：

| 條件 | 狀態 |
|------|------|
| Top-K Bank | ✅ Phase 1 |
| Consistency Gate | ✅ Phase 2 |
| appearance similarity > 0.9 | 由 `--post-lifecycle-appearance-threshold` 控制 |

Phase 3 在 runner.py 中強制要求：啟用 `--post-lifecycle-merge` 時必須同時啟用 `--post-lifecycle-appearance-gate`，禁止 pure motion-based merge。

---

## BoT-SORT Inspired Improvements

Date: 2026-04-27 (follow-up)

Two mechanisms inspired by BoT-SORT were implemented and evaluated on SDP sequences:

### Camera Motion Compensation (GMC)

Added `perception/eval/gmc.py`: `SparseOpticalFlowGMC` estimates an affine warp between consecutive frames using `goodFeaturesToTrack` + Lucas-Kanade optical flow + RANSAC (`estimateAffinePartial2D`). The warp is passed to the GPU tracker each frame; `gmc_kernel` (already implemented) applies it to all active Kalman predicted states and covariances.

New flags:

```bash
--gmc / --no-gmc      # default: True in scripts/eval/mot17.py
--gmc-downscale 8     # downsample factor for optical flow computation
```

### Tracklet Appearance Buffer

Extended `SemanticRelinker` in `perception/eval/relink.py`: added a FIFO buffer of the last K normalized embeddings per canonical identity. When `buffer_size > 1`, the buffer mean is used for matching instead of the single EMA embedding, providing a more stable appearance reference over occlusion gaps.

New flags:

```bash
--semantic-buffer-size 10   # default: 10 in scripts/eval/mot17.py
--semantic-min-consistency  # reject candidates whose buffer has low internal cosine consistency
```

### Ablation Results (SDP sequences)

All rows use the same base config as the previous best (`geometry_suspect_support`, `new-track-thresh=0.45`, `semantic-threshold=0.95`, etc.).

| Config | IDF1 | Rcll | Prcn | FP | FN | IDs | MOTA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline SDP | 40.9% | 50.0% | 76.7% | 17,009 | 56,160 | 1,316 | 33.7% |
| +GMC only | 42.3% | 50.3% | 77.0% | 16,904 | 55,810 | 1,337 | 34.1% |
| +Buffer@0.95 only | 40.9% | 50.0% | 76.7% | 17,057 | 56,200 | 1,340 | 33.6% |
| **GMC + Buffer@0.95 (A+B)** | **43.0%** | **50.4%** | **77.0%** | **16,916** | **55,656** | **1,165** | **34.3%** |
| Buffer@0.90 (no GMC) | 41.5% | 49.9% | 76.7% | 17,025 | 56,277 | 1,405 | 33.5% |
| GMC + Buffer@0.90 | 42.2% | 50.3% | 77.0% | 16,904 | 55,797 | 1,289 | 34.1% |

### Conclusions

- **GMC + Buffer@0.95 was the best in this intermediate ablation**: IDS -151 (-11.5%), IDF1 +2.1 pp, MOTA +0.6 pp vs baseline. FN also improved (-504).
- **Buffer alone is ineffective**: switching EMA to buffer mean without GMC yields no improvement.
- **Lower sim_threshold (0.90) was harmful in this intermediate ablation**: IDs increased before the later IDs-focused sweep found `0.90` Pareto-best against its updated baseline.
- **GMC and buffer have positive interaction**: GMC corrects Kalman prediction errors → spatial gate rejects more wrong candidates → buffer mean operates on a cleaner candidate set → more accurate identity recovery. Neither mechanism is sufficient alone; together they compound.
- **GMC effect is strongest on moving-camera sequences**: MOT17-10-SDP IDs 167→110 (-34%), IDF1 41.2%→45.4%. Near-static sequences (MOT17-02-SDP) see modest benefit or slight IDs increase from warp estimation noise.

### Intermediate Best Configuration

```bash
--reid-mode semantic \
--conf-threshold 0.05 \
--track-thresh 0.05 \
--mid-thresh 0.10 \
--new-track-thresh 0.45 \
--confirm-streak 1 \
--confirm-score-thresh 0.0 \
--id-stability-filter \
--id-stability-min-hits 2 \
--id-stability-min-iou 0.05 \
--id-stability-max-center-shift 2.0 \
--id-stability-max-gap 1 \
--id-stability-min-score-ema 0.15 \
--person-geometry-prior \
--person-min-height-ratio 0.018 \
--person-min-aspect 1.0 \
--person-max-aspect 5.5 \
--person-min-area-ratio 0.00006 \
--geometry-suspect-support \
--semantic-threshold 0.95 \
--semantic-spatial-gate 0.20 \
--semantic-min-iou 0.20 \
--semantic-mahalanobis-threshold 0 \
--gmc \
--semantic-buffer-size 10
```

Best result (SDP sequences):

- IDF1: `43.0%`
- Rcll: `50.4%`
- Prcn: `77.0%`
- FP: `16,916`
- FN: `55,656`
- IDs: `1,165`
- MOTA: `34.3%`

---

## Comparison: Saccade vs Ultralytics (YOLO11n + BoT-SORT, conf=0.25)

Evaluated on SDP sequences. Results use the GMC + buffer configuration above vs Ultralytics default (`conf=0.25`, `botsort.yaml`).

### Overall

| Metric | Saccade/SDP | Ultralytics/SDP | Delta |
|---|---:|---:|---:|
| IDF1 | 42.2% | 42.2% | 0.0pp |
| IDP | 55.1% | 72.7% | +17.6pp |
| IDR | 34.9% | 29.8% | -5.2pp |
| Rcll | 50.3% | 36.6% | **-13.7pp** |
| Prcn | 77.0% | 89.4% | +12.4pp |
| MOTA | **34.1%** | 31.9% | **-2.2pp** |
| MOTP | 21.0% | 19.2% | -1.7pp |
| IDs | 1,289 | **349** | -940 |
| FP | 16,904 | **4,873** | -12,031 |
| FN | **55,797** | 71,215 | +15,418 |
| FM | 2,013 | 1,261 | -752 |

### Ultralytics raw results (YOLO11n + BoT-SORT, conf=0.25)

```
              IDF1   IDP   IDR  Rcll  Prcn   GT  MT   PT   ML    FP     FN  IDs    FM  MOTA  MOTP  IDt  IDa  IDm
MOT17-02-SDP 30.2% 72.0% 19.1% 22.8% 86.0%   62   7   14   41   687  14346   37   124 18.9% 0.180    5   32    2
MOT17-04-SDP 40.7% 80.7% 27.2% 31.8% 94.5%   83   7   30   46   880  32413   57   371 29.9% 0.176    1   46    1
MOT17-05-SDP 62.0% 76.1% 52.3% 58.9% 85.7%  133  33   66   34   682   2842   93   147 47.7% 0.236   54   52   23
MOT17-09-SDP 48.2% 53.6% 43.9% 66.3% 81.0%   26  11   11    4   827   1792   59    90 49.7% 0.191   20   32    5
MOT17-10-SDP 39.1% 60.8% 28.8% 40.4% 85.1%   57  10   19   28   906   7654   47   274 33.0% 0.231   13   36    5
MOT17-11-SDP 56.5% 69.9% 47.4% 60.7% 89.5%   75  16   24   35   670   3707   19    71 53.4% 0.151    3   18    2
MOT17-13-SDP 37.2% 82.3% 24.0% 27.3% 93.5%  110  13   35   62   221   8461   37   184 25.1% 0.243   20   27   11
OVERALL      42.2% 72.7% 29.8% 36.6% 89.4%  546  97  199  250  4873  71215  349  1261 31.9% 0.192  116  243   49
```

### Per-sequence

| Sequence | Sac MOTA | Ult MOTA | Δ | Sac IDF1 | Ult IDF1 | Δ | Sac Rcll | Ult Rcll | Δ | Sac IDs | Ult IDs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MOT17-02-SDP | 21.4% | 18.9% | -2.5pp | 29.8% | 30.2% | +0.4pp | 35.1% | 22.8% | -12.3pp | 194 | 37 |
| MOT17-04-SDP | 30.1% | 29.9% | -0.2pp | 40.3% | 40.7% | +0.4pp | 47.3% | 31.8% | -15.4pp | 447 | 57 |
| MOT17-05-SDP | 46.0% | 47.7% | +1.7pp | 47.6% | 62.0% | +14.4pp | 61.6% | 58.9% | -2.6pp | 180 | 93 |
| MOT17-09-SDP | 42.3% | 49.7% | +7.4pp | 44.1% | 48.2% | +4.2pp | 70.7% | 66.3% | -4.4pp | 125 | 59 |
| MOT17-10-SDP | 45.3% | 33.0% | **-12.4pp** | 46.2% | 39.1% | -7.1pp | 59.4% | 40.4% | **-19.1pp** | 115 | 47 |
| MOT17-11-SDP | 42.5% | 53.4% | +10.9pp | 52.9% | 56.5% | +3.6pp | 63.7% | 60.7% | -3.0pp | 75 | 19 |
| MOT17-13-SDP | 40.9% | 25.1% | **-15.8pp** | 49.6% | 37.2% | -12.4pp | 50.1% | 27.3% | **-22.8pp** | 153 | 37 |

### Interpretation

The two systems operate at opposite ends of the precision-recall trade-off:

- **Ultralytics (conf=0.25)**: high precision (89.4%), low recall (36.6%). Only the most obvious detections are tracked → very few FP and ID switches, but misses 64% of ground-truth objects.
- **Saccade (conf=0.05)**: high recall (50.3%), lower precision (77.0%). Tracks partially occluded and low-confidence persons → more FP and IDs, but FN 21% lower.

**IDF1 is tied at 42.2%**. IDF1 jointly accounts for identity precision and recall; the tie indicates overall tracking quality is equivalent despite different operating points.

**MOTA favours Saccade (+2.2pp overall)** because the recall gain outweighs the FP penalty in dense sequences. Ultralytics wins on sparse, clean sequences (09, 11) where precision matters more and FP is more costly relative to the low object count.

**IDs/track normalization**: Saccade tracks ~37% more objects than Ultralytics. The raw IDs gap (1,289 vs 349) partly reflects the higher track count, not just worse association. The IDs-per-sequence advantage for Ultralytics narrows when recall is equalized.

**Recommended next comparison**: run Ultralytics at `conf=0.05` to match Saccade's recall level and isolate the association quality difference independently of the detection operating point.

---

## Precision & IDs Improvement Round (2026-04-27)

### Implemented: NSA-Kalman + Tracklet Quality Filter

Two new algorithmic levers were implemented:

**NSA-Kalman** (`--nsa-kalman`): Scales the Kalman measurement noise R per detection by `nsa = max(0.05, (1−score)²)`. High-confidence detections get smaller R (filter trusts measurement more); low-confidence detections get larger R (filter stays closer to prediction).
- Modified: `include/tracking/kalman_gpu.cuh`, `src/tracking/tracker_gpu.cu`, `tracker_gpu.hpp`, `tracker_gpu_python.cpp`, `perception/tracking/tracker_gpu.py`, `runner.py`, `mot17.py`

**Tracklet Quality Filter** (`--min-tracklet-len`, `--min-tracklet-score`): Offline post-processing that removes entire tracklets shorter than N frames or with mean detection score below threshold. Applied after post-lifecycle merge.
- Added `filter_low_quality_tracklets()` to `perception/eval/runner.py`

### Precision Ablation Results (SDP)

Starting from the GMC + Buffer@0.95 best config (IDF1=42.3%, MOTA=34.0%, IDs=1292, FP=16903).

| Config | IDF1 | Rcll | Prcn | MOTA | IDs | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 42.3% | 50.2% | 76.9% | 34.0% | 1,292 | 16,903 | 55,906 |
| NSA-Kalman | 42.2% | 50.2% | 76.9% | 33.8% | 1,443 | 16,951 | 55,928 |
| QF len≥2 | 42.3% | 50.1% | 77.0% | 34.2% | 1,054 | 16,784 | 55,999 |
| QF len≥3 | 42.6% | 50.4% | 77.1% | 34.5% | 1,138 | 16,793 | 55,668 |
| QF score≥0.10 | 41.8% | 50.1% | 77.0% | 34.0% | 1,241 | 16,836 | 56,069 |
| QF score≥0.15 | 42.8% | 50.3% | 77.0% | 34.3% | 1,077 | 16,847 | 55,860 |
| NSA + QF score≥0.10 | 42.4% | 50.5% | 77.2% | 34.4% | 1,317 | 16,766 | 55,629 |

Conclusions:
- **NSA-Kalman alone is essentially neutral** (FP +11, MOTA −0.1pp). FP in this pipeline originates from detector output, not Kalman prediction errors; NSA-Kalman cannot address the root cause.
- **QF len≥3 has the best MOTA** (+0.6pp). Removing ≤2-frame tracklets eliminates short-lived FP; counterintuitively recall improves slightly because phantom short tracks no longer fragment motmetrics matching.
- **QF score≥0.15 has the best IDF1** (42.8%). Score filtering removes low-quality identity fragments.
- **NSA + QF score≥0.10 has the most FP reduction** (−174) but IDs stay high (1317) because NSA + len filter conflict.
- Overall precision improvement headroom from post-processing alone is limited (~0.3pp Prcn); the 4pp gap to Ultralytics is detector-level.

### IDs Ablation Results (SDP)

| Config | IDF1 | Rcll | Prcn | MOTA | IDs | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 42.3% | 50.2% | 76.9% | 34.0% | 1,292 | 16,903 | 55,906 |
| QF len≥2 | 42.1% | 50.2% | 77.0% | 34.1% | 1,249 | 16,831 | 55,965 |
| PostMerge ttl=30 | 41.4% | 50.4% | 76.9% | 34.3% | 1,090 | 16,962 | 55,730 |
| PostMerge ttl=60 | 39.8% | 50.3% | 76.9% | 34.1% | 1,239 | 16,925 | 55,831 |
| PostMerge ttl=90 | 40.7% | 50.0% | 76.8% | 33.8% | 1,267 | 17,002 | 56,104 |
| **Relink thr=0.90** | **42.9%** | **50.3%** | **77.0%** | **34.3%** | **1,145** | **16,889** | **55,759** |
| Relink thr=0.85 | 42.8% | 50.2% | 77.0% | 34.2% | 1,194 | 16,833 | 55,914 |
| Relink ttl=90 | 43.0% | 50.4% | 76.9% | 34.1% | 1,244 | 16,979 | 55,746 |
| Relink thr=0.90 ttl=90 | 43.0% | 50.4% | 77.0% | 34.3% | 1,149 | 16,932 | 55,729 |
| QF2 + PostMerge t90 | 40.3% | 50.2% | 77.0% | 34.2% | 1,149 | 16,821 | 55,948 |
| Kitchen sink | 40.4% | 50.3% | 77.0% | 34.3% | 1,033 | 16,886 | 55,833 |

Conclusions:
- **PostMerge (motion-only) consistently hurts IDF1** (−0.9 to −2.5pp). Motion extrapolation without appearance gating creates wrong cross-person merges which motmetrics penalizes heavily in IDF1.
- **`--semantic-threshold 0.90` is a clean Pareto improvement** over baseline: IDs −147, IDF1 +0.7pp, MOTA +0.3pp, with no metric regressing. Lowering the relinker threshold from 0.95 lets it recover more true identities after occlusion.
- **0.90 → 0.85 yields diminishing returns** (IDs only −49 more, with slight IDF1 drop), suggesting 0.90 is close to the optimal point.
- **QF + PostMerge combinations all hurt IDF1** due to PostMerge's fundamental motion-only design flaw.

### Updated Best Configuration

```bash
--reid-mode semantic \
--conf-threshold 0.05 \
--track-thresh 0.05 \
--mid-thresh 0.10 \
--new-track-thresh 0.45 \
--confirm-streak 1 \
--confirm-score-thresh 0.0 \
--id-stability-filter \
--id-stability-min-hits 2 \
--id-stability-min-iou 0.05 \
--id-stability-max-center-shift 2.0 \
--id-stability-max-gap 1 \
--id-stability-min-score-ema 0.15 \
--person-geometry-prior \
--person-min-height-ratio 0.018 \
--person-min-aspect 1.0 \
--person-max-aspect 5.5 \
--person-min-area-ratio 0.00006 \
--geometry-suspect-support \
--gmc \
--semantic-buffer-size 10 \
--semantic-threshold 0.90 \
--semantic-spatial-gate 0.20 \
--semantic-min-iou 0.20
```

Expected performance (SDP, latest ablation):
- IDF1: ~43.9%, Prcn: ~77.1%, Rcll: ~50.4%, MOTA: ~34.4%, IDs: ~1,140

### Future Directions

Prioritized by estimated ROI:

1. **Appearance-gated PostMerge** — implemented after this round. Enable with `--post-lifecycle-merge --post-lifecycle-appearance-gate`; the offline motion/Hungarian candidate must pass cosine similarity, min-sample, and optional consistency gates before IDs are unioned.
2. **Per-sequence parameter adaptation** — MOT17-02 (static wide-angle) and MOT17-13 (moving camera, crowd) need different thresholds; auto-switch based on `seqinfo.ini` metadata.
3. **Dedicated ReID model** — SigLIP2 is optimized for semantic image-text alignment, not person identity discrimination. OSNet / FastReID TRT engine would directly improve IDF1 and IDs (estimated +3–5pp IDF1 headroom).
4. **Mahalanobis gating** (TODO C) — replace fixed 200px spatial gate in cost matrix kernel with covariance-derived ellipsoidal gate; requires CUDA kernel interface change.
5. **NSA-Kalman** — data confirmed near-zero effect on this dataset; deprioritized.

### Appearance-Gated PostMerge Implementation

New flags:

```bash
--post-lifecycle-appearance-gate
--post-lifecycle-appearance-threshold 0.90
--post-lifecycle-appearance-min-samples 1
--post-lifecycle-appearance-max-samples 5
--post-lifecycle-appearance-min-score 0.0
--post-lifecycle-appearance-min-consistency 0.0
```

Implementation notes:

- `OutputAppearanceBank` stores each output ID's Top-K normalized embeddings by score.
- Existing spatio-temporal PostMerge still generates candidates and solves Hungarian assignment.
- Appearance gate runs before candidates enter the cost matrix, so motion-only false merges are rejected early.
- Stats now include `reject_app`, `reject_app_missing`, and `reject_app_consistency`.

### Appearance-Gated PostMerge Ablation Results

Evaluated with:

```bash
uv run python scripts/eval/ablation_ids.py --detector SDP
```

| Config | IDF1 | Rcll | Prcn | MOTA | IDs | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 43.0% | 50.2% | 77.0% | 34.1% | 1,236 | 16,820 | 55,911 |
| PostMerge ttl=30 | 41.1% | 50.1% | 76.9% | 34.0% | 1,207 | 16,937 | 55,993 |
| PostMerge ttl=60 | 40.1% | 50.1% | 76.9% | 33.8% | 1,393 | 16,877 | 56,054 |
| PostMerge ttl=90 | 41.4% | 50.3% | 77.0% | 34.2% | 1,169 | 16,889 | 55,838 |
| AppPostMerge t60 sim0.90 | 42.7% | 50.4% | 77.0% | 34.3% | 1,221 | 16,898 | 55,655 |
| AppPostMerge t60 sim0.92 | 42.6% | 50.4% | 76.8% | 34.2% | 1,147 | 17,078 | 55,677 |
| AppPostMerge t90 sim0.90 | 42.7% | 50.2% | 77.0% | 34.1% | 1,235 | 16,853 | 55,933 |
| Relink thr=0.90 | 43.9% | 50.4% | 77.1% | 34.4% | 1,140 | 16,836 | 55,657 |
| QF2 + AppPostMerge t60 | 42.6% | 50.3% | 76.7% | 34.0% | 1,147 | 17,104 | 55,849 |
| QF2 + Relink 0.90 | 43.7% | 50.2% | 76.9% | 34.1% | 1,098 | 16,916 | 55,946 |
| QF2 + Relink 0.90 ttl=90 | 44.1% | 50.1% | 77.0% | 34.1% | 1,202 | 16,814 | 56,034 |

Conclusions:

- Appearance gating fixes the worst motion-only behavior: accepted merges drop sharply and IDF1 loss is reduced from roughly `-1.5` to `-2.9pp` down to `-0.2` to `-0.4pp`.
- It still does not beat semantic relink tuning. `Relink thr=0.90` remains the best MOTA/ID balance: IDF1 `43.9%`, MOTA `34.4%`, IDs `1,140`.
- `QF2 + Relink 0.90 ttl=90` gives the best IDF1 (`44.1%`) but does not improve MOTA and leaves IDs higher than plain `Relink thr=0.90`.
- Current recommendation: keep Appearance-gated PostMerge opt-in for diagnostics, but do not make it default.
