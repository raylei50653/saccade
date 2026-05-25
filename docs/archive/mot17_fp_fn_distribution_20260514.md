# MOT17 FP/FN Distribution Snapshot (2026-05-14)

目的：把目前 `MOT17` 的 `FP/FN` 分布用現成腳本整理成可復盤的固定紀錄，避免後續只剩對話摘要。

## Scope

- 日期：`2026-05-14`
- `FP` 分布：
  - 輸入：`results/mot17_sdp_external_fp_rows.csv`
  - 腳本：`scripts/eval/analyze_external_fp_rows.py`
  - 覆蓋 sequence：`MOT17-02/04/09/10-SDP`
  - 總 rows：`95,390`
- `FN` 分布：
  - 輸入：`results/MOT17_rule_promotion`
  - 腳本：`scripts/eval/analyze_fn.py`
  - 覆蓋 sequence：`MOT17-02/04/05/09/10/11/13-SDP`

## Repro Commands

```bash
uv run python scripts/eval/analyze_external_fp_rows.py \
  --rows-csv results/mot17_sdp_external_fp_rows.csv \
  --output-json results/mot17_sdp_external_fp_rows_analysis.json

uv run python scripts/eval/analyze_fn.py \
  --results results/MOT17_rule_promotion \
  --gt-root datasets/MOT17/train \
  --sequences MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP \
  --visibility-threshold 0.6 \
  --top-runs 12 \
  --csv results/mot17_rule_promotion_fn_runs.csv \
  --frame-csv results/mot17_rule_promotion_fn_frames.csv

uv run python scripts/eval/analyze_near_miss_offsets.py \
  --results results/MOT17_rule_promotion \
  --gt-root datasets/MOT17/train \
  --sequences MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP \
  --visibility-threshold 0.6 \
  --simulate-box-refine \
  --output-csv results/mot17_rule_promotion_near_miss_offsets.csv \
  --output-json results/mot17_rule_promotion_near_miss_offsets.json
```

## Output Files

- `results/mot17_sdp_external_fp_rows_analysis.json`
- `results/mot17_rule_promotion_fn_runs.csv`
- `results/mot17_rule_promotion_fn_frames.csv`
- `results/mot17_rule_promotion_near_miss_offsets.csv`
- `results/mot17_rule_promotion_near_miss_offsets.json`

## FP Snapshot

Counts from `results/mot17_sdp_external_fp_rows_analysis.json`:

- `TP = 2,386`
- `FP = 67,576`
- `IGNORE = 25,428`

Per-sequence row counts from `results/mot17_sdp_external_fp_rows.csv`:

| Sequence | TP | FP | IGNORE | Total |
|---|---:|---:|---:|---:|
| MOT17-02-SDP | 1,402 | 13,628 | 5,214 | 20,244 |
| MOT17-04-SDP | 47 | 38,183 | 11,543 | 49,773 |
| MOT17-09-SDP | 588 | 5,650 | 2,538 | 8,776 |
| MOT17-10-SDP | 349 | 10,115 | 6,133 | 16,597 |

Quantiles:

| Feature | TP median | FP median |
|---|---:|---:|
| score | 0.384 | 0.269 |
| height | 104.0 | 122.5 |
| aspect_ratio | 2.845 | 2.642 |

Score buckets:

| Score bucket | TP | FP |
|---|---:|---:|
| `[0.05, 0.10)` | 349 | 16,770 |
| `[0.10, 0.20)` | 404 | 12,224 |
| `[0.20, 0.40)` | 477 | 11,604 |
| `[0.40, 0.60)` | 350 | 9,781 |
| `[0.60, 0.80)` | 385 | 8,825 |
| `[0.80, 1.01)` | 421 | 8,372 |

Height buckets:

| Height bucket | TP | FP |
|---|---:|---:|
| `[0, 32)` | 0 | 668 |
| `[32, 64)` | 330 | 11,672 |
| `[64, 96)` | 712 | 10,854 |
| `[96, 128)` | 461 | 12,418 |
| `[128, 192)` | 183 | 19,916 |
| `[192, 256)` | 256 | 7,628 |
| `[256, 512)` | 369 | 3,103 |
| `[512, 4096)` | 75 | 1,317 |

Rule baseline simulation:

- `FP removed = 9,021`
- `FP reduction = 13.35%`
- `TP removed = 153`
- `recall_after = 93.6%`

Current reading:

- MOT17 `FP` 不是單純「低分小框尾巴」。
- `FP` 和 `TP` 在 `score` 上高度重疊。
- `FP` 中位高度甚至高於 `TP`，因此 generic low-score rule filter 只能小幅改善。

## FN Snapshot

Counts from `results/mot17_rule_promotion_fn_frames.csv`:

- `MISS / FN = 50,740`
- `high-vis FN (vis >= 0.6) = 9,563`
- `high-vis share = 18.85%`
- `FN height median = 137`
- `high-vis FN height median = 111`

Per-sequence summary:

| Sequence | MISS | high-vis | high-vis share | miss h median |
|---|---:|---:|---:|---:|
| MOT17-04-SDP | 22,099 | 3,397 | 15.4% | 161 |
| MOT17-02-SDP | 10,895 | 1,127 | 10.3% | 83 |
| MOT17-13-SDP | 5,393 | 2,170 | 40.2% | 67 |
| MOT17-10-SDP | 4,386 | 1,402 | 32.0% | 95 |
| MOT17-05-SDP | 3,343 | 535 | 16.0% | 122 |
| MOT17-11-SDP | 3,227 | 840 | 26.0% | 162 |
| MOT17-09-SDP | 1,397 | 92 | 6.6% | 274 |

All-FN height bins:

| Height bin | Count |
|---|---:|
| `<64` | 7,044 |
| `64-128` | 14,793 |
| `128-256` | 25,197 |
| `>=256` | 3,706 |

High-visibility FN height bins:

| Height bin | Count |
|---|---:|
| `<64` | 2,777 |
| `64-128` | 2,532 |
| `128-256` | 3,908 |
| `>=256` | 346 |

Best-IoU diagnostics against nearby predictions:

- `best_iou == 0`: `13,902`
- `best_iou >= 0.1`: `26,916`
- `best_iou >= 0.5`: `5,347`

Current reading:

- `FN` 主體不是單一 edge-case，而是中尺寸人群 recall 問題，峰值落在 `128-256 px`。
- `MOT17-04-SDP` 是最大的 `FN` 壓力源。
- `MOT17-13/10/11-SDP` 的 high-visibility miss 比例偏高，代表有較多「理應可追回」的 recall 損失。

## Box-Offset Hypothesis

暫時判讀：`FN` 裡確實有一塊可能是框偏移 / localization error，而不只是完全沒偵測到。

理由：

- `FN` 中只有 `13,902 / 50,740` 是 `best_iou == 0`
- 有大量 `FN` 附近存在 prediction，但 IoU 沒有過 match threshold
- 長 miss run 中常見 `best_iou_median` 落在 `0.1 ~ 0.4`

但不能把全部 `FN` 都解讀成框偏移：

- `best_iou == 0` 的比例仍然不低
- `MOT17-04-SDP` 有不少長 run 幾乎沒有可用鄰近框，更像真漏檢 / 遮擋 / 截斷問題

### Best-IoU Bucket Validation

用 `results/mot17_rule_promotion_fn_frames.csv` 依 `best_iou` 把 `FN` 分三桶：

1. `true miss`: `best_iou == 0`
2. `near miss`: `0 < best_iou < 0.5`
3. `threshold-sensitive`: `best_iou >= 0.5` 但仍為 `MISS`

整體 7-seq 結果：

| Bucket | Count | Share |
|---|---:|---:|
| `true miss` | 13,902 | 27.4% |
| `near miss` | 31,491 | 62.1% |
| `threshold-sensitive` | 5,347 | 10.5% |

這個結果支持「框偏移 / localization 問題是 `FN` 主因之一」：

- `62.1%` 的 `FN` 不是完全沒框，而是附近存在 prediction，但 IoU 不夠
- 這類 `FN` 更像 box center / size / partial-body coverage 不準，而非純 detector silence

但若只看 high-visibility (`vis >= 0.6`) 的 `FN`，情況更混合：

| Bucket | Count | Share |
|---|---:|---:|
| `true miss` | 4,235 | 44.3% |
| `near miss` | 4,690 | 49.0% |
| `threshold-sensitive` | 638 | 6.7% |

因此較準確的說法是：

- 整體 `FN` 以 `near miss` 為主，`box quality` 值得單獨優化
- 但 high-vis `FN` 仍有大塊 `true miss`，不能把所有 recall 問題都歸咎於框偏移

Per-sequence bucket shares：

| Sequence | true miss | near miss | threshold-sensitive |
|---|---:|---:|---:|
| MOT17-02-SDP | 5.1% | 80.2% | 14.6% |
| MOT17-04-SDP | 41.4% | 51.1% | 7.5% |
| MOT17-05-SDP | 13.9% | 79.5% | 6.7% |
| MOT17-09-SDP | 5.7% | 61.7% | 32.6% |
| MOT17-10-SDP | 14.9% | 72.5% | 12.6% |
| MOT17-11-SDP | 26.6% | 69.4% | 4.0% |
| MOT17-13-SDP | 39.6% | 46.9% | 13.6% |

讀法：

- `MOT17-02`、`MOT17-05`、`MOT17-10` 明顯偏 `near miss`，最像 localization-first 問題
- `MOT17-04`、`MOT17-13` 混入大量 `true miss`，不能只靠修框解
- `MOT17-09` 的 `threshold-sensitive` 比例高，較像 matching / assignment 也有貢獻

Height bucket 也支持這件事：

- 全部 `FN` 中，`128-256 px` 是 `near miss` 最大宗（14,435）
- high-vis `FN` 中，`128-256 px` 仍是 `near miss` 最大宗（1,892）

### Offset Summary

`scripts/eval/analyze_near_miss_offsets.py` 會輸出 best nearby prediction 的 box、score、track id，以及相對 GT 的 normalized offset。

Per-sequence `near_miss` median offsets:

| Sequence | center dx | center dy | width ratio | height ratio |
|---|---:|---:|---:|---:|
| MOT17-02-SDP | 0.121 | 0.579 | 2.137 | 1.970 |
| MOT17-04-SDP | 0.139 | 0.298 | 1.088 | 1.012 |
| MOT17-05-SDP | -0.031 | -0.010 | 1.825 | 1.635 |
| MOT17-09-SDP | -0.076 | 0.008 | 1.415 | 1.438 |
| MOT17-10-SDP | -0.113 | 0.026 | 1.154 | 1.101 |
| MOT17-11-SDP | 0.005 | 0.004 | 1.284 | 1.138 |
| MOT17-13-SDP | 0.023 | 0.046 | 1.143 | 1.091 |

Interpretation:

- `MOT17-02` is not a simple under-sized box case; the best nearby boxes are often larger and shifted downward.
- `MOT17-04` has moderate downward offset but near-normal width/height ratio, mixed with many true misses.
- Several sequences have median width/height ratio above 1.0, so blind expansion is unlikely to recover enough misses.

### Box Refinement Simulation

Offline simulation tested conservative transforms with max area growth `<= 1.20`.

Best simple candidates:

| Transform | recovered near miss | share | threshold-sensitive drops | median area growth |
|---|---:|---:|---:|---:|
| `width_expand_0.050` | 1,126 | 3.58% | 349 | 1.10 |
| `bottom_expand_0.150` | 1,063 | 3.38% | 1,211 | 1.15 |
| `uniform_expand_0.025` | 950 | 3.02% | 359 | 1.10 |
| `width_expand_0.100` | 931 | 2.96% | 512 | 1.20 |
| `vertical_expand_t0.050_b0.100` | 895 | 2.84% | 808 | 1.15 |

Result:

- Simple global expansion does not meet the `10%` near-miss recovery gate.
- Several transforms also push existing threshold-sensitive cases below IoU `0.5`.
- Runtime box expansion should not be promoted from this evidence alone.

### Stage Attribution Batch

After box-offset simulation, dense high-visibility near-miss windows were sampled from `MOT17-02-SDP` and `MOT17-10-SDP` and rerun with evaluator stage dumps:

```bash
uv run scripts/eval/mot17.py \
  --preset speed \
  --sequences MOT17-02-SDP \
  --debug-dump-seq MOT17-02-SDP \
  --debug-dump-frames 53-102 \
  --debug-dump-csv results/stage_dumps/MOT17-02-SDP_053-102.csv \
  --output results/tmp_stage_dump_02_053_102

uv run python scripts/eval/analyze_near_miss_stage_attribution.py \
  --near-miss-csv results/mot17_rule_promotion_near_miss_offsets.csv \
  --stage-dump-csv results/stage_dumps/MOT17-02-SDP_053-102.csv \
  --output-csv results/stage_dumps/MOT17-02-SDP_053-102_attribution.csv \
  --output-json results/stage_dumps/MOT17-02-SDP_053-102_attribution.json
```

Sampled windows:

| Window | total | raw good | post-merge good | lost after raw | tracker degraded | stage good final lost | raw never good |
|---|---:|---:|---:|---:|---:|---:|---:|
| `MOT17-02-SDP_053-102` | 804 | 409 | 215 | 194 | 293 | 137 | 177 |
| `MOT17-02-SDP_203-252` | 1,051 | 552 | 267 | 284 | 377 | 204 | 185 |
| `MOT17-10-SDP_253-302` | 518 | 293 | 134 | 154 | 157 | 72 | 119 |
| `MOT17-10-SDP_553-602` | 280 | 150 | 74 | 76 | 92 | 70 | 37 |
| **Aggregate** | **2,653** | **1,404** | **690** | **708** | **919** | **483** | **518** |

Aggregate stage-good counts:

| Stage | good boxes |
|---|---:|
| `raw` | 1,404 |
| `post_filter` | 690 |
| `post_nms` | 690 |
| `post_merge` | 690 |

Interpretation:

- Raw detector output often has an IoU `>= 0.5` candidate, but about half are lost by `post_filter`.
- `post_filter == post_nms == post_merge` in these windows, so NMS/merge is not the dominant loss point in this sample.
- `tracker_degraded` means `post_merge` had a better box than the final MOT output, not necessarily a tracker bug by itself. It points to final assignment/output competition/interpolation as the next place to inspect.
- `raw_never_good` remains material, so detector recall/localization still contributes and cannot be solved purely downstream.

### Final Output / Assignment Check

The `post_merge` candidates above were compared against the final MOT txt from the same stage-dump run using `scripts/eval/analyze_near_miss_final_output.py`:

```bash
uv run python scripts/eval/analyze_near_miss_final_output.py \
  --attribution-csv results/stage_dumps/MOT17-02-SDP_053-102_attribution.csv \
  --mot-result results/tmp_stage_dump_02_053_102/MOT17-02-SDP.txt \
  --output-csv results/stage_dumps/MOT17-02-SDP_053-102_final_output.csv \
  --output-json results/stage_dumps/MOT17-02-SDP_053-102_final_output.json
```

Only rows where `post_merge_iou >= 0.5` are shown below:

| Window | post-merge good | final GT match | final near-miss | candidate absent | similar box but GT miss | preserved metric miss | median final GT IoU | median final stage IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `MOT17-02-SDP_053-102` | 215 | 78 | 55 | 51 | 18 | 13 | 0.455 | 0.490 |
| `MOT17-02-SDP_203-252` | 267 | 63 | 94 | 91 | 14 | 5 | 0.240 | 0.175 |
| `MOT17-10-SDP_253-302` | 134 | 62 | 37 | 28 | 5 | 2 | 0.486 | 0.421 |
| `MOT17-10-SDP_553-602` | 74 | 4 | 36 | 29 | 5 | 0 | 0.140 | 0.126 |
| **Aggregate** | **690** | **207** | **222** | **199** | **42** | **20** | - | - |

Score split for `post_merge_iou >= 0.5`:

| Final output class | count | score < 0.28 | score < 0.15 | score >= 0.45 |
|---|---:|---:|---:|---:|
| `final_candidate_absent` | 199 | 86.4% | 68.3% | 3.0% |
| `final_near_miss` | 222 | 82.0% | 56.8% | 5.0% |
| `final_preserved_gt_match` | 207 | 13.5% | 2.4% | 75.8% |

Reading:

- A large share of good `post_merge` boxes are low-score boxes below `new_track_thresh = 0.28`, so they often cannot birth a new track.
- The CUDA tracker spawns new tracks only when `det_score >= effective_new_track_thresh`; final compaction writes only `confirmed` tracks updated in the current frame.
- This points to association/birth policy as the main output-side lever, especially conditional low-score birth for high-quality repeated detections.
- `final_near_miss` means the frame still emits a nearby box, but it is worse than the `post_merge` candidate; inspect tracker assignment/competition for these cases.
- `final_preserved_gt_match` rows are not misses under the same stage-dump run; earlier attribution mixed `results/MOT17_rule_promotion` final IoU with `--preset speed` stage dumps, so same-run final-output checks are required before drawing metric conclusions.

### Conditional Birth Sweep

Target: recover low-score `post_merge` good candidates on `MOT17-02-SDP` and `MOT17-10-SDP` without globally lowering `new_track_thresh`.

Shared command base:

```bash
uv run scripts/eval/mot17.py \
  --preset speed \
  --detector SDP \
  --sequences MOT17-02-SDP,MOT17-10-SDP \
  --module-lifecycle configs/modules/lifecycle.yaml \
  --output results/birth_sweep_speed_baseline_02_10
```

Variants tested:

- `baseline`
- `birth_quality_gate --birth-min-quality 0.45 --birth-quality-score-bias 0.20`
- `birth_consecutive_gate --birth-consecutive-frames 2 --birth-consecutive-iou 0.40 --birth-consecutive-boost 0.10 --birth-consecutive-min-score 0.15`
- `multi_birth_enabled`
- `multi_birth_enabled` with tighter geometry/evidence:
  `--multi-birth-min-score 0.15 --multi-birth-min-frames 3 --multi-birth-evidence-threshold 0.72 --multi-birth-min-aspect 1.8 --multi-birth-max-area-px 15000 --multi-birth-w-score 0.30 --multi-birth-w-motion 0.20 --multi-birth-w-quality 0.35 --multi-birth-w-streak 0.15`

Overall results:

| Variant | IDF1 | MOTA | IDs | FP | FN | Eval FPS |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 42.5 | 34.3 | 201 | 5,154 | 15,287 | 109.5 |
| `birth_quality_gate` | 40.6 | 32.0 | 251 | 6,509 | 14,595 | 97.2 |
| `birth_consecutive_gate` | 42.5 | 33.6 | 221 | 5,738 | 14,916 | 97.0 |
| `multi_birth` | 42.2 | 33.7 | 208 | 5,414 | 15,222 | 85.3 |
| `multi_birth_tight` | 42.5 | 34.0 | 204 | 5,277 | 15,257 | 89.0 |

Per-sequence deltas vs baseline:

| Variant | MOT17-02 MOTA | MOT17-02 FP | MOT17-02 FN | MOT17-10 MOTA | MOT17-10 FP | MOT17-10 FN |
|---|---:|---:|---:|---:|---:|---:|
| `birth_quality_gate` | 22.8 (-2.8) | 3,889 (+1,081) | 10,318 (-583) | 45.4 (-1.4) | 2,620 (+274) | 4,277 (-109) |
| `birth_consecutive_gate` | 24.4 (-1.2) | 3,326 (+518) | 10,611 (-290) | 46.9 (+0.1) | 2,412 (+66) | 4,305 (-81) |
| `multi_birth` | 24.6 (-1.0) | 3,035 (+227) | 10,866 (-35) | 46.8 (+0.0) | 2,379 (+33) | 4,356 (-30) |
| `multi_birth_tight` | 25.2 (-0.4) | 2,912 (+104) | 10,888 (-13) | 46.8 (+0.0) | 2,365 (+19) | 4,369 (-17) |

Reading:

- No tested birth policy beat baseline on this 2-seq slice.
- `birth_quality_gate` clearly over-promotes: it buys recall but blows up `FP`, `IDs`, and throughput.
- `birth_consecutive_gate` is materially safer than quality-only, but still loses on `MOTA` due to extra `FP`.
- Default `multi_birth` is the best direction among the three broad policies: smaller `FP` tax and almost neutral `MOT17-10`, but still not enough to justify promotion.
- Tighter `multi_birth` is the closest to baseline. It reduces the broad-policy regression substantially, but the net result is still `MOTA -0.3`, `FP +123`, `FN -30`, `IDs +3`, and `FPS -20.6`.

Decision:

- Do **not** lower `new_track_thresh` globally.
- Do **not** promote current birth gates as default.
- If this path continues, the next bar is not another blind sweep; it is promotion logging:
  capture which detections were boosted, their geometry/score history, and whether they became `TP` or `FP`.

## Follow-up

- Use the offset CSV to inspect per-sequence directionality before any runtime correction.
- 若要和 current default 直接比較，補一份 7-seq 同口徑的 detector-level `FP` rows
- 對 `MOT17-04-SDP` 單獨拆 `true miss` vs `near miss`，它目前是最大壓力來源
- If pursuing runtime refinement, prefer conditional policies over global expansion.
- For near-miss work, prioritize `post_filter` threshold/geometry decisions and final tracker output attribution before changing NMS/merge.
- Next experiment: replace the blind sweep with promotion-level diagnostics for `multi_birth_tight`, then split promoted rows into `TP / FP / ignored`.
