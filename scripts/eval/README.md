# scripts/eval

評估、調參、延遲分析的腳本目錄。

---

## 入口點

| 腳本 | 用途 |
|---|---|
| `mot17.py` | MOT17 主評估入口 |
| `detection_map.py` | detector-only mAP 評估（MOT-format GT） |
| `dancetrack.py` | DanceTrack 跨資料集評估 |
| `sportsmot.py` | SportsMOT 跨資料集評估 |
| `ablation_mot17.py` | 分組參數 ablation runner |
| `pipeline_contribution.py` | 模組累積貢獻分析 |
| `latency_report.py` | 延遲 profile 事後分析與比較 |
| `module_benchmark.sh` | 完整實驗流程封裝 |

---

## 工作流

### 1. 標準評估

```bash
uv run python scripts/eval/mot17.py \
  --engine models/yolo/yolo26s_960_batch1.engine \
  --detector SDP
```

載入設定的優先順序（低 → 高）：
`configs/mot17_baseline.yaml` → `--module-<name>` YAML → `--preset` → CLI flags

常用 profile：

```bash
# default speed baseline（native_960）
uv run python scripts/eval/mot17.py --preset speed --detector SDP

# lower-FP / higher-FPS profile（native_640）
uv run python scripts/eval/mot17.py --preset speed --detector SDP --tiling native_640
```

`native_640` 仍走 Saccade 原本的 tracking pipeline，只是 detector 輸入改為 640。
它通常會降低 `FP`、提高 `FPS`，但也會同步降低 `Recall / IDF1 / MOTA`，適合做對照與 profile，不是主預設。

### 2. 延遲分析

**快速 profile**（~3s，跳過 MOTMetrics）：

```bash
uv run python scripts/eval/mot17.py \
  --profile-stages --latency-only \
  --sequences MOT17-04-SDP \
  --max-frames 150 --warmup-frames 50 \
  --output runs/my_profile
```

輸出：console ASCII waterfall + `runs/my_profile/_stage_profile.json`

**事後分析**（不重跑）：

```bash
python scripts/eval/latency_report.py runs/my_profile/
python scripts/eval/latency_report.py runs/before/ --compare runs/after/
python scripts/eval/latency_report.py runs/my_profile/ --seq MOT17-04-SDP
```

**Stage 解讀**：

- `detect` — TRT 推論，通常最大瓶頸（~40-50%）
- `postprocess` — NMS + filter，第二大（~20-25%）
- `relink_write` — 含 background async 寫入，std 偏高屬正常
- `[unaccounted]` — frame_total 扣掉所有 stage 的殘差，< 1ms 正常
- `p95 >> mean` — 偶發長尾，值得調查

### 2.5 Detector mAP

```bash
uv run python scripts/eval/detection_map.py \
  --model models/yolo/yolo26s_960_batch1.engine \
  --sequences MOT17-04-SDP \
  --max-frames 100
```

用途：

- 只評估 detector，不跑 tracking / association / relink
- 對 MOT-format `gt/gt.txt` 計算 `mAP@0.5` 與 `mAP@0.5:0.95`
- 適合驗證 detector engine、conf threshold、preprocess 變更是否造成 detection regression
- 支援 `TRT .engine` 與 `Ultralytics .pt`，可用同一支腳本做對照

### 2.6 Birth Promotion Debug

```bash
uv run python scripts/eval/mot17.py \
  --preset speed \
  --sequences MOT17-02-SDP \
  --module-lifecycle configs/modules/lifecycle.yaml \
  --birth-quality-gate \
  --debug-birth-csv results/tmp_boost_debug.csv \
  --output results/tmp_boost_debug_run

uv run python scripts/eval/label_boosted_birth_rows.py \
  --boosted-csv results/tmp_boost_debug.csv \
  --results-dir results/tmp_boost_debug_run \
  --gt-root datasets/MOT17/train \
  --output-csv results/tmp_boost_debug_labeled.csv \
  --output-json results/tmp_boost_debug_labeled.json
```

用途：

- 記錄哪些 detection 被 `birth_quality_gate` / `birth_consecutive_gate` / `multi_birth` 提升分數
- 記錄該 boosted detection 是否真的進入 final MOT output
- 離線回填 final outcome：`tp` / `fp` / `ignore` / `dropped`

### 3. Ablation / 參數調整

```bash
# 分組 ablation
uv run python scripts/eval/ablation_mot17.py --category detection,geometry

# Bayesian 最佳化
uv run python scripts/eval/bayesian_optimizer.py

# 彙整 ablation 結果
uv run python scripts/eval/summarize_ablation_mot17.py results/ablation/
```

### 4. 模組貢獻分析

```bash
uv run python scripts/eval/pipeline_contribution.py --detector SDP
```

執行 `tracker_core → +gmc → +semantic → +bank → full_default` 累積切割，
每欄 Δprev 為單一模組的裸增益。輸出 `contribution_report.md/csv`。

### 5. 跨資料集驗證

MOT17 調參完成後，用作泛化 gate：

```bash
uv run python scripts/eval/sportsmot.py
uv run python scripts/eval/dancetrack.py
```

從 `configs/mot17_baseline.yaml` 出發，只在泛化確認後才引入資料集專用參數。

### 6. 完整實驗流程（module_benchmark.sh）

```bash
# 完整跑（profile → ablation → validate → contribution）
scripts/eval/module_benchmark.sh

# 單步
scripts/eval/module_benchmark.sh --mode profile --sequences MOT17-09-SDP --max-frames 80
scripts/eval/module_benchmark.sh --mode ablation --ablation-categories detection,geometry
scripts/eval/module_benchmark.sh --mode validate --engine models/yolo/yolo26s_960_batch1.engine
scripts/eval/module_benchmark.sh --mode contribution -- --match-thresh 0.78
```

Outputs: `results/module_benchmark/<timestamp>/`（含 `summary.txt`、`commands.txt`、`notes.md`）

---

## 輔助工具

| 腳本 | 用途 |
|---|---|
| `calculate_mota.py` | 對已有 .txt 結果重新計算追蹤指標 |
| `detection_map.py` | 對 MOT-format 序列直接重跑 detector 並計算 mAP |
| `convert_mot17.py` | MOT17 格式轉換 |
| `analyze_fn.py` | FN 根因診斷（逐框追蹤） |
| `analyze_near_miss_offsets.py` | FN near-miss 偏移診斷與 box refinement 離線模擬 |
| `analyze_near_miss_stage_attribution.py` | 將 FN near-miss 對齊 evaluator stage dump，定位 raw/filter/final output 流失點 |
| `analyze_near_miss_final_output.py` | 將 stage-attributed near miss 對齊同次 final MOT txt，拆 final output / birth / assignment 流失 |
| `label_boosted_birth_rows.py` | 將 boost 過的 birth detections 對齊 final MOT txt 與 GT，標成 tp/fp/ignore/dropped |
| `bench_yolo_batch.py` | detector engine batch 吞吐量 benchmark |
| `ablation_experiments.py` | ablation_mot17.py 使用的實驗定義集合 |
| `mot17_args.py` | 內部：mot17.py 的 argparse 委派 |

---

## 封存 / 實驗性

以下腳本為一次性實驗或外部 baseline 比較，不屬於常規工作流：

| 腳本 | 說明 |
|---|---|
| `mot17_public.py` | 用 MOT17 公開 detections 評估（det/det.txt） |
| `ultralytics_official_mot17.py` | Ultralytics 官方追蹤作為外部 baseline |
| `compare_framework_ultralytics.py` | 比較 Saccade vs Ultralytics 結果目錄 |
| `validate_last_vit_phase0.py` | LaSt-ViT Phase 0 原型驗證（已結案，No-Go） |
| `sweep_a7_quality.py` | A7 quality gate 參數掃描（已結案） |
| `coordinate_optimizer.py` | 座標最佳化實驗 |

---

## Script index

<!-- BEGIN generated script index -->
<!-- Generated by scripts/tools/build_scripts_index.py; do not edit this block by hand. -->

| Script | Status | Usage | Function |
|--------|--------|-------|----------|
| `_oao_feature_diff.py` | experiment | - | OAO gain attribution: characterise boxes that differ between baseline and B. |
| `_perseq_extract.py` | stable | cli | Per-sequence HOTA/AssA/IDF1 extractor for a finished output dir. |
| `_redirect.py` | stable | - | Shared helper that runpy-executes a canonical scripts/eval path for compat wrappers. |
| `ablation_experiments.py` | experiment | - | Historical multi-ablation experiment harness over MOT metrics tables. |
| `ablation_mot17.py` | stable | cli | Unified ablation runner for scripts/eval/mot17.py. |
| `analyze_05_cause.py` | experiment | cli | Attribution: is MOT17-05's occluder-mechanism regression from inaccurate boxes |
| `analyze_assoc_fn.py` | stable | - | Compatibility wrapper for scripts/eval/diagnostics/analyze_assoc_fn.py. |
| `analyze_confirm_prev_confirmed.py` | experiment | cli | Do cst040's new confirmations sit on a PREVIOUS-frame confirmed track? |
| `analyze_confirm_proximity.py` | experiment | cli | Does nearest-box distance separate the TP from the FP that cst040 newly confirms? |
| `analyze_crossing_swaps.py` | stable | - | Compatibility wrapper for scripts/eval/diagnostics/analyze_crossing_swaps.py. |
| `analyze_detection_fp_by_height.py` | diagnostic | cli | Detection-centric TP/FP split by box height x score band. |
| `analyze_external_fp_rows.py` | stable | - | Compatibility wrapper for scripts/eval/appearance/analyze_external_fp_rows.py. |
| `analyze_fn.py` | stable | - | Compatibility wrapper for scripts/eval/diagnostics/analyze_fn.py. |
| `analyze_fpn_dims.py` | experiment | cli | Analyze FPN feature dimension importance for ReID discrimination. |
| `analyze_fpn_embeddings.py` | experiment | cli | Analyze FPN embedding discriminability: intra-ID vs inter-ID cosine similarity. |
| `analyze_front_flag_exposure.py` | stable | - | Compatibility wrapper for scripts/eval/diagnostics/analyze_front_flag_exposure.py. |
| `analyze_near_miss_final_output.py` | stable | - | Compatibility wrapper for diagnostics/analyze_near_miss_final_output.py. |
| `analyze_near_miss_offsets.py` | stable | - | Compatibility wrapper for scripts/eval/diagnostics/analyze_near_miss_offsets.py. |
| `analyze_near_miss_stage_attribution.py` | stable | - | Compatibility wrapper for diagnostics/analyze_near_miss_stage_attribution.py. |
| `analyze_oao_attribution.py` | experiment | cli | OAO causal attribution: classify FP/FN changes between baseline (τ=0) and OAO (τ=0.3). |
| `analyze_oao_sweep.py` | experiment | cli | OAO (Occlusion-Aware Object) τ sweep analysis: metrics trend + per-sequence breakdown. |
| `analyze_occ_size.py` | diagnostic | cli | Attribution: cue-conflict — does box SIZE confirm the foot-y "front" call? |
| `analyze_occlusion_events.py` | diagnostic | cli | Frame-level occlusion event analysis: label every tracker→GT pair by occlusion state. |
| `analyze_pca_alt_combination.py` | experiment | - | Offline follow-up to probe_private_continuation_assignment.py: the hand-picked |
| `analyze_roi_dim_importance.py` | stable | - | Compatibility wrapper for scripts/eval/appearance/analyze_roi_dim_importance.py. |
| `analyze_score_distribution.py` | stable | - | Compatibility wrapper for scripts/eval/detector/analyze_score_distribution.py. |
| `bayesian_optimizer.py` | diagnostic | cli | Bayesian hyperparameter optimizer over MOT eval objectives. |
| `bench_recall_candidates.py` | diagnostic | cli | Head-only latency probes for recall-recovery architecture candidates. |
| `bench_reduction_bypass.py` | diagnostic | cli | Forward-latency: current head (spatial_reduction down/up) vs "no down/up". |
| `bench_yolo_batch.py` | diagnostic | cli | Benchmark YOLO TensorRT batch latency across batch sizes. |
| `bootstrap_mamba_size_recall.py` | diagnostic | cli | Paired moving-block bootstrap for size-binned detector recall. |
| `calculate_mota.py` | stable | cli | Compute MOTA/related metrics from tracker output vs GT. |
| `cheb_gr_osnet_gate.py` | stable | - | Compatibility wrapper for scripts/eval/appearance/cheb_gr_osnet_gate.py. |
| `compare_framework_ultralytics.py` | diagnostic | cli | Side-by-side MOTA comparison: Saccade vs Ultralytics (or any two result sets). |
| `concurrent_mot17.py` | stable | cli | Concurrent MOT17 Evaluator — batch-fused TRT (threading + batch4 engine) |
| `convert_mot17.py` | diagnostic | - | Convert/prepare MOT17 label layout for local eval trees. |
| `coordinate_optimizer.py` | experiment | - | Coordinate/param optimizer over mot17 CLI knobs (historical). |
| `dancetrack.py` | stable | cli | Run DanceTrack dataset eval through the saccade tracker pipeline. |
| `detection_map.py` | stable | cli | Detector-only mAP evaluation on MOT-format sequences. |
| `eval_conditioned.py` | stable | - | Compatibility wrapper for scripts/eval/baselines/eval_conditioned.py. |
| `eval_fpn_reid.py` | stable | - | Compatibility wrapper for scripts/eval/appearance/eval_fpn_reid.py. |
| `eval_gated_bytetrack.py` | stable | - | Compatibility wrapper for scripts/eval/baselines/eval_gated_bytetrack.py. |
| `eval_gated_detector.py` | archive-candidate | cli | Eval script for GatedYOLODetector (Method B). |
| `eval_mamba_head.py` | archive-candidate | cli | Eval script for MambaGatedDetector (Option F). |
| `eval_reid_1x1.py` | experiment | cli | Market-1501 ReID eval for 1×1 Conv dimension-reduction heads. |
| `eval_yolox_bytetrack.py` | experiment | cli | Eval: YOLOX-X detector + our GPUByteTracker. |
| `export_external_fp_rows.py` | stable | - | Compatibility wrapper for scripts/eval/appearance/export_external_fp_rows.py. |
| `fpn_raw_reid_market1501.py` | experiment | cli | Market-1501 ReID evaluation — raw YOLO FPN features (zero-training baseline). |
| `jde_market1501.py` | stable | - | Compatibility wrapper for scripts/eval/appearance/jde_market1501.py. |
| `label_boosted_birth_rows.py` | stable | - | Compatibility wrapper for scripts/eval/diagnostics/label_boosted_birth_rows.py. |
| `latency_report.py` | stable | cli | Standalone latency analyzer for _stage_profile.json files. |
| `mamba_size_binned_recall.py` | stable | - | Compatibility wrapper for scripts/eval/detector/mamba_size_binned_recall.py. |
| `measure_cst040_dedup.py` | experiment | cli | Measure cst040 + duplicate-suppression end-to-end. |
| `mlflow_logger.py` | diagnostic | - | Helper to log eval/train metrics and git identity to MLflow. |
| `module_benchmark.sh` | stable | - | Shell driver for per-module latency/accuracy benchmarks. |
| `mot17.py` | stable | cli | Flagship MOT17 eval entry: detector+tracker under presets, report metrics. |
| `mot17_all_sdp.py` | stable | cli | Batch-run MOT17 SDP sequences through the standard eval path. |
| `mot17_args.py` | stable | cli | Shared argparse argument groups used by mot17 eval entrypoints. |
| `mot17_public.py` | stable | - | Compatibility wrapper for scripts/eval/baselines/mot17_public.py. |
| `occ_rank.py` | diagnostic | cli | Post-hoc ranker over the tag-indexed occ-tune store — zero eval. |
| `occ_tune.py` | diagnostic | cli | Tag-indexed, incremental tuner for the same-height occlusion gate. |
| `oracle_height_birth_ceiling.py` | stable | - | Compatibility wrapper for experiments/oracle_height_birth_ceiling.py. |
| `oracle_occlusion_hold.py` | stable | - | Compatibility wrapper for scripts/eval/experiments/oracle_occlusion_hold.py. |
| `oracle_small_birth_ceiling.py` | stable | - | Compatibility wrapper for experiments/oracle_small_birth_ceiling.py. |
| `pipeline_contribution.py` | diagnostic | cli | Run cumulative pipeline cutoff experiments for MOT17 and summarize the |
| `print_assoc_basis.py` | diagnostic | cli | Print the resolved *association basis* (height / IoU / velocity / cost-weight |
| `probe_assoc_appearance_veto.py` | diagnostic | cli | mnv4 appearance-veto separability at PRIMARY association decision points. |
| `probe_camera_motion.py` | experiment | - | Capstone: per-sequence camera-motion magnitude. |
| `probe_ghost_rate_by_score.py` | experiment | - | Ghost-rate-by-score probe. |
| `probe_ghost_source.py` | experiment | - | Decompose low-score 'ghost' boxes by source. |
| `probe_lowiou_occ_gate.py` | diagnostic | cli | Can an OCCLUSION signal SAFELY relax the low-IoU association gate? |
| `probe_occ_activation_separability.py` | diagnostic | cli | Feasibility probe: can the Mamba HEAD ACTIVATION separate occluded from visible GT? |
| `probe_occ_pairwise_confound.py` | diagnostic | cli | Follow-up to registry #46 (probe_occ_activation_separability.py): does the |
| `probe_occ_separability.py` | diagnostic | cli | Feasibility probe: does the Mamba head's SCORE separate occluded from visible GT? |
| `probe_occ_swap_disambiguation.py` | diagnostic | cli | At occluder-ABSORB crossing-swaps: does an OCCLUSION signal disambiguate the |
| `probe_private_continuation_assignment.py` | experiment | cli | Does the geometric pairwise-position signal (gap_h/dx_norm, from the #46 |
| `probe_redundancy.py` | experiment | - | Redundancy probe for low-score REAL_BADBOX boxes. |
| `profile_analyze.py` | diagnostic | cli | Analyse per-frame CSV ledger from --profile-frame-csv. |
| `profile_cuda_kernels.py` | diagnostic | - | Profile CUDA kernel time for detector/tracker hot paths. |
| `reconnect_rate.py` | stable | - | Compatibility wrapper for scripts/eval/diagnostics/reconnect_rate.py. |
| `reid_id_benchmark.py` | stable | - | Compatibility wrapper for scripts/eval/appearance/reid_id_benchmark.py. |
| `run_confirm_gate_sweep.sh` | experiment | cli | Confirmation-gate sweep: shorten establishment latency (the 57% "establish/ |
| `run_m_matched.sh` | experiment | cli | Matched-M head retrain/eval driver at saturated fair operating point. |
| `run_native_assoc_search.sh` | experiment | cli | Native association hyperparam search driver (matched baseline suite). |
| `run_native_full_sweep.sh` | experiment | cli | Native full-config sweep driver (matched baseline suite). |
| `run_native_m_param.sh` | experiment | cli | Native M-parameter sweep driver. |
| `run_native_m_sweep.sh` | experiment | cli | Native M-architecture sweep driver. |
| `run_native_param_search.sh` | experiment | cli | Native multi-parameter search driver. |
| `run_native_pr_sweep.sh` | experiment | cli | Native precision/recall operating-point sweep. |
| `run_native_s_speed.sh` | experiment | cli | Native S-speed variant sweep. |
| `run_nopost_eval.sh` | experiment | - | Implicit vs explicit (matched seeds 42/43/44) re-evaluated with the post-processing |
| `run_occ_audit_offline.py` | diagnostic | cli | Offline occ-exit-audit ablation on an existing substrate's output files. |
| `run_offline_handover_ablation.py` | diagnostic | cli | Offline Cheb-GR handover ablation on an existing substrate's output files. |
| `run_teacher_head_matched_baseline.sh` | experiment | cli | Matched-baseline: original YOLO detect head (teacher) vs Mamba head through the |
| `run_threshold_strategies.sh` | stable | - | Thin redirect to experiments/run_threshold_strategies.sh. |
| `run_v14_conversion_ablation.sh` | archive-candidate | cli | Zero-training v14 conversion ablation. |
| `score_keyframe_filtered.py` | stable | cli | Score a tracker result dir against sparse-keyframe GT (PersonPath22 eval mode). |
| `select_ckpt_by_recall.py` | stable | cli | Post-training checkpoint selection by MOT17 tracking recall. |
| `sportsmot.py` | stable | cli | Run SportsMOT dataset eval through the saccade tracker pipeline. |
| `summarize_ablation_mot17.py` | stable | cli | Summarize MOT17 ablation experiment folders against a shared baseline. |
| `sweep_a7_quality.py` | experiment | - | Sweep A7/P5-2 quality-gate thresholds against MOT metrics. |
| `sweep_density_gating.py` | diagnostic | cli | Sweep: 局部軌跡密度自適應門控 (Density-Gating) 超參數搜索 |
| `sweep_external_fp_classifier.py` | experiment | cli | Sweep logistic TP/FP filters on CrowdHuman external rows. |
| `sweep_low_mt.sh` | experiment | - | Sweep low match_thresh values and collect overall metrics. |
| `train_cascade_stage2.py` | stable | - | Compatibility wrapper for scripts/eval/appearance/train_cascade_stage2.py. |
| `train_external_fp_classifier.py` | stable | - | Compatibility wrapper for appearance/train_external_fp_classifier.py. |
| `ultralytics_official_mot17.py` | stable | - | Compatibility wrapper for scripts/eval/baselines/ultralytics_official_mot17.py. |
| `validate_profiles.py` | diagnostic | cli | Validate that pipeline_contribution.py profiles correctly map to pipeline stages |
| `validate_roi_embeddings.py` | stable | - | Compatibility wrapper for scripts/eval/appearance/validate_roi_embeddings.py. |
| `verify_cpp_detector.py` | diagnostic | cli | Validation and benchmark script for C++ LibTorch + TensorRT MambaGatedDetector. |

<!-- END generated script index -->
