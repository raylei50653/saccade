# Diagnostic ↔ Experiment Semantic Review RESULT

**Scope:** 138 scripts from first-pass `diagnostic`/`experiment` seeds (stable set not re-reviewed).
**Mode:** read-only content review; no script edits / no commits.
**Date:** 2026-07-21

## 1. Per-script table (by subdirectory)

### `scripts/benchmarks` (13)

| path | current | proposed | changed? | reason |
|------|---------|----------|----------|--------|
| `scripts/benchmarks/benchmark_16_streams.py` | diagnostic | **diagnostic** | N | Multi-stream dispatcher stress microbench; reusable perf tool. |
| `scripts/benchmarks/benchmark_association.py` | diagnostic | **diagnostic** | N | GPU association/auction latency/throughput bench. |
| `scripts/benchmarks/bottleneck_annealer.py` | diagnostic | **diagnostic** | N | E2E pipeline bottleneck annealer; reusable profiler. |
| `scripts/benchmarks/dataloader_bench.py` | diagnostic | **diagnostic** | N | DataLoader throughput profiler (preload_to_ram vs workers). |
| `scripts/benchmarks/latency_e2e_report.py` | diagnostic | **diagnostic** | N | E2E latency report via run_eval; reusable perf diagnostic. |
| `scripts/benchmarks/mamba_crossscan_breakdown.py` | diagnostic | **diagnostic** | N | Cross-scan flip/stack vs SSM cost decomposition; reusable profiler (not a study driver). |
| `scripts/benchmarks/mamba_detect_breakdown.py` | diagnostic | **diagnostic** | N | Detect-stage subcomponent attribution; reusable. |
| `scripts/benchmarks/mamba_flip_alloc.py` | diagnostic | **diagnostic** | N | Tensor-alloc hypothesis microbench for cross-scan flips; reusable. |
| `scripts/benchmarks/mamba_head_breakdown.py` | diagnostic | **diagnostic** | N | Head submodule CUDA-event breakdown; reusable. |
| `scripts/benchmarks/mamba_head_kernelcount.py` | diagnostic | **diagnostic** | N | Kernel-launch counter for one head forward; reusable. |
| `scripts/benchmarks/mamba_train_prof.py` | diagnostic | **diagnostic** | N | Mamba train-step stage profiler with CLI phases; reusable. |
| `scripts/benchmarks/train_bottleneck_prof.py` | diagnostic | **diagnostic** | N | Training-step bottleneck via conditioned proxy; analysis tool. |
| `scripts/benchmarks/workbench_synthetic.py` | diagnostic | **diagnostic** | N | Synthetic workbench load generator for microbenches. |

### `scripts/ (root)` (6)

| path | current | proposed | changed? | reason |
|------|---------|----------|----------|--------|
| `scripts/download_crowdhuman_hf.py` | diagnostic | **diagnostic** | N | Reusable CrowdHuman HF downloader. |
| `scripts/download_external_datasets.py` | diagnostic | **diagnostic** | N | Reusable CrowdHuman/CityPersons downloader. |
| `scripts/download_kitti_tracking.py` | diagnostic | **diagnostic** | N | Reusable KITTI tracking downloader. |
| `scripts/download_market1501.py` | diagnostic | **diagnostic** | N | Reusable Market-1501 HF downloader. |
| `scripts/download_motsynth.py` | diagnostic | **diagnostic** | N | Reusable MOTSynth downloader. |
| `scripts/train_option_d.sh` | experiment | **archive-candidate** | Y | Option D 2-phase driver over train_conditioned.py; no_go #1 (harmful); docs/archive/option-d only. |

### `scripts/eval` (41)

| path | current | proposed | changed? | reason |
|------|---------|----------|----------|--------|
| `scripts/eval/_oao_feature_diff.py` | experiment | **experiment** | N | OAO gain attribution (baseline vs B); bound to no_go #7 revival analysis. |
| `scripts/eval/ablation_experiments.py` | experiment | **experiment** | N | Historical multi-ablation harness (a2/a3/a6/oao_tau…); named experiment catalog. |
| `scripts/eval/analyze_05_cause.py` | experiment | **experiment** | N | MOT17-05 occ-mechanism regression attribution; study-bound substrate analysis. |
| `scripts/eval/analyze_confirm_prev_confirmed.py` | experiment | **experiment** | N | cst040 confirm study: prev-frame overlap of new confirms; defaults results/conf_cst040. |
| `scripts/eval/analyze_confirm_proximity.py` | experiment | **experiment** | N | cst040 confirm study: NN distance of newly-confirmed TP vs FP. |
| `scripts/eval/analyze_fpn_dims.py` | experiment | **diagnostic** | Y | Path-arg Market-1501 FPN dim-importance tool; re-runnable on any mamba ckpt (supports no_go #16 evidence, not a closed-run driver). |
| `scripts/eval/analyze_fpn_embeddings.py` | experiment | **diagnostic** | Y | Path-arg FPN intra/inter-ID cosine probe on any seq/ckpt/head; reusable embedding QA. |
| `scripts/eval/analyze_oao_attribution.py` | experiment | **experiment** | N | Hardcoded oao_tau03 ablation dirs + FP/FN cause taxonomy; no_go #7. |
| `scripts/eval/analyze_oao_sweep.py` | experiment | **experiment** | N | Hardcoded TAU_EXPERIMENTS under ablation_mot17/association/oao_tau*; #7 sweep report. |
| `scripts/eval/analyze_pca_alt_combination.py` | experiment | **experiment** | N | Follow-up to private-continuation assignment; hardcoded pca_full7.npz; #46 family. |
| `scripts/eval/compare_framework_ultralytics.py` | diagnostic | **diagnostic** | N | Generic two-result-set MOT metrics compare; path args only. |
| `scripts/eval/convert_mot17.py` | diagnostic | **diagnostic** | N | Reusable MOT17 GT→YOLO label layout helper. |
| `scripts/eval/coordinate_optimizer.py` | experiment | **experiment** | N | Historical coordinate/param optimizer over mot17 knobs; research tuner. |
| `scripts/eval/eval_reid_1x1.py` | experiment | **experiment** | N | Market-1501 eval of DimReduceHead v9; measurement protocol for FPN-ReID #16 family. |
| `scripts/eval/eval_yolox_bytetrack.py` | experiment | **experiment** | N | YOLOX-X + GPUByteTracker isolation study; depends on /tmp/YOLOX. |
| `scripts/eval/fpn_raw_reid_market1501.py` | experiment | **experiment** | N | Zero-training raw FPN ReID baseline on Market-1501; core #16 protocol. |
| `scripts/eval/measure_cst040_dedup.py` | experiment | **experiment** | N | cst040 + lifespan-dedup intervention measure; named confirm-gate study. |
| `scripts/eval/mlflow_logger.py` | diagnostic | **diagnostic** | N | Reusable MLflow logging helper for any eval/train run. |
| `scripts/eval/probe_camera_motion.py` | experiment | **experiment** | N | Camera-motion vs hardcoded fw ΔIDF1 table; no_go #45 attribution capstone. |
| `scripts/eval/probe_ghost_rate_by_score.py` | experiment | **experiment** | N | Ghost-rate-by-score vs fw ΔIDF1; #45 fuse_score_weight probe. |
| `scripts/eval/probe_ghost_source.py` | experiment | **experiment** | N | Ghost-source decomposition with same fw Δ table; #45 follow-up. |
| `scripts/eval/probe_private_continuation_assignment.py` | experiment | **experiment** | N | Private-continuation A/B + gap/dx tie-break; #46 follow-up application. |
| `scripts/eval/probe_redundancy.py` | experiment | **experiment** | N | REAL_BADBOX redundant vs unique vs fw GO/NOGO; #45 follow-up. |
| `scripts/eval/profile_analyze.py` | diagnostic | **diagnostic** | N | Generic frame-ledger CSV profiler; any --profile-frame-csv run. |
| `scripts/eval/profile_cuda_kernels.py` | diagnostic | **diagnostic** | N | Torch profiler wrapper over run_eval; reusable perf diagnostic. |
| `scripts/eval/run_confirm_gate_sweep.sh` | experiment | **experiment** | N | Named confirm-gate arms (cst040/cst035/streak2/combo); confirm-latency study. |
| `scripts/eval/run_m_matched.sh` | experiment | **experiment** | N | Matched-M mamba vs native at fair OP; hardcoded v14replica ckpts. |
| `scripts/eval/run_native_assoc_search.sh` | experiment | **experiment** | N | Native e26 association-only search; fixed gated_det_native_full. |
| `scripts/eval/run_native_full_sweep.sh` | experiment | **experiment** | N | Native full-training epoch×confirm sweep; fixed study suite. |
| `scripts/eval/run_native_m_param.sh` | experiment | **experiment** | N | Native-M e26 param search (bridge/PC); fixed M ckpt. |
| `scripts/eval/run_native_m_sweep.sh` | experiment | **experiment** | N | Native-M architecture epoch sweep; fixed study suite. |
| `scripts/eval/run_native_param_search.sh` | experiment | **experiment** | N | Native e26 multi-param + private_continuation grid. |
| `scripts/eval/run_native_pr_sweep.sh` | experiment | **experiment** | N | Native e26 precision/recall OP sweep. |
| `scripts/eval/run_native_s_speed.sh` | experiment | **experiment** | N | Native-S vs mamba-S speed (eager/compile); fixed paired ckpts. |
| `scripts/eval/run_nopost_eval.sh` | experiment | **experiment** | N | v14replica implicit/explicit seeds with interp+bridge OFF; named study. |
| `scripts/eval/run_teacher_head_matched_baseline.sh` | experiment | **experiment** | N | Teacher YOLO head vs Mamba head matched baseline on v14replica teacher. |
| `scripts/eval/sweep_a7_quality.py` | experiment | **experiment** | N | A7/P5-2 quality-gate C++ patch+rebuild sweep; no_go #11; mutates tracker_gpu.cu. |
| `scripts/eval/sweep_external_fp_classifier.py` | experiment | **diagnostic** | Y | Fully parameterized logistic TP/FP OP sweep (--rows-csv, --model-json); reusable beyond cascade-filter #25. |
| `scripts/eval/sweep_low_mt.sh` | experiment | **experiment** | N | Low match_thresh×GMC grid on hardcoded fpn_reid_baseline + mamba_gt_960_v2. |
| `scripts/eval/validate_profiles.py` | diagnostic | **diagnostic** | N | Pipeline profile structure/docs/flags validation; maintenance diagnostic. |
| `scripts/eval/verify_cpp_detector.py` | diagnostic | **diagnostic** | N | C++ LibTorch+TRT detector validation/bench; reusable parity check. |

### `scripts/eval/appearance` (1)

| path | current | proposed | changed? | reason |
|------|---------|----------|----------|--------|
| `scripts/eval/appearance/mnv4_spatial_dilution_probe.py` | experiment | **experiment** | N | mnv4/LaSt-ViT spatial-dilution hypothesis; no_go #15 appearance family. |

### `scripts/eval/experiments` (4)

| path | current | proposed | changed? | reason |
|------|---------|----------|----------|--------|
| `scripts/eval/experiments/oracle_height_birth_ceiling.py` | experiment | **experiment** | N | Oracle ceiling for height×score birth filter; no_go #38. |
| `scripts/eval/experiments/oracle_occlusion_hold.py` | experiment | **experiment** | N | Phase-0 oracle for Occluded-hold / crossing-swap policy. |
| `scripts/eval/experiments/oracle_small_birth_ceiling.py` | experiment | **experiment** | N | Recall-side oracle for lowering birth thresh on small boxes; #38 companion. |
| `scripts/eval/experiments/run_v14_conversion_ablation.sh` | experiment | **experiment** | N | Zero-training v14 runtime conversion A/B; hardcoded parent/v14 ckpts. |

### `scripts/release` (1)

| path | current | proposed | changed? | reason |
|------|---------|----------|----------|--------|
| `scripts/release/build_slim_release.py` | diagnostic | **diagnostic** | N | Slim-release materializer from manifest; ops convertor, not a study. |

### `scripts/tools` (36)

| path | current | proposed | changed? | reason |
|------|---------|----------|----------|--------|
| `scripts/tools/analyze_fn_strata.py` | experiment | **experiment** | N | FN strata by height×visibility for 'P2 recovers resolution-limited FN?'; hardcoded MOT17_eval; study hypothesis. |
| `scripts/tools/average_top_trials.py` | experiment | **diagnostic** | Y | Generic Optuna CLI (--study/--storage/--min-value); no fixed study identity. |
| `scripts/tools/bench_pipeline_halves.py` | experiment | **experiment** | N | Odd/even double-buffer GPU-time ceiling spike; one-off research measurement. |
| `scripts/tools/birth_death_consistency.py` | experiment | **experiment** | N | Birth↔death appearance consistency for relink signal ceiling; #32/#2 family. |
| `scripts/tools/check_scan_bwd.py` | experiment | **diagnostic** | Y | CUDA selective-scan bwd vs JIT autograd numerical smoke; kernel correctness, not a metric study. |
| `scripts/tools/classify_gap_cause.py` | experiment | **diagnostic** | Y | Generic gap-cause classifier on any --csv + --mot-dir + GT; reusable relink-pool analysis tool. |
| `scripts/tools/cold_start_transfer.py` | experiment | **experiment** | N | MOT17→MOT20 cold-start of depth-order landmarks; no_go #39 transfer test. |
| `scripts/tools/compare_trials.py` | experiment | **diagnostic** | Y | Generic Optuna trial table CLI; reusable HPO helper. |
| `scripts/tools/convert_mot17_to_mp4.py` | diagnostic | **diagnostic** | N | Parallel ffmpeg MOT seq→MP4 viz helper. |
| `scripts/tools/convert_video_to_mot.py` | diagnostic | **diagnostic** | N | MP4→MOT folder helper; reusable data prep. |
| `scripts/tools/depth_ordering_auc.py` | experiment | **experiment** | N | Front/back AUC for depth cue; #39 companion. |
| `scripts/tools/depth_ordering_gate_sweep.py` | experiment | **experiment** | N | Offline (iou_thresh, foot_gap) sweep for same-height occ gate; #39. |
| `scripts/tools/depth_ordering_probe.py` | experiment | **experiment** | N | Core #39 GT probe: pre-occ geometry rank front/back at crossings? |
| `scripts/tools/determinism_check.py` | diagnostic | **diagnostic** | N | Multi-run bit-identical eval gate; regression diagnostic. |
| `scripts/tools/diagnose_id_switches.py` | experiment | **experiment** | N | IDSW gap stratification framed for P3-B dormant bank value; hypothesis-shaped buckets. |
| `scripts/tools/eval_golden.py` | diagnostic | **diagnostic** | N | Bit-exact golden capture/check for run_eval refactors. |
| `scripts/tools/format_tables.py` | diagnostic | **archive-candidate** | Y | One-shot string-replace that mutates src/saccade/perception/eval/runner.py print formats; migration leftover, not a re-runnable formatter. |
| `scripts/tools/gmc_rotation_probe.py` | experiment | **experiment** | N | GMC |rotation| vs horizon-justified bound; #41 / GMC plausibility probe. |
| `scripts/tools/gpu_check.py` | diagnostic | **diagnostic** | N | Env smoke: CUDA/TRT/GStreamer/YOLO engine. |
| `scripts/tools/graphify_cuda_patch.sh` | diagnostic | **diagnostic** | N | Idempotent re-patch of graphify for .cu/.cuh; maintenance. |
| `scripts/tools/horizon_convergence_probe.py` | experiment | **experiment** | N | Horizon design Exp1: scale-field→horizon convergence; no_go #41. |
| `scripts/tools/horizon_detector_test.py` | experiment | **experiment** | N | Homothety horizon on GT vs real dets; #41 deployment-survival test. |
| `scripts/tools/horizon_homothety_probe.py` | experiment | **experiment** | N | Vertex-homothety vanishing-point construction; #41 core method probe. |
| `scripts/tools/horizon_window_probe.py` | experiment | **experiment** | N | Sliding-window horizon motion for GMC signal; #41. |
| `scripts/tools/intra_track_consistency.py` | experiment | **experiment** | N | Within-track appearance stability upstream of relink gate; #32 family. |
| `scripts/tools/motion_norm_probe.py` | experiment | **experiment** | N | §8.4 displacement norm raw vs /h vs horizon depth_proxy; #41 motion-norm AUC. |
| `scripts/tools/probe_gate_separability.py` | experiment | **experiment** | N | Gated-YOLO per-scale separability for heatmap-cache fast path; Option-D-adjacent architecture probe. |
| `scripts/tools/remap_aligned.py` | diagnostic | **archive-candidate** | Y | Hardcoded local→global ID maps for results/demo/custom_seq_*; one-shot demo glue, not a general remap CLI. |
| `scripts/tools/sweep_href_variants.py` | experiment | **experiment** | N | Offline h_ref normalization variants for bridge score AUC/AP/LOSO; closed relink-bridge sweep. |
| `scripts/tools/sweep_velocity_variants.py` | experiment | **experiment** | N | Offline lost/cand exit-velocity estimators for bridge score; historical velocity/ring-cap sweep. |
| `scripts/tools/test_cufft_graph.py` | diagnostic | **diagnostic** | N | Minimal cuFFT graph capture smoke. |
| `scripts/tools/test_gmc_cudagraph.py` | diagnostic | **diagnostic** | N | Capture/replay of C++ estimate_into_direct; GMC impl diagnostic. |
| `scripts/tools/test_gpu_gmc.py` | diagnostic | **diagnostic** | N | Manual GPU GMC correctness/perf smoke. |
| `scripts/tools/test_letterbox_gpu.py` | diagnostic | **diagnostic** | N | cpp_letterbox_gpu vs PyTorch reference. |
| `scripts/tools/test_py_gmc.py` | diagnostic | **diagnostic** | N | Pure-PyTorch FFT GMC graph-capturable path smoke. |
| `scripts/tools/verify_gmc_direct.py` | diagnostic | **diagnostic** | N | estimate_into vs estimate_into_direct warp parity. |

### `scripts/train` (2)

| path | current | proposed | changed? | reason |
|------|---------|----------|----------|--------|
| `scripts/train/train_reid_head.py` | experiment | **archive-candidate** | Y | FPN emb cache → SupCon ReID head; ties to weak appearance/FPN ceiling (no_go #2/#16); not main-line MOT. |
| `scripts/train/train_temporal_yolo.py` | experiment | **archive-candidate** | Y | Early TemporalYOLOHybrid / Option C frozen-backbone trainer; superseded by train_joint then Mamba stack. |

### `scripts/train/temporal_yolo` (34)

| path | current | proposed | changed? | reason |
|------|---------|----------|----------|--------|
| `scripts/train/temporal_yolo/precompute_gmc.py` | diagnostic | **diagnostic** | N | Reusable GMC affine cache builder for MOT training. |
| `scripts/train/temporal_yolo/run_aug_loo.sh` | experiment | **experiment** | N | Named v14replica LOO augment ablation; hardcoded teacher/cache. |
| `scripts/train/temporal_yolo/run_frozenT1_probe.sh` | experiment | **experiment** | N | v14replica §4.1 frozen-SSM T=1 attribution probe. |
| `scripts/train/temporal_yolo/run_pp22_augment_ablation.sh` | experiment | **experiment** | N | PP22 single-factor augment ablation; no_go #54 family. |
| `scripts/train/temporal_yolo/run_pp22_augment_e30.sh` | experiment | **experiment** | N | PP22 augment-teacher downstream chain. |
| `scripts/train/temporal_yolo/run_pp22_dfl16_chain.sh` | experiment | **experiment** | N | PP22 DFL reg_max=16 chain; architecture knob #53. |
| `scripts/train/temporal_yolo/run_pp22_full_cadence_chain.sh` | experiment | **experiment** | N | PP22 full-cadence/interp plan driver; no_go #52/#54 family. |
| `scripts/train/temporal_yolo/run_pp22_heldout_e30.sh` | experiment | **experiment** | N | PP22 held-out e30 teacher→mamba chain. |
| `scripts/train/temporal_yolo/run_pp22_sr2_chain.sh` | experiment | **experiment** | N | PP22 spatial_reduction=2 chain; #53. |
| `scripts/train/temporal_yolo/run_pp22_teacher_strong.sh` | experiment | **experiment** | N | Strengthened PP22 teacher recipe (header: NOT yet run); named plan. |
| `scripts/train/temporal_yolo/run_reduction_candidates.sh` | experiment | **experiment** | N | Named reduction-variant arms vs mamba_gt_vgt_mamba_v14. |
| `scripts/train/temporal_yolo/run_t1joint_probe.sh` | experiment | **experiment** | N | T3/T1 curriculum Route 2 joint-loss probe. |
| `scripts/train/temporal_yolo/run_v14_frozen_bn_teacher.sh` | experiment | **experiment** | N | v14 full-e30 BN-adapt teacher control arm. |
| `scripts/train/temporal_yolo/run_v14_frozen_yolo_teacher.sh` | experiment | **experiment** | N | v14 full-e30 frozen-YOLO teacher control. |
| `scripts/train/temporal_yolo/run_v14_full_e30_replication.sh` | experiment | **experiment** | N | Named v14 e30 lineage rebuild (protocol entry, still study-bound experiment). |
| `scripts/train/temporal_yolo/run_v14_parent_n16_frozen_refit.sh` | experiment | **experiment** | N | v14 N=16 frozen-SSM refit causal probe. |
| `scripts/train/temporal_yolo/run_v14replica_consistency.sh` | experiment | **experiment** | N | Explicit consistency-loss follow-up (T3/T1 route 3; related #37). |
| `scripts/train/temporal_yolo/run_v14replica_consistency_sweep.sh` | experiment | **experiment** | N | Consistency-weight sweep wrapper on v14replica. |
| `scripts/train/temporal_yolo/run_v14replica_d256_ablation.sh` | experiment | **experiment** | N | d_model 128→256 capacity ablation on v14replica. |
| `scripts/train/temporal_yolo/run_v14replica_detail_b1h.sh` | experiment | **experiment** | N | B1-H detail-route experiment on replica lineage. |
| `scripts/train/temporal_yolo/run_v14replica_implicit_seed.sh` | experiment | **experiment** | N | Implicit-consistency seed control vs T3→T1. |
| `scripts/train/temporal_yolo/run_v14replica_implicit_sweep.sh` | experiment | **experiment** | N | Multi-seed sweep over implicit baseline. |
| `scripts/train/temporal_yolo/run_v14replica_seed.sh` | experiment | **experiment** | N | v14replica student-chain reseed (distill→GT1→GT2). |
| `scripts/train/temporal_yolo/run_v14replica_strip_oracle.sh` | experiment | **experiment** | N | Strip-oracle detail routing probe (related #36). |
| `scripts/train/temporal_yolo/run_v14replica_t3t1.sh` | experiment | **experiment** | N | Main T3→T1 curriculum runner; hardcoded replica paths. |
| `scripts/train/temporal_yolo/run_v14replica_t3t1_seed.sh` | experiment | **experiment** | N | Multi-seed T3→T1 paired validation. |
| `scripts/train/temporal_yolo/run_v14replica_t3t1_shared_seed.sh` | experiment | **experiment** | N | Shared-GT1 T3→T1 seed variant. |
| `scripts/train/temporal_yolo/run_v14replica_yolo26m.sh` | experiment | **experiment** | N | YOLO26m capacity contrast rebuild of replica lineage. |
| `scripts/train/temporal_yolo/train_conditioned.py` | experiment | **archive-candidate** | Y | Option D conditioned-gate trainer; temporal_yolo/README marks archive-candidate; no_go #1. |
| `scripts/train/temporal_yolo/train_gated_tp.py` | experiment | **archive-candidate** | Y | Older gated-TP recall trainer; README archive-candidate vs train_gated_detector/Mamba main-line. |
| `scripts/train/temporal_yolo/train_jde_distill.py` | experiment | **archive-candidate** | Y | OSNet→JDE KD on Market; ReID side-line; README archive-candidate for main-line MOT. |
| `scripts/train/temporal_yolo/train_jde_market.py` | experiment | **archive-candidate** | Y | Market-1501 JDE emb projector trainer; ReID experiment side-line; README archive-candidate. |
| `scripts/train/temporal_yolo/train_joint.py` | experiment | **archive-candidate** | Y | Option C joint Cross-Attn trainer; superseded by Mamba-head stable trainers; README archive-candidate. |
| `scripts/train/temporal_yolo/train_reid_1x1.py` | experiment | **archive-candidate** | Y | 1×1 dim-reduce ReID head on Market; ReID side-line; README archive-candidate. |

## 2. Summary

### proposed_label counts

| proposed_label | count |
|----------------|------:|
| diagnostic | 46 |
| experiment | 81 |
| archive-candidate | 11 |
| **total** | **138** |

**Changed:** 18 · **Unchanged:** 120

### Change lists (apply-ready)

#### diagnostic → experiment

_None._

#### experiment → diagnostic

- `scripts/eval/analyze_fpn_dims.py` — Path-arg Market-1501 FPN dim-importance tool; re-runnable on any mamba ckpt (supports no_go #16 evidence, not a closed-run driver).
- `scripts/eval/analyze_fpn_embeddings.py` — Path-arg FPN intra/inter-ID cosine probe on any seq/ckpt/head; reusable embedding QA.
- `scripts/eval/sweep_external_fp_classifier.py` — Fully parameterized logistic TP/FP OP sweep (--rows-csv, --model-json); reusable beyond cascade-filter #25.
- `scripts/tools/average_top_trials.py` — Generic Optuna CLI (--study/--storage/--min-value); no fixed study identity.
- `scripts/tools/check_scan_bwd.py` — CUDA selective-scan bwd vs JIT autograd numerical smoke; kernel correctness, not a metric study.
- `scripts/tools/classify_gap_cause.py` — Generic gap-cause classifier on any --csv + --mot-dir + GT; reusable relink-pool analysis tool.
- `scripts/tools/compare_trials.py` — Generic Optuna trial table CLI; reusable HPO helper.

#### → archive-candidate

- `scripts/tools/format_tables.py` — (diagnostic → archive-candidate) One-shot string-replace that mutates src/saccade/perception/eval/runner.py print formats; migration leftover, not a re-runnable formatter.
- `scripts/tools/remap_aligned.py` — (diagnostic → archive-candidate) Hardcoded local→global ID maps for results/demo/custom_seq_*; one-shot demo glue, not a general remap CLI.
- `scripts/train/temporal_yolo/train_conditioned.py` — (experiment → archive-candidate) Option D conditioned-gate trainer; temporal_yolo/README marks archive-candidate; no_go #1.
- `scripts/train/temporal_yolo/train_gated_tp.py` — (experiment → archive-candidate) Older gated-TP recall trainer; README archive-candidate vs train_gated_detector/Mamba main-line.
- `scripts/train/temporal_yolo/train_jde_distill.py` — (experiment → archive-candidate) OSNet→JDE KD on Market; ReID side-line; README archive-candidate for main-line MOT.
- `scripts/train/temporal_yolo/train_jde_market.py` — (experiment → archive-candidate) Market-1501 JDE emb projector trainer; ReID experiment side-line; README archive-candidate.
- `scripts/train/temporal_yolo/train_joint.py` — (experiment → archive-candidate) Option C joint Cross-Attn trainer; superseded by Mamba-head stable trainers; README archive-candidate.
- `scripts/train/temporal_yolo/train_reid_1x1.py` — (experiment → archive-candidate) 1×1 dim-reduce ReID head on Market; ReID side-line; README archive-candidate.
- `scripts/train/train_reid_head.py` — (experiment → archive-candidate) FPN emb cache → SupCon ReID head; ties to weak appearance/FPN ceiling (no_go #2/#16); not main-line MOT.
- `scripts/train/train_temporal_yolo.py` — (experiment → archive-candidate) Early TemporalYOLOHybrid / Option C frozen-backbone trainer; superseded by train_joint then Mamba stack.
- `scripts/train_option_d.sh` — (experiment → archive-candidate) Option D 2-phase driver over train_conditioned.py; no_go #1 (harmful); docs/archive/option-d only.

#### → stable

_None._ (v14 protocol shells keep `experiment`: path-stable for the protocol, but still named-study drivers, not general MOT workflow entrypoints like `mot17.py`.)

#### unclear

_None._

## 3. Method notes

1. **Inputs:** `scratchpad/diag_exp_review_input.tsv` (138 rows). `current_label` treated as seed only.
2. **Per-file evidence:** module docstring / header comments / `# status:` / first ~40–80 lines; full body for short shells and for archive suspects.
3. **Named-study cross-check:** `docs/reference/no_go_registry.md` (#1 Option D, #2/#16 ReID/FPN, #7 OAO, #11 P5-2, #15 LaSt-ViT, #32 appearance relink, #38 height birth, #39 depth ordering, #41 horizon, #45 fuse_score_weight, #46 private-continuation/occ geometry, #52–#54 PP22) and `scripts/train/temporal_yolo/README.md` (explicit archive-candidate family + stable protocol path list).
4. **Decision rule:** reusable path-arg tool that still works on new runs/data → `diagnostic`; hardcoded study tags, fixed result dirs, Δ tables, or named protocol arms → `experiment`; superseded / one-shot migration / dead Option C/D & ReID side-line trainers → `archive-candidate`.
5. **Conservative on archive:** only marked when README already says archive-candidate, or body is an obvious one-shot demo/migration patch, or no_go #1 Option D lineage. Historical `run_pp22_*` / `run_v14*` / attribution probes stay `experiment` (docs/results still cite them).
6. **Not reviewed:** the 118 already-stable scripts; no `# status:` lines were edited in this pass.
7. **Apply recipe (for follow-up):** for each changed path, set `# status: <proposed>`, regenerate indexes via `scripts/tools/build_scripts_index.py`, run `scripts/tools/check_scripts_structure.py --strict`.

