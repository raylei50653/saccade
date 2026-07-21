# scripts/tools

Maintenance checks, rendering/conversion helpers, and research diagnostics that
are not primary eval, train, model-build, or benchmark entrypoints.

Review pass: 2026-06-18. No files were moved or deleted in this pass; this
README is the classification index for later cleanup.

## Rules

- Keep scripts referenced by docs, `scripts/pre_push.sh`, or eval entrypoints
  path-stable until those callers are updated.
- Treat relink, occlusion, depth-ordering, and Mamba diagnostics as experiment
  families. Move or delete them as families, not one file at a time.
- `out/` and `__pycache__/` contents are generated local artifacts, not part of
  the reviewed script surface.
- Manual probes named `test_*.py` are not pytest tests. Rename, move, or document
  them before treating them as automated coverage.

## Continuous Decimal Hash

The decimal-hash determinism family has four scripts with layered roles:

| Script | Role |
|---|---|
| `check_continuous_decimal_hash.py` | Primitive: single batch with arbitrary `--sequences` order |
| `check_decimal_chain_routine.py` | **Routine regression sentinel**: fixed continuous chain `A,A,B,A,B,B` |
| `check_decimal_matrix_2x2.py` | **Forensic / directional diagnostic**: four directed A/B cells |
| `check_decimal_matrix_all7.py` | **Deep / release validation**: full all-7 order-contamination matrix |

They share hash, capture, comparison, and artifact-serialization primitives
from ``src/saccade/perception/eval/_decimal_hash_tools.py``.

**Responsibility split**

| Layer | Question it answers | When to use |
|---|---|---|
| Continuous chain (routine) | Same process, continuous execution: is final serialized-MOT decimal output self-consistent? | Automatic pre-push on determinism-sensitive paths |
| 2×2 matrix (forensic) | After a failure: is it A→B, B→A, or self instability? | Manual diagnosis; cleaner cell-level references |
| All-7 matrix (deep) | Broader order contamination across all SDP sequences? | Release / major CUDA, buffer, graph, postprocess changes |

Do **not** treat the 2×2 matrix as the routine pre-push guard.

### Routine continuous chain (pre-push sentinel)

Fixed chain executed in **one** Python evaluator process / runtime state::

    A, A, B, A, B, B

**Fixed sequences** (frozen 2026-07-10):

* Sequence A: ``MOT17-04-SDP``  (1 050 frames, 44 248 output records)
* Sequence B: ``MOT17-02-SDP``  (600 frames, 11 722 output records)

**Comparison contract**

* First occurrence of each sequence is that sequence's reference.
* Later same-sequence occurrences compare against their first occurrence.
* Fail if record count or serialized decimal hash differs.
* Does **not** attribute contamination direction or root cause.

**Pre-push / manual run:**

```bash
uv run python scripts/tools/check_decimal_chain_routine.py \
  --preset mamba_whole_graph_m --detector SDP --double-buffer
```

Runs automatically in ``scripts/pre_push.sh`` when staged or committed changes
touch determinism-sensitive paths (native/CUDA filter, NMS, postprocess, CUDA
graphs, decimal-hash tooling).  Path detection is fail-closed.  To skip:

```bash
SKIP_DETERMINISM_PREPUSH=1 git push
```

Skipping is visible in console output and does not count as a validation pass.

On divergence the routine chain prints and retains:

* sequence occurrence;
* reference hash vs observed hash;
* record counts;
* first divergent frame;
* differing serialized records;
* final verdict;
* artifact directory (``out/determinism/routine_chain_<timestamp>/``).

### 2×2 Matrix (forensic / directional diagnostic)

Retained for post-failure diagnosis.  Four directed cells, each a consecutive
pair in one process::

    A → A    B → A
    A → B    B → B

Use after the routine chain fails to separate self instability from directional
cross-sequence contamination and to obtain cleaner cell-level references.

```bash
uv run python scripts/tools/check_decimal_matrix_2x2.py \
  --preset mamba_whole_graph_m --detector SDP --double-buffer
```

Artifacts::

    out/determinism/matrix_2x2_<timestamp>/
    ├── manifest.json
    ├── A_to_A/
    ├── B_to_B/
    ├── B_to_A/
    ├── A_to_B/
    ├── runs.csv
    ├── hashes.csv
    ├── comparisons.csv
    └── summary.json

### All-7 Matrix (deep / release validation)

Full 28-run order-contamination layout across every SDP sequence (forward,
reverse, forward again)::

```bash
uv run python scripts/tools/check_decimal_matrix_all7.py \
  --preset mamba_whole_graph_m --detector SDP --double-buffer
```

The routine chain (and preferably a clean 2×2 forensic pass after any prior
failure) should pass before all-7 is used for deeper investigation or release
validation.

### Primitive (single batch)

`check_continuous_decimal_hash.py` runs the complete ordered sequence list once
inside one Python process and captures each final MOT result before a repeated
sequence can overwrite its output file. It excludes global track IDs, sorts
`frame,x,y,w,h,score`, converts bbox fields to `x100` integers and score to an
`x10000` integer, and emits `summary.json`, `runs.csv`, `hashes.csv`, and
`mismatches.csv` to the requested output directory. It validates final
serialized decimal output only; it does not claim to hash internal tensors.
The routine chain wrapper reuses the same capture/compare primitives with a
frozen sequence order.

The matching standard eval path is:

```bash
uv run scripts/eval/mot17.py --preset mamba_whole_graph_m --detector SDP --double-buffer
```

Use the same runtime path for ad-hoc in-process probes:

```bash
uv run python scripts/tools/check_continuous_decimal_hash.py \
  --sequences MOT17-04-SDP,MOT17-04-SDP,MOT17-02-SDP,MOT17-04-SDP,MOT17-02-SDP,MOT17-02-SDP \
  --output out/determinism/continuous_decimal_hash \
  --preset mamba_whole_graph_m --detector SDP --double-buffer
```

All options other than the tool's own flags are forwarded to `scripts/eval/mot17.py`.
Do not pass `--processes` or `--cpp-threads`: the tool rejects them to preserve
the single-process Python-evaluator contract.

For native postprocess attribution, `--stage-probe-frames 120-130` emits
ordered and multiset hashes for `detector_output`, `post_nms`, and
`tracker_input`. The default `passive` mode uses D2D snapshots and defers D2H
hashing until sequence completion; `fenced` synchronizes before each snapshot.
`SACCADE_DETERMINISTIC_FILTER_COMPACTION=1` is a diagnostic-only native control
that preserves source-index order during filter compaction. It is not a
production setting because its single-thread CUDA kernel is intentionally slow.

## Referenced / Path-Sensitive

These have known repo references outside this README.

| Script | Role | Reference |
|---|---|---|
| `add_occlusion_to_seq.py` | Add synthetic occlusion to a demo sequence | Bidirectional relink roadmap |
| `analyze_bidir_relink.py` | Analyze bidirectional relink raw dumps | Bidirectional relink analysis doc |
| `analyze_consistency_stats.py` | Preflight cross-frame consistency-loss statistics | Mamba curriculum doc and runner |
| `analyze_kalman_h_signal.py` | NSA/Kalman height-conditioned noise analysis | Kalman recalibration doc |
| `analyze_missed_relinks.py` | Feature distributions for missed relinks | Bidirectional relink analysis |
| `analyze_preloss_motion.py` | GMC-compensated pre-loss motion statistics | Offline relink candidate analysis |
| `analyze_turn_baseline.py` | Control probe for pre-loss turning signal | Offline relink candidate analysis |
| `build_relink_candidates.py` | Build labelled offline relink candidate pool | Offline relink docs and feature scripts |
| `cache_gt_tracks.py` | GT oracle cache generator for archived Option-D work | Archived Option-D docs |
| `check_api_layers.py` | API layering audit | `scripts/pre_push.sh` |
| `check_doc_links.py` | Relative markdown link checker | `scripts/pre_push.sh` |
| `check_gpu_contract.py` | GPU-first contract checker | `scripts/pre_push.sh` and docs |
| `check_h0_bridge_decision_trace_contract.py` | Static H0 pre-seal field/sidecar coverage checker | H0 full-decision capture declaration |
| `check_h0_phase_a_archives.py` | Verify every immutable H0 Phase-A evidence root through its archive codec | H0 Repair gate |
| `check_h0_repair_acceptance_matrix.py` | Validate the prospective Repair/Seal/Execution gate matrix | H0 qualification CI and PR review |
| `check_continuous_decimal_hash.py` | In-process ID-free final MOT decimal consistency probe | Continuous-run determinism primitive |
| `check_decimal_chain_routine.py` | Fixed continuous chain `A,A,B,A,B,B` sentinel | Routine pre-push regression guard |
| `check_decimal_matrix_2x2.py` | Forensic 2×2 order-contamination matrix | Post-failure directional diagnosis |
| `check_decimal_matrix_all7.py` | Full all-7 order-contamination matrix | Deep / release determinism validation |
| `check_determinism_paths.py` | Detect determinism-sensitive staged/committed changes | Pre-push hook (fail-closed) |
| `depth_ordering_auc.py` | Depth-ordering AUC analysis | Crossing-swap depth ordering doc |
| `depth_ordering_probe.py` | Crossing-swap depth-ordering probe | NO-GO registry and gate sweep |
| `diagnose_id_switches.py` | ID-switch diagnostic helper | NO-GO registry / oracle script |
| `eval_golden.py` | Golden relink-candidate evaluation helper | Offline relink candidate analysis |
| `gap_occupancy_features.py` | Gap-occupancy appearance feature analysis | Native tracker source note |
| `gmc_rotation_probe.py` | GMC rotation no-go probe | NO-GO registry |
| `mamba_assigner_diagnostics.py` | Mamba assigner diagnostic probe | Mamba dual-resolution plan |
| `mamba_relink_features.py` | Mamba relink feature analysis | NO-GO registry and v14 docs |
| `migrate_legacy_mamba_cache_manifest.py` | Migrate old Mamba cache manifests | v14 parent refit runner |
| `motion_norm_probe.py` | Motion-normalization probe | NO-GO registry |
| `optimize_relink_weight.py` | Offline relink gate weight optimization | Offline relink candidate analysis |
| `qualify_h0_phase_a.py` | Non-authoritative real-host build/runtime substrate qualification | H0 Repair gate |
| `qualify_h0_phase_a_child.py` | Synthetic no-capture child for the qualification runner-launch check | H0 Repair gate |
| `verify_h0_phase_a_archive.py` | Versioned immutable H0 Phase-A archive verifier | H0 Repair gate |
| `render_diffusion_debug.py` | Render bidirectional relink debug events | Bidirectional relink roadmap |
| `render_mot_result.py` | Render MOT result overlays to video | Imported by eval CLI |
| `set_mamba_checkpoint_runtime.py` | Patch runtime semantics into Mamba checkpoints | v14 conversion ablation runner |
| `sweep_speed_turn.py` | Speed-vs-turning sweep | Offline relink candidate analysis |
| `test_gmc_cudagraph.py` | CUDA graph capture test for C++ GMC | Referenced in eval evaluator |
| `validate_reach_gate.py` | Validate reach-gate model on relink candidates | Offline relink candidate analysis |

## Maintenance / CI Checks

| Script | Role |
|---|---|
| `check_api_layers.py` | Warning-only source layering audit |
| `check_association_tools.py` | AssA tools registry (**R**) vs disk/NO-GO; `--list` / `--print-recipe` |
| `check_doc_links.py` | Markdown relative-link checker |
| `check_gpu_contract.py` | GPU-first host-roundtrip scanner |
| `gpu_check.py` | Local GPU environment check |
| `vram_monitor.sh` | Local VRAM monitor |

## Internal Support

| Script | Role |
|---|---|
| `README.md` | This classification index |

## Rendering / Conversion / Remap

| Script | Role |
|---|---|
| `add_occlusion_to_seq.py` | Add synthetic occluder boxes to image sequences |
| `render_diffusion_debug.py` | Render bidirectional relink debug videos |
| `render_mot_result.py` | Render MOT result overlays |

## Relink / Appearance Diagnostics

| Script | Role |
|---|---|
| `analyze_bidir_relink.py` | Bidirectional relink candidate/outcome analysis |
| `analyze_missed_relinks.py` | Missed relink feature distribution analysis |
| `analyze_relink_stats.py` | Relink feature stats over binary dumps |
| `birth_death_consistency.py` | Long-lived track birth/death appearance consistency |
| `build_relink_candidates.py` | Build labelled offline relink candidate pool |
| `color_relink_features.py` | Color-histogram features for relink candidates |
| `eval_golden.py` | Golden relink candidate evaluation |
| `gap_occupancy_features.py` | Gap-occupancy signal for appearance/relink analysis |
| `intra_track_consistency.py` | Within-track appearance consistency analysis |
| `mamba_relink_features.py` | Mamba features on relink candidates |
| `optimize_relink_weight.py` | Offline speed-weighted relink gate optimization |
| `osnet_relink_features.py` | OSNet upper-bound appearance probe |
| `sweep_href_variants.py` | h-ref normalization sweep for bridge score |
| `sweep_velocity_variants.py` | Exit/entry velocity estimator sweep |
| `validate_reach_gate.py` | Reach-gate validation on candidate table |

## Motion / Geometry / Occlusion Diagnostics

| Script | Role |
|---|---|
| `analyze_kalman_h_signal.py` | Height-conditioned Kalman/NSA noise analysis |
| `analyze_preloss_motion.py` | Pre-loss turning and area-change statistics |
| `analyze_turn_baseline.py` | Interior-vs-preloss turning control |
| `cold_start_transfer.py` | MOT17-to-MOT20 same-height occlusion gate transfer probe |
| `depth_ordering_auc.py` | Depth-ordering score/AUC analysis |
| `depth_ordering_gate_sweep.py` | Depth-ordering gate sweep |
| `depth_ordering_probe.py` | Crossing-swap depth-ordering probe |
| `diagnose_id_switches.py` | ID-switch diagnostics |
| `gmc_rotation_probe.py` | Rotation sensitivity probe for GMC/depth assumptions |
| `motion_norm_probe.py` | Motion-normalization diagnostic |
| `occ_candidate_analyze.py` | Occlusion-candidate gate analysis |
| `occ_event_values.py` | Continuous-value table for same-height occlusion events |
| `sweep_speed_turn.py` | Speed-vs-turning sweep |

## Mamba / Detection / Training Probes

| Script | Role |
|---|---|
| `analyze_consistency_stats.py` | Consistency-loss pair-count statistics |
| `analyze_fn_strata.py` | Detector false-negative stratification by size/visibility |
| `cache_gt_tracks.py` | GT oracle cache generation for archived gated-detector training |
| `mamba_assigner_diagnostics.py` | Mamba assigner diagnostics |
| `migrate_legacy_mamba_cache_manifest.py` | Legacy Mamba cache manifest migration |
| `probe_gate_separability.py` | Per-scale gate separability probe |
| `set_mamba_checkpoint_runtime.py` | Add explicit runtime semantics to Mamba checkpoints |

## Benchmark / Tuning Utilities

| Script | Role |
|---|---|
| `bench_bank_scatter.py` | Native bank scatter hot-path benchmark |
| `bench_pipeline_halves.py` | Detection-half vs tracker-half GPU timing split |

## Manual GPU / Native Probes

| Script | Role |
|---|---|
| `test_gmc_cudagraph.py` | CUDA graph capture test for C++ GMC |

## Cleanup Candidates

These had no external references in the first scan. Review as families before
deleting.

| Family | Candidate files |
|---|---|
| Unreferenced relink/occlusion probes | `analyze_fn_strata.py`, `analyze_relink_stats.py`, `birth_death_consistency.py`, `cold_start_transfer.py`, `depth_ordering_gate_sweep.py`, `occ_candidate_analyze.py`, `occ_event_values.py`, `probe_gate_separability.py`, `sweep_href_variants.py` |

## Script index

<!-- BEGIN generated script index -->
<!-- Generated by scripts/tools/build_scripts_index.py; do not edit this block by hand. -->

| Script | Status | Usage | Function |
|--------|--------|-------|----------|
| `add_occlusion_to_seq.py` | stable | cli | Inject simulated occlusion boxes into MOT sequences. |
| `analyze_bidir_relink.py` | stable | - | Per-candidate analysis of the bidirectional bridge-relink dumps. |
| `analyze_consistency_stats.py` | stable | cli | GT-level statistics for the cross-frame consistency term (route 3). |
| `analyze_fn_strata.py` | experiment | - | False-negative stratification: is the recall gap resolution-limited (small boxes, |
| `analyze_kalman_h_signal.py` | stable | cli | NSA/Kalman h-conditioned noise analysis + Phase-0 fits. |
| `analyze_m_b1_5_t0_region_interpretation.py` | stable | cli | T0-B: Existing Atlas Region Interpretation Pack (read-only derivation). |
| `analyze_missed_relinks.py` | stable | - | Analyze missed relinks: GT-labeled distributions of all features. |
| `analyze_preloss_motion.py` | stable | cli | Pre-loss motion statistics (GMC-compensated): turning magnitude + box-area change. |
| `analyze_relink_stats.py` | stable | - | Analyze relink feature stats: Gaussian separation per gap bin. |
| `analyze_turn_baseline.py` | stable | cli | Control for the pre-loss turning signal: is mid-track turning just as high? |
| `audit_frozen_packet_exact_key_recoverability.py` | stable | cli | EK0 frozen-packet exact-key recoverability audit (pure consistency audit). |
| `audit_relink_safe_reject.py` | stable | cli | Offline B1 safe-reject audit: max FP removal under GT_hurt <= ε. |
| `audit_runtime_bridge_decision_path.py` | stable | cli | P0: outcome-blind runtime bridge decision-path identifiability audit. |
| `average_top_trials.py` | experiment | cli | Print top Optuna trials and mean of their parameters. |
| `bench_bank_scatter.py` | stable | cli | Benchmark: P0/P1 bank scatter hot-path gain. |
| `bench_pipeline_halves.py` | experiment | cli | Measure the GPU-time split between the two per-frame pipeline halves. |
| `birth_death_consistency.py` | experiment | cli | Birth-vs-death appearance consistency for long-lived tracks. |
| `build_h0_preseal_freeze.py` | stable | cli | Assemble the sole H0 pre-seal artifact (``h0_preseal_freeze_v3``). |
| `build_r1_bridge_replay.sh` | stable | - | Build the research-only device replay helper used by R1 host R0. |
| `build_relink_candidates.py` | stable | cli | Build a relink-candidate dataset from a no-relink / no-interp MOT dump. |
| `build_scripts_index.py` | stable | cli | Generate the scripts/ discovery index from each script's own header. |
| `cache_gt_tracks.py` | stable | cli | cache_gt_tracks.py — Phase 1 GT Oracle Cache Generator |
| `check_api_layers.py` | stable | cli | Saccade API Layering Audit. |
| `check_association_tools.py` | stable | cli | Check association recovery tools registry (R) against disk (D) and NO-GO (N). |
| `check_continuous_decimal_hash.py` | stable | cli | Validate ID-free final-MOT decimal output consistency in one process. |
| `check_decimal_chain_routine.py` | stable | cli | Routine pre-push continuous-chain determinism sentinel. |
| `check_decimal_matrix_2x2.py` | stable | cli | Forensic / directional 2×2 sequence-order determinism matrix. |
| `check_decimal_matrix_all7.py` | stable | cli | Full all-7 sequence order-contamination validation. |
| `check_determinism_paths.py` | stable | - | Detect whether staged/committed changes affect determinism-sensitive paths. |
| `check_doc_freshness.py` | stable | cli | Warn-only documentation freshness / fact-ownership checks. |
| `check_doc_links.py` | stable | cli | Check that relative markdown links in docs resolve to existing files. |
| `check_doc_stale_paths.py` | stable | cli | Fail if any tracked file references a pre-move (stale) doc path. |
| `check_doc_structure.py` | stable | cli | Warn-only documentation structure / research index coverage checks. |
| `check_gpu_contract.py` | stable | - | Saccade GPU-First Performance Contract Checker. |
| `check_h0_bridge_decision_trace_contract.py` | stable | cli | Statically admit H0's complete capture ABI before an owner seal. |
| `check_h0_phase_a_archives.py` | stable | - | Verify every committed H0 Phase-A evidence root through archive codecs. |
| `check_h0_repair_acceptance_matrix.py` | stable | - | Validate the prospective H0 repair/qualification acceptance matrix. |
| `check_headline_decision_contract.py` | stable | cli | Static guard for the headline tracker-decision contract (no GPU). |
| `check_scan_bwd.py` | experiment | - | Validate the CUDA selective-scan backward against the JIT autograd reference. |
| `check_scripts_structure.py` | stable | cli | Scripts structure contract: every script self-documents, and the index is fresh. |
| `classify_gap_cause.py` | experiment | cli | Classify relink gaps: person-person overlap vs non-person. |
| `cold_start_transfer.py` | experiment | cli | Cold-start transfer test: do the normalized occ-gate landmarks hold on MOT20? |
| `color_relink_features.py` | stable | cli | Offline AUC test: color-histogram appearance features for relink candidates. |
| `combo_gate_safe_region.py` | stable | cli | 2D combination-gate surface + safe-region audit. |
| `compare_trials.py` | experiment | cli | Compare a selected set of Optuna trial results. |
| `convert_mot17_to_mp4.py` | diagnostic | - | Encode MOT17 image sequences to MP4 for visualization. |
| `convert_safe_region_asset_r1.py` | stable | cli | R1: Deterministic RegionAsset conversion from sealed Q4.5 + T0 evidence. |
| `convert_video_to_mot.py` | diagnostic | cli | Convert MP4 video into a MOT-compatible sequence folder. |
| `depth_ordering_auc.py` | experiment | - | Compute discrimination AUC for the depth-ordering (occlusion front/back) signal, |
| `depth_ordering_gate_sweep.py` | experiment | cli | Tier-1 offline gate sweep for the same-height occlusion gate (GT-only, free). |
| `depth_ordering_probe.py` | experiment | cli | Depth-ordering probe — can pre-occlusion geometry tell which of two crossing |
| `determinism_check.py` | diagnostic | cli | Determinism gate: run N evals per config and verify byte-identical output. |
| `diagnose_id_switches.py` | experiment | cli | Diagnose ID switches by gap type to determine P3-B (Dormant Bank + HNSW) value. |
| `energy_transform_separability.py` | stable | cli | Energy transform separability audit (raw / log1p / sqrt / rank). |
| `eval_golden.py` | diagnostic | cli | Bit-exact golden regression gate for run_eval refactors. |
| `export_d0_runtime_capture.py` | stable | cli | Merge per-sequence Issue #112 native captures into D0's CSV contract. |
| `export_headline_bridge_decision_trace.py` | stable | cli | Canonicalize H0 records plus its independent native-universe sidecar. |
| `export_r1_temporal_reduction_capture.py` | stable | cli | Seal native shadow observations into the R1 temporal-reduction payload. |
| `format_tables.py` | diagnostic | - | Format benchmark/eval stage tables for reports. |
| `gap_occupancy_features.py` | stable | cli | Gap-occupancy (exclusion) features for relink candidates. |
| `gate_clean_color.py` | stable | - | Test the 'use color ReID only on non-occluded relink candidates' gate. |
| `gate_rule_search.py` | stable | cli | Constrained multi-gate rule search (not combinatorial thrashing). |
| `gate_rule_search_loo.py` | stable | cli | Leave-one-sequence-out validation for gate_rule_search policies. |
| `gmc_rotation_probe.py` | experiment | cli | GMC rotation distribution — does the default affine GMC leak spurious rotation? |
| `gpu_check.py` | diagnostic | - | Sanity-check PyTorch/CUDA/GPU environment. |
| `graphify_cuda_patch.sh` | diagnostic | cli | Re-apply graphify CUDA (.cu/.cuh) AST support after a graphify upgrade. |
| `gt_safe_region_area.py` | stable | cli | GT-safe region area in GT-CDF / tail-mass coordinates (not raw thr). |
| `h0_launch_hygiene_gate.py` | stable | cli | Non-authoritative launch-hygiene pre-authorization gate for H0 Phase A. |
| `h0_runtime_confinement.py` | stable | - | Linux fail-closed runtime confinement and file-input attestation. |
| `horizon_convergence_probe.py` | experiment | cli | Horizon-convergence probe — does the pedestrian bbox scale field actually |
| `horizon_detector_test.py` | experiment | cli | Horizon homothety on REAL detections vs GT — does the vertex-homothety horizon |
| `horizon_homothety_probe.py` | experiment | cli | Horizon-convergence probe via VERTEX HOMOTHETY (design §3) — does the |
| `horizon_window_probe.py` | experiment | cli | Frame-to-frame horizon motion — the GMC signal test. |
| `intra_track_consistency.py` | experiment | cli | Intra-track appearance consistency: how stable is a feature on the SAME id? |
| `loo_hurt_attribution.py` | stable | cli | LOO GT-hurt attribution → atom classification → repair LOO compare. |
| `mamba_assigner_diagnostics.py` | stable | cli | Measure real TaskAlignedAssigner capacity for the v14 Mamba detector. |
| `mamba_relink_features.py` | stable | cli | Mamba-head feature probe for relink-candidate association matching. |
| `migrate_legacy_mamba_cache_manifest.py` | stable | cli | Add a validated lineage manifest to a legacy feature-only Mamba cache. |
| `mine_relink_signals.py` | stable | cli | Batch deep-mine continuous relink signals on a B1 pairs CSV. |
| `motion_norm_probe.py` | experiment | cli | §8.4 motion normalization — does scale-normalizing displacement improve the |
| `occ_candidate_analyze.py` | stable | cli | Phase B: threshold discriminability + GT-movement from the real-run candidate dump. |
| `occ_event_values.py` | stable | cli | Per-event ACTUAL-VALUE table for the same-height occlusion gate (one run, no thresholding). |
| `optimize_relink_weight.py` | stable | cli | Offline optimisation of the speed-weighted relink gate score. |
| `osnet_relink_features.py` | stable | cli | Deep-ReID upper bound for relink-candidate appearance matching. |
| `probe_gate_separability.py` | experiment | - | Probe: is the spatial gate per-scale separable? |
| `probe_relink_occlusion_signal.py` | stable | cli | Does an explicit OCCLUSION signal separate true vs false relink bridges? |
| `qualify_h0_phase_a.py` | stable | cli | Run the repeatable, non-authoritative H0 substrate qualification gate. |
| `qualify_h0_phase_a_child.py` | stable | cli | Synthetic no-capture child used only by H0 substrate qualification. |
| `remap_aligned.py` | diagnostic | - | Remap/dedupe aligned track IDs (keep highest score per frame-id). |
| `remap_gpu_relinks.py` | stable | - | Remap track IDs after GPU relink using global ID mapping. |
| `render_diffusion_debug.py` | stable | cli | Render bidirectional relink debug events emitted by ``mot17.py``. |
| `render_mot_result.py` | stable | cli | Render a MOT-format tracking result onto the source frames and encode a video. |
| `repaired_tail_or_safe_region.py` | stable | cli | Safe-region thickness for frozen all-tail OR candidate (not new policy search). |
| `resolved_bridge_policy_config.py` | stable | cli | Single authority for `resolved_bridge_policy_config_v1` fingerprints. |
| `run_d0_runtime_shadow_fidelity.py` | stable | cli | D0 runtime shadow bridge fidelity — terminal verifier (Issue #112, v2). |
| `run_door0_ranking_probe.py` | stable | cli | Door 0 — ambiguous-band ranking-power probe runner. |
| `run_h0_phase_a.py` | stable | cli | A7/RC1 fail-closed Phase-A parent controller. |
| `run_h0_phase_a_child.py` | stable | - | RC1 fixed Phase-A runtime child (parent-only entry point). |
| `run_m_b1_5_stage2_q1q3.py` | stable | cli | M-B1.5 Stage 2 Q1–Q3 runner: D_online label join + safe-negative mass audit. |
| `run_m_b1_5_stage2_q4.py` | stable | cli | M-B1.5 Stage 2 Q4 runner: signal separability on D_online. |
| `run_m_b1_5_stage2_q45_atlas.py` | stable | cli | M-B1.5 Stage 2 Q4.5: structured threshold-combination atlas. |
| `run_m_b1_hook_ab.py` | stable | cli | Stage 1 A/B + Stage 1b action-path control runner for portable OR-tail hook. |
| `run_s0_safe_domain_runtime_transfer.py` | stable | cli | Execute sealed S0 Amendment 1: offline-to-runtime safe-axis transfer. |
| `run_safe_region_a1_audit.py` | stable | cli | A1 read-only acceptance audit for the safe-region A0 conversion pack. |
| `run_safe_region_assetization_r1.py` | stable | cli | Run Safe-Region Assetization R1 study (Phase A assets + Phase B linear probe). |
| `run_safe_region_assetization_r11.py` | stable | cli | Run R1.1 Transfer Failure Attribution Pack (authorized research only). |
| `set_mamba_checkpoint_runtime.py` | stable | cli | Create a weight-identical Mamba checkpoint with explicit runtime semantics. |
| `setup_services.sh` | stable | - | Saccade Systemd 服務安裝腳本 |
| `smoke_repaired_candidate_b2e2e.py` | stable | cli | Narrow B2/e2e smoke contract for frozen repaired candidate only. |
| `summarize_relink_pairs.py` | stable | cli | Summarize offline relink candidate pairs into a B1 study directory. |
| `sweep_href_variants.py` | experiment | cli | Offline sweep of h_ref normalization variants for the bridge score. |
| `sweep_speed_turn.py` | stable | cli | Sweep per-step speed vs turning: at what move-speed/box-height ratio does |
| `sweep_velocity_variants.py` | experiment | cli | Offline sweep of lost-track exit velocity estimation schemes for the bridge score. |
| `test_cufft_graph.py` | diagnostic | cli | Minimal cuFFT CUDA graph capture test. |
| `test_gmc_cudagraph.py` | diagnostic | cli | Capture C++ estimate_into_direct in torch.cuda.CUDAGraph. |
| `test_gpu_gmc.py` | diagnostic | - | Manual GPU GMC correctness/perf smoke test. |
| `test_letterbox_gpu.py` | diagnostic | - | Unit test: cpp_letterbox_gpu output matches PyTorch 3-op reference. |
| `test_py_gmc.py` | diagnostic | cli | Pure-Python GMC with PyTorch FFT, designed for CUDA graph capture. |
| `validate_reach_gate.py` | stable | cli | Validate the reach-gate model R_total(G) = s*G + R_search(G) against the |
| `verify_gmc_direct.py` | diagnostic | cli | Verify GMC estimate_into_direct produces identical warp as estimate_into. |
| `verify_h0_gctm_guarantee_registration.py` | stable | cli | Fail-closed validator for H0-to-GCTM guarantee registration records. |
| `verify_h0_phase_a.py` | stable | cli | Independent A7/RC1 aggregate verifier for Phase-A execution evidence. |
| `verify_h0_phase_a_archive.py` | stable | cli | Versioned archive verifier for immutable H0 Phase-A evidence. |
| `verify_h0_preseal_freeze.py` | stable | cli | Independently verify a canonical ``h0_preseal_freeze_v3`` artifact. |
| `verify_headline_bridge_decision_trace.py` | stable | cli | Replay and validate the sealed H0 bridge-decision trace from capture alone. |
| `verify_r1_temporal_reduction_replay.py` | stable | cli | Verify R1's estimator replay without reading labels or fitting a score. |
| `vram_monitor.sh` | archive-candidate | - | Empty stub (no implementation) for VRAM monitoring. |
| `weight_method_safe_region.py` | stable | cli | Compare weighting methods by GT-safe *productive region* (not best FP). |

<!-- END generated script index -->
