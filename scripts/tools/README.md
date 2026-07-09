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
