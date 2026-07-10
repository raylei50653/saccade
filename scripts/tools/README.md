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

The decimal-hash determinism family has three scripts with distinct roles:

| Script | Role |
|---|---|
| `check_continuous_decimal_hash.py` | Primitive: single batch execution with arbitrary `--sequences` order |
| `check_decimal_matrix_2x2.py` | Pre-push 2×2 matrix: four directed cells between fixed A/B sequences |
| `check_decimal_matrix_all7.py` | Full all-7 order-contamination matrix (28 runs) |

They share hash, capture, comparison, and artifact-serialization primitives
from ``src/saccade/perception/eval/_decimal_hash_tools.py``.

### 2×2 Matrix (pre-push regression guard)

The 2×2 matrix runs four directed cells::

    A → A    B → A
    A → B    B → B

Each cell means two full sequences executed consecutively within the same
process and runtime state.  The matrix detects:

1. same-sequence continuous-run instability (A→A, B→B self cells);
2. cross-sequence state or buffer contamination (B→A, A→B);
3. directional contamination differences (B→A vs A→B may differ);
4. final serialized-MOT decimal divergence.

**Fixed sequence pair** (frozen 2026-07-10):

* Sequence A: ``MOT17-04-SDP``  (1 050 frames, 44 248 output records)
* Sequence B: ``MOT17-02-SDP``  (600 frames, 11 722 output records)

B was selected from all-7 matrix evidence.  Its 4× smaller detection footprint
exercises substantially different postprocess buffer utilisation and CUDA-graph
static sizing on either side of the 04 footprint.  It was self-consistent in
all all-7 comparisons and the nearest-neighbour runs survived directional
cross-contamination at the record level.

**Manual run:**

```bash
uv run python scripts/tools/check_decimal_matrix_2x2.py \
  --preset mamba_whole_graph_m --detector SDP --double-buffer
```

All unrecognised options are forwarded to ``scripts/eval/mot17.py``.

**Pre-push integration:**

The matrix runs automatically in ``scripts/pre_push.sh`` when staged or
committed changes touch determinism-sensitive paths (native/CUDA filter, NMS,
postprocess, CUDA graphs, decimal-hash tooling).  To skip:

```bash
SKIP_DETERMINISM_PREPUSH=1 git push
```

Skipping is visible in console output and does not count as a validation pass.

The 2×2 matrix is a routine regression guard; it does **not** provide coverage
equivalent to the all-7 matrix.  For deeper validation, use the all-7 matrix.

### All-7 Matrix (deeper validation)

The all-7 matrix runs the full 28-run order-contamination layout across every
SDP sequence (forward, reverse, forward again)::

```bash
uv run python scripts/tools/check_decimal_matrix_all7.py \
  --preset mamba_whole_graph_m --detector SDP --double-buffer
```

The 2×2 matrix should pass before the all-7 matrix is used for deeper
investigation or release validation.

### Failure diagnostics

On divergence the 2×2 matrix prints:

* matrix cell (e.g. ``B_to_A``);
* preceding and target sequences;
* reference hash vs observed hash;
* record counts;
* first divergent frame;
* differing serialized records;
* artifact directory (``out/determinism/matrix_2x2_<timestamp>/``).

Artifacts are organized as::

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

### Primitive (single batch)

`check_continuous_decimal_hash.py` runs the complete ordered sequence list once
inside one Python process and captures each final MOT result before a repeated
sequence can overwrite its output file. It excludes global track IDs, sorts
`frame,x,y,w,h,score`, converts bbox fields to `x100` integers and score to an
`x10000` integer, and emits `summary.json`, `runs.csv`, `hashes.csv`, and
`mismatches.csv` to the requested output directory. It validates final
serialized decimal output only; it does not claim to hash internal tensors.

The matching standard eval path is:

```bash
uv run scripts/eval/mot17.py --preset mamba_whole_graph_m --detector SDP --double-buffer
```

Use the same runtime path for the in-process consistency probe:

```bash
uv run python scripts/tools/check_continuous_decimal_hash.py \
  --sequences MOT17-04-SDP,MOT17-04-SDP,MOT17-04-SDP,MOT17-04-SDP,MOT17-04-SDP \
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

For the order-contamination matrix, pass all blocks as one comma-separated
argument in this order: `04`, `04,02,04`, `02,04,02`, all seven SDP sequences
forward, all seven reverse, then all seven forward again. The resulting
`summary.json` compares every occurrence with the first occurrence of that
sequence, retaining the run index and full preceding sequence order.

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
| `check_continuous_decimal_hash.py` | In-process ID-free final MOT decimal consistency probe | Continuous-run determinism validation |
| `check_decimal_matrix_2x2.py` | Pre-push 2×2 order-contamination matrix | Routine determinism regression guard |
| `check_decimal_matrix_all7.py` | Full all-7 order-contamination matrix | Deeper determinism validation |
| `check_determinism_paths.py` | Detect determinism-sensitive staged/committed changes | Pre-push hook |
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
