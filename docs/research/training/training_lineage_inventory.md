<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-09-18 -->
<!-- doc-module: detection -->
<!-- Captured by scripts/provenance/training_lineage.py; regenerate rather than edit. -->

# Training lineage inventory (#421 · deliverable 1)

Captured 2026-09-18T12:25:31+00:00 on `DESKTOP-0FLA6SQ` at `bf3e75fc7de5` (dirty tree); tensor diff on, ONNX match on. Machine-readable twin: `report_data/training_lineage_inventory.json`. Role claims: `scripts/provenance/training_lineage_roles.json`.

This is a **captured snapshot of one workspace**, not a regenerable view: `runs/` and `models/` are gitignored. Every column except *role*, *note* and the `expected_*` claims is read from the artifact bytes. `unavailable` means the path does not exist here; no substitute was used. Numbers here are identities and counts, never quality metrics.

## Reading the tables

- **init** = warm-start parent recorded in the checkpoint's own `args.mamba_ckpt`; **teacher** / **base_yolo** / **cache** likewise from `args`. `unlisted:` = the artifact names a path that is not a declared role.
- **expected_*** = the role file's claim checked against the derived edge: `ok` / `mismatch` / `unverifiable` (checkpoint records no such edge).
- **sha attested** = whether `mamba_args.base_yolo_sha256` / `teacher_checkpoint_sha256` exist and verify against the file on disk (`not_recorded` for pre-2026-06-13 checkpoints).
- **SSM interior frozen** = along the init edge, every `A_log/D/conv1d/x_proj/dt_proj` tensor of `mamba_blocks` and `temporal_blocks` is bit-identical to the parent. This is the measured counterpart of the `scan_stop_grad` flag.
- **engine ← ckpt** = the engine's sibling ONNX matched bit-exactly against every checkpoint in the inventory (teachers BN-folded). `unique_exact` names the source; `partial` / `ambiguous` do not identify one.

## Family `s` — backbone yolo26s

### Artifacts

| role | kind | path | status | sha256 | size | epoch | seed |
|---|---|---|---|---|---:|---:|---:|
| `s.pretrained_yolo` | yolo_pt | `models/yolo/yolo26s.pt` | present | `646f8bc3fe0a` | 20.4 MB | — | — |
| `s.legacy_teacher` | gated_teacher_ckpt | `runs/gated_det_v1/best.ckpt` | present | `d2ace71d47c3` | 116.7 MB | 12 | — |
| `s.legacy_cache` | teacher_cache | `runs/trt_feat_cache_v2` | **unavailable** | — | — | — | — |
| `s.legacy_distill` | mamba_ckpt | `runs/mamba_distill_pixelshuffle_crossscan/best.ckpt` | **unavailable** | — | — | — | — |
| `s.legacy_parent_gt` | mamba_ckpt | `runs/mamba_gt_pixelshuffle_crossscan/best.ckpt` | present | `604810823cc9` | 121.2 MB | 30 | — |
| `s.legacy_v14` | mamba_ckpt | `runs/mamba_gt_vgt_mamba_v14/best.ckpt` | present | `7d424c82729a` | 121.2 MB | 58 | — |
| `s.controlled_refit` | mamba_ckpt | `runs/mamba_gt_v14_parent_n16_frozen_refit/best.ckpt` | present | `e3161423dd4e` | 121.2 MB | 29 | 42 |
| `s.adapted_teacher` | gated_teacher_ckpt | `runs/gated_det_v14replica/epoch_0012.ckpt` | present | `08d1d68dc8fb` | 116.8 MB | 12 | 20260612 |
| `s.teacher_cache` | teacher_cache | `runs/mamba_teacher_cache_v14replica` | **unavailable** | — | — | — | — |
| `s.distill` | mamba_ckpt | `runs/mamba_distill_v14replica/best.ckpt` | present | `6129cb71dd43` | 121.2 MB | 30 | 20260612 |
| `s.gt1` | mamba_ckpt | `runs/mamba_gt_v14replica_stage1/best.ckpt` | present | `ff40f07e9c27` | 121.2 MB | 29 | 20260612 |
| `s.gt2_plain` | mamba_ckpt | `runs/mamba_gt_v14replica_final/best.ckpt` | present | `c1acd2bb5064` | 121.2 MB | 29 | 20260612 |
| `s.t3t1_phase_a` | mamba_ckpt | `runs/mamba_gt_v14replica_t3/best.ckpt` | present | `0099ecde4bd5` | 128.6 MB | 15 | 42 |
| `s.t3t1_phase_b` | mamba_ckpt | `runs/mamba_gt_v14replica_t3_t1/best.ckpt` | present | `c161c88e50b8` | 126.2 MB | 15 | 42 |
| `s.implicit_s42` | mamba_ckpt | `runs/mamba_gt_v14replica_implicit_s42/best.ckpt` | present | `be8ef8612aca` | 121.2 MB | 30 | 42 |
| `s.implicit_s43` | mamba_ckpt | `runs/mamba_gt_v14replica_implicit_s43/best.ckpt` | present | `57ad6750f003` | 121.2 MB | 30 | 43 |
| `s.implicit_s44` | mamba_ckpt | `runs/mamba_gt_v14replica_implicit_s44/best.ckpt` | present | `a15ca3fdde8e` | 121.2 MB | 30 | 44 |
| `s.t3t1_phase_a_shared_s43` | mamba_ckpt | `runs/mamba_gt_v14replica_t3_shared_s43/best.ckpt` | present | `6a7cd5347772` | 128.6 MB | 15 | 43 |
| `s.t3t1_shared_s43` | mamba_ckpt | `runs/mamba_gt_v14replica_t3_t1_shared_s43/best.ckpt` | present | `b6f9f7e6222b` | 126.2 MB | 15 | 43 |
| `s.t3t1_phase_a_shared_s44` | mamba_ckpt | `runs/mamba_gt_v14replica_t3_shared_s44/best.ckpt` | present | `5ebb150937e6` | 128.6 MB | 15 | 44 |
| `s.t3t1_shared_s44` | mamba_ckpt | `runs/mamba_gt_v14replica_t3_t1_shared_s44/best.ckpt` | present | `d540887b48ae` | 126.2 MB | 15 | 44 |
| `s.distill_s13` | mamba_ckpt | `runs/mamba_distill_v14replica_s13/best.ckpt` | present | `a192dc6e935b` | 121.2 MB | 30 | 20260613 |
| `s.gt1_s13` | mamba_ckpt | `runs/mamba_gt_v14replica_s13_stage1/best.ckpt` | present | `5775a83dc8c0` | 121.2 MB | 29 | 20260613 |
| `s.gt2_plain_s13` | mamba_ckpt | `runs/mamba_gt_v14replica_s13_final/best.ckpt` | present | `8295101e71f3` | 121.2 MB | 29 | 20260613 |
| `s.t3t1_phase_a_s13` | mamba_ckpt | `runs/mamba_gt_v14replica_t3_s13/best.ckpt` | present | `16268a681cda` | 128.6 MB | 15 | 20260613 |
| `s.t3t1_s13` | mamba_ckpt | `runs/mamba_gt_v14replica_t3_t1_s13/best.ckpt` | present | `32476cb099c9` | 126.2 MB | 15 | 20260613 |
| `s.distill_s14` | mamba_ckpt | `runs/mamba_distill_v14replica_s14/best.ckpt` | present | `3d5737234488` | 121.2 MB | 30 | 20260614 |
| `s.gt1_s14` | mamba_ckpt | `runs/mamba_gt_v14replica_s14_stage1/best.ckpt` | present | `c7ed1093a8dc` | 121.2 MB | 29 | 20260614 |
| `s.gt2_plain_s14` | mamba_ckpt | `runs/mamba_gt_v14replica_s14_final/best.ckpt` | present | `1b2704ad8fcd` | 121.2 MB | 30 | 20260614 |
| `s.t3t1_phase_a_s14` | mamba_ckpt | `runs/mamba_gt_v14replica_t3_s14/best.ckpt` | present | `2c09699b0f5c` | 128.6 MB | 15 | 20260614 |
| `s.t3t1_s14` | mamba_ckpt | `runs/mamba_gt_v14replica_t3_t1_s14/best.ckpt` | present | `0af6582e613d` | 126.2 MB | 15 | 20260614 |
| `s.deployment_preset` | preset | `configs/presets/mamba_whole_graph.yaml` | present | `093b66ed1240` | 5.5 KB | — | — |
| `s.backbone_engine` | trt_engine | `models/yolo/yolo26s_backbone_640_best.engine` | present | `2ef3d4d40dfb` | 20.3 MB | — | — |
| `s.backbone_engine_v14replica_e12` | trt_engine | `models/yolo/yolo26s_backbone_640_v14replica_e12.engine` | present | `e244e8fe813a` | 20.4 MB | — | — |
| `s.cpp_head_script` | torchscript | `models/yolo/mamba_head_best.pt` | present | `16310ddafe76` | 44.4 MB | — | — |

### Mamba checkpoints — derived edges and checks

| role | init (derived) | teacher (derived) | cache (derived) | expected init | expected teacher | base_yolo sha | teacher sha | SSM interior frozen | tensors identical/changed/added | groups left bit-identical |
|---|---|---|---|---|---|---|---|---|---|---|
| `s.legacy_parent_gt` | s.legacy_distill (missing) | s.legacy_teacher | — | ok | ok | not_recorded | not_recorded | — | parent s.legacy_distill unavailable | — |
| `s.legacy_v14` | self (resume invocation) | s.legacy_teacher | s.legacy_cache (missing) | unverifiable | ok | not_recorded | not_recorded | — | no_init_parent_in_inventory | — |
| `s.controlled_refit` | s.legacy_parent_gt | s.legacy_teacher | s.legacy_cache (missing) | ok | ok | verified | verified | yes | 21/48/0 vs `s.legacy_parent_gt` | mamba_blocks.ssm_internal |
| `s.distill` | — | s.adapted_teacher | s.teacher_cache (missing) | — | ok | not_recorded | not_recorded | — | no_init_parent_in_inventory | — |
| `s.gt1` | s.distill | s.adapted_teacher | — | ok | ok | not_recorded | not_recorded | yes | 21/48/0 vs `s.distill` | mamba_blocks.ssm_internal |
| `s.gt2_plain` | s.gt1 | s.adapted_teacher | s.teacher_cache (missing) | ok | ok | not_recorded | not_recorded | yes | 21/48/0 vs `s.gt1` | mamba_blocks.ssm_internal |
| `s.t3t1_phase_a` | s.gt1 | s.adapted_teacher | s.teacher_cache (missing) | ok | ok | verified | verified | yes | 21/48/39 vs `s.gt1` | mamba_blocks.ssm_internal |
| `s.t3t1_phase_b` | s.t3t1_phase_a | s.adapted_teacher | s.teacher_cache (missing) | ok | ok | not_recorded | not_recorded | yes | 54/54/0 vs `s.t3t1_phase_a` | flow_gate_conv, mamba_blocks.ssm_internal, temporal_blocks.ssm_internal |
| `s.implicit_s42` | s.gt1 | s.adapted_teacher | s.teacher_cache (missing) | ok | ok | not_recorded | not_recorded | yes | 21/48/0 vs `s.gt1` | mamba_blocks.ssm_internal |
| `s.implicit_s43` | s.gt1 | s.adapted_teacher | s.teacher_cache (missing) | ok | ok | not_recorded | not_recorded | yes | 21/48/0 vs `s.gt1` | mamba_blocks.ssm_internal |
| `s.implicit_s44` | s.gt1 | s.adapted_teacher | s.teacher_cache (missing) | ok | ok | not_recorded | not_recorded | yes | 21/48/0 vs `s.gt1` | mamba_blocks.ssm_internal |
| `s.t3t1_phase_a_shared_s43` | s.gt1 | s.adapted_teacher | s.teacher_cache (missing) | ok | ok | not_recorded | not_recorded | yes | 21/48/39 vs `s.gt1` | mamba_blocks.ssm_internal |
| `s.t3t1_shared_s43` | s.t3t1_phase_a_shared_s43 | s.adapted_teacher | s.teacher_cache (missing) | ok | ok | not_recorded | not_recorded | yes | 60/48/0 vs `s.t3t1_phase_a_shared_s43` | flow_gate_conv, mamba_blocks.ssm_internal, temporal_blocks.proj, temporal_blocks.ssm_internal |
| `s.t3t1_phase_a_shared_s44` | s.gt1 | s.adapted_teacher | s.teacher_cache (missing) | ok | ok | not_recorded | not_recorded | yes | 21/48/39 vs `s.gt1` | mamba_blocks.ssm_internal |
| `s.t3t1_shared_s44` | s.t3t1_phase_a_shared_s44 | s.adapted_teacher | s.teacher_cache (missing) | ok | ok | not_recorded | not_recorded | yes | 60/48/0 vs `s.t3t1_phase_a_shared_s44` | flow_gate_conv, mamba_blocks.ssm_internal, temporal_blocks.proj, temporal_blocks.ssm_internal |
| `s.distill_s13` | — | s.adapted_teacher | s.teacher_cache (missing) | — | ok | not_recorded | not_recorded | — | no_init_parent_in_inventory | — |
| `s.gt1_s13` | s.distill_s13 | s.adapted_teacher | — | ok | ok | not_recorded | not_recorded | yes | 21/48/0 vs `s.distill_s13` | mamba_blocks.ssm_internal |
| `s.gt2_plain_s13` | s.gt1_s13 | s.adapted_teacher | s.teacher_cache (missing) | ok | ok | not_recorded | not_recorded | yes | 21/48/0 vs `s.gt1_s13` | mamba_blocks.ssm_internal |
| `s.t3t1_phase_a_s13` | s.gt1_s13 | s.adapted_teacher | s.teacher_cache (missing) | ok | ok | not_recorded | not_recorded | yes | 21/48/39 vs `s.gt1_s13` | mamba_blocks.ssm_internal |
| `s.t3t1_s13` | s.t3t1_phase_a_s13 | s.adapted_teacher | s.teacher_cache (missing) | ok | ok | not_recorded | not_recorded | yes | 60/48/0 vs `s.t3t1_phase_a_s13` | flow_gate_conv, mamba_blocks.ssm_internal, temporal_blocks.proj, temporal_blocks.ssm_internal |
| `s.distill_s14` | — | s.adapted_teacher | s.teacher_cache (missing) | — | ok | not_recorded | not_recorded | — | no_init_parent_in_inventory | — |
| `s.gt1_s14` | s.distill_s14 | s.adapted_teacher | — | ok | ok | not_recorded | not_recorded | yes | 21/48/0 vs `s.distill_s14` | mamba_blocks.ssm_internal |
| `s.gt2_plain_s14` | s.gt1_s14 | s.adapted_teacher | s.teacher_cache (missing) | ok | ok | not_recorded | not_recorded | yes | 21/48/0 vs `s.gt1_s14` | mamba_blocks.ssm_internal |
| `s.t3t1_phase_a_s14` | s.gt1_s14 | s.adapted_teacher | s.teacher_cache (missing) | ok | ok | not_recorded | not_recorded | yes | 21/48/39 vs `s.gt1_s14` | mamba_blocks.ssm_internal |
| `s.t3t1_s14` | s.t3t1_phase_a_s14 | s.adapted_teacher | s.teacher_cache (missing) | ok | ok | not_recorded | not_recorded | yes | 60/48/0 vs `s.t3t1_phase_a_s14` | flow_gate_conv, mamba_blocks.ssm_internal, temporal_blocks.proj, temporal_blocks.ssm_internal |

### Mamba checkpoints — recorded training treatment

| role | epochs | lr | clip_len | clip_stride | gt_ratio | add_temporal | scan_stop_grad | seqs | holdout | temporal blocks | params |
|---|---:|---:|---:|---:|---:|---|---|---|---|---|---:|
| `s.legacy_parent_gt` | 30 | 0.0001 | 4 | — | 0.5 | — | — | all (default) | none | no | 10,126,636 |
| `s.legacy_v14` | 60 | 0.0003 | 4 | — | 0.5 | False | — | all (default) | none | no | 10,126,636 |
| `s.controlled_refit` | 30 | 0.0001 | 4 | 4 | 0.0 | False | True | 7-seq explicit | none | no | 10,126,636 |
| `s.distill` | 30 | 0.001 | 1 | 0 | — | — | True | 7-seq explicit | none | no | 10,126,636 |
| `s.gt1` | 30 | 0.0001 | 4 | 8 | 0.5 | False | True | all (default) | none | no | 10,126,636 |
| `s.gt2_plain` | 30 | 0.0001 | 4 | 8 | 0.0 | False | True | all (default) | none | no | 10,126,636 |
| `s.t3t1_phase_a` | 15 | 0.0001 | 3 | 6 | 0.0 | True | True | all (default) | none | yes | 11,368,540 |
| `s.t3t1_phase_b` | 15 | 0.0001 | 1 | 2 | 0.0 | False | True | all (default) | none | yes | 11,368,540 |
| `s.implicit_s42` | 30 | 0.0001 | 4 | 8 | 0.0 | False | True | all (default) | none | no | 10,126,636 |
| `s.implicit_s43` | 30 | 0.0001 | 4 | 8 | 0.0 | False | True | all (default) | none | no | 10,126,636 |
| `s.implicit_s44` | 30 | 0.0001 | 4 | 8 | 0.0 | False | True | all (default) | none | no | 10,126,636 |
| `s.t3t1_phase_a_shared_s43` | 15 | 0.0001 | 3 | 6 | 0.0 | True | True | all (default) | none | yes | 11,368,540 |
| `s.t3t1_shared_s43` | 15 | 0.0001 | 1 | 2 | 0.0 | False | True | all (default) | none | yes | 11,368,540 |
| `s.t3t1_phase_a_shared_s44` | 15 | 0.0001 | 3 | 6 | 0.0 | True | True | all (default) | none | yes | 11,368,540 |
| `s.t3t1_shared_s44` | 15 | 0.0001 | 1 | 2 | 0.0 | False | True | all (default) | none | yes | 11,368,540 |
| `s.distill_s13` | 30 | 0.001 | 1 | 0 | — | — | True | 7-seq explicit | none | no | 10,126,636 |
| `s.gt1_s13` | 30 | 0.0001 | 4 | 8 | 0.5 | False | True | all (default) | none | no | 10,126,636 |
| `s.gt2_plain_s13` | 30 | 0.0001 | 4 | 8 | 0.0 | False | True | all (default) | none | no | 10,126,636 |
| `s.t3t1_phase_a_s13` | 15 | 0.0001 | 3 | 6 | 0.0 | True | True | all (default) | none | yes | 11,368,540 |
| `s.t3t1_s13` | 15 | 0.0001 | 1 | 2 | 0.0 | False | True | all (default) | none | yes | 11,368,540 |
| `s.distill_s14` | 30 | 0.001 | 1 | 0 | — | — | True | 7-seq explicit | none | no | 10,126,636 |
| `s.gt1_s14` | 30 | 0.0001 | 4 | 8 | 0.5 | False | True | all (default) | none | no | 10,126,636 |
| `s.gt2_plain_s14` | 30 | 0.0001 | 4 | 8 | 0.0 | False | True | all (default) | none | no | 10,126,636 |
| `s.t3t1_phase_a_s14` | 15 | 0.0001 | 3 | 6 | 0.0 | True | True | all (default) | none | yes | 11,368,540 |
| `s.t3t1_s14` | 15 | 0.0001 | 1 | 2 | 0.0 | False | True | all (default) | none | yes | 11,368,540 |

### Warm-start edges — what the recorded treatment changed

Keys both checkpoints recorded with different values. Flags that only the child records (added to the training script later) are counted, not interpreted.

- `s.legacy_parent_gt` → `s.controlled_refit`: `gt_ratio` 0.5→0.0; `seqs` ''→'MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP' (+30 keys only in child, 0 only in parent)
- `s.distill` → `s.gt1`: `batch_size` 8→4; `clip_len` 1→4; `clip_stride` 0→8; `lr` 0.001→0.0001; `seqs` 'MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP'→''; `use_pixel_shuffle` True→False (+20 keys only in child, 4 only in parent)
- `s.gt1` → `s.gt2_plain`: `gt_ratio` 0.5→0.0
- `s.gt1` → `s.t3t1_phase_a`: `add_temporal` False→True; `clip_len` 4→3; `clip_stride` 8→6; `epochs` 30→15; `gt_ratio` 0.5→0.0; `seed` 20260612→42; `warmup_epochs` 5→3 (+11 keys only in child, 0 only in parent)
- `s.t3t1_phase_a` → `s.t3t1_phase_b`: `add_temporal` True→False; `clip_len` 3→1; `clip_stride` 6→2 (+0 keys only in child, 10 only in parent)
- `s.gt1` → `s.implicit_s42`: `gt_ratio` 0.5→0.0; `seed` 20260612→42; `warmup_epochs` 5→3 (+9 keys only in child, 0 only in parent)
- `s.gt1` → `s.implicit_s43`: `gt_ratio` 0.5→0.0; `seed` 20260612→43; `warmup_epochs` 5→3 (+9 keys only in child, 0 only in parent)
- `s.gt1` → `s.implicit_s44`: `gt_ratio` 0.5→0.0; `seed` 20260612→44; `warmup_epochs` 5→3 (+9 keys only in child, 0 only in parent)
- `s.gt1` → `s.t3t1_phase_a_shared_s43`: `add_temporal` False→True; `clip_len` 4→3; `clip_stride` 8→6; `epochs` 30→15; `gt_ratio` 0.5→0.0; `seed` 20260612→43; `warmup_epochs` 5→3 (+9 keys only in child, 0 only in parent)
- `s.t3t1_phase_a_shared_s43` → `s.t3t1_shared_s43`: `add_temporal` True→False; `clip_len` 3→1; `clip_stride` 6→2
- `s.gt1` → `s.t3t1_phase_a_shared_s44`: `add_temporal` False→True; `clip_len` 4→3; `clip_stride` 8→6; `epochs` 30→15; `gt_ratio` 0.5→0.0; `seed` 20260612→44; `warmup_epochs` 5→3 (+9 keys only in child, 0 only in parent)
- `s.t3t1_phase_a_shared_s44` → `s.t3t1_shared_s44`: `add_temporal` True→False; `clip_len` 3→1; `clip_stride` 6→2
- `s.distill_s13` → `s.gt1_s13`: `batch_size` 8→4; `clip_len` 1→4; `clip_stride` 0→8; `lr` 0.001→0.0001; `seqs` 'MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP'→''; `use_pixel_shuffle` True→False (+21 keys only in child, 4 only in parent)
- `s.gt1_s13` → `s.gt2_plain_s13`: `gt_ratio` 0.5→0.0
- `s.gt1_s13` → `s.t3t1_phase_a_s13`: `add_temporal` False→True; `clip_len` 4→3; `clip_stride` 8→6; `epochs` 30→15; `gt_ratio` 0.5→0.0; `warmup_epochs` 5→3
- `s.t3t1_phase_a_s13` → `s.t3t1_s13`: `add_temporal` True→False; `clip_len` 3→1; `clip_stride` 6→2
- `s.distill_s14` → `s.gt1_s14`: `batch_size` 8→4; `clip_len` 1→4; `clip_stride` 0→8; `lr` 0.001→0.0001; `seqs` 'MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP'→''; `use_pixel_shuffle` True→False (+21 keys only in child, 4 only in parent)
- `s.gt1_s14` → `s.gt2_plain_s14`: `gt_ratio` 0.5→0.0
- `s.gt1_s14` → `s.t3t1_phase_a_s14`: `add_temporal` False→True; `clip_len` 4→3; `clip_stride` 8→6; `epochs` 30→15; `gt_ratio` 0.5→0.0; `warmup_epochs` 5→3
- `s.t3t1_phase_a_s14` → `s.t3t1_s14`: `add_temporal` True→False; `clip_len` 3→1; `clip_stride` 6→2

### Gated teachers

| role | base_yolo (derived) | epoch | provenance recorded | git commit | dirty | tensors |
|---|---|---:|---|---|---|---:|
| `s.legacy_teacher` | s.pretrained_yolo | 12 | no | — | — | 711 |
| `s.adapted_teacher` | s.pretrained_yolo | 12 | yes | d1d3198917b3 | dirty | 711 |

### Teacher caches

| role | status | manifest | teacher (manifest) | decode | frames |
|---|---|---|---|---|---:|
| `s.legacy_cache` | **unavailable** | — | — | — | — |
| `s.teacher_cache` | **unavailable** | — | — | — | — |

### Engines — attribution via sibling ONNX

| role | onnx | verdict | matches (hits / initializers) |
|---|---|---|---|
| `s.backbone_engine` | `models/yolo/yolo26s_backbone_640_best.onnx` | **unique_exact** → `s.legacy_teacher` | 146/146 for each best; 0 other checkpoints with fewer hits |
| `s.backbone_engine_v14replica_e12` | `models/yolo/yolo26s_backbone_640_v14replica_e12.onnx` | **unique_exact** → `s.adapted_teacher` | 146/146 for each best; 0 other checkpoints with fewer hits |

### Deployment forward — `s.deployment_preset` (`configs/presets/mamba_whole_graph.yaml`)

- checkpoint: `runs/mamba_gt_v14replica_t3_t1/best.ckpt` → node `s.t3t1_phase_b`; same-sha aliases: none
- temporal blocks: present; BYPASSED under whole-graph single-frame forward (effective T=1)
- final-stage `gt_ratio`: 0.0; runtime gate teacher: `null (backbone-only; no teacher forward)`
- backbone: TRT engine models/yolo/yolo26s_backbone_640_best.engine — attributed **unique_exact** to ['s.legacy_teacher']
- deployed backbone vs the teacher the head was trained against: **DIFFERENT** (engine ← `s.legacy_teacher`; head trained against ['s.adapted_teacher'])
- head engine: `none (PyTorch head inside whole graph)`; embedding: reid_mode='off'; graphs: {'use_whole_graph': True, 'use_cuda_graph': True, 'use_tracker_graph': True}

## Family `m` — backbone yolo26m

### Artifacts

| role | kind | path | status | sha256 | size | epoch | seed |
|---|---|---|---|---|---:|---:|---:|
| `m.pretrained_yolo` | yolo_pt | `models/yolo/yolo26m.pt` | present | `401cea9ab23a` | 44.3 MB | — | — |
| `m.adapted_teacher` | gated_teacher_ckpt | `runs/gated_det_yolo26m_v14replica/epoch_0012.ckpt` | present | `5f1cb461cb06` | 251.9 MB | 12 | 20260612 |
| `m.teacher_cache` | teacher_cache | `runs/mamba_teacher_cache_yolo26m_v14replica` | **unavailable** | — | — | — | — |
| `m.teacher_cache_gpu_decode` | teacher_cache | `runs/mamba_teacher_cache_yolo26m_v14replica_gpu_decode` | present | — | dir | — | — |
| `m.distill` | mamba_ckpt | `runs/mamba_distill_yolo26m_v14replica/best.ckpt` | present | `5bc14397102e` | 121.8 MB | 30 | 20260612 |
| `m.gt1` | mamba_ckpt | `runs/mamba_gt_yolo26m_v14replica_stage1/best.ckpt` | present | `fb8434556a2a` | 121.8 MB | 29 | 20260612 |
| `m.gt2_plain` | mamba_ckpt | `runs/mamba_gt_yolo26m_v14replica_final/best.ckpt` | present | `1960e28f8924` | 121.8 MB | 29 | 20260612 |
| `m.t3t1_phase_a` | mamba_ckpt | `runs/mamba_gt_yolo26m_v14replica_t3/best.ckpt` | present | `ca41aa449fe5` | 129.2 MB | 14 | 20260612 |
| `m.t3t1_phase_b` | mamba_ckpt | `runs/mamba_gt_yolo26m_v14replica_t3_t1/best.ckpt` | present | `a5a0e7091cd8` | 126.8 MB | 15 | 20260612 |
| `m.t3t1_phase_b_gpu_decode` | mamba_ckpt | `runs/mamba_gt_yolo26m_v14replica_t3_t1_gpu_decode/best.ckpt` | present | `477fe1ab41bd` | 126.8 MB | 15 | 20260612 |
| `m.deployment_preset` | preset | `configs/presets/mamba_whole_graph_m.yaml` | present | `496c4ec22b49` | 5.0 KB | — | — |
| `m.backbone_engine` | trt_engine | `models/yolo/yolo26m_backbone_640_best.engine` | present | `fdf0f4550f58` | 40.2 MB | — | — |
| `m.head_engine` | trt_engine | `models/yolo/mamba_head_26m.engine` | present | `c0c0e3e1a99e` | 22.5 MB | — | — |

### Mamba checkpoints — derived edges and checks

| role | init (derived) | teacher (derived) | cache (derived) | expected init | expected teacher | base_yolo sha | teacher sha | SSM interior frozen | tensors identical/changed/added | groups left bit-identical |
|---|---|---|---|---|---|---|---|---|---|---|
| `m.distill` | — | m.adapted_teacher | m.teacher_cache (missing) | — | ok | verified | verified | — | no_init_parent_in_inventory | — |
| `m.gt1` | m.distill | m.adapted_teacher | — | ok | ok | verified | verified | yes | 21/48/0 vs `m.distill` | mamba_blocks.ssm_internal |
| `m.gt2_plain` | m.gt1 | m.adapted_teacher | m.teacher_cache (missing) | ok | ok | verified | verified | yes | 21/48/0 vs `m.gt1` | mamba_blocks.ssm_internal |
| `m.t3t1_phase_a` | m.gt1 | m.adapted_teacher | m.teacher_cache (missing) | ok | ok | verified | verified | yes | 21/48/39 vs `m.gt1` | mamba_blocks.ssm_internal |
| `m.t3t1_phase_b` | m.t3t1_phase_a | m.adapted_teacher | m.teacher_cache (missing) | ok | ok | verified | verified | yes | 60/48/0 vs `m.t3t1_phase_a` | flow_gate_conv, mamba_blocks.ssm_internal, temporal_blocks.proj, temporal_blocks.ssm_internal |
| `m.t3t1_phase_b_gpu_decode` | m.t3t1_phase_b | m.adapted_teacher | m.teacher_cache_gpu_decode | — | ok | verified | verified | yes | 60/48/0 vs `m.t3t1_phase_b` | flow_gate_conv, mamba_blocks.ssm_internal, temporal_blocks.proj, temporal_blocks.ssm_internal |

### Mamba checkpoints — recorded training treatment

| role | epochs | lr | clip_len | clip_stride | gt_ratio | add_temporal | scan_stop_grad | seqs | holdout | temporal blocks | params |
|---|---:|---:|---:|---:|---:|---|---|---|---|---|---:|
| `m.distill` | 30 | 0.001 | 1 | 0 | — | — | True | 7-seq explicit | none | no | 10,175,788 |
| `m.gt1` | 30 | 0.0001 | 4 | 8 | 0.5 | False | True | all (default) | none | no | 10,175,788 |
| `m.gt2_plain` | 30 | 0.0001 | 4 | 8 | 0.0 | False | True | all (default) | none | no | 10,175,788 |
| `m.t3t1_phase_a` | 15 | 0.0001 | 3 | 6 | 0.0 | True | True | all (default) | none | yes | 11,417,692 |
| `m.t3t1_phase_b` | 15 | 0.0001 | 1 | 2 | 0.0 | False | True | all (default) | none | yes | 11,417,692 |
| `m.t3t1_phase_b_gpu_decode` | 15 | 0.0001 | 1 | 2 | 0.0 | False | True | all (default) | none | yes | 11,417,692 |

### Warm-start edges — what the recorded treatment changed

Keys both checkpoints recorded with different values. Flags that only the child records (added to the training script later) are counted, not interpreted.

- `m.distill` → `m.gt1`: `batch_size` 8→4; `clip_len` 1→4; `clip_stride` 0→8; `lr` 0.001→0.0001; `seqs` 'MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP,MOT17-10-SDP,MOT17-11-SDP,MOT17-13-SDP'→''; `use_pixel_shuffle` True→False (+28 keys only in child, 4 only in parent)
- `m.gt1` → `m.gt2_plain`: `gt_ratio` 0.5→0.0
- `m.gt1` → `m.t3t1_phase_a`: `add_temporal` False→True; `clip_len` 4→3; `clip_stride` 8→6; `epochs` 30→15; `gt_ratio` 0.5→0.0; `warmup_epochs` 5→3
- `m.t3t1_phase_a` → `m.t3t1_phase_b`: `add_temporal` True→False; `clip_len` 3→1; `clip_stride` 6→2
- `m.t3t1_phase_b` → `m.t3t1_phase_b_gpu_decode`: no recorded key differs (+8 keys only in child, 0 only in parent)

### Gated teachers

| role | base_yolo (derived) | epoch | provenance recorded | git commit | dirty | tensors |
|---|---|---:|---|---|---|---:|
| `m.adapted_teacher` | m.pretrained_yolo | 12 | yes | 06e539ce2955 | dirty | 771 |

### Teacher caches

| role | status | manifest | teacher (manifest) | decode | frames |
|---|---|---|---|---|---:|
| `m.teacher_cache` | **unavailable** | — | — | — | — |
| `m.teacher_cache_gpu_decode` | present | mamba-teacher-cache-v2 | m.adapted_teacher | torchvision_nvjpeg | 5316 |

### Engines — attribution via sibling ONNX

| role | onnx | verdict | matches (hits / initializers) |
|---|---|---|---|
| `m.backbone_engine` | `models/yolo/yolo26m_backbone_640.onnx` | **unique_exact** → `m.adapted_teacher` | 170/170 for each best; 0 other checkpoints with fewer hits |
| `m.head_engine` | `models/yolo/mamba_head_26m.onnx` | **ambiguous** → `m.distill`, `m.gt1`, `m.gt2_plain`, `m.t3t1_phase_a`, `m.t3t1_phase_b`, `m.t3t1_phase_b_gpu_decode` | 16/61 for each best; 25 other checkpoints with fewer hits |

### Deployment forward — `m.deployment_preset` (`configs/presets/mamba_whole_graph_m.yaml`)

- checkpoint: `runs/mamba_gt_yolo26m_v14replica_t3_t1/best.ckpt` → node `m.t3t1_phase_b`; same-sha aliases: none
- temporal blocks: present; BYPASSED under whole-graph single-frame forward (effective T=1)
- final-stage `gt_ratio`: 0.0; runtime gate teacher: `runs/gated_det_yolo26m_v14replica/epoch_0012.ckpt`
- backbone: TRT engine models/yolo/yolo26m_backbone_640_best.engine — attributed **unique_exact** to ['m.adapted_teacher']
- deployed backbone vs the teacher the head was trained against: **same** (engine ← `m.adapted_teacher`; head trained against ['m.adapted_teacher'])
- head engine: `models/yolo/mamba_head_26m.engine`; embedding: reid_mode='off'; graphs: {'use_whole_graph': True, 'use_cuda_graph': True, 'use_tracker_graph': True}

## Cross-reference — `report_data/tables/mamba_checkpoint_provenance.csv`

| role | experiment (csv) | sha256 |
|---|---|---|
| `s.legacy_v14` | `legacy_v14` | sha256_agrees |
| `s.gt2_plain` | `replica_20260612` | sha256_agrees |
| `s.t3t1_phase_b` | `t3t1_seed42` | sha256_agrees |
| `s.gt2_plain_s13` | `replica_20260613` | sha256_agrees |
| `s.t3t1_s13` | `t3t1_20260613` | sha256_agrees |
| `s.gt2_plain_s14` | `replica_20260614` | sha256_agrees |
| `s.t3t1_s14` | `t3t1_20260614` | sha256_agrees |

## Problems

None: every existing artifact was read as its declared kind.
