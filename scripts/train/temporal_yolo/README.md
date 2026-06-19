# temporal_yolo Training Scripts

This directory contains the training and reproduction scripts for the YOLO +
Mamba-head detector lineage. It is not a generic script dump: the active
reference for this area is
`docs/modules/detection/mamba-v14-replication-protocol.md`.

Review pass: 2026-06-18.

## Internal Support

| Path | Role |
|---|---|
| `README.md` | This protocol-oriented classification index |

## Stable Protocol Entrypoints

These paths are referenced by the v14 replication protocol or support the
current Mamba-head training lineage. Keep paths stable unless the protocol doc is
updated in the same change.

| Path | Role |
|---|---|
| `run_v14_full_e30_replication.sh` | One-command e30 teacher rebuild / v14 replica flow |
| `build_mamba_teacher_cache.sh` | Teacher feature cache builder used by the protocol |
| `train_gated_detector.py` | Gated teacher training |
| `train_mamba_head.py` | Distillation stage for Mamba head |
| `train_mamba_gt.py` | GT adaptation stages, including T3->T1 curriculum |
| `run_v14replica_t3t1.sh` | Main T3->T1 curriculum runner |
| `run_v14replica_t3t1_seed.sh` | Paired multi-seed T3->T1 validation |
| `run_v14replica_t3t1_shared_seed.sh` | Shared-seed T3->T1 validation variant |
| `run_v14replica_seed.sh` | Plain replica multi-seed baseline |
| `run_v14_parent_n16_frozen_refit.sh` | Controlled frozen-SSM N=16 refit |
| `run_v14replica_yolo26m.sh` | YOLO26m capacity contrast |

## Attribution And Follow-Up Runners

These scripts are experiment runners that preserve important attribution context.
They are not daily entrypoints, but they should not be deleted while the related
docs/results still cite the experiments.

| Path | Role |
|---|---|
| `run_v14_frozen_bn_teacher.sh` | Teacher-prior / frozen-BN control |
| `run_v14_frozen_yolo_teacher.sh` | Frozen-YOLO teacher control |
| `run_frozenT1_probe.sh` | Frozen-SSM / T=1 continued-training attribution |
| `run_t1joint_probe.sh` | Joint T-clip + T=1 loss probe |
| `run_v14replica_consistency.sh` | Temporal consistency follow-up |
| `run_v14replica_consistency_sweep.sh` | Consistency-weight sweep wrapper |
| `run_v14replica_implicit_seed.sh` | Implicit baseline seed runner |
| `run_v14replica_implicit_sweep.sh` | Implicit baseline multi-seed sweep |
| `run_v14replica_strip_oracle.sh` | Strip detail routing oracle probe |
| `run_v14replica_detail_b1h.sh` | B1-H detail route experiment |

## Training Code Under Review

These are reusable training loops or support scripts, but their current status is
less directly tied to the accepted main-line protocol.

| Path | Current status |
|---|---|
| `cache_trt_feats.py` | Support utility; keep until cache workflows are retired |
| `precompute_gmc.py` | Support utility; keep if GMC precompute remains useful |
| `train_mamba_trt.py` | Experimental TRT-feature training path |
| `train_mamba_cached.py` | Experimental cached-feature training path |
| `train_conditioned.py`, `configs/conditioned.yaml` | Older conditioned training path; archive-candidate |
| `train_joint.py`, `configs/joint.yaml` | Older joint training path; archive-candidate |
| `train_gated_tp.py` | Older gated-TP training path; archive-candidate |
| `train_jde_distill.py`, `train_jde_market.py`, `train_reid_1x1.py` | ReID/JDE experiments; archive-candidate for main-line MOT work |

## Cleanup Policy

- Keep protocol entrypoints stable.
- For attribution runners, prefer adding a header note over moving first.
- If a runner is moved later, update:
  - `docs/modules/detection/mamba-v14-replication-protocol.md`
  - `docs/modules/detection/research/mamba-t3t1-curriculum-20260613.md`
  - any report-data provenance table that cites the old path
- Archive candidates should move as a family, not one file at a time, so related
  configs and run scripts stay together.
