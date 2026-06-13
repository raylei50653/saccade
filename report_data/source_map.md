# Source Map

This index separates the central Mamba evidence from supporting tracking work
and negative results.

## Central method

| Source | Manuscript use |
|---|---|
| `docs/modules/detection/mamba-v14-replication-protocol.md` | Exact historical reconstruction, training chain, commands, and artifact lineage |
| `docs/modules/detection/mamba-v14r-training-protocol.md` | Strict-clean protocol and publication requirements |
| `docs/modules/detection/mamba-head-training.md` | General Mamba head training design |
| `docs/modules/detection/option-f-mamba-head.md` | Architecture motivation and original design choices |
| `docs/modules/detection/research/mamba-t3t1-curriculum-20260613.md` | T3-to-T1 discovery, mechanism probes, curriculum-order boundary, and future directions |
| `report_data/mamba_v14_frozen_ssm_audit.md` | Frozen-SSM interpretation and trainable parameter audit |
| `report_data/mamba_v14_training_audit.md` | Training-stage and checkpoint audit |
| `report_data/gated_det_v1_training_audit.md` | Teacher lineage and adaptation evidence |

## Deployment

| Source | Manuscript use |
|---|---|
| `docs/modules/detection/research/mamba-cuda-graph-bug.md` | Correctness issue and stream-binding fix |
| `docs/modules/detection/research/whole-graph-kernel-fragmentation.md` | Runtime bottleneck analysis |
| `docs/modules/detection/research/kernel-fusion-plan.md` | Pointwise fusion plan and measured speedup |
| `docs/modules/detection/mamba_whole_graph_analysis.md` | Whole-pipeline graph architecture |
| `report_data/mamba_whole_graph_analysis.md` | Paper-facing runtime summary |

## Supporting tracking ablations

| Source | Manuscript use |
|---|---|
| `docs/modules/semantic/research/bidirectional_relink_roadmap.md` | Bridge-relink method definition |
| `docs/modules/semantic/research/bidir_relink_data_analysis.md` | Bridge candidate and error analysis |
| `docs/modules/semantic/research/offline_relink_candidate_analysis.md` | Offline relink candidate study |
| `docs/modules/semantic/research/relink_normalization_gate_analysis.md` | Scale/normalization gate ablation |
| `docs/modules/geometry/research/fp_fn_recovery_and_gmc.md` | GMC and FP/FN recovery analysis |
| `report_data/relink_gates_and_formulas.md` | Compact formulas and gate definitions |

Use these as fixed tracker components or supporting ablations. Do not fold
their gains into the Mamba curriculum claim without a factorial comparison.

## Limitations and negative results

| Source | Manuscript use |
|---|---|
| `docs/modules/reid/research/semantic_relink_and_crop.md` | Appearance and semantic relink analysis |
| `docs/modules/reid/research/last_vit_integration_analysis.md` | ReID integration limitations |
| `docs/modules/trigger/research/dynamic_trigger.md` | Dynamic-trigger cost/benefit result |
| `docs/modules/lifecycle/research/tentative_confirmed_state.md` | Lifecycle-state ablation |
| `docs/modules/detection/research/mamba-dual-resolution-original-detail-plan.md` | Dense-detail branch proposal and outcome context |
| `report_data/no_go_summary.md` | Consolidated NO-GO decisions |

These records are useful for a limitations section because they show which
plausible alternatives did not explain or improve the main result.
