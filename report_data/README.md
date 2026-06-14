# Saccade Paper Data

This directory collects paper-facing evidence for the current research thread:
**a Mamba detection head trained with temporal constraints and deployed as a
single-frame, CUDA-graph inference pipeline for MOT**.

## Start here

1. [paper_direction.md](paper_direction.md): proposed thesis, contributions,
   experiment hierarchy, and claims that are currently defensible.
2. [algorithms.md](algorithms.md): paper method, equations, curriculum
   pseudocode, inference semantics, and attribution boundary.
3. [mamba_training_recipe.md](mamba_training_recipe.md): paper-ready training
   method extracted from the v14 replication protocol.
4. [evidence_and_limitations.md](evidence_and_limitations.md): provenance,
   leakage, seed pairing, and missing publication-grade experiments.
5. [tables/mamba_tracking_overall.csv](tables/mamba_tracking_overall.csv):
   recomputed MOT metrics and runtime.
6. [tables/mamba_t3t1_pairs.csv](tables/mamba_t3t1_pairs.csv): temporal
   curriculum comparisons, with valid pairing explicitly marked.
7. [tables/mamba_t3t1_per_sequence.csv](tables/mamba_t3t1_per_sequence.csv):
   sequence-level IDF1/MOTA deltas.
8. [tables/mamba_checkpoint_provenance.csv](tables/mamba_checkpoint_provenance.csv):
   checkpoint seed, schedule metadata, parameter count, and SHA-256.
9. [paper_tables.md](paper_tables.md): compact tables and captions that can be
   adapted directly into a manuscript.
10. [source_map.md](source_map.md): map from `docs/modules` research records to
   proposed manuscript sections and evidence roles.
11. [mamba_curriculum_progress.md](mamba_curriculum_progress.md): current
    T3-to-T1 result, mechanism evidence, boundary experiments, and next step.

The former full-pipeline formula collection is retained as
[pipeline_algorithms_reference.md](pipeline_algorithms_reference.md). It is a
tracker implementation reference, not the paper's proposed algorithm.

Runtime dataflow for the deployed production baseline is documented in
[mamba_whole_graph_dataflow.md](mamba_whole_graph_dataflow.md) (per-stage
latency, Nsight kernel verification) and
[mamba_whole_graph_pipeline_flow.md](mamba_whole_graph_pipeline_flow.md)
(three-layer CUDA-graph architecture). These describe the `mamba_whole_graph`
preset specifically; the generic `speed`/`baseline` versions live under
`docs/reference/pipeline_flow.md` and `docs/DATAFLOW.md`.

## Evidence classes

| Class | Meaning | Current material |
|---|---|---|
| A | Publication-grade held-out result | Not available yet |
| B | Controlled development ablation | T3-to-T1 paired seeds 20260613/20260614 |
| C | Historical replication evidence | legacy v14 and v14 replica |
| D | Negative-result / limitation evidence | ReID ceiling, trigger cost, lifecycle and detail-branch NO-GO |

All existing Mamba results are class B or C because the replicated teacher,
cache, and distillation stages used all seven MOT17-SDP training sequences.

## Rebuild

```bash
uv run python report_data/build_paper_assets.py
```

The script reads existing MOT result files and checkpoint metadata. It does not
run detector inference or training.

Generated outputs:

- `paper_metrics.json`
- `tables/mamba_tracking_overall.csv`
- `tables/mamba_t3t1_pairs.csv`
- `tables/mamba_t3t1_per_sequence.csv`
- `tables/mamba_curriculum_boundaries.csv`
- `tables/mamba_checkpoint_provenance.csv`
- `tables/bridge_ablation.csv`
- `figures/mamba_t3t1_paired_metrics.png`
- `figures/mamba_t3t1_per_sequence_idf1.png`

## Source of truth

- Training: `docs/modules/detection/mamba-v14-replication-protocol.md`
- Clean protocol requirements:
  `docs/modules/detection/mamba-v14r-training-protocol.md`
- Frozen-SSM audit: `report_data/mamba_v14_frozen_ssm_audit.md`
- Runtime optimization:
  `docs/modules/detection/research/mamba-cuda-graph-bug.md`,
  `whole-graph-kernel-fragmentation.md`, and `kernel-fusion-plan.md`
- Tracking metrics: recomputed from `results/*/MOT17-*-SDP.txt`
- HOTA/DetA/AssA: vendored TrackEval through
  `src/saccade/perception/eval/metrics.py`
