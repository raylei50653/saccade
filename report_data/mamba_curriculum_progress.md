# Mamba T3-to-T1 Curriculum: Current Progress

Primary research record:
`docs/modules/detection/research/mamba-t3t1-curriculum-20260613.md`.

## Current conclusion

The project now supports a stronger contribution than simply replacing a
detection head with Mamba:

> Temporal Mamba blocks act as a training-time shaping constraint. A following
> T1 readaptation stage transfers the benefit into a stateless single-frame
> detector, improving downstream association without temporal inference.

This is the current paper main line.

## Evidence status

### Main result

- Two valid same-seed curriculum pairs are available.
- Seed 20260613 improves IDF1 by +2.27 and HOTA by +2.04.
- Seed 20260614 improves IDF1 by +0.42 and HOTA by +0.26.
- The seed-42 T3-to-T1 run reaches IDF1 75.45 and HOTA 67.71, but its plain
  comparison checkpoint uses seed 20260612. It is positive replication
  evidence, not a strict paired ablation.
- Difficult sequences 02, 05, 10, and 13 improve in both valid paired seeds.

### Mechanism closure

The Phase-A checkpoint was evaluated with and without streaming temporal
inference:

| Evaluation | IDF1 | MOTA | FPS | Interpretation |
|---|---:|---:|---:|---|
| T1 bypass | 69.1 | 72.0 | 103.6 | Phase A without temporal inference |
| T3 streaming | 69.8 | 71.6 | 46.6 | Only +0.7 IDF1 with large runtime and FP cost |
| Final T3-to-T1 | 75.4 | 77.6 | 217.3 | Benefit is realized after T1 readaptation |

The useful effect is therefore training-time shaping, not deployment-time
temporal recurrence.

### Association pathway

- Strongest run: AssA +4.06 points and 109 fewer ID switches relative to the
  unpaired plain reference.
- Detection recall remains approximately stable.
- Mamba features used directly as ReID embeddings have hard-pool AUC 0.438,
  with about 0.001 difference from plain features.

The supported explanation is improved box/score consistency transmitted
through IoU association, not learned identity discrimination.

### Curriculum boundary

| Ordering | IDF1 | MOTA | HOTA | DetA | AssA | Result |
|---|---:|---:|---:|---:|---:|---|
| T3-to-T1 | 75.4 | 77.6 | 67.7 | 69.7 | 66.0 | Best association point |
| T3-to-T1 then SSM-ft | 73.8 | 79.4 | 66.7 | 71.0 | Detection improves; shaping erased |
| SSM-ft then T3-to-T1 | 74.3 | 78.8 | 66.9 | 70.7 | Association only partly restored |

This establishes a DetA-AssA tradeoff along SSM freedom. The final objective in
the curriculum determines which feature structure is retained.

### Weight-space boundary

Linear interpolation between SSM-ft and T3-to-T1 checkpoints produces no
synergy peak. The T3-to-T1 association solution is concentrated near its
endpoint and cannot be recovered by a simple model soup.

Exact recomputed values are in
`tables/mamba_curriculum_boundaries.csv`.

## Remaining work

1. Resolve the strict-clean training lineage and held-out evaluation.
2. Produce a true third same-seed plain/T3-to-T1 pair.
3. Compare against equal-budget CNN, MLP, temporal convolution, and attention
   controls.
4. Test joint full-gradient plus T1 loss and an explicit GMC-warped temporal
   consistency loss.
5. Validate on a second dataset.
6. Pool latency samples across all sequences for final P95/P99 reporting.

## Paper positioning

The defensible novelty is the combination of:

1. frozen-SSM training for small MOT data;
2. temporal Mamba shaping followed by T1 readaptation;
3. stateless single-frame deployment;
4. mechanism evidence separating consistency from identity discrimination;
5. curriculum-order experiments exposing the DetA-AssA tradeoff.

Do not describe the contribution only as a "Mamba detection head." The
curriculum, deployment semantics, and boundary evidence are what make the work
methodologically distinctive.
