# Paper Direction: Temporally Shaped Mamba Detection for Real-Time MOT

## Recommended thesis

> Temporal structure can be used as a training-time regularizer for a Mamba
> detection head, improving downstream association stability while retaining a
> single-frame deployment graph and near-identical detector recall.

This is stronger and better supported than claiming a generic new Mamba
detector. The observed gain is primarily association quality rather than raw
recall:

- plain replica: IDF1 73.0-73.4, HOTA 65.3-65.5;
- T3-to-T1: IDF1 73.4-75.4, HOTA 65.7-67.7;
- best run: IDF1 75.4, MOTA 77.6, HOTA 67.7, AssA 66.0;
- deployment remains single-frame because temporal blocks are bypassed in the
  whole-graph evaluator;
- all variants remain around 215-220 FPS on the recorded RTX 5070 Ti Laptop
  runs.

These are development-set results, not held-out benchmark claims.

## Candidate contributions

### 1. Small-data frozen-SSM detection head

The 10.13M-parameter single-frame head uses frozen internal SSM dynamics and
trains the projection, gate/readout, upsampling, and prediction heads. The
historical gradient bug was converted into an explicit `--scan-stop-grad`
regime. Temporal training checkpoints contain 11.37M parameters because they
also retain training-only temporal blocks; those blocks are bypassed in the
deployment evaluator.

Defensible result:

- the legacy behavior is reproducible across a reconstructed training chain;
- three plain student-chain seeds span only about 0.4 IDF1 points;
- unfreezing from a converged checkpoint improves MOTA/HOTA but not recall,
  showing that lower training loss is not a sufficient deployment selector.

### 2. T3-to-T1 temporal shaping

GT2 is split into:

1. 15 epochs with temporal processing, `T=3`, stride 6;
2. 15 epochs of single-frame readaptation, `T=1`, stride 2.

Inference uses only the spatial path. The strongest interpretation is that
training-time temporal mixing regularizes box/score consistency, which then
improves IoU-based association.

Evidence:

- valid paired seed 20260613: approximately +2.2 IDF1;
- valid paired seed 20260614: approximately +0.4 IDF1;
- the original seed-42 T3-to-T1 run is an independent positive replication,
  but not a strict pair with the seed-20260612 plain checkpoint;
- recall remains stable while AssA, IDs, and FP improve in the strongest run;
- using the feature as a ReID embedding gives hard-pool AUC 0.438, so the gain
  is consistency rather than identity discriminability.

Mechanism and boundary evidence:

- Phase-A streaming T=3 inference adds only about +0.7 IDF1 over its T=1
  evaluation, remains far below the final T3-to-T1 checkpoint, and more than
  doubles latency;
- full-gradient SSM fine-tuning after T3-to-T1 raises DetA/MOTA but erases the
  AssA gain;
- reversing the order only partially restores association;
- weight interpolation has no synergy peak, indicating a narrow and fragile
  consistency solution rather than a freely mergeable weight direction.

The original `replica_20260612` versus `t3t1_seed42` comparison is **not
paired** because the T3-to-T1 script omitted `--seed` and used default seed 42.

### 3. Single-frame high-throughput deployment

The deployment path captures resize, TensorRT backbone, Mamba head, decode,
NMS, GMC, and tracking in CUDA graphs.

Measured engineering results from module research:

- custom selective scan stream fix restored graph correctness;
- head CUDA graph produced about +15% FPS over eager;
- pointwise compilation was bit-exact and added about +4.7% FPS;
- current result artifacts record roughly 215-223 FPS at 4.5-4.7 ms/frame.
- valid paired T3-to-T1 artifacts record post-decode-to-output P99 latency of
  5.19 and 5.25 ms on MOT17-13-SDP.

This should be presented as systems co-design supporting the method, not as an
independent algorithmic novelty unless the paper targets a systems venue.

The production timer starts after `next(stream_iter)` has returned the decoded
GPU frame and ends after tracking output generation. It therefore excludes
decode/fetch wait and should be labeled **post-decode-to-output latency**, not
camera-to-result or full video end-to-end latency.

## Supporting rather than central contributions

- GPU GMC is an important fixed baseline component.
- Speed-weighted bidirectional bridge relink improves IDF1 73.0 to 75.1 and
  AssA 63.8 to 66.6 on the recorded development evaluation.
- ReID, dynamic trigger, lifecycle, and dense-detail studies are useful
  limitation and negative-result sections.

Do not combine all modules into one claimed method. That would obscure the
Mamba training contribution and make attribution weak.

## Proposed paper structure

1. Introduction: small-data MOT detection and deployment constraints.
2. Related work: state-space vision heads, temporal training, MOT association,
   and CUDA-graph inference.
3. Method:
   frozen-SSM Mamba head, staged teacher-to-GT curriculum, T3-to-T1 shaping.
4. Deployment:
   whole-detect graph and fused pointwise execution.
5. Experiments:
   clean split, architecture controls, temporal curriculum, SSM gradient
   regime, runtime, and downstream association.
6. Analysis:
   recall versus AssA, per-sequence behavior, feature consistency versus
   discriminability.
7. Limitations:
   current replication leakage, two valid paired seeds, and MOT17 appearance
   ceiling.

## Minimum experiment matrix before submission

| Question | Required comparison |
|---|---|
| Does Mamba help? | same backbone and budget: CNN/MLP head vs Mamba head |
| Does frozen SSM help? | full-gradient vs stop-gradient from the same distill checkpoint |
| Does temporal shaping help? | plain GT2 vs T3-only vs T3-to-T1, at least 3 valid paired seeds |
| Is the gain association-driven? | recall, DetA, AssA, IDs, FP/FN, box jitter |
| Does it generalize? | strict-clean MOT17 split plus DanceTrack/SportsMOT or official test |
| Is deployment efficient? | eager, head graph, whole graph, compile/fusion; latency distribution |
