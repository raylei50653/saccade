# Saccade NO-GO Registry

> **Purpose**: decision index for directions that were rejected, parked, or revived.
> This file should answer: **what was this for, what signal was found, and where is
> the evidence?** Do not expand full experiment narratives here.
>
> Detailed historical notes are preserved in
> [no_go_registry_details.md](no_go_registry_details.md). Module-level writeups and
> ADRs remain the source of truth for full measurements.
>
> Last updated: 2026-07-04.

## How To Use This Registry

- Use this page before opening a new experiment branch. If the same **use case** and
  **signal** already failed, do not repeat it unless the revival condition changed.
- Keep each row short: one use case, one signal verdict, links to evidence.
- If a result has valuable details, put them in a module `research/` document or in
  [no_go_registry_details.md](no_go_registry_details.md), then link it here.
- A neutral result is not automatically dead. Mark whether the signal is absent,
  blocked by a mechanism, or valid but not actionable.

## Signal Verdicts

| Verdict | Meaning | Revival rule |
|---------|---------|--------------|
| `harmful` | Metrics regress when enabled. | Only retry if the operating point or upstream assumption changes. |
| `no signal` | The proposed signal is near-random or physically unavailable. | Do not retry without a new modality or feature source. |
| `blocked` | Signal exists, but the current mechanism cannot expose it. | Retry only after the blocker is removed. |
| `not actionable` | Signal is measurable, but already dominated by a better proxy or cannot separate gain from harm. | Retry only with a different action point. |
| `cost-bound` | Signal may exist, but expected gain is too small for implementation/runtime cost. | Retry only if cost drops or the target metric changes. |
| `revived` | Former NO-GO that became useful after mechanism redesign. | Keep the old failure mode documented. |

## Registry

| # | Item | Use Case | Signal Verdict | Evidence |
|---|------|----------|----------------|----------|
| <a id="1"></a>1 | Option D Track-Conditioned YOLO | Condition detection on tracker state. | harmful: tracker gate did not improve detection and collapsed IDF1. | [details](no_go_registry_details.md), [archive](../archive/option-d/) |
| <a id="2"></a>2 | Appearance ReID Bank | Online identity memory under GMC. | no signal / cost-bound: no IDF1 gain, large FPS hit. | [ReID module](../modules/reid/README.md), [details](no_go_registry_details.md) |
| <a id="3"></a>3 | Semantic Relink | Reconnect lost tracks with semantic/appearance cues. | blocked: age gate rejected most candidates; later bypassed by bidirectional bridge. | [semantic module](../modules/semantic/README.md), [details](no_go_registry_details.md) |
| <a id="4"></a>4 | Appearance capability ceiling | Establish whether MOT17 appearance embeddings can identify people. | no signal: multiple models and mechanisms hit the same weak identity ceiling. | [ReID module](../modules/reid/README.md), [details](no_go_registry_details.md) |
| <a id="5"></a>5 | Tiled Detection | Recover small/far people by 960p tiling. | harmful: truncation and score pollution doubled FP. | [legacy pipeline reference](PIPELINE_REFERENCE.md), [details](no_go_registry_details.md) |
| <a id="6"></a>6 | Motion-based Relinking | Reconnect tracks using motion extrapolation. | blocked: candidates were mostly removed before the signal could act. | [details](no_go_registry_details.md) |
| <a id="7"></a>7 | OA-SORT OAO | Penalize overlap ambiguity in association. | revived: spatial cost failed; duration-ramp exposed the usable temporal signal. | [revival analysis](../research/eval/oao_duration_ramp_revival_20260617.md), [details](no_go_registry_details.md) |
| <a id="8"></a>8 | NSA-Kalman | Adapt Kalman noise by detection score. | blocked / not actionable: score signal exists, but state/output tradeoffs hurt IDF1. | [neutral attribution](../research/eval/neutral_nogo_signal_attribution_20260612.md), [details](no_go_registry_details.md) |
| <a id="9"></a>9 | PostMerge | Merge fragmented tracklets after tracking. | blocked: good AUC, poor base rate and disabled appearance path made precision too low. | [neutral attribution](../research/eval/neutral_nogo_signal_attribution_20260612.md), [details](no_go_registry_details.md) |
| <a id="10"></a>10 | Per-frame Detection Cap / Adaptive Cap | Cut dense-frame detections before tracking. | harmful: dense MOT17 frames contain real people, not mostly FP. | [legacy pipeline reference](PIPELINE_REFERENCE.md), [details](no_go_registry_details.md) |
| <a id="11"></a>11 | P5-2 Stage2 QualityGate | Filter low-quality births. | no actionable gain: IDF1/MOTA stayed in noise. | [lifecycle module](../modules/lifecycle/README.md), [details](no_go_registry_details.md) |
| <a id="12"></a>12 | P5-3 ConsecutiveBirthGate | Require consecutive evidence before birth. | no actionable gain: statistically neutral. | [lifecycle module](../modules/lifecycle/README.md), [details](no_go_registry_details.md) |
| <a id="13"></a>13 | P5-4 Scene-Adaptive | Scene-adaptive lifecycle thresholds. | no actionable gain: historical negative/neutral result. | [lifecycle module](../modules/lifecycle/README.md), [details](no_go_registry_details.md) |
| <a id="14"></a>14 | P5-5 Proximity Birth Gate | Suppress births near existing tracks. | harmful: removed true positives and hurt recall. | [lifecycle module](../modules/lifecycle/README.md), [details](no_go_registry_details.md) |
| <a id="15"></a>15 | LaSt-ViT pre-hoc embedding quality | Improve ReID crops with LaSt-ViT quality. | no signal: untrained SigLIP2/LaSt-ViT features did not separate identities. | [LaSt-ViT analysis](../modules/reid/research/last_vit_integration_analysis.md), [details](no_go_registry_details.md) |
| <a id="16"></a>16 | ROI FPN ReID | Use detector FPN features as identity embeddings. | no signal: detector features encode person/geometry, not individual identity. | [details](no_go_registry_details.md) |
| <a id="17"></a>17 | Horizontal-flip TTA | Improve detection accuracy through TTA. | no actionable gain: changes stayed within noise. | [detection module](../modules/detection/README.md), [details](no_go_registry_details.md) |
| <a id="18"></a>18 | MOT20 mixed training | Improve MOT17 via additional crowded training data. | harmful: domain shift degraded metrics. | [detection module](../modules/detection/README.md), [details](no_go_registry_details.md) |
| <a id="19"></a>19 | Pose box expansion | Expand boxes using pose/geometry prior. | no signal: static FP could not be separated spatially. | [detection module](../modules/detection/README.md), [details](no_go_registry_details.md) |
| <a id="20"></a>20 | GMC FG Mask | Mask foreground during global motion compensation. | not actionable: PCR is dominated by background texture. | [geometry module](../modules/geometry/README.md), [details](no_go_registry_details.md) |
| <a id="21"></a>21 | Velocity-direction gate | Use motion direction to gate associations. | blocked: fast-object signal exists, but slow-object noise dominates globally. | [neutral attribution](../research/eval/neutral_nogo_signal_attribution_20260612.md), [details](no_go_registry_details.md) |
| <a id="22"></a>22 | Cheb-GR offline tracklet merge | Merge tracklets with graph re-ranking. | no actionable gain at the tested safe operating point. | [semantic module](../modules/semantic/README.md), [details](no_go_registry_details.md) |
| <a id="23"></a>23 | Birth-time lost-bank relink | Attach newborns to recently lost identities. | no signal for long gaps: appearance rank-1 is too weak. | [semantic module](../modules/semantic/README.md), [details](no_go_registry_details.md) |
| <a id="24"></a>24 | YOLO non-end2end output | Use cxcywh non-end2end YOLO output. | harmful: overall metrics regressed. | [detection module](../modules/detection/README.md), [details](no_go_registry_details.md) |
| <a id="25"></a>25 | Cascade Filter | Remove FP using CrowdHuman-to-MOT17 filtering rules. | not actionable: TP/FP scores overlap heavily. | [details](no_go_registry_details.md) |
| <a id="26"></a>26 | Pose Bio gate | Use biometric pose gate for relinking. | cost-bound: too few vetoes for large FPS cost. | [details](no_go_registry_details.md) |
| <a id="27"></a>27 | Narrow person score bonus | Boost narrow/far detections. | harmful: raised FP and reduced IDF1. | [details](no_go_registry_details.md) |
| <a id="28"></a>28 | Mamba temporal block | Add SSM temporal blocks to the detection head. | harmful: training/gradient path did not converge usefully. | [Mamba head training](../modules/detection/mamba-head-training.md), [details](no_go_registry_details.md) |
| <a id="29"></a>29 | Per-channel SSM A + MOT20 mix | Add per-channel temporal dynamics and mixed data. | harmful: DetA regressed under domain shift. | [Option F](../modules/detection/option-f-mamba-head.md), [details](no_go_registry_details.md) |
| <a id="30"></a>30 | Cheb-GR standalone | Replace classic k-reciprocal ranking. | not better than baseline fixed-k method. | [semantic module](../modules/semantic/README.md), [details](no_go_registry_details.md) |
| <a id="31"></a>31 | Relink bridge scale gate | Add scale gate to bridge relink. | no actionable signal on the speed direction; only selected bridge usage survived. | [semantic module](../modules/semantic/README.md), [details](no_go_registry_details.md) |
| <a id="32"></a>32 | Appearance relink gate | Gate relink with color histogram / OSNet appearance. | no signal: AUC near random, short-gap behavior reversed. | [offline relink analysis](../modules/semantic/research/offline_relink_candidate_analysis.md), [details](no_go_registry_details.md) |
| <a id="33"></a>33 | occ_cover live relink | Gate relink by occupancy along the gap path. | blocked: long-gap candidates are structurally removed by track buffer. | [semantic module](../modules/semantic/README.md), [details](no_go_registry_details.md) |
| <a id="34"></a>34 | GMC box-residual correction | Feed common residuals back into innovations. | not actionable: GT affine residual upper bound did not transfer to innovation space. | [GMC residual correction](../research/eval/gmc_residual_correction_20260612.md), [details](no_go_registry_details.md) |
| <a id="35"></a>35 | Mamba head features as relink embeddings | Reuse detection features for identity relink. | no signal: consistency improved boxes/scores, not identity discriminability. | [T3/T1 curriculum](../modules/detection/research/mamba-t3t1-curriculum-20260613.md), [details](no_go_registry_details.md) |
| <a id="36"></a>36 | Small-object high-resolution recovery | Recover tiny objects with dense/high-res routing. | cost-bound: expected IDF1 ceiling is below deployment cost. | [strip detail routing](../modules/detection/research/mamba-strip-detail-routing-design.md), [details](no_go_registry_details.md) |
| <a id="37"></a>37 | Explicit temporal consistency loss | Force adjacent-frame class logits to match. | harmful: hard consistency suppresses legitimate score dynamics. | [T3/T1 curriculum](../modules/detection/research/mamba-t3t1-curriculum-20260613.md), [details](no_go_registry_details.md) |
| <a id="38"></a>38 | Height-conditioned birth threshold | Raise/lower new-track threshold by box height. | no actionable signal: detection-layer precision gradient did not transfer to output. | [score distribution](../modules/detection/research/mamba-score-distribution-20260613.md), [details](no_go_registry_details.md) |
| <a id="39"></a>39 | Depth-ordering crossing-swap fix | Use front/back depth order to prevent crossing swaps. | blocked / overfit: real depth signal, but sequence-specific and identity-limited. | [depth ordering](../modules/semantic/research/depth_ordering_crossing_swap.md), [details](no_go_registry_details.md) |
| <a id="40"></a>40 | tile-PCR affine GMC | Estimate per-tile affine/similarity camera motion. | no actionable 2D signal: background parallax breaks a single image warp. | [geometry module](../modules/geometry/README.md), [details](no_go_registry_details.md) |
| <a id="41"></a>41 | Horizon / depth prior | Estimate horizon for GMC or motion normalization. | not actionable: horizon is measurable, but MOT17 lacks profitable pitch/roll DOF. | [details](no_go_registry_details.md) |
| <a id="42"></a>42 | Auction freshness bid | Prefer recently updated tracks during association. | harmful: fixes a switch subset but hurts correct stale-track reacquisition. | [math model](math_model.md), [details](no_go_registry_details.md) |
| <a id="43"></a>43 | Auction stability bid / Mahalanobis-as-cost | Use height stability or covariance as association tie-breaker. | not actionable: motion-layer tie-breakers cannot separate true crossing identities. | [math model](math_model.md), [details](no_go_registry_details.md) |
| <a id="44"></a>44 | Interpolation FP reduction | Reduce FP introduced by track interpolation. | blocked upstream: interpolation FP mostly comes from bad endpoints / wrong bridges. | [math model](math_model.md), [details](no_go_registry_details.md) |
| <a id="45"></a>45 | Higher fuse_score_weight | Add detection score into association cost. | not actionable: sequence-specific effects cannot be separated by score/ghost/motion probes. | [Mamba whole-graph analysis](../modules/detection/mamba_whole_graph_analysis.md), [details](no_go_registry_details.md) |
| <a id="46"></a>46 | Head activation occlusion signal | Train/use visibility signal from Mamba head activations. | not actionable: occlusion signal exists, but geometry proxies dominate downstream. | [details](no_go_registry_details.md) |
| <a id="47"></a>47 | FPN / raw backbone relink features | Use raw detector backbone features for relink identity. | no signal: revive attempts lowered IDF1 as more links were accepted. | [details](no_go_registry_details.md) |
| <a id="48"></a>48 | Occlusion-gated appearance relink | Use appearance only in clean/non-occluded relink windows. | not actionable: clean subset is small and already geometry-saturated. | [offline relink analysis](../modules/semantic/research/offline_relink_candidate_analysis.md), [details](no_go_registry_details.md) |
| <a id="49"></a>49 | occ-gated velocity damping | Damp Kalman velocity while coasting through occlusion. | harmful: drift exists, but damping creates FP and loses moving occluded tracks. | [project direction ADR](../decisions/018-project-main-line-direction.md), [details](no_go_registry_details.md) |
| <a id="50"></a>50 | NSA gating/output decoupling | Apply adaptive R only to gating and emit measurement boxes. | not actionable for IDF1: mechanism improves localization/HOTA but trades off IDF1. | [details](no_go_registry_details.md) |
| <a id="51"></a>51 | Predict-through-occlusion coast | Emit predicted boxes during occlusion gaps. | harmful for IDF1/MOTA: recall gain is dominated by FP from Kalman drift. | [details](no_go_registry_details.md) |
| <a id="52"></a>52 | PP22 full-cadence + GT-interp training | Fix PP22 temporal cadence for Mamba training. | no transfer signal: cadence shaping did not survive T=1 deployment. | [PP22 cadence plan](../research/training/pp22_full_cadence_interp_training_plan.md), [PP22 findings](../research/training/pp22_stress_test_findings.md) |
| <a id="53"></a>53 | Detection-head architecture recall knobs | Use DFL, spatial reduction, or deeper YOLO-like head for deploy recall. | cost-bound / marginal: @0.001 gains do not map to deploy threshold. | [PP22 findings](../research/training/pp22_stress_test_findings.md), [details](no_go_registry_details.md) |
| <a id="54"></a>54 | PP22 full training as MOT17 transfer source | Use PP22 as training source for MOT17 deployment. | harmful / domain-bound: PP22 is better as a stress test than a transfer source. | [PP22 findings](../research/training/pp22_stress_test_findings.md), [details](no_go_registry_details.md) |
| <a id="55"></a>55 | occ-exit identity audit | Cut IDs after occlusion using clean crop appearance audit. | blocked: GT-clean probe does not transfer to runtime geometry-clean gates. | [clean FIFO substrate](../modules/semantic/research/clean_fifo_bank_substrate_20260704.md), [details](no_go_registry_details.md) |
| <a id="56"></a>56 | Causal online Cheb-GR handover | Apply live newborn-to-dead identity claims in C++ streaming tracker. | harmful: live feedback compounds wrong claims even after port correctness fixes. | [semantic module](../modules/semantic/README.md), [details](no_go_registry_details.md) |
| <a id="57"></a>57 | Sync online ReID in tracker critical path | Run dynamic ReID synchronously in double-buffer tracking. | cost-bound: ~20% throughput loss for about +0.2 IDF1. | [semantic module](../modules/semantic/README.md), [details](no_go_registry_details.md) |
| <a id="58"></a>58 | Offline handover quality gates / crop filters / prototype banks | Use crop quality or prototype compression to improve offline handover. | mixed: quality gates no-go; clean-FIFO sparse bank is the usable signal. | [signal map](../modules/semantic/research/chebgr_handover_signal_map_20260704.md), [sparse bank](../modules/semantic/research/sparse_key_embedding_bank_20260704.md), [details](no_go_registry_details.md) |

## Reusable Lessons

| Lesson | Applies To |
|--------|------------|
| Appearance remains the main identity wall on MOT17. | #2, #4, #15, #16, #23, #32, #35, #47, #48, #55, #57 |
| A true probe signal can still be unusable if a better local proxy already dominates. | #34, #39, #41, #46 |
| Global gates fail when gain and harm are entangled across sequences. | #21, #38, #42, #43, #45, #49 |
| Long-gap recall is not safely solved by prediction alone. | #23, #33, #49, #51 |
| Use deploy operating points, not permissive probe thresholds, for GO/NO-GO. | #36, #38, #52, #53 |
| Do not average identity embeddings for Cheb-GR; preserve raw clean samples. | #58 |

## Current Stable GO Counterparts

| Module | Status | Reference |
|--------|--------|-----------|
| GPU GMC phase correlation | default ON | [math model](math_model.md) |
| Mamba whole-graph preset | production baseline | [MOT17 config](mot17_default_config.md), [frozen v2 ablation](benchmarks/frozen_v2_ablation.md) |
| GPUByteTracker + Sinkhorn-Auction | default ON | [ADR 015](../decisions/015-sinkhorn-auction-hybrid-association.md), [math model](math_model.md) |
| GPU bidirectional bridge relink | preset default ON | [semantic module](../modules/semantic/README.md), [math model](math_model.md) |
| OA-SORT OAO duration-ramp | preset default ON | [revival analysis](../research/eval/oao_duration_ramp_revival_20260617.md) |
| Clean-FIFO sparse embedding bank | offline/async substrate candidate | [sparse bank](../modules/semantic/research/sparse_key_embedding_bank_20260704.md) |
