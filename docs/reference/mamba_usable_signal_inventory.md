# Mamba usable signal inventory

_Last updated: 2026-06-30. Purpose: separate usable signals from rejected or
weak signals before combining mechanisms._

## Current Readout

The useful signal is not another standalone FPN reduction. The useful signal is
in **suppressed/private detector candidates plus previous-track geometry**.

Implemented first combination:

```text
candidate_nms=0.60 private pool
  + no private births
  + max 1 private candidate per frame
  + low-score association stage only
```

MOT17-SDP 7-seq interpolation-on result is a small positive tradeoff, not a
strong standalone solution: IDF1 76.4 -> 76.5, MOTA 78.6 -> 78.8, IDs 467 ->
459, FN 20613 -> 20127, FP 2952 -> 3180. A per-track geometry-only selector was
tried next; it slightly helped MOT17-02 no-interp IDs but raised FP under
interpolation. A first suppressor/public-ownership heuristic also regressed
MOT17-02 no-interp IDF1. A cheap sparse symmetric detection-field probe improved
FN only marginally while increasing IDs/FP. The next missing piece is not a
broader private pool or raw pre-tracker geometry: it is direct tracker
association cost or a learned motion-conditioned score for deciding which
private candidate is a real continuation.

## Usable Signals

| signal | evidence | use | guardrail |
|---|---|---|---|
| Constrained private candidate pool | MOT17-02, score 0.25, `candidate_nms=0.70`: private boxes 10,932; unique recoverable GT 313/604; score AUC to recoverable subset 0.770; P@100 by score 59% | Source candidates for NMS-suppressed/crowded misses | Private boxes must not be emitted globally and must not start new IDs |
| Previous-box / motion geometry | MOT17-02, score 0.25, `candidate_nms=0.70`: active baseline FN 203; unique active FN covered 161/203; nearest-motion covered 147/203; best-last-IoU AUC 0.843 | Rank/filter private candidates for active/lost track continuation | Applies only to existing tracks; does not solve births or never-detected objects |
| Detector score within constrained private pool | `candidate_nms=0.60/0.70`: recoverable AUC 0.817/0.770; P@100 50%/59% | First-stage candidate ranking and score guard | Weakens when the pool is too broad; not enough alone at NMS 0.90 |
| Eval-only private continuation prototype | 7-seq interp-on: IDF1 +0.1, MOTA +0.2, IDs -8, FN -486, FP +228 with `candidate_nms=0.60`, min 0.10, top1, low-stage-only | Production sanity check that suppressed evidence survives tracker competition | Not a default yet; per-seq mixed and FP rises |
| Per-track geometry selector | MOT17-02 no-interp improves IDs 123 -> 120 and FN 8238 -> 8230 vs global top1, but interp-on regresses to IDF1 57.2, FP 269 vs global top1 IDF1 57.3, FP 244 | Diagnostic only; proves one-private-per-track geometry is insufficient | Do not promote without suppressor/public ownership or tracker association cost |
| Sparse symmetric local support | MOT17-02 no-interp sparse-symmetric top1 gives FN 8234 vs global top1 8238, but IDs 135 vs 123; interp-on FP 300 vs global top1 244 | Possible feature for a later learned/tracker-cost gate | No-go as standalone selector |
| Raw NMS 0.55/0.60 sweep | MOT17-02 tracking probes show direct recall/IDF1 gain, but FP/IDs rise | Upper-bound proof and fallback operating point | Raw global NMS is a probe, not the final design |
| Assigner conflict diagnostics | 4-8 px zero-positive before conflict 0.133%, lost-all after conflict 4.161% | Justifies conflict-focused handling, and later P2 only if private continuation leaves this bucket | Does not justify broad assigner swap or broad P2 first |

## Not Primary Signals

| signal | status | reason |
|---|---|---|
| ReID / PTDS appearance branch | not primary | Existing ReID evidence marks MOT17 appearance weak operationally. Quick sanity check: SigLIP2 drops to rank-1 29.6% at gap 121+; TransReID is 38.4% overall and 8.9% at gap 121+. |
| Candidate NMS 0.90 private pool | upper-bound only | Covers more GT in theory, but recoverable subset ranking is weak: score AUC 0.609 and P@100 11%. |
| Standalone reduction replacements | no-go | `space-to-depth`, `blur-conv`, and `wavelet` do not produce a useful recall gain as standalone head replacements. |
| Direct NGLA/NBCD assigner swap | no-go | Removes tiny zero-pre cases but increases conflict anchors and does not improve post-conflict positives. |
| Pre-tracker suppressor ownership heuristic | no-go as standalone | MOT17-02 no-interp `suppressor_aware` top1 low-stage gives IDF1 53.5 vs baseline 54.7 and global top1 54.3; it adds too few useful boxes and does not reduce IDs. |
| Detection-field sparse symmetric selector | no-go as standalone | It is recall-biased but not identity-safe: no-interp FN -4 vs global top1 but IDs +12; interp-on FP rises to 300. |
| Full-resolution / global higher-resolution scan | closed/reserve | Full scan is ~12x head-only; P3/global sr=2 is ~3x head-only. Use only if sparse mechanisms fail and evidence requires Mamba-side recovery. |

## Missing Signal Gates

| gate | question | required before |
|---|---|---|
| Tracker association-cost gate | Can we rescue a private box only when the tracker cost for that track prefers the private candidate over the suppressing public candidate? | Calling A2 solved |
| Cross-sequence private/motion signal | Do private-pool ranking signals keep the same direction outside MOT17-02? | Defaulting the private-continuation prototype |
| Actual tracker private-continuation A/B | Current prototype is small positive interp-on but mixed per-seq; per-track geometry-only does not beat global top1 under interpolation | Calling A2 solved |
| Non-active FN taxonomy | Of the score-0.25 baseline FN not covered by active motion, how many are low-score continuation, birth-confirmation, localization, or absent/far? | Choosing P2 vs ROI vs low-score birth policy |
| Trainable CenterTrack-lite head signal | Does previous-track heatmap or motion-conditioned head output beat hand-coded previous-box geometry? | Adding a trained track-conditioned head branch |
| ReID private-pool separability | Can appearance separate same-track private candidates from wrong private boxes after motion gating? | Reintroducing PTDS-style ReID/dense similarity |

## Combination Order

1. Keep **private continuation only** with no private births.
2. Keep global top1 low-stage as the current reference selector.
3. Add a tracker association-cost selector; use per-track geometry and
   suppressor relation/sparse symmetric support only as features, not the
   decision by themselves.
4. Keep private boxes in the low-score association stage unless a stronger gate proves safe.
5. Evaluate no-interp tracking first, then production interpolation-on.
6. If active FN improves but non-active FN remains, split residuals into low-score continuation, birth, localization, and absent/far buckets.
7. Add P2 or high-res ROI only if the residual bucket demands new spatial evidence.
8. Add ReID or dense similarity only if a private-pool separability gate proves it adds signal beyond motion.

## Evidence Artifacts

- `report_data/private_candidate_separability_mot17_02_nms060_full.json`
- `report_data/private_candidate_separability_mot17_02_nms070_full.json`
- `report_data/private_candidate_separability_mot17_02_full.json`
- `report_data/motion_private_candidate_separability_mot17_02_nms060_full.json`
- `report_data/motion_private_candidate_separability_mot17_02_nms070_full.json`
- `report_data/reduction_candidates_full30/mot17_private060_min010_top1_lowstage_interp_7seq/`
- `report_data/reduction_candidates_full30/mot17_baseline_interp_7seq_current/`
- `report_data/reduction_candidates_full30/mot17_private060_min010_pertrack_c2_top1_lowstage_nointerp/`
- `report_data/reduction_candidates_full30/mot17_private060_min010_pertrack_top1_lowstage_interp/`
- `report_data/reduction_candidates_full30/mot17_private060_min010_suppressor_claimed_top1_lowstage_nointerp/`
- `report_data/reduction_candidates_full30/mot17_private060_min010_sparse_sym_top1_lowstage_nointerp/`
- `report_data/reduction_candidates_full30/mot17_private060_min010_sparse_sym_top1_lowstage_interp/`
- `docs/reference/mamba_recall_candidate_experiments.md`
- `scripts/eval/detector/probe_private_candidate_separability.py`
- `scripts/eval/detector/probe_motion_private_candidate_separability.py`
