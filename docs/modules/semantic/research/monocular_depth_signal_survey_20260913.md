# Monocular depth (YOLO26-depth) signal survey (2026-09-13)

> Part of the relink / crossing-swap / AssA investigation; hub =
> [`offline_relink_candidate_analysis.md`](offline_relink_candidate_analysis.md) §0.
> Do not confuse with [`depth_ordering_crossing_swap.md`](depth_ordering_crossing_swap.md)
> (that probe's "depth" is a **cheap pixel-geometry proxy** — `foot_y`/`area` — for
> **front/back ordering** at crossing-swaps, already GO and in production as
> `occ_state_*`). This note asks a different question: does a **neural monocular
> depth model** (Ultralytics YOLO26-depth) add anything on top of what this
> project's own pixel-geometry features already give the relink cost gate, and
> is there a better application elsewhere in the stack.

**Verdicts:**
- **Relink/assignment cost gate — NO-GO for the tested designs.** Per-box neural
  depth (single-frame or trend-fit) is far weaker than the existing pixel-geometry
  features it would compete with, and naive fusion measurably *hurts* both tested
  baselines. Not attributed to a fixed cause; see §4.
- **Direct depth concatenation into the ReID/appearance embedding —
  architecturally rejected, not tested end-to-end.** Concatenating depth into
  the normalized identity feature vector contradicts what that embedding is
  trained to represent; see §5. Other multimodal interactions remain untested.
- **Static per-camera neural calibration for relink — NO-GO for the tested
  design.** The depth surfaces are stable, but for the exploratory top-two rule
  at margins 0.10/0.20/0.30, the cheap position-to-scale surface duplicates
  every candidate decision in the activated groups; see §6. Other non-relink
  uses of a calibrated scene map remain untested.

Full scripts, CSVs, MANIFEST (env/provenance) and heatmap renders:
[`results/analysis/depth_signal_probe_20260913/`](../../../../results/analysis/depth_signal_probe_20260913/)
(local disk, `results/` is gitignored — the doc below is the citable record).

## 0. What YOLO26-depth is

Ultralytics YOLO26 ships a monocular depth task (`yolo26{n,s,m,l,x}-depth`) as of
2026-01: a separate model (own backbone + head, not fused with detection),
open-ended log-depth output in metres, trained on a ~2.19M-image indoor+outdoor
mix (NYU Depth V2 among others). This project's detector is a heavily modified
YOLO26 backbone (layers 0-22) feeding a custom `MambaDetectionHead`
(`mamba_gated_detector.py`) inside a captured CUDA graph — the stock depth head
is **not** available on that path; every measurement here used the stock
`yolo26n-depth.pt` checkpoint in an isolated venv (`ultralytics==8.4.150`,
`torch==2.14.0`), never the project's pinned `.venv` or its CUDA graph.

A basic monotonic sanity check passed before any signal claim: on
MOT17-02-SDP f100, five GT boxes' patch-median depth ordered **exactly** by box
height (356px→7.07m … 110px→17.54m). This single-frame, five-box check only
supported proceeding with the broader probe; it did not establish domain
transfer by itself.

## 1. Method

Reused the existing GT-labelled relink funnel
([`results/analysis/reid_repair_funnel_20260905`](../../../../results/analysis/reid_repair_funnel_20260905/),
`oracle_cands.csv`: 13,953 enumerated lost→candidate pairs, 161 GT-true breaks
across 119 `(seq, lost_id)` groups) rather than a fresh eval — this is exactly
the failure population the funnel already diagnosed as **pairing-separability
limited** (not coverage, not model-capacity — see
[`offline_relink_candidate_analysis.md`](offline_relink_candidate_analysis.md)
and the funnel's own `results/analysis/reid_repair_funnel_20260905/README.md`).

Metric convention matches the funnel's own established one (per-event margin,
rank-1 fraction) rather than inventing a new one: for every group with a known
true continuation, rank all candidates by a signal and ask whether the true one
ranks first. Chance level = mean(1/candidates-in-group).

## 2. Single-frame depth: weak, and much weaker than existing geometry

Per-box median depth (torso-centred patch, side = 0.3×h, scaled by the funnel's
own height estimate), sampled at `lost_last_frame` / `cand_first_frame`.

| signal (rank-1, n=88 groups) | rate | chance |
|---|---|---|
| depth margin `\|depth_lost − depth_cand\|` | 10.2% | 4.9% |
| existing `dist_h` (pixel) | 39.8% | same |
| naive rank-sum fusion (dist_h + depth) | 28.4% (**worse than dist_h alone**) | same |

Restricted to the subset where `dist_h` cannot already separate (a wrong
candidate within 1.5× the true pair's `dist_h`, n=65): depth margin still beats
chance (35.4% vs 21.9%), while `dist_h` itself — predictably — drops below its
own chance level there (18.5%). **Depth's usable signal, where it exists, is
conditional (a tie-breaker inside the geometry-ambiguous band), not additive
across the whole population.**

## 3. Visual inspection: low spatial resolution + one demonstrated (non-general) failure mode

Heatmap renders
([`renders/`](../../../../results/analysis/depth_signal_probe_20260913/renders/))
show the depth maps are **low spatial resolution** — smooth scene-level
gradients with soft blobs around near objects, not sharp per-instance
boundaries. Closely-spaced pedestrians blur together; much of what the per-box
median reads may be a smoothed proxy for image-row position rather than new
information.

One deep failure case (`failure_id17`, true_rank 30/49) visually shows the
sampling patch landing on a closer *occluder* instead of the (partially hidden)
target at both ends, producing a spurious near-perfect depth "match" against an
unrelated, fully-visible wrong candidate. **This mechanism is real but does not
generalize**: joining all 161 true rows against MOT17's own GT visibility gives
`r(visibility, depth_margin) = +0.41` — margin goes *up* with visibility, the
opposite of what occlusion-contamination-as-the-driver would predict in
aggregate (low-vis median margin 3.0m vs high-vis 7.1m, n=140/21; gap length is
not a confound, r=0.11). **What drives most of the large-margin tail is
unresolved — do not attribute it to occlusion contamination.**

## 4. Trend-fit depth (mirrors production `fwd_resid`/`bwd_resid`/`bridge_dist`): does not rescue it

Built the depth-space analogue of `scripts/tools/build_relink_candidates.py`'s
own `_velocity`/`anchor=4` trend fit (mean per-frame depth velocity over the
last/first 4 **GT** frames, extrapolated across `gap`) — `depth_fwd_resid`,
`depth_bwd_resid`, `depth_bridge_dist`.

| signal (rank-1, n=87) | rate |
|---|---|
| raw single-frame depth diff | 12.6% |
| `depth_bridge_dist` (trend-fit) | 13.8% |
| existing `bridge_dist` (pixel) | **54.0%** |
| existing `dist_h` (pixel) | 40.2% |

Averaging depth over more frames does not clean up whatever limits it (plausibly
the resolution ceiling in §3, which temporal averaging cannot fix). Note in
passing: `bridge_dist` is a stronger baseline than `dist_h` on this subset
(54.0% vs 40.2%) — not previously compared this way in the funnel notes.

In `bridge_dist`'s own blind spot (competitor within 1.5× true `bridge_dist`,
n=57): `depth_bridge_dist` still beats chance (38.6% vs 27.6%) while
`bridge_dist` drops to 29.8% there — same conditional pattern as §2, smaller in
absolute terms because `bridge_dist` leaves a smaller gap to fill.

**Naive rank-sum fusion of `bridge_dist` + `depth_bridge_dist`: 33.3%, worse
than `bridge_dist` alone (54.0%).** Same failure mode as §2, now reproduced
against the stronger of the two existing baselines. Two independent naive-fusion
attempts, two measured regressions — treat additive fusion of this signal as a
closed door pending a non-additive design (e.g. gated only inside the
`dist_h`/`bridge_dist` ambiguous band), not attempt a third fusion this way.

## 5. Why direct depth concatenation does not belong in the ReID/appearance vector

Considered and rejected without a production run: `cheb_gr.py` states all
input features are assumed L2-normalized, and the whole re-ranking machinery
(per-node μ/σ, Chebyshev threshold `T = μ − λσ`, graph-conv propagation) is
calibrated on the statistics of that normalized appearance-distance
distribution — no trained weights to absorb a mis-scaled extra dimension.

More fundamentally: an identity embedding's job is to be invariant to exactly
the things depth is not. Among the 88 groups with usable depth, the **true**
(GT-correct) match's own depth margin has **median 2.68m** (58% > 2m, 36% > 5m)
over a median 59-frame (~2s) gap — normal motion, not noise. Baking depth into
the "identity" vector would push the same real person further apart in feature
space purely for having walked, directly opposing what relink is trying to do
across a gap.

## 6. Scene-geometry calibration: stable maps, no neural relink value

Motivating idea:
[`bridge_gate_cross_dataset_20260808.md`](../../../reference/benchmarks/bridge_gate_cross_dataset_20260808.md)
established that a bridge height-ratio gate candidate tuned on MOT17
(+0.642 IDF1, with `relink_bridge_px=0.4` fixed) does not transfer to
MOT20/DanceTrack (near-neutral / clearly harmful) — the geometry is real but
not domain-general in pixel space. A
**static, per-camera** ground-plane/perspective calibration
(fit once, not run per-frame) could convert those thresholds to metres —
sidestepping the real-time-budget objection to §2/§4 entirely, since it is not
a live per-frame inference cost.

Tested whether the neural depth is doing real geometric work here, or is
redundant with a cheap `box height ↔ image position` control using the same
GT-assisted calibration data. 4 sequences (different camera geometries), 40
sampled frames each, clean GT boxes only (`class==1`, `conf==1`, `vis≥0.8`),
regressed `log(neural_depth)` on `log(h)` alone, then `+foot_y`, then `+foot_x`
(`scripts/scene_geometry.py`):

| seq | n | R² log(h) alone | R² +foot_y | R² +foot_x |
|---|---|---|---|---|
| MOT17-02-SDP | 347 | 0.945 | 0.955 | 0.959 |
| MOT17-04-SDP | 717 | 0.777 | 0.874 | 0.874 |
| MOT17-09-SDP | 175 | 0.784 | 0.868 | 0.880 |
| MOT17-05-SDP | 114 | 0.916 | 0.922 | 0.925 |

**Camera-dependent, not uniform.** MOT17-02/05: box height alone already
explains 92-95% — the neural model is close to redundant with a cheap `1/h`
law there. MOT17-04/09: box height alone explains only ~78%; `foot_y` recovers
another 8-10 points — the neural model is capturing real scene structure a
pure size cue misses. (Caveat: `log_h` and `foot_y` are correlated, so
individual regression coefficients are not stable/interpretable — only the R²
increments are being read.) This first regression only established that the
model reacts to camera geometry; it did not establish incremental downstream
value.

### 6.1 Foreground-masked static calibration follow-up

The recommended follow-up was run on the three static cameras (02/04/09), 40
evenly sampled frames each. Clean GT boxes locate feet and mask all foreground;
depth is sampled from the ground immediately below each person. Per-camera
spatial cells are temporally median-aggregated, then robust quadratic surfaces
are fitted from normalized foot position to (a) neural ground depth and (b)
expected box height, the cheap control. This is a **GT-assisted capability
ceiling**, not a deployable calibrator. Full artifacts:
[`depth_scene_calibration_probe_20260913/`](../../../../results/analysis/depth_scene_calibration_probe_20260913/).

The fitted neural surfaces are stable under alternating-frame validation:
temporal R² = 0.971 / 0.899 / 0.876 for 02/04/09. The failure is downstream
incremental value:

| score (lower is better) | rank-1, all 39 | rank-1, bridge-hard 26 |
|---|---:|---:|
| current `bridge_dist` | **19/39 (48.7%)** | 6/26 (23.1%) |
| raw height residual | 8/39 (20.5%) | 4/26 (15.4%) |
| neural perspective-corrected height residual | 9/39 (23.1%) | 3/26 (11.5%) |
| cheap position-corrected height residual | 9/39 (23.1%) | 4/26 (15.4%) |
| neural scene-depth difference | 12/39 (30.8%) | 9/26 (34.6%) |
| cheap expected-scale difference | **14/39 (35.9%)** | **11/26 (42.3%)** |

Perspective correction does not rescue the height gate. Neural scene-depth
looks conditionally useful in the bridge-hard subset, but the cheap control is
stronger. In an exploratory runtime-identifiable top-two rule, the current
bridge margin 0.05 is neutral (19→19). Post-hoc margins 0.10/0.20/0.30 give
19→21/22/24 with neural scene-depth, but the cheap surface produces **exactly
the same candidate decision in every activated group at every tested margin**.
Pair-level Spearman correlation is ρ=0.9815. All five apparent fixes at margin
0.30 occur on MOT17-02; 04 has no activation and 09 is unchanged.

**Verdict: neural calibration for relink is NO-GO for this design.** A stable
depth map is not enough; its usable ordering is already encoded by cheap
camera-position geometry. The post-hoc margin sweep licenses only a separate,
cheap-only hypothesis with a frozen rule and held-out cameras. It does not
license production wiring or an end-to-end depth run. Metric scene analytics
and automatic homography generation remain separate, untested applications.

## 7. Not done here

No production code path was touched. No cost-function or embedding change was
made. No end-to-end IDF1/IDs run — everything above is the same offline
rank-1/margin/R² proxy this project already uses to decide whether an
end-to-end run is worth spending
([`reid_repair_funnel_20260905/README.md`](../../../../results/analysis/reid_repair_funnel_20260905/README.md):
"read ranking and margin first, only then decide"). Going from §2/§4/§6's negative
proxy to an actual IDs number would require wiring the feature into the real
merge decision and running tracker+evaluator — an integration step, not a
signal-measurement one, and not justified by the measured result. No external
camera, moving-camera calibration, detector-derived foreground mask, or metric
ground-plane accuracy was tested.
