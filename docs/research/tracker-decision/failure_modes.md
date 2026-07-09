# Geometry-Side Failure Modes

Known decision-layer failure modes affecting matching, birth, death, and relink.

Status labels: **mitigated** (baseline already addresses), **tradeoff** (GO with known cost), **NO-GO path** (do not use listed fix), **open**.

---

## Near-person crossing

### Symptom

ID switch when two people walk close / briefly overlap (sparse scenes like MOT17-05), or over-suppression (FN) if anti-confusion is too strong.

### Likely cause

Inter-track IoU makes OAO fire on **transient** overlaps; pure spatial OAO cannot separate “brief crossing” from “persistent crowd confusion.”

### Related knobs

`oao_tau`, `oao_ramp_frames`, `occ_state_*`, `match_thresh`

### Probe / evidence

Duration-ramp revival: spatial signals 6/6 NO-GO; time axis separates 05 (~10f) vs 04 (~49f).  
`docs/research/eval/oao_duration_ramp_revival_20260617.md`, no_go OAO entry.

### Current status

**Mitigated** with `oao_tau=0.50`, `oao_ramp_frames=25` (tradeoff: residual seq-level variance).

### Possible fix

Do not re-enable spatial OAO family as default. Per-seq adapt rejected for headline (`per_seq_adapt: false`).

---

## Long occlusion recovery

### Symptom

Person exits occlusion with a **new ID**; IDs++ / AssA drop.

### Likely cause

Lost track expired (`track_buffer` / bridge ttl); bridge gates fail (px/h/margin); or young track never hits `bridge_at`.

### Related knobs

`track_buffer`, `relink_bridge_*`, `confirm_streak`, `interpolate_max_gap`

### Probe / evidence

Bridge gate formulas; m relaxation of px/h recovered AssA on small boxes.

### Current status

**Tradeoff** — geometry bridge on; appearance reconnect off (#57). Long pure-appearance re-entry remains hard.

### Possible fix

Async appearance sidecar (not sync critical path). Tighten only if false merges dominate.

---

## Box jitter amplification

### Symptom

Track boxes shake; intermittent unmatched frames; flicker confirm.

### Likely cause

Measurement noise large vs R; low `kalman_r_scale`; detector box unstable on small objects.

### Related knobs

`kalman_r_scale`, GMC, det quality (out of tracker)

### Probe / evidence

m retune 2.8 → 3.5 closed IDF1 gap; residual/kalman research notes.

### Current status

**Mitigated** on m with higher r_scale; s at 2.8.

### Possible fix

Avoid NSA double-comp (#8). Prefer detector stability over association hacks.

---

## Private continuation false keepalive

### Symptom

Ghost / wrong track continues in crowd or moving camera; FP up (known MOTA trade).

### Likely cause

Wider-NMS private candidates attach to wrong active track via prior IoU; score clamp stops birth but not wrong continue.

### Related knobs

`private_prior_iou_threshold`, `private_min_score`, `private_max_candidates`, `private_candidate_nms_iou`

### Probe / evidence

Preset comment: FP 2589→3386 with recall/IDF1 gain; seq 13 weaker.

### Current status

**Tradeoff** (GO for recall); watch moving-cam.

### Possible fix

Stricter prior IoU; per-seq off only if policy allows; not global NMS raise (IDs++).

---

## Late birth from low-score true positive

### Symptom

True person appears many frames before an ID is born; FN then late fragment.

### Likely cause

`new_track_thresh` / confirm gates; score under Mamba calibration; private path cannot birth.

### Related knobs

`new_track_thresh`, `confirm_streak`, `confirm_score_thresh`, `track_thresh`

### Probe / evidence

new_track 0.20 aggregate mirage / cross-seq fail (preset comment, registry-style #38).

### Current status

**Tradeoff** at 0.28.

### Possible fix

Do not chase aggregate-only birth lowers. Detector recall levers preferred for systematic late birth.

---

## Camera motion under-compensation

### Symptom

Mass ID disruption on moving-camera sequences; tracks lag scene.

### Likely cause

GMC off, coarse downscale, low PCR warp shrink, residual parallax (single warp limit).

### Related knobs

`gmc`, `gmc_downscale`, PCR env; secondary `kalman_r_scale`

### Probe / evidence

GMC module docs; tile-PCR affine NO-GO #40; FG mask NO-GO #20.

### Current status

**Mitigated** with GPU GMC ds=4; residual camera error remains.

### Possible fix

Not fuse_score (#45). Not FG mask. Motion residual terms need cand-cap review.

---

## Wrong relink after disappearance

### Symptom

After a gap, ID A continues as person B.

### Likely cause

Loose `bridge_px` / wide h-band / low margin; two people in same place.

### Related knobs

`relink_bridge_px`, `h_lo/h_hi`, `margin`, `dir_bonus`, `ttl`

### Probe / evidence

m looser gates: intentional AssA trade; appearance veto ineffective when clean emb missing (#57).

### Current status

**Tradeoff** (geometry precision-first); app veto not baseline.

### Possible fix

Raise margin; optional spatial gate; never sync ReID for tiny gains.

---

## Height-ratio gate too strict / too loose

### Symptom

Strict: missed small-object bridges. Loose: scale-mismatched merges.

### Likely cause

Single global h-band for all sizes; m/s noise differs.

### Related knobs

`relink_bridge_h_lo`, `relink_bridge_h_hi`

### Probe / evidence

s 0.75–1.33 vs m 0.6–1.7 documented in m preset.

### Current status

**Mitigated** via **per-preset** bands (not per-seq).

### Possible fix

Keep preset split; avoid one-size CLI default for both backbones.

---

## Fuse-score / score-in-cost regressions

### Symptom

Aggregate IDF1/HOTA drop when raising `fuse_score_weight`; per-seq bipolar.

### Likely cause

Score is not a separable ghost signal across sequences (#45 probes).

### Related knobs

`fuse_score_weight`

### Probe / evidence

no_go_registry #45 four orthogonal probes.

### Current status

**NO-GO path** — keep 0.0.

### Possible fix

None in cost matrix; use cascade thresholds only.

---

## Interpolation FP from bad endpoints

### Symptom

Large share of FP from filled boxes; turning interpolate off removes FP but loses FN/IDs badly.

### Likely cause

Wrong-bridge or unmatched endpoints (not path curvature).

### Related knobs

`interpolate_max_gap`, min track len; upstream association/birth

### Probe / evidence

no_go #44 — GMC-aware / Hermite / Bézier interpolators NO-GO.

### Current status

**Tradeoff** — keep interpolate; fix endpoints via tracker decisions.

### Possible fix

Improve association/relink; do not sophisticate fill geometry first.

---

## Candidate list eviction (crowd)

### Symptom

Crowded frames: true match never confirms; tracks stall tentative.

### Likely cause

Mahalanobis-only high-cost candidates fill `K_MAX_CANDIDATES` before true low-cost IoU match.

### Related knobs

`cand_cost_cap` (= max(dda, match_thresh, stage2)), maha gate, match_thresh

### Probe / evidence

Comments in `stage1_cost_fused_kernel` enqueue path.

### Current status

**Mitigated** by cost-cap enqueue filter.

### Possible fix

Do not add terms that systematically lower cost of non-overlapping pairs (`math_model_implementation.md` §2.2).

---

## Related

- Knobs: [assoc_knobs.md](assoc_knobs.md)
- Scoring: [scoring_semantics.md](scoring_semantics.md)
- Relink: [relink_bridge.md](relink_bridge.md)
- Registry: [../../reference/no_go_registry.md](../../reference/no_go_registry.md)
