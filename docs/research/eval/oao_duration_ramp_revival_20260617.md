# OA-SORT OAO Revival via Duration-Ramp (2026-06-17)

**Status:** GO — `oao_tau=0.30, oao_ramp_frames=25` set as `mamba_whole_graph` preset default.
**Supersedes:** registry #7 (OAO ⚪ 被遮蔽 → ✅ GO).
**Eval:** MOT17 train, SDP, `mamba_whole_graph` preset, single-stream, 7 sequences.

## TL;DR

Plain OAO (`cost[t][d] += tau·occ_coeff[t]`, `occ_coeff` = max inter-track IoU) was a
registered NO-GO on the old ~48% baseline. On the current baseline
(whole-graph head + bidirectional bridge relink) it **flips to +1.6 IDF1 / +2.6 AssA**
— but the gain is concentrated in MOT17-04 and it inflicts a **−3.4 IDF1 regression on
MOT17-05**. Six per-detection / spatial discriminators all failed to separate 05's harm
from the 10/13 benefit. The **duration axis** (consecutive overlapped frames) is the only
one that separates them: 05's overlaps are transient crossings (~10 frames), 04's are
persistent crowding (~49 frames). Ramping the penalty by overlap duration
(`tau·occ·min(1, frames/ramp)`) keeps 04/10/13 and recovers most of 05. Final
**`tau=0.30, ramp_frames=25` Pareto-dominates plain OAO on every metric.**

## 1. Baseline numbers (this study)

| config | IDF1 | HOTA | AssA | MOTA | FP | FN | FPS |
|---|---|---|---|---|---|---|---|
| baseline (OAO off) | 75.9 | 68.1 | 66.2 | 77.9 | 3598 | 20732 | 223 |
| plain OAO tau=0.25 | 77.5 | 69.7 | 68.8 | 78.2 | 2416 | 21654 | 220 |

OAO is essentially free (220 vs 223 fps — within noise; `occ_coeff` is a light per-track
kernel and the penalty rides on the already-computed cost matrix).

## 2. The gain is concentrated in MOT17-04 (42% of GT weight)

Per-seq ΔIDF1 (plain OAO − baseline): 04 **+2.7**, 13 +3.0, 10 +1.6, 02 +0.5, 09 +0.5,
11 −0.7, **05 −3.4**. AssA gain is almost entirely 04 (+4.6).

MOT17-04 is unique: **static camera** (reliable `occ_coeff` on predicted states) ×
**high recall 91.5%** (detections exist to re-route) × **45 ppl/frame** (most assignment
conflicts). It is detection-saturated (Prcn 99.8), so all headroom is association — exactly
where OAO acts. 04 alone = 47,557 GT = **42% of all train GT**, so it dominates the
weighted aggregate. **ex-04 (6-seq) gain is only +0.6 IDF1 / +0.3 AssA** — the headline
+1.6 is largely an MOT17-04 artifact. (Lesson: track ex-04 / per-seq, not just the
aggregate — see registry note on inflated totals.)

Mechanism in 04 (output diff): OAO **re-routes** (adds 102 TP boxes under correct IDs,
0 FP, FN −23, IDs flat) rather than suppressing — it defers the greedy IoU grab in
ambiguous overlaps so the detection lands on the correct track. In 10/13 it **suppresses
FP** (ghost track loses the contested detection). In 05 it **suppresses TP** (sparse,
FP-poor: the few overlaps are real side-by-side people) → FN +149.

## 3. Six spatial discriminators all NO-GO

Goal: keep the 04/10/13 benefit, drop the 05 harm. All measured on `tau≈0.25`.

| signal | idea | result |
|---|---|---|
| **contention gate** | penalise only if the detection is also claimed by t's max-overlap partner (partner-pred IoU ≥ thr) | NO-GO — recovered 05 (+2.3) but killed 10's gain (−3.4); 10/13 benefit is NOT from contested detections. Net 76.9 |
| **score weight** | `penalty·(1 − w·det_score)` — spare confident boxes | NO-GO — killed 04 re-routing (−4.1 at w=0.5) and did NOT recover 05 (−0.3). Gain lives ON high-score overlaps; 05 harm not score-separable. 75.3 |
| **union coverage** | `occ_coeff` = fraction of t covered by union of others (8×8 grid) | redistribution — boosted 10/11, lost 04/13, 05 still −3.3. 76.8 |
| **crowd `(1−1/N)`** | scale by local people count within radius·h | best headline 77.8 (lets tau go higher safely) but 05 still −3.2, ex-04 = plain |
| **height gate** | only same-height partners (\|h_t−h_j\| ≤ g·max h) contribute | gate 0.3 ≈ no-op; gate 0.15 recovered 05 (−0.8) but lost 10/13 FP; ex-04 66.8 < plain. Worst-seq −0.8 (most uniform but lower total) |
| **foot gate** | only same-foot-line partners (\|footy_t−footy_j\| ≤ g·h_ref) contribute | **worse** — gate 0.15 collapsed 04 (+0.1): a dense crowd spans a depth (foot) range, so foot is too strict; height (scale) is the better same-region proxy for crowds |

**Wall:** in every spatial axis, 05's harmful overlaps are geometrically the same kind of
configuration as 04's beneficial overlaps; the difference is scene-level (05 sparse/FP-poor;
re-routing has nowhere to go) not a local per-box property. Same wall as P5-4
SceneAdaptivePolicy (scene classifiable but adjustment incompatible).

## 4. Breakthrough: the temporal axis

Overlap persistence per track (baseline output, IoU>0.2 run-length):

| seq | mean run (frames) | OAO ΔIDF1 |
|---|---|---|
| 04 | **48.8** | +2.7 |
| 02/09/10/13 | 14–21 | mixed |
| **05** | **10.2** | −3.4 |

05 (transient crossings) and 04 (persistent crowding) sit at opposite ends of the
duration axis — the **only** measured axis where harm and benefit separate. Ramp the
penalty by a per-track consecutive-overlapped-frame counter:

```
occ_coeff[t] *= min(1, occ_duration[t] / ramp_frames)
```

Transient overlaps get a damped penalty (real track survives the brief crossing);
persistent crowds reach the full penalty (re-routing preserved).

## 5. Final sweep — `ramp_frames × tau`

| config | IDF1 | HOTA | AssA | ex-04 IDF1 | ex-04 AssA | MOT17-05 IDF1 |
|---|---|---|---|---|---|---|
| baseline | 75.9 | 68.1 | 66.2 | 66.5 | 54.1 | 73.9 |
| plain OAO (.25) | 77.5 | 69.7 | 68.8 | 67.1 | 54.5 | 70.6 |
| ramp20 t0.25 | 77.6 | 69.7 | 69.1 | 67.2 | 55.0 | 71.4 |
| ramp20 t0.30 | 77.3 | 69.6 | 68.8 | 66.7 | 54.6 | 71.5 |
| ramp25 t0.25 | 77.4 | 69.6 | 68.9 | 67.0 | 54.8 | 71.5 |
| **ramp25 t0.30** | **77.6** | **69.9** | **69.1** | **67.2** | **55.3** | **72.3** |

**`tau=0.30, ramp_frames=25` ≥ plain OAO on every axis** (Pareto-dominant): total IDF1/HOTA/AssA
all max, ex-04 AssA +0.8 over plain (robust set, not 04-inflated), and the MOT17-05
regression halved (73.9 baseline → 70.6 plain → **72.3**). The high-tau + long-ramp combo
is the sweet spot: crank the penalty for persistent crowds while damping transient
crossings harder.

## 6. Implementation

`src/tracking/tracker_gpu.cu` — `set_oao_params(tau, contest_thresh=-1, score_w=-1,
occ_mode=0, crowd_radius=0, height_gate=0, foot_gate=0, ramp_frames=0)`. All six failed
discriminators are retained as **default-off, bit-exact** ablation knobs; the duration ramp
uses a per-track `d_occ_duration_` counter updated in `compute_track_occlusion_kernel`
(reset on inactive slot / no-overlap frame). Preset default lives in
`configs/presets/mamba_whole_graph.yaml`.

CLI: `--oao-tau --oao-ramp-frames --oao-contest-thresh --oao-score-w --oao-occ-mode
--oao-crowd-radius --oao-height-gate --oao-foot-gate`.

## 7. Caveats

- MOT17 **train**, single-stream, this GPU. Test-server generalization not measured.
- MOT17-05 is still −1.6 vs baseline (not fully neutral); fully recovering it (ramp20 t0.40
  hit 05 ≈ 0) costs total/04, so it's left partially recovered for the Pareto-best point.
- The headline +1.6 is 04-weighted; the durable/transferable gain is the ex-04 +0.7 IDF1 /
  +1.2 AssA (vs baseline), which duration-ramp improves over plain.
