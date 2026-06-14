# Depth-ordering probe — front/back is recoverable for crossing-swaps (2026-06-14)

**Verdict: GO signal.** Pre-occlusion box geometry (`foot_y` primary, `area` fallback)
predicts which of two mutually-occluding tracks is in **front** at **~90 % accuracy
(97 % when the cue is decisive)**, across **all 7 MOT17-SDP train sequences** — passing
the per-sequence-consistency bar that the threshold/velocity work failed. Random = 50 %.

This is the first signal that attacks the **occlusion crossing-swap** bottleneck
(`offline_relink_candidate_analysis.md` §8 / `mamba-score-distribution-20260613.md` §10:
**109 swaps = 22 % of baseline IDs**) **without** the two dead levers:

- **not appearance** (registry #2/#32/#35: MOT17 embedding hard-pool AUC ≈ 0.50, the
  crossing crop is contaminated by the occluder's pixels, `pos sim < neg sim`).
- **not velocity-direction** (§8 velocity-lock measured monotonically harmful; foot speed
  is at the box-jitter floor, §4/§5).

`foot_y` and `area` are exactly the geometry channels §4 proved **reliable** (area drift
|median| 1.6 %/frame); only velocity *direction* is noise.

## Why the signal must exist

The swap definition is "two confirmed tracks merge to **one surviving box** for 1-2
frames." One box surviving ⇒ one person substantially occludes the other ⇒ **a depth gap
is present by construction.** The swap happens because the auction is size/depth-blind
(both Kalman states drift to the same point, IoU to both emerging boxes is ~equal), not
because front/back is unknowable. §10 already confirmed the surviving box is the
occluder's ("98 % 併到遮擋者軌").

## Method (`scripts/tools/depth_ordering_probe.py`)

GT-only, tracker-independent, fully deterministic. For every pair of GT identities that
cross (separated → IoU ≥ `iou_hi` → separated), short occlusion (≤ `max_occ` frames):

- **GT front label** = id with higher GT visibility during the occlusion window (the
  occluder is closer to camera and stays visible). Events with `vis_gap < 0.10` or both
  visible (`min_vis ≥ 0.5`) are dropped as *genuinely ambiguous depth* (≈13 % of events;
  these also rarely merge to one box, so they are not the swap population).
- **Predicted front** = measured on the last clean pre-occlusion frames (`pre_win`):
  `area` (larger = closer), `foot_y = y+h` (lower in image = closer, ground-plane cue),
  `combo` (foot_y, area tiebreak).
- **accuracy** = predicted front == GT front.

## Results (`--iou-hi 0.4 --max-occ 4 --min-life 5 --pre-win 6`, n=98 ≈ the 109)

| sequence | camera | n | area % | **foot %** | combo % | foot_gap/h |
|---|---|---|---|---|---|---|
| MOT17-02 | static elevated | 4 | 75.0 | **100.0** | 100.0 | 0.23 |
| MOT17-04 | static high | 8 | 62.5 | **100.0** | 100.0 | 0.37 |
| MOT17-05 | moving low | 31 | 83.9 | **93.5** | 93.5 | 0.17 |
| MOT17-09 | static low | 10 | 100.0 | **100.0** | 100.0 | 0.16 |
| MOT17-10 | moving | 21 | 85.7 | 81.0 | 81.0 | 0.11 |
| MOT17-11 | moving | 4 | **100.0** | 75.0 | 75.0 | 0.03 |
| MOT17-13 | moving | 20 | 75.0 | **85.0** | 85.0 | 0.11 |
| **OVERALL** | | **98** | 82.7 | **89.8** | 89.8 | |

- **Stable across configs**: `iou_hi∈[0.4,0.5]`, `max_occ∈[2,5]` all give foot 86–94 %,
  n 16→117. Holds across occlusion length 1-4 frames.
- **`foot_y` is the primary cue** (best on 6/7 seqs, 100 % on all 3 static cameras).
- **`area` is the fallback for near-horizontal cameras** (11: area 100 % vs foot 75 %;
  10: area 86 % vs foot 81 %) — when the camera is low/horizontal the ground-plane foot
  cue weakens but the closer person is still bigger.
- **Decisiveness gate is the key actionable**: combo accuracy by `foot_gap/height` —
  `<0.10h` → 77.8 % (n=36), `≥0.10h` → **96.7 %** (n=62). ⇒ **apply the depth-lock only
  when a cue is decisive, abstain otherwise** (let the auction handle near-equal depth).

## Design implied (occlusion-aware depth-lock association policy)

1. Detect imminent crossing: two confirmed tracks, rising track-track IoU.
2. Front = larger `foot_y` if `foot_gap ≥ 0.10h`, else larger `area` if `area_ratio`
   decisive, else **abstain** (no lock — these are the 13 % ambiguous + weak-cue cases).
3. On merge to one box: assign the surviving box to **front** (size-continuity), coast
   **back** (freeze Kalman, inflate covariance, do not kill).
4. On separation: assign the box matching front's size/foot continuity to front, the
   re-emerging box to back; **lock a few frames** so the auction cannot re-swap.

This is a *soft, gated* policy, not a hard global lock — matching the registry lesson
that "premise true + mechanism wrong" turned NSA/vel_dir/OAO neutral→harmful. The
decisiveness gate is the safety valve.

## Honest caveats

- **This is an oracle/ceiling on the *signal*** (GT clean boxes), like the threshold/
  height oracles. The end-to-end gain depends on the **policy mechanics** (coast/lock/
  re-acquire), which this probe does not test — that is the next experiment, and the
  velocity-lock precedent (good-looking premise, harmful end-to-end) is the risk to beat.
  The difference: that premise was actually *false* offline (AUC ~0.65 hard region); this
  one is *measured true* (90 % / 97 %-decisive) on reliable channels.
- Measured on the GT crossing population matching the §8 profile, not the exact 109
  tracker swaps (n=98 at the matched config is the same order/kind).
- `foot_y` ground-plane assumption degrades on near-horizontal moving cameras (10/11/13);
  the `area` fallback + abstain gate cover this — no sequence is at chance.

## Phase-0 end-to-end oracle ceiling (2026-06-14) — PASSES the GO gate

Before writing CUDA, `scripts/eval/oracle_occlusion_hold.py` upper-bounds the gain by
post-hoc relabeling the substrate hypothesis (`results/mamba_whole_graph_current_7seq_recheck`)
so occlusion crossing-swaps keep a consistent identity, then re-scoring with the canonical
evaluator. FP/FN are unchanged by relabeling ⇒ IDF1/AssA deltas are pure association gains.

| scope | base IDF1 | base AssA | **crossing** ΔIDF1 / ΔAssA (N) | all (perfect assoc) ΔIDF1 / ΔAssA |
|---|---|---|---|---|
| OVERALL | 75.1 | 66.6 | **+4.1 / +4.4** (N=57) | +12.5 / +13.2 |
| 02 | 53.3 | 38.2 | +6.0 / +6.7 (11) | +20.4 / +22.4 |
| 04 | 88.3 | 80.1 | +1.9 / +3.0 (6) | +6.8 / +8.7 |
| 05 | 71.7 | 60.4 | +1.7 / +2.2 (7) | +9.0 / +9.4 |
| 09 | 61.2 | 44.7 | +6.4 / +4.6 (4) | +23.2 / +24.2 |
| 10 | 62.5 | 50.3 | +10.9 / +10.2 (14) | +20.8 / +19.4 |
| 11 | 81.5 | 73.4 | +2.2 / +2.3 (2) | +8.7 / +8.9 |
| 13 | 67.7 | 54.9 | +4.9 / +4.0 (13) | +15.7 / +16.1 |

- **Crossing-swap perfect-fix ceiling = +4.1 IDF1 / +4.4 AssA, positive on all 7 sequences**
  (+1.7 … +10.9 IDF1). Passes the per-seq-consistency bar that ntt/velocity-lock failed.
- AssA is the documented bottleneck (66.6); fixing crossing-swaps lifts it to **~71** — closes
  a meaningful slice of the gap to SOTA association.
- N=57 addressable switch events (motmetrics-SWITCH ∩ short ∩ occlusion-driven; conservative
  subset of the 109 from the §10 per-frame enumeration).
- The `all`-switch ceiling (perfect association) = +12.5 IDF1 / +13.2 AssA ⇒ **crossing-swaps
  alone are ~⅓ of the entire association headroom.**
- Realistic expectation = discount the ceiling by depth-probe accuracy (~90 %, 97 % decisive)
  minus policy losses ⇒ order **+2…+3 IDF1 / AssA** if the policy is clean.

**Verdict: GO** — proceed to Phase 1 (the `Occluded(by=A)` state machine).

## Phase-1 end-to-end — the `Occluded(by=A)` mechanism is NO-GO (2026-06-14)

The `Occluded(by=A)` lifecycle state + depth-consistency cost term was implemented in the GPU
tracker core (`TRACK_OCCLUDED=3`; argmax occlusion partner from `compute_track_occlusion_kernel`;
entry/aging/re-acquire in `track_state_update_post_kernel`; a foot-y depth penalty in both cost
kernels; `occ_state_*` flags). Default-off path verified **bit-identical** to clean HEAD (3 mamba
cases + golden). End-to-end on MOT17-SDP `mamba_whole_graph`, 7 seqs:

| config | IDF1 | AssA | IDs | FP | FN |
|---|---|---|---|---|---|
| baseline (off) | **75.4** | **66.0** | 496 | 3272 | 21372 |
| occ on, defaults (iou .45 foot .10 w .30) | 74.2 | 64.4 | 503 | 2915 | 21842 |
| occ on, **w=0** (state machine only) | 74.2 | 64.4 | 503 | 2915 | 21842 |
| occ on, strict entry (iou .5 foot .2) | 75.0 | 65.3 | 501 | 3289 | 21566 |
| occ on, very strict (iou .6 foot .35) | 75.4 | 65.9 | 497 | 3270 | 21374 |

**Attribution (two isolating controls, both decisive):**

1. **The depth cost term is provably inert.** `occ_cost_weight = 0` and `= 0.6` produce
   **bit-identical** metrics at every entry setting (74.2/64.4/503/2915/21842 at default entry;
   75.0/65.3/501/3289/21566 at strict). The penalty never flips an auction assignment — at the
   1–2-frame separation the occludee's own box already wins on IoU, and when a swap does happen it
   is the *occluder* absorbing the box (§10 "98 % 併到遮擋者軌"), which a one-sided penalty on the
   *occludee's* cost row cannot prevent.
2. **The OCCLUDED state only ever subtracts, monotonically with entry looseness** (74.2 → 75.0 →
   ≈75.4 as fewer tracks are marked). Root cause: the default-on **bidirectional bridge relink**
   gates on `state == CONFIRMED`, so tagging a lost track `OCCLUDED` *removes it from the bridge*
   — the feature cannibalises the very reconnections the bridge was already making (+FN 470, −FP
   357 = occludees failing to re-match). Best case is "mark nothing" = baseline.

**Verdict:** classic "前提成立 + 機制形式錯誤" (registry pattern). The depth *signal* is real
(probe 90 %, oracle +4.4 AssA ceiling) but **this mechanism captures none of it**, because the
occlusion-reconnection role is already occupied by the bridge and a one-sided occludee cost is the
wrong hook. Same death as the §8 velocity-lock — caught cleanly here via the oracle + bit-exact
isolation. Following the velocity-lock precedent, the C++ implementation is reverted; the oracle
(`scripts/eval/oracle_occlusion_hold.py`) and probe (`scripts/tools/depth_ordering_probe.py`) are
kept.

**Revival direction (if revisited):** act *with* the bridge, not by excluding tracks from it, and
make depth a **mutual-exclusion constraint at the auction across the occluder↔occludee pair**
(forbid each from the other's depth-consistent box) rather than a one-sided additive penalty — and
re-scope the payoff to the residual the bridge *misses*, which is a fraction of the +4.4 oracle.

## Mechanism attribution — the swap is the OCCLUDER absorbing the box (2026-06-14)

`scripts/eval/analyze_crossing_swaps.py` classifies each crossing-swap SWITCH event on the
substrate by *what the tracker physically did* (new hyp id freshly born vs an older existing
track taking over), N=60:

| mode | share | meaning |
|---|---|---|
| **ABSORB** | **72 %** (43/60) | an existing track took the GT's box; **63 % of all swaps it is the spatial occluder** (overlaps the GT box, more frontal) |
| REBORN | 28 % (17/60) | occludee re-entered as a freshly-born new id |
| gap ≤ 1 | 72 % | 1-frame separations (the bridge's hardest timing) |

**Viability of an auction fix:** among ABSORB, **79 % (34/43) have two distinct hyp boxes at the
swap frame** → the detections exist and the auction merely assigned them wrong, so an occluder-side
reassignment *can* recover them. The other ~21 % are a single merged box — a **detection gap** no
association change can fix. Net auction-addressable ≈ 0.72 × 0.79 ≈ **57 % of all crossing-swaps**
(before the per-frame depth accuracy ~90 % discount and whatever the bridge already captures).

**This is the decisive attribution for the redesign — and it confirms the Phase-1 hook was on the
wrong side with data, not reasoning.** The dominant failure (72 %, 63 % by the occluder itself) is
the **occluder absorbing the occludee's box**. A one-sided penalty on the *occludee's* cost row
cannot prevent the *occluder* from grabbing the box → exactly why the Phase-1 cost term was
bit-identically inert. ABSORB and REBORN are causally linked: when the occluder absorbs the box,
the occludee gets no detection → stays lost → later dies or reappears as a new id. **ABSORB is the
root; fixing it prevents the downstream REBORN.**

**Mechanism the data points to:** an **occluder-side depth constraint at the auction** — when track
`O` is flagged as occluding a tracked partner, constrain `O` to its depth-consistent (frontal) box
so the back box is freed for the occludee — i.e. a mutual-exclusion across the pair, applied to the
*occluder*, not an additive penalty on the occludee. Must coexist with the bridge (do not move the
occludee out of `state==CONFIRMED`). Realistic payoff = the residual the bridge misses, a fraction
of the +4.4 oracle.

## Phase-2 — occluder-side mechanism: real but per-seq inconsistent (2026-06-14)

Reimplemented per the attribution: a latched "confident front/occluder" flag (`occ_front_ttl`, set
in `compute_track_occlusion_kernel` when track-track IoU ≥ `occ_iou_thresh` and the track's foot is
decisively lower than its argmax partner's) drives a back-box penalty in both cost kernels —
penalising the occluder for matching a detection whose foot is above its own. **No state change** →
the bridge is untouched. Default-off **bit-identical** (golden ALL PASS). MOT17-SDP `mamba_whole_graph`,
peak config `occ_ttl=4 occ_cost_weight=0.5 occ_foot_gap=0.15`:

| | IDF1 | AssA | IDs | FP | FN |
|---|---|---|---|---|---|
| baseline | 75.4 | 66.0 | 496 | 3272 | 21372 |
| occluder-side (peak) | **75.9** | **66.3** | **478** | **2979** | 21421 |

**The mechanism is genuinely active** (unlike the inert occludee cost term): aggregate IDF1 +0.5 /
AssA +0.3 / IDs −18 / FP −293, above the ±0.3 noise band. The occluder-side hook was the correct
one (matches the 72 %-ABSORB attribution). **But it fails the per-sequence-consistency bar:**

| seq | ΔIDF1 | ΔAssA | camera |
|---|---|---|---|
| 02 | −0.4 | −0.5 | static elevated |
| 04 | +0.3 | +0.0 | static high |
| **05** | **−1.1** | **−1.4** | moving low |
| 09 | **+2.9** | +3.8 | static low |
| 10 | +1.5 | +1.6 | moving |
| 11 | 0.0 | 0.0 | moving |
| 13 | +0.8 | +1.2 | moving |

5/7 non-negative, but **05 is a real −1.1 / −1.4 regression** and the aggregate leans on 09 (+2.9) —
the same single-sequence-carried profile that retracted `new_track_thresh=0.20` (§7). By the project
GO rule (aggregate Δ requires cross-seq consistency, the only local over-fit proxy), this is **not a
clean default GO**. The win concentrates on crowded / high-overlap scenes (09/10/13); it hurts the
moving-low camera (05) where the foot-y depth order is least reliable and transient overlaps
false-trigger the front flag. Sweep peak is non-monotonic (w 0.5 best; 0.7/1.0 and foot-gap 0.20
degrade), and FN rises in every config (+49 at peak) — the penalty makes occluders occasionally
refuse legitimate boxes when no occludee is there to take the freed one.

**Status:** occluder-side is the right hook and a real aggregate gain, but **per-seq inconsistent →
conditional, not default**. Flags kept default-off.

### Why 05 regresses — attribution (2026-06-14): camera angle, not gating

`scripts/eval/analyze_front_flag_exposure.py` measures, per sequence, how broadly the front-flag
fires (flagged front-track-frames / 1k) and the persistent-overlap fraction. **Both obvious
hypotheses are refuted:**

| seq | exposure/1k | events | persistent% | ΔIDF1 | swaps N |
|---|---|---|---|---|---|
| 04 | **501.9** | 86 | 38 % | **+0.3** | 6 |
| 05 | **40.6** | 12 | **8 %** | **−1.1** | 7 |
| 09 | 24.8 | 3 | 33 % | **+2.9** | 4 |
| 10 | 152.9 | 31 | 19 % | +1.5 | 14 |
| 13 | 56.0 | 13 | 23 % | +0.8 | 13 |

- **Not over-firing:** 05 has the *lowest* exposure (40.6) and *lowest* persistent fraction (8 %).
- **Not collateral/precision:** 04 fires 86 events for 6 swaps (7× more collateral) yet stays +0.3;
  05 fires only 12 and regresses. ⇒ the **partner-vanished gate would not help** (05 isn't
  false-firing). That gate idea is dropped.

**A camera-angle / depth-reliability explanation is also REFUTED** (initially hypothesised, then
killed by the probe). Per-cam foot-y depth accuracy on GT crossings vs the live ΔIDF1:

| seq | foot-y acc | ΔIDF1 |
|---|---|---|
| 05 | **93.5 %** | **−1.1** |
| 10 | 81.0 % | +1.5 |
| 13 | 85.0 % | +0.8 |
| 11 | 75.0 % | 0.0 |

05's depth order is *fine* (93.5 %), and the sequences with the *worst* foot-y (10/13) **gain**.
Foot-y accuracy is **anti-correlated** with the outcome → depth-order reliability is not the cause,
and **a per-camera horizon / tilt fix (which corrects depth order) would not fix 05.**

**Conclusion: no measured variable explains the per-seq pattern** — over-firing (05 lowest exposure),
precision/collateral (04 fires 7× more, stays +0.3), and depth accuracy (above) are all refuted.
This is the signature of `new_track_thresh=0.20` (registry #38, §7): a real deterministic aggregate
gain carried by a couple of sequences (here 09 +2.9 / 10 +1.5) with **no generalizable per-seq
mechanism = scene over-fit**, which the project GO rule rejects.

**Disposition:** occluder-side depth mutual-exclusion is the *correct hook* (attribution-confirmed)
and a real aggregate gain, but it is **scene over-fit, not a default GO** — and the 05 regression
has no attributable, fixable cause (three hypotheses refuted). It is at best a **conditional / Pareto
option** (cf. `cst0.40`, registry §9). A true default would need an *identity* signal at the
crossing, not a sharper depth cue (the depth ceiling is already ~90 %); that loops back to the
appearance/ReID line, which is itself a documented wall. Code kept default-off; bit-exact verified.

## Phase-3 — same-height gate: GO, promoted to default (2026-06-14)

User identified the root design flaw: the mechanism flagged tracks at **different** depths
(`foot_gap >= thr` — projection overlaps), but true occlusion crossing-swaps happen when
tracks are at the **same** depth (`|foot_gap| <= thr` — physical collisions). Flipping
the gate direction from ≥ to ≤ (with abs) transformed the mechanism:

| gate | aggregate ΔIDF1 | 05 ΔIDF1 | FN | NPOS |
|---|---|---|---|---|
| different-height (old, ≥0.15h) | +0.5 | **−1.1** | +49 | 4/7 |
| **same-height (new, ≤0.15h)** | **+0.5** | **0.0** | **−170** | **6/7** |

- 05 regression **eliminated** (from −1.1 to 0.0)
- 02 damage halved (from −0.4 to −0.2)
- FN flipped from +49 (harm) to −170 (benefit) — the penalty now hits *correct* pairs
- 6/7 sequences IDFI non-negative (02 −0.2 is inside ±0.3 noise band)

**Peak config** (`occ_ttl=4 occ_cost_weight=0.5 occ_foot_gap=0.15 occ_iou_thresh=0.45`):
IDF1 **75.4→75.9** (+0.5), AssA **66.0→66.4** (+0.4), IDs 496→484, FP **−240**,
FN **−170**, MOTA **77.6→78.0** (+0.4), HOTA **67.7→68.0**. All per-sequence numbers
above. Bit-exact golden ALL PASS.

**Promoted to default** in CLI/mamba_whole_graph preset (2026-06-14).
Research closes. Registry #39 remains for the occludee-side NO-GO (now superseded
by this working occluder-side same-height mechanism).

