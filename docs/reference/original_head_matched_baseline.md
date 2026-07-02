# Original (YOLO) detect head vs Mamba head — matched baseline (MOT17)

_2026-07-01. Fills a hole flagged by the user: **the project never ran the
original detection-head architecture through the identical tracker, so there was
no control proving the Mamba head is better than the original architecture.**
This is that control._

## 1. The missing control

The deployed system (`mamba_whole_graph`, MOT17 IDF1 78.6) runs a **Mamba
detection head** on a gated YOLO backbone. Every prior head comparison was
detector-only recall (and only on PersonPath22, see
`project_mamba_head_compresses_recall_go`). No run ever put the **original YOLO
detect head** through the *same tracker* on MOT17. So "the Mamba head earns its
place" had no baseline: the treatment arm existed, the control arm did not.

## 2. Design — single variable = the head

Both arms share **everything** except the detection head:

| | treatment | control |
|---|---|---|
| head | Mamba head (`runs/mamba_gt_v14replica_t3_t1/best.ckpt`) | native YOLO detect head |
| backbone | `gated_det_v14replica/epoch_0012` (PyTorch) | **same object** |
| tracker | full pipeline (GMC / relink / OAO / interp / multiplicative cost) | **same** |
| eval | 7 MOT17-SDP, `mamba_pyt_backbone` preset, eager | **same** |

`epoch_0012` is the *correct* shared backbone: the Mamba head is SHA-locked to it,
and it is literally the teacher head's own backbone — so this isolates the head
cleanly. (The deployed 78.6 uses a legacy-teacher TRT backbone + whole-graph
runtime; that is a deployment artifact, not the clean head control. See §6.)

Implementation: `TeacherHeadDetector`
(`src/saccade/perception/temporal_yolo/teacher_head_detector.py`) wraps the gated
teacher's native detect head as a drop-in detector (`detect_raw → [B,N,6]`,
person-filtered, gate-free), so it flows through the identical tracker.
CLI: `--teacher-head-ckpt`. Run driver:
`scripts/eval/run_teacher_head_matched_baseline.sh`.

## 3. The calibration trap (why a naive control is meaningless)

At the **frozen mamba-tuned thresholds** the teacher head collapses to **IDF1
47.5 / Rcll 35.2**. This is NOT a detection failure — it is calibration. The
frozen `confirm_score_thresh=0.50` is tuned for the Mamba head's *saturated* score
distribution (median ~0.93); the YOLO head's scores are lower and its confident
detections sparse, so almost no track ever confirms. The gating knob is
**confirmation**, not birth: lowering `new_track_thresh` did nothing; lowering
`confirm_score_thresh` recovered everything.

⇒ **A fair head A/B must tune each head to its own operating point.** Comparing
two heads at one head's thresholds measures calibration, not architecture.

## 4. Result — dead heat on aggregate (each head at its fair point)

7-seq aggregate, `epoch_0012` backbone, identical tracker:

| head (fair point) | IDF1 | MOTA | Rcll | Prcn | AssA | IDs | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Mamba** (cst 0.50) | **71.4** | 72.5 | 77.7 | 94.3 | 61.0 | 553 | 36.4 |
| **Original YOLO** (cs2, cst 0.15) | **71.0** | 69.2 | 72.5 | 96.3 | 59.8 | 486 | **59.7** |
| Original @ naive mamba-thr (cst 0.50) | 47.5 | 34.8 | 35.2 | 99.1 | 58.6 | 61 | 61.2 |

The original architecture, fairly calibrated, is **within 0.4 IDF1 of the Mamba
head at 1.6× the throughput**, with higher precision and fewer ID switches. The
Mamba head's only genuine edge is **recall (+5.2pp)**, which buys only +0.4 IDF1.

(Mamba's own confirm sweep: cst 0.30 = IDF1 70.8 — the frozen 0.50 is optimal for
the crowd-dominated aggregate; lowering it trades recall for IDs/FP. Teacher
sweep: cst 0.20→70.9, cst 0.15→71.0, cst 0.30→69.2.)

**Robustness — same conclusion with the FULL deployed tracker** (adding
`private_continuation`, so the tracker config is byte-identical to
`mamba_whole_graph` minus the whole-graph runtime):

| head + private_continuation | IDF1 | MOTA | Rcll | Prcn | IDs | FPS |
|---|---:|---:|---:|---:|---:|---:|
| Mamba | 69.6 | 70.8 | 78.5 | 91.7 | 641 | 37.2 |
| Original YOLO | **70.1** | 68.9 | 73.1 | 95.2 | 507 | **60.3** |

So across both tracker configs the gap flips sign but stays inside ±0.5 IDF1
(no-PC: mamba +0.4; +PC: teacher +0.5) — a statistical **tie either way**, with
the original head 1.6× faster in both. (private_continuation happens to *hurt* the
Mamba head on this eager/epoch_0012 path, −1.8; it is +0.4 only on the deployed
legacy-backbone whole-graph — a backbone/runtime-specific interaction, not a head
property.)

## 5. Where each head wins — a clean scene split

Per-sequence IDF1 (mamba vs original, each at its fair point):

| seq | scene | mamba | original | winner | recall Δ (mamba−orig) |
|---|---|---:|---:|---|---:|
| 02 | crowded | 51.3 | 49.2 | mamba +2.1 | +5.1 |
| 04 | very crowded | 83.9 | 81.1 | **mamba +2.8** | **+8.2** |
| 05 | sparse / moving | 72.9 | 76.5 | orig +3.6 | −2.6 |
| 09 | sparse | 63.4 | 68.5 | **orig +5.1** | −2.6 |
| 10 | moving cam | 54.8 | 59.2 | orig +4.4 | +5.0 |
| 11 | moving cam | 78.6 | 76.7 | mamba +1.9 | +3.1 |
| 13 | moving cam | 61.7 | 63.9 | orig +2.2 | +3.0 |

**The Mamba head wins exactly where its design targets — crowded, small-object,
dense scenes (02/04): recall +5 to +8pp.** The original head wins 4/7 sparse /
moving-camera sequences, where full-resolution detection + cleaner calibration
beat the Mamba head's ÷4-downsample recall on people that aren't crowded. The
aggregate is a tie only because crowded MOT17-04 (biggest GT weight) pulls the
Mamba head up.

## 6. Interpretation — for the contribution framing

- **The Mamba head is NOT globally superior to the original architecture.** It is
  a **crowd / small-object recall specialist** that trades throughput (1.6×
  slower) and sparse-scene accuracy for dense-scene recall. On the balanced
  MOT17 aggregate the two heads are a statistical tie (+0.4 IDF1).
- This is consistent with, and now *quantifies on MOT17*, the PP22 finding
  (`project_mamba_head_compresses_recall_go`): the ÷4 spatial reduction costs
  recall, but only crowded/small-object regimes make the head the deciding factor.
- The safe, defensible contribution claims remain the ones the mainline already
  makes (`project_main_line_framing`, `PROJECT_SHOWCASE`): the **system /
  real-time runtime** and the **tracker delta on a fixed detector** — NOT
  "the Mamba head beats the conventional head." A head-superiority claim on
  accuracy is unsupported (dead heat) and, on speed, false (1.6× slower).
- If the Mamba head is to be justified, it is as a *crowded-scene* module, or on
  its runtime/temporal-gating properties — not as a general-purpose detection head.

### Caveats
- `epoch_0012` (replica) backbone is weaker than the deployed legacy-teacher TRT
  backbone; both heads share it, so the **head delta is valid**, but absolute
  numbers (71.x) sit below the deployed 78.6. A deployed-quality version would
  need the teacher head exported to run on the legacy backbone (the Mamba head's
  detect head being SHA-locked to `epoch_0012` makes `epoch_0012` the honest
  shared choice here).
- Eager (`--no-compile`) → FPS is a lower bound; the *relative* 1.6× (teacher
  faster, no ÷4 scan round-trip) is the real signal.
- MOT17 train = in-sample for both (backbone/head trained on these 7 seqs);
  relative head comparison holds, absolute numbers optimistic.

## 8. Native architecture ceiling — full retraining (2026-07-02)

The matched control (§2–5) deliberately froze the shared `epoch_0012` backbone, so
the native head there only ever saw the teacher's 12-epoch joint training while the
Mamba head got its full ~90-epoch head curriculum. The user's follow-up: give the
**native architecture** a fair, unconstrained post-training and find its true
ceiling. Fresh full-network run (backbone + native detect head) from `yolo26s.pt`,
same 7 MOT17-SDP in-sample, 30 epochs. `runs/gated_det_native_full`,
`scripts/train/temporal_yolo/train_gated_detector.py` (`--lr-gate 1e-3 --lr-yolo
1e-5 --gt-ratio 0.5 --clip-grad 1.0 --warmup-epochs 3 --seed 42`).

**Stability**: `clip-grad 1.0` + the loop's non-finite-batch skip cleared the known
teacher **e20 NaN wall** (only 4 batches skipped over 30 epochs). Train loss
4.79 (e12) → **4.13 (e26 plateau)** → 4.15 (e30) — the legacy e12 cap left real
loss on the table.

Eval sweep (fair point cs2 / cst0.15 / ntt0.20, `mamba_pyt_backbone`, eager):

| epoch | IDF1 | MOTA | Rcll | Prcn | AssA | IDs | FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| e12 | 69.9 | 69.1 | 72.2 | 96.5 | 59.3 | 506 | 2951 |
| e20 | 72.4 | 70.0 | 72.3 | 97.4 | 61.2 | 436 | 2148 |
| **e26** | **73.3** | 70.6 | 73.0 | 97.3 | 62.3 | 461 | 2300 |
| e30 | 72.2 | 70.5 | 72.7 | 97.5 | 60.8 | 425 | 2063 |

**Two findings:**

1. **Training past e12 helps (+3.4 IDF1 to e26), but the gain is association, not
   recall.** e12→e26: AssA +3.0, FP −650, precision up — but **recall is flat
   (72.2 → 73.0)**. More backbone training buys *cleaner boxes* (→ better
   association), it does **not** move recall. ⇒ **Recall is not backbone-bound**
   (consistent with the occlusion-bound recall story,
   `project_recall_lever_separability`). e30 mildly over-fits back to 72.2.
2. **The fully-trained native head (73.3) beats the eager Mamba head (71.4, §4)**
   — because it now has a fully-trained backbone instead of the frozen shared one.
   Still ~5pp below the deployed 78.6, but that remaining gap is **runtime /
   preset / backbone-lineage** (whole-graph TRT + private_continuation + legacy
   engine), **not the head**.

### PR curve — does the Mamba head separate hard-real from false better? No.

Native e26 pushed down-threshold vs the Mamba head (epoch_0012):

| head @ point | Rcll | Prcn | FP | IDF1 |
|---|---:|---:|---:|---:|
| native cst0.15 | 73.0 | 97.3 | 2300 | 73.3 |
| native cst0.05 | 75.0 | 96.2 | 3365 | 72.8 |
| native cst0.02 | 75.4 | 96.1 | 3475 | 72.7 |
| mamba cst0.50 | 77.7 | 94.3 | 5261 | 71.4 |
| mamba cst0.30 | 81.8 | 90.6 | 9539 | 70.8 |

Recall-per-FP slope: **native +2.4 recall / +1175 FP (0.0020)** vs **mamba +4.1
recall / +4278 FP (0.0010)**. The native head's PR curve is **steeper = it
separates real from fake more cleanly**; the Mamba head's higher absolute recall is
a **shift to a lower-precision / higher-FP operating point** (saturation lifts real
*and* false detections together), not better separation. Pushed to its floor the
native head only reaches 96.1% precision — it simply **doesn't have the batch of
high-scored false detections** the Mamba head's saturated scores manufacture. This
confirms the score-distribution reading in §3 and `project_mamba_score_distribution`
(saturated median 0.93) / `project_fuse_score_weight_nogo` (real/ghost overlap in
score space): the Mamba head's crowd-recall edge is **overconfidence traded for
recall**, not a cleaner real/false decision boundary.

### Eval-parameter search — 73.3 is the eager ceiling, no knob moves it

Swept the eval-time association levers on native e26 (fair point cs2/cst0.15). **None
transfer**; every recall-recovery lever that helps another config fails or backfires
on the native head:

| lever | IDF1 | vs base | verdict |
|---|---:|---:|---|
| base (cst0.15) | **73.3** | — | ceiling |
| + private_continuation (deployed params) | 72.5 → 71.9 | −0.8…−1.4 | NO-GO — FP 2300→4153, AssA −1.7 |
| + relink-bridge relax (h [0.6,1.7], px 0.4) | 72.8 | −0.5 | NO-GO — the yolo26m small-box lever |
| + relink-sim 0.85 / private-min 0.05 | 73.3 | ±0 | no-op — no low-score candidates exist |
| lower confirm-thresh (→0.02) for recall | ≤72.8 | − | trades IDF1 down (see PR table) |

Why they all fail: they recover *NMS-suppressed / low-score* detections. The native
head is already at a clean, calibrated, high-precision (97%) point — its residual FN
are **genuine occlusion** (occlusion-bound), not recoverable boxes. private_continuation
helps the deployed Mamba head (+0.4) precisely because saturated scores make suppressed
crowd boxes look real; the honest native head has no such headroom. ⇒ **The eager
native ceiling is 73.3; the remaining ~5pp to the deployed 78.6 is backbone-lineage
(legacy TRT engine > our fresh replica-level backbone) + whole-graph runtime, reachable
only by a TRT export, NOT by eval params and NOT by the head.**

Driver: `scripts/eval/run_native_{full,pr,param,assoc}_search.sh`;
numbers `results/native_{full_sweep,pr_sweep,param_search,assoc_search}.txt`.

## 9. yolo26m backbone — the conclusion FLIPS (2026-07-02)

Repeated the whole study on the bigger `yolo26m` backbone (256/512/512 FPN vs
s 128/256/512). Native-m full retraining (`runs/gated_det_native_full_m`, same
recipe from `yolo26m.pt`, survived NaN wall, loss 3.93@e12 → 3.27@e26 — much lower
than s, the bigger backbone fits better).

**Native-m epoch sweep (fair point, eager pyt path):**

| epoch | IDF1 | MOTA | Rcll | Prcn | AssA | IDs | FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| m_e12 | 75.5 | 75.6 | 78.0 | 97.6 | 63.5 | 462 | 2183 |
| **m_e26** | **76.4** | 77.1 | 79.1 | 98.0 | 63.7 | 458 | 1778 |
| m_e30 | 76.0 | 76.9 | 78.6 | 98.3 | 63.5 | 432 | 1522 |

**Recall moved with capacity**: native-s recall was stuck ~73 across all epochs;
native-m recall = 79.1 (+6pp). ⇒ refine §8: recall is not improved by more
*training* of a fixed backbone, but *is* improved by more backbone *capacity*
(bigger FPN). The occlusion wall is what remains after detection capacity is maxed.

**Clean same-backbone m head A/B** (both on the m teacher's `epoch_0012_m`,
each head at its fair point):

| head | IDF1 | MOTA | HOTA | AssA | Rcll | Prcn | IDs | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **mamba-m** (cst0.50, T3→T1) | **77.4** | 79.2 | 69.3 | **68.3** | 81.6 | 97.5 | 370 | 39 |
| native-m (cst0.15) | 75.2 | 75.6 | 63.4 | 63.4 | 78.0 | 97.5 | 460 | 64 |

**On the m backbone the Mamba head WINS +2.2 IDF1 (77.4 vs 75.2) — the s tie does
NOT hold.** Breakdown: recall +3.6 (saturation, same as s) **+ AssA +4.9 (new)**.

**The AssA gap is architectural, not a training-budget artifact:**
- native-m *full* training (own 30ep backbone, e26): AssA 63.7 — identical to the
  matched 63.4. More training did **not** move native AssA.
- The +4.9 is the Mamba head's **T3→T1 temporal-consistency shaping** (cross-frame
  blocks force temporally consistent features, +3–5 AssA; `project_t3t1_temporal_consistency`).
  The native per-frame YOLO head **structurally has no temporal path**, so it cannot
  replicate this — and on the richer m features the shaping pays off much more than
  on s (s matched AssA gap was only +1.2).

**Cross-backbone summary (eager, fair points):**

| backbone | native head | mamba head | Δ (mamba−native) |
|---|---:|---:|---|
| s, same `epoch_0012` | 71.0 | 71.4 | +0.4 — **tie** |
| m, same `epoch_0012_m` | 75.2 | 77.4 | **+2.2 — mamba wins** |
| m, native own full backbone | 76.4 | (77.4) | +1.0 — mamba still ahead |

**Revised verdict**: the §6 "Mamba head has no advantage" claim was **s-specific**.
On a bigger backbone the Mamba head's two properties — saturation recall (crowd
specialist) **and** T3→T1 temporal-consistency AssA — compound into a genuine +2.2
IDF1 win, and the durable half (AssA) is a real **architectural capability** the
native head lacks. The trade-off (1.6× slower: 39 vs 64 FPS eager) stands. So:
**backbone capacity and the Mamba head's advantage scale together** — which also
explains the s tie (small-backbone features don't give the temporal shaping enough
to work with). A head-superiority claim is now *supported on m*, not on s.

**native-m eval-param search — 76.4 is the ceiling, same as s**: bridge-relax
−0.7, private_continuation −1.3, both −1.4 (all inflate FP, AssA locked ~63.7).
No eval knob closes the architectural AssA gap; the T3→T1 advantage is train-time
only. So native-m's honest ceiling is 76.4 (eager), vs mamba-m 77.4 (matched
backbone) / 79.5 (deployed TRT whole-graph + private_continuation).

Drivers: `run_native_m_sweep.sh`, `run_m_matched.sh`, `run_native_m_param.sh`;
numbers `results/native_m_sweep.txt`, `results/m_matched.txt`, `results/native_m_param.txt`.

## 7. Artifacts
- `src/saccade/perception/temporal_yolo/teacher_head_detector.py` — the control detector.
- `scripts/eval/mot17.py` `--teacher-head-ckpt` — CLI entry.
- `scripts/eval/run_teacher_head_matched_baseline.sh` — 7-seq driver.
- `results/teacher_matched_baseline_summary.txt` — aggregate numbers.
- `results/matched_{mamba_headline_cst050,teacher_fair_cst015}/` — per-seq MOT outputs.
- `runs/gated_det_native_full/` — full native retraining (§8); `scripts/eval/run_native_{full,pr}_sweep.sh`.
