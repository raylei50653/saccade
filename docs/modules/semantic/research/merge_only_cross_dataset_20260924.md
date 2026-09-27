# Merge-only identity repair — cross-dataset validation (2026-09-24)

<!-- doc-status: active -->
<!-- doc-promotion: ledger -->
<!-- doc-date: 2026-09-24 -->
<!-- doc-module: semantic -->

> **This note answers one question.** Does the existing merge-only output-layer
> repair (Cheb-GR tracklet merge, 2026-09-05 operating point) hold its identity
> gain outside MOT17? Should it be retained globally, gated conditionally, or
> parked? (#459)
>
> **This note does not claim.** It does not retune merge, interpolation,
> detector, tracker or routing. It does not change a shipping preset. It does not
> revisit handover ↔ merge chaining (#335 / #402). It does not claim that
> MOT20 results cover MOT20-03/05. It does not decompose metric deltas causally
> into merge events.

Predecessor: [output_layer_repair_chaining_revalidation_20260912.md](output_layer_repair_chaining_revalidation_20260912.md) (#402).
Machine-readable packet: [evidence/merge_only_cross_dataset_20260924/](evidence/merge_only_cross_dataset_20260924/)
(`table.json` is the summary; the per-dataset `results.json`, `merge_events.json` and `no_interp/` files hold the detail).

---

## 1. Verdict

**Gated conditionally. Not retained globally. Not promoted.**

- **Identity direction: generalizable across the three datasets.** merge-only
  raises IDF1, HOTA and AssA on MOT17, MOT20 (01+02) and DanceTrack. This holds
  with and without the shipping interpolation.
- **Net metric outcome: dataset-specific.** On DanceTrack the shipping
  interpolation, run after merge, turns a clean merge into **MOTA −3.6 /
  FP +16,712 / DetA −1.6**. MOT17 and MOT20 do not show this.
- **Feasibility: not general.** At the fixed operating point the merge stage
  cannot run on MOT20-03 or MOT20-05 (GPU OOM). On MOT20-02 it costs about
  110 ms per frame.

Retaining merge globally would ship the DanceTrack MOTA/FP regression and an
OOM on dense scenes. Parking it would discard an identity gain that appeared on
all three datasets. Merge-only therefore stays an **opt-in offline repair**,
subject to the two conditions in §6. The #459 exit label is:
**identity gain `generalizable` with named limits; net-metric outcome
`dataset-specific`**.

---

## 2. Execution identity

| Item | Value |
|---|---|
| Substrate commit | `b3ac5725` (clean `main`, after #457 closed) |
| Replay commit | `eb465b49` (`dirty=false`; measurement-only harness instrumentation `0df87c65`, event labeller `30b94e7b`) |
| Preset | `mamba_whole_graph_m`, `--double-buffer --no-gpu-decode`, `reid_mode: off` |
| Substrate | tracker output with interpolation **off**, frozen; arms are replays on the identical substrate |
| Arms | `base` (interpolation only) vs `merge_only` (merge → interpolation) |
| Merge (not retuned) | `mobilenetv4_reid` visclean engine, `max_cost=0.45`, `max_gap=60`, `n_samples=50` |
| Interpolation | shipping m: `max_gap=35`, `min_track_len=5`, `min_h=0` |
| Datasets | MOT17 train 7-seq SDP · MOT20 train **MOT20-01, MOT20-02** · DanceTrack train, all 40 sequences |
| GPU | RTX 5070 Ti Laptop (12 GB), WSL2; all GPU work under `resctl machine-bench` |

MOT20 and DanceTrack are the same external sets used for the bridge-gate check
([bridge_gate_cross_dataset_20260808.md](../../../reference/benchmarks/bridge_gate_cross_dataset_20260808.md)).
DanceTrack is outside both the detector's and the ReID model's training
provenance. The base cell reproduces that note's shipped cell exactly
(IDF1 34.474).

---

## 3. Results — shipping interpolation after the stage

| dataset | seqs | base IDF1 | ΔIDF1 | ΔHOTA | ΔAssA | ΔDetA | ΔMOTA | ΔIDs | ΔFP | per-seq IDF1 ↑/↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| MOT17 | 7 | 80.45 | **+0.83** | +0.36 | +0.69 | +0.03 | −0.06 | −31 | +540 | 3 / 4 |
| MOT20 (01+02) | 2 | 51.09 | **+1.04** | +0.58 | +0.89 | +0.13 | +0.14 | −47 | +316 | 2 / 0 |
| DanceTrack | 40 | 34.47 | **+3.48** | +1.09 | +1.87 | **−1.61** | **−3.63** | +17 | **+16,712** | 34 / 6 |

The MOT17 cells reproduce #402's merge-only row to full precision
(81.2767 / 74.7374 / 74.1728 / 313).

## 4. Localization — the same replay with interpolation off

This step was run because DanceTrack's IDF1 and MOTA moved in opposite
directions. It changes no parameter. It only removes the downstream stage.

| dataset | ΔIDF1 | ΔHOTA | ΔAssA | ΔDetA | ΔMOTA | ΔIDs | ΔFP | per-seq IDF1 ↑/↓/= |
|---|---|---|---|---|---|---|---|---|
| MOT17 | +0.74 | +0.29 | +0.57 | +0.01 | +0.04 | −43 | 0 | 5 / 2 / 0 |
| MOT20 (01+02) | +0.83 | +0.41 | +0.72 | −0.02 | +0.02 | −50 | +7 | 2 / 0 / 0 |
| DanceTrack | +3.97 | +1.49 | +1.89 | −0.34 | +0.16 | **−574** | 0 | 37 / 1 / 2 |

**Reading.** Merge by itself only relabels rows. On every dataset it is
IDF1/AssA-positive, and it reduces IDs. The DanceTrack MOTA/FP regression
appears only when interpolation runs **after** merge: interpolation fills the
gaps that merge created, including gaps across wrong joins (see §5). This is a
merge × interpolation interaction. Merge relabelling alone does not cause it.
The MOTA/FP/FN differences of 0–7 with interpolation off come from identity
changes altering the metric's frame matching.

---

## 5. Accepted-merge events

Each accepted pair is labelled against GT. A tracklet takes the majority GT id
from per-frame Hungarian matching at IoU ≥ 0.5, provided that id covers at
least 50 % of the tracklet's rows; otherwise the pair is *undetermined*. Only
first-pass accepts are labelled; transitive pairs are not expanded.

| dataset | accepted | same GT | different GT | undetermined | judged precision |
|---|---|---|---|---|---|
| MOT17 | 85 | 48 | 33 | 4 | **59 %** |
| MOT20 (01+02) | 97 | 59 | 13 | 25 | **82 %** |
| DanceTrack | 2,678 | 645 | 1,219 | 814 | **35 %** |

MOT17's 59 % agrees with #335's independent funnel reading of 60 % (39/65 direct
edges).

Medians, same-GT vs different-GT:

| dataset | cost | gap (frames) | shorter track length | CV motion residual (÷ box h) |
|---|---|---|---|---|
| MOT17 | 0.356 vs 0.401 | 16 vs 28 | 5 vs 4 | 0.62 vs 0.84 |
| MOT20 | 0.385 vs 0.430 | 20 vs 12 | 4 vs 1 | 0.31 vs 0.35 |
| DanceTrack | 0.415 vs 0.425 | 19 vs 19 | 1 vs 1 | **0.42 vs 1.02** |

Judged precision by cost band:

| cost | MOT17 | MOT20 | DanceTrack |
|---|---|---|---|
| < 0.30 | 0.79 (14) | 1.00 (11) | 0.75 (87) |
| 0.30–0.37 | 0.70 (23) | 0.93 (15) | 0.42 (210) |
| 0.37–0.42 | 0.63 (27) | 0.74 (19) | 0.36 (576) |
| 0.42–0.46 | **0.24** (17) | 0.74 (27) | **0.28** (991) |

Observations. These are descriptive; no gate is proposed.

- **DanceTrack merges are mostly single-row fragments** (median shorter length 1).
  The appearance cost barely separates correct from wrong joins there
  (0.415 vs 0.425). This fits DanceTrack's uniform appearance. The IDF1 gain
  coexists with 65 % wrong judged joins. The wrong joins are mostly one-row
  fragments, so each costs little identity. This is **not** a causal
  decomposition of +3.5.
- **The motion residual separates correct from wrong joins on DanceTrack and
  MOT17, but not on MOT20.** It is the first candidate variable if localization
  is ever reopened. It is not a validated gate.
- **The high-cost band 0.42–0.46 is where wrong joins concentrate** on MOT17
  and DanceTrack. MOT20 stays at 0.74 even there.
- **MOT17's gain is concentrated.** With interpolation on, 10 (+5.17) and
  02 (+2.56) carry it; 11 (−0.69, 3/3 joins wrong), 05 (−0.60), 13 (−0.16) and
  04 (−0.01) lose. Per-sequence sign follows the per-sequence join precision.

---

## 6. Failure modes and conditions

| # | failure mode | dataset | evidence | condition it implies |
|---|---|---|---|---|
| F1 | **Merge × shipping interpolation creates FPs** | DanceTrack | MOTA −3.63, FP +16,712 with interpolation on; MOTA +0.16, FP 0 with it off | do not run the shipping interpolation over merge-created gaps where join precision is low |
| F2 | **Dense-scene infeasibility (OOM)** | MOT20-03, MOT20-05 | `mot20/full_set_oom_excerpt.txt`: sample-level Cheb-GR builds dense S×S matrices; S≈37.8k (03) and 67.7k (05) samples vs 26.1k for MOT20-02, which ran | cap the per-sequence sample count, or skip merge, until the implementation is sparse |
| F3 | **Quadratic runtime** | MOT20-02 | merge stage 308 s for 2,782 frames | offline only; see §7 |
| F4 | **Frame-name portability** | DanceTrack | `dancetrack/8digit_filename_failure_excerpt.txt`: merge extractor hardcodes `%06d`; DanceTrack uses `%08d` | the replay used a symlink mirror; the production surface cannot read DanceTrack as-is |
| F5 | **Low join precision out of domain** | DanceTrack | 35 % judged precision | identity gain survives, but correctness of individual joins does not |

Conditions for any future retention beyond opt-in offline use: **F1** (the
interpolation interaction) and **F2** (feasibility) must be resolved first.
F3 and F4 are engineering items. Their fixes (exact sparse / temporally
pre-gated computation, frame-name discovery) are **not** part of this
validation and remain unscheduled by owner decision (2026-09-24).

---

## 7. Runtime

Merge is an offline output-layer stage. It does not touch the tracker loop, so
tracker FPS is unaffected by construction. Its own cost is reported as host wall
time per frame processed (median of 3 in-process repeats; the first repeat
includes engine warm-up).

| dataset | frames | tracker capture FPS (substrate, serial) | merge stage median (extract + merge) | ms / frame |
|---|---|---|---|---|
| MOT17 | 5,316 | 138.4 | 56.2 s + 18.0 s | ≈ 14 |
| MOT20 (01+02) | 3,211 | 111.7 (all 4 seqs) | 55.6 s + 300.2 s | ≈ 110 |
| DanceTrack | 41,796 | 147.0 | 187.8 s + 99.7 s | ≈ 7 |

The MOT20 cost is dominated by the merge step itself, not by crop extraction.
This is consistent with pairwise cost growing with tracklet count (1,084 on
MOT20-02). The attribution to the pairwise loop versus the dense re-ranking was
not profiled.

## 8. Determinism

- **Tracker substrate, cross-process:** MOT20 4/4 and DanceTrack 40/40 files
  byte-identical across two independent captures. MOT17 7/7 byte-identical to
  #402's 2026-09-12 substrate (`85b95ac1`).
- **Replay, in-process:** base and merge_only n=3 each on all three datasets;
  MOT outputs byte-identical within each arm (IDF1 range 0.0).
- Interpolation-off decomposition: n=1. It runs on the same deterministic
  substrate and harness.

---

## 9. What would change this verdict

- MOT20-03/05 results after an exact sparse implementation. Currently only
  2 of the 4 MOT20 sequences are covered.
- Any evidence that the interpolation interaction can be removed without
  retuning (e.g. interpolation not bridging merge-created gaps), measured on all
  three datasets.
- A fourth domain. Evidence exists for crowded pedestrian and dance scenes;
  sports is untested.
