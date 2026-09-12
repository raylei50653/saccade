# Output-layer identity repair chaining — current-head revalidation (2026-09-12)

<!-- doc-status: active -->
<!-- doc-promotion: ledger -->
<!-- doc-date: 2026-09-12 -->
<!-- doc-module: semantic -->

> **This note answers one question.** On current `main` (`85b95ac1`), do the
> five output-layer identity-repair arms still rank as they did on 2026-09-05:
> is merge-only best, and does either chaining order beat it?
>
> **This note does not claim.** It does not overwrite the 2026-09-05 historical
> table. It does not restore `--cheb-gr-postproc-order` as a production runtime
> surface. It does not promote merge, handover, or chaining into a shipping
> preset. It does not treat extra accepted links as an accuracy metric.

Historical companion (unchanged):
[output_layer_repair_chaining_20260905.md](output_layer_repair_chaining_20260905.md).

Machine-readable packet:
[evidence/output_layer_repair_chaining_revalidation_20260912/](evidence/output_layer_repair_chaining_revalidation_20260912/).

---

## 1. Question

On current HEAD, with one frozen MOT17 7-seq SDP tracker substrate and the
2026-09-05 operating points, do these five arms keep the same *directional*
conclusion?

1. base (interpolation only)
2. handover only
3. merge only
4. handover → merge
5. merge → handover

Absolute cells from 2026-09-05 are **not** a time series to subtract against.
Compare ranking, sign of effect, and qualitative conclusion.

---

## 2. Current-head execution identity

| Item | Value |
|---|---|
| Measurement date | 2026-09-12 |
| Substrate commit | `85b95ac1` (clean `main`; merge of #401; tracker capture `dirty=0`) |
| Replay harness commit | `7f7edcbe` (measurement-only harness; worktree `dirty=false` at replay start) |
| Harness | `scripts/eval/experiments/run_output_layer_repair_chaining.py` |
| Host | `DESKTOP-0FLA6SQ` (Linux WSL2) |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU, sm_120 |
| Torch / CUDA | 2.11.0+cu130 / 13.0 |
| Preset | `mamba_whole_graph_m` |
| ReID (tracker) | `reid_mode: off` |
| Detector | MOT17 train-half, 7-seq **SDP** |
| Decode | `--no-gpu-decode` (CPU JPEG). Shipping headline default is GPU decode; this run follows the current per-config reproducibility contract for this preset, not the 2026-09-05 slogan “`--no-gpu-decode` ⇒ deterministic”. |
| Scheduling | `--double-buffer` (event barrier; whole-graph + CUDA graph + tracker graph + `main_nms_graphed`) |
| Interpolation | **off during tracker capture**; applied **after** identity stages with shipping `mamba_whole_graph_m` knobs (`max_gap=35`, `min_track_len=5`, `min_h=0`) |
| Merge | `mobilenetv4_reid`, `max_cost=0.45`, `max_gap=60`, `n_samples=50` |
| Handover | `mobilenetv4_reid`, `min_head=2`, `margin=0.05`, `max_cost=0.45`, `decide_n=5` |
| Embedding engine | `models/embedding/mobilenetv4_reid_visclean_224.engine` |
| Quality filter | shipping no-op (`min_tracklet_len=1`, `min_tracklet_score=0`) |

Published runtime-identity coordinate was **not** republished. This work did not
modify `decision_relevant` / `identity_semantics` production paths; it added an
isolated research harness. The coordinate on disk at measurement time:

| Axis | Digest (prefix) |
|---|---|
| implementation | `a92f241943539ba1…` |
| decision_surface | `9b7faeb0f76a4348…` |
| identity_semantics | `03ec01c5374061b7…` |
| environment | `85f5d180e089f4f5…` |
| runtime_inputs | `0b839df0b8914195…` |

Source: [runtime_identity.generated.json](../../../reference/runtime_identity.generated.json).
Witness head of that publication is older than `85b95ac1`; it is recorded here
as the then-current published coordinate, not as a claim that this measurement
was the identity probe.

Substrate MOT SHA-256:
[substrate_sha256.json](evidence/output_layer_repair_chaining_revalidation_20260912/substrate_sha256.json).
Exact commands:
[commands.md](evidence/output_layer_repair_chaining_revalidation_20260912/commands.md).

---

## 3. Protocol

Current production still runs the two repairs as **mutually exclusive**
(`if handover: … elif merge: …` in `evaluator.py` / `cpp_runner.py`). There is
no `--cheb-gr-postproc-order`. That is the same production shape #401 left in
place.

Chaining was therefore **not** re-introduced as a config flag. The harness
replays both stages on a frozen pre-interpolation MOT substrate:

1. Capture tracker output with merge off, handover off, interpolation off.
2. For each arm, apply the named stages **in order**. Each stage re-crops
   `img1` and rebuilds embeddings from the **previous stage's lines**.
3. Apply shipping interpolation once, after the identity stages.
4. Score with project motmetrics (IDF1/MOTA/IDs, full precision) and vendored
   TrackEval (HOTA/AssA).

This is the #335 chaining semantics (sequential, re-extract, interpolate
downstream). It is not “both stages scoring the original track set at once”.
No shipping preset, production config key, or evaluator default dispatch was
changed.

Reproducibility: do **not** inherit “`--no-gpu-decode` ⇒ bit-exact”. The
current contract is
[nogpudecode_reproducibility_20260907.md](../../../research/eval/nogpudecode_reproducibility_20260907.md)
block D: `mamba_whole_graph_m --double-buffer --no-gpu-decode` 7-seq, n=20,
0 divergences. That licenses N=1 for the **tracker substrate** as exploratory
comparison. It does **not** license the treatment arms. Those were repeated.

---

## 4. Five-arm result table

Display cells are 1 decimal, same convention as the historical note. Full
precision is in
[results.json](evidence/output_layer_repair_chaining_revalidation_20260912/results.json).

| arm | IDF1 | MOTA | HOTA | AssA | IDs | final tracks | accepted links |
| --- | ---: | ---: | ---: | ---: | --: | -----------: | -------------: |
| base | 80.4 | 81.6 | 74.4 | 73.5 | 344 | 804 | 0 |
| handover only | 80.8 | 81.5 | 74.5 | 73.7 | 338 | 750 | 54 |
| **merge only** | **81.3** | 81.6 | **74.7** | **74.2** | **313** | 719 | 85 |
| handover → merge | 81.1 | 81.4 | 74.6 | 73.9 | 323 | 704 | 100 |
| merge → handover | 81.2 | 81.4 | 74.7 | 74.1 | 317 | 700 | 104 |

Full precision (percentage points):

| arm | IDF1 | MOTA | HOTA | AssA | IDs |
| --- | ---: | ---: | ---: | ---: | --: |
| base | 80.4471 | 81.6104 | 74.3771 | 73.4854 | 344 |
| handover only | 80.7671 | 81.5293 | 74.5122 | 73.7218 | 338 |
| merge only | 81.2767 | 81.5552 | 74.7374 | 74.1728 | 313 |
| handover → merge | 81.0846 | 81.4225 | 74.5767 | 73.9362 | 323 |
| merge → handover | 81.2365 | 81.4465 | 74.6806 | 74.1224 | 317 |

`accepted links` = sum of stage accepts (`handovers` + `merges`). That matches
the 2026-09-05 *event* counts (54 handovers / 85 merges), **not** that note's
`link pairs` column (59 / 114 / 127 / 133), which used a different count.
Do not subtract the two link columns.

---

## 5. Diagnostics

These counts are observability, **not** accuracy metrics.

| arm | stage | accepted | no_embedding | reject_cost | reject_temporal | reject_same_component | reject_component_overlap | components |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| handover only | handover | 54 | 0 | 253 | — | — | — | — |
| merge only | merge | 85 | 124 | 36005 | 235 | 6 | 33 | 62 |
| handover → merge | handover then merge | 54 then 46 | 0 / 124 | 253 / 30183 | — / 197 | — / 2 | — / 10 | — / 32 |
| merge → handover | merge then handover | 85 then 19 | 124 / 0 | 36005 / 246 | 235 / — | 6 / — | 33 / — | 62 / — |

Second stage still finds accepts in both orders, and fewer than it found
alone (merge 85→46 after handover; handover 54→19 after merge). That drop is
**not** leftover opportunity the first stage “stole”: the first stage changes
the candidate units, and interpolation still runs afterwards.

---

## 6. Repeatability / uncertainty

Tracker substrate: one capture under the block-D contract above (n=20, 0/20
divergences for this preset + `--double-buffer` + `--no-gpu-decode`).

Treatment arms: n=3 **within-process** replays from the same frozen MOT files
for `merge_only`, `handover_then_merge`, and `merge_then_handover`. One
`TRTFeatureExtractor` is built once and reused; this is not three fresh
processes.

| arm | n | MOT files identical | IDF1 range | HOTA range | IDs range |
| --- | --: | --- | ---: | ---: | ---: |
| merge only | 3 | yes | 0 | 0 | 0 |
| handover → merge | 3 | yes | 0 | 0 | 0 |
| merge → handover | 3 | yes | 0 | 0 | 0 |

Observed within-process replay range on this substrate is **0** (bit-identical
MOT outputs). That is a per-arm observation for this frozen input + one shared
extractor, not a general determinism proof and not three independent processes.

The 0.1-level 1-decimal gap between the two orders is 0.152 IDF1 in full
precision (`81.2365 − 81.0846`). That gap is larger than the observed
within-process replay range. It is still **not** used as a production order
effect: neither order beats merge-only.

---

## 7. Comparison with 2026-09-05

Compare ranking and sign, not absolute drift-as-regression.

| Claim from 2026-09-05 | Current HEAD |
|---|---|
| merge-only best on IDF1 / HOTA / AssA / IDs | **same** |
| handover-only positive vs base on IDF1 / IDs | **same** (IDF1 +0.32, IDs −6) |
| neither chaining order beats merge-only | **same** |
| more accepts ≠ better identity metrics | **same** (100 / 104 accepts, both below merge-only) |
| 0.1-level order gap is not a restack reason | **same** |

At 1 decimal, the quality cells and `final tracks` / IDs **match** the
historical table. That is a directional-and-magnitude replication of the
reported quality table. It is **not** a claim that the two tracker binaries
were bit-identical, and it is not a license to splice the two tables into one
time series. Decode path, commits, and link-count definitions differ; only
the quality ranking is being compared.

Second-stage handover after merge still found **19** accepts, the same number
the historical note recorded. Merge-after-handover found 46 here vs 68 there;
that column used different link counting, so the 46 vs 68 is not interpreted
as a merge-rate regression.

---

## 8. Supported conclusions

On this current-head measurement contract:

1. **Merge-only still beats base** on IDF1 (+0.83), HOTA (+0.36), AssA (+0.69),
   IDs (−31). MOTA is slightly below base in full precision (−0.055) and ties
   at 1 decimal (both 81.6), matching the historical MOTA picture.
2. **Handover-only still has a positive identity gain** vs base (IDF1 +0.32,
   IDs −6, HOTA +0.14, AssA +0.24) with a small MOTA loss (−0.08).
3. **Neither chaining order beats merge-only** on IDF1 / HOTA / AssA / IDs.
4. **More accepted links still do not become better identity metrics.** Both
   chained arms applied more accepts (100 / 104 vs 85) and finished below
   merge-only.
5. **The two orders differ by 0.15 IDF1**, which is larger than the observed
   within-process replay range of 0, and still irrelevant to the restack
   question because both lose to merge-only.
6. **The 2026-09-05 directional conclusion replicates** on current HEAD.

---

## 9. Unsupported conclusions

- Not a promotion of merge, handover, or chaining into shipping config.
- Not a claim that extra links, extra events, or extra components are quality.
- Not a claim that order is a useful production variable.
- Not a claim that `--no-gpu-decode` is generally deterministic.
- Not a GPU-decode headline number. Shipping default decode is GPU; this
  revalidation used the current reproducibility-contracted decode for this
  preset.
- Not a bit-exact identity between 2026-09-05 tracker output and this
  substrate, even though the 1-decimal quality table matches.
- Not a merge-gate for #401. That PR already filed the historical observation.

---

## 10. Decision / follow-up

**Do not restack the two output-layer repairs.** Current HEAD still says
merge-only is the better of these five arms; chaining still spends extra
accepts for a worse identity table.

No production runtime surface is added. Thresholds were not tuned. The
historical 2026-09-05 note stays the snapshot of that day.

Follow-up that this note does **not** open: locating why the 1-decimal quality
table matched the 2026-09-05 snapshot so closely, or re-running the same arms
on GPU-decode headline identity. Those are separate measurements if someone
needs them. They are not required to answer the ranking question above.
