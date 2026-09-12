# Output-layer identity repair chaining — historical complementarity probe (2026-09-05)

<!-- doc-status: active -->
<!-- doc-promotion: ledger -->
<!-- doc-date: 2026-09-05 -->
<!-- doc-module: semantic -->

> **This note answers one question.** On the 2026-09-05 tracker, did stacking
> the two output-layer identity repairs (offline handover and Cheb-GR tracklet
> merge) beat merge-only, in either order?
>
> **This note does not claim.** It does not claim that current `main` still
> produces the same numbers. It does not promote chaining as a production path.
> It does not treat extra links, extra events, or a 0.1-level order gap as
> accuracy evidence. It does not reopen the merge-only adoption decision.

Experiment record: [PR #335](https://github.com/raylei50653/saccade/pull/335)
(`feat/postproc-chain-order`). Measurement code: `f60e814b` (chaining) and
`668bbd56` (merge decision log). Then-`main` parent: `1d620127` (2026-09-03,
#333). Filed here on 2026-09-12 so the result survives without merging that
branch.

---

## 1. What was measured

The two repairs had been mutually exclusive (`elif` behind the handover
branch). #335 made the order an explicit experiment variable so they could be
run in series, each stage re-extracting embeddings from the previous stage's
lines.

| Item | Value |
|---|---|
| Date | 2026-09-05 |
| Then baseline | `1d620127` + shipping `reid_mode: off` |
| Preset | `mamba_whole_graph_m` |
| Benchmark | MOT17 train-half, 7-seq **SDP** |
| Protocol | `--double-buffer --no-gpu-decode` |
| Merge operating point | `cheb_gr_merge_enabled`, `mobilenetv4_reid`, `max_cost=0.45` |
| Handover operating point | `cheb_gr_offline_handover`, `mobilenetv4_reid`, `min_head=2`, `margin=0.05` |

The PR reports that three earlier repeats × three arms on this
`--no-gpu-decode` family were ±0.00 on every quality cell. That is a
historical determinism note for *that* then-current configuration. It is not
a general `--no-gpu-decode` guarantee (see
[nogpudecode_reproducibility_20260907.md](../../../research/eval/nogpudecode_reproducibility_20260907.md))
and it was **not** re-checked on current HEAD.

---

## 2. Historical observation

| arm | IDF1 | MOTA | HOTA | AssA | IDs | final tracks | link pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 80.4 | 81.6 | 74.4 | 73.5 | 344 | 804 | 0 |
| handover only | 80.8 | 81.5 | 74.5 | 73.7 | 338 | 750 | 59 |
| **merge only** | **81.3** | 81.6 | **74.7** | **74.2** | **313** | 719 | 114 |
| handover → merge | 81.1 | 81.4 | 74.6 | 73.9 | 323 | 704 | 127 |
| merge → handover | 81.2 | 81.4 | 74.7 | 74.1 | 317 | 700 | 133 |

Second-stage link counts from the per-stage records (still historical):

- after handover (59 links), merge found **68** where alone it found 114
- after merge (114 links), handover found **19** where alone it found 59

The second stage kept finding links in both orders, and found fewer than it
did alone. That drop is **not** a measure of opportunity the first stage
removed: the first stage changes the candidate units, and interpolation
downstream changes the rows again.

Event counts were also recorded (54 handovers vs 85 merges; 59 vs 114 link
pairs over 49 vs 62 components). **Event counts are not a complementarity
metric** and are not used as one here.

FPS was 265–277 against a 270 base. That supports only "no visible front-end
throughput change", not a claim about post-process wall time.

---

## 3. What this supports — and only this

On that 2026-09-05 snapshot:

1. **Neither chaining order beat merge-only** on IDF1 / HOTA / AssA / IDs.
2. **More links did not mean better identity metrics.** Both chained arms
   applied more links (127 / 133 vs 114) and finished below merge-only.
3. **The 0.1 gap between the two orders is not an order effect.** It is not
   claimed as a difference.

These are historical observations. They are **not** restated as current-HEAD
production numbers. Current `main` has moved ~170 commits since the
then-parent; the values above have not been re-measured.

---

## 4. What was not carried forward

| #335 piece | Disposition | Why |
|---|---|---|
| MOT17 table above | **this note** | historical research result |
| `cheb_gr_merge_output_tracklets(..., decision_log=)` | **forward-ported** as an opt-in diagnostics surface | distinguishes `has_embedding` / `no_embedding` / `reject_cost` / `reject_temporal` / `reject_same_component` / `reject_component_overlap` / `accepted`; default `None` is a no-op |
| `--cheb-gr-postproc-order` `handover_then_merge` / `merge_then_handover` | **not ported** | the experiment is complete; current main has no remaining research use that needs this runtime entry |
| chain YAML modules | **not ported** | temporary experiment configs |
| per-stage dump / recovered link-record machinery in `evaluator.py` | **not ported** | existed only to make the two orders comparable |
| `math_model.md` §12 transcription | **not ported** | attestation churn; the stages were already in the code, and ADR 022 / #334 reframed that governance problem |

Default runtime output is unchanged: evaluator and cpp_runner still do not
pass a merge `decision_log`, and they still do not chain the two repairs.

This note does **not** change the merge-only adoption decision, Cheb-GR
thresholds, or any shipping preset.
