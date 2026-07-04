# Cheb-GR Offline Handover — The Clean Signal, Its Use, and Basis

> Created: 2026-07-04. Corresponds to registry
> [#58](../../../reference/no_go_registry.md) and semantic TODO item 2
> ("Cheb-GR offline handover candidate-edge 統計").
>
> Purpose of this doc: harvest item 2 into **one clean signal + what it is
> for + the experiment that earns it**, so no future consumer re-derives it.

---

## 1. The clean signal

**`best_cost` (Cheb-GR k-reciprocal robust-min distance of newborn-B head
against archive-A bank) is the single durable handover-confidence
discriminator, and its reliability zones are stable across genuinely
different tracking conditions.**

Cross-condition applicability map (3 genuine conditions — see §3), same_gt
rate = fraction of candidates in the zone that are a true A→B identity:

| zone | expression | class | same_gt across 3 conds |
|---|---|---:|---|
| **veto** | `best_cost >= 0.5` | stable-veto | 0.01–0.03 |
| **veto** | `center_dist_norm >= 2` | stable-veto | 0.02–0.04 |
| **veto** | `margin <= 0.05` | stable-veto | 0.03–0.06 |
| **support** | `center_dist_norm < 0.5` | stable-support | 0.36–0.42 |
| **support** | `margin >= 0.12` | stable-support | 0.38–0.43 |
| accept | `best_cost <= 0.25` | condition-sensitive | 0.72–0.88 |
| accept | `best_cost <= 0.35 && margin >= 0.12` | condition-sensitive | 0.68–0.74 |
| accept | `margin >= 0.12 && candidate_count >= 5` | condition-sensitive | 0.61–0.75 |
| pollution | `head_tail_neighbor_iou >= 0.7` | stable-high-pollution | 0.45–0.71 |
| pollution | `match_iou ∈ [0.1,0.3)` | stable-high-pollution | 0.31–0.40 |

`best_cost` alone is the strongest single feature (AUC ≈ 0.91, monotone —
lower is more trustworthy). `margin` is second (AUC ≈ 0.83) but non-monotone
in the tail (`>= 0.3` decays: sparse-candidate degenerate). `center_dist_norm`
is a **geometry support/veto**, not identity proof. `neighbor_iou` /
`head_tail_neighbor_iou` / `match_iou` are **ReID crop-pollution context**,
not gates (AUC ≈ 0.53).

Map artifacts:
`results/diag_applicability_m_s_cs030_chebgr_offline_20260704.{md,json}`
(`synthesize_handover_applicability.py`, schema
`cheb_gr_offline_handover_summary/v1`).

---

## 2. What it is for (and what it is NOT for)

**Use A — confidence characterization / where-not-to-trust.** Any future
handover/relink/claim consumer (offline merge, occ-exit audit #55, a future
async sidecar #57) can read the zones as **sanity floors**: a candidate with
`best_cost >= 0.5` or `center_dist_norm >= 2` is ~97–99% a wrong link *in
any condition*, so it must never be auto-accepted. These floors are
condition-robust and cheap to check.

**Use B — the operating variable is `best_cost`.** Tune/threshold/rank on
`best_cost`; treat `margin` as a bounded second signal (needs an upper bound,
see §1); treat the IoU family as pollution context for **crop selection**
(link to [[clean_fifo_bank_substrate]] / #58: high `head_tail_neighbor_iou`
frames are the ones the visclean gate should skip), never as an accept gate.

**NOT for — an add-on gate on the offline handover.** This is the decisive
negative (§3, "veto ablation"): the stable-veto zones are **descriptive, not
actionable**. The handover's own `max_cost` operating point already avoids
the veto zones, so bolting an explicit veto on top removes nothing net — even
forced active in the highest-pollution regime it is net-neutral. Likewise the
accept zones are **condition-sensitive** (rates and spreads move with the
regime), so they cannot be hard-wired as a default whitelist; they are at
most a gray-zone candidate priority. **Both tracker/output-changing levers
(veto-cut, accept-tighten) are falsified across conditions — offline handover
sits at its signal frontier at this operating point.**

The value of this map is therefore *diagnostic*: it explains **why the
handover cannot be squeezed further with these features**, not a new gate.

---

## 3. Experimental basis

### 3.1 Three genuine conditions (and how genuineness was earned)

The map's stability claim requires conditions that actually shift the
handover-candidate distribution — not cosmetic reruns.

- **`--detector SDP/FRCNN/DPM` is structurally a no-op** and cannot be a
  condition: `scripts/eval/mot17.py:132-142` only selects `MOT17-XX-<suffix>`
  **directories**; those hold identical images and detections always come
  from the learned mamba detector (`mot17.py:267-299`) — public `det/det.txt`
  is never read. All three suffixes are byte-identical by construction
  (verified: `compare_detector_suffix_runs.py`). Line closed.
- **`--new-track-thresh` is a weak lever**: nt0.20 IDF1 79.9/IDs 330,
  nt0.10 79.8/316 vs nt0.28 baseline 79.5/335; distinct-id sum 763→750,
  births ±2% and **non-monotone**. Reason: the real birth control is the
  confirm gate (`confirm_score_thresh 0.50 + confirm_streak 3`);
  new_track_thresh only sizes a tentative pool that is mostly culled.
- **`--confirm-score-thresh 0.30` is the genuine 3rd condition**: distinct-id
  sum 763→**885 (+16%)**, FP 2155→**3235 (+50%)**, Rcll 83.7→86.0 — a
  higher-recall / higher-pollution regime that stresses exactly the veto and
  pollution zones.

| condition | preset knob | baseline (no handover) | handover | cand rows / accepted |
|---|---|---|---|---|
| m@cs0.50 | `mamba_whole_graph_m` | 79.5 / 335 | 80.3 / 311 | 323 / 57 |
| s@cs0.50 | `mamba_whole_graph` | 78.4 / 425 | 78.9 / 407 | 402 / 75 |
| **m@cs0.30** | `--confirm-score-thresh 0.30` | 79.9 / 411 | 80.5 / 385 | 336 / — |

Backbone (m↔s) and confirm-gate (0.50↔0.30) are orthogonal distribution
shifts; the veto/support rates above hold across all three.

### 3.2 Veto ablation — the "descriptive not actionable" proof

`center_dist_norm >= 2` is the **only** stable-veto that directly hits
*accepted* wrong-links (the other two are candidate-stage, and the handover
already avoids them). Tested in the most favorable regime (cs0.30, +50% FP):

| cs0.30 run | IDF1 | IDs |
|---|---|---|
| no-handover | 79.9 | 411 |
| handover | 80.5 | 385 |
| handover + `--cheb-gr-offline-center-dist-veto 2.0` | **80.6** | 386 |

The veto is **active** — it changed 2154 MOT output lines (a vetoed handover
flips a whole tracklet; concentrated in seq-13 1528 / seq-11 261 / seq-05 201)
— yet the metric is flat (+0.1 IDF1, +1 IDs). The accepted-and-vetoed
handovers are a wash: the handover's `max_cost` gate had already extracted the
separable value.

This confirms the frozen-substrate #58 result (`center_dist_veto=2.0` +0.06 /
noise; `pollution_veto=0.5` −0.46; `neighbor_iou_max=0.5` crop-filter −0.26)
now holds under a genuine cross-condition test, not just the m-substrate A/B.

### 3.3 Tooling & reproduction

- Log: `mot17.py … --module-lifecycle configs/modules/cheb_gr_offline_mnv4.yaml --cheb-gr-offline-log`
- Annotate + registry/summary: `scripts/eval/diagnostics/cheb_gr_offline_handover_report.py`
- Cross-run compare: `scripts/eval/diagnostics/compare_handover_summaries.py`
- Applicability map: `scripts/eval/diagnostics/synthesize_handover_applicability.py` (nargs ≥ 2 summaries)
- Detector-suffix guard: `scripts/eval/diagnostics/compare_detector_suffix_runs.py`
- Veto flags (default off): `--cheb-gr-offline-center-dist-veto` / `--cheb-gr-offline-pollution-veto` / `--cheb-gr-offline-neighbor-iou-max`
- Regression tests: `tests/unit/eval/test_cheb_gr_offline_handover_report.py`,
  `test_compare_handover_summaries.py`, `test_synthesize_handover_applicability.py`

---

## 4. One-line takeaway

`best_cost` is the durable handover-confidence signal with condition-stable
veto/support zones — **but the handover's own cost gate already spends it in
full**, so the map's job is to explain the frontier and give future consumers
condition-robust sanity floors, not to add a gate.

Related: [[chebgr-handover-applicability-map]],
[[clean_fifo_bank_substrate]], registry
[#58](../../../reference/no_go_registry.md),
[#56](../../../reference/no_go_registry.md) (live claims NO-GO),
[#55](../../../reference/no_go_registry.md) (occ-exit audit).
