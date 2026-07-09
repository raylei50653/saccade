# #55 occ-exit audit — WP3 promotion decision (2026-07-09)

**Branch:** `research/occ-exit-audit-p55-wp3-report`  
**Depends on:** WP1a #74 · WP1b #75 · WP2 #76  
**Objective:** **RESEARCH** evidence + promotion decision only  

---

## Decision (headline)

| Field | Value |
|:--|:--|
| **Promotion** | **`split_feat_pr`** |
| Runtime gate in this PR | **No** (`gate_implemented: false`) |
| Production preset / headline YAML | **Unchanged** |
| Default-on sequence conditioning | **No** |

**Rationale (short):** On the frozen `mamba_whole_graph_m` no-ReID substrate, cosine occ-exit audit (bank reference + Cheb-GR **log-only** probe) is **net negative** (ΔIDF1 **−0.27pp**, ΔIDs **+47**). WP2 labels give **1** `enable_candidate` (MOT17-11) vs **4** `harmful` and **2** `abstain`. That is a local positive signal, not a safe default-off allowlist of ≥2 clean enables with isolatable harm. Any future gate belongs in a separate **`feat/`** PR with explicit design + A/B, not a silent research merge.

**Explicit:** WP3 does **not** implement `occ_audit_seq_allowlist`, does not change lifecycle/evaluator behavior, and does not flip any sequence default-on.

---

## Method

### Why frozen substrate (not live tracker re-run)

Same protocol as #55 offline ablations and the bank-reference probe:

- Control = frozen MOT txts (identical tracker output, zero GPU-decode nondeterminism).  
- Treatment = post-process audit only (line rewrite + decision log).  
- Score both with the same motmetrics call.

This answers “does cosine audit help, and where does Cheb-GR disagree?” without confounding tracker noise.

### Control

```text
results/diag_m_no_reid_current_20260704
preset lineage: mamba_whole_graph_m, reid off
MOT17 train SDP ×7
```

### Treatment

Same substrate, offline:

```bash
.venv/bin/python scripts/eval/diagnostics/run_occ_audit_wp3_promotion.py \
  --substrate results/diag_m_no_reid_current_20260704 \
  --out-dir results/occ_exit_p55_wp3 \
  --tau 0.45 --ref-n 5 --min-ref 2 --audit-crops 3 \
  --audit-window 30 --min-occ 2 --cov 0.4 --bank-n 20 \
  --chebgr-max-cost 0.45 --chebgr-margin 0.0 \
  --engine models/embedding/mobilenetv4_reid_visclean_224.engine \
  --model-type mobilenetv4_reid
```

Treatment path per sequence:

1. `build_filled_bank` (visclean front-occ FIFO, `bank_n=20`)  
2. `extract_audit_embeddings_post_exit`  
3. `occ_exit_audit_lines_from_bank(..., enabled=True, chebgr_probe=True, decision_log=...)`  
4. Write MOT txts + `_occ_audit.csv`

**Contract reminder:** `chebgr_probe=True` only adds decision-log columns.  
**Cuts / `flag_frame` / rewritten lines follow the cosine path only.**  
Metric deltas therefore measure **cosine occ-exit audit**, while `flag_delta` measures **cosine vs Cheb-GR agreement**.

### Artifacts (local; `results/` is gitignored)

| Path | Content |
|:--|:--|
| `results/occ_exit_p55_wp3/treatment/` | treatment MOT + `_occ_audit.csv` (152 rows) |
| `results/occ_exit_p55_wp3/occ_audit_metrics.json` | control/treatment IDF1·IDs + deltas |
| `results/occ_exit_p55_wp3/occ_audit_seq_applicability.{json,md}` | WP2 map |
| `results/occ_exit_p55_wp3/wp3_summary.json` | aggregate + promotion + per-seq notes |

Regenerate anytime with the command above (`--skip-extract` reuses treatment txt/csv).

---

## Aggregate metrics

| | IDF1 | IDs |
|:--|--:|--:|
| Control (substrate) | **79.51** | **335** |
| Treatment (cosine audit + chebgr log) | **79.25** | **382** |
| Δ (treatment − control) | **−0.27 pp** | **+47** |

Net tradeoff is **worse identity** (more switches) with a small IDF1 drop — consistent with historical #55 NO-GO (global audit bipolar / net harm on crowded sequences).

---

## Per-sequence metrics + flag_delta

| seq | type | ΔIDF1 (pp) | ΔIDs | audited | cos_flags | chebgr_flags | same | cos_only | chebgr_only | WP2 rec |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| MOT17-02-SDP | crowded_static | −0.44 | +12 | 28 | 14 | 12 | 24 | 3 | 1 | `harmful` |
| MOT17-04-SDP | crowded_static | −0.28 | +2 | 8 | 2 | 2 | 8 | 0 | 0 | `abstain` |
| MOT17-05-SDP | moving_low | **+0.21** | +12 | 25 | 12 | 12 | 21 | 2 | 2 | `abstain` |
| MOT17-09-SDP | static_low | −0.51 | +3 | 8 | 3 | 5 | 6 | 0 | 2 | `harmful` |
| MOT17-10-SDP | moving | −0.41 | +9 | 30 | 11 | 9 | 26 | 3 | 1 | `harmful` |
| MOT17-11-SDP | indoor_static | **+0.18** | +2 | 17 | 5 | 6 | 16 | 0 | 1 | **`enable_candidate`** |
| MOT17-13-SDP | moving_night | −0.36 | +7 | 36 | 16 | 14 | 22 | 8 | 6 | `harmful` |

**Global flag_delta (152 audited episodes):**  
`same` 123 · `cosine_only` 16 · `chebgr_only` 13  

Cheb-GR and cosine **mostly agree** (~81% same). Disagreements are not rare noise-only: 13 `chebgr_only` / 16 `cosine_only` across the set, concentrated more on 13 (8 cos_only / 6 chebgr_only).

---

## Per-sequence rationale

### `enable_candidate`

- **MOT17-11-SDP** — small positive IDF1 (+0.18), IDs not material-worse (+2), 17 audited, useful flags ≥2, high agreement (`same` 16). Only sequence that clears the WP2 enable bar.

### `abstain`

- **MOT17-04-SDP** — mild IDF1 drop (−0.28) near harm band, only 8 audited / 2 flags; low mass, no clear enable.  
- **MOT17-05-SDP** — IDF1 up (+0.21) but IDs **+12** (material fragmentation); mixed tradeoff, not enable.

### `harmful`

- **MOT17-02-SDP** — IDF1 −0.44, IDs +12; classic crowded static penalty.  
- **MOT17-09-SDP** — IDF1 −0.51; few audits but clear score loss; slight chebgr_only excess (2).  
- **MOT17-10-SDP** — IDF1 −0.41, IDs +9.  
- **MOT17-13-SDP** — IDF1 −0.36, IDs +7; largest disagreement mass (cos_only 8 / chebgr_only 6) — Cheb-GR does **not** cleanly “fix” the harm; it mostly tracks cosine cuts.

### `insufficient_evidence`

- None (all 7 seqs had metrics + ≥5 audited under WP2 thresholds).

---

## WP2 recommendation counts

| recommendation | n |
|:--|--:|
| enable_candidate | 1 |
| abstain | 2 |
| harmful | 4 |
| insufficient_evidence | 0 |

Scene-type rollup: harm hits `crowded_static` (02), `static_low` (09), `moving` (10), `moving_night` (13); only `indoor_static` (11) enables.

---

## Promotion criteria application

| Rule | Required | Observed |
|:--|:--|:--|
| `promote_default_off_gate` | ≥2 enable_candidate, harmful isolatable, aggregate not clearly worse | **1** enable; **4** harmful; aggregate **−0.27 / +47** → **fail** |
| `split_feat_pr` | local strong signal needing gate design validation | **1** enable + multi-seq harm + bad aggregate → **match** |
| `research_only` | mixed thin map, keep tools | Would fit if no enable and no clear no_go; weaker than split given 11’s local win |
| `no_go` | multi-seq clear harm **and** no enable | Harm multi-seq yes, but **one** enable exists → prefer split over hard no_go |

### Why not `no_go`?

Historical #55 already called global audit NO-GO. WP3 **confirms net negative** on this substrate, but also shows:

1. A **real** positive seq (11) under the same knobs.  
2. Cheb-GR probe is **usable log infrastructure** (columns present, mostly consistent with cosine).  
3. WP2 tooling correctly isolates harm vs enable without a runtime gate.

Killing the research line would discard (1)–(3). The right next engineering step is a **narrow feat PR**, not more default-off probe-only merges that pretend to gate.

### Why not `promote_default_off_gate`?

Allowlisting only MOT17-11 would satisfy “isolatable harm,” but the written bar was **≥2** enable_candidates and aggregate not clearly worse. Aggregate fails hard (IDs +47). Shipping a 1-seq allowlist as “default-off gate” without feat-level validation would re-create #55’s premature promotion pattern.

---

## What this does **not** authorize

- No `occ_audit_seq_allowlist` / lifecycle flag  
- No evaluator sequence gate  
- No `mamba_whole_graph*.yaml` change  
- No live critical-path audit  
- No claim that Cheb-GR graph cuts are GO (probe is still log-only; metrics are cosine cuts)

---

## Recommended next steps (outside WP3)

If/when a feat PR is opened (`feat/occ-exit-conditional-audit`):

1. **Default-off** allowlist experiment: e.g. enable audit only on MOT17-11 (and maybe re-check 05 with IDs-aware knobs).  
2. Measure **allowlist-only** aggregate vs full audit vs control on the same frozen substrate.  
3. Optionally add a **second treatment** that applies cuts from `chebgr_flag` (not log-only) for a true graph-decision A/B — that is a **new** experiment, not WP3.  
4. Do **not** promote global `--occ-audit` on headline presets.

If no capacity for feat work: keep WP1–WP2 tools as research instrumentation; leave #55 as registry NO-GO for global audit; status = **research_only tooling + split_feat recommendation**.

---

## Verification

```bash
# classifier + promotion unit tests
.venv/bin/pytest -q tests/unit/eval/test_occ_audit_seq_conditioning.py

# regenerate evidence (GPU + TRT engine required)
.venv/bin/python scripts/eval/diagnostics/run_occ_audit_wp3_promotion.py \
  --substrate results/diag_m_no_reid_current_20260704 \
  --out-dir results/occ_exit_p55_wp3
```

Related:  
[WP2 report](occ_exit_audit_p55_wp2_seq_conditioning_20260709.md) ·  
[scope](occ_exit_audit_p55_scope_20260709.md) ·  
registry [#55](../../../reference/no_go_registry.md#55)
