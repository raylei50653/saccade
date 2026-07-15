# #55 occ-exit audit — WP2 sequence conditioning (2026-07-09)

**Branch:** `research/occ-exit-audit-p55-wp2-seq-conditioning`  
**Depends on:** WP1a Cheb-GR probe core (#74) · WP1b config/evaluator wiring (#75)  
**Objective (O1):** **RESEARCH** + **DEBUG** — applicability map only  

---

## Intent

```text
PR intent:
  semantic / RESEARCH + DEBUG
  per-seq / scene-type applicability for occ-exit audit
  NO sequence gate implementation
  NO production preset / live path promotion
```

WP1 shipped a default-off Cheb-GR graph decision **probe** that logs
`chebgr_*` + `flag_delta` next to cosine decisions without changing
`flag_frame` / cuts / `stats.flags`.

WP2 turns those logs (plus optional metric pairs) into a **recommendation
table**:

| recommendation | meaning |
|:--|:--|
| `enable_candidate` | metrics non-negative / positive, useful flags not one-offs |
| `abstain` | mixed signal, noise band, or low useful-flag mass |
| `harmful` | clear metric harm, or chebgr_only domination without benefit |
| `insufficient_evidence` | missing CSV / too few audits / no metric pair |

**This PR does not wire any of these labels into evaluator gating.**

---

## Deliverables

| Piece | Path |
|:--|:--|
| Pure classifier / aggregator | `src/saccade/perception/eval/occ_audit_seq_conditioning.py` |
| CLI diagnostic | `scripts/eval/diagnostics/analyze_occ_audit_seq_conditioning.py` |
| Unit tests | `tests/unit/eval/test_occ_audit_seq_conditioning.py` |
| This report | `docs/experiments/occ_exit_audit_p55/wp2_seq_conditioning.md` |

---

## Classification rules (conservative)

Defaults (`Thresholds`):

| knob | default | role |
|:--|--:|:--|
| `min_audited` | 5 | below → `insufficient_evidence` |
| `min_useful_flags` | 2 | one-off flags cannot become `enable_candidate` |
| `idf1_noise_pp` | 0.15 | \|ΔIDF1\| inside band treated as noise |
| `idf1_harm_pp` | 0.30 | ΔIDF1 ≤ −this → `harmful` |
| `ids_material` | 5 | material IDs change (lower IDs is better) |
| `chebgr_only_domination` | 0.70 | share of disagreements that are `chebgr_only` |

Logic sketch:

```text
if audited < min_audited:
    insufficient_evidence
elif no metric pair:
    if chebgr_only dominates disagreements and count ≥ min_useful_flags:
        harmful   # suspicious probe mass without evidence of benefit
    else:
        insufficient_evidence
elif ΔIDF1 ≤ -idf1_harm_pp OR (ΔIDs ≥ ids_material and ΔIDF1 not clearly positive):
    harmful
elif chebgr_only dominates AND no metric upside:
    harmful
elif ΔIDF1 ≥ 0 AND ΔIDs not material-worse AND useful_flags ≥ min:
    enable_candidate
else:
    abstain
```

`flag_delta` values (from WP1 probe log):

- `same` — cosine and Cheb-GR agree  
- `cosine_only` — cosine flags, Cheb-GR does not  
- `chebgr_only` — Cheb-GR flags, cosine does not  

Probe-off logs (no `chebgr_*` columns) are **tolerated**: cosine tallies
still aggregate; recommendations stay `insufficient_evidence` until a
metric pair and/or probe-on log is supplied.

---

## Scene-type rollup (MOT17 train labels)

Coarse labels for rollup only (not a gate):

| seq prefix | type |
|:--|:--|
| MOT17-02, MOT17-04 | `crowded_static` |
| MOT17-05 | `moving_low` |
| MOT17-09 | `static_low` |
| MOT17-10 | `moving` |
| MOT17-11 | `indoor_static` |
| MOT17-13 | `moving_night` |

These map 7 MOT17-train sequences used throughout the eval harness. They are
**not** a production 13-class taxonomy; extend `MOT17_SEQ_TYPE` if a richer
type map is later validated.

---

## How to run

```bash
# Probe-on log only (no metrics → expect insufficient_evidence / rare harmful)
.venv/bin/python scripts/eval/diagnostics/analyze_occ_audit_seq_conditioning.py \
  --occ-audit-csv results/<run>/_occ_audit.csv \
  --out-json results/<run>/occ_audit_seq_applicability.json \
  --out-md results/<run>/occ_audit_seq_applicability.md

# With control vs treatment metric deltas
.venv/bin/python scripts/eval/diagnostics/analyze_occ_audit_seq_conditioning.py \
  --occ-audit-csv results/<treatment>/_occ_audit.csv \
  --metrics-json results/<pair>/occ_audit_metrics.json \
  --out-json results/<pair>/occ_audit_seq_applicability.json \
  --out-md results/<pair>/occ_audit_seq_applicability.md
```

Metrics JSON shape:

```json
{
  "per_sequence": {
    "MOT17-04-SDP": {
      "idf1_control": 79.5,
      "idf1_treatment": 79.2,
      "ids_control": 40,
      "ids_treatment": 48
    },
    "MOT17-05-SDP": {
      "idf1_delta": 0.4,
      "ids_delta": -3
    }
  }
}
```

Eval flags needed for a probe-on log (still default-off; research only):

```text
--occ-audit --occ-audit-bank-reference --occ-audit-log --occ-audit-chebgr-probe
```

---

## Seed read (historical cosine-only log)

Source (probe **off**, no metric pair):

```text
results/analysis_m_semantic_delayed_claim_control_20260703_occaudit_tau0.45/_occ_audit.csv
```

| seq | type | audited | cosine_flags | recommendation |
|:--|:--|--:|--:|:--|
| MOT17-02-SDP | crowded_static | 94 | 42 | `insufficient_evidence` |
| MOT17-04-SDP | crowded_static | 129 | 36 | `insufficient_evidence` |
| MOT17-05-SDP | moving_low | 51 | 20 | `insufficient_evidence` |
| MOT17-09-SDP | static_low | 29 | 13 | `insufficient_evidence` |
| MOT17-10-SDP | moving | 107 | 30 | `insufficient_evidence` |
| MOT17-11-SDP | indoor_static | 48 | 15 | `insufficient_evidence` |
| MOT17-13-SDP | moving_night | 120 | 53 | `insufficient_evidence` |

**Read:** episode mass is highest on 02/04/10/13; flag rate is material
everywhere. Without (a) Cheb-GR probe columns and (b) control/treatment
IDF1–IDs pairs, **no sequence is an enable_candidate**. This matches the
conservative contract and the historical #55 NO-GO note that global audit
was bipolar (hurt crowded 02/04; helped some others).

Unit tests encode the classification contracts on synthetic rows (positive /
mixed / harmful / empty / probe-off tolerance). They do **not** claim live
MOT17 metrics.

---

## What WP2 deliberately does **not** do

- No `occ_audit_seq_allowlist` / gate flag in lifecycle or evaluator  
- No `mamba_whole_graph*.yaml` change  
- No live critical-path promotion  
- No sparse bank C++ sidecar  
- No WP3 promotion decision  

---

## WP3 handoff

WP3 should:

1. Run a frozen-substrate pair: control (no audit) vs treatment  
   (`occ-audit` + bank-reference + chebgr-probe + log).  
2. Build `occ_audit_metrics.json` from TrackEval / motmetrics per-seq.  
3. Re-run this diagnostic; publish the first real enable/abstain/harmful map.  
4. Only then decide whether a **default-off** sequence gate (or a split
   `feat/occ-exit-conditional-audit`) is warranted.

---

## Verification

```bash
.venv/bin/pytest -q tests/unit/eval/test_occ_audit_seq_conditioning.py

.venv/bin/python scripts/eval/diagnostics/analyze_occ_audit_seq_conditioning.py \
  --occ-audit-csv results/analysis_m_semantic_delayed_claim_control_20260703_occaudit_tau0.45/_occ_audit.csv \
  --out-md /tmp/occ_audit_wp2_smoke.md
```

Related: [scope.md](scope.md),
WP1a #74, WP1b #75, registry [#55](../../reference/no_go_registry.md).
