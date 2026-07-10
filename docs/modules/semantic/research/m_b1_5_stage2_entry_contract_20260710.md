# M-B1.5 Stage 2 entry contract — \(D_{\text{online}}\) claim firewall

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-10 -->
<!-- doc-module: semantic -->

**Role:** Stage 2 **entry + acceptance contract** (claim firewall). Not evidence tables.  
**Prerequisite:** Stage 1 substrate study `m_b1_hook_ab_20260710T071001Z_stage1_close` (B-audit 244-row table) · [e2e note](m_b1_hook_stage1_e2e_20260710.md).  
**Q1–Q3 evidence:** [m_b1_5_stage2_q1q3_d_online_audit_20260710.md](m_b1_5_stage2_q1q3_d_online_audit_20260710.md) · study `m_b1_5_stage2_q1q3_20260710` (**PASSED / PASSED / SUFFICIENT**).  
**Q4 evidence:** [m_b1_5_stage2_q4_separability_20260710.md](m_b1_5_stage2_q4_separability_20260710.md) · study `m_b1_5_stage2_q4_20260710` (**C weak/unstable**).  
**Q4.5 atlas:** [m_b1_5_stage2_q45_threshold_atlas_20260710.md](m_b1_5_stage2_q45_threshold_atlas_20260710.md) · study `m_b1_5_stage2_q45_20260710` (**B isolated_safe_points_only**).  
**Plan body:** [two-stage plan §14+](m_b1_to_m_b1_5_two_stage_plan_20260710.md).  
**Thread:** [m_b1_online_hook_20260709.md](../../../research/threads/m_b1_online_hook_20260709.md).

---

## 0. Stage 1 final scope (locked — do not re-litigate)

```text
stage1_overall: CLOSED

hook_mechanism:              VALIDATED
  policy load → eval entry → signal cross → atom fire → reject → output Δ
  (P/F controls + B-audit counters/table)

measurement_substrate:       VALIDATED
  trusted baseline + full online population + recon + A0 + det + runtime naming
  + default-off / preset boundary

frozen_policy_online_relevance: NULL_support_mismatch
  evaluated=244  triggered=0  rejected=0
  effective_intervention_coverage=0
  policy_effect_supported: no
  reason: vacuous_online_threshold

e2e_safe_for_default_off: yes
  = null-effect mount is safe
  ≠ policy is online-effective
  ≠ effect fully power-tested under freeze thr

production_promotion: BLOCKED
```

Stage 1’s real job was:

> **建立可信的觀測與介入基礎，而不是證明 frozen policy 有效。**

That job is done.

---

## 1. Stage 2 primary substrate

```text
D_online = Stage 1 online B-audit full event table
         = 244 hook-eligible rows at production bridge placement
         (NOT global offline pairs; NOT offline q85 re-search as first move)
```

Canonical machine-readable source:

```text
out/signal_study/m_b1_hook_ab_20260710T071001Z_stage1_close/
  hook_candidate_events.{csv,parquet}
  atom_summary.csv
  per_sequence_summary.csv
  baudit_summary.json
```

Offline pairs / freeze thr remain **context only**. Frozen five-atom **signal definitions** may be reused; frozen offline thr is **not** truth on \(D_{\text{online}}\).

---

## 2. Ordered Stage 2 questions (do not skip)

First question is **not** “threshold 要設多少”.

| Order | Question | If fail → |
|:--|:--|:--|
| **Q1** | 244 rows 的 GT / FP / ambiguous outcome 是否完整可重建？ | fix labels / join keys before any thr work |
| **Q2** | 各 signal 在 \(D_{\text{online}}\) 的實際 support 與分布？ | declare support-empty families |
| **Q3** | FP mass 是否足以支持 reject 型研究？ | underpowered → placement / ranking path |
| **Q4** | GT 與 FP 是否存在穩定分離？ | no separation → change signal family, not thr chase |
| **Q5** | 若 Q3∧Q4：再研究 threshold / Boolean safe region | only after Q1–Q4 pass |
| **Q6** | 若無 FP mass：signal 不足 vs **hook placement 太晚**？ | earlier gate / ranking — not offline q85 |

### Three legal terminal conclusions

```text
A. FP mass > 0  AND  GT/FP separation stable
   → enter conditional safe-region (restricted Boolean grammar; plan §18)

B. FP mass > 0  AND  no stable separation
   → change signal family / features
   → DO NOT hard-tune thresholds on inseparable support

C. FP mass ≈ 0  (or effective reject power underpowered)
   → placement too late relative to production gates
   → study earlier placement or ranking/margin
   → DO NOT conclude “need better offline thr”
```

All three are **valid Stage 2 outcomes**. Only A continues into safe-region thickness / LOO work.

---

## 3. Claim firewall (G0–G4)

Every Stage 2 claim must state: **claim → falsifier → observables → reject conditions → maximum promotion**.

### G0 — Population & labels

| | |
|:--|:--|
| **Claim class** | “we measured on \(D_{\text{online}}\)” |
| **Falsifier** | events missing · recon fail · labels unjoinable |
| **Observables** | n_events=244 · fire_class counts · join_key coverage · GT/FP/ambiguous rates |
| **Reject if** | n_events ≠ native eligible · recon errors · GT/FP incomplete without declared ambiguous policy |
| **Max promotion** | substrate_ready / substrate_blocked |

### G1 — Effect admissibility (intervention power)

| | |
|:--|:--|
| **Claim class** | “policy/clause had online effect” |
| **Firewall** | `triggered == 0` → **effect claim inadmissible** |
| **Firewall** | `decision_changed == 0` → **downstream effect claim inadmissible** |
| **Firewall** | effective coverage insufficient → **underpowered intervention** (not “safe productive”) |
| **Observables** | n_triggered · n_rejected · n_decision_changed · per-seq coverage |
| **Max promotion** | effect_supported / null_vacuous / underpowered / unsafe |

Locked Stage 1 instance:

```text
triggered=0, rejected=0, decision_changed=0
→ freeze policy: effect claim inadmissible
→ online_relevance: NULL_support_mismatch
```

### G2 — Safety (GT preservation)

| | |
|:--|:--|
| **Claim class** | “GT-safe reject region” |
| **Falsifier** | GT_hurt > ε on declared domain · LOO GT hurt · worst-seq GT hurt |
| **Observables** | GT_hurt · GT_hurt_rate · per-seq GT_hurt · LOO flags |
| **Reject if** | any undeclared ε; silent averaging hides worst seq |
| **Max promotion** | gt_safe_on_domain / gt_unsafe / inconclusive |

### G3 — Productivity (FP mass)

| | |
|:--|:--|
| **Claim class** | “productive FP removal” |
| **Falsifier** | FP_removed=0 · support too thin · single-seq dominance of FP mass |
| **Observables** | FP_total · FP_removed · productive_safe · seq contribution HHI / max-seq share |
| **Reject if** | **single-sequence dominance** of support or FP mass without multi-seq check |
| **Max promotion** | productive / unproductive / underpowered |

Portability firewall:

```text
single-sequence dominance → portability claim blocked
```

### G4 — Region stability (threshold / Boolean)

| | |
|:--|:--|
| **Claim class** | “stable safe region / thick boundary” |
| **Falsifier** | neighboring thr unsupported · subset monotonicity break · LOO boundary jump |
| **Observables** | ordered-family subset checks · safe interval thickness · LOO boundary shift |
| **Reject if** | **neighboring threshold unsupported** → stable-region claim blocked |
| **Max promotion** | region_stable / point_only / unstable |

---

## 4. Claim template (required per hypothesis)

Before any grid or sweep, register in `hypothesis_registry.json` (plan §16):

```text
hypothesis_id:
claim_type:          # singleton_sufficiency | ...
claim_text:          # one sentence
falsifier:           # what observation kills the claim
observables:         # columns / derived metrics
reject_conditions:   # G0–G4 gates that apply
maximum_promotion:   # never above this ladder rung
domain:              D_online (244) | declared subset
negative_controls:
```

Promotion ladder (cannot skip):

```text
substrate_ready
  → support_characterized
    → fp_mass_measured
      → separation_tested
        → point_safe (if any)
          → region_stable (only if G4)
            → loo_portable (only if multi-seq + LOO)
              → e2e_candidate (only after eng re-entry; not Stage 2 alone)
                → production   # OUT OF SCOPE for Stage 2
```

---

## 5. Explicit inadmissible claims (Stage 2 start)

```text
triggered == 0
  → effect claim inadmissible

decision_changed == 0
  → downstream effect claim inadmissible

effective coverage insufficient
  → underpowered intervention
  → cannot promote to “e2e productive policy”

single-sequence dominance
  → portability claim blocked

neighboring threshold unsupported
  → stable-region claim blocked

e2e_safe_for_default_off (Stage 1 null-effect)
  → does NOT imply online policy power
  → does NOT unlock production

offline q85 / offline FP=8721
  → not online pruning power on D_online
```

---

## 6. Forbidden Stage 2 work (from plan §19 + Stage 1 discipline)

```text
- unrestricted Boolean mining then post-hoc safety claim
- adaptive thr expansion chasing best FP in-session
- silent default-on / production preset change
- treating freeze offline thr as online truth
- starting at thr sweep before Q1–Q4
- upgrading soft single-seq wins to portable claims
```

---

## 7. Stage 2 first deliverable (entry work package)

Minimal machine-readable pack before any “safe region” headline:

```text
out/signal_study/m_b1_5_<stamp>/
  d_online_events.parquet          # 244 + GT/FP/ambiguous joins
  label_join_report.json           # Q1: coverage / fail modes
  signal_support_summary.csv       # Q2
  fp_mass_summary.json             # Q3: FP mass + per-seq shares
  separation_audit.parquet         # Q4: optional before thr
  claim_log.jsonl                  # each claim + G-gate verdict
  hypothesis_registry.json
  summary.md                       # legal conclusion A | B | C
```

**Stage 2 entry is satisfied only when Q1–Q3 are answered** (Q4 may share the first study).  
Thr / Boolean grids **must not** start until entry pack exists.

**Status 2026-07-10:** entry pack **fulfilled** through Q4.5.

```text
entry contract: fulfilled
Q1–Q3: completed  (study m_b1_5_stage2_q1q3_20260710 · SUFFICIENT)
Q4: completed     (study m_b1_5_stage2_q4_20260710 · C weak/unstable)
Q4.5: completed   (study m_b1_5_stage2_q45_20260710 · B isolated_safe_points_only)
current terminal: isolated_safe_points_only
next authorized direction:
  ranking / assignment-relative audit
secondary:
  thin-edge diagnostics
  absolute MOT frame instrumentation
threshold/hook-policy promotion: blocked
production_preset: unchanged
```

Evidence narrative: [Stage 2 final](m_b1_5_stage2_q45_threshold_atlas_20260710.md).  
G0–G4 claim firewall rules above remain normative (not reopened).

---

## 8. Relationship to Stage 1 artifacts

| Stage 1 fact | Stage 2 use |
|:--|:--|
| eligible=244, rejected=0, all zero-fire | prior: freeze thr outside support |
| full event signals + margins | primary feature matrix |
| join_key / track ids / frame | GT-FP rebuild |
| recon pass | G0 substrate trust |
| rebased A0 + det | eng baseline for any later e2e re-entry |
| P/F action path | mechanism already proven; do not re-prove with thr search |

---

## 9. One-line mainline

```text
M-B1 Stage 1: CLOSED
hook mechanism: validated
measurement substrate: validated
frozen policy online relevance: NULL
Stage 2 entry: fulfilled (Q1–Q4.5 complete)
current terminal: isolated_safe_points_only
production promotion: blocked
next authorized: ranking / assignment-relative audit
secondary: thin-edge diagnostics · absolute MOT frame instrumentation
```
