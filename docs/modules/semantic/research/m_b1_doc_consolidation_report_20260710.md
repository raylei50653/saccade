# M-B1 / M-B1.5 documentation consolidation report

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-10 -->
<!-- doc-module: semantic -->

**Role:** Information-preservation report for M-B1/M-B1.5 doc governance.  
**Not** a research evidence note (research terminals unchanged at claim level).  
**Research claims changed:** **no** (Q4.5 remains `isolated_safe_points_only`)  
**Production preset:** **unchanged**

**PR dependency:** consolidation on top of PR #89 (`research/m-b1-5-stage2-q45-atlas`, HEAD `6df1739b`).  
This round rebases onto evaluator **v4** and repairs provenance / preservation contracts.

---

## 0. Preservation contract (locked)

```text
contract_kind: claims_and_evidence_preserved
NOT: full method reprint of every deleted note

What canonicals MUST retain:
  - research questions / terminals / claim firewalls
  - quantitative evidence (tables, AUC, FP, LOO folds, atlas counts)
  - substrate + study_id / out/signal_study paths
  - production boundaries
  - authorized next steps
  - explicit historical reversals

What MAY remain git-only (not inlined into canonicals):
  - long method formalizations (ε-constrained optimization writeups,
    submodular greedy OR trajectories, signal-role taxonomies beyond
    grammar headlines, extended intermediate plots/tables)

Requirement for every deleted source:
  - basename listed in migration table
  - unique_information_status ∈ {claims_and_evidence_preserved,
                                  preserved_as_historical_context,
                                  intentionally_retained_as_separate_contract}
  - source_commit_or_blob_sha recorded (recover: git show <blob>)
```

**Why not `fully_preserved`:** reprinting every method section would re-inflate
canonicals toward the original ~21 notes. Reviewable evidence chain =
claims + numbers + study paths **in** docs, methods **recoverable** from git blobs.

---

## 0b. Attestation (post v4 rebase repair)

```text
information_loss_detected: no   # under claims_and_evidence_preserved contract
broken_references_detected: no
semantic_state_conflicts_detected: no
research_claims_changed: no
q45_counts_synced_to_v4: yes    # 154 / 1·153·0 / 0 interior / 0 region / 0 exact LOSO
provenance_self_refs_fixed: yes
terminal_namespaces_declared: yes
artifact_provenance_preserved: yes
stale_token_scan_clean: yes     # see §7
doc_consolidation: PASSED_UNDER_CLAIMS_AND_EVIDENCE_CONTRACT
```

### Semantic consistency audit

Retained + canonical `m_b1*.md` scanned for stale **current-tense** tokens and
pre-v4 atlas language:

```text
stale research-state tokens (must be historical-marked if present):
  Stage 1 overall: OPEN | online_blocked | not e2e_safe_for_default_off |
  Stage 2 not started | B-audit pending | next: new signal family |
  next=default-off hook | next=B2/e2e

pre-v4 atlas tokens (must NOT appear as current truth):
  productive_safe cells: 211 | AND: 210 | loo_stable_region_but_seq_thin
  (except explicit historical / superseded callouts)

bare terminal letters (must be namespaced if current):
  terminal C  (Q4) without q4_separability_grade / stage2_entry_terminal_after_q4

self-referential Absorbed from:
  Absorbed from: <this canonical basename>  → forbidden

stale nav language:
  hub Tier B  (history canonical uses §1–§18 + blob SHAs, not Tier B)
```

| Result | Count |
|:--|--:|
| unmarked stale current-state hits | **0** |
| unmarked ledger `next=` (default-off hook / B2/e2e) | **0** |
| bare Q4 `terminal C` without namespace | **0** |
| current-tense pre-v4 atlas count hits (`211`/`210` as truth) | **0** |
| self-referential Absorbed-from | **0** |
| hub Tier B refs | **0** |

---

## 1. Before / after inventory

Scope: `docs/modules/semantic/research/m_b1*.md`  
(plus companion references under `docs/modules/semantic/`, `docs/research/`, code docstrings).

```text
files_before (m_b1*.md at start of consolidation):  25
files_after  (m_b1*.md on disk):                      8
files_deleted:                                       21
files_merged (into new canonicals):                  21
files_retained (standalone contracts/plan):           4
files_created (3 canonicals + this report):           4
```

### After set (8)

| File | Role |
|:--|:--|
| `m_b1_research_history_20260709_20260710.md` | **NEW** offline history canonical |
| `m_b1_stage1_online_hook_final_20260710.md` | **NEW** Stage 1 final |
| `m_b1_5_stage2_d_online_final_20260710.md` | **NEW** Stage 2 final (Q1–Q4.5 **v4**) |
| `m_b1_doc_consolidation_report_20260710.md` | **NEW** this report |
| `m_b1_repaired_eps0_loo_pass_candidate_20260709.md` | **RETAINED** freeze identity |
| `m_b1_portable_or_tail_hook_contract_20260709.md` | **RETAINED** hook ABI (code refs) |
| `m_b1_5_stage2_entry_contract_20260710.md` | **RETAINED** Stage 2 firewall (code refs) |
| `m_b1_to_m_b1_5_two_stage_plan_20260710.md` | **RETAINED** plan body (runner refs) |

Committed evidence pack (not in `m_b1*.md` count):  
`docs/modules/semantic/research/evidence/m_b1_5_stage2_q45_20260710/`

---

## 2. Per-file migration table

Blob SHAs resolve on tips that still contained the file  
(offline/Stage1 notes: pre-consolidation `16031c47` tree;  
Q4.5 atlas note: v4 tip `6df1739b`).

| Original file | Action | Canonical destination | Preserved in canonical | unique_information_status | source_blob_sha | links_updated | deletion_status |
| ------------- | ------ | --------------------- | ---------------------- | ------------------------- | --------------- | ------------- | --------------- |
| `m_b1_bridge_discriminability_20260709.md` | merge+delete | history §1 | full/hard AUC, thr recall, substrate, study | claims_and_evidence_preserved | `37ec792162d4417fe14355b90127848761c4cbbc` | yes | deleted |
| `m_b1_signal_mine_batch_20260709.md` | merge+delete | history §2 | hard AUC rank, study, tool | claims_and_evidence_preserved | `717953f94a765a0f418bc26a05c0372c50fa0361` | yes | deleted |
| `m_b1_signal_scale_linear_log_20260709.md` | merge+delete | history §3 | monotone AUC invariance, study | claims_and_evidence_preserved | `1cdf69f50290cdd184477c8359059d9db92b0ad0` | yes | deleted |
| `m_b1_energy_transform_separability_20260709.md` | merge+delete | history §4 | d′/Fisher protocol, no-AUC-for-transform | claims_and_evidence_preserved | `7831848638d6ee017b4ccd8958e6be707779306d` | yes | deleted |
| `m_b1_signal_distribution_stability_20260709.md` | merge+delete | history §5 | cross-seq thr hurt std, kurtosis | claims_and_evidence_preserved | `d8994592ab169a96f7220975c6ef34310c4e987b` | yes | deleted |
| `m_b1_gate_coverage_7seq_20260709.md` | merge+delete | history §6 | L0 coverage numbers | claims_and_evidence_preserved | `4228b22afc7c69588541048594a8a0911d9f0de9` | yes | deleted |
| `m_b1_combo_gate_safe_region_20260709.md` | merge+delete | history §7 | recoverability vs marginal FP | claims_and_evidence_preserved | `fd1a41e758eb20ea78c6b260c662744f308bbae4` | yes | deleted |
| `m_b1_gt_safe_region_area_20260709.md` | merge+delete | history §8 | GT-tail area ε0 isolated | claims_and_evidence_preserved | `00ec9848c1d5f31d547e9ebf84c1d6fd181a20f9` | yes | deleted |
| `m_b1_weight_method_safe_region_20260709.md` | merge+delete | history §9 | no thick ε0 weight plateau | claims_and_evidence_preserved | `248ea4ac46e6cef1ec28e6f195e6a8a5938d1190` | yes | deleted |
| `m_b1_gate_rule_search_architecture_20260709.md` | merge+delete | history §10 | atoms→AND→OR grammar headline, in-sample FP9130 | claims_and_evidence_preserved | `69498921a6e8bab4c7571c080a19e7ddbfd523d4` | yes | deleted |
| `m_b1_policy_card_eps0_or5_20260709.md` | merge+delete | history §10 | unrepaired in-sample card; **superseded for LOO** | preserved_as_historical_context | `718283d9e2e27329ea9d04a7df0e89cb99d99fad` | yes | deleted |
| `m_b1_gate_rule_search_loo_20260709.md` | merge+delete | history §11 | loo_partial 5/7, mean teFP 1278 | claims_and_evidence_preserved | `787e00132ee17f6eeb9316b3783adbd780e5d760` | yes | deleted |
| `m_b1_loo_hurt_atom_repair_20260709.md` | merge+delete | history §12 | ban_gap+ban_zone 7/7, 97.3% retained | claims_and_evidence_preserved | `df90cc232c39ad93bce8017b6f6b0c5aa7d28441` | yes | deleted |
| `m_b1_repaired_tail_or_safe_region_20260709.md` | merge+delete | history §13 | safe%~56, p80~14–15, LOO region | claims_and_evidence_preserved | `ad442570d51a2d00b8d40701926475f0ed8030fa` | yes | deleted |
| `m_b1_repaired_candidate_b2e2e_smoke_contract_20260709.md` | merge+delete | history §14 | offline_pass / online_blocked, FP8721 | claims_and_evidence_preserved | `2f01aa9c0b13347220fe3419aad88835230f4200` | yes | deleted |
| `m_b1_offline_safe_region_phase_20260709.md` | merge+delete | history header + §15–§18 | phase CLOSED nav contract, tool index | claims_and_evidence_preserved | `06877af91c0e54ba81ea743259482908bed6b8a9` | yes | deleted |
| `m_b1_hook_stage1_e2e_20260710.md` | merge+delete | stage1 final Part A | Stage 1 evidence tables | claims_and_evidence_preserved | `25d770fbe43a878feac901e9076690c39e9a5911` | yes | deleted |
| `m_b1_hook_stage1_wire_20260710.md` | merge+delete | stage1 final Part B | wire inventory, host dbg index map | claims_and_evidence_preserved | `37aed64990cc69c609ae74e667e3e2ff731ff4c9` | yes | deleted |
| `m_b1_5_stage2_q1q3_d_online_audit_20260710.md` | merge+delete | stage2 final Part A | Q1–Q3 counts, taxonomy, study paths | claims_and_evidence_preserved | `7b745472203610556774cf642bd8b86a453f21ea` | yes | deleted |
| `m_b1_5_stage2_q4_separability_20260710.md` | merge+delete | stage2 final Part B | AUC 0.588, cohort 23/64, grade C | claims_and_evidence_preserved | `68c9126051dfcea4f767bb45fc027d96808b772a` | yes | deleted |
| `m_b1_5_stage2_q45_threshold_atlas_20260710.md` | merge+delete | stage2 final Part C–D | **v4** atlas 154/0, evaluator gates, frame provenance | claims_and_evidence_preserved | `e25574d9bc28955e2b94bfb0c1e053e4382b8935` | yes | deleted |
| `m_b1_repaired_eps0_loo_pass_candidate_20260709.md` | retain | — | freeze identity living card | intentionally_retained_as_separate_contract | (retained) | nav links only | retained |
| `m_b1_portable_or_tail_hook_contract_20260709.md` | retain | — | hook ABI; code/docstring refs | intentionally_retained_as_separate_contract | (retained) | yes | retained |
| `m_b1_5_stage2_entry_contract_20260710.md` | retain | — | G0–G4; code/runner refs | intentionally_retained_as_separate_contract | (retained) | yes | retained |
| `m_b1_to_m_b1_5_two_stage_plan_20260710.md` | retain | — | plan body; runner refs | intentionally_retained_as_separate_contract | (retained) | yes | retained |

### Method detail intentionally left in git (example)

`m_b1_gate_rule_search_architecture_20260709.md` blob  
`69498921a6e8bab4c7571c080a19e7ddbfd523d4` still holds:

- ε-constrained optimization formalization  
- Pareto / monotone AND pruning / submodular greedy OR method duties  
- signal role taxonomy  
- ε=0.01 results  
- greedy marginal trajectory  
- anti brute-force / post-hoc overfit argumentation  

Canonical history §10 keeps **claims + grammar + FP9130 headline** only.

---

## 3. Unique-information audit

| Category | Status | Where |
|:--|:--|:--|
| research questions | preserved | history §1–14; stage2 Parts A–D |
| substrate identities | preserved | history §0; stage1 studies; stage2 D_online |
| study/artifact paths | preserved | all sections list `out/signal_study/...` + evidence pack |
| key quantitative results | preserved | AUC, FP, LOO folds, 244, 23/64, atlas **154**/0 |
| per-stage terminals | preserved | offline CLOSED; S1 CLOSED; Q4 grade C; Q4.5 terminal B |
| terminal namespaces | preserved | `q4_separability_grade` / `stage2_entry_terminal_after_q4` / `q45_atlas_terminal` |
| claim limitations | preserved | history §17; stage1/2 claim firewalls |
| negative results | preserved | loo_partial, online NULL, support mismatch, atlas B |
| failed hypotheses | preserved | weight plateau none; singleton thr inseparable |
| support mismatches | preserved | freeze thr vs D_online max; Stage1 NULL |
| LOO / sequence dependence | preserved | history §11–13; stage2 Q4 LOO; Q4.5 nested LOSO |
| production boundaries | preserved | preset unchanged everywhere |
| authorized next steps | preserved | ranking/assignment after valid assignment key |
| historical reversals | preserved | Stage1 1a/1b split; unrepaired OR-5 superseded; pre-v4 211 superseded |
| frame provenance | preserved | stage2 Part C (host counter ≠ MOT frame; competition untrusted) |
| full method formalizations | git-recoverable | blob SHAs in §2 |

---

## 4. Terminal letter namespaces

| Field | Rubric | Current value |
|:--|:--|:--|
| `q4_separability_grade` | Q4 effect-size / LOO / pure-neg grade | **C** (weak/unstable) |
| `stage2_entry_terminal_after_q4` | Entry-contract legal A/B/C | **B** (mass>0, no stable separation) |
| `q45_atlas_terminal` | Q4.5 atlas taxonomy A/B | **B** (`isolated_safe_points_only`) |

Entry-contract legal Stage 2 terminals:

- **A** — mass + stable separation  
- **B** — mass, no stable separation  
- **C** — FP mass insufficient / placement too late  

These are **not** interchangeable with Q4 grades or Q4.5 atlas letters.

---

## 5. Reference audit

### Updated entry points

| Location | Change |
|:--|:--|
| `docs/modules/semantic/README.md` | 3 canonicals + retained contracts + this report |
| `docs/modules/semantic/TODO.md` | S1/S2 finals; Q4.5 terminal language |
| `docs/research/threads/m_b1_online_hook_20260709.md` | v4 one-liner + namespaces |
| `docs/research/threads/README.md` | one-line status |
| `docs/research/eval/signal_analysis_ledger.md` | doc links → history/finals |
| retained contracts + plan | evidence links → finals; plan top summary **154** |

### Code / runner string references (must remain valid)

| Reference | Status |
|:--|:--|
| `portable_or_tail.py` → hook contract + two-stage plan | **intact** |
| `d_online_stage2.py` / q4 / q45 / runners → entry contract | **intact** |
| `run_m_b1_hook_ab.py` → two-stage plan | **intact** |
| study IDs under `out/signal_study/` | **not docs**; unchanged |
| evidence pack under `docs/.../evidence/m_b1_5_stage2_q45_20260710/` | **committed** |

### Dead-link / stale-token check

```text
rg deleted basenames under docs/ + src/ + scripts/ (excl. out/, report, history §18 tables):
  → hits only intentional provenance tables / this report

rg '211|210|loo_stable_region_but_seq_thin' as current truth:
  → 0 (v4 uses 154 / 153 / edge_candidate)

rg self-referential Absorbed from:
  → 0
```

---

## 6. Explicit non-goals

```text
- did not re-run experiments
- did not change Q4.5 research terminal (still isolated_safe_points_only)
- did not upgrade isolated safe points
- did not rewrite NULL_support_mismatch as success
- did not change production preset
- did not merge retained contracts into narrative docs
- did not delete artifact paths
- did not claim full method reprint (contract = claims_and_evidence_preserved)
```

---

## 7. Checker commands (re-runnable)

```bash
# dead basenames (expect report/history provenance tables only)
rg -n 'm_b1_5_stage2_q45_threshold_atlas|m_b1_gate_rule_search_architecture|m_b1_hook_stage1_e2e' \
  docs src scripts --glob '!**/.git/**'

# pre-v4 counts as current truth (expect 0 outside explicit historical notes)
rg -n 'productive_safe cells: 211|AND:    210|loo_stable_region_but_seq_thin' \
  docs/modules/semantic/research docs/research

# unmarked ledger next= (expect historical-marked only)
rg -n 'next=default-off hook|next=B2/e2e' docs/research/eval/signal_analysis_ledger.md

# bare Q4 terminal C (expect namespaced form only)
rg -n 'terminal C separability|terminal C weak' docs/research docs/modules/semantic

# stale hub Tier B
rg -n 'hub Tier B' docs/research docs/modules/semantic

# self-ref Absorbed from
rg -n 'Absorbed from:.*m_b1_research_history_20260709_20260710|Absorbed from:.*stage1_online_hook_final' \
  docs/modules/semantic/research

# blob recoverability spot-check
git cat-file -t 69498921a6e8bab4c7571c080a19e7ddbfd523d4   # blob
git cat-file -t e25574d9bc28955e2b94bfb0c1e053e4382b8935   # blob
```

---

## 8. Canonical documents (reading order)

1. [Offline history](m_b1_research_history_20260709_20260710.md)
2. [Freeze candidate card](m_b1_repaired_eps0_loo_pass_candidate_20260709.md)
3. [Hook ABI contract](m_b1_portable_or_tail_hook_contract_20260709.md)
4. [Stage 1 final](m_b1_stage1_online_hook_final_20260710.md)
5. [Stage 2 entry contract](m_b1_5_stage2_entry_contract_20260710.md)
6. [Stage 2 D_online final](m_b1_5_stage2_d_online_final_20260710.md) — **v4 truth**
7. [Two-stage plan](m_b1_to_m_b1_5_two_stage_plan_20260710.md)
8. Thread: [m_b1_online_hook_20260709.md](../../../research/threads/m_b1_online_hook_20260709.md)
9. Evidence pack: [evidence/m_b1_5_stage2_q45_20260710/](evidence/m_b1_5_stage2_q45_20260710/)
