---
doc-status: active
doc-promotion: none
doc-date: 2026-07-10
doc-module: semantic
---

# MOT M-B1 → M-B1.5 two-stage implementation plan

> **Role:** Stage 1 + Stage 2 engineering contract (single source; methods + full-table rules).  
> **Navigation:** [m_b1 online hook thread](../../../research/threads/closed/m_b1_online_hook_20260709.md) · [hook ABI](m_b1_portable_or_tail_hook_contract_20260709.md).  
> **Evidence:** [Stage 1 final](m_b1_stage1_online_hook_final_20260710.md) · [Stage 2 final](m_b1_5_stage2_d_online_final_20260710.md).  
> **Not evidence tables:** freeze card / ledger / study dirs own numbers.

### Top summary (current — 2026-07-10)

```text
Stage 1: CLOSED
Q1–Q3: SUFFICIENT
Q4: q4_separability_grade C (weak/unstable; best oriented AUC 0.588)
     → stage2_entry_terminal_after_q4: B
Q4.5 (evaluator v4): q45_atlas_terminal B = isolated_safe_points_only
productive_safe cells: 154   (single 1 / AND 153 / OR 0)
coordinate-union interior: 0
stable region candidates: 0
exact_absolute nested LOSO portable: 0
selected unresolved: 21 (blocks candidate)
competition-relative columns: untrusted (invalid_frame_provenance)
new-signal path: PARKED / secondary
next preferred direction: ranking / assignment-relative audit
  (after valid assignment-group key)
threshold/hook-policy promotion: blocked
production preset: unchanged
```

0. Purpose

工作分成兩個嚴格分離的階段（**both executed** as of 2026-07-10）：

Stage 1 — M-B1 frozen hook + action-path validation  
  **execution_status: completed · terminal: stage1_overall CLOSED**
  1a: policy load + evaluation-entry (eligible counters)  **PASSED**
  freeze B: online relevance NULL (support mismatch)      **observed**
  1b: plumbing controls prove signal→atom→reject→decision **PASSED**
  B-audit full table + recon + rebased A0 + det + runtime **PASSED**

Stage 2 — M-B1.5 conditional-domain safe-negative audit  
  **execution_status: completed through Q4.5 · thr/hook-policy promotion blocked**
  在 production baseline 已接受的條件域 D_online 內，
  是否存在穩定可泛化的 safe-negative region
  （不是再對全 offline pairs 做 q85）
  Q1–Q3: m_b1_5_stage2_q1q3_20260710 (SUFFICIENT mass)
  Q4: m_b1_5_stage2_q4_20260710 → q4_separability_grade C
      (weak/unstable; best AUC 0.588) → stage2_entry_terminal_after_q4 B
  Q4.5 v4: m_b1_5_stage2_q45_20260710 → q45_atlas_terminal B
      isolated_safe_points_only · 154 productive-safe · 0 region candidates
  next preferred: ranking / assignment-relative audit — not thr-as-rule

核心原則：

不得把「eligible>0 且 freeze 未 reject」寫成「hook 全鏈已驗證」。
1a ≠ 1b ≠ Stage 1 CLOSED。
不得在 Stage 1 對 production freeze 做 thr search；plumbing control thr 是預先固定的測試臂。
不得在 Stage 2 直接修改 production preset。

---

1. Global invariants

兩階段共同遵守：

GT preservation first
FP pruning second
tracking metric gain last

目前主要安全邊界：

epsilon = 0
per-seq GT_hurt = 0
LOO GT_hurt = 0

除非文件明確標示，以下詞彙：

necessary
sufficient
safe
implication

都只代表：

«在指定 candidate universe、sequence set、label source 與 evaluation substrate 上成立的 empirical relation。»

不代表對未知資料分布的全域數學證明。

---

2. Full-table retention requirement

兩階段都必須保留完整 machine-readable 結果。

不得只保留：

- pass rows
- safe rows
- selected candidate
- headline summary
- 最終 structural clause

所有被實際評估的資料都要保留，包括：

unsafe
unproductive
low-support
redundant
LOO-failed
negative-control
zero-effect

要求：

All summaries, boundaries, classifications, and selected candidates
must be reproducible from the retained full tables.

完整表格是研究母資料；Markdown 與 JSON summary 只負責呈現結論。

---

Stage 1 — M-B1 default-off online hook validation

3. Stage 1 goal

實作 research-only、default-off 的 online hook，精確套用 frozen M-B1 portable policy。

Stage 1 只回答：

1. frozen offline policy 能否被正確 replay 到 online path？
2. online/e2e coupling 後是否仍安全？
3. 五個 frozen atoms 實際如何觸發？
4. rejection 主要來自 singleton fire 還是 co-fire？
5. hook 是否改變 reconnect、tracking metrics、runtime 或 determinism？

Stage 1 不回答：

- 哪個 threshold 更好？
- 哪個 atom 是必要條件？
- 是否應改成 structural OR？
- 是否存在更好的 clause？

---

4. Read first

1. "docs/modules/semantic/research/m_b1_research_history_20260709_20260710.md"
2. "docs/modules/semantic/research/m_b1_repaired_eps0_loo_pass_candidate_20260709.md"
3. "docs/modules/semantic/research/m_b1_portable_or_tail_hook_contract_20260709.md"
4. "docs/research/eval/signal_analysis_ledger.md" §5

Optional:

- PR #83

---

5. Locked state

### Current lifecycle (2026-07-10)

```text
candidate_id: m_b1_repaired_eps0_loo_pass_20260709
offline freeze: LOO_pass_region_candidate · offline_smoke_pass
Stage 1 overall: CLOSED
freeze online relevance: NULL_support_mismatch
e2e_safe_for_default_off: yes  (= null-effect mount only)
production preset: unchanged / promotion blocked
Stage 2 Q4.5: isolated_safe_points_only
```

Portable policy:

out/signal_study/m_b1_repaired_eps0_loo_pass_20260709/portable_policy.json

Frozen policy shape:

OR of 5 singleton tail_q85 atoms
no zone
no gap
ban_gap
ban_zone

### Historical checkpoint as of 2026-07-09 (pre–Stage 1 close)

```text
prior checkpoint (superseded):
  online_blocked
  not e2e_safe_for_default_off
  next then: implement default-off hook + A/B
superseded by: Stage 1 CLOSED · Stage 2 Q4.5 terminal B
```

---

6. Stage 1 implementation scope

6.1 Policy loader

實作 frozen portable-policy loader。

必須驗證：

schema version
candidate_id
ordered atom ids
signal names
predicate directions
threshold values
ban_gap
ban_zone
policy file hash

禁止：

- runtime refit
- threshold override
- policy repair
- atom reorder that changes semantics
- 缺欄位時靜默 fallback
- 從其他 study directory 自動挑另一份 policy

policy 不相容時應 fail closed，不能默默繼續 hook-on。

---

6.2 Default-off control

新增 research-only 控制介面，可使用：

CLI flag
environment variable
research config field

要求：

default = off
production preset unchanged
hook-off path does not require policy file
no silent enable

hook-disabled 路徑應盡量不引入額外 candidate materialization、policy evaluation 或 audit allocation。

---

6.3 Hook application point

hook 必須套在 contract 指定的 online candidate decision point。

不得：

- 移動到不同 pipeline stage
- 改變 baseline candidate generation
- 改變 baseline ranking
- 改變非被 reject candidate 的分數
- 修改其他 gate / relink / association policy

語義必須是：

baseline candidate set
→ evaluate frozen predicates
→ reject candidate if any frozen atom fires
→ continue existing baseline logic

---

6.4 Audit modes

至少區分：

hook off
hook on, audit off
hook on, audit on

runtime 報告必須分開，避免把 audit serialization 成本算進 policy overhead。

---

7. Stage 1 full event table

Stage 1 必須輸出完整 event-level table。

建議檔名：

hook_candidate_events.parquet

每個到達 hook evaluation point 的 candidate 至少一列。

最低欄位：

run_id
sequence
frame
event_id
runtime_candidate_id

policy_candidate_id
policy_file_hash
policy_schema_version

atom_bitmask
fired_atom_ids
n_atoms_fired
fire_class:
  zero
  singleton
  cofire

rejected_by_hook

baseline_rank
baseline_accepted_candidate
hook_accepted_candidate
accepted_competitor_changed

baseline_reconnect_decision
hook_reconnect_decision
reconnect_decision_changed
reconnect_outcome

local_assignment_changed
downstream_identity_changed

audit_timestamp_or_order

若某些 downstream effects 無法在同一 row 即時取得，可用穩定的 "event_id" 在後處理 join。

完整表必須包含：

zero-fire candidates
singleton-fire candidates
co-fire candidates
rejected candidates
fired but decision-neutral candidates

不得只輸出最終 rejection subset。

可另外輸出 derived subset：

rejected_events.parquet

但它必須由完整表格派生。

---

8. Stage 1 derived tables

8.1 Atom summary

atom_summary.csv

欄位：

atom_id
n_fired
n_singleton
n_cofired
n_rejected
n_decision_changed
n_sequences_fired
n_sequences_singleton

8.2 Per-sequence summary

per_sequence_summary.csv

欄位：

sequence
n_hook_eligible
n_zero_fire
n_singleton
n_cofire
n_rejected
n_competitor_changed
n_reconnect_changed
reconnect_success_delta
reconnect_miss_delta

8.3 A/B metrics

ab_metrics.json

至少包含：

IDF1
AssA
HOTA
MOTA
IDs
FP
FN
reconnect metrics
per-sequence metrics

8.4 Runtime

runtime.json

分開報告：

hook-disabled overhead
hook-enabled policy overhead
audit-enabled overhead

8.5 Determinism

determinism.json

至少包含：

hook-off baseline compatibility hash
hook-on repeated-run output hashes
event-table hash
summary artifact hashes

---

9. Stage 1 A/B matrix

A0 — Existing baseline reference

existing B2 baseline
old code or trusted reference artifact

A1 — New code, hook disabled

same B2 substrate
new implementation
hook off

要求：

A1 == A0

檢查：

- metrics identity
- result-file hash identity
- decision/output identity
- production preset identity
- disabled-path runtime無實質退化

B — Frozen hook enabled

same B2 substrate
same run conditions
frozen policy enabled

比較：

A1 vs B

B-audit — Frozen hook with audit enabled

用於確認完整表格與 derived summaries。

不可用 B-audit runtime 直接代表 policy runtime。

---

10. Stage 1 reconciliation assertions

必須程式化檢查：

n_hook_eligible
=
n_zero_fire + n_singleton + n_cofire

n_rejected
=
count(rejected_by_hook = true)
=
singleton_rejected + cofire_rejected
=
sum(per-seq rejected)

per_atom_singleton
<=
per_atom_fired

derived summaries
=
group-by results from hook_candidate_events.parquet

summary 不得使用另一套獨立計數邏輯。

---

11. Stage 1 headline

唯一 headline：

e2e_safe_for_default_off: yes / no

判斷不能只看 IDF1。

至少考慮：

default-off invisibility
GT/reconnect safety
per-sequence effects
IDF1 / AssA / HOTA / MOTA
IDs / FP / FN
runtime overhead
determinism

建議 classification：

e2e_safe_for_default_off
online_effect_neutral_but_safe
online_unsafe
online_inconclusive

---

12. Stage 1 forbidden work

禁止：

rule search
threshold sweep
repair
new atoms
learned weights
runtime refit
zone/gap atoms
structural OR redesign
necessary/sufficient-condition mining
production preset change
silent default-on

Stage 1 A/B 結束後必須停止。

失敗或 neutral 不代表 offline M-B1 candidate 被推翻，只代表 frozen application 的 online status。

---

13. Stage 1 artifacts

out/signal_study/<m_b1_hook_ab_id>/
  manifest.json
  portable_policy.snapshot.json

  hook_candidate_events.parquet
  rejected_events.parquet

  atom_summary.csv
  per_sequence_summary.csv

  ab_metrics.json
  runtime.json
  determinism.json

  summary.json
  summary.md

"manifest.json" 至少記錄：

study_id
git commit
candidate_id
policy path
policy hash
sequence set
candidate-universe identity
runtime config
hook flag state
audit mode
evaluator version
artifact hashes

---

Stage 2 — M-B1.5 parameterized implication and safe-region audit

```text
execution_status: completed through Q4.5 (2026-07-10)
q45_atlas_terminal: B
terminal: isolated_safe_points_only
productive_safe: 154 (v4)
stable_region_candidates: 0
coordinate_union_interior: 0
exact_absolute_nested_loso_portable: 0
threshold/hook-policy promotion: blocked
method body below remains normative full-table contract
```

14. Stage 2 entry condition

```text
execution_status: completed
entry pack: m_b1_5_stage2_q1q3_20260710 · SUFFICIENT
Q4: completed · q4_separability_grade C → stage2_entry_terminal_after_q4 B
Q4.5: completed · q45_atlas_terminal B (v4 · 154 productive-safe · 0 region)
```

Stage 1 overall is **CLOSED** (eng + B-audit + identity + det + runtime).
Stage 2 may now use the online full event table as primary domain substrate;
Stage 2 PRs stay separate from Stage 1 hook implementation.

**Authoritative Stage 2 entry + claim firewall:**  
[m_b1_5_stage2_entry_contract_20260710.md](m_b1_5_stage2_entry_contract_20260710.md)  
(G0–G4 · Q1–Q6 order · three legal terminals A/B/C · claim template).

不得和 Stage 1 hook implementation 放在同一個 policy-remodeling PR。

**Ordered entry (do not skip) — historical plan order; all steps completed:**

1. Online full B-audit event table exists (eligible=244) — **done in Stage 1**
2. Rebuild GT / FP / ambiguous outcomes on those 244 rows (Q1) — **done**
3. Signal support & distribution on \(D_{\text{online}}\) (Q2) — **done**
4. Measure FP mass **inside** \(D_{\text{online}}\) (Q3) — **done · SUFFICIENT**
5. Test GT/FP separation (Q4) — **done · q4_separability_grade C**; thr Boolean atlas Q4.5 — **done · q45_atlas_terminal B (v4)**

**Claim firewall (summary):**  
`triggered==0` → effect claim inadmissible ·  
`decision_changed==0` → downstream effect inadmissible ·  
insufficient coverage → underpowered ·  
single-seq dominance → portability blocked ·  
neighbor thr unsupported → stable-region blocked.

Stage 2 可使用：

- M-B1 offline candidate rows (context only; not the primary domain)
- frozen five-atom **signal definitions** (not frozen offline thr as truth)
- underlying continuous signals on **online** events
- Stage 1 full hook event table / B-audit table
- 明確註冊的 context/support predicates that define \(D_{\text{online}}\)

Stage 1 online-safe **does not** imply offline thr is online-relevant. Status must stay split:

```text
Stage 1: eng safety + wiring + e2e_safe
Stage 2: conditional safe-negative region on D_online
```

If B-audit shows insufficient FP mass in the 244, Stage 2 may conclude **placement too late** rather than “need better thr.”

---

15. Stage 2 goal

Stage 2 的目標不是再對 \(D_{\text{offline}}\) 做 q85，也不是找最高分布林規則，而是：

```text
max_C  FP_removed(C | D_online)
s.t.   GT_hurt(C | D_online) ≤ ε
```

並測量不同 signal value / 必要·充分條件 / AND·OR 結構

如何改變：

GT-safe reject domain
productive FP-removal domain
safe-region boundary
safe-region thickness
safe-region area
connectedness
per-sequence robustness
LOO portability

核心問題：

«How do parameter values and restricted Boolean relations reshape the GT-safe and productive reject regions?»

---

16. Hypothesis registry

所有 hypothesis 必須先註冊，再執行。

建議檔名：

hypothesis_registry.json

每個 hypothesis 至少包含：

hypothesis_id
claim_type

signal_family
signal_names
predicate_directions
transforms

value_grid
Boolean_form
context_scope

candidate_universe
label_source
sequence_set
evaluation_units

expected_role
negative_controls

"claim_type" 建議限制為：

singleton_sufficiency
necessary_GT_envelope
conditional_sufficiency
OR_branch_stability
extreme_singleton
moderate_consensus
necessary_violation_with_support
context_modifier

禁止從結果表中臨時挑出漂亮 clause，再補寫 hypothesis。

---

17. Parameter grids

每個連續 signal 建立受限且預先定義的 value grid。

例如：

q80
q85
q90
q95

或 frozen threshold 的固定鄰域：

q82.5
q85
q87.5
q90

每個 study 必須記錄：

value representation
quantile reference population
direction
bounds
grid spacing
normalization

不得在同一輪根據結果 adaptive expansion 追逐最佳點。

---

17.1 Predicate ordering and monotonicity invariants

對具有明確 threshold ordering 的 predicate family，必須先驗證集合巢狀關係，再進行 safe-region 解讀。

例如 upper-tail predicate：

P_q95 ⊆ P_q90 ⊆ P_q85 ⊆ P_q80

lower-tail predicate 則依定義方向建立對應的巢狀順序。

對任何宣告為 ordered family 的相鄰值 a、b，至少檢查：

expected subset relation
observed subset relation
support monotonicity
GT_hurt monotonicity
FP_removed monotonicity

對 upper-tail tightening，理論上應滿足：

support(strict) <= support(loose)
GT_hurt(strict) <= GT_hurt(loose)
FP_removed(strict) <= FP_removed(loose)

若不成立，不得直接解讀為真實 domain geometry；必須先分類原因：

predicate direction mismatch
quantile reference population mismatch
normalization mismatch
missing-value semantics
candidate-universe mismatch
implementation error
declared ordering invalid

建議輸出：

monotonicity_results.parquet

最低欄位：

study_id
hypothesis_id
predicate_family

loose_predicate_id
strict_predicate_id
loose_parameter_values
strict_parameter_values

expected_subset
observed_subset
subset_violation_count

loose_support
strict_support
support_monotonic

loose_GT_hurt
strict_GT_hurt
GT_hurt_monotonic

loose_FP_removed
strict_FP_removed
FP_removed_monotonic

status
failure_reason

只有通過 ordering audit 的 predicate family，才可用其 value grid 計算 first-safe boundary、safe interval 與 thickness。

---

18. Allowed Boolean grammar

18.1 Singleton sweep

P_a

目的：

- 找 first safe value
- 找 safe interval
- 測量 threshold thickness
- 測量 FP capacity 衰減

18.2 Pairwise AND

P_a AND Q_b

目的：

- 測試兩個單獨不充分條件是否共同充分
- 建立 2D safe/productive surface

18.3 Pairwise OR

P_a OR Q_b

目的：

- 測試不同 singleton-safe domains 聯集後的安全性
- 測量 hard OR 對 branch threshold 的敏感度

每個 OR branch 必須獨立接受 singleton sufficiency audit。

18.4 Necessary-condition violation

NOT N_a

其中：

GT => N_a

目的：

- 建立 GT support envelope
- 測量 envelope 外的 FP mass

18.5 Necessary violation with supporting evidence

NOT N_a AND P_b

目的：

- 測試必要條件 violation 是否過寬
- 測試額外 evidence 是否形成 productive safe domain

18.6 Extreme singleton or moderate consensus

P_extreme
OR
(P_moderate AND Q_moderate)

目的：

- 測試極端異常是否可 singleton reject
- 測試中度異常是否需要共識

18.7 Union of validated sufficient modes

S1 OR S2 OR ... OR Sk

每個 branch 必須先獨立通過 sufficiency audit。

---

19. Forbidden search space

Stage 2 禁止：

arbitrary Boolean enumeration
unrestricted negation
arbitrary depth > 2
learned Boolean weights
learned score fusion
runtime refit
new production gate
same-data unrestricted discovery followed by final safety claim
context predicate naked OR without independent sufficiency

Stage 2 是 restricted-domain analysis，不是 unrestricted rule mining。

---

20. Stage 2 evaluation layers

每個 hypothesis 必須在至少三層輸出結果。

20.1 Point-level audit

每一組具體 parameter values 一列。

20.2 Per-sequence audit

每一組 parameter values × sequence 一列。

20.3 Region/topology audit

對整個 grid 計算：

safe area
productive safe area
connected components
boundary
selected-point margin

另外保留 implication 與 minimality 關係。

---

21. Stage 2 full point table

建議檔名：

point_results.parquet

每個實際評估點都必須保留，包括 unsafe 與 zero-productivity points。

最低欄位：

study_id
hypothesis_id
predicate_id
predicate_expr

signal_family
signal_names
Boolean_form
claim_type

parameter_values
grid_coordinate
context_scope

GT_total
GT_hurt
GT_hurt_rate

FP_total
FP_removed
FP_removed_rate

support_count
safe
productive_safe

safe_neighbor_count
declared_neighbor_count
all_declared_neighbors_safe
robust_safe
topology_class

global_safe
worst_seq_safe
LOO_pass

n_candidate_support
n_unique_event_support
n_unique_track_pair_support
n_sequence_support

is_empirical_sufficient_reject
is_empirical_GT_necessary
is_inclusion_minimal
productive_support_pass

same_family_or_cross_family
n_unique_signal_families

negative_control_status
failure_reason
notes

不得只保留 safe points。

---

22. Stage 2 per-sequence table

建議檔名：

per_sequence_results.parquet

採 long-form 格式：

study_id
hypothesis_id
predicate_id
parameter_values
grid_coordinate

sequence

GT_total
GT_hurt
GT_hurt_rate

FP_total
FP_removed
FP_removed_rate

support_count
safe
productive_safe

n_candidate_support
n_unique_event_support
n_unique_track_pair_support

不得只把 per-seq 結果塞進單一 JSON 欄位。

此表必須能直接支援：

per-seq boundary
worst-sequence boundary
sequence-specific failure mode
sequence grouping
replotting

而不必重跑 evaluator。

---

23. Stage 2 region table

建議檔名：

region_results.parquet

最低欄位：

study_id
hypothesis_id
region_id

Boolean_form
parameter_space
parameter_bounds
grid_definition
measure_definition

grid_adjacency
distance_metric
skipped_point_connectivity
boundary_definition

n_grid_points
n_safe_points
n_productive_safe_points
n_robust_safe_points
n_productive_robust_safe_points

safe_area_ratio
productive_safe_area_ratio
robust_safe_area_ratio
productive_robust_safe_area_ratio

n_connected_safe_components
n_connected_productive_components
n_connected_robust_components
largest_connected_component_size
largest_productive_component_size
largest_robust_component_size

boundary_point_count
interior_safe_point_count
isolated_safe_point_count

selected_point
selected_component_id
selected_topology_class
distance_to_unsafe_boundary
distance_to_grid_edge

global_boundary
worst_sequence_boundary
LOO_boundary
boundary_spread

safe area 必須明確綁定 parameterization 與 measure。

定義：

SafeRegion_epsilon
=
{theta in Theta | GT_hurt(policy_theta) <= epsilon}

safe_area_ratio
=
measure(SafeRegion_epsilon) / measure(Theta)

不同 parameterization 下的 safe-area ratio 不得直接比較。

---

23.1 Topology, adjacency, and distance definitions

所有 connected-component、boundary、interior 與 distance 指標，都必須綁定明確的 grid topology。

每個 study 必須在 hypothesis registry 或 manifest 中宣告：

grid dimensionality
grid coordinate ordering
grid adjacency
distance metric
grid spacing
skipped-point treatment
grid-edge treatment
boundary definition

建議預設：

1D grid:
  adjacent index neighbors

2D rectangular grid:
  4-neighbor adjacency

更高維 grid:
  axis-aligned one-step adjacency

除非 hypothesis registry 明確指定，不得使用 diagonal adjacency。

skipped/error point 預設：

not traversable
does not connect two safe components
does not count as safe
must remain visible as an unknown point

distance_to_unsafe_boundary 必須明確定義。

對均勻離散 grid，建議使用：

minimum graph distance from the point
to an evaluated unsafe point within the declared parameter space

同時保留：

distance_to_grid_edge

避免 selected point 位於研究邊界時，因 grid 外未評估而被誤判為厚安全內部。

若 grid spacing 非均勻，另報：

index_distance_to_unsafe
parameter_distance_to_unsafe

boundary point 定義：

safe point with at least one declared adjacent evaluated unsafe point

interior safe point 定義：

safe point whose declared adjacent evaluated points are all safe

isolated safe point 定義：

safe point with no adjacent safe point

unknown-neighbor point：

至少一個 declared neighbor 為 skipped、error 或 grid 外未評估，因此不能自動分類為 robust interior。

拓撲結果不得依賴 artifact row order。

---

24. Necessary-condition table

建議檔名：

necessary_condition_results.parquet

每個 proposed necessary predicate 必須保留完整 2×2 support：

GT satisfying N
GT violating N
FP satisfying N
FP violating N

欄位：

hypothesis_id
predicate_id
parameter_values

GT_satisfy
GT_violate
FP_satisfy
FP_violate

GT_coverage
GT_violation_rate
FP_violation_rate
violation_productivity

global_necessary
per_seq_necessary
LOO_necessary

必須區分：

necessity strength
violation productivity

一條幾乎永遠為真的 predicate 即使 empirical necessary，也不代表有 reject 價值。

---

25. Implication and minimality tables

25.1 Implication edges

implication_edges.parquet

欄位：

source_predicate_id
target_predicate_id
relation_type

logical_by_threshold_order
empirical_subset
same_support
support_overlap

source_safe
target_safe

用途：

- 建立 value implication lattice
- 找出 equivalent predicates
- 找出 nested safe regions
- 區分邏輯 implication 與 empirical sufficiency

25.2 Minimality results

minimality_results.parquet

欄位：

parent_clause_id
subclause_id
dropped_atom_ids

parent_GT_hurt
subclause_GT_hurt

parent_FP_removed
subclause_FP_removed
FP_retention_vs_parent

is_inclusion_minimal
productive_support_pass
per_seq_support
LOO_pass

必須區分：

logical minimality
productive minimality

一個只砍掉單一孤立 FP event 的 clause，不應因為 inclusion-minimal 就自動晉級。

---

26. Negative controls

建議檔名：

negative_controls.parquet

每個主要 predicate family 視可行性加入：

opposite tail
central quantile band
same-support random predicate

目的：

- 檢查 sparse support 偶然 GT0
- 檢查 tail 語義是否真有結構
- 檢查低 support 本身是否足以產生假安全

可選工具 sanity check：

label permutation
sequence permutation

permutation 結果不得混入主要 empirical safety claim。

---

27. Boundary views

每個 predicate family 至少比較：

global first-safe boundary
per-sequence first-safe boundaries
worst-sequence first-safe boundary
LOO selected boundary
LOO holdout result

必要欄位：

global_first_safe_value
per_seq_first_safe_value
worst_seq_first_safe_value
LOO_selected_value
LOO_holdout_pass
boundary_spread

重點不是只有 selected value，而是：

safe interval
distance to unsafe boundary
boundary spread
productive capacity across interval

---

27.1 Robust safe interior classification

Stage 2 不只區分 safe / unsafe，還必須區分 safe point 在域中的位置。

最低 topology_class：

unsafe
isolated_safe
boundary_safe
interior_safe
productive_boundary_safe
productive_interior_safe
unknown_neighbor_safe

robust_safe 的預設定義：

safe = true
AND all declared adjacent evaluated neighbors are safe
AND no declared neighbor is skipped/error/unknown
AND the point is not protected only by the study grid edge

對 k-neighborhood robustness，可選擇另外報告：

robust_safe_k1
robust_safe_k2

但 k 值必須預先註冊，不得在結果後選擇。

建議衍生指標：

robust_safe_area_ratio
productive_robust_safe_area_ratio
largest_robust_component_size
selected_point_robust
selected_point_neighbor_safety
selected_point_distance_to_unsafe
selected_point_distance_to_grid_edge

selected candidate 不得只因為 safe 或 productive_safe 就自動晉級。

優先級應為：

productive robust interior
productive interior
productive boundary
isolated productive safe point

isolated safe point 可以保留作研究證據，但不得被描述成具有 domain thickness。

若 selected point 位於 grid edge，應標示：

edge_censored = true

此時只能說在已評估方向上未觀察到 unsafe boundary，不能宣稱完整厚度。

---

28. Same-family and cross-family composition

每個 AND／co-fire hypothesis 必須標示：

same_family
cross_family

並輸出：

n_unique_signal_families

兩個高度相關的同 family atoms 不應被自動解讀為兩份獨立 evidence。

Stage 2 應能回答：

同 family 共現只是冗餘嗎？
跨 family 共現是否形成新的 sufficient domain？

---

29. Stage 2 artifact structure

out/signal_study/<m_b1_5_study_id>/
  manifest.json
  hypothesis_registry.json

  point_results.parquet
  per_sequence_results.parquet
  region_results.parquet

  necessary_condition_results.parquet
  implication_edges.parquet
  minimality_results.parquet
  monotonicity_results.parquet
  negative_controls.parquet

  selected_inventory.json
  summary.json
  summary.md

"manifest.json" 至少記錄：

study_id
git commit
candidate-universe hash
label/source hash
signal schema hash
hypothesis registry hash
Boolean grammar version

value grids
parameter bounds
measure definitions

grid adjacency
distance metric
skipped-point connectivity
boundary definition
robust-neighborhood definition

expected hypothesis count
expected point count
expected per-sequence row count
expected implication-edge count where derivable

sequence set
evaluator version
source artifact paths
artifact hashes

---

30. Stage 2 validation requirements

Pre-run cardinality declaration

執行前必須從 hypothesis registry 與 value grids 計算並寫入 manifest：

expected_hypothesis_count
expected_point_count
expected_per_sequence_row_count
expected_region_count
expected_control_count

若不同 Boolean form 的 grid 維度不同，必須逐 hypothesis 記錄 expected point count，不得只用單一粗略乘法。

若預期 cardinality 超過 study contract 設定的上限，必須在執行前 fail，不能在跑到一半後靜默截斷。

Table completeness

number of point rows
=
sum(expected grid points for every registered hypothesis)

number of per-sequence rows
=
sum(expected grid points × declared sequence count)

除非 manifest 明確記錄 skipped/error rows。

每個 skipped point 也必須留下：

status
failure_reason

完成後必須輸出：

actual_hypothesis_count
actual_point_count
actual_per_sequence_row_count
actual_region_count
actual_control_count
completeness_pass

Monotonicity validation

所有宣告 ordered threshold family 的結果，都必須能在 monotonicity_results.parquet 中對帳。

若 ordering audit 失敗：

- 對應 family 不得產生可信 first-safe boundary
- 不得將 apparent non-monotonic island 解讀成真實 stable region
- summary 必須列出 violation count 與原因

Summary reproducibility

以下所有數字必須能從 full tables 重新產生：

safe_area_ratio
productive_safe_area_ratio
robust_safe_area_ratio
productive_robust_safe_area_ratio
connected-component counts
selected-point topology class
distance to unsafe boundary
first_safe_value
worst-sequence boundary
LOO pass
monotonicity status
minimal clause
selected inventory

Stable identifiers

所有 tables 使用穩定：

hypothesis_id
predicate_id
region_id
event_id

不得以 row order 作為跨 artifact join key。

---

31. Stage 2 final inventory

Stage 2 結束時，將所有 predicate/classification 分成：

validated sufficient singleton
empirical GT necessary predicate
minimal sufficient clause
context modifier
diagnostic only
unsafe
insufficient support
non-portable
redundant / equivalent

每個 inventory item 另外保留 topology classification：

isolated_safe
boundary_safe
robust_safe_interior
productive_robust_interior
edge_censored
unknown_neighbor

predicate role 與 topology class 是兩個不同維度，不得互相取代。

輸出：

selected_inventory.json

但完整失敗與未選 rows 仍保留在 full tables。

Stage 2 只能推薦下一階段 structural policy family，不得：

- 修改 frozen M-B1 hook
- 修改 production preset
- 宣稱 production-safe
- 靜默整合 structural OR

---

32. PR and document separation

建議拆成至少兩個 PR。

PR 1 — M-B1 frozen hook

包含：

policy loader
default-off hook
full event audit
A/B runner
runtime/determinism checks
hook result note
status updates

不含：

threshold sweep
structural clauses
implication mining

PR 2 — M-B1.5 domain audit

包含：

hypothesis registry
parameterized predicate generator
restricted Boolean evaluator
full point/per-seq/region tables
implication/minimality analysis
monotonicity and topology validation
robust-safe classification
negative controls
M-B1.5 result note

不含：

production hook replacement
preset change

---

33. Final execution order

1. Implement frozen policy loader.
2. Add default-off online hook.
3. Prove hook-off baseline identity.
4. Run B2 vs B2+hook A/B.
5. Retain complete hook event table.
6. Publish Stage 1 e2e status.
7. Freeze Stage 1 artifacts.

8. Register Stage 2 hypotheses, value grids, topology rules, and expected cardinalities.
9. Validate predicate ordering and generate monotonicity expectations.
10. Generate all restricted predicate/Boolean points.
11. Retain complete point and per-sequence tables.
12. Reconcile actual cardinalities against the manifest.
13. Compute safe-region topology, connected components, boundaries, and robust interiors.
14. Compute necessary/sufficient implication relations.
15. Run minimality and negative-control audits.
16. Publish classified predicate inventory with role and topology class.
17. Stop before structural-policy integration.

---

34. One-line summary

Stage 1 replays the frozen M-B1 hard OR through a research-only,
default-off online hook and retains a complete candidate-event audit.

Stage 2 independently evaluates bounded predicate values and restricted
Boolean forms, retaining every evaluated point so that necessary,
sufficient, AND, and OR effects on the GT-safe and productive reject
domains can be reproduced without rerunning candidate extraction.

This contract is complete when:

- Stage 1 can establish the frozen policy's online/e2e status without remodeling it.
- Stage 2 can reproduce every point, boundary, implication, topology class, and selected inventory item from retained full tables.
- No isolated or edge-censored safe point is promoted as a stable region without explicit qualification.
- Structural-policy integration remains a separate downstream task.