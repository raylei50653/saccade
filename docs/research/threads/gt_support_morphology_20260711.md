---
doc-status: parked
doc-promotion: navigation-only; not evidence
owner-module: semantic
work-class: mainline-study
wip-role: parked
created: 2026-07-11
---

# GT-Support Morphology on Boolean Atom Lattices

## Status

| Item | Status |
|:--|:--|
| Program | **PARKED** — PR-D accepted boundary preserved；restricted-closure prototype and PR-E are **SUSPENDED pending re-charter** |
| Research object | μ_GT placement distribution on the atom lattice（**不是** per-cell risk field —— step-0 判定其不可識別） |
| Step-0 audit | **recorded** · [note](../../modules/semantic/research/gt_support_morphology_step0_20260711.md) · committed packet: [evidence/gt_support_morphology_step0_20260711/](../../modules/semantic/research/evidence/gt_support_morphology_step0_20260711/manifest.json) |
| Procedure | framework §19 **v1 sealed** via [PR #100](https://github.com/raylei50653/saccade/pull/100) merge |
| Boolean closure-domain line | **PR-B normative doc merged** · [PR #101](https://github.com/raylei50653/saccade/pull/101) · [doc](../../modules/semantic/research/boolean_closure_domain_line_20260711.md) |
| Step-0 verdict | **`UNRESOLVED`**（no valid cluster-aware UCB · core 未求解 · nested 未重跑）+ bounded descriptive hypothesis: corner-concentrated placement + far-Hamming motion-violation-enriched tail |
| Far-Hamming descriptive tail | 4/209 tracks，**4/4 全在 MOT17-10-SDP**（sequence clustering 為實據）；motion-atom 集中（speed_mismatch 4/4 · dir_cos 3/4 · resid_mean 3/4 · log_h_ratio 0/4）；nominal CP x=4→4.33% 不得跨界 |
| Forensics (PR-C / #102) | **`ACCEPTED_WITH_LIMITS`** ([PR #104](https://github.com/raylei50653/saccade/pull/104)) · 3×`TRUE_LONG_GAP_REENTRY` + 1×`UNRESOLVED` · aggregate **`ROLE_REVERSAL_SUPPORTED`** · L1 single-seq (MOT17-10) only · authorizes partial-order audit only · [note](../../modules/semantic/research/escape_tail_forensic_20260711.md) · [packet](../../modules/semantic/research/evidence/escape_tail_forensic_20260711/manifest.json) |
| Partial-order audit (PR-D gate / #106 · [PR #107](https://github.com/raylei50653/saccade/pull/107)) | **`ACCEPTED_WITH_LIMITS`** · terminal **`GLOBAL_PARTIAL_ORDER_READY`** · global=`{dist_h, log_h_ratio}` · conditional=`{bridge_dist}`+motion · context=`{score_m_bridge, gap}` · [note](../../modules/semantic/research/boolean_atom_partial_order_20260711.md) · [packet](../../modules/semantic/research/evidence/boolean_atom_partial_order_20260711/manifest.json) |
| Restricted closure prototype | **SUSPENDED pending re-charter** · historical global-map boundary = `{dist_h, log_h_ratio}` only; conditional atoms remain **forbidden** in any future global solve |
| Nested per-fold rerun | **SUSPENDED** with its prototype charter; not run |
| Production / presets / ledger | **unchanged** |
| §20 classification (2026-07-12) | completed outputs = **diagnostic result**；restricted-closure prototype + PR-E = **SUSPENDED pending re-charter**（見下方區塊） |

## §20 re-classification (2026-07-12)

Role-aligned experiment contract v1 sealed（[framework §20](../contracts/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)，PR #133 merged）後的簿記重分類。純 §20.4 output-class 標籤，不改任何已 accepted 的結果、limit、或本線 PARKED 狀態與 resume condition。

**已完成產出 = diagnostic result（§20.4）：**

| 產出 | §20.2 declaration（追認） | §20.4 output class |
|:--|:--|:--|
| Step-0 identifiability + placement audit（#100） | target layer=coarse gate · intent=boundary diagnostic | **diagnostic result**（identifiability verdict + boundary morphology） |
| PR-C escape-tail forensic（#104） | target layer=coarse gate · intent=boundary diagnostic | **diagnostic result**（exceptional-tail attribution） |
| PR-D partial-order audit（#107） | target layer=coarse gate · intent=capability map | **diagnostic result**（capability map：atom 偏序資格） |

依 §20.5,這些結果不得直接或經重貼標籤升格為 design recommendation;依 §20.7,它們是 diagnostic,不佔主線節奏,完成不計主線 transition。

**未動工項目 = SUSPENDED pending re-charter：**

- **Restricted-closure prototype**（原 step-2b,PR-D 授權的 separate post-merge task）：**SUSPENDED**。原 selection rule——max FP removed、vs frozen OR-tail、candidate-only——已被 §20 廢止（§20.2「maximize FP removed alone is invalid」；§20.4「best performer alone is invalid」）。
- **PR-E nested held-out validation**（原「首個可超 L1 的確認單元」）：**SUSPENDED**。其確認對象是上述 prototype 的 closure candidate,charter 隨之失效。

**Re-charter 條件（resume 時二選一,先於任何求解宣告）：**

1. **Gate 方向（design evaluation @ coarse gate,§20.3）**：問題改為「CORE 簡化在大 margin 下能否以更簡結構達到最低 obvious-negative coverage」——不是「closure 能安全推到多遠」;或
2. **Score 方向（design evaluation @ score-ranking,§20.3）**：ambiguous band 內 atom interactions 是否穩定改善 GT vs FP 的相對排序（event-local ranking metrics）。

若 re-charter 後的答案仍然只是「gate 多移除 FP」,本線**不續**（§20.6 futility）。PARKED 狀態與 resume condition（gap-conditioned motion probe 釋放 semantic WIP lock 後 owner 明示再授權）不變;re-charter 是 resume 時的**額外**前置條件。

## Current boundary

```text
step-0 = pooled + in-sample + median-split ⇒ L1 descriptive ceiling
procedure v1 sealed (#100); PR-B line doc sealed (#101)
PR-C forensic ACCEPTED_WITH_LIMITS（PR #104）
  aggregate = ROLE_REVERSAL_SUPPORTED（L1 single-seq MOT17-10 bound）
  → authorizes partial-order audit only; NOT global motion closure / MWC / veto / production

Partial-order audit (PR #107 / #106):
ACCEPTED_WITH_LIMITS

Accepted map:
- global_orderable: dist_h, log_h_ratio
- conditional_orderable: bridge_dist, speed_mismatch, dir_cos, resid_mean
  (short-gap proposal-only; bridge_dist = motion-extrapolation composite)
- context_only: score_m_bridge, gap
- unresolved: ∅

Aggregate terminal:
GLOBAL_PARTIAL_ORDER_READY (accepted with limits)

Restricted closure prototype / PR-E:
SUSPENDED pending re-charter; not a current authorized task
  historical global-map limit = {dist_h, log_h_ratio}
  bridge_dist and other conditional atoms MUST NOT enter any future global solve
  Door 0 T2 closes the score continuation for its tested 12-member class;
  only a gate re-charter remains as a parked existing branch

不得 veto escape tail（protected GT mass）
零 exposure cells = UNRESOLVED，不是障壁（framework §19.2）
不碰 safe-region assetization 的 closed gates（A1 / terminal B / R2–R4）
```

## Read first

1. [Framework §19 — GT-support morphology predeclared procedure](../contracts/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)
2. [布林閉包域研究線 normative doc（PR-B）](../../modules/semantic/research/boolean_closure_domain_line_20260711.md)
3. [Step-0 note](../../modules/semantic/research/gt_support_morphology_step0_20260711.md)
4. [Partial-order audit note](../../modules/semantic/research/boolean_atom_partial_order_20260711.md)
5. Study artifacts: `out/signal_study/gt_support_morphology_step0_20260711/`
6. 前線 handoff: [safe-region assetization thread](closed/safe_region_assetization_20260710.md)（A1 CLOSED；R1.1 role-reversal 症狀 = 本線 escape-tail 機制的描述性前身）

## Artifacts

- `out/signal_study/gt_support_morphology_step0_20260711/step0_identifiability_audit.py` + `results.txt`
- `out/signal_study/gt_support_morphology_step0_20260711/hamming_profile.py` + `hamming_profile.txt`
- 上游 pool: `out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv`
- Partial-order packet: [evidence/boolean_atom_partial_order_20260711/](../../modules/semantic/research/evidence/boolean_atom_partial_order_20260711/manifest.json)

## Current step

**PARKED; no current work.** Door 0's `T2` closes the pre-existing **score**
continuation for its tested 12-member class, so that branch has **no
continuation**. The only retained existing branch is the **gate** direction,
which remains parked until an explicit owner re-charter and semantic WIP
authorization. It must not be described as an already-authorized restricted
closure prototype.

PR ladder（2026-07-11 owner 定版；#106 將 partial-order gate 與 closure prototype 拆開）：

```text
PR-A  #100  Step-0 + morphology procedure seal（MERGED = seal）
PR-B  #101  布林閉包域研究線 normative doc（MERGED）
PR-C  #102  4-track escape-tail forensic（MERGED #104；aggregate
            ROLE_REVERSAL_SUPPORTED；ACCEPTED_WITH_LIMITS）
PR-D  #106  partial-order audit gate（PR #107 ACCEPTED_WITH_LIMITS；
            terminal GLOBAL_PARTIAL_ORDER_READY；
            global = {dist_h, log_h_ratio}）
            → restricted-closure prototype = separate post-merge task
PR-E        nested held-out validation（outer-fold full replay；
            首個可超 L1 的確認單元）
```

## Acceptance

```text
step-1 forensics: DONE — 3×TRUE_LONG_GAP_REENTRY + 1×UNRESOLVED
       aggregate ROLE_REVERSAL_SUPPORTED（MOT17-10 single-seq bound）
step-2 partial-order audit: ACCEPTED_WITH_LIMITS (PR #107)
       terminal GLOBAL_PARTIAL_ORDER_READY
       global = dist_h · log_h_ratio
       conditional = bridge_dist · speed_mismatch · dir_cos · resid_mean
       context_only = score_m_bridge · gap
step-2b restricted-closure prototype: SUSPENDED pending re-charter
       historical global-map limit ONLY = {dist_h, log_h_ratio}
step-3 nested per-fold: SUSPENDED with the prototype charter; no current
       confirmatory unit is authorized
closing hypothesis: GT 是否形成 corner-concentrated core
       + 可解釋的 conditional escape tail？
```

## Must not

- 把零 exposure 解讀為高風險 / 障壁；
- veto escape tail；motion atoms、bridge_dist 或 score_m_bridge 直接當全域 closure 維度；
- 在 restricted-closure global solve 中使用 bridge_dist 或其他 conditional atoms；
- merge tree / barrier height / per-cell UCB 進主線（低於宣告密度）；
- 以 pooled audit 或單序列 forensic 充當 confirmatory 證據；
- 將 partial-order audit 與 MWC prototype 併入同一 PR；
- 動 production preset、evidence_ledger、或 safe-region closed gates。

## History

- 2026-07-11: Line opened（owner 選定 next scientific uncertainty）。Step-0 identifiability + placement audit recorded：per-cell risk field 不可識別（任何 k 僅 1 cell 達 ε≤0.05）；GT placement 可識別（M₀=97.1% @ k=5 median-split；4/209 escape tail，motion-atom 集中，log_h_ratio 0/4）。研究物件轉換為 μ_GT；framework §19 procedure v1 起草（PROPOSED）。
- 2026-07-11: [PR #100](https://github.com/raylei50653/saccade/pull/100) opened as **procedure v1 seal unit**；review fixes #1–#5 landed；**MERGED = seal**。
- 2026-07-11: Owner 定版 **PR ladder A–E**；[PR #101](https://github.com/raylei50653/saccade/pull/101) **PR-B** boolean closure-domain normative doc **MERGED**。
- 2026-07-11: **PR #100 review = REQUEST_CHANGES（research seal 層；engineering PASS）**，五點修正全數落地：①§19.5 加 UCB validity（residual clustering 下 plain CP = nominal diagnostic，不得跨 ε_morph；須 cluster-aware 或再聚合）②§19.4 GT trial 改 set-valued semantics（Z_u 全集 + H_C(u)=1[Z_u∩C=∅]；min-d_H representative 降 descriptive-only）③Step-0 terminal 改 **UNRESOLVED** + bounded descriptive hypothesis（`CORE_PLUS_…` token 撤回；「out-of-core mass」保留給 C* 求解後）④core 定義補全：C* = argmin retained-FP s.t. valid UCB[P(H_C=1)] ≤ ε_morph；up-set 方向 + deterministic tie-breaks + complexity cap + Ω/missing-value 宣告 ⑤committed reproduction packet（[evidence/](../../modules/semantic/research/evidence/gt_support_morphology_step0_20260711/manifest.json)：gt_rows / occupancy k4–k8 / tail_tracks / cp_ucb + scripts + pairs.csv SHA seal）。**副發現：tail 4/4 全在 MOT17-10-SDP** —— sequence clustering 為實據非假設，直接支持 UNRESOLVED，並成為 PR-C forensic 的關鍵 context。
- 2026-07-11: **PR-C / issue #102 escape-tail forensic** via [PR #104](https://github.com/raylei50653/saccade/pull/104). Review blockers fixed (signal non-tautology · scene sheets · operationalization honesty). Research acceptance **`ACCEPTED_WITH_LIMITS`**: 3×TRUE + 1×UNRESOLVED · aggregate `ROLE_REVERSAL_SUPPORTED` · L1 single-seq only · authorizes partial-order audit only.
- 2026-07-11: **PR-D gate / issue #106 Boolean-atom partial-order audit** via [PR #107](https://github.com/raylei50653/saccade/pull/107). **Initial operational terminal** = `GLOBAL_PARTIAL_ORDER_READY` with global=`{bridge_dist, dist_h, log_h_ratio}`. Research-owner review found `bridge_dist` provenance misclassification and incorrect `score_m_bridge` unit claim; revisions demoted `bridge_dist`, fixed units/guards/DAG, and held status UNDER REVIEW. **Final research acceptance `ACCEPTED_WITH_LIMITS`**: terminal `GLOBAL_PARTIAL_ORDER_READY` · global=`{dist_h, log_h_ratio}` · conditional includes `bridge_dist` · authorizes only a separate restricted-closure prototype on the accepted global pair.
- 2026-07-11: Parked before the restricted-closure prototype started so the independent gap-conditioned motion probe can hold semantic WIP=1. Accepted PR-D roles and the resume boundary remain unchanged; no closure prototype artifact exists yet.
- 2026-07-12: **§20 re-classification（bookkeeping）**。Contract v1（PR #133）生效後追認分類：step-0 / PR-C / PR-D 全部 = diagnostic result（coarse-gate boundary diagnostic / capability map）；restricted-closure prototype 與 PR-E 標 **SUSPENDED pending re-charter**（原 max-FP-removed selection rule 被 §20 廢止；re-charter = gate 方向或 score 方向二選一）。PARKED 狀態與 resume condition 不變。同 PR 將 framework §19 procedure v1 抽出為獨立檔 [procedures/gt_support_morphology_procedure_v1.md](../eval/procedures/gt_support_morphology_procedure_v1.md)（§19.x 編號保留；§19 留 tombstone）。
