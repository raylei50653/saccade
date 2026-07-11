---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-11
---

# GT-Support Morphology on Boolean Atom Lattices

## Status

| Item | Status |
|:--|:--|
| Program | **ACTIVE** — semantic sole active（接替 safe-region assetization A1 closed 後的 research mainline） |
| Research object | μ_GT placement distribution on the atom lattice（**不是** per-cell risk field —— step-0 判定其不可識別） |
| Step-0 audit | **recorded** · [note](../../modules/semantic/research/gt_support_morphology_step0_20260711.md) · committed packet: [evidence/gt_support_morphology_step0_20260711/](../../modules/semantic/research/evidence/gt_support_morphology_step0_20260711/manifest.json) |
| Procedure | framework §19 **v1 sealed** via [PR #100](https://github.com/raylei50653/saccade/pull/100) merge |
| Boolean closure-domain line | **PR-B normative doc merged** · [PR #101](https://github.com/raylei50653/saccade/pull/101) · [doc](../../modules/semantic/research/boolean_closure_domain_line_20260711.md) |
| Step-0 verdict | **`UNRESOLVED`**（no valid cluster-aware UCB · core 未求解 · nested 未重跑）+ bounded descriptive hypothesis: corner-concentrated placement + far-Hamming motion-violation-enriched tail |
| Far-Hamming descriptive tail | 4/209 tracks，**4/4 全在 MOT17-10-SDP**（sequence clustering 為實據）；motion-atom 集中（speed_mismatch 4/4 · dir_cos 3/4 · resid_mean 3/4 · log_h_ratio 0/4）；nominal CP x=4→4.33% 不得跨界 |
| Forensics (PR-C / #102) | **`ACCEPTED_WITH_LIMITS`** ([PR #104](https://github.com/raylei50653/saccade/pull/104)) · 3×`TRUE_LONG_GAP_REENTRY` + 1×`UNRESOLVED` · aggregate **`ROLE_REVERSAL_SUPPORTED`** · L1 single-seq (MOT17-10) only · authorizes partial-order audit only · [note](../../modules/semantic/research/escape_tail_forensic_20260711.md) · [packet](../../modules/semantic/research/evidence/escape_tail_forensic_20260711/manifest.json) |
| Partial-order audit (PR-D gate / #106 · [PR #107](https://github.com/raylei50653/saccade/pull/107)) | **UNDER REVIEW** — research acceptance **pending** · operational map revised (see Current boundary); **not** accepted; does **not** authorize MWC · [note](../../modules/semantic/research/boolean_atom_partial_order_20260711.md) · [packet](../../modules/semantic/research/evidence/boolean_atom_partial_order_20260711/manifest.json) |
| Restricted closure prototype | **BLOCKED** until PR #107 receives research acceptance |
| Nested per-fold rerun | **not run**（升 L2+ 的 confirmatory unit） |
| Production / presets / ledger | **unchanged** |

## Current boundary

```text
step-0 = pooled + in-sample + median-split ⇒ L1 descriptive ceiling
procedure v1 sealed (#100); PR-B line doc sealed (#101)
PR-C forensic ACCEPTED_WITH_LIMITS（PR #104）
  aggregate = ROLE_REVERSAL_SUPPORTED（L1 single-seq MOT17-10 bound）
  → authorizes partial-order audit only; NOT global motion closure / MWC / veto / production

Partial-order audit (PR #107 / #106):
UNDER REVIEW — research acceptance pending

Initial operational map (packet; provisional, not accepted):
- candidate global: dist_h, log_h_ratio
- bridge_dist: conditional_orderable (motion-extrapolation-derived;
  initial PR misclassified as pure geometry / global — corrected under review)
- motion atoms: conditional proposal only
- score_m_bridge / gap: context_only

Aggregate terminal:
PENDING — GLOBAL_PARTIAL_ORDER_READY not yet accepted
  (initial operational terminal in packet remains provisional)

Restricted closure prototype:
BLOCKED until PR #107 receives research acceptance

不得 veto escape tail（protected GT mass）
零 exposure cells = UNRESOLVED，不是障壁（framework §19.2）
不碰 safe-region assetization 的 closed gates（A1 / terminal B / R2–R4）
不得把 packet 的 operational terminal 當成 research acceptance
```

## Read first

1. [Framework §19 — GT-support morphology predeclared procedure](../eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)
2. [布林閉包域研究線 normative doc（PR-B）](../../modules/semantic/research/boolean_closure_domain_line_20260711.md)
3. [Step-0 note](../../modules/semantic/research/gt_support_morphology_step0_20260711.md)
4. Study artifacts: `out/signal_study/gt_support_morphology_step0_20260711/`
5. 前線 handoff: [safe-region assetization thread](safe_region_assetization_20260710.md)（A1 CLOSED；R1.1 role-reversal 症狀 = 本線 escape-tail 機制的描述性前身）

## Artifacts

- `out/signal_study/gt_support_morphology_step0_20260711/step0_identifiability_audit.py` + `results.txt`
- `out/signal_study/gt_support_morphology_step0_20260711/hamming_profile.py` + `hamming_profile.txt`
- 上游 pool: `out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv`

## Current step

**PR-D gate / #106 via [PR #107](https://github.com/raylei50653/saccade/pull/107): UNDER REVIEW — research acceptance pending.**

Engineering has landed a revised operational packet (bridge_dist demoted; score_m_bridge units fixed). That is **not** research acceptance. **Do not** open restricted-closure / MWC work from this thread until the research owner accepts #107.

PR ladder（2026-07-11 owner 定版；#106 將 partial-order gate 與 closure prototype 拆開）：

```text
PR-A  #100  Step-0 + morphology procedure seal（MERGED = seal）
PR-B  #101  布林閉包域研究線 normative doc（MERGED）
PR-C  #102  4-track escape-tail forensic（MERGED #104；aggregate
            ROLE_REVERSAL_SUPPORTED；ACCEPTED_WITH_LIMITS）
PR-D  #106  partial-order audit gate（[PR #107](https://github.com/raylei50653/saccade/pull/107)
            UNDER REVIEW — research acceptance pending;
            operational terminal provisional only）
            → restricted-closure prototype only AFTER acceptance
PR-E        nested held-out validation（outer-fold full replay；
            首個可超 L1 的確認單元）
```

## Acceptance

```text
step-1 forensics: DONE — 3×TRUE_LONG_GAP_REENTRY + 1×UNRESOLVED
       aggregate ROLE_REVERSAL_SUPPORTED（MOT17-10 single-seq bound）
step-2 partial-order audit: UNDER REVIEW (PR #107)
       research acceptance: PENDING
       initial operational terminal = GLOBAL_PARTIAL_ORDER_READY (provisional)
       revised candidate global = dist_h · log_h_ratio
       bridge_dist role under re-audit (motion-extrapolation-derived)
step-2b restricted-closure prototype: BLOCKED
       (until PR #107 research acceptance)
step-3 nested per-fold: 整條鏈（atom 發現/定向/二值化/verdict）per-fold 重跑，
       verdict 以 sealed boundaries 判定；此為 L2+ 的 confirmatory unit
closing hypothesis: GT 是否形成 corner-concentrated core
       + 可解釋的 conditional escape tail？
```

## Must not

- 把零 exposure 解讀為高風險 / 障壁；
- veto escape tail；motion atoms、bridge_dist 或 score_m_bridge 直接當全域 closure 維度；
- merge tree / barrier height / per-cell UCB 進主線（低於宣告密度）；
- 以 pooled audit 或單序列 forensic 充當 confirmatory 證據；
- 將 packet 的 provisional `GLOBAL_PARTIAL_ORDER_READY` 當成 research acceptance；
- 在 #107 research acceptance 前開 restricted-closure / MWC / 最佳化任務；
- 將 partial-order audit 與 MWC prototype 併入同一 PR；
- 動 production preset、evidence_ledger、或 safe-region closed gates。

## History

- 2026-07-11: Line opened（owner 選定 next scientific uncertainty）。Step-0 identifiability + placement audit recorded：per-cell risk field 不可識別（任何 k 僅 1 cell 達 ε≤0.05）；GT placement 可識別（M₀=97.1% @ k=5 median-split；4/209 escape tail，motion-atom 集中，log_h_ratio 0/4）。研究物件轉換為 μ_GT；framework §19 procedure v1 起草（PROPOSED）。
- 2026-07-11: [PR #100](https://github.com/raylei50653/saccade/pull/100) opened as **procedure v1 seal unit**；review fixes #1–#5 landed；**MERGED = seal**。
- 2026-07-11: Owner 定版 **PR ladder A–E**；[PR #101](https://github.com/raylei50653/saccade/pull/101) **PR-B** boolean closure-domain normative doc **MERGED**。
- 2026-07-11: **PR #100 review = REQUEST_CHANGES（research seal 層；engineering PASS）**，五點修正全數落地：①§19.5 加 UCB validity（residual clustering 下 plain CP = nominal diagnostic，不得跨 ε_morph；須 cluster-aware 或再聚合）②§19.4 GT trial 改 set-valued semantics（Z_u 全集 + H_C(u)=1[Z_u∩C=∅]；min-d_H representative 降 descriptive-only）③Step-0 terminal 改 **UNRESOLVED** + bounded descriptive hypothesis（`CORE_PLUS_…` token 撤回；「out-of-core mass」保留給 C* 求解後）④core 定義補全：C* = argmin retained-FP s.t. valid UCB[P(H_C=1)] ≤ ε_morph；up-set 方向 + deterministic tie-breaks + complexity cap + Ω/missing-value 宣告 ⑤committed reproduction packet（[evidence/](../../modules/semantic/research/evidence/gt_support_morphology_step0_20260711/manifest.json)：gt_rows / occupancy k4–k8 / tail_tracks / cp_ucb + scripts + pairs.csv SHA seal）。**副發現：tail 4/4 全在 MOT17-10-SDP** —— sequence clustering 為實據非假設，直接支持 UNRESOLVED，並成為 PR-C forensic 的關鍵 context。
- 2026-07-11: **PR-C / issue #102 escape-tail forensic** via [PR #104](https://github.com/raylei50653/saccade/pull/104). Review blockers fixed (signal non-tautology · scene sheets · operationalization honesty). Research acceptance **`ACCEPTED_WITH_LIMITS`**: 3×TRUE + 1×UNRESOLVED · aggregate `ROLE_REVERSAL_SUPPORTED` · L1 single-seq only · authorizes partial-order audit only.
- 2026-07-11: **PR-D gate / issue #106 Boolean-atom partial-order audit** opened as [PR #107](https://github.com/raylei50653/saccade/pull/107). **Initial operational terminal** = `GLOBAL_PARTIAL_ORDER_READY` with global=`{bridge_dist, dist_h, log_h_ratio}`. **Research-owner review** found `bridge_dist` provenance misclassification (motion-extrapolation composite, not pure geometry) and incorrect `score_m_bridge` unit-incompatibility claim. Revision in flight: demote `bridge_dist`, fix units/guards. **Research acceptance pending revision** — active status is UNDER REVIEW; restricted-closure remains BLOCKED. Evidence history retains the initial operational map; do not treat it as accepted.
