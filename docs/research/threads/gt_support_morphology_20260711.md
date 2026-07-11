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
| Procedure | framework §19 **v1 — PROPOSED**；seal unit = [PR #100](https://github.com/raylei50653/saccade/pull/100)（Draft；review fixes #1–#5 applied 2026-07-11；merge = owner seal） |
| Step-0 verdict | **`UNRESOLVED`**（no valid cluster-aware UCB · core 未求解 · forensic 未跑 · nested 未重跑）+ bounded descriptive hypothesis: corner-concentrated placement + far-Hamming motion-violation-enriched tail |
| Far-Hamming descriptive tail | 4/209 tracks，**4/4 全在 MOT17-10-SDP**（sequence clustering 為實據）；motion-atom 集中（speed_mismatch 4/4 · dir_cos 3/4 · resid_mean 3/4 · log_h_ratio 0/4）；nominal CP x=4→4.33% 不得跨界 |
| Forensics | **not run**（排在 procedure seal 之後） |
| Nested per-fold rerun | **not run**（升 L2+ 的 confirmatory unit） |
| Production / presets / ledger | **unchanged** |

## Current boundary

```text
step-0 = pooled + in-sample + median-split ⇒ L1 descriptive ceiling
procedure v1 未 seal ⇒ 一切 verdict 皆 exploratory
不得 veto escape tail（protected GT mass）
零 exposure cells = UNRESOLVED，不是障壁（framework §19.2）
不碰 safe-region assetization 的 closed gates（A1 / terminal B / R2–R4）
```

## Read first

1. [Framework §19 — GT-support morphology predeclared procedure](../eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)
2. [Step-0 note](../../modules/semantic/research/gt_support_morphology_step0_20260711.md)
3. Study artifacts: `out/signal_study/gt_support_morphology_step0_20260711/`
4. 前線 handoff: [safe-region assetization thread](safe_region_assetization_20260710.md)（A1 CLOSED；R1.1 role-reversal 症狀 = 本線 escape-tail 機制的描述性前身）

## Artifacts

- `out/signal_study/gt_support_morphology_step0_20260711/step0_identifiability_audit.py` + `results.txt`
- `out/signal_study/gt_support_morphology_step0_20260711/hamming_profile.py` + `hamming_profile.txt`
- 上游 pool: `out/signal_study/m_b1_gate_coverage_7seq_20260709T121326Z/pairs.csv`

## Current step

**PR 1（Draft）= procedure seal unit**：framework §19（含 ε_morph=5% / CP-UCB 界線與四 terminals）+ Step-0 note + 治理配線。Review 中定版界線，owner merge = seal。**Forensic 結果不得進 PR 1**（時間邊界：機制資訊不可反饋 boundary）。Seal 前不跑 forensics、不跑 per-fold。

PR ladder（2026-07-11 owner 定版，五段）：

```text
PR-A  #100  Step-0 + morphology procedure seal（本 PR；merge = seal）
PR-B        布林閉包域研究線 normative doc（placement morphology →
            partial-order audit → MWC → exact GT-UCB validation →
            optional gate compression；含 Verdict A–E 完成條件）
PR-C        4-track escape-tail forensic（五類別；判 motion role-reversal
            是否真機制）
PR-D        restricted closure prototype（global_orderable atoms only；
            parametric MWC vs frozen OR-tail；read-only candidate-only；
            conditional closure probe 依 PR-C 結果併入）
PR-E        nested held-out validation（outer-fold full replay；
            首個可超 L1 的確認單元）
```

## Acceptance

```text
step-1 forensics: 4 條 escape-tail tracks 各落入唯一預宣告類別
step-2 nested per-fold: 整條鏈（atom 發現/定向/二值化/verdict）per-fold 重跑，
       verdict 以 sealed boundaries 判定；此為 L2+ 的 confirmatory unit
closing hypothesis: GT 是否形成 corner-concentrated core
       + 可解釋的 conditional escape tail？
```

## Must not

- 在 procedure seal 前跑 forensics 或宣稱 verdict；
- 把零 exposure 解讀為高風險 / 障壁；
- veto escape tail；motion atoms 直接當全域 closure 維度；
- merge tree / barrier height / per-cell UCB 進主線（低於宣告密度）；
- 以 pooled audit 充當 confirmatory 證據；
- 動 production preset、evidence_ledger、或 safe-region closed gates。

## History

- 2026-07-11: Line opened（owner 選定 next scientific uncertainty）。Step-0 identifiability + placement audit recorded：per-cell risk field 不可識別（任何 k 僅 1 cell 達 ε≤0.05）；GT placement 可識別（M₀=97.1% @ k=5 median-split；4/209 escape tail，motion-atom 集中，log_h_ratio 0/4）。研究物件轉換為 μ_GT；framework §19 procedure v1 起草（PROPOSED）。
- 2026-07-11: [PR #100](https://github.com/raylei50653/saccade/pull/100)（Draft）opened as **procedure v1 seal unit**：§19 含 ε_morph=5% CP-UCB 界線 + 四 terminals；boundaries 在 review 定版；forensic 結果排除在外（時間邊界）。
- 2026-07-11: **PR #100 review = REQUEST_CHANGES（research seal 層；engineering PASS）**，五點修正全數落地：①§19.5 加 UCB validity（residual clustering 下 plain CP = nominal diagnostic，不得跨 ε_morph；須 cluster-aware 或再聚合）②§19.4 GT trial 改 set-valued semantics（Z_u 全集 + H_C(u)=1[Z_u∩C=∅]；min-d_H representative 降 descriptive-only）③Step-0 terminal 改 **UNRESOLVED** + bounded descriptive hypothesis（`CORE_PLUS_…` token 撤回；「out-of-core mass」保留給 C* 求解後）④core 定義補全：C* = argmin retained-FP s.t. valid UCB[P(H_C=1)] ≤ ε_morph；up-set 方向 + deterministic tie-breaks + complexity cap + Ω/missing-value 宣告 ⑤committed reproduction packet（[evidence/](../../modules/semantic/research/evidence/gt_support_morphology_step0_20260711/manifest.json)：gt_rows / occupancy k4–k8 / tail_tracks / cp_ucb + scripts + pairs.csv SHA seal）。**副發現：tail 4/4 全在 MOT17-10-SDP** —— sequence clustering 為實據非假設，直接支持 UNRESOLVED，並成為 PR-C forensic 的關鍵 context。
