---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-12
task-type: mainline-diagnostic-probe
production-impact: none
default-behavior-change: forbidden
---

# Ambiguous-band ranking-power probe (Door 0)

## Status

| Item | Status |
|:--|:--|
| Program | **DECLARED — awaiting seal**（declaration PR open;exec行須 seal 後） |
| Mainline position | realignment sealed 順序 **step ③**（①contract v1 PR #133 → ②bookkeeping PR #134 → ③本 probe → ④/⑤依 terminal） |
| §20.2 declaration | [declaration doc](../../modules/semantic/research/ambiguous_band_ranking_power_probe_declaration_20260712.md)（target layer=score-ranking · intent=capability map · output class=diagnostic result only） |
| Core question | 凍結 signal family 在 gate-retained ambiguous band 內,對 GT vs FP 的相對排序是否存在**穩定、可解釋、LOO 保留**的改善(超越 production `s0=score_m_bridge` baseline)? |
| Substrate | frozen pairs `0ae38967…`(7-seq SDP;21,789 gt_valid rows;340 GT);coarse gate=production h-window proxy;**210/205 rankable events**(gate 前/後) |
| Probe family | 6 single-atom + 6 second-order AND conditions;lexicographic demotion(parameter-free;不 fit λ);q85/q15 band quantiles |
| Terminals | T1 `RANKING_SIGNAL_PRESENT`→開 step ④ / T2 `NO_USABLE_RANKING_POWER_IN_CLASS`→封 **Door-0 complexity class**(step ⑤ class-scoped;unexplained residual set w.r.t. class;不及於連續訊號/有限 λ)/ T3 `NO_HEADROOM`(H1 PWA∧H2 top-1 joint)→step ④ 在 Door-0 resolution 下不必要 / T0 `UNRESOLVED/INVALID-STUDY`→只關實驗;reachable-slice limit rule 適用全部 terminal |
| Prior risk | [幾何 AUC≈0.55 前例](../../modules/semantic/research/boolean_closure_domain_line_20260711.md)——band 內 ranking power 可能本來就弱;此即 Door 0 存在的理由(先驗證有訊號再投資 ④) |
| Production / presets / ledger | **unchanged**(read-only probe) |

## Current step

**Declaration PR open = seal unit.** Execution(single run)authorized only
after merge. Runner script written post-seal against the declaration
verbatim;results PR carries the committed evidence packet + exactly one
terminal.

## Read first

1. [Declaration(§20.2 block+boxes+terminals)](../../modules/semantic/research/ambiguous_band_ranking_power_probe_declaration_20260712.md)
2. [Framework §20 contract](../eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)
3. [Production substrate mapping(consumer split;s0 proxy 語意)](../../modules/semantic/research/production_substrate_mapping_20260711.md)
4. 上游 capability maps:[GT-support morphology thread](gt_support_morphology_20260711.md)(diagnostic;SUSPENDED prototype 與本 probe 的 re-charter score 方向相接)

## Acceptance

```text
seal: declaration PR merged by owner
run:  V1–V5 validity gate → H headroom → 12 candidates × B1–B6 boxes
out:  exactly one terminal(T0–T3)+ full 12-candidate capability map
      (no cherry-pick;P3 hard subset+reachable slice reported)
then: T1→step ④ design-evaluation charter;T2→step ⑤ closure statement;
      T3→step ④ closed as unnecessary;T0→fix validity blocker only
```

## Must not

- seal 前執行 ranking 計算(counts-only recon 已揭露於 declaration §8);
- seal 後偏離宣告(atoms/quantiles/metrics/boxes/terminals)——偏離=作廢重宣告;
- 以 best performer 取代 boxes;futility 後擴家族;
- 動 production preset、evidence_ledger、safe-region closed gates。

## History

- 2026-07-12: Realignment step ③ 開工。Counts-only recon(event 結構/pool 大小,無 ranking 指標):210 rankable events(gate 後 205,per-seq min 12);reachable slice(s0≤0.4)僅 34 events→降 descriptive-only。Declaration doc 完成,PR open = seal unit。
- 2026-07-12: **Owner seal review = SEAL BLOCKED(四組 terminal-affecting 自由度)→ Revision 2 落地**:①H/T3 改 joint(H1 PWA≥0.98 ∧ H2 top-1≥0.98),撤「算術不可能」錯誤理由,明定 H-over-boxes precedence;②凍結 rank tie policy(pessimistic against GT)與 quantile estimator(`numpy.quantile method="linear"`),B3 明定 metric-level re-aggregation 不重 fit threshold(refit 只在 B5);③B6 改機械判定(B6a fire-rate P(c|FP)>P(c|GT) ∧ B6b flip decomposition n_good>n_bad,均 strict),排除 reviewer 事後判斷;④T2 更名 `NO_USABLE_RANKING_POWER_IN_CLASS` 並限 Door-0 complexity class(擴 class=新宣告+owner charter),reachable-slice limit rule 改為適用全部 terminal。等第二輪 seal review。
