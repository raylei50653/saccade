# Eval Experiment Notes

Evaluation / ablation notes: commands, metric summaries, adopt / reject, follow-ups.

**Index contract:** this table lists every note in this directory (except this README).
See [../../ownership/doc_structure_contract.md](../../ownership/doc_structure_contract.md).

## Index

| 文件 | 內容 |
|------|------|
| **[statistical_robust_feasible_set_estimation_under_asymmetric_loss.md](statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)** | **跨研究數學契約：非對稱損失下的 feasible/productive-safe set、region geometry、有限樣本、transfer 與 claim ladder** |
| **[runtime_quantity_fidelity_protocol.md](runtime_quantity_fidelity_protocol.md)** | **normative — 任何聲稱代表 production runtime quantity 的量,不得因公式名稱/形狀相似而繼承語義,須先過本流程**（由 Issue #112 提煉）· core lemma：**同一個 `f`,不同的時域化約算子 `R`** → `s0 = f(R_off(x))` ≠ `bdist = f(R_ker(x))` · 八步：宣告（threshold/ranking/morphology）→ shadow capture（**逐 byte 相同**須有 regression test）→ 版本化 key + 釘死 ID universe（local vs global）→ partition 窮盡守恆（unmatched/unemitted **不是缺值**,不得進 agreement 分母）→ 看 terminal 前封門檻 → 五項驗證（threshold agreement **confusion 不得相抵**／數值誤差**尾分位**／rank correlation／**關鍵區域** coverage／estimator-shift mechanism）→ 四分 terminal（faithful／threshold-only／rank-only／unfaithful；**不可互相補償**）→ 失敗做 **append-only amendment**,不得改歷史 sealed evidence · **例外**：純 offline domain 的 morphology/capability-map 訊號可保留,但**必須明確標示**不可轉移 · faithful replay 須逐項復刻 production estimator,不得重 fit |
| **[boolean_composition_semantics_contract.md](boolean_composition_semantics_contract.md)** | **Boolean 正式語義補丁：Ω/Θ 分型、三值 predicate、universe identity、threshold edge、role closure、canonical grammar 與 closed-loop firewall** |
| **[safe_region_asset_contract.md](safe_region_asset_contract.md)** | **R0-B RegionAsset 規範契約 — ACCEPTED**（RB1–RB9 PASS；E1 applied；R1 eng. MERGED；A0 retained；A1 research acceptance open；see [thread](../threads/closed/safe_region_assetization_20260710.md) · [R1 note](../../modules/semantic/research/safe_region_asset_r1_conversion_20260710.md) · [PR #95](https://github.com/raylei50653/saccade/pull/95) history） |
| [signal_table_schema.md](signal_table_schema.md) | 契約：A/B1/B2；§0.3 風格；§0.4 L0 safe-reject；**§0.5 Gate vs Score**；§8 能用/缺什麼 |
| **[signal_analysis_ledger.md](signal_analysis_ledger.md)** | **深度訊號分析總帳**（一 gate/訊號一列；study + note pointer） |
| [../../modules/semantic/research/m_b1_research_history_20260709_20260710.md](../../modules/semantic/research/m_b1_research_history_20260709_20260710.md) | **M-B1 offline phase CLOSED** — 方法摘要見 §1–§14；原始方法細節由 source blob SHA 回溯（見 consolidation report） |
| [m_b1_substrate_smoke_20260709.md](m_b1_substrate_smoke_20260709.md) | D1：`mamba_whole_graph_m` offline_relink/B1 資料規格煙測（pointer → study_dir） |
| [m_b2_reconnect_bridge_ab_20260709.md](../../modules/semantic/research/m_b2_reconnect_bridge_ab_20260709.md) | D3：m B2 bridge on/off reconnect + e2e（study_dir master） |
| [fp_classifier_external_only_plan.md](fp_classifier_external_only_plan.md) | 0-shot FP classifier 設計 |
| [gmc_residual_correction_20260612.md](gmc_residual_correction_20260612.md) | GMC residual 共模修正 ablation |
| [kalman_h_recalibration_20260612.md](kalman_h_recalibration_20260612.md) | Kalman H 重校準 |
| [neutral_nogo_signal_attribution_20260612.md](neutral_nogo_signal_attribution_20260612.md) | Neutral NO-GO 訊號歸因 |
| [oao_duration_ramp_revival_20260617.md](oao_duration_ramp_revival_20260617.md) | OAO duration-ramp 復活筆記 |
| [scoring_energy_20260630.md](scoring_energy_20260630.md) | Scoring energy 分析 |
| [wsl2_d2h_pinned_memory_leak_20260517.md](wsl2_d2h_pinned_memory_leak_20260517.md) | WSL2 D2H pinned memory leak post-mortem |

## Archived related

- [Rerank Phase 2](../../archive/rerank_phase2.md)
- [MOT17 FP/FN Distribution Snapshot](../../archive/mot17_fp_fn_distribution_20260514.md)
