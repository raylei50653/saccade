# Eval Experiment Notes

Evaluation / ablation notes: commands, metric summaries, adopt / reject, follow-ups。**本目錄只放筆記與總帳。**

> **規範層不在這裡。** 數學框架（feasible set / claim ladder / independence unit）、
> runtime 量的 fidelity protocol、gate-vs-score 分層、Boolean 組合語義、RegionAsset
> 打包契約 → **[../contracts/](../contracts/README.md)**。
> 開 gate / safe-region / reject-rule 研究前先讀那裡，**不要自造統計**。

**Index contract:** this table lists every note in this directory (except this README).
See [../../ownership/doc_structure_contract.md](../../ownership/doc_structure_contract.md).
標為 *(pointer)* 的列位於其他目錄，於此僅作導覽。

## Index

| 文件 | 內容 |
|------|------|
| **[signal_analysis_ledger.md](signal_analysis_ledger.md)** | **深度訊號分析總帳**（一 gate/訊號一列；study + note pointer）· 契約＝[../contracts/signal_table_schema.md](../contracts/signal_table_schema.md) |
| [procedures/gt_support_morphology_procedure_v1.md](procedures/gt_support_morphology_procedure_v1.md) | GT-support morphology 執行 procedure v1（study-specific；規範層見 [../contracts/](../contracts/README.md)） |
| *(pointer)* [../../modules/semantic/research/m_b1_research_history_20260709_20260710.md](../../modules/semantic/research/m_b1_research_history_20260709_20260710.md) | **M-B1 offline phase CLOSED** — 方法摘要見 §1–§14；原始方法細節由 source blob SHA 回溯（見 consolidation report） |
| [m_b1_substrate_smoke_20260709.md](m_b1_substrate_smoke_20260709.md) | D1：`mamba_whole_graph_m` offline_relink/B1 資料規格煙測（pointer → study_dir） |
| *(pointer)* [m_b2_reconnect_bridge_ab_20260709.md](../../modules/semantic/research/m_b2_reconnect_bridge_ab_20260709.md) | D3：m B2 bridge on/off reconnect + e2e（study_dir master） |
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
