# Training Experiments

每次訓練實驗一個獨立文檔。記錄：配置、架構、loss 曲線、eval 指標、結論。

## 格式

```markdown
# 實驗：簡短描述

日期：YYYY-MM-DD
狀態：進行中 / GO / NO-GO

## 命令

## 配置
| 參數 | 值 |
|------|-----|

## 架構


## 結果
| epoch | loss | metric |
|-------|------|--------|

## 結論
```

新文請加狀態標記（[契約 C3](../../ownership/doc_structure_contract.md)）：

```html
<!-- doc-status: active | parked | closed | archived -->
<!-- doc-promotion: none | ledger | report_data | archive | no_go -->
<!-- doc-date: YYYY-MM-DD -->
```

## 索引

| 實驗 | 日期 | 狀態 |
|------|------|------|
| [jde-market-1501](jde-market-1501.md) | 2026-05-25 | 進行中 (v4) |
| [pp22_full_cadence_interp_training_plan](pp22_full_cadence_interp_training_plan.md) | 2026-07 | plan |
| [pp22_stress_test_findings](pp22_stress_test_findings.md) | 2026-07 | findings |
| [training_lineage_inventory](training_lineage_inventory.md) | 2026-09-18 | captured snapshot（#421 deliverable 1；由 `scripts/provenance/training_lineage.py` 產生，不手改） |
| [training_comparison_matrix](training_comparison_matrix.md) | 2026-09-18 | generated view（#421 deliverable 2；由 `scripts/provenance/training_comparison.py` 從 inventory JSON + `training_comparisons.json` 導出，`--check` 驗新鮮度，不手改） |
| [training_eval_contract](training_eval_contract.md) | 2026-09-19 | frozen contract（#421 deliverable 3；由 `scripts/provenance/training_eval_contract.py freeze` 從 `training_eval_contract.json` 的 `declared` 區塊 + matrix JSON 導出，`check` 驗新鮮度；runner `run`/`validate-pair` 以 `run_manifest.json` v3 `runtime_identity` fail-closed；不手改 .md，只改 JSON 的 `declared`） |
| [training_eval_campaign](training_eval_campaign.md) | 2026-09-19 | run inventory（#421 deliverable 3 closeout；由 `training_eval_contract.py campaign` 從 gitignored `results/training_eval_contract/` 盤點成 `report_data/training_eval_campaign.json` + 本文；18 recipes 各 repeat n=6 + formal 3、同一 clean commit/contract/lease；只是 inventory，baseline 與 pair verdict 歸 deliverable 4 `validate-pair`；不手改） |
