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
