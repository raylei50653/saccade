# TF32-off head drift stage localization：r1 結案（無 terminal）

<!-- evidence-tier: exploratory -->

> 狀態：**結案，無 terminal**。本研究不再執行任何 attempt。後繼研究：[`tf32_off_head_drift_stage_465_r2`](../tf32_off_head_drift_stage_465_r2/declaration.md)。

- **Attempt 001**：`invalid`，criterion `V_FORMAT`，`terminal: null`（[attempts/001](attempts/001/attempt.json)）。freeze commit `763db72d`，tag `freeze/tf32_off_head_drift_stage_465/1`。
- runner 在 `V_FORMAT` 階段停止，**沒有執行任何 arm 間比較**，沒有產生任何比較量。
- 原因：宣告的 `V_FORMAT` 要求 probe row 為 x2 ≥ x1、y2 ≥ y1，這個 xyxy 排序假設不符合 frozen input。study.yaml 在 freeze 後不可修改，重跑只會在同一處失敗，因此不再嘗試 attempt 002／003。
- 修正後的方法由新的宣告處理（r2），本研究的宣告與 attempt 保持原樣。
