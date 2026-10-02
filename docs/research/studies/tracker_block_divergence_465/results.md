# Tracker block divergence：r1 結案（無 terminal）

<!-- evidence-tier: exploratory -->

> 狀態：**結案，無 terminal**（`INVALID_V_REPEAT / NO_TERMINAL`）。本研究不再執行任何 attempt。後繼研究：[`tracker_block_divergence_465_r2`](../tracker_block_divergence_465_r2/declaration.md)。

- **Attempt 001**：`invalid`，criterion `V_REPEAT`，`terminal: null`（[attempts/001](attempts/001/attempt.json)）。freeze commit `68bf34bb`，tag `freeze/tracker_block_divergence_465/1`。raw record `results/tracker_block_divergence_465/20261002T030926Z`（`MANIFEST.json` SHA-256 `d6090e7e…`）。
- 五個 replay 全部完成，`V_REPLAY` 與 `V_RECORD` 成立；`V_REPEAT` 在 MOT17-02 f2（`R_C#1` 對 `R_C#2` 的 dump bytes）失敗。runner 在 `V_REPEAT` 停止，**沒有計算任何 R_C 對 R_T／R_E 的比較**。attempt 001 的任何跨 arm 觀測都不是任何 terminal 的 evidence。
- 原因：宣告的 `V_REPEAT` 要求 dump 段逐位元相同，但 `CND` 列在同一個 track 內的順序來自 `atomicAdd` 分配的 candidate slot（`src/tracking/tracker_gpu.cu`），producer 沒有承諾這個順序。study.yaml 在 freeze 後不可修改，重跑會在同一類位置再失敗，因此不再嘗試 attempt 002／003。
- 修正後的 repeat equivalence 由新的宣告（r2）處理。本研究的宣告、tag 與 attempt 保持原樣；r2 的規則不回溯套用到 attempt 001，它不會被重新判為 valid。
