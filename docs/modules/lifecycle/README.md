# Track Lifecycle Module (軌跡生命週期模組)

## 📐 模組職責
負責軌跡生命週期狀態機（Tentative / Confirmed / Lost）管理、死鎖軌跡驅逐與底層 Tracker LRU 釋放。

## 🟢 目前現況
* 支援 `AsyncDispatcher` 的 LRU Tracker 機制（預設最大 8 個串流），超限時自動析構釋放 CUDA 顯存。
* ID 穩定性過濾器（Hits & IoU 門檻）已在評測線路上收斂。

## 🔗 I/O & Dataflow

| | |
|---|---|
| **Pipeline stage** | `[11] track` 內狀態機 + Tracker LRU（見 [pipeline_flow.md](../../reference/pipeline_flow.md)） |
| **輸入** | track hits / age / score / IoU |
| **輸出** | Tentative / Confirmed / Lost 狀態 + Tracker LRU 釋放（預設 max 8 streams） |
| **上游 → 下游** | `tracker update → id_stability filter (hits & IoU) → 狀態機 → LRU evict / output id` |

## ⚖️ GO / NO-GO 決策

> 完整脈絡見 [TODO_history.md](../../TODO_history.md)。

| 日期 | 項目 | 結論 |
|------|------|------|
| 2026-05-05 | D2-B new_track_threshold | ✅ GO，IDF1 44.0% / IDs 515 |
| — | id_stability filter（hits & IoU 門檻） | ✅ default |
| 2026-05-18 | P5-5 Proximity Birth Gate | ❌ NO-GO（prox=0.3 已 FN +1038 / Rcll -5.6pp） |
| 2026-05-11 | P5-4 Scene-Adaptive / P5-3 Consecutive birth gate | ❌ NO-GO / 統計中性 |

## 📋 模組 TODO

詳見 [TODO.md](TODO.md)。
