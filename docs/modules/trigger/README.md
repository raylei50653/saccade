# Trigger Module (觸發機制)

## 📐 模組職責
負責非同步 ReID 特徵處理與 RAG 觸發。

## 🟢 目前現況
* `async_reid=True` 是 ReID 分支啟用時的非同步 side-stream 能力；現行 `mamba_whole_graph` headline preset 使用 `reid_mode: "off"`，因此不把 async ReID 當 headline baseline 的預設成本/收益來源。
* 整合顯存降級廣播，在 FAST_PATH 以上時自動跳過 RAG 觸發。

## 🔗 I/O & Dataflow

| | |
|---|---|
| **Pipeline stage** | `reid_budget` + ReID trigger decision（見 [pipeline_flow.md](../../reference/pipeline_flow.md)） |
| **輸入** | detections + tracker 狀態（need_reid / hit streak）+ VRAM level |
| **輸出** | 本幀 / track 是否做 embedding extraction；RAG 觸發決策 |
| **上游 → 下游** | `reid_work_enabled → fixed-interval / need_reid_frame() / DynamicReIDController.should_reid()（受 MIN_REID_GAP 限制）→ reid_crop` |

## ⚖️ GO / NO-GO 決策

> 完整脈絡見 [TODO_history.md](../../TODO_history.md)。

| 日期 | 項目 | 結論 |
|------|------|------|
| — | async_reid side-stream 提取 | ✅ GO，ReID 分支啟用時可用；headline baseline ReID off |
| 2026-04-28 | DynamicReIDController（cooldown_frames + birth_death_lost_min） | ✅ GO（`p3_t40_cd8_bm3` 精度優先最佳） |
| — | Saccade Heartbeat 影格更新隔離閘控 | ✅ GO |

## 📋 模組 TODO

詳見 [TODO.md](TODO.md)。
