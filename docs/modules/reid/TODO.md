# ReID — 模組 TODO

> 全局進度矩陣與 Baseline 見 [docs/TODO.md](../../TODO.md)。本檔只列 ReID / Feature Bank 模組待辦。

## 待辦（暫緩 — 卡在特徵品質）

- [ ] **ReID 尋回身份 — 阻塞於特徵能力上限**：2026-06-03 全面調查確認 appearance 對 MOT17
  ID 一致性無 headroom（5 個現成模型 + online/offline/relink 機制 + SR + 域 head 訓練全 NO-GO）。
  根因：MOT17 行人在 embedding 空間本質難分（清晰大框 rank-1 也僅 57%）。完整脈絡見
  [appearance_ceiling_mot17](../../research/reid/appearance_ceiling_mot17.md)。
  - **解鎖條件**：取得**MOT/crowd 域訓練（小框+遮擋魯棒）的 ReID 特徵**（如自訓 backbone）。
  - **重測流程**：先過 `scripts/eval/reid_id_benchmark.py`（gap 31+ rank-1 明顯 > ~37%/13% 才繼續）
    → relink C++ 基建現成（`--relink-enabled`，default off）→ `scripts/eval/reconnect_rate.py` 驗收。
