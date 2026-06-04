# ReID Module (外觀特徵與去重)

## 📐 模組職責
負責目標裁剪 (Crop)、特徵向量提取 (SigLIP 2) 與 Feature Bank 長期外觀特徵庫維護。

## 🟢 目前現況
* **特徵提取器**已升級至 **SigLIP 2**，具備更高的對比學習表徵能力。
* **Feature Bank (向量化版) 底層實現**：
  * 使用預先配置的固定大小張量（`self.features = torch.zeros((max_ids, feat_dim))`）來存儲外觀特徵。
  * 提供 `_slot_map: Dict[Tuple[int, int], int]` 字典，實現從 `(track_id, stream_id)` 到張量 Row Index 的 $O(1)$ 雙重哈希索引槽尋找，完全消除了呼叫 GPU `nonzero()` 產生的同步開銷。
  * `find_matches_batch` 與 `find_cross_camera_matches`：使用向量化矩陣乘法 `torch.mm(queries, targets.t())` 並行計算所有待關聯目標與 Feature Bank 的餘弦相似度。跨相機匹配使用更高的嚴格相似度門檻（`self.threshold + 0.03`）。
* ⚠️ **ReID + Appearance Bank 暫緩執行**：待 Temporal YOLO 驗證後，再行評估是否疊加。
* ℹ️ **Cheb-GR re-ranking 不屬本模組**：core 程式碼物理上放在 `perception/reid/cheb_gr.py`（commit 也 tag `reid`），但功能為 re-ranking / tracklet 關聯，歸 [semantic](../semantic/README.md)；本模組維持暫緩，不因 Cheb-GR 重啟。

## 🔗 I/O & Dataflow

| | |
|---|---|
| **Pipeline stage** | `[5] reid_bank_sync` → `[6] budget` → `[7] crop` → `[8] extract`（目前 default OFF，見 [pipeline_flow.md](../../reference/pipeline_flow.md)） |
| **輸入** | detections + 原圖（ROI crop） |
| **輸出** | appearance embedding（SigLIP 2，384-d）→ Feature Bank + tracker association |
| **上游 → 下游** | `detections →(trigger 決策)→ [7] crop → [8] SigLIP2 extract → embedding → bank / semantic 關聯`；⚠️ 整條 ReID 分支在 GMC 之前 |

> 職責分界：**reid** = 特徵抽取 + Feature Bank；用這些特徵做關聯 / rerank 的是 [semantic](../semantic/README.md)，觸發決策是 [trigger](../trigger/README.md)。

## ⚖️ GO / NO-GO 決策

> 完整脈絡見 [TODO_history.md](../../TODO_history.md)；近期待辦見 [TODO.md](TODO.md)。

| 日期 | 項目 | 結論 |
|------|------|------|
| — | SigLIP 2 升級 | ✅ GO，特徵表徵更強 |
| 2026-04-28 | Dynamic ReID trigger（cooldown + birth/death） | ✅ GO |
| 2026-05-02 | LaSt-ViT pre-hoc embedding quality | ❌ NO-GO（+0.09pp，SigLIP2 未訓練無區分力）；kernels 保留 |
| 2026-05-19 | ROI FPN embedding ReID | ❌ NO-GO（cos_thr 全設定 IDs↑、IDF1 持平） |
| 2026-06-03 | **Appearance 能力上限調查（5 模型 + 4 機制 + SR + 域訓練）** | ❌ **NO-GO 結案**：MOT17 身份在 embedding 空間本質難分（清晰 200+px 框 rank-1 僅 57%、intra-inter gap ~0.03、長 gap rank-1 崩 10–37%）；換模型/加機制/SR 皆撞同一上限。細節：[appearance_ceiling_mot17](../../research/reid/appearance_ceiling_mot17.md) |
| — | ReID stack 疊加 | ⏸️ 暫緩，待 **MOT-域訓練的 ReID 特徵**（非 Market）；先過 `reid_id_benchmark.py` gate |

## 📋 模組 TODO

詳見 [TODO.md](TODO.md)。
