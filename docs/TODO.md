# Saccade TODO — 具體實作清單

> 主 TODO 只保留目前待辦、近期 ablation 結論與下一步方向。已完成項、設計規範與 C++ 路線圖已移至 [docs/TODO_history.md](/docs/TODO_history.md)。

---

## 歸檔標準

- 主 TODO 只保留三類內容：
  - 目前真的還要做的事項
  - 近期仍會影響決策的 ablation 結論
  - 下一輪已排定的實驗 / 實作 backlog
- 內容應移入 [docs/TODO_history.md](/docs/TODO_history.md) 的情況：
  - 已完成，且後續不再需要逐步追蹤
  - 已收斂並明確放棄，不再作為近期 default 候選
  - 已被新方向取代，只需保留背景與結論
  - 屬於長篇實作過程、舊路線圖或階段性 milestone，而不是當前待辦
- 歸檔時原則：
  - 主 TODO 保留高訊號摘要與最終結論
  - 細節、過程、舊參數掃描與已結案子項移入 history
  - 若某方向之後重新啟動，再從 history 摘回主 TODO，而不是在主 TODO 長期保留已結案脈絡

---

## 當前 Baseline（2026-05-11，最後更新）

| preset | IDF1 | MOTA | IDs | Rcll | FPS |
|--------|------|------|-----|------|-----|
| **speed**（yolo26s） | **52.0%** | **41.6%** | **475** | 55.0% | **97.9** |
| **baseline**（yolo26m） | **51.4%** | **43.5%** | **502** | 59.0% | ~85 |

已 default 的 flag：`fuse_score_weight=0.4`、`interp`、`fp_hard_filter`（area=40000）、`kalman_r_scale=0.75`、`async_reid`、`pipeline_relink`、`gmc gpu`、`detection_quality_scaling`。

---

## 待辦事項

| 優先 | 項目 | 行動 | 預期收益 |
|------|------|------|---------|
| P2 | **測試覆蓋率提升（66% → 70%+）** | 見下方覆蓋率任務清單 | 穩定性、CI 保護、開發信心 |
| P3 | **Detector 訓練資料改善** | pred_h = 61.4% of gt_h，77% 近似 FP 有真實 GT；需補足腿/腳標注 | 根本解決 FN 問題；目前所有 score-gate 手段天花板已見 |

### 測試覆蓋率任務清單（P2）

> 詳細報告：[docs/TEST_COVERAGE.md](/docs/TEST_COVERAGE.md)

| 優先 | 模組 | 覆蓋率 | 未覆蓋行 | 狀態 |
|------|------|--------|----------|------|
| P2-1 | `perception/eval/evaluator.py` | 40% | 734 | **待實作** |

**目標**：
- 🔄 短期 v4：`perception/eval/evaluator.py` (40%)
- 📋 中期：`perception/eval/evaluator.py` (40%), `perception/eval/detection.py` (49%)
- 📋 長期：API 模組、media 模組、native 測試

---

---

## 算法方向探索（2026-05-17 新增）

> 背景：2026-05-17 speed preset（yolo26s）7-seq SDP 全量評估，參數優化已觸天花板。
>
> Current results: IDF1=52.4%, MOTA=41.7%, IDs=464, FP=14,687, FN=50,356, Rcll=55.2%, FPS=131.8
>
> Key insight: yolo26s 檢測能力弱於 yolo26m（FN +3,800），但 FPS 優勢顯著（131 vs ~98）。
> 所有 threshold 調整（conf=0.02~0.05, track=0.02~0.05, nms=0.35~0.5, fuse_score=0.0~0.4）
> MOTA 波動 <0.5pp。**需要新算法架構，而非參數調校。**

### ❌ Option D — Track-Conditioned YOLO（NO-GO，2026-05-19 結案）

> Phase 2（50 epochs）eval 完成。IDF1 31.7% / MOTA 24.5% vs baseline 52.0% / 41.6%，差距 -20pp。
> Gate ablation 確認 gate 無貢獻（gate-on 38.3% vs gate-off 38.2%，∆<0.2pp）。
> 根因：100 queries recall 天花板（34.9% vs baseline 55%）+ Phase 2 gt_ratio→0 使 decoder 繞過 gate。
> Checkpoints 保留：`runs/conditioned_p1_v2/best.ckpt`、`runs/conditioned_p2/best.ckpt`。

---

### 📋 中長期 Backlog

#### 2. ReID + Appearance Bank

| 項目 | 內容 |
|------|------|
| **問題** | 遮擋後若無視覺特徵，僅靠 motion 預測容易匹配失敗 |
| **思路** | 啟用 ReID stack（siglip2 或更輕量 model），在遮擋後使用 embedding 尋回身份 |
| **狀態** | 📋 暫緩，待 Temporal YOLO 驗證後再評估是否需要疊加 |

#### 3. Detector 資料集補強與微調

| 項目 | 內容 |
|------|------|
| **問題** | yolo26s 對於被遮擋或只有腿/腳的行人檢測能力弱 |
| **思路** | 針對遮擋與小目標，使用包含更多半身/肢體標註的資料集重新微調 YOLO |
| **狀態** | 📋 暫緩，將優先觀察 Temporal YOLO 是否能透過時序資訊彌補此缺陷 |

---

## Recent Ablation Conclusions（2026-05-10 ~ 05-11）

- **FP 模組 C：bank_weighted_mean（待測）**
  - 以 quality_score 加權 bank 樣本均值，降低低品質 embedding 影響
  - 只在 `--appearance-bank` 開啟時有效；ReID stack 測試待排。

---

## Historical Links

- 歷史 TODO / 設計規範 / C++ 路線圖：[docs/TODO_history.md](/docs/TODO_history.md)
- Tracking base 與 relink sweep：[docs/experiments/tracking/fp_fn_recovery_and_gmc.md](/docs/experiments/tracking/fp_fn_recovery_and_gmc.md)
- ReID backbone refresh 歸檔：[docs/experiments/reid/semantic_relink_and_crop.md](/docs/experiments/reid/semantic_relink_and_crop.md)
