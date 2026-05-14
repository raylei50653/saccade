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
| ~~P1~~ | ~~FPS anomaly 根因（match ≥ 0.73）~~ | **✅ 已結案（2026-05-11）**：全量實測（7 seq + profile-stages）確認 match=0.72 vs 0.73 差距僅 0.16ms/frame（82.3 vs 81.3 FPS），`bg_relink_wait=0.00ms`，**anomaly 已消失**。推測 2026-05-07 的 async_reid + pipeline_relink + fused letterbox 等優化將 frame total 從 ~8.4ms 壓至 12ms（其中 detect 佔 6ms），原本的臨界條件不再觸發。**`match=0.72` 安全邊界解除**。 | ✅ 已完成 |
| P3 | **Detector 訓練資料改善** | pred_h = 61.4% of gt_h，77% 近似 FP 有真實 GT；需補足腿/腳標注 | 根本解決 FN 問題；目前所有 score-gate 手段天花板已見 |

---

## Recent Ablation Conclusions（2026-05-10 ~ 05-11）

- **fuse_score_weight=0.4（2026-05-11）✅ 已設為 default（baseline preset）**
  - botsort-style：cost = 1 − IoU × (1 − 0.4 × score)，讓低信心 det 更難匹配
  - 7-seq SDP baseline preset：IDF1 +1.7pp → 51.4%，MOTA +1.6pp → 43.5%，FP **-13%**，Rcll -0.6pp，IDs -1
  - 0.4 為 Pareto 最優；≥0.75 會讓 IDs 翻倍

- **FP 模組 A：stage2_match_thresh（2026-05-11）❌ 語意反向，no-go for tuning**
  - `max_cost` 是上限；提高 = 更寬鬆 = FP 增加。0.5 在 fuse_score_weight=0.4 下已適當。
  - 參數已暴露（`--stage2-match-thresh`），保留為可調，但不調整 default。

- **FP 模組 B：birth_low_score_thresh（2026-05-11）微小增益，不設 default**
  - 出生分數 < thresh 的 track 需多一次 confirm；最佳 blst=0.28 → FP -63、IDs -3、IDF1 +0.1pp
  - 效益太小；`--birth-low-score-thresh`（default 0.0=off）保留為可調。

- **FP 模組 C：bank_weighted_mean（待測）**
  - 以 quality_score 加權 bank 樣本均值，降低低品質 embedding 影響
  - 只在 `--appearance-bank` 開啟時有效；ReID stack 測試待排。

- **P5-4 Per-sequence scene adaptive policy（2026-05-11）❌ NO-GO（narrow_bonus 策略）**
  - `SceneAdaptivePolicy` 分類邏輯正確（只有 MOT17-02 觸發 `crowded_narrow`：avg_aspect=1.97，最低）
  - 7-seq SDP speed preset：IDF1 -0.1pp，MOTA -0.2pp，IDs +5，FP +201，FN -107，FPS **-8**
  - MOT17-02 單序列：IDF1 -0.6pp（預期 +1.4pp）— 根因：speed preset `new_track_thresh=0.28` 低，bonus=0.05 把低品質框推過門檻
  - 舊版 +1.4pp 是在 `new_track_thresh=0.35` 下測得，不適用 speed preset
  - 框架保留 `--scene-adapt-enabled`（預設 off），classification 邏輯可供未來其他策略重用

- **P5-1 Multi-signal birth policy（2026-05-11）❌ NO-GO（2-run 確認）**
  - 2026-05-11 初測 7-seq：IDF1 ±0，IDs +12，FP +453，FN -530，FPS **-20**
  - 根因：Python-side O(K×C) IoU matching 每幀開銷；sub-threshold TP 比例太低
  - `--multi-birth-enabled` 保留（預設 off）
  - **2026-05-14 2-run 深度掃描（MOT17-02/10-SDP）：33 configs, 5-run avg 驗證**
    - 所有配置 MOTA 集中在 33.8–34.1%，run-to-run 波動 ±0.15–0.25%（在誤差內）
    - 最佳單策略：A `iou=0.90 ratio=1.03`（FPS 117.8，不降）、B `e=0.75 r=0.85`（FPS 92，-22%）
    - 組合策略（A+B, A+C）不如單獨使用（重疊處理同一批檢測）
    - **結論：後處理層面提升 <0.2%，不抵銷開銷。需從檢測層面下手。**
  - 新增 debug 工具：`--debug-birth-csv` + `label_boosted_birth_rows.py`（56 測試通過）
  - 856 boosted rows 全部 dropped（17.8% 真正 missed，需進一步分析）

- **P5-2 Stage 2 Quality Gate（2026-05-10）❌ NO-GO**
  - 與 `detection_quality_scaling=True` 完全重疊，零效果。只在 `--no-detection-quality-scaling` 時有意義。

- **P5-3 Consecutive-Frame Birth Gate（2026-05-10）❌ NO-GO**
  - Motion gate 有效過濾靜態 FP，但最佳結果統計中性（IDF1 +0.1pp）。sub-threshold TP 比例太低。

- **Tracklet Interpolation（2026-05-10）✅ 已設為 default**
  - `max_gap=20, min_track_len=5`：speed IDF1 +0.3pp，IDs -34，Rcll +2.0pp，FPS 不變

- **Pose-Guided Box Expansion（2026-05-10）❌ No-Go**
  - IDF1 不退但 FPS -60%（pose engine 每幀都跑），box 擴展引發 ID switches。detector training data 才是根本解。

- **FPS anomaly：match ≥ 0.73 時 FPS 驟降（2026-05-09）✅ 已結案（2026-05-11）**
  - 原症狀：`match=0.72` ~119 FPS；`match ≥ 0.73` FPS -28（+2.65ms），全 seq 受影響
  - Step 1（bench）：孤立 C++ tracker 無異常（1.3–1.5ms 全平）→ 根因在 Python
  - Step 2（實測）：7 seq `--profile-stages` 確認 0.72 vs 0.73 差距僅 0.16ms/frame，`bg_relink_wait=0.00ms`
  - **結論：anomaly 已被 2026-05-07 pipeline 優化消除（async_reid + fused letterbox）。`match=0.72` 安全邊界已解除。**

---

## Historical Links

- 歷史 TODO / 設計規範 / C++ 路線圖：[docs/TODO_history.md](/docs/TODO_history.md)
- Tracking base 與 relink sweep：[docs/experiments/tracking/fp_fn_recovery_and_gmc.md](/docs/experiments/tracking/fp_fn_recovery_and_gmc.md)
- ReID backbone refresh 歸檔：[docs/experiments/reid/semantic_relink_and_crop.md](/docs/experiments/reid/semantic_relink_and_crop.md)
