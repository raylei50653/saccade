# FP Classifier External-Only Plan

日期：2026-05-13

## 目標

建立一條 **不使用 MOT17 eval/test sequence 標註做訓練** 的 `FP/TP` classifier 路線，讓最終在 MOT17 上的效果可視為 `external-only` 驗證，而不是 in-domain fitting。

## 0-shot 定義

在這個 repo 裡，`0-shot` 指的是：

- classifier 訓練 **不能** 使用 `MOT17` 最終評估 sequence 的 GT
- classifier 訓練 **不能** 使用由同一批 `MOT17` sequence 產生的 `TP/FP` 標籤
- `MOT17` 只用來做最後 inference / evaluation

這裡的 `0-shot` 不是指 foundation-model 式的純 prompt zero-shot，而是指：
**對最終 MOT17 驗證集零訓練暴露（zero training exposure）**。

## 為什麼要 external-only

之前的 in-domain `FP/TP` classifier prototype 雖然有效，但有明確 leakage：

- 用 `MOT17` 同一批 sequence 的 GT 產生 `TP/FP` 標籤
- 再回頭改善同一批 sequence 的 tracking 結果

這種作法可用來驗證方向是否有訊號，但不適合作為正式 baseline 或 default preset 依據。

## 資料來源優先順序

1. `CrowdHuman`
2. `CrowdHuman + CityPersons`
3. `COCO person` 僅作補充，不作主訓練來源

理由：

- `CrowdHuman` 對密集行人、遮擋、重疊框更接近 MOT17
- `CityPersons` 與街景監控的幾何與尺度分佈更接近 MOT
- `COCO` 太通用，行人場景密度不夠高

## 訓練資料生成原則

不要直接拿外部資料的 GT box 訓練 classifier。正確流程是：

1. 用實際部署的 detector 跑 `CrowdHuman / CityPersons`
2. 把 detector predictions 與外部 GT 做 match
3. 產生 detection-level `TP/FP` 標籤
4. 抽取與線上 pipeline 一致的 pre-tracker features

建議保留的 feature：

- `score`
- `width`
- `height`
- `aspect_ratio`
- `area_ratio`
- `center_x_norm`
- `center_y_norm`
- `edge_margin_norm`
- `touches_edge`
- `geometry_quality`

## 實驗流程

### Phase 1：external dataset adapter

- 支援 `CrowdHuman`
- 支援 `CityPersons`
- 統一輸出成與現有 eval pipeline 相容的 detection-level rows

### Phase 2：external-only classifier training

- train on `CrowdHuman` only
- train on `CrowdHuman + CityPersons`
- 先只做 pre-tracker structural classifier

### Phase 3：MOT17 inference-only evaluation

在 `MOT17 SDP 7 seq` 上只做：

- baseline
- classifier score penalty variants

不使用 MOT17 label 參與 classifier 訓練。

### Phase 4：generalization check

若 external-only 版本在 MOT17 有穩定收益，再補：

- `DanceTrack`
- `SportsMOT`

確認不是只對 MOT17 有利。

## 成功條件

external-only 版本至少要滿足：

- `FP` 明顯下降
- `IDF1 / MOTA` 不退
- 不依賴 MOT17 eval/test labels

若只能在 in-domain 有效、external-only 無法複現，則不能升為 default。

---

## 實驗結果摘要（2026-05-14 更新）

### CrowdHuman External-Only Results

使用 `yolo26m` detector + `CrowdHuman val` GT，產生 detection-level rows：

| 方法 | Precision | Recall | FP kept | FP reduction |
|------|-----------|--------|---------|-------------|
| 原始 | 15.0% | 100% | 386,652 | 0% |
| Rule baseline (default) | 52.4% | 76.8% | 47,661 | 87.7% |
| Logistic classifier | 82.2% | 53.6% | 13,064 | 96.6% |
| **Cascade (Stage1+Stage2)** | **59.6%** | **72.4%** | **33,533** | **91.3%** |

Cascade filter 為目前最佳平衡：
- Stage 1 rule baseline（零成本）砍掉 87.7% FP
- Stage 2 logistic classifier（在 Stage 1 輸出上訓練）再砍 9.5% FP
- Stage 2 僅對 ~100K rows 推理，FPS 影響極小

實作：`scripts/eval/train_cascade_stage2.py` + `models/external_fp/cascade_stage2_logistic.json`

### MOT17 Generalization Check（2026-05-14）

**結論：Cascade 不適用 MOT17**

MOT17 FP 分佈與 CrowdHuman 截然不同，導致 cascade filter 完全失效：

| 特徵 | CrowdHuman FP | MOT17 FP |
|------|---------------|----------|
| score 中位數 | 0.008 | **0.269**（34x 差距） |
| height 中位數 | 66px | **122px**（比 TP 高） |
| score TP/FP 差距 | 0.457 | **0.116** |

Rule baseline 在 MOT17 僅砍掉 13.3% FP（vs CrowdHuman 87.7%）。
Cascade model（CrowdHuman-trained）泛用效果：P=4.5%, R=84.4%, FPrem=37.2%。

**根本原因**：MOT17 的 YOLO FP 品質遠高於 CrowdHuman，FP 分數與 TP 幾乎完全重疊（中位數差距僅 0.116）。不存在「低分 FP」區可被 rule 或 logistic 分離。

**後續策略**：若要在 MOT17 上應用 cascade，需重新訓練 Stage 2 model（使用 MOT17 detector output），不適用 CrowdHuman-trained 模型。
