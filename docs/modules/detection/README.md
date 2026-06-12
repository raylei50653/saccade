# Temporal YOLO — 設計文件索引

> Option D（Track-Conditioned YOLO）已於 2026-05-19 結案 NO-GO，設計文件移至
> [docs/archive/option-d/](../../archive/option-d) 保留供歷史參考。

本目錄保留**當前 active** 的 Temporal YOLO 設計：

| 文件 | 內容 | 狀態 |
|------|------|------|
| [option-e-v2-design.md](option-e-v2-design.md) | Quality-Gated Temporal Feature Fusion | GO (MOTA 54.2%) |
| [option-f-mamba-head.md](option-f-mamba-head.md) | Mamba SSM Detection Head | active prototype |
| [mamba-head-training.md](mamba-head-training.md) | Mamba head 完整訓練流程（distill→GT-ft、版本譜系、高解析重訓）| reference |
| [research/mamba-dual-resolution-original-detail-plan.md](research/mamba-dual-resolution-original-detail-plan.md) | 640 Mamba global + 原始解析度 detail branch 研究計畫 | proposed |

## 設計演進總覽

```
Option B  →  Option C  →  Option D  →  Option E  →  Option E-v2  →  Option F
凍結 YOLO    聯合訓練     Gate+Decoder  Gate+標準     Quality-Gated    Mamba SSM
                            (NO-GO)     Head+FT       Temporal Fusion   Head
                                        IDF1 57.2%    MOTA 54.2%        active
```

## 程式碼位置

### 核心模組 (`src/saccade/perception/temporal_yolo/`)

| 模組 | 內容 |
|------|------|
| `model.py` | Option B/C 基礎架構 (TrackQueryDecoder, Lifecycle) |
| `yolo_joint.py` | Option C: YOLOFeaturePyramid + FPNSequenceProjection |
| `yolo_conditioned.py` | Option D: TrackerGateInput + TrackSpatialGate (NO-GO) |
| `yolo_gated_detector.py` | Option E: GatedYOLODetector, build_gated_yolo_detector |
| `temporal_fusion.py` | Option E-v2: TemporalFeatureFusion, AlphaTierConfig |
| `mamba_head.py` | Option F: MambaDetectionHead, MambaBlock, EmbeddingProjector |
| `mamba_gated_detector.py` | Option F: MambaGatedDetector, build_mamba_gated_detector |
| `loss.py` | AuctionMatcher, TemporalTrackingLoss |
| `dataset.py` | MOT17TemporalClip, build_mot17_dataloader |
| `dataset_joint.py` | DanceTrackTemporalClip, build_joint_dataloader |
| `reid_head.py` | ReIDHead, supcon_loss |
| `roi_embedder.py` | FPNCropEmbedder, ROIEmbeddingBank |

### 訓練腳本 (`scripts/train/temporal_yolo/`)

| 腳本 | 內容 |
|------|------|
| `train_joint.py` | Option C: Joint YOLO + Decoder |
| `train_conditioned.py` | Option D: Track-Conditioned YOLO |
| `train_gated_detector.py` | Option D revised: GatedYOLODetector |
| `train_gated_tp.py` | GatedDetector + TP Recall Loss |
| `train_mamba_head.py` | Option F: Mamba distillation (MSE) |
| `train_mamba_gt.py` | Option F: Mamba head GT fine-tuning |
| `train_jde_market.py` | JDE embedding projector on Market-1501 |

### 共享訓練基礎設施 (`src/saccade/perception/temporal_yolo/`)

| 模組 | 內容 |
|------|------|
| `training_utils.py` | `save_checkpoint`, `load_checkpoint`, `strip_compiled_keys` |
| `box_utils.py` | `xyxy_to_cxcywh_norm`, `make_yolo_batch`, `build_gate_inputs` |
| `train_config.py` | `TrainingConfig`, `add_common_training_args`, `build_optimizer_and_scheduler` |
| `data_pipeline.py` | `DataPreloader`, `FeatureCache`, `resize_letterbox_float` |

## 訓練流程規範

所有訓練腳本遵循 **3-phase pipeline**：

```
Phase 1 — PREPARE  只做一次，處理將被重複用的數據
  1a. DataPreloader  → 多線程 JPEG 解碼 → uint8 tensor 存 RAM
  1b. FeatureCache   → 預算 frozen encoder 輸出 → 存 .pt 檔，後續 epoch 跳過 encoder

Phase 2 — TRAIN    只碰 trainable 參數
  - 數據從 FeatureCache 讀取（快）或即時 encoder forward（慢）

Phase 3 — SAVE     checkpoint 由 training_utils.save_checkpoint() 統一處理
```

設計原則：
- **預算優於即時算**：frozen 的 data pipeline 應在 Phase 1 完成，Phase 2 只做 trainable 部分
- **FeatureCache 存最終輸出，不存中間特徵**：pooled embedding (384 float) 而非 FPN features (5.7MB)
- **DataPreloader 只存 uint8 原圖**：不做 resize/float，避免 RAM 浪費
- **API 命名去底線前綴**：共用函數不加 `_`（如 `save_checkpoint`），腳本私有函數加 `_`（如 `_parse_pid`）

## 歷史文件

- Option D 完整設計 (4 文件) → [docs/archive/option-d/](../../archive/option-d)
- Option D 訓練產物：`runs/conditioned_p1_v2/best.ckpt`、`runs/conditioned_p2/best.ckpt`

# Detection Module (偵測模組)

## 📐 模組職責
負責前級 YOLO 圖像目標偵測、Mamba Head 時空特徵融合與 NMS 抑制，為後續追蹤提供高置信度的 Object Box。

## 🟢 目前現況
* **Mamba SSM Head** 已作為 `mamba_optimal` 生產配置落地，採用 PixelShuffle 上取樣與 Stretch-Resize 域一致性。評測幀率達到 93.8–110.2 FPS（精度 MOTA 76.6% / IDF1 71.3%）。
* **Flow-Gated 特徵調製機制**：取代了 Option E-v2 中使用 `F.grid_sample` 對低解析度 FPN 特徵圖（如 $20 	imes 20$ P5）進行空間 Warp 導致鋸齒偽影與精度下降（MOTA -2.1pp）的問題。當前方案在 `mamba_head.py` 中將累計 GMC Affine matrix 轉成稠密流場 (Dense Flow) 後，以 `torch.cat([x_up, flow_i], dim=1)` 的通道拼接形式與特徵圖融合，並利用卷積層自適應學習空間門控調製 `(1.0 + gate)`，避開了特徵插值扭曲。
* **CUDA Graph 推理優化**：
  * 解決了自訂 CUDA 算子 `selective_scan_fwd` 運行於 Legacy Default Stream 導致 CUDA Graph 捕獲丟失 Scan Kernel 的重大 Bug（目前使用 `torch.cuda.current_stream().cuda_stream` 強制流綁定）。
  * 實現了圖安全 (Graph-Safe) 的檢測後處理解碼，使用預先計算的 Anchor 網格（`_precompute_anchor_grid`）替換了不可捕獲的 `torch.arange` / `torch.full` 動態分配算子，推理提速達 15%。

## 🔗 I/O & Dataflow

| | |
|---|---|
| **Pipeline stage** | `[3] detect` + `[4] postprocess`（6 sub-stages，見 [pipeline_flow.md](../../reference/pipeline_flow.md)） |
| **輸入** | preprocessed frame tensor（`native_960` 960×960，或 tiled `960p_2x2/3x2`）；Mamba head 額外吃 FPN 特徵 + geometry 的 GMC dense flow（時序 gate） |
| **輸出** | fused `boxes / scores / classes`（filter→NMS→cross-tile merge→quality/birth gate 後） |
| **上游 → 下游** | `streaming/[2] preprocess → [3] detect (YOLO26 + Mamba head) → [4] postprocess → [11] tracker` |

## ⚖️ GO / NO-GO 決策

> 完整脈絡見 [TODO_history.md](../../TODO_history.md)；近期待辦見 [TODO.md](TODO.md)。

| 日期 | 項目 | 結論 |
|------|------|------|
| 2026-05-27 | Option F Mamba Gated Detector（PixelShuffle + Cross-Scan + Flow-Gated） | ✅ GO，當前 preset `mamba_optimal` |
| 2026-06-02 | mamba_head CUDA graph（stream-bind fix） | ✅ GO，+15% FPS，已 default |
| 2026-05-22 | Option E-v2 Quality-Gated Temporal Fusion | ✅ GO（後被 Option F 取代） |
| — | ST-Mamba 時序 buffer | ⏸️ 單幀持平、時序不 work → 轉 VGT-Mamba |
| 2026-05-19 | Option D Track-Conditioned YOLO | ❌ NO-GO（IDF1 31.7%，gate 無貢獻） |
| 2026-05-18 | Horizontal-flip TTA | ❌ NO-GO（精度雜訊內） |
| 2026-06-01 | MOT20 混訓 | ❌ NO-GO（domain shift 退步） |
| 2026-05-10~11 | Pose box expansion / P5 birth gates / stage2 quality gate | ❌ NO-GO（靜態 FP 無法靠 spatial 區分） |

## 📋 模組 TODO

詳見 [TODO.md](TODO.md)。
