# Temporal YOLO — 設計文件索引

> Option D（Track-Conditioned YOLO）已於 2026-05-19 結案 NO-GO，設計文件移至
> [docs/archive/option-d/](../../archive/option-d/) 保留供歷史參考。

本目錄保留**當前 active** 的 Temporal YOLO 設計：

| 文件 | 內容 | 狀態 |
|------|------|------|
| [option-e-v2-design.md](option-e-v2-design.md) | Quality-Gated Temporal Feature Fusion | GO (MOTA 54.2%) |
| [option-f-mamba-head.md](option-f-mamba-head.md) | Mamba SSM Detection Head | active prototype |

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

- Option D 完整設計 (4 文件) → [docs/archive/option-d/](../../archive/option-d/)
- Option D 訓練產物：`runs/conditioned_p1_v2/best.ckpt`、`runs/conditioned_p2/best.ckpt`
