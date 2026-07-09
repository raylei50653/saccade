# Temporal YOLO — 設計文件索引

> **detection 特例（Doc Structure C2）：** 本檔為「協議/程式碼索引庫 + 下方模組卡」雙結構，其餘模組僅用一頁卡片。
>
> Option D（Track-Conditioned YOLO）已於 2026-05-19 結案 NO-GO，設計文件移至
> [docs/archive/option-d/](../../archive/option-d) 保留供歷史參考。
### 設計 / 訓練協議（頂層）

| 文件 | 內容 | 狀態 |
|------|------|------|
| [option-f-mamba-head.md](option-f-mamba-head.md) | Mamba SSM Detection Head lineage（現行 baseline 已升到 `mamba_whole_graph`）| active |
| [mamba-head-training.md](mamba-head-training.md) | Mamba head 完整訓練流程（distill→GT-ft、版本譜系、高解析重訓）| reference |
| [mamba-v14r-training-protocol.md](mamba-v14r-training-protocol.md) | v14-R canonical protocol、split/seed、選模與 provenance 規範 | canonical |
| [mamba-v14-replication-protocol.md](mamba-v14-replication-protocol.md) | legacy v14 復刻協議（刻意保留 02 洩漏結構，驗證 recipe 可重現性）| reference |
| [mamba_whole_graph_analysis.md](mamba_whole_graph_analysis.md) | mamba_whole_graph production baseline 深度分析報告 | analysis |
| [option-e-v2-design.md](option-e-v2-design.md) | Quality-Gated Temporal Feature Fusion（已被 Option F 取代）| superseded |

### research/ — 分析與計畫

| 文件 | 內容 | 狀態 |
|------|------|------|
| [research/mamba-t3t1-curriculum-20260613.md](research/mamba-t3t1-curriculum-20260613.md) | T3→T1 temporal-consistency curriculum（首超 v14，IDF1 75.4）| ✅ GO |
| [research/mamba-score-distribution-20260613.md](research/mamba-score-distribution-20260613.md) | 分數分佈歸因（95k GT）：飽和左尾、框高主導、人群密度走漏檢非壓分；門檻 sweep → ntt0.20 過擬合撤回(維持0.28)、框高條件化雙向 NO-GO（registry #38） | ❌ 結案 |
| [research/mamba-cuda-graph-bug.md](research/mamba-cuda-graph-bug.md) | CUDA-graph eval bug 根因（selective_scan stream-bind fix）| ✅ 已修 |
| [research/whole-graph-kernel-fragmentation.md](research/whole-graph-kernel-fragmentation.md) | nsys kernel fragmentation 分析（~372 kernels/frame，fragmentation-bound）| analysis |
| [research/kernel-fusion-plan.md](research/kernel-fusion-plan.md) | elementwise kernel fusion 計畫（對應上文 fragmentation）| proposed |
| [research/mamba-dual-resolution-original-detail-plan.md](research/mamba-dual-resolution-original-detail-plan.md) | 640 Mamba global + 原始解析度 detail branch 研究計畫 | proposed |
| [research/mamba-strip-detail-routing-design.md](research/mamba-strip-detail-routing-design.md) | 小目標 strip detail routing 設計（registry #36 ROI NO-GO，parked）| ⏸ parked |
| [research/holdout_generalization_plan.md](research/holdout_generalization_plan.md) | Holdout / generalization 計畫 | plan |

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
| `training_utils.py` | checkpoint、seed/split、warmup+cosine、RNG resume、artifact hash |
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
- Option D historical training artifacts：runs/conditioned_p1_v2/best.ckpt、runs/conditioned_p2/best.ckpt（目前不在 tree）

# Detection Module (偵測模組)

## 📐 模組職責
負責前級 YOLO 圖像目標偵測、Mamba Head 時空特徵融合與 NMS 抑制，為後續追蹤提供高置信度的 Object Box。

## 🟢 目前現況
* **Mamba SSM Head** 是現行 `mamba_whole_graph` baseline 的 detector lineage。`mamba_optimal` 是其上一代 head-CUDA-graph preset；目前 headline preset 改用 whole-detect CUDA graph、`native_640`、ReID off，凍結指標為 IDF1 78.2 / MOTA 78.4 / HOTA 70.2 / AssA 69.7 / IDs 413 / 269.47 FPS（`frozen_v2`, 2026-06-21）。
* **Flow-Gated 特徵調製機制**：取代了 Option E-v2 中使用 `F.grid_sample` 對低解析度 FPN 特徵圖（如 $20 	imes 20$ P5）進行空間 Warp 導致鋸齒偽影與精度下降（MOTA -2.1pp）的問題。當前方案在 `mamba_head.py` 中將累計 GMC Affine matrix 轉成稠密流場 (Dense Flow) 後，以 `torch.cat([x_up, flow_i], dim=1)` 的通道拼接形式與特徵圖融合，並利用卷積層自適應學習空間門控調製 `(1.0 + gate)`，避開了特徵插值扭曲。
* **CUDA Graph 推理優化**：
  * 解決了自訂 CUDA 算子 `selective_scan_fwd` 運行於 Legacy Default Stream 導致 CUDA Graph 捕獲丟失 Scan Kernel 的重大 Bug（目前使用 `torch.cuda.current_stream().cuda_stream` 強制流綁定）。
  * 實現了圖安全 (Graph-Safe) 的檢測後處理解碼，使用預先計算的 Anchor 網格（`_precompute_anchor_grid`）替換了不可捕獲的 `torch.arange` / `torch.full` 動態分配算子，推理提速達 15%。

## 🔗 I/O & Dataflow

| | |
|---|---|
| **Pipeline stage** | `detect` + `postprocess`（現行 stage name 見 [pipeline_flow.md](../../reference/pipeline_flow.md)） |
| **輸入** | 現行 `mamba_whole_graph` baseline 使用 `native_640` / `preprocess=none`；`native_960` 與 tiled `960p_2x2/3x2` 是 legacy comparison / ablation 路線。Mamba head 額外吃 FPN 特徵 + geometry 的 GMC dense flow（時序 gate） |
| **輸出** | fused `boxes / scores / classes`（filter→NMS→cross-tile merge→quality/birth gate 後） |
| **上游 → 下游** | `streaming/ingest_preprocess → detect (YOLO26 + Mamba head) → postprocess → track` |

## ⚖️ GO / NO-GO 決策

> 完整脈絡見 [TODO_history.md](../../TODO_history.md)；近期待辦見 [TODO.md](TODO.md)。

| 日期 | 項目 | 結論 |
|------|------|------|
| 2026-06-18 | `mamba_whole_graph` production baseline | ✅ headline preset：whole-detect CUDA graph + T3→T1 ckpt lineage，native_640，ReID off |
| 2026-05-27 | Option F Mamba Gated Detector（PixelShuffle + Cross-Scan + Flow-Gated） | ✅ GO，`mamba_optimal` lineage；已被 `mamba_whole_graph` supersede |
| 2026-06-02 | mamba_head CUDA graph（stream-bind fix） | ✅ GO，+15% FPS，已 default |
| 2026-06-13 | T3→T1 temporal-shaping curriculum | ✅ promoted into current `mamba_whole_graph` lineage via `runs/mamba_gt_v14replica_t3_t1/best.ckpt`; 原始 research note 仍保留當時的 multi-seed/clean-split caveat。見 [research/mamba-t3t1-curriculum-20260613.md](research/mamba-t3t1-curriculum-20260613.md) |
| 2026-06-12 | v14 全鏈復刻（recipe 可重現性驗證） | ✅ 結案：replica IDF1 73.4 vs legacy 75.1，殘差 ~2pp = seed/scheduler 不可恢復因素；v14 非 lucky checkpoint。**衍生**兩條未結案線（T3→T1 上、frozen-SSM「unfreeze 無明顯變化」）。見 [mamba-v14-replication-protocol.md](mamba-v14-replication-protocol.md) |
| 2026-05-22 | Option E-v2 Quality-Gated Temporal Fusion | ✅ GO（後被 Option F 取代） |
| — | ST-Mamba 時序 buffer | ⏸️ 單幀持平、時序不 work → 轉 VGT-Mamba |
| 2026-05-19 | Option D Track-Conditioned YOLO | ❌ NO-GO（IDF1 31.7%，gate 無貢獻） |
| 2026-05-18 | Horizontal-flip TTA | ❌ NO-GO（精度雜訊內） |
| 2026-06-01 | MOT20 混訓 | ❌ NO-GO（domain shift 退步） |
| 2026-05-10~11 | Pose box expansion / P5 birth gates / stage2 quality gate | ❌ NO-GO（靜態 FP 無法靠 spatial 區分） |
| 2026-06-13 | 小目標高解析度恢復（B1-H dense / 1024 unified / strip-oracle routing） | 🔻 ROI NO-GO（成本判定 · parked，registry [#36](../../reference/no_go_registry.md)）；增益天花板 <0.5pp IDF1 vs 高解析度部署成本，strip Phase 1 已實作 default off，設計見 [research/](research/mamba-strip-detail-routing-design.md) |

## 📋 模組 TODO

詳見 [TODO.md](TODO.md)。
