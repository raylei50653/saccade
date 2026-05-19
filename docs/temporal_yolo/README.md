# Temporal YOLO — 設計文件索引

> **[結案 2026-05-19 — NO-GO]** Option D 實作並訓練完成（Phase 1+2），但 gate 無實質貢獻（∆ <0.2pp），
> IDF1 31.7% vs baseline 52.0%，差距 -20pp。根因：100 queries recall 天花板 + Phase 2 gt_ratio→0。
> Checkpoints 保留：`runs/conditioned_p1_v2/best.ckpt`、`runs/conditioned_p2/best.ckpt`。
> **本目錄保留作設計參考，不代表當前開發方向。**

本目錄記錄「讓 YOLO 具備時序感知能力」的完整設計演進，
從最初的 Option B（凍結 backbone）到 Option D（外部 tracker 回饋驅動特徵提取）。

## 文件

| 文件 | 內容 |
|------|------|
| [architecture.md](architecture.md) | **五個選項**的完整架構對比與技術細節（含 Option E 實測結果）|
| [pipeline.md](pipeline.md) | Option C/D 完整資料流：tensor shapes、跨幀狀態、訓練迴圈 |
| [track-conditioned-design.md](track-conditioned-design.md) | Option D 詳細設計：TrackerGateInput + TrackSpatialGate |
| [implementation-plan.md](implementation-plan.md) | 分階段實作計畫、風險與里程碑 |

## 設計演進總覽

```
Option B  →  Option C  →  Option D  →  Option E
凍結 YOLO    聯合訓練     Gate+Decoder  Gate+標準Head+Fine-Tune
(已實作)     (已實作)     (NO-GO)       (✅ 當前 baseline, IDF1 57.2%)
```

### 核心問題

標準 YOLO 對每幀「從零開始」——它不知道上一幀哪裡有人在被追蹤。
現有 ByteTrack tracker 輸出的空間先驗（位置 + Kalman 速度）完全沒有被特徵提取利用到。

**Option D 的目標**：把 ByteTrack 的 `TrackerGateInput`（確認軌跡 boxes + Kalman vx/vy）
渲染成每尺度 Gaussian heatmap，乘到 YOLO FPN 輸出上，
讓 backbone 在已知目標位置（及其預測下一幀位置）產生更強的特徵。

Option D **不使用 Track Queries** 作為 gate 輸入——Track Queries 訓練初期不可靠，
會污染訓練信號；ByteTrack 已是穩定的外部系統，從 day 1 就乾淨。

## 程式碼位置

```
src/saccade/perception/temporal_yolo/
├── model.py              # Option B/C 基礎架構（TrackQueryDecoder, Lifecycle）
├── yolo_joint.py         # Option C：YOLOFeaturePyramid + FPNSequenceProjection
├── yolo_conditioned.py   # Option D：TrackerGateInput + TrackSpatialGate（待實作）
├── loss.py               # AuctionMatcher（移植自 auction.hpp，取代 scipy）
├── dataset.py            # MOT17TemporalClip（支援 track_cache_dir 待擴充）
└── evaluator.py          # MOTREvaluator

train/temporal_yolo/
├── train_joint.py        # Option C 訓練腳本
├── train_conditioned.py  # Option D 訓練腳本骨架（等 yolo_conditioned.py）
└── configs/
    ├── joint.yaml
    └── conditioned.yaml

scripts/train/
└── train_temporal_yolo.py  # 舊版統一入口（--mode frozen|joint），仍可用但新開發走 train/
```
