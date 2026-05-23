# Temporal YOLO — 設計文件索引

> Option D（Track-Conditioned YOLO）已於 2026-05-19 結案 NO-GO，設計文件移至
> [docs/archive/option-d/](../archive/option-d/) 保留供歷史參考。

本目錄保留**當前 active** 的 Temporal YOLO 設計：

| 文件 | 內容 | 狀態 |
|------|------|------|
| [option-e-v2-design.md](option-e-v2-design.md) | Quality-Gated Temporal Feature Fusion | ✅ GO (MOTA 54.2%) |
| [option-f-mamba-head.md](option-f-mamba-head.md) | Mamba SSM Detection Head | 🔄 active prototype (`feat/option-f-mamba` branch) |

## 設計演進總覽

```
Option B  →  Option C  →  Option D  →  Option E  →  Option E-v2  →  Option F
凍結 YOLO    聯合訓練     Gate+Decoder  Gate+標準     Quality-Gated    Mamba SSM
                            (NO-GO)     Head+FT       Temporal Fusion   Head
                                        IDF1 57.2%    MOTA 54.2%        active
```

## 程式碼位置

```
src/saccade/perception/temporal_yolo/
├── model.py              # Option B/C 基礎架構 (TrackQueryDecoder, Lifecycle)
├── yolo_joint.py         # Option C: YOLOFeaturePyramid + FPNSequenceProjection
├── yolo_conditioned.py   # Option D: TrackerGateInput + TrackSpatialGate (NO-GO)
├── loss.py               # AuctionMatcher
├── dataset.py            # MOT17TemporalClip
└── evaluator.py          # MOTREvaluator

train/temporal_yolo/
├── train_joint.py
├── train_conditioned.py
└── configs/
```

## 歷史文件

- Option D 完整設計 (4 文件) → [docs/archive/option-d/](../archive/option-d/)
- Option D 訓練產物：`runs/conditioned_p1_v2/best.ckpt`、`runs/conditioned_p2/best.ckpt`
