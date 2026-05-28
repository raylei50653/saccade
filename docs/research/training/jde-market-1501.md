# JDE Market-1501 訓練 — 架構迭代記錄

> 記錄 JDE embedding projector 在 Market-1501 上的訓練架構演進。
> 每一版記錄：變更內容、設計理由、結果、結論。

---

## v1 — letterbox + global pool, 只訓練 projector

**日期**：2026-05-25（前）
**命令**：
```bash
uv run train/temporal_yolo/train_jde_market.py \
    --yolo-weights models/yolo/yolo26s.pt \
    --teacher-ckpt runs/gated_det_v1/best.ckpt \
    --mamba-ckpt runs/mamba_gt_960_v2/best.ckpt \
    --market-root datasets/Market-1501-v15.09.15 \
    --run-dir runs/jde_market_v1 \
    --epochs 30 --batch-size 64 --lr 3e-4
```

**架構**：
```
Market-1501 crop → letterbox 640×640 → teacher YOLO → Mamba head
    → pool_embeddings_global (spatial mean) → projector (384→256→128) → SupCon
```
- `--train-emb-head`：否（預設關閉）
- emb_head：frozen，隨機初始化
- projector：trainable

**結果**：
| epoch | loss |
|-------|------|
| 1 | 3.95 |
| 30 | 2.83 |

loss 收斂良好，但 Market-1501 eval rank-1 = 0.12%（隨機 0.13%），embedding 完全無效。

**結論**：`pool_embeddings_global` 在 letterbox 後的 640×640 圖上做 spatial mean — 95% 區域是 padding，人體訊號被淹沒。emb_head 隨機且 frozen，projector 無法學到有意義的特徵。

---

## v2 — 同 v1，加 FeatureCache 加速

**日期**：2026-05-25
**命令**：同上，加 `--precompute`

**變更**：
- 新增 `DataPreloader` + `FeatureCache`（預算 frozen encoder 輸出 → 19 MB .pt）
- 訓練時直接讀預算 embedding，每 epoch 0.7s

**結果**：
| epoch | loss |
|-------|------|
| 1 | 3.95 |
| 30 | 2.83 |

rank-1 = 0.18%（隨機 0.13%），仍無效。架構問題與 v1 相同，FeatureCache 只加速了瓶頸計算，不改變結果。

---

## v3 — RoIAlign pool + 訓練 emb_head

**日期**：2026-05-25
**命令**：同上，預設 `--train-emb-head`，加 `resize_letterbox_batch_gpu_with_bbox`

**變更**：
- `pool_embeddings_global` → `pool_embeddings(bbox)`（RoIAlign output_size=1 精確擷取 person 區域）
- `--train-emb-head` 預設為 True（emb_head 隨機初始化必須被訓練）
- 新增 `resize_letterbox_batch_gpu_with_bbox`：letterbox 同時記錄 person bbox 坐標
- FeatureCache encoder_fn 也改用 RoIAlign

**結果**：

所有 epoch loss 鎖死在 **4.1431**。

```
epoch  1  best_loss=4.1431
epoch  2  best_loss=4.1431
...
epoch 40  best_loss=4.1431
```

**根因分析**：

4.1431 = ln(63) = ln(batch_size - 1)。RoIAlign 對隨機初始化的 per-pixel embeddings 做 spatial average → 每個 person crop 的 pooled embedding 幾乎完全相同 → projector 輸出常數向量 → SupCon 收斂到最大熵基線（所有 embedding 相等，無法區分任何 ID）。

emb_head 是一個 tiny 2-layer CNN（256→128→128），從隨機初始化開始，RoIAlign 的梯度訊號太弱無法推動學習。

**結論**：NO-GO。RoIAlign pool 對隨機初始嵌入無效。

---

## v4 — stretch-resize + global pool + 訓練 emb_head

**日期**：2026-05-25（當前）
**命令**：
```bash
uv run train/temporal_yolo/train_jde_market.py \
    --yolo-weights models/yolo/yolo26s.pt \
    --teacher-ckpt runs/gated_det_v1/best.ckpt \
    --mamba-ckpt runs/mamba_gt_960_v2/best.ckpt \
    --market-root datasets/Market-1501-v15.09.15 \
    --run-dir runs/jde_market_v4 \
    --epochs 30 --batch-size 64 --lr 3e-4
```

**變更**：
- `resize_letterbox` → `resize_stretch_batch_gpu`（直接拉伸到 640×640，無 padding）
- `pool_embeddings(bbox)` → `pool_embeddings_global`（無 padding 所以 global mean 有效）
- `--train-emb-head` 維持 True
- 移除 `--precompute`（emb_head 訓練中無法預算）

**架構**：
```
Market-1501 crop → stretch 640×640 (F.interpolate on GPU)
    → teacher YOLO → Mamba head → pool_embeddings_global
    → projector (384→256→128) → SupCon loss
```

| 組件 | 參數 | 狀態 |
|------|------|------|
| YOLOv26s + teacher + mamba blocks | ~13M | frozen |
| emb_head (3× Conv→SiLU→Conv) | 935K | **training** |
| EmbeddingProjector (384→256→128) | 131K | **training** |

**目前進度**（epoch 1-7）：

| epoch | loss |
|-------|------|
| 1 | 3.8818 |
| 2 | 3.5993 |
| 3 | 3.4196 |
| 4 | 3.2609 |
| 5 | 3.1235 |
| 6 | 3.0170 |
| 7 | 2.9532 |

loss 穩定下降，待 30 epoch 完成後進行 Market-1501 eval。

---

## 設計檢討

### 為何 stretch-resize 有效而 letterbox + RoIAlign 無效？

| 方案 | pooling 方式 | 問題 |
|------|-------------|------|
| letterbox + global_pool | spatial mean over 640×640 | 95% 為 padding，人體訊號被淹沒 |
| letterbox + RoIAlign pool | RoIAlign(bbox, output_size=1) | 隨機初始化 emb_head → 所有 cropped embedding 幾乎相同 → gradient=0 |
| **stretch + global_pool** | **spatial mean over 640×640** | **100% 為人體像素，global mean 直接有效；隨機初始化 emb_head 的 pixel 差異在 640×640 尺度上足夠大** |

### 後續可行方向

1. **emb_head 預訓練**：先用更大的 learning rate 或更簡單的 loss 預訓練 emb_head，再用 SupCon fine-tune
2. **Mamba checkpoint 包含 emb_head**：重訓 `train_mamba_head.py` 或 `train_mamba_gt.py` 時設定 `emb_dim=128`，讓 emb_head 獲得檢測相關的預訓練
3. **三階段訓練**：
   - Phase A：stretch + global_pool + 較高 LR 預訓練 emb_head + projector
   - Phase B：切換到 letterbox + RoIAlign，fine-tune projector（emb_head frozen）
   - Phase C：解凍全部，聯合 fine-tune
4. **ReID backbone 蒸餾**：用 SigLIP2 或 OSNet 在 Market-1501 上產生 teacher embeddings，用 SupCon 蒸餾到 projector
