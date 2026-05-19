# Option D：Track-Conditioned YOLO Neck — 詳細設計

## 核心動機

現有 ByteTrack-style tracker 成本極低，已能穩定輸出每幀的追蹤狀態。
這些資訊只被用於 Re-ID / 關聯，從未回饋到特徵提取階段。

**目標**：把 tracker 的空間先驗（「這裡有人，而且正在往這個方向移動」）
注入 YOLO FPN，讓 backbone 在已知目標區域產生更強的特徵。

### 為什麼用外部 tracker 而不是 Track Queries

| | Track Queries 自身輸出 | 外部 ByteTrack |
|--|----------------------|---------------|
| 訓練初期品質 | 不可靠（隨機初始化）| **穩定（已訓好）** |
| Train/inference 一致性 | 需 curriculum 橋接 | **天然一致** |
| 自我污染風險 | 高（雜訊 gate → 雜訊梯度）| 無 |
| 架構複雜度 | 高（Track Queries + gate + decoder）| **低（直接用 tracker 輸出）** |

---

## Tracker 可用欄位

### `TrackResult`（`tracker.update()` 直接輸出，每幀都有）

```python
TrackResult:
    x1, y1, x2, y2   # 絕對像素座標
    obj_id            # Track ID
    score             # 偵測分數
    det_idx           # 匹配到哪個 det；-1 = 純預測（無匹配，遮擋中）
```

### `TrackStateSnapshot`（`tracker.get_state_snapshots()`）

```python
TrackStateSnapshot:
    obj_id, class_id
    age               # 存活幀數（越大越穩定）
    score
    state             # [cx, cy, a, h, vx, vy, va, vh]  ← Kalman 狀態，含速度！
    covariance        # 8×8 共變異數矩陣（反映不確定性）
```

### `TrackCandidateSnapshot`（`tracker.get_tentative_candidates()`）

```python
TrackCandidateSnapshot:
    obj_id, age
    hit_streak              # 已連續命中幀數
    required_confirm_streak # 需再命中幾幀才確認
    score, x1, y1, x2, y2
```

---

## TrackerGateInput：統一輸入介面

```python
@dataclass
class TrackerGateInput:
    # ── 必要：確認軌跡 ──
    confirmed_boxes  : Tensor        # (N, 4)  [x1,y1,x2,y2] 絕對像素
    confirmed_scores : Tensor        # (N,)

    # ── 建議：Kalman 速度（下一幀預測位置）──
    # 來源：TrackStateSnapshot.state[4:6] = [vx, vy]（px/frame）
    # None = 不做預測，heatmap 打在歷史位置
    velocities       : Tensor | None  # (N, 2)  [vx, vy]

    # ── 可選：候選軌跡（尚未確認，給較弱的 gate 信號）──
    tentative_boxes  : Tensor | None  # (M, 4)
    tentative_ratios : Tensor | None  # (M,)  hit_streak / required（0~1）

    # ── 圖像尺寸（正規化用）──
    img_hw           : tuple[int, int]

    @classmethod
    def from_tracker_results(
        cls,
        track_results   : list[TrackResult],
        state_snapshots : list[TrackStateSnapshot],   # get_state_snapshots()
        candidates      : list[TrackCandidateSnapshot], # get_tentative_candidates()
        img_hw          : tuple[int, int],
    ) -> "TrackerGateInput":
        # state_snapshots 按 obj_id 建 lookup → 取 vx, vy
        vel_lookup = {
            s.obj_id: (s.state[4], s.state[5])   # vx, vy
            for s in state_snapshots
        }
        boxes, scores, vels = [], [], []
        for tr in track_results:
            boxes.append([tr.x1, tr.y1, tr.x2, tr.y2])
            scores.append(tr.score)
            vx, vy = vel_lookup.get(tr.obj_id, (0.0, 0.0))
            vels.append([vx, vy])

        t_boxes, t_ratios = [], []
        for c in candidates:
            t_boxes.append([c.x1, c.y1, c.x2, c.y2])
            t_ratios.append(c.hit_streak / max(c.required_confirm_streak, 1))

        return cls(
            confirmed_boxes  = torch.tensor(boxes,   dtype=torch.float32),
            confirmed_scores = torch.tensor(scores,  dtype=torch.float32),
            velocities       = torch.tensor(vels,    dtype=torch.float32) if vels else None,
            tentative_boxes  = torch.tensor(t_boxes, dtype=torch.float32) if t_boxes else None,
            tentative_ratios = torch.tensor(t_ratios,dtype=torch.float32) if t_ratios else None,
            img_hw           = img_hw,
        )
```

`from_tracker_results()` 統一轉換格式，training / inference 共用同一路徑，確保無 distribution shift。

---

## TrackSpatialGate：heatmap 生成機制

### 輸入 → heatmap 的三步驟

```
Step 1  Velocity Prediction（來自 Kalman state[4:6]）
  vx, vy = TrackStateSnapshot.state[4], state[5]   (px/frame)
  predicted_cx = cx + vx    (dt = 1 frame，與 tracker 內部預測一致)
  predicted_cy = cy + vy
  → heatmap 打在「下一幀預計位置」，比歷史位置對 YOLO 更有用
  → velocities=None 時 fallback 到歷史位置（訓練 v1 GT oracle 用）

Step 2  Gaussian Rendering（每個尺度獨立）
  每個 track 以 (predicted_cx, predicted_cy) 為中心畫 Gaussian：
    sigma = box_size * sigma_scale              (確認軌跡，預設 0.5)
    sigma = box_size * sigma_scale * 1.5        (det_idx=-1，純預測，較寬)
  強度權重 = sigmoid(confirmed_score)           (確認軌跡)
           × 0.5 if det_idx == -1              (遮擋中，降權)
           或 tentative_ratio * 0.3            (候選軌跡，更弱)
  所有 track 的 heatmap 取 max → (B, 1, H_s, W_s)

Step 3  Learnable Gate（per-scale）
  gate_s = 1 + alpha_s * heatmap_s
  alpha_s：per-scale 可學習純量，初始化為 0
  → 訓練初始 gate=1，等效不注入，不破壞 pretrained YOLO 行為
```

### 為什麼用 Kalman 速度而非 Python MotionModel

| | Kalman vx/vy（`TrackStateSnapshot`） | Python MotionModel（EMA） |
|--|--------------------------------------|--------------------------|
| 來源 | C++ tracker 內部，推論必然有 | 僅 pure-Python eval 路徑 |
| 一致性 | **訓練 = 推論** | 訓練/推論需各自維護 |
| 精度 | 卡爾曼濾波，考慮 Q/R 噪聲 | EMA，較粗糙 |
| 額外成本 | `get_state_snapshots()` 已有 | 需獨立維護一份 registry |

結論：使用 Kalman 速度，training pipeline 的快取生成腳本呼叫 `get_state_snapshots()` 取 `state[4:6]`。

### det_idx = -1 的特殊處理

純預測軌跡（遮擋中）：
- 位置不確定性高 → sigma 放大 1.5×
- Gate 強度折半（× 0.5）
- 目的：提示 backbone「這裡可能有人，但不確定」

---

## 架構全圖

```
Frame_{t-1}
  │
  ├─ YOLO26s ──→ ByteTrack
  │                  │
  │          TrackResult × N        (每幀都有)
  │          TrackStateSnapshot × N (get_state_snapshots)
  │          TrackCandidateSnapshot (get_tentative_candidates)
  │                  │
  │          TrackerGateInput.from_tracker_results(...)
  │                  │
  ▼                  ▼
Frame_t → YOLO26s Backbone (layers 0~10)
                   │
                   │   TrackSpatialGate
                   │   ├─ Velocity prediction
                   │   ├─ Gaussian rendering per scale
                   │   └─ gate_p3/p4/p5
                   │         │
                   ▼         ▼
           YOLO Neck (layers 11~22)  ← FPN feats × gate
           P3_gated / P4_gated / P5_gated
                   │
                   ▼
           標準 YOLO Detect Head 或 TrackQueryDecoder
```

---

## 訓練策略

### 訓練時的 tracker 輸入來源（按優先順序）

**v1：GT 軌跡（快速驗證 gate 有沒有用）**
- 直接從 MOT17 `gt.txt` 讀 GT boxes
- Oracle 上界，確認 gate 機制有效
- `velocities = None`（GT 沒有速度，用固定 sigma）

**v2：預先快取 ByteTrack 輸出（推薦長期方案）**
```bash
# 一次性前處理：跑 ByteTrack，逐幀存 tracker state
uv run scripts/tools/cache_tracker_states.py \
    --data-root /path/to/MOT17 \
    --yolo-weights models/yolo/yolo26s.pt \
    --out-dir datasets/mot17_track_cache

# cache 格式：
# datasets/mot17_track_cache/{seq}/frame_{t:06d}.pt
# 內容：{'boxes': Tensor(N,4), 'scores': Tensor(N,), 'velocities': Tensor(N,2), ...}
```
- 與推論完全一致（同一個 ByteTrack）
- 訓練快（無 tracker overhead）
- 可重複使用

**v3：On-the-fly（不建議）**
- 訓練時每 batch 跑 ByteTrack → 太慢

### 訓練損失

使用**標準 YOLO detection loss**（對 GT boxes），不需要 Hungarian matching：

```
L = L_box + L_score     (YOLO 原本的損失)
```

這比 Track Query 的 Hungarian matching 穩定得多，且不需要維護 lifecycle manager。

### 兩階段訓練

**Phase 1（凍結 YOLO，只訓 gate.alpha）**
- 驗證 gate 能學到有意義的位置先驗
- `alpha` 應該從 0 收斂到某個正值
- 用 GT 軌跡（oracle）

**Phase 2（全部解凍，差分 LR）**
- `lr_backbone=1e-5, lr_gate=5e-5, lr_decoder=1e-4`
- 切換到 ByteTrack 快取輸出

---

## 風險評估

| 風險 | 可能性 | 緩解方式 |
|------|--------|----------|
| Alpha 學到 0（gate = no-op） | 中 | Phase 1 用 GT oracle 確保 gate 有用再解凍 |
| Velocity 預測偏差（快速運動）| 低 | sigma 自動放寬；ByteTrack 已處理快速運動 |
| False track gate 增強錯誤區域 | 低-中 | 只用 `score > 0.5` 且 `age >= 3` 的穩定 track |
| det_idx=-1 軌跡位置不準 | 中 | 放寬 sigma × 1.5 + 強度 × 0.5 |

---

## 開放問題

1. **要不要保留 Track Query Decoder？**
   可以完全移除（輸出改用標準 YOLO Detect Head），也可以保留。
   移除更簡單，但喪失跨幀 query 的優勢；保留可以疊加兩種機制。
   建議 v1 先移除，驗證 gate 增益後再考慮疊加。

2. **Tentative tracks 值不值得渲染？**
   新出現的目標（hit_streak=1）可能是 FP，渲染可能有害。
   建議先只用確認軌跡，之後消融。

3. **Sigma scale 用固定值還是從 covariance 自適應？**
   v1 固定（0.5），v2 可從 `covariance[0,0], covariance[1,1]` 推算。

4. **Gate 用 P3+P4+P5 全部，還是只用 P5？**
   P5（大目標、全局語意）影響最大；P3（小目標、邊緣）更精細。
   建議預設 P3+P4+P5，消融確認各自增益。
