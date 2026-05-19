# Option D 實作計畫

## 前置條件

- [x] Option C 架構已完成（`yolo_joint.py`）
- [ ] ByteTrack tracker cache 已生成（v2 需要）
- [ ] MOT17 訓練資料可存取
- [ ] GPU VRAM ≥ 12GB（BF16 訓練）

---

## Phase 0：現狀

| 項目 | 狀態 | 備註 |
|------|------|------|
| `model.py` | ✅ | Option B 基礎架構 |
| `yolo_joint.py` | ✅ | Option C 聯合訓練 |
| `loss.py` | ✅ | Auction Matcher（移除 scipy，移植自 `auction.hpp`） |
| `train/temporal_yolo/train_joint.py` | ✅ | |
| `train/temporal_yolo/train_conditioned.py` | ✅ | 骨架（等 yolo_conditioned.py） |
| `docs/temporal_yolo/` | ✅ | 本文件 |

### loss.py：Auction 取代 scipy Hungarian

`HungarianMatcher` 已重命名為 `AuctionMatcher`，`HungarianMatcher` 保留為別名（向下相容）。

演算法來源：直接移植 `include/tracking/auction.hpp` 的 `AuctionAlgorithm::Solve`，
與 `tracker_gpu.cu` 的 CUDA kernel `parallel_auction_shmem_kernel` 同一演算法族。

- **無 scipy dependency**
- 速度：O(N²) per iteration vs Hungarian O(N³)；N≤200 的追蹤場景快 2~5×
- 一致性：matcher 與 tracker 內部使用相同的分配哲學

---

## Phase 1：核心模組實作

### 1-A  `TrackerGateInput` dataclass

**檔案**：`src/saccade/perception/temporal_yolo/yolo_conditioned.py`

```python
@dataclass
class TrackerGateInput:
    confirmed_boxes:   Tensor          # (N, 4) [x1,y1,x2,y2] 絕對像素
    confirmed_scores:  Tensor          # (N,)
    velocities:        Tensor | None   # (N, 2) [vx,vy] px/frame
    tentative_boxes:   Tensor | None   # (M, 4)
    tentative_ratios:  Tensor | None   # (M,)  hit_streak/required
    img_hw:            tuple[int, int]

    @classmethod
    def from_tracker_results(
        cls,
        track_results: list[TrackResult],
        state_snapshots: list[TrackStateSnapshot],
        candidate_snapshots: list[TrackCandidateSnapshot],
        img_hw: tuple[int, int],
    ) -> "TrackerGateInput": ...
```

驗收：`from_tracker_results` 能從 tracker 直接轉換，shapes 正確。

### 1-B  `GaussianHeatmapRenderer`

```python
class GaussianHeatmapRenderer(nn.Module):
    def forward(
        self,
        gate_input: TrackerGateInput,
        hw: tuple[int, int],          # target (H_s, W_s)
    ) -> Tensor:                       # (B, 1, H_s, W_s)
```

邏輯：
1. 正規化 boxes + velocity prediction → normalized (cx, cy)
2. 渲染 Gaussian，sigma = box_size * sigma_scale
3. det_idx=-1（純預測）：sigma × 1.5，強度 × 0.5
4. Tentative tracks：強度 × `tentative_ratio * 0.3`
5. max 合併所有 heatmap

### 1-C  `TrackSpatialGate`

```python
class TrackSpatialGate(nn.Module):
    # per-scale learnable alpha（初始化為 0）
    alpha_p3: nn.Parameter
    alpha_p4: nn.Parameter
    alpha_p5: nn.Parameter

    def forward(
        self,
        gate_input: TrackerGateInput,
        scales: tuple[str, ...],       # e.g. ('p3', 'p4', 'p5')
        feat_hws: dict[str, tuple],    # {'p3': (80,80), 'p4': (40,40), 'p5': (20,20)}
    ) -> dict[str, Tensor]:            # {'p3': (B,1,80,80), ...}
```

### 1-D  `TemporalYOLOConditioned`

```python
class TemporalYOLOConditioned(nn.Module):
    pyramid  : YOLOFeaturePyramid   # 複用 yolo_joint.py
    gate     : TrackSpatialGate     # 新增
    fusion   : FPNSequenceProjection | None  # 可選，接 decoder
    # detect head 用 YOLO 原本的 Detect（或複用 TrackQueryDecoder）

    def forward(
        self,
        frame      : Tensor,
        gate_input : TrackerGateInput | None = None,  # None = 不注入
    ) -> dict
```

`gate_input=None` 時等效標準 YOLO 推論，確保向下相容。

### 驗收測試

```python
# alpha=0 → gate 全為 1 → 輸出與無 gate 相同
gate = TrackSpatialGate()
out = gate(gate_input, scales=('p5',), feat_hws={'p5': (20,20)})
assert (out['p5'] == 1.0).all()

# alpha > 0 後 → heatmap 區域 gate > 1
gate.alpha_p5.data.fill_(0.5)
out = gate(gate_input, ...)
assert out['p5'].max() > 1.0

# velocity prediction：heatmap 中心在 predicted 位置
# （box + velocity，而非原始 box）

# 梯度流：loss.backward() 後 alpha_p5.grad is not None
```

---

## Phase 2：訓練腳本整合

### 2-A  GT 快取工具（v1 用）

**檔案**：`scripts/tools/cache_gt_tracks.py`

從 MOT17 `gt.txt` 生成 per-frame `.pt` 檔，格式與 `TrackerGateInput` 相容。
（不含速度，`velocities=None`）

```bash
uv run scripts/tools/cache_gt_tracks.py \
    --data-root /path/to/MOT17 \
    --out-dir datasets/mot17_gt_cache
```

### 2-B  ByteTrack 快取工具（v2 用）

**檔案**：`scripts/tools/cache_tracker_states.py`

跑完整 pipeline（YOLO + ByteTrack），存 `TrackerGateInput` 相容格式，含速度。

```bash
uv run scripts/tools/cache_tracker_states.py \
    --data-root /path/to/MOT17 \
    --yolo-weights models/yolo/yolo26s.pt \
    --out-dir datasets/mot17_track_cache
```

### 2-C  `MOT17TemporalClip` 擴充

在 `dataset.py` 的 `__getitem__` 加載 `TrackerGateInput`：

```python
# 若 track_cache_dir 有指定，載入前一幀的 gate_input
gate_input = load_gate_input(cache_dir, seq, frame_id - 1)
return {
    "frames": ...,
    "gt_boxes": ...,
    "gate_inputs": gate_input,  # 新增
}
```

### 2-D  `train_conditioned.py` 更新

主要改動：training loop 傳入 `gate_input`：

```python
out = model(frame_t, gate_input=gate_inputs[b][t-1])
```

**損失函數**：Option D 輸出改用標準 YOLO detection head（非 Track Query decoder），
因此損失為標準 YOLO box + score loss，不走 `AuctionMatcher`。

`AuctionMatcher`（`loss.py`）仍用於 Option B/C 的 Track Query 訓練路徑，
Option D 的訓練腳本直接使用 Ultralytics 的 detection loss 計算。

---

## 訓練執行順序

```bash
# Step 1：生成 GT 快取（v1 驗證用）
uv run scripts/tools/cache_gt_tracks.py --data-root /MOT17

# Step 2：Phase 1 — 凍結 YOLO，只訓 gate.alpha
uv run train/temporal_yolo/train_conditioned.py \
    --data-root /MOT17 \
    --track-cache datasets/mot17_gt_cache \
    --phase 1 --epochs 10 \
    --resume runs/joint/best.ckpt   # 從 Option C 熱啟動

# Step 3：確認 alpha 非零 + gate map 視覺化
uv run scripts/tools/visualize_gate.py --ckpt runs/conditioned_p1/best.ckpt

# Step 4：生成 ByteTrack 快取（v2）
uv run scripts/tools/cache_tracker_states.py --data-root /MOT17

# Step 5：Phase 2 — 全部解凍
uv run train/temporal_yolo/train_conditioned.py \
    --data-root /MOT17 \
    --track-cache datasets/mot17_track_cache \
    --phase 2 --epochs 50 \
    --resume runs/conditioned_p1/best.ckpt
```

---

## 消融實驗設計

| Exp | Gate 輸入 | Velocity | Tentative | 目的 |
|-----|-----------|----------|-----------|------|
| A | 無（baseline） | — | — | Option C baseline |
| B | GT oracle | 無 | 無 | Gate 有效性上界 |
| C | ByteTrack 快取 | 無 | 無 | 實際推論等效 |
| D | ByteTrack 快取 | 有 | 無 | Velocity 增益 |
| E | ByteTrack 快取 | 有 | 有 | Tentative 增益 |

---

## 里程碑

| 里程碑 | 驗收標準 |
|--------|----------|
| Phase 1 模組完成 | unit tests pass |
| Phase 1 訓練 | `alpha_p5 > 0`，gate heatmap 視覺上對齊目標位置 |
| Phase 2 訓練 | MOT17 val IDF1 > Option C baseline |
| 消融完成 | Velocity 是否有增益有定論 |
