# Option F: Mamba SSM 檢測頭 (Detection Head)

> **狀態**：主動生產環境候選方案 (Active Production Candidate, 2026-05-31)
> 將標準 YOLO 的檢測頭 (Detect Head) 替換為 Mamba 選擇性狀態空間 (S6) 模組，以建模長程空間與時間依賴性。
> 已完全整合 Gated YOLO 骨幹網路 (支援 PyTorch 與 TensorRT 引擎) 以及 GPUByteTracker。

---

## 架構總覽

Option F 將標準卷積 YOLO 檢測頭替換為 **Mamba 選擇性狀態空間 (S6) 模組**，將空間與時間特徵視為序列進行處理。

```
                  ┌──────────────────────────────────────────┐
                  │        YOLO backbone (layers 0-22)       │
                  └────┬────────────────┬────────────────┬───┘
                       │                │                │
                       ▼                ▼                ▼
                      P3                P4               P5
                 (128ch, H/8, W/8)  (256, H/16)      (512, H/32)
                       │                │                │
                       │          [空間閘控模組]         │
                       ▼                ▼                ▼
                   Gated P3          Gated P4         Gated P5
                       │                │                │
                       │                ▼                ▼
                       │           步長卷積降採樣    步長卷積降採樣
                       │            (Reduce 4x)      (Reduce 4x)
                       │                │                │
                       ▼                ▼                ▼
                   輕量級卷積        Mamba 掃描       Mamba 掃描
                  (Hybrid 模式)     (序列長度 L)     (序列長度 L)
                       │                │                │
                       │                ▼                ▼
                       │           PixelShuffle     PixelShuffle
                       │             上採樣器         上採樣器
                       │                │                │
                       ▼                ▼                ▼
                 [與原始 FPN 輸入進行通道拼接 (Skip Connection)]
                       │                │                │
                       ▼                ▼                ▼
                   cls_head          cls_head         cls_head
                   reg_head          reg_head         reg_head
                 (emb_head)        (emb_head)       (emb_head)
```

### 核心設計原則：
1. **空間維度縮減 (Spatial Reduction)**：在進入 Mamba 模組前，先利用步長卷積將高解析度特徵圖 (P4, P5) 的寬高降採樣 4 倍，使序列長度 $L$ 縮小 16 倍，確保計算成本在可控範圍內 ($L \le 400$)。
2. **PixelShuffle 學習型上採樣 (P1 優化)**：取代傳統的雙線性插值，採用基於卷積的 PixelShuffle 上採樣器 (3×3 Conv + PixelShuffle) 實現更豐富的空間特徵融合。若加載的權重缺少上採樣參數，會自動且安全地退回到雙線性插值。
3. **殘差跳躍連接 (Skip Connections)**：原始的 FPN 特徵圖通過 1×1 卷積進行投影後，與上採樣重建的 Mamba 輸出特徵圖進行通道拼接（總通道數為 `d_model * 2`），再送入最終的分類頭與回歸頭。

---

## 核心程式碼檔案

### 感知與模型相關 (`src/saccade/perception/temporal_yolo/`)
* **[mamba_head.py](../../../src/saccade/perception/temporal_yolo/mamba_head.py)**：純 PyTorch 實作的 `MambaBlock` 與 `MambaDetectionHead`，包含 `_selective_scan_jit` 以及調用 C++/CUDA 自定義選擇性掃描算子的擴充模組包裝。
* **[mamba_gated_detector.py](../../../src/saccade/perception/temporal_yolo/mamba_gated_detector.py)**：偵測器整合類別 `MambaGatedDetector`，將 Gated YOLO 骨幹 (支援 PyTorch/TensorRT) 與 `MambaDetectionHead` 封裝，並無縫整合 `GPUByteTracker`。
* **[yolo_gated_detector.py](../../../src/saccade/perception/temporal_yolo/yolo_gated_detector.py)**：基礎閘控偵測器邏輯、配置 (`GatedDetConfig`) 與 Hook 捕捉輔助函數。

### CUDA 核心與底層擴充 (`src/tracking/`, `include/perception/`, `src/perception/`)
* **[mamba_scan.cu](../../../src/tracking/mamba_scan.cu)** / **[mamba_scan.cuh](../../../include/tracking/mamba_scan.cuh)**：自定義 CUDA 選擇性掃描前向算子 (`selective_scan_fwd`)。
* **[mamba_gated_detector.cpp](../../../src/perception/mamba_gated_detector.cpp)** / **[mamba_gated_detector.hpp](../../../include/perception/mamba_gated_detector.hpp)**：C++ 端的綁定與推理解析執行介面。

---

## 架構優化設計

Mamba 檢測頭相較於標準架構導入了以下 5 大關鍵優化：

### P1: PixelShuffle 學習型上採樣
為解決傳統雙線性插值在還原序列特徵時造成的邊緣模糊與網格效應，採用了學習型上採樣層：
```python
nn.Sequential(
    nn.Conv2d(d_model, d_model * (spatial_reduction ** 2), 3, padding=1),
    nn.PixelShuffle(spatial_reduction),
)
```
* **相容退回機制 (Fallback)**：當加載舊版 (僅包含雙線性插值) 訓練的權重檔案時，`load_state_dict` 會捕捉到缺少 `upsample` 層參數的狀況，並自動將 `upsample_loaded` 設為 `False`，平滑地退回雙線性插值，確保生產部署相容性。

### P2: 四向交叉掃描 Mamba (Cross-Scan)
標準的 1D 序列掃描會使 2D 影像丟失垂直方向的空間上下文。我們在 `_cross_scan_mamba` 中實作了四向掃描：
1. 將特徵圖沿著 4 個空間維度旋轉/翻轉 (水平正向、轉置、垂直翻轉、水平翻轉)。
2. 將其 reshape 並合併到 batch 維度 ($4B$) 同步進行 Mamba 掃描。
3. 掃描完成後，反向旋轉/翻轉還原並求均值 (Mean Pool)。

### P3: 混合尺度 FPN 頭 (Hybrid Head)
在細粒度的 P3 尺度上 (stride 8，在 640 解析度下 sequence 長度為 $80 \times 80 = 6400$)，完全進行狀態空間掃描極耗計算資源。當開啟 `use_hybrid_head=True`：
* **P3 尺度**：繞過 Mamba 掃描，直接使用輕量級的深度可分離卷積區塊 (EfficientViT 樣式) 進行快速特徵處理。
* **P4 & P5 尺度**：維持採用高效的 Mamba SSM 狀態空間模組。

### P2-ST: 時空狀態空間模型 (Temporal Mamba)
針對連續視訊串流，引入跨畫面時間序列掃描：
* 在長度為 $T$ 的時間快取中，利用 **全域運動補償 (GMC)** 的仿射矩陣對空間特徵進行扭曲對齊 (Warp Align)。
* 將對齊後的特徵沿時間軸拼接，送入 `temporal_blocks` 進行時序關聯性掃描，極大增強了動態目標的時序一致性。

### JDE: 聯合檢測與 ReID 嵌入
Mamba 頭可同步輸出像素級的 ReID 外觀嵌入特徵 (`emb_dim > 0`)：
* 多尺度 ReID 特徵圖透過 ROI Align 或中心/全域池化提取目標外觀。
* 將其送入 `EmbeddingProjector` (包含 BatchNorm 與 ReLU 的投影層) 投射為 128 維 $L_2$ 正規化特徵，該投影層已在 Market-1501 資料集上預訓練完成。

---

## CUDA 核心效能評測 (Kernel Benchmark)

底層 CUDA 核心實作於 `src/tracking/mamba_scan.cu`，每個 $(B, D)$ 管道映射到一個 CUDA Block，並使用 Warp 級別的 Shuffle 指令進行關聯掃描。

| 規格配置 (B, L, D, N) | 耗時 (ms) | 運算效能 (GFLOPS) |
|-----|------|--------|
| (1, 400, 128, 16) 類似 P5 | 0.19 ms | 25.7 |
| (1, 1600, 128, 16) 類似 P4 | 0.75 ms | 26.3 |
| (1, 6400, 128, 16) 類似 P3 | 2.97 ms | 26.5 |
| (1, 6400, 256, 16) 兩倍通道 D | 2.98 ms | 52.8 |
| (4, 6400, 128, 16) B=4 | 4.24 ms | 74.1 |
| (8, 6400, 128, 16) B=8 | 5.84 ms | 107.8 |
| (1, 400, 128, 32) 狀態數 N=32 | 0.19 ms | 51.6 |
| (1, 6400, 128, 32) 狀態數 N=32 | 2.98 ms | 52.8 |

### 效能分析：
* 掃描時間相較於序列長度 $L$ 與 Batch 大小 $B$ 呈現極佳的線性縮放特徵。
* 狀態數 $N \le 32$ 時，能完美在單個 CUDA Warp 內完成關聯掃描，無額外通訊開銷。
* 較大的通道數 $D$ 能提供更高的 Block 並行 occupancy，進而使 GFLOPS 效能翻倍。

---

## 訓練管線與流程

所有 Mamba 檢測頭相關的訓練腳本皆存放於 `scripts/train/temporal_yolo/`，並嚴格遵循**三階段訓練管線規範 (3-Phase Pipeline)**：

### 第一階段：特徵蒸餾訓練 (`train_mamba_head.py`)

利用預訓練的 `GatedYOLODetector` 教師模型，將其卷積檢測頭 (`cv2` 與 `cv3` 分支) 的 raw 預測特徵圖，蒸餾到學生模型的 `MambaDetectionHead` 中。

```
MOT17 影像 → GatedYOLODetector (凍結骨幹網路 + 閘控模組)
                    │
           [Hook 閘控 FPN: P3, P4, P5]
                    │
      ┌─────────────┴─────────────┐
      ▼                           ▼
教師卷積預測頭             Mamba 檢測頭 (Trainable)
cv2[i], cv3[i] (Frozen)            │
      │                           ▼
  t_reg, t_cls               s_reg, s_cls
      │                           │
      └────── 均方誤差 (MSE Loss) ─┘
```

#### 特徵快取極速加速 (Feature Cache)
為最大化 Phase 1 訓練效率，可將教師模型的 FPN 特徵圖預先快取至硬碟/記憶體，完全繞過骨幹網路的前向計算：
1. **特徵預計算模式 (Precompute)**:
   ```bash
   uv run scripts/train/temporal_yolo/train_mamba_head.py \
       --data-root datasets/MOT17 \
       --precompute-dir storage/feature_cache/mot17_gated_v1
   ```
2. **快速訓練模式 (Fast Train)**:
   加載預計算好的快取特徵檔案，使每輪 Epoch 僅需處理可訓練的 Mamba 參數：
   ```bash
   uv run scripts/train/temporal_yolo/train_mamba_head.py \
       --data-root datasets/MOT17 \
       --cache-dir storage/feature_cache/mot17_gated_v1 \
       --epochs 20 --batch-size 8 --lr 1e-3
   ```

---

### 第二階段：地面真值細微調 (`train_mamba_gt.py`)

此階段拋棄 MSE 蒸餾損失，改用 MOT17 Ground-Truth 框進行端到端監督，以最大化提升追蹤效能指標 (MOTA, IDF1)。

* **損失函數**：保持骨幹網路與閘控模組凍結，對加載了蒸餾權重的 `MambaDetectionHead` 使用 Ultralytics 的 `v8DetectionLoss` (DFL + CIoU + BCE Loss) 進行訓練。
* **空間閘模擬回饋 (Gate Feedback)**：利用前一訊框 of Ground-Truth 框渲染為動態的高斯熱圖 (`TrackerGateInput`) 送入閘控，以在訓練時模擬真實追蹤中的回饋閉環：
  ```bash
  uv run scripts/train/temporal_yolo/train_mamba_gt.py \
      --data-root datasets/MOT17 \
      --teacher-ckpt runs/gated_det_v1/best.ckpt \
      --mamba-ckpt runs/mamba_distill_v1/best.ckpt \
      --epochs 30 --batch-size 4 --lr 1e-4 --gt-ratio 0.5
  ```

---

## 生產環境預設配置 (Production Presets)

在生產部署中，最佳評估配置定義在 `configs/presets/mamba_optimal.yaml` 中，其針對 Mamba 的預測分佈調整了關聯匹配參數：

```yaml
# configs/presets/mamba_optimal.yaml
mamba_ckpt: runs/mamba_gt_vgt_mamba_v14/best.ckpt
fpn_backbone_engine: models/yolo/yolo26s_backbone_640_best.engine
reid_mode: "off"         # 關閉 ReID，以達到極限 108 FPS 運作速度
gmc: true
gmc_downscale: 4        # 高精度運動關聯
match_thresh: 0.50       # 針對 Mamba 邊框優化的較低匹配閥值
new_track_thresh: 0.28
confirm_streak: 3
```

此配置能在相機劇烈晃動下保持極其穩定且精準的動態邊框追蹤，且完全不引入額外的 ReID 特徵提取延遲。

---

## 各元件實測增益 (MOT17-SDP Ablation)

> 基準：yolo26s baseline tracker，IDF1 ≈ 52%。所有 Mamba 結果使用 `mamba_optimal` preset (match_thresh=0.50, confirm_streak=3, interpolate=true)。

### 當前最佳：**v14 — N=16 Cross-Scan**（2026-05-31）

`runs/mamba_gt_vgt_mamba_v14/best.ckpt`

| IDF1 | MOTA | FP | FN | Recall | Prcn | loss |
|------|------|-----|-----|--------|------|------|
| **72.4%** | **77.3%** | 5754 | 19077 | 83.0% | 94.2% | 2.87 |

架構：Cross-Scan + PixelShuffle + **N=16 SSM fix**（無 temporal blocks）

---

### 元件拆解

| 配置 | IDF1 | MOTA | FP | FN | Recall |
|------|------|------|-----|-----|--------|
| Baseline (no Mamba) | 52.0% | 43.5% | — | — | — |
| v2 spatial only (`--no-temporal`) | 49.4% | 49.0% | 17246 | 39044 | 65.2% |
| v2 full (spatial + temporal) | 71.6% | 76.7% | 5252 | 20352 | 81.9% |
| vgt_v2 checkpoint | 69.4% | 74.4% | 4437 | 23752 | 78.8% |
| **Cross-Scan (N=1, 原版)** | 71.6% | 76.7% | 5252 | 20352 | 81.9% |
| Cross-Scan + N=16 fix（不重訓）| 72.2% | 76.3% | 5431 | 20522 | 81.7% |
| **v14 Cross-Scan + N=16 retrain** | **72.4%** | **77.3%** | 5754 | **19077** | **83.0%** |

### Temporal Blocks 的雙重貢獻

Temporal blocks 透過 **eval 時的 sliding window buffer**（T=3~4 幀）在每個空間位置做幀間 SSM 掃描，提供兩種增益：

| 機制 | v2 | v8 |
|------|-----|-----|
| **FP 壓制**（不一致偵測被過濾）| 17246 → 5327（**-69%**）| 9775 → 8125（-17%）|
| **FN 回收**（遮擋目標靠歷史幀補回）| 39044 → 20329（**+18715**）| 30256 → 21949（+8307）|
| 總 IDF1 增益 | **+21.8pp** | +5.9pp |

> **關鍵發現**：v2 的 temporal blocks 從未在訓練時看過多幀（frame-by-frame 訓練），但在 eval buffer 模式下能自然學到強力的時序一致性 gate。v8 用 all-frames loss 明確訓練 temporal，反而弱化了這個 gate 行為。

---

## 訓練版本歷程與教訓

| 版本 | 訓練方式 | IDF1 | FP | 問題根因 |
|------|----------|------|-----|---------|
| **v2** (current best) | Frame-by-frame 空間訓練，temporal 未激活 | **71.2%** | **5327** | — |
| v6 | `--scratch` 全隨機初始化 | 32.0% | — | MOT17 7 條 seq 不足以訓練偵測頭 |
| v7 | `--add-temporal`，time-major stacking，last-frame loss | 68.2% | 15727 | stacking 順序錯誤；last-frame loss 無法抑制 FP |
| v8 | batch-major stacking，all-frames loss | 70.5% | 8125 | temporal FP gate 被弱化（v2 −69% → v8 −17%）|
| v10 | **凍結 temporal**，只訓練 spatial | TBD | TBD | 保留 v2 temporal gate + 提升 spatial 品質 |

### 訓練設計原則（從失敗中學到的）

1. **不能從零訓練偵測頭**：MOT17 資料量遠不足以訓練 spatial detection，必須從預訓練 checkpoint 繼承。
2. **Stacking 順序影響 SSM 語意**：`torch.cat(time-major)` 讓 SSM 的「時序序列」是不同 batch item 的同一幀，不是單一 sample 的多幀。正確方式：`torch.stack(dim=1).reshape(B*T, ...)`（batch-major）。
3. **All-frames loss ≠ 保留 temporal gate**：All-frames loss 讓每幀偵測品質好，但不直接優化「抑制跨幀不一致的 FP」。v2 的 temporal gate 效果是訓練時未激活、eval 時 emergent 的行為。
4. **Freeze temporal + 訓練 spatial** 是目前最有希望的方向：v8 no-temporal 的空間品質（FP=9775，Recall=73.1%）已明顯優於 v2（FP=17246，Recall=65.2%），若能保留 v2 的 temporal gate（FP −69%，FN +18715），理論上可超越 71.2%。

---

## 時序特徵方向結案（2026-05-31）

所有在 v14 Cross-Scan 空間基線上增加時序模組的嘗試均無法超越純空間模型。以下為完整 ablation 結果與根本原因分析。

### 實驗結果

| 版本 | 方法 | T=1 eval | Recurrent eval | 結論 |
|------|------|----------|----------------|------|
| v14（基線）| Cross-Scan 純空間 | **72.4% IDF1** | — | 最佳 |
| v15 | + SSM temporal blocks | 58.6% (−13.8pp) | 66.0% (−6.4pp) | NO-GO |
| v17 | + TemporalAttentionBlock | 69.2% (−3.2pp) | 69.5% (−2.9pp) | NO-GO |

v15 temporal 的 FP 壓制有效（T=1: 32,630 → recurrent: 13,934，−57%），但空間路徑退步的代價遠超過時序增益。

### 根本原因

**1. 訓練 code path 問題（all-frames loss 從不激活純 T=1 路徑）**

```
訓練 t=0：x_up → temporal_block(h=0) → x_cat   ← 仍通過 temporal block
eval T=1： x_up → x_cat                          ← temporal block 完全 bypass
```

All-frames loss 的 t=0 不等於 T=1 eval 模式。訓練 30 epoch 後，空間 features 優化成「temporal block 會在後面補正」，T=1 eval 時 temporal block 不在，空間退步暴露。

**2. SSM 特有的 train-test 分布不一致**

SSM temporal blocks 有 hidden state `h`。訓練時每個 clip 從 `h₀=0` 重置；eval 時 recurrent 模式下 `h` 攜帶跨 clip 的連續歷史。模型從未在訓練中見過「有長歷史的 h」，導致推論行為和訓練分布不一致。TemporalAttentionBlock 沒有 hidden state，理論上規避了此問題，但仍因空間退步而失敗。

**3. Cross-Scan 已提供全局空間 context，時序增益空間有限**

4 方向掃描讓每個位置能看到整張圖的空間資訊，部分替代了時序需要做的「區分背景 vs 真實移動目標」功能。空間基線已夠強，temporal 的邊際增益不足以彌補它帶來的訓練複雜度。

**4. MOT17 訓練資料量不足**

MOT17 僅 7 條 SDP 序列，temporal blocks 沒有足夠多樣的時序樣本來學習有意義的跨幀模式。每個 epoch 約 300-400 個 clip，不足以讓時序 SSM 收斂到有效的狀態轉移。

### 結論與未來方向

v14（Cross-Scan + N=16，純空間）為 Option F 最終版本。

若未來要重新嘗試時序方向，需同時解決以下條件：
- **更大時序資料集**（MOT17 × 3 以上的序列多樣性）
- **Recurrent rollout 訓練**（在 clip 間傳遞 hidden state，消除 train-test 不一致）
- 或改用 **純 frame-by-frame 訓練 + eval-only buffer**（復現 v2 的 emergent temporal gate，但需重新設計訓練流程）

---

## CUDA SSM Kernel 注意事項

CUDA kernel 接受 `A.shape[0]` 作為 `dstate` 參數。`A_log` 在 checkpoint 中的形狀為 `(1, N)`，因此 `A.shape[0]=1` 而非 `N=16`，導致 kernel 只用 1 個狀態維度。

- **正確路徑**：使用 TRT backbone（fp16）→ CUDA kernel → `dstate=1`（有效狀態數不足）
- **原始路徑**：不使用 TRT backbone（fp32）→ JIT scan → `N = A.shape[1] = 16`（正確）

v2 的 71.2% 是在 fp32 JIT path 下達成的（TRT backbone 啟用後會降至 ~49%）。長期修法：將 kernel 呼叫改為 `A.numel()` 取得正確 dstate。
