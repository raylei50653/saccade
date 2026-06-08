# Saccade mamba_whole_graph Preset — 深度分析報告

> 生產環境基準 (production baseline)，2026-06-06 實測
> GPU: RTX 5070 Ti Laptop (12 GB), Driver: CUDA 12.x
> Preset: `configs/presets/mamba_whole_graph.yaml`

---

## 1. 總覽

```
┌─────────────────────────────────────────────────────────────┐
│                   mamba_whole_graph                          │
│                                                             │
│  Frame (JPEG)                                               │
│      │                                                      │
│      ▼ DALI NVDEC                                           │
│  GPU tensor (B, 3, 1080, 1920)                              │
│      │                                                      │
│      ▼ stretch-resize 640 (no letterbox)                    │
│                                                             │
│  ┌────── Whole-Detect CUDA Graph ──────┐                    │
│  │  TRT Backbone (yolo26s layers 0-22) │                    │
│  │      → P3(128,80,80)                │                    │
│  │      → P4(256,40,40)                │                    │
│  │      → P5(512,20,20)                │                    │
│  │           │                          │                    │
│  │      Mamba Detection Head           │                    │
│  │       input_proj(1x1) → downsample  │                    │
│  │       → MambaBlock × 2 (per scale)  │                    │
│  │       → cross_scan (4-dir)          │                    │
│  │       → PixelShuffle upsample      │  ← ~278 kernel      │
│  │       → cls_head + reg_head         │    launches 合併為   │
│  │           │                          │    1 graph replay  │
│  │      Postprocess Decode             │                    │
│  │       dist2bbox → sigmoid → top-K   │                    │
│  └─────────────────────────────────────┘                    │
│      │                                                      │
│      ▼ boxes/scores/classes (max 300 det)                   │
│                                                             │
│  ┌────── Tracker CUDA Graph ──────┐                         │
│  │  GPU GMC → ByteTracker update  │                         │
│  │   Kalman predict → IoU gate    │                         │
│  │   → Association → update       │                         │
│  └────────────────────────────────┘                         │
│      │                                                      │
│      ▼ track results → interpolate → MOT output             │
│                                                             │
│  Production: 175 FPS, 5.7 ms/frame                          │
│  7-seq SDP: IDF1 73.4%, MOTA 76.9%, HOTA 66.6%             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 三層 CUDA Graph 架構

mamba_whole_graph 的核心創新是**三層 CUDA graph**，將原本分散的數百個 kernel launch 合併為三次 graph replay：

| 層 | Graph | 捕獲內容 | 合併 kernel 數 | 效能 |
|:--|-------|---------|:---:|------|
| **L1** | Whole-Detect Graph | resize → TRT backbone → Mamba head → postprocess decode | ~280 → 1 | detect: ~7.4ms → ~3.25ms |
| **L2** | NMS Graph | C++/CUDA NMS + filter pipeline | ~5 → 1 | 內建於 postprocess |
| **L3** | Tracker Graph | GPUByteTracker update_into (Kalman + association) | ~15 → 1 | track: ~1.4ms → ~0.53ms |

### 2.1 Whole-Detect Graph — 關鍵設計

```
無 graph（eager）：                 有 graph：
─────────────────────              ─────────
resize         (host)              ┌── replay ──┐
   ↓ kernel launch                 │  resize     │
TRT backbone   (host)              │  backbone   │
   ↓ kernel launch                 │  Mamba head │
Mamba head     (host)              │  postproc   │
   ↓ (278 次 PyTorch kernel launch) │  decode     │
input_proj     (kernel)            └─────────────┘
downsample     (kernel)               1 次 GPU submission
MambaBlock×6   (kernel)
cross_scan     (kernel)
upsample       (kernel)
cls_head       (kernel)
reg_head       (kernel)
   ↓ kernel launch
postprocess    (host)
   ↓ (decode kernels)

~280 次 kernel launch              1 次 graph replay
~60 µs launch overhead total       ~0 µs launch overhead
```

**Graph-safe 設計要點**：
- TRT backbone 從專用 stream 改為 current stream (`execute_async_v3`)，避免跨 stream 依賴
- postprocess 從動態 masking 改為固定 shape top-K（`_postprocess_mamba_fixed`），確保輸出 shape 固定
- anchor grid 預計算（`_precompute_anchor_grid`），避免在 graph capture 中使用 `torch.arange`
- 座標 rescale 參數透過 `fill_()` 在主機端寫入預配置 tensor

### 2.2 Tracker Graph — 關鍵設計

- 預配置固定 size 的輸入/輸出 buffer（`max_assoc=256`, `max_objs=2048`）
- 每個 frame 以 `copy_inputs()` 將即時檢測資料複製到靜態 buffer，然後 replay
- 所有 CUDA kernel 以 fixed grid 啟動（padding 至 `max_assoc` size），在 kernel 內部透過 score/IoU 過濾無效欄位
- **限制**：不支援 per-detection embeddings（`embed_ptr=0`）— 因此 ReID mode 必須為 `off`。當 `reid_mode ≠ off` 時，tracker graph 會自動禁用

---

## 3. Mamba Detection Head 架構

### 3.1 層次結構（per FPN scale P3/P4/P5）

```
FPN feature [B, C_i, H_i, W_i]   (C_i ∈ {128, 256, 512})
    │
    ▼ input_proj: Conv2d(C_i → 128, 1×1)
[B, 128, H_i, W_i]
    │
    ├──→ skip connection ──────────────────────┐
    │                                          │
    ▼ downsample: Conv2d(128 → 128, 4×4, s=4)  │
[B, 128, H_i/4, W_i/4]                         │
    │                                          │
    ▼ flatten → [B, L, 128], L = H_i×W_i/16    │
    │                                          │
    ▼ MambaBlock × 2 (per scale)               │
    │  ┌──────────────────────────┐            │
    │  │ in_proj: 128 → 512       │            │
    │  │ conv1d(k=4) + SiLU       │            │
    │  │ SSM (selective_scan)     │            │
    │  │   d_state=16, d_inner=256│            │
    │  │   A_log: learnable decay │            │
    │  │   dt/B/C: input-dependent│            │
    │  │ gate: y ⊙ SiLU(z)        │            │
    │  │ out_proj: 256 → 128      │            │
    │  └──────────────────────────┘            │
    │                                          │
    ▼ Cross-Scan (4-direction)                 │
    │  原始 + x-flip + y-flip + xy-flip        │
    │  各自過 MambaBlock → 平均                │
    │                                          │
    ▼ unflatten → [B, 128, H_i/4, W_i/4]      │
    │                                          │
    ▼ PixelShuffle upsample (×4)               │
    │  Conv(128 → 128×16, 3×3) → PixelShuffle  │
    │  學習型上取樣，取代 F.interpolate         │
[B, 128, H_i, W_i]                              │
    │                                          │
    ▼ cat ← skip connection ──────────────────┘
[B, 256, H_i, W_i]
    │
    ├──→ cls_head: Conv→SiLU→Conv(num_classes)
    │    [B, 80, H_i, W_i]
    │
    └──→ reg_head: Conv→SiLU→Conv(reg_max×4)
         [B, 4, H_i, W_i]
```

### 3.2 關鍵架構選擇

| 設計 | 選擇 | 理由 |
|------|------|------|
| 上取樣 | **PixelShuffle** (學習型) | MOTA +14.3pp vs F.interpolate；backward 階段有 gradients 指導 |
| 掃描方向 | **Cross-Scan** (4 向) | 消除 row-major bias，零參數增長（4 向共享 MambaBlock） |
| P3 處理 | **MambaBlock** (非 hybrid) | `use_hybrid_head=False`；v14 ckpt 在 P3 也使用 Mamba |
| Temporal | **無** (`T=0`) | 時序 SSM/attention 在 v15/v17 全面退步 (IDF1 -13.8pp)，production 不使用 |
| Embedding | **無** (`emb_dim=0`) | ReID mode = off，不需 appearance 分支 |
| Per-channel A | **無** | per_channel_a + MOT20 混訓退步 (DetA -1.8pp) |

### 3.3 MambaBlock — Selective Scan (S6)

核心 SSM 運算 (`selective_scan_fwd`，CUDA kernel)：

狀態空間模型定義：
```
h_t = Ā_t · h_{t-1} + B̄_t · u_t    (狀態更新)
y_t = C_t · h_t                     (輸出)
```

其中 Ā = exp(Δt · A), B̄ = Δt · B 為離散化後的參數，全由當前輸入 u_t 決定（input-dependent）。

**MambaBlock 內部資料流**：
```
u ∈ R^{B×L×128}
  │
  ▼ in_proj: u → [x, z]，各 ∈ R^{B×L×256}
  │
  ▼ x 分支：
  │   conv1d(x, k=4, causal)  →  local mixing
  │   SiLU
  │   x_proj → [dt, B, C]
  │     dt:  R^{B×L×256×16}
  │     B:   R^{B×L×256×16}
  │     C:   R^{B×L×256×16}
  │   selective_scan_fwd(x, dt, A, B, C) → y
  │
  ▼ Gate: y ⊙ SiLU(z)
  │
  ▼ out_proj: 256 → 128
```

---

## 4. Pipeline Stage 詳細分析

### 4.1 效能數據 (MOT17-04-SDP, 200 frames, warmup 50)

| Stage | Mean (ms) | P95 (ms) | P99 (ms) | 佔比 | 備註 |
|-------|:---------:|:--------:|:--------:|:----:|------|
| detect | 3.25 | 4.55 | 4.63 | 51.0% | Whole-detect CUDA graph replay |
| fetch | 2.08 | 2.80 | 3.22 | 32.7% | DALI NVDEC JPEG decode |
| postprocess | 1.84 | 3.25 | 3.41 | 28.9% | Native C++/CUDA pipeline |
| track | 0.53 | 0.64 | 0.69 | 8.3% | Tracker CUDA graph |
| ingest_preprocess | 0.20 | 0.24 | 0.47 | 3.1% | Stretch-resize |
| materialize | 0.15 | 0.21 | 0.40 | 2.3% | GPU→host transfer |
| relink_write | 0.11 | 0.14 | 0.18 | 1.7% | Bridge relink |
| gmc | 0.04 | 0.06 | 0.07 | 0.6% | GPU phase correlation |
| **frame_total** | **6.36** | **7.83** | **8.68** | — | with profiling sync |

### 4.2 Postprocess 子階段 (CUDA event timing, 無 sync 失真)

| 子階段 | Mean (ms) | P95 (ms) | 佔 GPU |
|--------|:---------:|:--------:|:------:|
| post_seg_fp_hard | **0.96** | 2.30 | 54.5% |
| post_seg_native | 0.30 | 0.33 | 17.0% |
| post_seg_python_tail | 0.17 | 0.21 | 9.7% |
| post_seg_prep | 0.16 | 0.21 | 9.1% |
| post_seg_slice_quality | 0.13 | 0.14 | 7.4% |
| post_seg_tail_filter | 0.05 | 0.06 | 2.8% |
| **GPU total** | **1.76** | | 100% |
| CPU overhead | 0.08 | | |

**FP hard filter 佔 postprocess GPU 時間的 54.5%** — 是 postprocess 內最大單一開銷。
這是一個 per-box 檢查 kernel（面積 > 40000px² 的低分框標記為可疑），每秒需要處理 300 個 raw box。

### 4.3 Production vs Profile 效能差異

| 模式 | FPS | Latency | 原因 |
|------|:---:|:-------:|------|
| production (無 --profile-stages) | **175.4** | 5.70 ms | CUDA graph pipeline 全速 |
| profile (--profile-stages) | 157.1 | 6.36 ms | cudaSynchronize() 打斷 graph pipeline |

**Profiling 吃掉 ~10% 效能**（vs 舊 speed preset 的 ~30%）。
差異小於 speed preset 因為：① mamba graph 已將多數 kernel 合併為單一 replay，sync 的邊界數少很多；② fetch 仍在 graph 外，但 async NVDEC 與 graph replay 原本就是並行的。

### 4.4 Nsight Systems GPU Kernel-Level View (MOT17-04-SDP, 20 frames)

> 來源：Nsight Systems GPU trace (`new_profile_analysis.md`)
> 驗證了 Python 層級 profiling 的結論，並提供精確的 GPU kernel 時間分布。

**4 CUDA Graphs 確認**：78 次 `cudaGraphLaunch` / 20 frames ≈ **3.9 graph launches/frame**。

|< Graph | 每幀 launch 數 | 對應 |
|:-------|:-------------:|------|
| WholeDetectGraph | 1 | TRT backbone + Mamba head + decode |
| TrackerGraph | 1 | GPUByteTracker update |
| NMSGraph | 1 | C++/CUDA NMS pipeline |
| GMCGraph | ~1 | Phase correlation FFT |

**Top 10 GPU Kernels (20-frame run)**：

| Rank | Kernel | GPU Time % | Total (ms) | 類別 |
|:----:|:-------|:--------:|:---------:|------|
| 1 | `cutlass_tensorop_s1688fprop_optimized_tf32_64x64` | 9.6% | 1.92 | TRT conv |
| 2 | `cutlass_tensorop_s1688fprop_optimized_tf32_128x64` | 7.9% | 1.58 | TRT conv |
| 3 | **`selective_scan_fwd_kernel<float>`** | **6.8%** | 1.36 | **Mamba SSM** |
| 4 | `saccade::kernel::stage1_cost_fused_kernel` | 5.3% | 1.06 | Tracker assoc |
| 5 | `cudnn::nchwToNhwcKernel` | 4.3% | 0.86 | Layout convert |
| 6 | `at::native::vectorized_elementwise_kernel` | 3.8% | 0.76 | PyTorch ops |
| 7 | `at::native::unrolled_elementwise_kernel` | 3.6% | 0.71 | PyTorch ops |
| 8 | `nvjpeg::rgba2rgb` | 3.2% | 0.63 | JPEG decode |
| 9 | `sm80_xmma_fprop_implicit_gemm_indexed` | 2.3% | 0.46 | TRT fused conv |
| 10 | `saccade::kernel::fused_sinkhorn_topk_kernel` | 1.2% | 0.24 | Sinkhorn match |

**關鍵發現**：
- **TRT 卷積合計 ~20%** — backbone inference 是 GPU 最大運算開銷
- **Mamba SSM kernel = 6.8%** — selective_scan_fwd 每幀僅 ~68µs，極高效
- **Tracker association = 6.5%** — stage1_cost_fused + sinkhorn_topk
- **PyTorch elementwise ops = 7.4%** — 未進入 graph 的 residual 操作（chiefly in postprocess python_tail）
- **nvjpeg rgba2rgb = 3.2%** — DALI decode 的顏色轉換部分

**CUDA API Host 端**（20 幀總計 0.58s）：

| API Call | Time % | Calls | 說明 |
|:---------|:-----:|:-----:|------|
| `cuLibraryLoadData` | **50.7%** | 49 | 一次性 GPU library 載入（短序列下放大；長序列接近 0%） |
| `cudaLaunchKernel` | 24.4% | 2,794 | 未進入 graph 的 kernel launch (~140/frame) |
| `cudaStreamSynchronize` | 10.1% | 1,681 | 初始階段 ~84 syncs/frame；穩態降至 ~19.3/frame |
| `cudaMemcpyAsync` | 2.7% | 1,982 | D2H/H2D 傳輸 |
| `cudaGraphLaunch` | 0.7% | 78 | **4 graph 的 replay，僅佔 host 時間 0.7%** |

**Nsight 數據的核心意義**：
1. 確認了 4 個 CUDA graph **確實在工作**（78 次 graph launch，host 時間僅 0.7%）
2. 仍有 2,794 次未進入 graph 的 kernel launch（來自 postprocess 和 python tail）
3. Mamba SSM kernel 在 GPU 上的實際開銷極小（1.36ms / 20 framse = 68µs/frame）
4. 最大的剩餘 GPU 開銷在 TRT 卷積和 PyTorch elementwise ops

---

## 5. 模組貢獻分析 (Ablation)

> 條件：mamba_whole_graph preset, 2 seq SDP (MOT17-04, MOT17-10), 150 frames
> 方法：cumulative cutoff — 每一層依次開啟更多模組

| Step | Module | IDF1 | MOTA | IDs | FPS | Δ IDF1 | Δ FPS |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | Bare tracker (GMC OFF) | 18.3% | 10.9% | 106 | 176.6 | base | base |
| 1 | **+ GPU GMC** | 20.3% | 11.6% | 39 | 192.1 | **+2.0pp** | **+15.5** |
| 2 | + ReID branch + relink | 20.0% | 11.6% | 41 | 165.9 | **−0.3pp** | −26.3 |
| 3 | + Appearance bank | 20.0% | 11.6% | 41 | 162.2 | ±0.0pp | −3.6 |
| 4 | + Async pipeline | 20.0% | 11.6% | 41 | 161.8 | ±0.0pp | −0.4 |

**關鍵發現**：
1. **GMC 是唯一正向貢獻模組** — IDF1 +2.0pp, IDs −63% (106→39)
2. **ReID 在 GMC ON 下為負貢獻** — IDF1 −0.3pp (從 +2.0pp 退回 +1.7pp)，且 FPS −26.3
3. **GMC = 100% 的 IDF1 增益來源** — 第 2-4 層的 ReID/bank/async 零貢獻（且有害）

這個結論與舊 speed preset 一致，但在 mamba_whole_graph 下更極端：
ReID 不僅零增益，還**降低** IDF1（可能因為 semantic relink 在強 GMC+mamba 偵測下引入錯誤的身份合併）。

---

## 6. 與 speed preset 的關鍵差異

| 面向 | speed preset | mamba_whole_graph | 影響 |
|------|:-----------:|:-----------------:|------|
| 偵測器 | YOLO26s end-to-end TRT | TRT Backbone + Mamba head | **IDF1 +21.4pp, Rcll +26.2pp** |
| Detect CUDA graph | 無 | Whole-detect: TRT + head + decode | **detect: 7.4→3.25ms (2.3×)** |
| Tracker CUDA graph | 無 | GPUByteTracker graph | **track: 1.4→0.53ms (2.6×)** |
| 輸入解析度 | 960×960 (letterbox) | 640×640 (stretch-resize) | 更小的運算量 + 域一致性 |
| ReID | 部分 ON (heartbeat) | OFF | 省去 SigLIP2 TRT 5-8ms/frame |
| Detection Quality Scaling | ON (score×geometry) | OFF | Mamba head 內建品質信號 |
| match_thresh | 0.66 | 0.50 | 適配 Mamba 分數分佈 |
| kalman_r_scale | 1.0 | 2.8 | 更信任 motion model（強 GMC） |
| fuse_score_weight | 0.4 | **0.0** | IoU-only matching（Mamba scores 不需 fusion） |
| interpolate_max_gap | 20 | **35** | 更長的 gap 容忍（~1.17s vs ~0.67s） |

---

## 7. 設計取捨與限制

### 7.1 What You Get

- **極低延遲**：5.7 ms/frame (175 FPS) — 比舊 speed preset 快 **79%**
- **極高精度**：IDF1 73.4%, MOTA 76.9% — 比舊 speed preset **IDF1 +21.4pp, MOTA +35.3pp**
- **CUDA graph 穩定**：三層 graph 架構經完整驗證 (parity check, bit-exact)
- **生產級代碼**：`configs/presets/mamba_whole_graph.yaml` 一鍵設定所有參數

### 7.2 What You Give Up

- **ReID / appearance** — 必須關閉（tracker graph 不支援 per-det embeddings）
- **動態 batch size** — whole-detect graph 固定 shape，不同解析度需重新 capture
- **Temporal Mamba** — production 不使用（v15/v17 退步），但 checkpoint 仍支援
- **Hybrid head (P3 conv)** — v14 ckpt 使用 full Mamba，非 hybrid
- **Detection quality scaling** — Mamba head 分數已內部校準，不需 external geometry boost

### 7.3 何時不適合

- 跨攝影機 ReID / long-term identity
- 需要 appearance features 的下游任務
- 動態輸入解析度變化的場景（graph re-capture overhead）
- 需要 per-detection embeddings 做 association 的場景

---

## 8. 優化空間

> **深入分析**：whole_graph 的 kernel 級剖析(每幀 372 kernel、SM Issue 僅 22%、碎片化來源、fp16/channels_last NO-GO 實測)見 [whole-graph-kernel-fragmentation.md](research/whole-graph-kernel-fragmentation.md)。重點:detect 路徑是 **latency/碎片化-bound**(非 compute/memory-bound),最大可動槓桿為「融合 104 個 torch pointwise kernel」與「selective_scan block-size 重構」。

### 8.1 立即可行（不改架構）

| 方向 | 估計收益 | 方法 |
|------|:------:|------|
| FP hard filter 移至 postprocess 前段 | postprocess 0.96→0.5ms | 整併 FP hard filter kernel 到 native postprocess pipeline，消除額外 kernel launch |
| Backbone 換 yolo26n | detect 3.25→~2.0ms | 更小的 backbone，犧牲少量精度 |
| DALI prefetch tuning | fetch 2.08→~1.0ms | DALI pipeline depth 調優、async prefetch |

### 8.2 中期（小幅度架構改動）

| 方向 | 估計收益 | 說明 |
|------|:------:|------|
| Mamba head INT8 quantization | detect −30% | 將 Mamba head 量化為 INT8 (目前 backbone 已 FP16) |
| Postprocess all-native | postprocess −0.17ms | 消除 python_tail 段，完全 C++/CUDA |
| 合併 postprocess 到 whole-detect graph | postprocess 1.84→0 | 將目前獨立的 postprocess 也納入 whole-detect graph（目前已有 NMS graph，但非 fully merged） |

### 8.3 長期（架構變更）

| 方向 | 說明 |
|------|------|
| 端到端 CUDA graph | 將 fetch (DALI) 也納入 graph — 目前 fetch 是 graph 外最昂貴的 stage (2.08ms) |
| Mamba head export to TRT | 將 PyTorch Mamba head 匯出為 TensorRT engine，消除 PyTorch dispatch overhead |

---

## 9. 附錄：YAML 參數完整解讀

```yaml
# configs/presets/mamba_whole_graph.yaml

# --- Detection & Tiling ---
tiling: native_640              # 單張 640×640，無 tiling
preprocess: none                # stretch-resize，無 letterbox
track_person_only: false        # 所有類別都追蹤 (非 person_only)
person_geometry_prior: false    # 不套用 geometry prior filter
detection_quality_scaling: false # Mamba head 已內建品質信號

# --- Mamba Detector ---
mamba_ckpt: runs/mamba_gt_vgt_mamba_v14/best.ckpt
# v14 = PixelShuffle upsampling + Cross-Scan + full Mamba at P3
# 訓練流程: Phase 1 distillation (teacher FPN) + Phase 2 GT fine-tuning
fpn_backbone_engine: models/yolo/yolo26s_backbone_640_best.engine
# YOLO26s layers 0-22 (backbone + FPN neck), FP16 TRT engine

# CUDA graphs
use_whole_graph: true   # 捕獲 TRT + Mamba head + decode → 單一 replay
use_cuda_graph: true    # 同時啟用 head-only graph (被 whole-graph 取代)
use_tracker_graph: true # 捕獲 GPUByteTracker update → 單一 replay

# --- GMC ---
gmc: true               # 全域運動補償 (GPU phase correlation)
gmc_downscale: 4        # FFT 前降採樣倍數
gmc_fg_mask: false      # 前景遮罩 (無效，已 NO-GO)

# --- ReID ---
reid_mode: "off"        # 完全關閉 ReID (tracker graph 不支援 embeddings)

# --- Tracker ---
match_thresh: 0.50      # IoU 關聯門檻 (Mamba 專屬)
new_track_thresh: 0.28  # 新軌跡出生門檻
kalman_r_scale: 2.8     # Kalman R 矩陣縮放 >1 = 更信任 motion model
confirm_streak: 3       # 軌跡確認所需連續幀數
confirm_score_thresh: 0.50
fuse_score_weight: 0.0  # IoU-only matching (Mamba score 不需 fusion)
id_stability_filter: false
per_seq_adapt: false
geometry_suspect_support: false

# --- Post ---
interpolate_tracklets: true    # 軌跡線性插值
interpolate_max_gap: 35        # 最大插值 gap (~1.17s @30fps)
interpolate_min_track_len: 5   # 最短軌跡長度才插值
```

---

*最後更新：2026-06-06，基於 mamba_whole_graph 生產環境實測*
