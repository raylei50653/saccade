# Geometry Module (幾何與卡爾曼模組)

## 📐 模組職責
負責軌跡的幾何形變約束、卡爾曼濾波運動預測與 GMC (全局運動補償) 相機漂移修正。

## 🟢 目前現況
* **GPU GMC (全局運動補償)** 已全面落地。相機運動對齊全階段均在 GPU (CUDA) 上完成：包括 GPU 影像傅立葉變換 (FFT)、相位互相關 (Phase Correlation)、以及將原先單線程 $O(N)$ 複雜度改為 256 線程並行 Reduction 運算的 **Peak Find 尋峰算子**。這使單幀 GMC 耗時從 0.71ms 壓減至 0.28ms（加速 12.5×），避免了 Host 端與 Device 端的數據頻繁拷貝。
* **幾何先驗硬限制與自適應檢測品質縮放 (Detection Quality Scaling)** 仍保留為模組能力；現行 `mamba_whole_graph` headline preset 關閉 `detection_quality_scaling` 與 `id_stability_filter`，避免和 Mamba 分佈重校準重疊。
* **時序流對齊引導**：GMC 生成的仿射變換矩陣現在會被寫入 `StreamState.gmc_mat_buffer` 隊列中，供 `detection` 模組的 `_gmc_matrices_to_flow` 算子調用，融合成時序流場。

## 🔗 I/O & Dataflow

| | |
|---|---|
| **Pipeline stage** | `gmc`（獨立階段）+ `postprocess` 幾何先驗 + `track` Kalman（見 [pipeline_flow.md](../../reference/pipeline_flow.md)） |
| **輸入** | 當前 + 前一幀（GMC 的 FFT / phase-correlation）；detections（可選幾何先驗縮放、id_stability） |
| **輸出** | `gmc_warp` + `gmc_uncertain`（→ tracker 消耗、→ detection Mamba flow gate）；可選 geometry quality scaling（→ postprocess）；Kalman predict/update 協方差 |
| **上游 → 下游** | `prev/cur frame → gmc (GPU FFT→PCR→peak-find) → gmc_warp → track；detections → 幾何先驗 → postprocess` |

## ⚖️ GO / NO-GO 決策

> 完整脈絡見 [TODO_history.md](../../TODO_history.md)；近期待辦見 [TODO.md](TODO.md)。

| 日期 | 項目 | 結論 |
|------|------|------|
| 2026-05-07 | GPU GMC（FFT/PCR + peak-find 並行 reduction） | ✅ GO，單幀 0.71→0.28ms（12.5×） |
| 2026-06-01 | kalman_r_scale=2.8（動態 R 矩陣） | ✅ GO，IDF1 +1.9pp / IDs -82 / FP -940 |
| — | NSA-Kalman（Noise Scale Adaptive） | ❌ NO-GO（無效） |
| 2026-05 | gmc_fg_mask | ❌ NO-GO（背景紋理主導 PCR peak）；保留 PCR uncertain thr=8.0 作保護 |
| 2026-06-01 | vel_dir gate | ❌ NO-GO |
| 2026-06-12 | GMC box-residual 共模修正（innovation 自迴饋） | ❌ NO-GO（全 4 模式負，最佳 lost-only 74.7 vs 75.1）；GT affine 共模上限不轉移到 innovation 空間，registry [#34](../../reference/no_go_registry.md) |
| 2026-06-12 | GMC 旋轉系列（LK affine / box-residual probe） | ❌ NO-GO（LK affine −0.8）；tile phase-corr→affine 為唯一未否證路線 |
| 2026-06-01 | MOT 輸出框 clip | ⚠️ 禁止（GT 大量出界，clip 打斷 IoU → MOTA -6.9pp） |

## 📋 模組 TODO

詳見 [TODO.md](TODO.md)。
