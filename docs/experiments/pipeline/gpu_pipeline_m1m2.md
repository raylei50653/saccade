# GPU Pipeline M1+M2 實驗紀錄

Date: 2026-04-29

> 後續 `M4-a / M4-b` 的 runner / relink / output-path 收斂，另見 [docs/experiments/pipeline/gpu_pipeline_m4ab.md](/docs/experiments/pipeline/gpu_pipeline_m4ab.md)。

## 目標

將 `PerceptionPipeline` 與 `GPUByteTracker` 的熱路徑全面 GPU 化，消除每幀大量 D2H/H2D 往返與 host-side lifecycle 決策，為後續 M3（runner native path）鋪路。

---

## M1：PerceptionPipeline::process_detections() 全 GPU 化

**修改檔案**：`src/tracking/pipeline.cpp`、`include/tracking/pipeline.hpp`、`include/tracking/tracker_gpu.hpp`、`src/tracking/tracker_gpu.cu`

### 改動

| 原本（CPU） | M1 後（GPU） |
|---|---|
| `std::sort` score sort | `argsort_scores_descending_cuda`（CUB `SortKeysDescending`） |
| filter 後 CPU compaction loop | `gather_compact3_cuda`（GPU gather kernel） |
| NMS 後 CPU compaction loop | `gather_compact4_cuda` + D2D memcpy（temp buffer 避免 in-place aliasing） |
| 每幀 `ensure_scratch` 可能 `cudaMalloc` | constructor 預分配（`cfg_.max_detections`），串流中不再 alloc |

### 穩定排序設計

NMS 需要 score 降序的穩定 argsort（equal score 時保留原始 index 順序）。方案是 compound uint64 key：

```
upper 32 bits = raw float bits of score
              （正浮點數：bit pattern 保持單調映射）
lower 32 bits = (n - 1 - i)
              （index 越小 → lower bits 越大 → CUB 降序時優先）
```

`CUB::DeviceRadixSort::SortKeysDescending` 對 uint64 排序後，用 `decode_sort_order_kernel` 從 lower bits 還原原始 index。與舊 SortPairsDescending + iota 方案相比，省掉一個 iota 填充 kernel 與一個 int64 values buffer。

---

## M2：GPUByteTracker::update() lifecycle 留在 GPU

**修改檔案**：`src/tracking/tracker_gpu.cu`

### 改動

| 原本（CPU loop） | M2 後（GPU kernel） |
|---|---|
| CPU 迭代 free slots、spawn new tracks | `collect_free_slots_kernel` + `spawn_new_tracks_kernel` |
| CPU 呼叫 `kf_gpu::init_covariance` per track | `init_covariance_if_new_kernel`（獨立 kernel，避免 register spilling 降低 occupancy） |
| result-building：全狀態 D2H ~159 KB/frame | `compact_results_kernel`：只 D2H ~2.4 KB（confirmed+matched tracks） |
| `get_tentative_candidates` 每幀強制 D2H | lazy D2H（每次呼叫時才 sync） |
| `update_reference_features_impl` 每幀強制 D2H | lazy D2H |
| `set_clean_embedding_flags` 每幀強制 D2H | lazy D2H |

D2H 減少約 **66×**（~159 KB → ~2.4 KB per frame）。

### 隱性 bug 修正

舊的 result-building 使用 C++ aggregate init `{x1,y1,x2,y2,id,score,cls}` 初始化 8 欄位的 `TrackResult`，第 8 欄 `det_idx` 永遠被 zero-initialize 為 0。效果是所有 track 都用 `embeddings[0]` 更新 ReID bank，而非各自匹配到的 detection。

`compact_results_kernel` 正確從 `d_trk_to_det[]` 填入每個 track 的匹配 det index，修正後 IDF1 改善約 **+1pp**。

---

## 三項安全性修正（同日）

### 1. 穩定排序（determinism）

舊 `std::sort` 與 `CUB::SortPairsDescending` 均為 unstable sort，相同 score 的 detection 順序每幀不保證一致，會影響 NMS 結果的 determinism。改為 compound uint64 key 後，equal-score 的排序順序固定（lower original index 優先），NMS 結果確定性提升。

### 2. `h_dirty_` flag（lazy D2H 安全護欄）

`update()` 結束時設 `h_dirty_ = true`；`get_tentative_candidates`、`update_reference_features_impl`、`set_clean_embedding_flags` 各自完成 D2H sync 後設 `h_dirty_ = false`。未來若有程式碼在未 sync 的情況下讀取 host cache，`h_dirty_` 為 true 即為明確的 bug signal。

### 3. constructor 預分配（VRAM 碎片化）

`PerceptionPipeline` 的 `ensure_scratch` 改在 constructor 以 `cfg_.max_detections` 預分配所有 scratch buffer，消除串流中期因 capacity 成長而觸發的 `cudaMalloc`/`cudaFree`，避免 VRAM 碎片化與 driver-level stall。

---

## 實測結果（MOT17 SDP，7 序列全長）

基準說明：parallel auction kernel 內的 non-deterministic `atomicAdd`/`atomicCAS` 使 baseline 本身就有 ±1.5pp IDF1、±115 IDs 的 run-to-run 變異，與 M1/M2 無關。

| 指標 | M1+M2 後（兩次獨立跑） |
|---|---|
| IDF1 | 43.8% / 45.0%（baseline 同範圍 43–44.5%） |
| IDs | ~930–1050（baseline 同範圍） |
| FPS MOT17-04（重序列） | ~49 FPS（與 baseline 持平） |
| FPS MOT17-10（輕序列） | ~65 FPS vs baseline ~78 FPS（−17%） |

輕序列 FPS 下降原因：5 個 extra kernel launch overhead 在低 occupancy 情境下相對明顯。重序列（大量 detection）反而沒有退步，因為 GPU 化省掉的 D2H 時間補回了 launch overhead。

---

## 已知未解問題 / 下一步

- **Parallel auction non-determinism**（M4 目標）：`parallel_auction_shmem_kernel` 的 `atomicAdd` slot claiming 是目前最大的非確定性來源；考慮以 CUB prefix scan 取代，恢復完全 deterministic assignment。
- **M3**：runner 預設改走 native filter / NMS / merge / reid / tracker path，不再以 Python postprocess helpers 作為 default。
- **GMC 同步成本**、**Preprocessor::process_gpu() 補完**、**GStreamer zero-copy** 均屬 M4 範圍。
