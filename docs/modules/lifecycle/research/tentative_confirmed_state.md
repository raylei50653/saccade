# Tracking 改進規劃：Tentative/Confirmed 狀態機 + 三層 Detection 分級

## 目標 (Objective)

現行 `GPUByteTracker` 採用標準 ByteTrack 兩段匹配，new track 初始化門檻固定為 `high_thresh=0.5`。
這導致 score 在 0.25~0.5 之間的真實目標永遠無法被追蹤，是目前 FN=58,830 >> FP=16,257、Recall=47.6% 的主因之一。

**核心改進**：引入 Tentative/Confirmed 狀態機，讓中分 detection 有觀察期，而非直接捨棄。

---

## 設計說明

### Detection 三層分級

| 分級 | Score 範圍 | 行為 |
|------|-----------|------|
| High | `score ≥ 0.5` | Stage 1 匹配 + 允許初始化新 track（直接 Tentative） |
| Mid  | `0.40 ≤ score < 0.5` | Stage 1 匹配（延續已有 track）+ 允許初始化新 Tentative track |
| Low  | `0.1 ≤ score < 0.40` | Stage 2 匹配（只延續已有 track，不建立新 track） |
| Discard | `score < 0.1` | 丟棄 |

> 目前 `high_thresh=0.5`、`track_thresh=0.1`，New track 只來自 High。
> 改動後：High + Mid 均可建立新 Tentative track。

### 狀態機 (State Machine)

```
Detection (High/Mid, unmatched)
         │
         ▼
    [Tentative]  ──── 連續 N 幀匹配 + avg_score > 0.5 ────▶  [Confirmed]
         │                                                         │
         │ 1~2 幀未匹配                                           │ max_age 幀未匹配
         ▼                                                         ▼
     [Deleted]                                               [Deleted]
```

- **Tentative → Confirmed**：連續 `confirm_streak` 幀（建議 N=3）成功匹配，且最近 N 幀 `avg_score > confirm_score_thresh`（建議 0.5）
- **Tentative → Deleted**：只要 1 幀未匹配即刪除（快速清除雜訊）
- **Confirmed → Deleted**：連續 `max_age`（目前 30）幀未匹配
- **輸出**：只輸出 **Confirmed** 狀態的 track（Tentative 不寫 MOT 結果，不影響 IDs 指標）

---

## 實作路徑 (`src/tracking/tracker_gpu.cu`)

### 1. 資料結構擴充

現行 `TrackPool` 已有：`h_active_raw_[]`、`h_age_[]`、`h_scores_[]`

新增欄位：
```cpp
// Track state machine
std::vector<int>   h_state_;         // 0=Empty, 1=Tentative, 2=Confirmed
std::vector<int>   h_hit_streak_;    // 連續成功匹配幀數
std::vector<float> h_score_sum_;     // 最近 N 幀 score 累積（用於算平均）
```

GPU 端同步（D2H/H2D）加入對應 device buffer。

建議採用 **SoA (Structure of Arrays)** 佈局以確保 Memory Coalescing（現行已是 SoA，繼續沿用）。

### 2. Detection 分組邏輯

```cpp
std::vector<int> high_dets, mid_dets, low_dets;
for (int d = 0; d < num_dets; ++d) {
    float s = h_det_scores_inp[d];
    if      (s >= high_thresh_)  high_dets.push_back(d);  // ≥ 0.5
    else if (s >= mid_thresh_)   mid_dets.push_back(d);   // 0.40~0.5
    else if (s >= track_thresh_) low_dets.push_back(d);   // 0.1~0.40
}
```

新增參數：`mid_thresh_`（預設 0.40）。

### 3. 匹配順序

```
Stage 1a: high_dets → 所有 Confirmed + Tentative tracks（IoU, match_thresh）
Stage 1b: mid_dets  → Stage 1a 未匹配的 Confirmed + Tentative tracks（IoU, match_thresh）
Stage 2:  low_dets  → Stage 1 全部未匹配的 Confirmed tracks（IoU, min_iou=0.5）
           └─ Tentative tracks 不參與 Stage 2（避免用低分延命）
```

### 4. Track 初始化

```cpp
// 未匹配的 high/mid dets → 初始化 Tentative
for (int d : unmatched_high_mid) {
    int slot = find_empty_slot();
    h_state_[slot]      = TENTATIVE;
    h_hit_streak_[slot] = 1;
    h_score_sum_[slot]  = score;
    // ... Kalman init ...
}
```

### 5. 狀態更新 Kernel（建議獨立函式）

```cpp
void updateTrackStates() {
    for (int t : active_trks) {
        if (matched) {
            h_hit_streak_[t]++;
            h_score_sum_[t] += h_scores_[t];
            if (h_state_[t] == TENTATIVE
                && h_hit_streak_[t] >= confirm_streak_
                && h_score_sum_[t] / confirm_streak_ >= confirm_score_thresh_) {
                h_state_[t] = CONFIRMED;
            }
        } else {
            if (h_state_[t] == TENTATIVE) {
                deactivate(t);   // 立即刪除
            } else {
                h_hit_streak_[t] = 0;
                h_score_sum_[t]  = 0.0f;
                if (h_age_[t] > max_age_) deactivate(t);
            }
        }
    }
}
```

### 6. 輸出過濾

```cpp
// 只輸出 Confirmed tracks（age == 0 代表本幀有匹配）
if (h_active_raw_[i] && h_state_[i] == CONFIRMED && h_age_[i] == 0) {
    results.push_back(...);
}
```

---

## 新增超參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `mid_thresh` | 0.40 | Mid-conf 分界，低於此不建立新 track |
| `confirm_streak` | 3 | Tentative → Confirmed 所需連續匹配幀數 |
| `confirm_score_thresh` | 0.50 | 升確認所需平均 score |

於 `set_params()` 介面中新增，`scripts/eval/mot17.py` 對應新增 `--mid-thresh`、`--confirm-streak`、`--confirm-score-thresh` CLI 參數（主邏輯實作於 `src/saccade/perception/eval/runner.py`）。

---

## 記憶體預估

現行 `max_objs_=300`，新增 3 個 int/float 陣列：
- `h_state_`：300 × 4B = 1.2 KB
- `h_hit_streak_`：300 × 4B = 1.2 KB
- `h_score_sum_`：300 × 4B = 1.2 KB

GPU 端同步 buffer 相同，**總額外 VRAM < 10 KB**，可忽略。

---

## 預期效果

- **Recall 提升**：score 0.40~0.49 的真實目標從「永遠 FN」變為「觀察後輸出」
- **FP 抑制**：Tentative 需連續 3 幀才升 Confirmed，單幀雜訊偵測不會進入輸出
- **IDs 影響**：Tentative 期間 track 不輸出，不會產生短命 ID 切換

---

## 狀態

- [ ] ADR 撰寫（如架構委員會認定為重大變更）
- [x] `tracker_gpu.cu` 實作
- [x] `tracker_gpu.hpp` / `tracker_gpu_python.cpp` 介面更新
- [x] `scripts/eval/mot17.py` 新增 CLI 參數（邏輯位於 `src/saccade/perception/eval/runner.py`）
- [x] MOT17 full SDP A/B 驗證（baseline vs tentative state machine）
- [x] `docs/benchmarks/` 更新比較結果

## Full SDP A/B 驗證（2026-04-26）

| Config | Output | IDF1 | Rcll | Prcn | FP | FN | IDs | MOTA | FPS |
|--------|--------|------|------|------|----|----|-----|------|-----|
| Baseline off | `results/reid_sdp_off` | 44.6% | 44.4% | 80.1% | 12,432 | 62,396 | 604 | 32.8% | 154.67 |
| Tentative default probe (`mid=0.25`, `confirm_score=0.40`) | `results/tentative_sdp_off` | 45.8% | 48.5% | 75.2% | 17,959 | 57,872 | 656 | 31.9% | 151.75 |
| Tentative tuned default (`mid=0.40`, `confirm_score=0.50`) | `results/tentative_sdp_m040_c050` | 45.5% | 45.9% | 78.6% | 14,005 | 60,786 | 575 | 32.9% | 149.94 |
| Tentative geometry scale (`mid=0.40`, `confirm_score=0.50`, median-height scale) | `results/tentative_sdp_geometry` | 45.0% | 45.5% | 78.8% | 13,724 | 61,199 | 548 | 32.8% | 153.29 |
| Tentative geometry EMA (`beta=0.80`, `max_step=0.05`) | `results/tentative_sdp_geometry_ema` | 44.8% | 45.3% | 78.9% | 13,610 | 61,408 | 551 | 32.7% | 146.88 |
| Tentative median EMA (`beta=0.80`, `loosen=0.08`, `tighten=0.03`, `min_samples=5`) | `results/tentative_sdp_geometry_median_ema` | 44.7% | 45.3% | 78.9% | 13,609 | 61,460 | 545 | 32.7% | 144.77 |
| Adaptive confirmation, fixed threshold | `results/tentative_sdp_adaptive_fixed` | 45.3% | 45.5% | 78.7% | 13,863 | 61,172 | 563 | 32.7% | 148.94 |
| Adaptive confirmation + geometry | `results/tentative_sdp_geometry_adaptive_v2` | 44.7% | 45.1% | 78.9% | 13,567 | 61,697 | 541 | 32.5% | 144.52 |

結論：`mid=0.40`、`confirm_score=0.50` 是目前較好的預設，保留 IDF1/IDs 改善，同時避免 `mid=0.25` 帶來的 FP 爆量。
幾何縮放版進一步降低 IDs（575 → 548）且吞吐回升，但 IDF1/MOTA 低於固定 tuned default，因此目前保留為 `--geometry-mid-scale` 選配模式。
EMA + step clamp 可降低門檻抖動，但這組參數未改善 full SDP accuracy；保留為幾何縮放內部穩定機制，後續需調 `beta/max_step` 或改成場景切換時重置。
median EMA + asymmetric step 將 IDs 進一步降到 545，但 FN 增加且 FPS 下降，不適合作為預設 accuracy profile。
Adaptive confirmation 目前只適合實驗：它可把 IDs 壓到 541~563，但 FN/MOTA 代價高於固定 tuned default。

## Lazy ReID Arbiter 入口（2026-04-26）

已新增 `get_tentative_candidates()` snapshot API，先作為 Phase 1 的觀察入口，不改追蹤決策。

Candidate snapshot 欄位：
- `obj_id`
- `class_id`
- `age`
- `hit_streak`
- `required_confirm_streak`
- `score`
- `x1, y1, x2, y2`

MOT17 eval 新增 profiling 參數：
- `--profile-lazy-reid-candidates`
- `--lazy-reid-min-hit-streak`（預設 2）

Smoke profile：
- Command: `MOT17-09-SDP --max-frames 80 --profile-stages --profile-lazy-reid-candidates`
- Lazy ReID candidates: `0.20/frame`，`16 total`

Lazy embedding profile：
- Command: `MOT17-09-SDP --max-frames 80 --profile-stages --profile-lazy-reid-embeddings --reid-model transreid`
- Lazy ReID candidates: `0.20/frame`，`16 total`
- Lazy ReID crops: `0.44/frame`
- Self-consistency pairs: `17`
- Mean cosine: `0.852`
- Pass@0.85: `58.8%`
- Arbiter dry-run checks: `16`
- Arbiter dry-run approve: `10` (`62.5%`)
- Synchronous profile cost: `2.32 ms/frame`

## 性能優化紀錄 (2026-04-26)

### 成果：Covariance Transfer 優化
針對 GPU-CPU 數據傳輸進行了精簡，減少了不必要的同步點。

**Benchmark 對比：**

| N (目標數) | P50 Before | P50 After | P99 Before | P99 After | 改善幅度 (P99) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 100 | 0.767 ms | 0.579 ms | 2.295 ms | 1.695 ms | -26% |
| 300 | 1.348 ms | 1.116 ms | 3.662 ms | 2.398 ms | -35% |
| 500 | 2.479 ms | 2.459 ms | 4.062 ms | 3.832 ms | -6% |
| 1000 | 7.555 ms | 7.069 ms | 11.970 ms | 8.892 ms | -26% |

### 成果：GPU Zero-Sync Association (Phase 3 實裝)
已成功將 CPU 端的 Greedy Matching 完全替換為 GPU 原生的 Fused Sinkhorn + Parallel Auction Algorithm。
不再需要透過 `d_states_` 的 H2D 傳輸來進行 CPU 關聯，徹底解決了 N=500+ 時的 CPU 延遲瓶頸。

**Benchmark 對比 (N=目標數/偵測數)：**

| N (目標數) | P50 (CPU Greedy) | P50 (GPU Auction) | P99 (CPU Greedy) | P99 (GPU Auction) | 改善幅度 (P50 / P99) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 100 | 0.579 ms | 0.602 ms | 1.695 ms | 1.076 ms | - / -36% |
| 300 | 1.116 ms | 0.627 ms | 2.398 ms | 2.306 ms | -43% / -3% |
| 500 | 2.459 ms | 0.650 ms | 3.832 ms | 1.957 ms | -73% / -48% |
| 1000 | 7.069 ms | 0.669 ms | 8.892 ms | 2.017 ms | -90% / -77% |

### 瓶頸分析與結論
1.  **當前狀態**：
    - O(N×M) 的關聯運算已完全轉移至 GPU。
    - N=1000 時的關聯時間從原本的 **~7ms** 崩跌式下降至 **~0.67ms**，吞吐量提升了超過 10 倍。
    - `d_states_` 傳輸已消除，全管線真正達成 **Zero-Copy/Zero-Sync**（直至最後產出 TrackResult 陣列才進行輕量 D2H）。
    - 狀態機 (Tentative/Confirmed) 已順利整合至 GPU Kernel 內並維持正確性。
2.  **下一步 (Goal)**：
    - 將此改動推上主分支，並準備針對 SigLIP 2 的 Semantic Drift Pipeline 進行整合測試。
