# Saccade TODO — 具體實作清單

> 主 TODO 只保留目前待辦、近期 ablation 結論與下一步方向。已完成項、設計規範與 C++ 路線圖已移至 [docs/TODO_history.md](/docs/TODO_history.md:1)。

---

## 歸檔標準

- 主 TODO 只保留三類內容：
  - 目前真的還要做的事項
  - 近期仍會影響決策的 ablation 結論
  - 下一輪已排定的實驗 / 實作 backlog
- 內容應移入 [docs/TODO_history.md](/docs/TODO_history.md:1) 的情況：
  - 已完成，且後續不再需要逐步追蹤
  - 已收斂並明確放棄，不再作為近期 default 候選
  - 已被新方向取代，只需保留背景與結論
  - 屬於長篇實作過程、舊路線圖或階段性 milestone，而不是當前待辦
- 歸檔時原則：
  - 主 TODO 保留高訊號摘要與最終結論
  - 細節、過程、舊參數掃描與已結案子項移入 history
  - 若某方向之後重新啟動，再從 history 摘回主 TODO，而不是在主 TODO 長期保留已結案脈絡

---

## Recent Ablation Conclusions

- **MOT17-02-SDP FN 診斷與窄人加分（2026-05-06）**
  - `FN` 根因已定位：不是單純 `IDs / relink`，而是 `raw` 階段已有一部分窄人正確框，但在 `post_filter` 前後因低分被淘汰，或被鄰近大框取代。
  - 代表案例：
    - 高可見 `MISS` 仍有 `1453` 個，其中 `363` 個 `best_iou==0`
    - `gt_id=59 / 29` 在 `raw` 階段常有 `IoU 0.7~0.9` 的正確窄人框，但分數僅 `0.008~0.04`
  - 全局放寬低分地板不可用：
    - `track=0.02, new_track=0.20`：`FN -457`，但 `FP +897`，`MOTA 26.5 → 23.9`
    - `crowd_low_score_mode` 亦只帶來小幅 `FN` 改善，`FP` 反噬更大，`MOTA` 全面回退
  - 已新增實驗開關：
    - `--narrow-person-score-bonus`
    - `--narrow-person-max-width-ratio`
    - `--narrow-person-min-height-ratio`
    - `--narrow-person-min-aspect`
    - `--narrow-person-max-aspect`
  - 目前 `MOT17-02-SDP` 最佳實驗候選：
    - `--narrow-person-score-bonus 0.05`
    - `--narrow-person-max-width-ratio 0.015`
    - `--narrow-person-min-aspect 2.4`
  - 單序列結果（vs baseline `IDF1 30.8 / MOTA 26.5 / FP 1894 / FN 11616 / IDs 153`）：
    - `IDF1 32.2 / MOTA 27.5 / FP 1757 / FN 11547 / IDs 164`
  - **結論：**「選擇性保留窄人低分框」方向成立，且優於全局降 `track/new-track`；但尚未做 7-seq 驗證，**不納入 default**。

- **MOT17 current default（2026-05-06 更新）**
  - 當前最佳 live 評測設定（SDP 7 序列）：
    - `--gmc --gmc-mode gpu`
    - `--reid-trigger-mode event_any`
    - `--cross-tile-merge`
    - `--match-thresh 0.78`
    - `--semantic-threshold 0.91`
    - `--detection-quality-scaling`（已納入 default）
    - `--reid-budget 0.2`（已納入 default）
    - `--new-track-thresh 0.45`（與 `runner.py` fallback 對齊，2026-05-05）
    - D2-C CUDA tentative isolation（已納入 default，2026-05-05）
  - 最近一輪 SDP 7 序列結果（2026-05-06，`mot17.py --detector SDP`）：**IDF1 47.9%，MOTA 40.7%，IDs 648，FP 10,821，FN 55,103，Recall 50.9%，Eval FPS 51.5**。

- **Tiled Detector 流程診斷（2026-05-05）**
  - `native_960` 在 `MOT17-04-SDP / MOT17-10-SDP` 上明顯優於 `960p_2x2 tiled`：
    - tiled baseline：`IDF1 43.6 / MOTA 31.6 / IDs 187 / FP 8516 / FN 32584 / Rcll 46.0`
    - `native_960`：`IDF1 47.4 / MOTA 41.4 / IDs 151 / FP 4222 / FN 31007 / Rcll 48.7`
  - 結論：目前主問題不像 tracker threshold，而像 detection/tile 流程本身引入 seam duplicate、截斷框與分數校準污染。
  - **判斷：這是流程問題，不是單點門檻問題。**

- **Cross-tile Duplicate Merge 與參數調校（2026-05-05）**
  - 已完成 seam-aware duplicate 判定與 native CUDA/C++ port。
  - 已新增 tile diagnostics：`pre_merge_seam / post_merge_seam / merged_clusters / compression`。
  - 對 `tiled_seam_coord_weight` 與 `tiled_best_blend` 進行了徹底的網格掃描（極端傾向 best box、單純平均、0.5/0.5 等）。
  - 結果：無論參數如何調整，tiled 的 FP 始終落在 ~8000 水準，且 FN 亦高於 `native_960`（FP ~4200）。這證實了 tiled 帶來的 truncation 與 score 污染是流程層面的根本問題，無法透過 representative box 參數或 tracker / relink 補救。
  - 結論：**停止在 tiled 上繼續堆疊調參，正式將 `native_960` 設為預設 baseline。**

- **D2-C CUDA Tentative Track Isolation（2026-05-05）**
  - Stage 1/1b confirmed-only，Stage 1c tentative tracks 只拿剩餘 detection（`d_det_to_trk_` 過濾已匹配）
  - IDs 534 → 515（-3.6%），MOTA 34.8%（不變），Recall 44.7%（不變），IDF1 44.0%（-0.2pp，noise 範圍）
  - **已納入 default**：`src/tracking/tracker_gpu.cu` Stage 1/1b state=2，Stage 1c state=1

- **D1 診斷結論（2026-05-05）**
  - **60.5% 的 IDs 為 primary association 震盪**（match_gap ≤ 5f，target 仍可見）— 不是 ReID 問題
  - ReID budget=0（D1-a）：IDs -4%，FPS 崩至 17fps，不值得
  - Phase 3 gate（D1-b）：不是 IDF1 缺口根因，disabling 無效
  - P3-B (match_gap ≥ 91f)：僅 26 / 664（3.9%），天花板過低，**確認放棄**

- **46.28% → 44.2% IDF1 缺口根因確認（2026-05-05）**
  - A6 關閉（D2-A-1）：IDF1 僅 +0.1pp，MOTA -4.1pp — A6 不是根因
  - Phase 3 gate 關閉（D1-b）：無改善
  - **結論：缺口來自 M4-b 架構改動，已深植，不追查**

- **YOLO 門檻調查（2026-05-03）**
  - `conf_threshold=0.05` 因 `keep_mask = scores > min(conf, track_thresh)` 而對 FP 完全無效。
  - `track_thresh` 從 0.05 → 0.10：FN +814、MOTA -0.7pp，FP 幾乎不變。
  - **結論：保持現有設定（conf=0.05, track=0.05）。**

- **MOT17 default（2026-04-30）— 歷史最高（stale cache）**
  - `IDF1 46.28%，MOTA 35.09%，IDs 857，FP 16048`（cached .txt）
  - 差距已確認為 M4-b code 變更，不再追查。

- **A5 Post-Merge V2**：`max_cost=0.8` 是 appearance-weighted post-merge 安全邊界；**不納入 default**。若未來 online tracking IDF1 提升，可再評估。

- **A2 Reference Quality Gate**：`clean_score_threshold=0.65` 已捕捉主要品質訊號；margin/aspect bounds 無系統性增益（≤ 0.2pp IDF1），**不納入 default**。

- **GMC fg-mask**：MOT17 背景紋理足以主導 Phase Correlation peak，`--gmc-fg-mask` 無增益，**不納入 default**。

---

- **Async ReID Pipelining（2026-05-06）**
  - `--async-reid`：將 `reid_extract` 提交至 side CUDA stream，與 main stream 的 GMC 估算重疊（~1ms overlap）
  - Sync point 在 GMC 之後、`tracker.update_into` 之前，tracker 仍拿到新鮮 embeddings，無準確度退化
  - 7-seq SDP A/B：IDF1/MOTA/IDs 完全不變，**Eval FPS 54.90 → 56.34（+2.6%，-0.46ms/frame）**
  - 注意：`--profile-stages` 的 `torch.cuda.synchronize()` 會序列化 GPU 工作，在 profile 模式下看不到增益
  - 實作：`--async-reid` flag；`runner.py` kwargs `async_reid=True`
  - 結論：**增益真實但很小（~0.5ms）。目前不納入 default，保留為實驗性 flag。**

## Inter-Frame Relink Pipelining — 實作計畫（2026-05-06）

**背景**：GPU 在 `frame_total` 中只有 68% 使用率；`relink_write`（~5.4ms CPU Python）是最大空轉段。
`detect`（~5.95ms GPU）期間 `torch.cuda.synchronize()` 會釋放 GIL，理論上可讓 background CPU work 同步執行。

### 架構

- `ThreadPoolExecutor(max_workers=1)`：main thread 執行 GPU work（fetch → detect → reid → track），bg thread 執行 `relink_write` CPU Python。
- GIL 釋放：`detect` 的 `torch.cuda.synchronize()` 釋放 GIL ~5.95ms → bg thread 可在此期間執行 Python/CPU 工作。

### 資料合約

**Main thread 在 submit 前預先 materialize（避免 CUDA stream 衝突）：**

| 資料 | 操作 | 原因 |
|------|------|------|
| `boxes_gpu` | `.cpu()` | 是 `tracker_result_buffers` 的 view，frame N+1 的 `update_into` 會覆寫 |
| `fused_boxes / fused_scores / embeddings / ...` | `.cpu()` | 避免 bg thread 在 null stream 上啟動 GPU kernel 干擾 main thread postprocess |
| `motion_candidate_ids` + `motion_snapshots` | 主 thread 呼叫 | `get_motion_snapshots_for_track_ids` 需要 main CUDA stream context |

**Background task 簽名：** `(frame_id, track_results, host_batch_cpu, fused_boxes_cpu, ..., prev_track_ids_snap) → (list[str], set[int])`

**Shared mutable objects（物件參考不變，只有內部狀態改變）：** `primary_appearance_bank`, `lifecycle_merger`, `identity_resolver`, `global_id_mapper`, `relinker`, `dynamic_reid` — 透過 closure 捕獲。

### Sync 時機

```
Frame N:   fetch → detect → postprocess → [SYNC bg(N-1)] → reid → GMC → track → [SUBMIT bg(N)]
Frame N+1: fetch → detect → postprocess → [SYNC bg(N)]   → reid → ...
                                  ↑
          bg(N) 在此前必須完成（sync before first mutable state access: dynamic_reid.should_reid）
```

- Detect(5.95ms) + Postprocess(2.25ms) = **8.2ms 重疊窗口**（mean）
- Relink_write mean 5.4ms < 8.2ms → sync 通常免等待
- Relink_write P95 11.7ms vs window P95 12.3ms → 極端幀可能短暫 stall，但 mean 大幅改善

### Safety Invariants

1. `boxes_gpu` 必須 `.cpu()` 再傳給 bg（避免 tracker_result_buffers 被 frame N+1 覆寫）
2. `get_motion_snapshots_for_track_ids` 在 main thread 執行（需要 main CUDA stream）
3. `ThreadPoolExecutor(max_workers=1)`：bg 任務依序執行，frame N 完成後才開始 frame N+1
4. `profile_stages=True` 時停用 pipelining（`torch.cuda.synchronize()` 會序列化，且計時會混亂）
5. Sync 在 `dynamic_reid.should_reid()` 之前（第一個 shared mutable state 存取點）

### Flag

`--pipeline-relink`：啟用 inter-frame relink pipelining（實驗性）

### 實測結果（2026-05-06，7-seq SDP back-to-back A/B）

| Metric | Baseline | `--pipeline-relink` | Δ |
|--------|----------|---------------------|---|
| IDF1 | 47.9% | 47.9% | 0 |
| MOTA | 40.7% | 40.7% | 0 |
| IDs | 651 | 655 | noise |
| **Eval FPS** | **70.05** | **71.80** | **+1.75 (+2.5%)** |
| mean frame | 14.28ms | 13.93ms | **-0.35ms** |

**增益比預期小的原因：**
- 無 `--profile-stages` 時 relink_write 實際只有 ~2ms wall-clock（5.4ms 是 profiling 強制 sync 造成的放大）
- Python GIL 競爭在 bg thread 執行 Python loops 時增加 main thread 的等待開銷 ~1ms
- 綜合：~2ms 隱藏，~1.5ms 新增開銷，net ~0.35ms/frame

**結論：** 增益真實但很小（~0.5%）。目前不納入 default，保留為實驗性 flag。如未來 relink_write 工作量增加（更多 tracks）或 detect 時間增加，增益會放大。

---

## Active Context

- 目前 `siglip2` 仍是最高 IDF1 ceiling。
- `MOT17-02-SDP` 的主問題已確認偏向 detection/post-filter 對窄人低分框不友善，不是單純 relink 問題。
- `--narrow-person-score-bonus 0.05 --narrow-person-max-width-ratio 0.015 --narrow-person-min-aspect 2.4` 是目前僅在 `MOT17-02-SDP` 上表現最好的窄人加分候選，尚待 7-seq 驗證。
- `transreid` 與 `osnet` 已完成對比，但都未超過當前 `siglip2` base。
- OSNet engine 已建立，可用 `uv run python scripts/model/build_osnet.py` 重建。
- Phase 1 multi-sample rerank 結果已存 `results/ablation_rerank/`，可用 `--skip-run` 重新讀取。
- `native_960` 現在是 detector path 的重要 control；所有 tiled 修正都應先對照它，而不是只和舊 tiled baseline 比。
- tiled path 目前已知最大風險是 seam 汙染進入 tracker，拖髒 association、appearance bank 與 semantic relink。
- E2E 延遲現況（2026-05-06）：無 profiling 時實際約 14ms/frame（~70 FPS 7-seq avg）。`--pipeline-relink` 可再壓 ~0.35ms（+2.5%），目前不納入 default。`--profile-stages` 測量的 5.4ms relink_write 包含強制 sync 開銷，實際 wall-clock 約 2ms。

---

## Historical Links

- 歷史 TODO / 設計規範 / C++ 路線圖： [docs/TODO_history.md](/docs/TODO_history.md:1)
- Tracking base 與 relink sweep： [docs/experiments/tracking/fp_fn_recovery_and_gmc.md](/docs/experiments/tracking/fp_fn_recovery_and_gmc.md:1)
- ReID backbone refresh 歸檔： [docs/experiments/reid/semantic_relink_and_crop.md](/docs/experiments/reid/semantic_relink_and_crop.md:252)

---

## 🎯 專案里程碑

最後更新：2026-05-06

**E2E Latency 優化：relink_write -32%（2026-05-05）**：
- cProfile + wall-clock 雙管齊下定位瓶頸：`_prepare_track_candidates` 佔 `relink_write` 9.3ms/frame。
- Fix 1：`_refresh_track`（`tracker_gpu.py:673`）pairwise cosine O(K²) → `||mean||²` O(1)。
- Fix 2：`_build_prepared_candidates`（`runner.py:1067`）Phase 3 的每 candidate GPU scalar 提取批次化到 Phase 2 的單一 D2H。
- 結果：`relink_write` **8.0ms → 5.4ms (-32%)**，P95 **17.5ms → 11.7ms (-33%)**。精度不變。

**D2-C CUDA Tentative Track Isolation 完成（2026-05-05）**：
- Stage 1/1b 改為 confirmed-only（state=2），Stage 1c 新增 tentative tracks 搶剩餘 detection。
- IDs 534 → 515（-3.6%），Recall 44.7%（不變），MOTA 34.8%（不變），IDF1 44.0%（-0.2pp noise）。
- 已納入 default，當前最佳：**IDF1 44.0%，MOTA 34.8%，IDs 515，FP 10,595**。

**Detection / Tiling 流程收斂（2026-05-05）**：
- `native_960` 在 `MOT17-04-SDP / MOT17-10-SDP` 上同時贏 `FP / FN / IDs / MOTA`，且 FPS 更高。
- 經過深度掃描 `tiled_seam_coord_weight` 等代表框常數，確認無法透過後處理修復 tiled 的 seam truncation / score 污染（FP 始終約 8000，是 native_960 的兩倍）。
- **已停止調校 tiled，並將 `native_960` 與對應的 `yolo26s_960_batch1.engine` 設為 CLI 預設基準。**

**Inter-Frame Relink Pipelining 完成（2026-05-06）**：
- `ThreadPoolExecutor(max_workers=1)` + CPU pre-materialization 實作完成，精度完全不變。
- 7-seq A/B：**70.05 → 71.80 FPS (+2.5%，-0.35ms/frame)**。
- 增益受限於實際 relink_write wall-clock ~2ms（profile 量到的 5.4ms 含強制 sync 放大）與 GIL 競爭。
- **不納入 default**；`--pipeline-relink` flag 保留為實驗性選項。

**下一輪 backlog（高優先）**：
- 以 `native_960` 為新 baseline，重新評估 MOT17 全序列上的 tracker threshold（例如 `--match-thresh`, `--new-track-thresh`）。
- 尋找 `native_960` 尚未最佳化的瓶頸，進一步突破 IDF1 47.4% / MOTA 41.4%（目前 MOT17-04/10 水準）。

**B1 + A1–A8 全系列完成（2026-05-01 ~ 2026-05-05）**：
- Bank Zero-Copy P0→P2：5.8×（T=50）/ 11.9×（T=200）增益，固定成本壓到 ~175µs/frame。
- Detection Quality Scaling（A6）：MOTA +1.9pp，FP -28.8%，IDs -23.5%；已納入 default。
- GPU GMC（A8）：43.4% IDF1，優於 CPU GMC 41.9% 與 no GMC 41.7%。
- Quality-Aware Sinkhorn（A7）：IDs -2.2%，Recall 無損。
- Budgeted ReID（A3）：20% budget 帶來 +24% FPS，IDF1 持平。
