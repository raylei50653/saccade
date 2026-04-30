# Saccade TODO — 具體實作清單

> 主 TODO 只保留目前待辦與近期方向。已完成項、設計規範與 C++ 路線圖已移至 [docs/TODO_history.md](/home/ray/developer/ai/saccade/docs/TODO_history.md:1)。

---

## Current Focus

- [ ] **Rerank Phase 2：Bank Inject（C）+ Reciprocal Margin（D）— 收斂結論**
  - **Phase 1 結論（2026-04-29）**：`buffer_size × rerank_mode` 多樣本 scoring 無效——11 的 IDF1 系統性下降（-2.8～-3.6pp），02 的 FP/IDs 未改善。問題根源不在 appearance scoring，而在 reference 品質與 false accept 過濾。
  - **實作已完成（2026-04-29）**：
    - C++ `SemanticRelinker` 新增 `rerank_mode`、`reciprocal_margin`、`inject_reference()`、`alias`/`features` 屬性
    - `runner.py` 修正：`semantic_bank_inject` 與 `reciprocal_margin > 0` 不再強制 Python fallback（僅 non-mean rerank_mode 才用 Python）
    - `relink.py` 修正：Python `inject_reference()` 強制 `.cpu()` 防止 device mismatch
    - `ablation_rerank.py` 修正：base 加 `--no-semantic-bank-inject --semantic-buffer-size 1`；C/D/CD 均加 `--semantic-buffer-size 1`
  - **結果輸出**：`scripts/eval/output/ablation_rerank.txt` 與 `scripts/eval/output/current_margin_m*`
  - **Phase 2 結論**：
    - `buf=1 EMA` 基線上，`C+D: inject+margin=0.05` 最強，可作為 stripped reference path 的最佳組合。
    - 但在 current documented base 上，沒有任何 reciprocal margin 能乾淨取代 default。
    - `margin=0.15` 只適合保留為 `IDs/MOTA` 傾向的可選 variant，不應進 default。
  - **下一步**：停止 reciprocal margin default tuning，轉向 Phase 3 reference-quality / false-accept filtering。

- [ ] **Pipeline GPU 化 85%+**
  - 目標：把 L1 感知熱路徑推到接近全 GPU，優先移除 `PerceptionPipeline` 與 `GPUByteTracker` 內的 D2H/H2D 往返與 host-side lifecycle 決策。
  - `M1`：`PerceptionPipeline::process_detections()` 全 GPU 化 ✅ **完成（2026-04-29）**
    - CPU score sort → `argsort_scores_descending_cuda`（CUB `SortKeysDescending` + 複合 uint64 key 保證穩定排序）
    - filter 後 CPU compaction → `gather_compact3_cuda`（GPU gather kernel）
    - NMS 後 CPU compaction → `gather_compact4_cuda` + D2D memcpy（temp buffer 避免 aliasing）
    - scratch 改為 constructor 預分配（`cfg_.max_detections`），消除串流中期 `cudaMalloc`
    - 驗收：boxes / scores / classes 不再為 postprocess 做 host roundtrip ✅
  - `M2`：`GPUByteTracker::update()` lifecycle 留在 GPU ✅ **完成（2026-04-29）**
    - `collect_free_slots_kernel` + `spawn_new_tracks_kernel` + `init_covariance_if_new_kernel` 取代 CPU spawn loop
    - `compact_results_kernel`：只 D2H ~2.4 KB 結果 vs 舊 ~159 KB 全狀態（~66× 減少）
    - `get_tentative_candidates` / `update_reference_features_impl` / `set_clean_embedding_flags` 改 lazy D2H
    - 新增 `h_dirty_` flag：`update()` 結束後設為 true，lazy sync 後清除；防止未來代碼讀取 stale host arrays
    - 修正隱性 bug：舊 result-building 的 aggregate init 讓 `det_idx` 永遠為 0（所有 track 用 `embeddings[0]` 更新 ReID bank）；新 compact kernel 正確填入匹配的 det index
    - 驗收：`update()` 內大段 state D2H/H2D 顯著縮小 ✅
  - **M1+M2 實測結果（MOT17 SDP 7 序列全長，單次基準）**：
    - IDF1: ~43–45%（baseline 同樣有 ±1.5pp run-to-run 變異，源於 parallel auction kernel atomics）
    - IDs: ~930–1050（baseline run-to-run 變異 ±115；M1+M2 在同一噪聲範圍內）
    - FPS 重序列（MOT17-04）：~49 FPS，與 baseline 持平
    - FPS 輕序列（MOT17-10）：~65 FPS vs baseline ~78 FPS（−17%，5 個 extra kernel launches 的 overhead，低 occupancy 時顯著）
  - `M3`：主路徑統一走 native facade
    - `runner` 預設改走 native filter / NMS / merge / reid / tracker path
    - 不再以 Python postprocess helpers 作為 default
    - **目前狀態（2026-04-29）**：已接上 `PerceptionPipeline`，default path 可走 native `filter + NMS`，且在 `reid_crop_layout=full`、C++ `Cropper` / `FeatureExtractor` 可用時走 native `extract_reid()`
    - **端到端驗收（MOT17-09-SDP smoke / 50-frame warm run，2026-04-29）**：
      - native vs 強制 Python fallback 無明顯指標回歸
      - 5-frame smoke：IDF1 / MOTA / IDs / FP / FN 完全一致；只見 bbox 小數位浮點差
      - 50-frame warm run：native2 `59.25 FPS`, `IDF1 8.367%`, `MOTA 2.516%`, `IDs 0`, `FP 103`, `FN 5088`
      - 50-frame warm run：fallback2 `58.38 FPS`, `IDF1 8.366%`, `MOTA 2.498%`, `IDs 0`, `FP 104`, `FN 5088`
      - 結論：M3 已證明 native facade 可作為 default path；目前差異量級仍在既有 non-deterministic 噪聲範圍內
    - **限制**：M3 不等於「CPU 只管流程」；目前仍有少量 count/result D2H 與 Python-side semantic / bank / output logic
  - `M3.5`：CPU 只保留流程控制，資料面留在 GPU
    - **目標定義**：
      - CPU 只做模式切換、排程與最終輸出
      - frame / detections / embeddings / tracker state / relink state 由 GPU 持有
      - CPU 不再承接每幀 box / embedding / track-result materialization
    - **已完成（2026-04-29）**：
      - `M3.5-a`：`GPUByteTracker` 新增 caller-provided GPU result buffer path，避免每幀固定回 `std::vector<TrackResult>`
      - `M3.5-b`：runner 改吃 GPU result view；CPU 只在輸出邊界一次性讀取必要欄位
      - `M3.5-c`：appearance bank / output bank 改為 GPU-resident，且 relinker bank inject 改成窄查詢 API，避免整張 alias/features map host materialization
      - `M3.5-d`：`PerceptionPipeline` 新增 `process_detections_into()`，把 `n_filtered / n_nms` 內部 host count sync 改成 device-side count 傳遞；runner 只在單一流程邊界讀取 `post_count`
    - **M3.5 驗收（MOT17-09-SDP，2026-04-29）**：
      - 5-frame smoke：`results/tmp_m35d_countless_smoke_fix2` 正常產生追蹤輸出，semantic relink `attempts=7`
      - 50-frame smoke：`results/tmp_m35d_countless_50` 正常完成，`46.41 FPS`，semantic relink `attempts=11`
      - 結論：`M3.5-d` 的 countless path 已恢復正常，未見空輸出或 relink 停擺
    - **目前仍需 CPU 的資料面**：
      - semantic relink 本體仍是 Python / CPU resident
      - MOT output assembly / post lifecycle merge / quality filter 仍在 Python
      - runner 仍需在 `_materialize_gpu_track_results()` / `post_count` 邊界讀回少量結果供流程層消費
    - **下一步順序**：
      - `M4-a`：semantic relink motion sync 已從 full snapshot push 收斂成 candidate-id pull；`semantic_mahalanobis_threshold <= 0` 時整條 motion path 可直接跳過 ✅ **完成（2026-04-29）**
      - `M4-b`：native identity resolve pass 三階段全完成 ✅ **完成（2026-04-30）**
        - **Phase A**：Python `IdentityResolver` composer，runner 改走 `resolve_pass`，舊兩段 fallback chain 降為 else-branch
        - **Phase B**：`TrackletLifecycleMergerCpp` + `IdentityResolverCpp`（`tracker_gpu_python.cpp`）；runner 優先走 C++ path；`SemanticRelinkerCpp::resolve_cpp()` 新增 C++ internal API，inputs 只解析一次
        - **Phase C**：刪除 `_relink_prepared_candidates`、`RelinkedTrackCandidate`、`relinker` param from `_resolve_frame_tracks`；lifecycle-only fallback 直接走 lifecycle，不再過 relink 中介
        - **驗證**：`test_phase_b_parity`（8）+ `test_identity_resolver_parity`（5）+ 原有 16 + e2e 3 = **32 passed**；C++ vs Python byte-equal（IDs / alias / stats）
        - 實驗紀錄：[docs/experiments/pipeline/gpu_pipeline_m4b_identity_resolver.md](/home/ray/developer/ai/saccade/docs/experiments/pipeline/gpu_pipeline_m4b_identity_resolver.md)
      - `M4-c`：最後再評估是否需要進一步消除 `post_count` / result-count 單點回讀
  - `M4`：補齊剩餘同步瓶頸
    - parallel auction atomics 非確定性 → 考慮 deterministic assignment（CUB scan）
    - GMC 降同步成本
    - `Preprocessor::process_gpu()` 補完
    - 最後才評估 GStreamer ingest 真 zero-copy
  - 建議順序：`M1 -> M2 -> M3 -> M3.5 -> M4`

## Active Context

- 目前 `siglip2` 仍是最高 IDF1 ceiling。
- `transreid` 與 `osnet` 已完成對比，但都未超過當前 `siglip2` base。
- OSNet engine 已建立，可用 `uv run python scripts/model/build_osnet.py` 重建。
- Phase 1 multi-sample rerank 結果已存 `results/ablation_rerank/`，可用 `--skip-run` 重新讀取。
- Phase 2 ablation script 已修正，直接 `--fast` 執行即可（舊 `base/` 結果仍有效）。

## Historical Links

- 歷史 TODO / 設計規範 / C++ 路線圖： [docs/TODO_history.md](/home/ray/developer/ai/saccade/docs/TODO_history.md:1)
- Tracking base 與 relink sweep： [docs/experiments/tracking/fp_fn_recovery_and_gmc.md](/home/ray/developer/ai/saccade/docs/experiments/tracking/fp_fn_recovery_and_gmc.md:1)
- ReID backbone refresh 歸檔： [docs/experiments/reid/semantic_relink_and_crop.md](/home/ray/developer/ai/saccade/docs/experiments/reid/semantic_relink_and_crop.md:252)

---

## 🎯 專案里程碑

最後更新：2026-04-30

下一步：`M4-a` ✅、`M4-b` ✅（Phase A/B/C 全完成）。runner 熱路徑的 identity resolve 現為單一 C++ pass；`_relink_prepared_candidates` 與 `RelinkedTrackCandidate` 已清除。下一階段評估 `M4-c`（`post_count` 單點回讀是否值得消除），或轉進 Rerank Phase 3 reference-quality / false-accept filtering。
