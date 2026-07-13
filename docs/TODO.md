# Saccade TODO — 具體實作清單

> 主 TODO 只保留目前待辦、近期 ablation 結論與下一步方向。已完成項、設計規範與 C++ 路線圖已移至 [TODO_history.md](TODO_history.md)。

---

## 歸檔標準

- 主 TODO 只保留三類內容：
  - 目前真的還要做的事項
  - 近期仍會影響決策的 ablation 結論
  - 下一輪已排定的實驗 / 實作 backlog
- 內容應移入 [TODO_history.md](TODO_history.md) 的情況：
  - 已完成，且後續不再需要逐步追蹤
  - 已收斂並明確放棄，不再作為近期 default 候選
  - 已被新方向取代，只需保留背景與結論
  - 屬於長篇實作過程、舊路線圖或階段性 milestone，而不是當前待辦
- 歸檔時原則：
  - 主 TODO 保留高訊號摘要與最終結論
  - 細節、過程、舊參數掃描與已結案子項移入 history
  - 若某方向之後重新啟動，再從 history 摘回主 TODO，而不是在主 TODO 長期保留已結案脈絡

---

## 系統模組實作進度表

| 系統分層/模組 | 當前實作狀態 | 關鍵特徵與已完成里程碑 |
|---|---|---|
| **Media & Streaming** (媒體接入) | 🟢 高效能工業級 | Canonical RTSP 規範與讀寫分離、GstClient C++ 零拷貝解碼 (5-Buffer 狀態機 + Per-buffer CUDA Stream)、DALI GPU 預處理、 Watchdog 自動斷線恢復 |
| **L1 Perception** (感測/追蹤) | 🟢 穩定落地 | YOLO26s/m/l TRT 偵測推理、GPUByteTracker 雙階段關聯、GMC 運動補償與光線自適應、動態預測協方差 R 矩陣 |
| **L2 Deduplication** (去重) | 🟢 穩定落地 | SigLIP 2 特徵提取、Saccade Heartbeat 影格更新隔離、AsyncEmbeddingDispatcher 雙串流、FeatureBank 向量記憶庫 |
| **L3–L4 Storage** (快取/儲存) | 🟢 穩定運行 | Redis RPUSH + Pipelining 批次寫入、Micro-batching (100ms 聚合，降低 90% QPS)、ChromaDB 混合向量語意查詢、資料過期快取清理與冷備份 |
| **L5 Cognition** (認知推理) | 🟢 穩定落地 | LlamaIndex + local BAAI 嵌入 + llama3 邊緣 Agentic RAG 本地推理、高熵事件觸發、視覺二次查詢與跨鏡頭 ReID 關聯 |
| **L6 Resource** (資源調度) | 🟢 穩定運行 | ResourceManager 階梯式資源降級管理（NORMAL → REDUCED → FAST_PATH → EMERGENCY）、滯後控制（Hysteresis）預防切換震盪 |
| **Infra & CI/CD** (基礎設施) | 🟢 維運完備 | Systemd modular user 服務生命週期管理 (`saccade-*`)、Github Action (Ruff / Mypy / Pytest / C++ CUDA 容器自動編譯與推送 GHCR) |

---

## 當前 Baseline（2026-06-21 `frozen_v2` run）

<!-- fact-owner: current-baseline = docs/TODO.md -->
> 本節是「當前 baseline 數字」的唯一事實來源（single fact owner）。其他入口文件只鏡射並回連此處，不另立數字。

| preset | IDF1 | MOTA | IDs | Rcll | FP | FPS | 備註 |
|--------|------|------|-----|------|-----|-----|------|
| **YOLO26s + Mamba + GPU tracker** (`mamba_whole_graph`) | **78.2%** | **78.4%** | **413** | 81.0% | 2589 | **269.47** | **當前 baseline**，`frozen_v2`、整圖 CUDA graph + double-buffer + bidir bridge relink + GMC cuFFT graph + same-height occlusion gate + **OAO duration-ramp**，ReID off；HOTA 70.2/DetA 70.9/AssA 69.7/Prcn 97.2 |
| **speed**（yolo26s） | **52.0%** | **41.6%** | **475** | 55.0% | 14687 | **97.9** | Baseline s |
| **baseline**（yolo26m） | **51.4%** | **43.5%** | **502** | 59.0% | — | ~85 | Baseline m |
| **gated_det_v1**（Option E） | **56.9%** | **52.5%** | **515** | 56.2% | 3712 | ~71 | |
| **e-v2 α_tier**（Option E-v2） | **55.6%** | **54.2%** | **545** | 57.3% | **2932** | ~37 | |
| **mamba_optimal**（Option F, P1 PixelShuffle） | **71.2%** | **76.3%** | 665 | 82.3% | 6050 | **100.9** | 單向掃描 |
| **mamba_optimal**（head-only graph，前身） | **73.4%** | **77.1%** | **533** | 81.0% | 3774 | 116.7 | whole_graph 前身（同精度、無整圖加速）；HOTA 66.7/AssA 64.0 |
| **P3 Hybrid** (conv P3 + Mamba P4/P5) | **72.8%** | 75.6% | 652 | 82.0% | 6503 | 77.6 | 歷史 head variant |
| **P2-ST** (Spatio-Temporal) T=1 eval | 71.6% | 75.4% | 689 | 81.9% | 6543 | 92.0 | 時序頭，單幀推理 |
| **VGT Flow-Gated** (GMC flow gate) T=1 | **72.9%** | 76.1% | 659 | **82.5%** | 6454 | 85.6 | 歷史高分 head variant，flow 為輸入非 warp |
| VGT T=3 (buffer) | 41.0% | 40.2% | 1603 | 68.7% | 30378 | 40.8 | 時序 buffer 不 work，訓練/推理時序不一致 |

當前 `mamba_whole_graph` preset 的關鍵 default：`native_640`、`preprocess=none`、`use_whole_graph`、`use_cuda_graph`、`use_tracker_graph`、`reid_mode=off`、`gmc_downscale=4`、`relink_bridge_enabled`、`match_thresh=0.50`、`new_track_thresh=0.28`、`kalman_r_scale=2.8`、`interpolate_max_gap=35`、`oao_tau=0.50`、`oao_ramp_frames=25`、`multiplicative_cost=true`、`sinkhorn_lambda=10`、`stability_cost_w=0.20`。

legacy `native_960` presets (`speed` / `baseline`) remain useful for comparison, but they are no longer the current baseline.

---

## 模組 TODO 索引

> 模組專屬待辦已物理拆分至各 `docs/modules/<name>/TODO.md`，它們是 sole-active 的唯一 live state。 [DEVELOPMENT.md 模組現狀總覽](../DEVELOPMENT.md#模組現狀總覽)只提供穩定入口。

| 模組 | TODO | 模組 | TODO |
|------|------|------|------|
| detection | [↗](modules/detection/TODO.md) | semantic | [↗](modules/semantic/TODO.md) |
| geometry | [↗](modules/geometry/TODO.md) | trigger | [↗](modules/trigger/TODO.md) |
| motion | [↗](modules/motion/TODO.md) | streaming | [↗](modules/streaming/TODO.md) |
| reid | [↗](modules/reid/TODO.md) | storage | [↗](modules/storage/TODO.md) |
| lifecycle | [↗](modules/lifecycle/TODO.md) | cognition | [↗](modules/cognition/TODO.md) |
| | | resource | [↗](modules/resource/TODO.md) |

---

## 跨模組待辦

| 優先 | 項目 | 行動 | 預期收益 |
|------|------|------|---------|
| P2 | **測試覆蓋率提升（66% → 70%+）** | 見下方覆蓋率任務清單；按模組落地（如 lifecycle 切片見 [lifecycle/TODO.md](modules/lifecycle/TODO.md)） | 穩定性、CI 保護、開發信心 |

### 測試覆蓋率任務清單（P2）

> 詳細報告：[TESTING.md](TESTING.md)

| 優先 | 模組 | 覆蓋率 | 未覆蓋行 | 狀態 |
|------|------|--------|----------|------|
| P2-1 | `perception/eval/evaluator.py` | 40% | 734 | **部分完成**：lifecycle helper slice 已補；剩餘 `run_eval` branch coverage |

**目標**：
- 🔄 短期 v4：`perception/eval/evaluator.py` (40%)
- 📋 中期：`perception/eval/evaluator.py` (40%), `perception/eval/detection.py` (49%)
- 📋 長期：API 模組、media 模組、native 測試

---

## 算法方向探索（已結案，詳見 history）

> 背景：2026-05-17 yolo26s 參數優化觸天花板（threshold 調整 MOTA 波動 <0.5pp）→「需要新架構而非調參」。後續 Option D/E/F 探索結論已歸檔至 [TODO_history.md](TODO_history.md) 的「Algorithm Direction Exploration」節：
>
> - **Option D**（Track-Conditioned YOLO）❌ NO-GO（2026-05-19）：IDF1 31.7%，gate 無貢獻、recall 天花板。
> - **Option E-v2**（Quality-Gated Temporal Fusion）✅ GO（2026-05-22）：MOTA 54.2% / FP -21%；後被 Option F 取代。
> - **Option F**（Mamba Gated Detector）✅ 結案（2026-05-27）→ `mamba_optimal` 為 Mamba head lineage；current headline preset 是其 whole-graph / tracker-core evolution `mamba_whole_graph`。PixelShuffle 上取樣 + Stretch-Resize 域一致 → IDF1 71.2% / MOTA 76.3% / Rcll 82.3%。訓練流程、breakthrough 與核心精調見 history。當前指標見上方 baseline 表。
>
> Mamba 檢測頭中長期優化（ST-Mamba / VGT-Mamba / Hybrid Mamba-ViT / CUDA-graph✅）已拆分至 [detection/TODO.md](modules/detection/TODO.md)。

---

## Historical Links

- 歷史 TODO / 設計規範 / C++ 路線圖：[TODO_history.md](TODO_history.md)
- Tracking base 與 relink sweep：[fp_fn_recovery_and_gmc.md](modules/geometry/research/fp_fn_recovery_and_gmc.md)
- ReID backbone refresh 歸檔：[semantic_relink_and_crop.md](modules/reid/research/semantic_relink_and_crop.md)
