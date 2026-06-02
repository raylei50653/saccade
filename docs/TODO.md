# Saccade TODO — 具體實作清單

> 主 TODO 只保留目前待辦、近期 ablation 結論與下一步方向。已完成項、設計規範與 C++ 路線圖已移至 [docs/TODO_history.md](/docs/TODO_history.md)。

---

## 歸檔標準

- 主 TODO 只保留三類內容：
  - 目前真的還要做的事項
  - 近期仍會影響決策的 ablation 結論
  - 下一輪已排定的實驗 / 實作 backlog
- 內容應移入 [docs/TODO_history.md](/docs/TODO_history.md) 的情況：
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

## 當前 Baseline（2026-05-28 更新）

| preset | IDF1 | MOTA | IDs | Rcll | FP | FPS | 備註 |
|--------|------|------|-----|------|-----|-----|------|
| **speed**（yolo26s） | **52.0%** | **41.6%** | **475** | 55.0% | 14687 | **97.9** | Baseline s |
| **baseline**（yolo26m） | **51.4%** | **43.5%** | **502** | 59.0% | — | ~85 | Baseline m |
| **gated_det_v1**（Option E） | **56.9%** | **52.5%** | **515** | 56.2% | 3712 | ~71 | |
| **e-v2 α_tier**（Option E-v2） | **55.6%** | **54.2%** | **545** | 57.3% | **2932** | ~37 | |
| **mamba_optimal**（Option F, P1 PixelShuffle） | **71.2%** | **76.3%** | 665 | 82.3% | 6050 | **100.9** | 單向掃描 |
| **mamba_optimal**（P2 Cross-Scan 並行） | 71.3% | **76.6%** | **614** | 82.0% | 5391 | 93.8 | **當前 preset** |
| **P3 Hybrid** (conv P3 + Mamba P4/P5) | **72.8%** | 75.6% | 652 | 82.0% | 6503 | 77.6 | 最高 IDF1 |
| **P2-ST** (Spatio-Temporal) T=1 eval | 71.6% | 75.4% | 689 | 81.9% | 6543 | 92.0 | 時序頭，單幀推理 |
| **VGT Flow-Gated** (GMC flow gate) T=1 | **72.9%** | 76.1% | 659 | **82.5%** | 6454 | 85.6 | **歷史最高 IDF1**，flow 為輸入非 warp |
| VGT T=3 (buffer) | 41.0% | 40.2% | 1603 | 68.7% | 30378 | 40.8 | 時序 buffer 不 work，訓練/推理時序不一致 |

已 default 的 flag：`fuse_score_weight=0.4`、`interp`、`fp_hard_filter`（area=40000）、`kalman_r_scale=0.75`、`async_reid`、`pipeline_relink`、`gmc gpu`、`detection_quality_scaling`。

---

## 待辦事項

| 優先 | 項目 | 行動 | 預期收益 |
|------|------|------|---------|
| P2 | **測試覆蓋率提升（66% → 70%+）** | 見下方覆蓋率任務清單 | 穩定性、CI 保護、開發信心 |
| P3 | **Detector 訓練資料改善** | pred_h = 61.4% of gt_h，77% 近似 FP 有真實 GT；需補足腿/腳標注 | 根本解決 FN 問題；目前所有 score-gate 手段天花板已見 |

### 測試覆蓋率任務清單（P2）

> 詳細報告：[docs/TESTING.md](/docs/TESTING.md)

| 優先 | 模組 | 覆蓋率 | 未覆蓋行 | 狀態 |
|------|------|--------|----------|------|
| P2-1 | `perception/eval/evaluator.py` | 40% | 734 | **待實作** |

**目標**：
- 🔄 短期 v4：`perception/eval/evaluator.py` (40%)
- 📋 中期：`perception/eval/evaluator.py` (40%), `perception/eval/detection.py` (49%)
- 📋 長期：API 模組、media 模組、native 測試

---

---

## 算法方向探索（已結案，詳見 history）

> 背景：2026-05-17 yolo26s 參數優化觸天花板（threshold 調整 MOTA 波動 <0.5pp）→「需要新架構而非調參」。後續 Option D/E/F 探索結論已歸檔至 [TODO_history.md](/docs/TODO_history.md) 的「Algorithm Direction Exploration」節：
>
> - **Option D**（Track-Conditioned YOLO）❌ NO-GO（2026-05-19）：IDF1 31.7%，gate 無貢獻、recall 天花板。
> - **Option E-v2**（Quality-Gated Temporal Fusion）✅ GO（2026-05-22）：MOTA 54.2% / FP -21%；後被 Option F 取代。
> - **Option F**（Mamba Gated Detector）✅ 結案（2026-05-27）→ **當前 production preset `mamba_optimal`**：PixelShuffle 上取樣 + Stretch-Resize 域一致 → IDF1 71.2% / MOTA 76.3% / Rcll 82.3%。訓練流程、breakthrough 與 3 項核心精調（`match_thresh=0.50`、`interpolate_max_gap=35`、`gmc_downscale=4`）見 history。當前指標見上方 baseline 表。

---

## Mamba 檢測頭中長期優化待辦（2026-05-27 新增）

> 已完成項（Pixel-Shuffle、Cross-Scan，均已設為當前 preset）已歸檔至 [TODO_history.md](/docs/TODO_history.md)。

| 優先 | 項目 | 具體思路 | 預期收益 | 狀態 |
| :---: | :--- | :--- | :--- | :---: |
| **P2** | **時空聯合 Mamba (Spatio-Temporal SSM)** | 將連續 T 幀的 FPN 特徵做空間 cross-scan 後再做時間軸 SSM 掃描。**已訓練（stride=1, clip_len=3），單幀推理與 cross-scan 持平，時序 buffer 因固定位置掃描無法追蹤移動物體而不 work。** | 未達預期，需 GMC/Kalman 引導的對齊機制（見 VGT-Mamba）。 | 🔴 VGT 進行中 |
| **P2-VGT** | **VGT-Mamba (Velocity-Guided Temporal)** | 用 GMC affine flow 將歷史幀 FPN 特徵 warp 對齊到當前幀後再做 temporal Mamba。GMC 已預計算（127 KB），訓練中（Phase 1）。 | 根治時序對齊，從源頭降低遮擋 FN | 🔄 訓練中 |
| **P3** | **混合 Head 架構 (Hybrid Mamba-ViT)** | 在低層 FPN (P3) 採用硬體效率極高的 **EfficientViT-style CGA** 卷積頭；在高層 FPN (P5) 採用 **Mamba 頭** 捕捉全局語義。 | 兼顧 Mamba 的全局上下文建模優勢與 EfficientViT 對 TensorRT / GPU 架構極友好的推論速度。 | 📋 待實作 |
| ~~**P1**~~ | ~~**mamba_head CUDA graph(修 eval 整合 bug)**~~ | ✅ **完成(2026-06-02)**。真因:custom `selective_scan_fwd` CUDA op 跑在 legacy default stream(stream 0)→ CUDA-graph capture 不錄 → replay 缺 scan kernel → cls 飽和 → ~10× FP → MOTA 崩。修法:pybind binding 加 `stream_ptr` + op 傳 `torch.cuda.current_stream().cuda_stream` + graph path 改 `torch.cuda.make_graphed_callables`。隔離 bit-exact、full-SDP parity(噪音內)、FPS 95.5→110.2(+15%)。詳見 [research](research/pipeline/mamba_head_cuda_graph_eval_bug_20260602.md)。 | **+15% FPS 達成,精度持平** | ✅ 已 default(`use_cuda_graph: true`) |

---

### 📋 中長期 Backlog

#### 2. ReID + Appearance Bank

| 項目 | 內容 |
|------|------|
| **問題** | 遮擋後若無視覺特徵，僅靠 motion 預測容易匹配失敗 |
| **思路** | 啟用 ReID stack（siglip2 或更輕量 model），在遮擋後使用 embedding 尋回身份 |
| **狀態** | 📋 暫緩，待 Temporal YOLO 驗證後再評估是否需要疊加 |

#### 3. Detector 資料集補強與微調

| 項目 | 內容 |
|------|------|
| **問題** | yolo26s 對於被遮擋或只有腿/腳的行人檢測能力弱 |
| **思路** | 針對遮擋與小目標，使用包含更多半身/肢體標註的資料集重新微調 YOLO |
| **狀態** | 📋 暫緩，將優先觀察 Temporal YOLO 是否能透過時序資訊彌補此缺陷 |

---

## Historical Links

- 歷史 TODO / 設計規範 / C++ 路線圖：[TODO_history.md](TODO_history.md)
- Tracking base 與 relink sweep：[fp_fn_recovery_and_gmc.md](research/tracking/fp_fn_recovery_and_gmc.md)
- ReID backbone refresh 歸檔：[semantic_relink_and_crop.md](research/reid/semantic_relink_and_crop.md)
