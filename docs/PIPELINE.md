# Saccade 演算法主線精煉 (Pipeline Distilled)

> **這份文件是什麼**：把整條感知/追蹤主線**按處理流程順序**濃縮成單檔 —— 每個階段只回答四件事：**職責 / 現行最優 / 關鍵 GO / 關鍵 NO-GO**。
> **這份文件不是什麼**：不是細節倉庫。所有數據、ablation、歸因實驗都在「補充文檔」連結裡，本檔只留結論。
>
> 配套文件分工：
> - **演算法決策精煉（本檔）** → 按流程讀「現在用什麼、為什麼、哪些死路」
> - **[stage 計時 / 延遲剖析](reference/pipeline_flow.md)** → `time_stage()` 實際串行順序與每階段耗時
> - **[NO-GO 全局登記表](reference/no_go_registry.md)** → 每條死路的完整數據與歸因（本檔以 `#N` 引用）
> - **[模組物理文檔庫](README.md)** → 各模組 architecture / research / ADR
>
> 最後更新：2026-06-17

---

## 0. 全鏈一覽

實際串行順序（對應 `evaluator.py` 的 `time_stage()`，6 個演算法階段）：

```text
[前處理]    fetch + ingest_preprocess   DALI/NVDEC 解碼 → letterbox → AdaptiveFramePool
     │
[ssm-head]  detect                      YOLO26 backbone + Mamba gated head
     │
[後處理]    postprocess                 filter → NMS → cross-tile merge → FP/quality gating
     │
[GMC]       gmc                         GPU phase-correlation warp（相機運動）
     │
[ID分配]    track                       GPUByteTracker + Kalman + Sinkhorn-Auction
     │
[relink]    relink_write                GPU 雙向橋接 relink + identity resolve
     │
[output]    MOT txt / metrics
```

> ⚠️ ReID 分支（bank_sync→budget→crop→extract）在計時上位於 GMC **之前**，但**預設全關**（見 §6），故精煉上歸入 relink 一併討論。

**現行 production 疊加帳（每步相對前一步的裸增益）：**

| 疊加 | 模組 | 增益 |
|---|---|---|
| 0 | bare tracker | start |
| +1 | **GPU GMC** | IDF1 **+2.8pp**、IDs −133（唯一壓倒性貢獻模組） |
| +2 | **GPU 雙向橋接 relink** | IDF1 **+2.1**、AssA +2.8、IDs −13.6%、FP −14% |
| +3 | **depth 同高遮擋 gate** | IDF1 +0.5、AssA +0.4（crossing-swap） |

現行最優 ≈ **IDF1 ~75.9 / MOTA ~78 / HOTA ~68 / AssA ~66.4**（Option F mamba whole-graph + 上表三疊加，7-seq MOT17-SDP）。

**結構性鐵律（先讀，省得重蹈）：**
1. **GMC 壓倒性主導** —— 開啟後其他關聯類模組多半冗余（∆<0.4pp）。調參一律以 `tracker_core_gmc` 為基準。
2. **Appearance 天花板** —— MOT17 身份在 embedding 空間本質難分（5 模型 × 4 機制 × SR × 域訓練全撞同一上限，清晰 200px 框 rank-1 僅 57%）。外觀方向已結案。
3. **「上限不轉移」反覆出現** —— GT/oracle 算得出的增益，到了 innovation / 輸出 / 特徵空間就蒸發（#5/#34/#38/#41）。離線 oracle 漂亮 ≠ 端到端 GO。
4. **時序難進特徵層** —— Mamba temporal block、per-channel SSM A 全因 grad 崩潰退步；時序一致性只能當 emergent/soft，不能硬編 loss（#28/#29/#37）。
5. **聚合 Δ 須逐序列一致** —— test GT 在 MOTChallenge server 不可本地評，per-seq 一致性是唯一本地過擬代理。靠單一序列撐起的聚合增益＝場景過擬（#38/#39）。

---

## 1. 前處理（fetch + ingest_preprocess）

**職責**：從 frame source 取影格（DALI/NVDEC 解碼）→ letterbox/gamma/contrast → AdaptiveFramePool → 備妥 detector tensor。**幾乎純效能議題，無精度 GO/NO-GO**。

**現行最優**：
- 解碼走 **NVJPG 硬體引擎**（jpg util ~55%）；瓶頸是**逐張阻塞呼叫餵不滿引擎（per-call bubble）**，非 CPU/IDCT/color。batched 2×（1300fps）、雙 thread 1.65× 可填 bubble。
- **letterbox_gpu fused kernel** + `async_reid` + `pipeline_relink` default ON → e2e ~107 FPS / 9.29ms/frame。
- 訓練側同型瓶頸：teacher-cache H2D `stack_scales` 45% + per-step sync 23%（非 launch-bound）；半精傳輸 + GPU 累積消 sync 得 3.0× bit-exact。

**關鍵 GO**：letterbox_gpu fused kernel、e2e latency opt（async + pipeline relink）、batched JPEG decode。

**關鍵 NO-GO / blocked**：
- **高解析度 head**：960 head 幾何 crash（input 須 ÷128）；1024 能跑但 640-head 不轉移（IDF1 74→39）。要 hi-res 須 head 重訓。
- **NV12 融合 decode+letterbox 路徑** —— kernel/Python 寫好但沒接 build/pybind、沒打過 baseline，**目前不算數**。
- **奇偶雙緩衝 detect(N)‖tracker(N-1)** —— clean-room +23% 但 correctness 0/250 不符；未驗 GMC+postprocess sync。
- channels_last(NHWC) head（mamba SSM reshape 強制 NCHW 邊界，轉置只被搬走非消除）、fp16 mamba head（association 退步 −0.7pp）。

**補充文檔**：JPEG decode 引擎 (`memory project_jpeg_decode_engine_underfed`)・e2e latency opt (`memory project_e2e_latency_opt`)・[streaming 架構](modules/streaming/architecture.md)（L3 媒體接入，附錄 A）・高解析 blocked (`memory project_hires_960_mamba_head_blocked`)

---

## 2. ssm-head（偵測模型）

**職責**：YOLO26 backbone + Mamba gated detection head 產出 raw `boxes/scores/classes`（清理在 §3 後處理）。

**現行最優**：
- **Option F Mamba Gated Detector**（production preset）：IDF1 71.2% / MOTA 76.3% / Rcll 82.3%（偵測層貢獻）。
- engine `yolo26s_960`、`--tiling native_960`、PixelShuffle head、v14 配方。
- **whole-graph CUDA graph** eval = 最優純加速（157 FPS、bit-exact、+15% FPS）；曾有 capture 漏錄 `selective_scan` 的 bug，已修（傳 current_stream + `make_graphed_callables`）。
- v14 內部 SSM（A_log/D/conv1d/x_proj/dt_proj）**從未被訓練**（scan 無 grad_fn）；「N=1 curriculum」是 eval artifact。
- 分數分佈：飽和左尾、median 0.93、門檻坐 0.3% 薄尾、框高主導。

**關鍵 GO**：Option F、whole-graph CUDA graph、**T3→T1 GT2 curriculum**（顯式 staging AssA +3.2，3/3 seed 全勝 p<0.001 —— 但增益全經 box/score 穩定性走 IoU 路徑傳導）。

**關鍵 NO-GO**：
- Tiled detection 960p 2×2/3×2（#5）—— FP ~8000，truncation + score 污染。
- Option D track-conditioned YOLO（#1）—— IDF1 31.7%，gate 無貢獻。
- Mamba temporal block v15/v17（#28）、per-channel SSM A（#29）—— R1→R2 grad 崩潰。
- frozen-YOLO / BN-only（AdaBN）—— IDF1 −17.6，BN 重校只救 21% gap，主導是真權重學習。
- 小目標高解析度恢復（#36，⏸ parked）—— 640 resize 對小目標不可逆，增益天花板 <0.5pp，成本判定。
- 框高條件化出生門檻（#38）、顯式跨幀一致性 loss（#37）、Mamba head 特徵作 relink embedding（#35）—— consistency ≠ discriminability。
- yolo26m 容量對照 —— detection/MOTA/FP win 但 AssA −3.0（DetA↔AssA 權衡），IDF1 parity；殘餘結構性，閉合須 training-side。

**補充文檔**：[Mamba Head 設計](modules/detection/option-f-mamba-head.md)・[v14-R 訓練規範](modules/detection/mamba-v14r-training-protocol.md)・[whole-graph 分析](modules/detection/mamba_whole_graph_analysis.md)・[CUDA Graph Bug](modules/detection/research/mamba-cuda-graph-bug.md)・[分數分佈研究](modules/detection/research/mamba-score-distribution-20260613.md)・[T3→T1 curriculum](modules/detection/research/mamba-t3t1-curriculum-20260613.md)・[strip 高解析設計](modules/detection/research/mamba-strip-detail-routing-design.md)

---

## 3. 後處理（postprocess）

**職責**：清理 detector raw 輸出 —— filter → NMS → cross-tile merge → FP/quality gating。內含 **6+ 隱藏 sub-stages**（均計入 postprocess 時間）。

**現行最優**：
- 主路徑優先走 native `PerceptionPipeline.process_detections_into()`，否則 Python fallback。
- default ON：`fp_hard_filter`（area=40000，移除 9021 FP / 僅 153 TP）、`detection_quality_scaling`（geometry-aware score boost）、`cross_tile_merge`（seam-aware：seam-near 放寬 dup 判定、降座標融合權重）。
- 延遲：postprocess ~3.16ms 主因 raw_boxes=300 全量 NMS；1.87ms unattributed 實為 Python tail（CPU/launch-bound，非可砍 GPU）。
- ⚠️ cross-tile merge **不再視為穩定增益來源**，僅 tiled path 的必要補救，高密場景仍是風險點。

**關鍵 GO**：FP hard filter（area=40000）、detection quality scaling、geometry priors（hard 幾何過濾）。

**關鍵 NO-GO（一整排出生門全死）**：
- Per-frame / Adaptive detection cap（#10）—— 密集場景壓至 ~21 破壞 recall；「密集=FP 多」假設在 MOT17 相反（密集=真實人多）。
- P5-2 Stage2 QualityGate（#11）、P5-3 ConsecutiveBirthGate（#12）—— 統計中性，靜態 FP 無法靠 spatial IoU 區分。
- P5-5 Proximity Birth Gate（#14）—— prox=0.3 → FN +1038 / Rcll −5.6pp。
- P5-1 Multi-birth —— FN −530 被 FP+453/IDs+12 抵消，FPS −20。
- P5-4 Scene-Adaptive narrow bonus（#13）、Narrow person score bonus（#27）、Cascade Filter（#25，MOT17 FP score 與 TP 重疊嚴重）。
- ⚠️ **MOT 輸出框不可 clip** —— GT 大量出界，clip 打斷 IoU 比對（MOTA −6.9pp）。

**補充文檔**：postprocess 優化 (`memory project_pipeline_optimizations`)・[postprocess tail 歸因](research/pipeline)（CPU/launch-bound）・出生門系列 (`memory project_p5_*`)・registry [#10–#14](reference/no_go_registry.md)

---

## 4. GMC（全局運動補償）

**職責**：估計幀間相機運動 warp，補償後再給 tracker 做關聯。

**現行最優**：
- **GPU phase-correlation**（cuFFT）、`--gmc-mode gpu`、`--gmc-downscale 8`、CUDA-graph 化（`cufftSetStream` cache）。**default ON**。
- 貢獻 **IDF1 +2.8pp、IDs −133** —— 全 pipeline 唯一顯著貢獻模組。
- PCR quality feedback：`0 < pcr < threshold` 標 uncertain（影響 ReID budget）。

**關鍵 GO**：GPU GMC 本身。就這一個，但它撐起整條鏈。

**關鍵 NO-GO（旋轉/affine 系列已系統性結案）**：
- GMC FG mask（#20）—— 背景紋理主導 PCR peak。
- Box-residual 共模修正（#34）—— GT affine 共模上限（13:41%）**不轉移到 innovation 空間**（Kalman 速度已吸收持續殘差）。
- tile-PCR affine（#40，史上最佳 affine 嘗試，勝 LK）—— 結構天花板：**2D 影像 warp 無法表示 3D 視差**；走路相機背景本身橫跨多深度，affine 額外 DOF 只放大固有 misfit。
- Horizon / depth prior（#41）—— 地平線**估得準**（GT 上限 4/7 收斂、撐過 detector）但**無利可圖**：MOT17 相機是 pan/平移非 pitch/roll（horizon 不敏感、GMC 輸出 max 2.5°），下游尺度有更優本地代理（物件自身框高）。
- LK affine —— −0.8，序列 10 崩 −4.6。

> **根因（終局）**：MOT17 相機運動是 pan/translation，純平移 GMC 取一階共同位移即足，殘餘視差交給 per-track Kalman 速度態。旋轉/affine 路線整方向關閉；復活需有真 camera roll 的資料集（空拍/穿戴式）。

**補充文檔**：[GMC 與卡爾曼消融](modules/geometry/research/fp_fn_recovery_and_gmc.md)・[box-residual 結案](research/eval/gmc_residual_correction_20260612.md)・registry [#40](reference/no_go_registry.md) / [#41](reference/no_go_registry.md)

---

## 5. ID分配（關聯 + 生命週期）

**職責**：predict/update tracker 狀態、detection↔track 關聯、Kalman 更新、生命週期狀態機（Tentative/Confirmed/Lost）、ID 穩定性過濾。

**現行最優**：
- **GPUByteTracker + Sinkhorn-Auction 混合關聯** —— 關聯延遲 0.67ms（10× 提升）。default ON。
- preset 參數：`match=0.66`、`new_track=0.28`（IDF1/MOTA/Rcll 三者 Pareto 最優；CLI raw default 為 0.75/0.35）。
- **Kalman R scale 0.75** default ON；歷史最優 2.8 版（IDF1 +1.9/IDs −82）為另一支線。
- default ON：`interpolation`（max_gap 35）、`id_stability_filter`、`geometry_suspect_support`。
- 生命週期：`lifecycle_merge` 預設 **OFF**（GMC 下冗余）。
- **depth 同高遮擋 gate**（commit c418872b，**第一個純幾何/無外觀修復**，default ON）—— 修 crossing-swap（佔 22% IDs）：同高 occlusion gate（`|foot_gap|≤0.15h`），IDF1 75.4→75.9 / AssA 66.0→66.4 / MOTA→78.0。

**關鍵 GO**：GPUByteTracker、Sinkhorn-Auction、Kalman R scale、interpolation、ID stability filter、depth 同高 gate。

**關鍵 NO-GO**：
- **NSA-Kalman**（#8 ⚪ 被遮蔽）—— 前提真（ρ=−0.52）但與 r_scale 雙重補償；f(score) v2 重校準證訊號可用（IDF1 +1.5）但 DetA −1.44、移動相機退步，default 不開。
- **OA-SORT OAO**（#7 ⚪）—— occ 訊號真（AUC 0.727）但「整列加 cost 不改排序」機制形式錯誤。
- **vel_dir gate**（#21 ⚪）—— fast AUC 0.751 被 46% 慢速噪聲（AUC 0.526）淹沒；需 speed-conditioned。
- depth occludee-side Phase-1（#39 一部分）—— cost term bit-identically inert、hook 錯邊；正解是 occluder-side 同高 gate（見上 GO）。
- Kalman h recalibration（g(h) −0.4，h 代理假說反證）。

> **⚪ = 被遮蔽（非結構天花板）**：前提成立但機制形式錯誤/失準。blocker 移除後可組合復活，見 registry「復活前例」。

**補充文檔**：[GPU Tracker 深度解析](modules/geometry/tracker_deep_dive.md)・[生命週期狀態機](modules/lifecycle/research/tentative_confirmed_state.md)・[depth-ordering crossing-swap](modules/semantic/research/depth_ordering_crossing_swap.md)・[Motion 參數](modules/motion/README.md)・[中性 NO-GO 歸因](research/eval/neutral_nogo_signal_attribution_20260612.md)

---

## 6. relink（斷鏈重連 + identity resolve）

**職責**：把 lost track 重連回現役 detection；將 local track id 解析成穩定 identity 輸出。可選 appearance / motion / lifecycle merge。

**現行最優**：
- **GPU 雙向中點橋接 relink**（px=0.25 + scale gate）—— **preset default ON**：IDF1 **+2.1**、AssA +2.8、IDs −13.6%、FP −14%（06-11 全指標嚴格優勢）。專案級大增益。
- 機制關鍵：**farewell archive + 中點外推**改變候選生成，**繞過 age gate 結構**（age gate 原本拒掉 86–89% relink 候選 —— 這正是 motion/semantic relink 單獨測試中性的根因）。
- `async_reid` + `pipeline_relink` default ON（純 FPS，零精度損失）。
- semantic relink / appearance bank：default **OFF**（GMC 下冗余）。

**關鍵 GO**：GPU 雙向橋接 relink、async_reid、pipeline_relink。

**關鍵 NO-GO（relink gate 訊號天花板）**：
- Semantic relink（#3）、Appearance ReID Bank（#2）—— GMC 下冗余 / 零增益高代價（FPS −17.3）。
- **Appearance 能力上限**（#4，已結案）—— 5 模型 × 4 機制 × SR × 域訓練全撞同一天花板。
- Motion-based relinking（#6 ⚪）—— age gate 攔 89% 候選（已由 bidir bridge 復活）。
- Scale gate 單獨走 speed 方向（#31）、Appearance relink gate 顏色/OSNet（#32，已結案 AUC≈0.50）、occ_cover live relink（#33，長 gap 被 track_buffer=30 結構性消滅）。
- Mamba head 特徵作 relink embedding（#35）、Birth-time lost-bank relink（#23，長 gap rank-1 僅 13–33%）、Cheb-GR tracklet merge（#22）、PostMerge（#9）、ROI FPN ReID（#16）、LaSt-ViT（#15）。

> **根因**：幾何/運動殘差對「真 vs 假橋接」AUC≈0.55（近隨機）、外觀 gate AUC≈0.50；長 gap（80+）目前無單一可靠訊號。唯一已驗證正向是雙向橋接本身。閉合殘量需交叉點 **identity 訊號** —— 但那繞回 appearance 牆（#2/#32/#35）。

**補充文檔**：[雙向 relink roadmap](modules/semantic/research/bidirectional_relink_roadmap.md)・[bidir 數據分析](modules/semantic/research/bidir_relink_data_analysis.md)・[離線候選分析](modules/semantic/research/offline_relink_candidate_analysis.md)・[normalization gate](modules/semantic/research/relink_normalization_gate_analysis.md)・[appearance 天花板](research/reid/appearance_ceiling_mot17.md)・[ReID 架構](modules/reid/architecture.md)

---

## 附錄 A：基礎設施（流程外）

感知熱路徑之外的子系統，不在演算法主線，但完整性列此（前處理 §1 的 RTSP/DALI 接入面也屬 L3 streaming）：

| 層 | 模組 | 職責 | 文檔 |
|---|---|---|---|
| L3 媒體接入 | **streaming** | RTSP 解碼、DALI/NVDEC 預處理、零拷貝 | [架構](modules/streaming/architecture.md)・[RTSP 規範](modules/streaming/runbooks/rtsp_contract.md) |
| L4 長期記憶 | **storage** | Redis 微批寫入、ChromaDB 向量索引、混合檢索 | [架構](modules/storage/architecture.md)・[API/Schema](modules/storage/api_spec.md) |
| L5 認知推理 | **cognition** | 本地 Llama 邊緣 Agentic RAG、事件分析 | [架構](modules/cognition/architecture.md) |
| L6 資源健康 | **resource** | VRAM 監測、三階 Hysteresis 降級 | [降級架構](modules/resource/architecture_degradation.md)・[VRAM OOM](modules/resource/runbooks/vram_oom.md) |

---

## 附錄 B：怎麼維護這份檔

- 本檔只放**結論一行**；任何數據/ablation/歸因細節寫進對應 `modules/*/research/` 或 [registry](reference/no_go_registry.md)，這裡只引 `#N` 或連結。
- 新增 GO/NO-GO 結論時：先更新 registry（含歸因），再回填本檔對應階段的一行。
- baseline 數字變動時更新 §0 疊加帳與各階段「現行最優」。
- 寫作規範見 [DOC_MAINTENANCE.md](DOC_MAINTENANCE.md)。
</content>
