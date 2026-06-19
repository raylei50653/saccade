# Detection — 模組 TODO

> 全局進度矩陣、Baseline 表與已 default flags 見 [docs/TODO.md](../../TODO.md)。本檔只列偵測模組待辦。

## 待辦

| 優先 | 項目 | 具體思路 | 預期收益 | 狀態 |
| :---: | :--- | :--- | :--- | :---: |
| P2-VGT | **VGT-Mamba (Velocity-Guided Temporal)** | 用 GMC affine flow 將歷史幀 FPN 特徵 warp 對齊到當前幀後再做 temporal Mamba。GMC 已預計算（127 KB）。 | 根治時序對齊，從源頭降低遮擋 FN | 🔄 訓練中（Phase 1） |
| P3 | **混合 Head 架構 (Hybrid Mamba-ViT)** | 低層 FPN (P3) 採 EfficientViT-style CGA 卷積頭；高層 FPN (P5) 採 Mamba 頭捕捉全局語義。 | 兼顧全局上下文建模與 TensorRT / GPU 友好的推論速度 | 📋 待實作 |
| P3 | **偵測器訓練資料改善 / 資料集補強** | `pred_h ≈ 61.4% of gt_h`，77% 近似 FP 其實有真實 GT；針對遮擋與小目標補足半身/腿/腳標註後重新微調。 | 根本解決 FN；目前所有 score-gate 手段天花板已見 | 📋 暫緩（待時序 YOLO 驗證） |

## 近期結論 / 已完成

- ✅ **mamba_head CUDA graph eval bug（2026-06-02）**：真因為 custom `selective_scan_fwd` CUDA op 跑在 legacy default stream（stream 0）→ CUDA-graph capture 不錄 scan kernel → replay 缺 scan → cls 飽和 → ~10× FP → MOTA 崩。修法：pybind binding 加 `stream_ptr` + op 傳 `torch.cuda.current_stream().cuda_stream`，graph path 改 `torch.cuda.make_graphed_callables`。隔離 bit-exact、full-SDP parity（噪音內）、FPS 95.5→110.2（+15%）。已 default `use_cuda_graph: true`。詳見 [research/mamba-cuda-graph-bug.md](research/mamba-cuda-graph-bug.md)。
- ⏸️ **ST-Mamba (Spatio-Temporal SSM)**：已訓練（stride=1, clip_len=3），單幀推理與 cross-scan 持平；時序 buffer 因固定位置掃描無法追蹤移動物體而不 work（T=3 IDF1 41.0% 崩）。已由 VGT-Mamba 取代。
- ✅ **Option F（Mamba Gated Detector）結案（2026-05-27）**→ `mamba_optimal` 是 Mamba head lineage；目前 production headline preset 已升到 `mamba_whole_graph`（whole-detect CUDA graph + T3→T1 ckpt lineage）。設計與精調見 [option-f-mamba-head.md](option-f-mamba-head.md)，歷史脈絡見 [TODO_history.md](../../TODO_history.md)。
