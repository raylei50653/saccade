# Production Pipeline Profiling TODO

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-09-17 -->

基準固定於 the current `mamba_whole_graph_m` + SDP + `--double-buffer`
production path. Measurement contract:
[production_db_critical_path_contract.md](production_db_critical_path_contract.md).
Closure (2026-09-17):
[production_db_critical_path_20260917.md](production_db_critical_path_20260917.md)
+ [production_db_critical_path_20260917.json](../../reference/benchmarks/production_db_critical_path_20260917.json).

本輪工作的目的不是直接最佳化，而是確認目前 ~350 FPS 區間的主要限制究竟來自實際演算法計算，還是 fixed-capacity work、memory staging、同步與排程造成的 structural overhead。

來源地圖：[Production frame path：代碼閱讀地圖](../../reference/production_pipeline_code_map.md)
與 [MOT17 mamba_whole_graph_m SDP double-buffer path](mot17_mamba_whole_graph_m_sdp_double_buffer.md)。
上述 FPS 區間是背景範圍；可比較的 throughput protocol 由 contract 的 P 層定義。
#419/#431–#433 的 routing harness FPS 不是 production baseline。

* [x] **P1 — 建立不改變 production scheduling 的 profiling 基線**

  Contract + scan-anchored nsys JSON + `--profile-frame-csv` + clean P0.
  Warmed production: **347.83 FPS / 2.875 ms**. Detect span **2.65 ms**.
  `--profile-stages` 仍會關掉 double-buffer，未用作 production 數字。

* [x] **P2 — 驗證 fixed-capacity tracker work 是否形成性能底限**

  Auction 34–35 µs 與 sinkhorn 75–77 µs 在 11 vs 44 tracks 上不變。
  Occlusion 148–165 µs 近似固定。NMS select 39 vs 243 µs **會**隨 occupancy 變。
  結論：association compute 是 Tcap/Dcap floor；NMS 不是。

* [x] **P3 — 調查 host synchronization 與跨幀 overlap 損失**

  `cudaStreamSynchronize` API 23 µs（成本 A）。`outside_detect_remainder`
  **0.23 ms**（P period − D detect span；成本 B，opportunity）。B 才是
  throughput 項。禁止用 production period − nsys GPU-union busy 當 bubble。
  Host ledger 的 `post_graph_count_wait` ~1.8 ms 是 CPU 等 detect，不是額外 GPU 工作。

* [x] **P4 — 拆解 GMC 的 compute 與 memory-traffic 成本**

  FFT+downscale 暴露約 6 µs。1080p DtoD staging ~47 MB，落在 exposed memcpy
  0.11 ms 裡；640×480 無 8 MB 以上 staging。不要因 byte count 先做 #341。

* [x] **P5 — 評估 private continuation 與 multi-pass association 的實際工作價值**

  S0/S1：常跑 + 有效工作。S1b/S1c：常跑 + 少量工作。S2：常跑 + 幾乎沒工作
  （0.00–0.04 assigns/frame）。Private append 每幀加 0.17–3.1 框。
  未移除任何 pass。

* [x] **P6 — Production performance bottleneck closure**

  Primary: detector whole-graph（class D），scan 0.38 ms 是其內最大可攻切片。
  Secondary: 0.23 ms detect-to-detect tail（B+C）。
  Tertiary: fixed-capacity association 暴露 ~0.15 ms（A）。
  Tracker/GMC/NMS 整體接近 overlap 飽和（class G）——藏住的 duration 不是 FPS 槓桿。

## Boundary

本輪只做 measurement / attribution，不直接改演算法、不以 profiling instrumentation
下的 FPS 取代 production throughput，也不因單一 kernel utilization 低就直接提出最佳化方案。

下一步 optimization 另開 issue / PR，必須對準上列已被證明在 production critical
path 上的 bottleneck。
