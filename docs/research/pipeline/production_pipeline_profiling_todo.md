# Production Pipeline Profiling TODO

<!-- doc-status: proposed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-09-06 -->

基準固定於 `806c52cf` 的 production path。
本輪工作的目的不是直接最佳化，而是確認目前 270–350 FPS 區間的主要限制究竟來自實際演算法計算，還是 fixed-capacity work、memory staging、同步與排程造成的 structural overhead。

來源地圖：[Production frame path：代碼閱讀地圖](../../reference/production_pipeline_code_map.md)。
完整 source commit：`806c52cf8ced0836c80606559f7c38a5fcc546a3`；對應 `mamba_whole_graph` + SDP + `--double-buffer`。
上述 FPS 區間是待釐清的背景範圍，不是本清單新建立的量測結果；可比較的 throughput protocol 由 P1 定義。
本文件目前為待辦規劃，P1–P6 均未執行。

* [ ] **P1 — 建立不改變 production scheduling 的 profiling 基線**

  建立可觀測正常 `--double-buffer` production path 的 profiling 方法，避免既有 `--profile-stages` 因改變 barrier／double-buffer eligibility 而量到不同 pipeline。產出可信的 steady-state timeline 與後續任務共用的 measurement contract。

* [ ] **P2 — 驗證 fixed-capacity tracker work 是否形成性能底限**

  調查 tracker 固定 `Tcap=2048`、`Dcap=1024` 的 dense cost／association 工作量，確認 runtime 是否主要跟容量上限而非實際 active tracks / detections 成長。以 workload sweep 判斷 padding work 是否是目前 tracker latency 的結構性來源。

* [ ] **P3 — 調查 host synchronization 與跨幀 overlap 損失**

  針對 postprocess count D2H + stream synchronization，量化它是否造成 CPU stall、GPU bubble 或切斷 detector N+1 與 tracker N 的 overlap。目標是區分同步本身的成本與同步造成的排程損失。

* [ ] **P4 — 拆解 GMC 的 compute 與 memory-traffic 成本**

  分離 GMC 原圖 staging、downscale／preprocess、FFT／phase correlation 與結果傳遞成本，確認 GMC 在 production throughput 中主要受計算限制還是資料搬運／cache pollution 限制。

* [ ] **P5 — 評估 private continuation 與 multi-pass association 的實際工作價值**

  對 private NMS／prior scan，以及 S0→S1→S1b→S1c→S2 五輪 association 做 workload-aware measurement。確認哪些 stage 實際處理有效候選、哪些主要是固定排程成本，並把 tracking benefit 與 runtime cost 對在一起，而不是僅看 kernel latency。

* [ ] **P6 — Production performance bottleneck closure**

  整合 P1–P5 的量測，建立 production pipeline 的瓶頸排序與 scaling model，判斷下一步應優先處理 fixed-capacity computation、同步／排程、memory traffic，還是實際 detector／tracker compute。只有這一步完成後才決定是否開 optimization 工作項。

## Boundary

本輪只做 measurement / attribution，不直接改演算法、不以 profiling instrumentation 下的 FPS 取代 production throughput，也不因單一 kernel utilization 低就直接提出最佳化方案。
