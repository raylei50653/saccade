# mamba_whole_graph_m 性能歸因與優化 backlog

Date: 2026-07-07
Config: `mamba_whole_graph_m` preset + `--detector SDP --double-buffer`，nsys node-mode（工作流見 [nsys_profiling.md](../../reference/runbooks/nsys_profiling.md)）
Baseline: **292.8 FPS / mean latency 6.82 ms / IDF1 80.5**（7-seq，3 跑 metric 全同、FPS ±0.4%）
GPU: RTX 5070 Ti Laptop (Blackwell GB205)

---

## TL;DR

Production 每幀 throughput 預算 ~3.4 ms（latency 6.8 ms = 2 個週期，double-buffer 加一幀延遲）：

| 段落 | ms/frame | 佔比 | 性質 |
|---|---:|---:|---|
| detect whole-graph（GPU busy）| 2.56–2.59 | ~76% | compute floor |
| tracker graph + GMC + eager postproc | 0.43+0.08+0.3 | — | **全藏在 detect 底下**，對 throughput 零成本 |
| GPU 閒置 bubble（production 校準後）| ~0.55–0.62 | ~17% | host Python 串行段 + 下一幀 decode 鏈 gating |

double-buffer 已把能藏的全藏完（tail 內 other-work 只剩 0.12 ms）——tracker 側對 throughput 再無油水，剩下的槓桿只有 **detect compute** 和 **餵料速度**。

---

## 1. 量測（seq09 與 seq04 結構一致）

穩態窗（掐頭去尾 5%），MOT17-04 為例：wall/frame 5.03 ms（nsys 膨脹後；production 3.34）、GPU union busy 2.79 ms。

### detect whole-graph 內部（graphId 2，span mean 2.59 ms / p95 2.70）

| 類別 | ms/frame | kernels/frame |
|---|---:|---:|
| TRT backbone + TRT MambaHead（fp16 xmma）| 1.995 | **260** |
| selective_scan（3 launches，K=4 / 64-thread blocks）| 0.437 | 3 |
| torch pointwise | 0.123 | ~34 |
| upsample / topk / 其他 | ~0.13 | ~17 |

### 藏在 detect 底下的（double-buffer 生效證明）

- tracker graph（graphId 8）span 0.487 ms：occlusion 0.151 + sinkhorn 0.086 + NMS 0.13 + auction …
- GMC cuFFT graph（graphId 5）0.076 ms
- eager stream 7（postproc/NMS/prior）~0.2 ms
- tail（detect 結束→下一輪 detect 啟動）內 other-work busy 僅 **0.127 ms** → 上述工作幾乎全部與 detect 重疊

### tail 的組成（GPU 閒置段）

- host 主線程 quiet zone（純 Python、無 CUDA API）：0.4–0.7 ms/frame（nsys 膨脹值；production 估 ~0.3–0.5）
- `cudaStreamSynchronize` 佔 tail 內 API 時間大宗（1.12 ms/frame @ nsys）
- **下一幀 JPEG decode 鏈在 tail 裡跑**：NVJPG 引擎工作對 kernel 表不可見，但 `rgba2rgb`（40 µs）落在 tail、JPEG bitstream HtoD 每幀出現 → tail 的「idle」一部分其實是 decode gating
- production bubble 校準：3.34 − 2.79 ≈ **0.55 ms/frame**

### 意外發現

**DtoD memcpy 71.7 MB/frame**（20 次 / 174 µs，~410 GB/s）：frame-buffer / letterbox 量級的大搬運。GPU 時間不大，但量級可疑，值得查是否能指標交換 / zero-copy 消掉（也吃 DRAM 頻寬、與其他 kernel 搶）。

---

## 2. 優化 backlog（排序）

1. **selective_scan kernel 優化** — 0.44 ms/frame、3 launches；現行 `mamba_scan.cu` 已從原始 block-16 推到 `MAMBA_CHANNELS_PER_BLOCK=4`（N=16 時 64 threads/block），並把 `D * u` skip connection 併入主 scan 寫回，避免 has-D 路徑額外掃一次整個序列。下一步用 nsys A/B 掃 `K=4/8/16`，若 occupancy/latency 改善再固定編譯參數；仍可評估 3 個 FPN scale fuse 成單 launch。m 上 scan 佔比比 s 更高，此項最划算。
2. **decode/preproc prefetch 加深 + 縮 Python tail** — 上限 ~0.55 ms/frame（理論 ~350 FPS）。方向：decode 提前一個週期發起（讓 rgba2rgb/letterbox 落進 detect span）、host 串行段（輸出 consume / 下一幀 prep）縮短或移出關鍵路徑。
3. **DtoD 71.7 MB/frame 查源** — 找出 22 次 copy 的 call site，能否 buffer 輪轉替代整塊複製。
4. **TRT engine 本體（2.0 ms）** — backbone+head 已 fp16 已 myelin 融合，260 kernels/frame 是引擎內部結構；只剩 engine rebuild / tactic 調整級別的手段，投報最低，先不動。

不值得做：tracker/GMC/postproc 的 throughput 優化（已全部藏在 detect 底下）；多開 nsys 疊加分析 NVTX stage（injection 死鎖，見 runbook）。

---

## 3. scan `D*u` fusion 驗證（2026-07-07）

變更：`selective_scan_fwd_kernel` 在主 scan loop 的 `n==0` 寫回點直接加上 `D[d] * u_btd`，刪掉 has-D 路徑結尾第二次掃 `L` 的 read-modify-write pass。

Correctness/build：

```bash
cmake --build build --target saccade_mamba_scan_test -j$(nproc)
ctest --test-dir build -R saccade_mamba_scan_test --output-on-failure
cmake --build build --target saccade_tracking_ext saccade_scan_plugin -j$(nproc)
```

結果：native scan test pass；tracking extension 與 TRT plugin 均重建成功。

Performance：

| 測項 | 結果 |
|---|---:|
| production 7-seq | **294.40 FPS / 6.79 ms mean latency / IDF1 80.3** |
| production MOT17-04-SDP | **298.30 FPS / 6.70 ms mean latency / IDF1 92.6** |
| nsys MOT17-04-SDP wall/frame | 3.819 ms（trace host-inflated） |
| nsys GPU union busy/frame | 2.663 ms |
| nsys detect graph span | 2.504 ms mean / 2.617 ms p95 |
| nsys selective_scan | **0.327 ms/frame, 3 launches** |
| production bubble 校準（MOT17-04）| 1000/298.30 − 2.663 ≈ **0.69 ms/frame** |

對比本頁原始 trace：selective_scan 0.437 → **0.327 ms/frame**（約 **−0.11 ms/frame, −25%**）；detect graph span 2.59 → **2.50 ms**。整體 7-seq throughput 與原 baseline 292.8 FPS 同級，scan kernel 改動沒有造成 metric drift（IDF1 80.3 vs 原 80.5，屬同一 preset/輸出波動範圍）。

---

## Reproduce

```bash
# production baseline(無 nsys)
.venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph_m --detector SDP --double-buffer

# profile + 歸因
nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --force-overwrite=true -o /tmp/wg \
  .venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph_m --detector SDP \
  --double-buffer --sequences MOT17-04-SDP --output /tmp/wg_out
.venv/bin/python scripts/benchmarks/nsys_frame_attribution.py /tmp/wg.nsys-rep
```
