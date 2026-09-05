# Saccade frame budget — where the 3.02 ms goes (2026-09-05)

> **本文回答的問題**:headline 單流 MOT path 的每幀時間花在哪裡,以及**哪些數字這套工具
> 根本量不出來**。後者是本文的主要產出:production 小 kernel 的 exact L2 hit rate 與
> per-frame DRAM ledger 在這張卡的 counter interface 下**不可識別**,不是尚未量。
>
> 全部數字出自 2026-09-05、`mamba_whole_graph_m` preset、MOT17 train / SDP、
> RTX 5070 Ti Laptop GPU、main `1d620127`(clean)。未變更任何 driver、顯示設定、
> 主機設定或 repo 程式碼。
>
> ⚠️ **證據封存在本機,不在版控**:`~/.local/state/saccade/perf/l2-source-attribution-20260905/`
> 與 `l2-investigation-20260905/`(raw CSV、`.ncu-rep`、sweep 結果)。它們的價值是**可審計**,
> 不等於需要長期版本化。結論層在本文;重跑驗證用
> [verify_l2_evidence.py](../../../scripts/eval/diagnostics/verify_l2_evidence.py)。
>
> ⚠️ 本文尚未 promote 到 `evidence_ledger` 或 `report_data`,scope 限本機與本 preset。

A per-kernel `nsys` profile of the headline single-stream MOT path. The pipeline is no
longer host-bound — the GPU is busy 93.6% of every frame — and three of the remaining
gaps are structural, not tuning.

| | |
|:--|:--|
| preset | `mamba_whole_graph_m` |
| data | MOT17 train / SDP |
| gpu | RTX 5070 Ti Laptop · GB205 · 46 SM · 36 MiB L2 · 672 GB/s peak · 12 GiB |
| driver | 610.57 · WSL2 |
| date | 2026-09-05 |

| Throughput | Frame period | GPU busy | SM clock |
|--:|--:|--:|--:|
| **349.64 fps** | **3.022 ms** | **93.6 %** | **2497 MHz** |
| 7-seq SDP, IDF1 80.3 | latency 5.71 ms (double-buffered) | union across 11 streams | boost cap 3090 · 135 W of 140 W |

Steady-state window: MOT17-04-SDP frames 100–400. Trace cost −3.7% throughput, so the
window is representative.

---

## §1 Protocol

Baseline is the documented headline command, run clean:

```bash
scripts/eval/mot17.py --preset mamba_whole_graph_m --detector SDP --double-buffer
```

All seven SDP sequences, 4 966 frames. The trace is a separate 400-frame run over
MOT17-04-SDP under `nsys --trace=cuda --cuda-graph-trace=node`. Host-side stage times
come from a third run with `--profile-frame-csv`, which inserts no syncs.

> **Why node granularity is not optional.** nsys defaults to `--cuda-graph-trace=graph`,
> which collapses each of the four captured CUDA graphs — TRT detect, tracker update,
> main NMS, GMC — into a single opaque node. At that setting the trace shows 16 014
> kernels and a GPU that looks 94% *idle*. At node granularity it shows 149 204 kernels
> and a GPU that is 93.6% *busy*. Every number below is from the node-granularity trace.

### Per-sequence latency

| Sequence | Frames | FPS | mean ms | p95 | p99 | σ |
|:--|--:|--:|--:|--:|--:|--:|
| MOT17-02-SDP | 550 | 344.6 | 5.79 | 6.82 | 7.88 | 0.61 |
| MOT17-04-SDP *(traced)* | 1000 | 347.5 | 5.74 | 6.43 | 7.95 | 0.47 |
| MOT17-05-SDP *(640×480)* | 787 | 388.8 | 5.13 | 5.88 | 6.32 | 0.44 |
| MOT17-09-SDP | 475 | 348.2 | 5.73 | 6.63 | 6.93 | 0.49 |
| MOT17-10-SDP | 604 | 344.0 | 5.80 | 6.76 | 7.41 | 0.53 |
| MOT17-11-SDP | 850 | 347.6 | 5.74 | 6.59 | 7.32 | 0.52 |
| MOT17-13-SDP *(heaviest camera motion)* | 700 | 327.2 | 6.10 | 8.38 | 10.98 | 1.11 |

Spread across 1080p sequences is 327–348 FPS; MOT17-13's p99 of 10.98 ms is the only
real tail, and it tracks camera motion (GMC and association both grow).

`docs/reference/mot17_default_config.md` still records ~241 FPS for this preset — that
row predates threaded decode and is stale.

---

## §2 One frame, eleven streams

Every GPU event inside a single 2 975.5 µs frame, grouped by the stream it ran on.
Double-buffering means detect for frame *N+1* is already running while the tracker
finishes frame *N*, so lanes overlap by design — the sum of lane busy time (3.56 ms)
exceeds the frame period (3.02 ms) by 1.18×.

| Lane (CUDA stream) | Busy µs | % of frame | Dominant work |
|:--|--:|--:|:--|
| Detect — TRT backbone + Mamba head `164` | 2019.5 | 68 % | TRT 1651 µs ×219, scan 228 µs ×2 |
| Tracker association `171` | 314.6 | 11 % | 304 µs ×22 native kernels |
| NMS — private-candidate pass `7` | 245.6 | 8 % | NMS 127 µs ×2, memcpy 81 µs ×20 |
| Frame ingest — cast, scale, 24.9 MB D2D `24` | 206.3 | 7 % | 105 µs elementwise + 101 µs memcpy ×2 |
| Detect side stream (double-buffer) `166` | 178.4 | 6 % | TRT 122 µs ×21, scan 56 µs ×1 |
| NMS — main pass `169` | 165.2 | 6 % | NMS 137 µs ×2 |
| GMC (cuFFT) `170` | 54.3 | 2 % | 51 µs ×8 |
| nvJPEG decode `160` | 46.6 | 2 % | 29 µs decode + 17 µs H2D |
| TRT tail streams `165/168/167` | 46.4 | 2 % | TRT 46 µs ×16 |

388 kernel and memcpy events in the frame. The detect lane is saturated; everything else
is filling gaps around it.

### Host stage ledger

MOT17-04-SDP, 1 000 frames, warmup excluded, `--profile-frame-csv` (changes no scheduling).

| Host stage | mean ms | median | max | What it is |
|:--|--:|--:|--:|:--|
| `total_ms` | 5.925 | 5.900 | 9.083 | End-to-end frame latency |
| `post_graph_count_wait_ms` | **1.884** | 1.912 | 5.309 | **Blocking sync on GPU — not work** |
| ├ `post_pre_nms_ms` | 1.243 | 1.291 | 4.943 | Waiting for detect to land |
| └ `post_finalize_ms` | 0.642 | 0.585 | 3.440 | Filter, quality gate, private continuation |
| `track_ms` | 0.313 | 0.258 | 1.172 | Tracker update + materialize |
| `gmc_ms` | 0.041 | 0.032 | 0.283 | Global motion compensation |
| `fetch_ms` | 0.018 | 0.016 | 0.243 | Frame fetch |
| `detect_ms` | 0.004 | 0.004 | 0.050 | Launch only; the wait moved to `pre_nms` |
| `output_ms` | 0.003 | 0.003 | 0.025 | MOT-format emit |

The CPU is not the bottleneck; it is parked on `cudaStreamSynchronize`.

---

## §3 Compute budget

Kernel time summed across all streams, divided by 300 frames. Because the streams
overlap, these add to 118% of the frame period rather than 100% — read them as *cost*,
not as a partition of wall time.

| Group | µs/frame | % period | launches/frame |
|:--|--:|--:|--:|
| TRT backbone + Mamba head (conv / gemm / myelin) | **2121.6** | 70.2 % | 267.2 |
| Tracker association (occlusion / sinkhorn / auction / kalman) | 386.5 | 12.8 % | 22.1 |
| Mamba `selective_scan` (fp16, one per FPN level) | 323.6 | 10.7 % | 3.0 |
| NMS — bitmask + select, **two passes per frame** | 313.6 | 10.4 % | 4.0 |
| torch elementwise / copy / topk | 262.6 | 8.7 % | 41.9 |
| memcpy engine | 208.2 | 6.9 % | 30.1 |
| GMC (grayscale / cuFFT / subpixel peak) | 62.5 | 2.1 % | 8.0 |
| Other native post (private continuation / gather / pad) | 45.2 | 1.5 % | 17.1 |
| nvJPEG decode (dedicated NVJPG engine) | 29.2 | 1.0 % | 1.0 |
| *frame period* | *3022* | *100 %* | |

TRT at 70% is healthy — that is the model doing its job. The attackable surface is the
other 30%, and three items in it are structural waste rather than tuning headroom.

---

## §4 Memory and bandwidth

PCIe is not in play. Per frame the host sends 0.23 MB (JPEG bytes) and reads back
0.07 MB (~7 KB detection results, nine times).

| Direction | ops/frame | MB/frame | GPU µs/frame | effective |
|:--|--:|--:|--:|--:|
| **Device → Device** | 20.1 | **75.68** | 181 | 417.7 GB/s |
| Host → Device | 1.0 | 0.23 | 17 | 13.5 GB/s |
| Device → Host | 9.1 | 0.07 | 10 | 6.8 GB/s |

The D2D total decomposes almost entirely into one size class:

```
size 24300.0 KiB × 3.02 per frame = 75.06 MB/frame   ← 99.2% of all D2D
```

24 300 KiB = 1920×1080×3×4 B — `pool.frame_buffer`, the full-resolution fp32 CHW frame
(`src/saccade/perception/eval/pool.py:69`).

### D2D copy bandwidth versus working set (measured on this GPU)

| Buffer | read+write | Note |
|--:|--:|:--|
| 1 MB | 228.3 GB/s | launch-bound |
| 4 MB | 601.9 GB/s | |
| 8 MB | 1264.2 GB/s | L2-resident |
| 16 MB | **1826.3 GB/s** | L2-resident (src+dst = 32 MB < 36 MB) |
| **24.4 MB** | **700.3 GB/s** | ← `pool.frame_buffer`; src+dst = 48.7 MB, spills L2 |
| 32 MB | 576.5 GB/s | DRAM |
| 48 MB | 577.1 GB/s | DRAM |
| 96 MB | 572.8 GB/s | DRAM |
| 256 MB | 576.9 GB/s | DRAM — 86% of the 672 GB/s pin peak |

Inside L2 the fabric delivers 1.83 TB/s; past it, 575 GB/s. `frame_buffer` lands just
past the edge. Memory-controller utilization during the real run sits at 42–49%, so the
pipeline is **not** globally bandwidth-bound — but this one chain is.

### VRAM is not a constraint here

Peak device memory over the whole run is **1 942 MiB of 12 227 MiB**, and that figure
includes roughly 1.1 GiB already resident before the process started. Single-stream, the
`ResourceManager` degradation ladder (85 / 92 / 96%) can never trigger.

Power is the real ceiling instead: **135 W median against a 140 W cap**, holding the SM
clock at 2 497 MHz against a 3 090 MHz boost.

---

## §5 Three structural findings

Ranked by cost per frame. All three are shape problems — a launch geometry, a duplicated
pass, a temporary — rather than parameters to tune.

### F1 — The NMS select pass is a single thread · 251.9 µs/frame (8.3%) · ×2 per frame

Launch geometry read straight off the trace is `grid=(1,1,1) block=(1,1,1)` over 801
launches, averaging 121.4 µs each. One CUDA thread runs a sequential greedy select on a
46-SM GPU. It is the largest non-TRT kernel in the pipeline.

```cpp
// src/tracking/tracker_gpu.cu:6235
nms_select_counted_kernel<<<1, 1, 0, stream>>>(...)

// :5889
if (blockIdx.x != 0 || threadIdx.x != 0) return;
for (int order_pos = 0; order_pos < valid_count; ++order_pos) { ... }
```

Measured: 604 launches in the 300-frame window · 75.86 ms total · 125.60 µs mean.

### F1b — The second NMS pass is private-continuation · ~157 µs/frame

Half of the NMS bill is a whole extra bitmask + select round, gated on
`private_candidate_nms_iou (0.70) > nms_threshold` — which the preset satisfies.
Whatever is done about F1's geometry, it is paid twice until this gate changes.

```cpp
// src/tracking/pipeline.cpp:443
&& cfg_.private_candidate_nms_iou > cfg_.nms_threshold
// :447
auto launch_private_candidate_nms = [&] { ... }
```

### F2 — Occlusion scores a 2 048-slot table with 2 048 threads · 169.0 µs/frame (5.6%) · 0.03 waves/SM (ncu)

The launch is `grid=(8,1,1) block=(256,1,1)` — 2 048 threads in only eight blocks
for 46 SMs. Inactive outer slots return early; each active track scans the 2 048-slot
`max_objs` table. The earlier 4.2 M iteration estimate incorrectly included inactive
outer slots, and **42.9 detections/frame** is not an active-track count. The small-grid
and work-imbalance diagnosis is confirmed by ncu in §6.

```cpp
// src/tracking/tracker_gpu.cu:3515
kernel::compute_track_occlusion_kernel<<<blocks, threads, 0, stream>>>(...)

// :319
int t = blockIdx.x * blockDim.x + threadIdx.x;
if (!active[t]) { /* reset occlusion outputs */ return; }
for (int j = 0; j < max_objs; ++j) { if (j == t || !active[j]) continue; ... }
```

Careful here — `occ_partner`, `occ_partner_all` and `occ_duration` all carry semantics
into association. A geometry or compaction rewrite must preserve inactive-slot resets
and partner/duration semantics; it still needs correctness and metric A/B checks.

### F3 — Frame ingest materializes fp32 temporaries · ~147 µs isolated saving in the original probe · bit-exact verified

One expression produces two fp32 temporaries (cast and scaled output), then copies
into the pool buffer. The original trace reports ~75 MB/frame of large D2D copies
across the pipeline; attributing all three copies to this one expression is not
established. The isolated counter experiment and fresh timing comparison are in §6.

```python
# src/saccade/perception/eval/stages.py:1843
pool.frame_buffer.copy_(frame_gpu.permute(2, 0, 1).float() / 255.0)
#                       cast temporary      scaled temporary    final buffer copy
```

> **Call site correction (2026-09-05).** This section previously cited
> `src/saccade/perception/eval/evaluator.py:425`. That copy of the expression sits inside
> `if getattr(cfg, "workbench", False) and wb is not None:` (`evaluator.py:414`), and
> `workbench` defaults to `False` (`config.py:1280`); `scripts/eval/mot17.py:125` also
> rejects `--workbench` together with `--private-continuation`, which this preset enables
> (`private_continuation_enabled: true`). The profiled headline command therefore never
> executes that line. The live ingest for this preset is `stages.py:1843`, inside
> `_run_detect` (`stages.py:1809`). The expression text is identical at both sites, so the
> measurements, the bit-exactness checks and the §6 probe are unaffected — only the
> citation was wrong.

Original probe, as supplied: measured on this GPU at 1920×1080, and checked for bit-exactness against the current
expression over all 256 `uint8` values and on a random 1080p tensor — `torch.equal` is
true for both rewrites:

| Form | µs | Δ |
|:--|--:|--:|
| `buf.copy_(src.permute(2,0,1).float() / 255.0)` *(current)* | 191.1 | — |
| `buf.copy_(src.permute(2,0,1)); buf.div_(255.0)` | 68.0 | −123.1 |
| **`torch.div(src.permute(2,0,1), 255.0, out=buf)`** | **44.1** | **−147.0** |

### What the total is worth — and what it isn't

F1 + F2 + F3 come to roughly **570 µs**, 19% of the 3 022 µs frame period. That is a
bound on *GPU time removed*, not on FPS gained: these kernels sit on streams that
overlap, so wall-clock recovery is strictly less than the sum. The GPU is 93.6% busy, so
conversion should be decent — but it needs a paired A/B, not an estimate.

F3 is the only one of the three that is a pure rewrite with proven bit-exactness; F1 and
F2 change kernel behaviour and need the usual metric check.

---

## §6 Nsight Compute — counters available; geometry confirmed

**2026-09-05 follow-up:** the earlier `ERR_NVGPUCTRPERM` block no longer reproduces.
The pipeline now collects hardware counters without sudo. The retained run used
**ncu 2026.2.1.0**, repository HEAD `1d620127e06cb713b0b4e266316a95b767a3d3ea`
(clean), CUDA user-mode driver **610.57.04**, Windows KMD **616.56**, and the same
RTX 5070 Ti Laptop GPU. Present access is verified; which host setting changed is unknown.

### Protocol and sampling limits

Four separate 160-frame MOT17-04-SDP runs retain the preset, compilation, CUDA graphs
and double-buffering. Each collects six matching launches. Skip counts are **kernel
matches, not frames**: NMS 200, occlusion 100, scan 300, Sinkhorn 100. Initialization
and graph warmup can consume matches, so these are targeted launch samples, not the
original frames 100–400 window. NMS combines the main/private passes; scan mixes FPN
sequence lengths. This is not an exhaustive profile of the TRT backbone.

Explicit controls: `--graph-profiling node --cache-control all --clock-control none`,
default kernel replay, eight collection passes per sampled launch. Clocks remain
dynamic; cache flushing and serialization differ from normal execution. The durations
below are **ncu replay durations**, not replacements for §3's nsys µs/frame or evidence
of a production regression. The profiled runs' FPS is not usable as production FPS.

### Launch geometry and achieved occupancy

Arithmetic means over six launches; time ranges show min–max. SM throughput uses
`sm__throughput.avg.pct_of_peak_sustained_elapsed`. The two warp columns use
`sm__warps_active.avg.pct_of_peak_sustained_{active,elapsed}`, respectively.

| Kernel | Grid × block | ncu µs/launch, mean [range] | SM throughput % | Active-cycle occupancy % | Elapsed-cycle warp utilization % | Waves/SM |
|:--|:--|--:|--:|--:|--:|--:|
| NMS select | 1 × 1 | 211.05 [194.62–223.55] | 0.03 | 2.08 | 0.048 | <0.001¹ |
| Occlusion | 8 × 256 | 265.74 [254.08–280.22] | 0.11 | 4.49 | 0.10 | 0.03 |
| Selective scan, fp16 | (4,64,1) × 64 | 283.41 [65.31–631.84] | 18.24 | 23.06 | 22.35 | 0.23 |
| Fused Sinkhorn | 2,048 × 128 | 128.10 [117.82–139.49] | 92.73 | 46.63 | 45.31 | 7.42 |

¹ ncu rounds NMS waves/SM to `0.00`; `1 / (46 × 24)` is approximately 0.00091,
using the reported resident-block limit. A resident warp does not imply all 32 lanes
are useful: NMS has **one thread**, so its 2.08% active occupancy (about 1/48 warps)
does not mean 2.08% useful work across the GPU.

**F1 and F2's geometry diagnosis is supported.** Occlusion's theoretical occupancy
is 100%, yet achieved occupancy is 4.49%. Only eight SMs can receive its eight blocks,
and inactive tracks return early, further concentrating the remaining work. The earlier
0.17 was `8/46` **blocks/SM**, not waves. With six resident blocks/SM, ncu's waves are
`8/(46×6) ≈ 0.029`. The device reports **1,536 resident threads/SM**, or 70,656 across
46 SMs, rather than 94,000. This is a residency capacity, not a minimum for performance.

The source also rules out unconditional 4.2 M inner iterations: inactive outer slots
return before the loop. Slot visits scale with **active tracks × 2,048**; the reported
42.9 detections/frame is not an active-track count. Partner, duration and inactive-slot
reset semantics still need preservation in any rewrite.

Sinkhorn's sampled SM throughput is high despite moderate occupancy; raising occupancy
alone is not an optimization criterion. Scan has a small number of waves, but these
counters do not prove which instruction or dependency limits it.

### L2 and DRAM: raw readings are not an exact traffic ledger

| Kernel | Raw L2 hit rate % | Raw L1 hit rate %, mean | Raw DRAM read MB/launch [min–max] | Interpretation |
|:--|:--|--:|--:|:--|
| NMS select | **invalid**, maximum 4,842.34 | 51.94 | 0.012–6.668 | No usable L2 estimate |
| Occlusion | 1.11–90.26 | 79.24 | 0.025–4.924 | Too variable for a residency claim |
| Selective scan | 35.54–85.51 | 63.25 | 0.126–3.467 | Mixed lengths; cache-flushed replay |
| Fused Sinkhorn | 96.55–97.58 | 91.84 | 0.047–4.867 | High replay L2 hit rate; DRAM attribution remains uncertain |

MB here means 1,000,000 bytes. Do not clamp invalid hit rates or average them into a
physical cache claim. A narrower **L2-only** retry still needed three passes and returned
**18.67–520.08% for NMS**; occlusion returned **1.03–63.86%**. Reducing the counter set
did not resolve the attribution problem. Zero DRAM write counts in some rows do not
mean the kernel performed no stores; stores may remain cached beyond the measured interval.

NVIDIA documents interference from display/copy/video engines in shared L2/DRAM counters,
and out-of-range ratios from multi-pass collection. This GPU drives a display and the
pipeline decodes/copies asynchronously. Those are plausible contributors, not an isolated
cause established by these runs. See [NVIDIA's profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#range-and-precision).

### L2 anomaly follow-up — source scope and replay consistency

A controlled follow-up reproduced the anomaly using the **unchanged NMS kernel body**
in a standalone CUDA harness, both with and without a CUDA graph. One ordinary launch
reported **263,138 hits / 18,843 total sectors = 1,396.48%**. The CSV arithmetic matches
ncu; the underlying multi-pass counters are inconsistent. Saccade decode/tracker
concurrency is not required to reproduce it.

The interfering traffic has since been identified by source and excluded; the
paragraphs below supersede the earlier "responsible Windows engine/process has not been
identified" and the L1TEX-based recommendation that followed it.

**Source.** The L2 total partitions exactly by source node — `srcnode_gpc + srcnode_hub
+ srcnode_fbp`, residual zero in 24/24 single-pass rows — and every anomalous total is a
large `srcnode_hub` term, i.e. L2 clients that are not the SM. Sweeping the measurement
window over 250x (168 us to 42.3 ms) with an idle kernel gives a background rate of
**85.25 sectors/us** by least squares, against **84.48** predicted by one 2560x1600x32bpp
framebuffer per 165 Hz display refresh — a ratio of **1.009**. Over the same sweep the
kernel side (`srcnode_gpc`) is fixed at 320 sectors. In a separate 60-launch sample,
11 of the non-zero hub values fall within 1.0006-1.0054 framebuffers. The NVIDIA GPU
holds the only active display mode on this host; WSL's `nvidia-smi` cannot see the 15
Windows C+G processes that the host-side `nvidia-smi.exe` lists. This is a rate-and-size
match plus demonstrated independence from kernel work, not a timestamped capture, and
scanout is not owned by a named process. A display-mode A/B would settle it: 60 Hz
predicts 30.7 sectors/us, 1920x1080 predicts 259,200-sector bursts. Neither was run.

**Exclusion.** Scope to `srcnode_gpc` **and** collect in one pass; both are required.
Gpc-scoped *standard* counters still fail `total = hit + miss` in 11 of 12 rows and
still exceed 100% (80.13-112.53%), so multi-pass replay is inconsistent for a second
reason that has nothing to do with other engines. Gpc-scoped *realtime* counters
conserve 12/12 with a 0.37 pp spread. Nothing on the Windows host needs to change.

**But gpc is not the data path.** It includes instruction/constant fetch, whose share of
gpc is 4.3% for the isolated NMS probe, 31.7% for pipeline NMS and **62.7%** for pipeline
occlusion, and `gpc - tex - gcc` leaves a further unnamed 440-1,232 sectors. A gpc-scoped
hit rate answers "what this launch does to L2", not "does this kernel's data stay
resident".

| Pipeline kernel | L1TEX standard, 4 passes | L1TEX realtime, 1 pass | gpc realtime, 1 pass | tex realtime, 1 pass |
|:--|--:|--:|--:|--:|
| NMS select | 85.57-86.07% | 100.00% | 95.86-96.97% | 95.52-100.00%, median 100.00 |
| Occlusion | 5.70-6.35% | 25.51-28.28% | 70.23-71.13% | 22.68-26.26%, median 25.25 |

**The pipeline rows remain unresolved.** Four methods give four answers, and the data
path moves only **992 sectors** (NMS) and **792** (occlusion) per launch — the size at
which the documented realtime quantization reproduces exactly, pipeline NMS returning a
median 100.00% with zero misses. Conservation passing is not evidence against
quantization: `hit == total, miss == 0` satisfies it trivially. Do not pick the method
with the tightest spread. The only defensible cache figure from this work is the
**isolated** NMS probe at 1,024 slots, where the data path moves 16,976 sectors:
**86.60%** (`srcunit_tex` realtime) or 85.49-85.86% (`srcnode_gpc` realtime) — a
standalone harness with the production kernel body, not the production binary.

**DRAM cannot be repaired this way.** `dram__` metrics have no `srcnode` or `srcunit`
dimension and no `_realtime` variant on this chip, so every DRAM figure is chip-global
and multi-pass by construction. The DRAM columns above stay unusable.

Two collection traps worth carrying: `ncu` accepts a **misspelled metric name without
any error**, silently omitting the column and corrupting any subtraction built on it;
and adding metrics silently raises the pass count, so read it back out of the log.

The source attribution evidence pack (`~/.local/state/saccade/perf/l2-source-attribution-20260905/`,
**local retained evidence, not a repository artefact**) holds the sweep, the negative
controls and the raw CSVs. Re-verify it with
[verify_l2_evidence.py](../../../scripts/eval/diagnostics/verify_l2_evidence.py), which
recomputes every number above and passes 26/26 checks. The predecessor investigation
(`~/.local/state/saccade/perf/l2-investigation-20260905/`) retains the original reports
and precision controls.
NVIDIA documents both [multi-pass/shared-engine interference](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#range-and-precision)
and [realtime counter precision](https://docs.nvidia.com/cuda/developer-preview/13.4/nsight-compute/ComputeTriage/index.html#current-limitations-to-account-for-in-triage).

### F3: isolate the whole conversion, including its final copy

`ingest_probe.py` obtains the CUDA JPEG decoder's actual layout, then fills it with
seeded random uint8 values. The HWC view has stride **(1920, 1, 2073600)**; converting
back to CHW exposes contiguous planes. The destination is contiguous fp32 CHW.
All forms pass `torch.equal` on the random 1080p input, and both rewrites pass the
exhaustive 256-value uint8 check on **PyTorch 2.11.0+cu130**.

One whole conversion is enclosed by `cudaProfilerStart/Stop` after 100 warmups.
Range replay includes the current expression's final copy, which a kernel-name filter
alone would omit. Each cell below is one range result, collected in four passes;
`all` and `none` are separate runs. They are sensitivity probes, not confidence intervals.

| Form | Cache control | DRAM read MB | DRAM write MB | L2 hit % |
|:--|:--|--:|--:|--:|
| Current | all | 15.222 | 55.401 | 29.09 |
| Current | none | 16.504 | 67.891 | 28.83 |
| Copy + in-place divide | all | 6.308 | 33.865 | 58.91 |
| Copy + in-place divide | none | 5.179 | 33.777 | 87.40 |
| Fused `torch.div(..., out=buf)` | all | 6.252 | 10.087 | 0.41 |
| Fused `torch.div(..., out=buf)` | none | 1.439 | 22.598 | 73.03 |

The fused form reduces observed DRAM traffic in both cache modes, supporting removal
of intermediate materialization. Its near-zero cold-cache L2 hit rate is not a failure:
a single pass with little reuse can move fewer bytes while achieving fewer cache hits.
These are measured memory-controller bytes, not logical tensor read/write sizes; cached
writes, replay state and other engines prevent treating them as exact application bytes.
The three 24.9 MB D2D copies in the original pipeline trace are not independently
attributed to this single expression by this experiment.

A separate **unprofiled**, shuffled-order CUDA-event comparison used nine paired
trials, 200 operations/form/trial after warmup. Median operation times were
**196.40 µs current, 84.47 µs in-place, 49.74 µs fused**. The fused saving was
**144.73 µs median paired difference** (range 136.39–148.24 µs).
This supports the scale of the original ~147 µs isolated saving; it is not a paired
production pipeline run and establishes no FPS or tracking-metric improvement.
`compare_ingest.py` and `paired-ingest.json` retain the trial order and all measurements.

### Reproduce and retain evidence

Run from `/home/ray/developer/ai/saccade`. The executable scripts, exact argv JSON,
logs, `.ncu-rep`, base-unit CSV, `summary.json`, timing results and environment snapshot
are retained in **`/home/ray/.local/state/saccade/perf/ncu-20260905/`**. Commands below
create new reports; choose unused export paths when repeating.

```bash
# Query metric names supported by the installed ncu/device.
ncu --query-metrics

# Example family sample; the collector repeats this with each family's filter/skip.
timeout 180s ncu --graph-profiling node --replay-mode kernel \
  --clock-control none --cache-control all --kernel-name-base demangled \
  --kernel-name 'regex:compute_track_occlusion' --launch-skip 100 --launch-count 6 \
  --section LaunchStats --section Occupancy \
  --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_elapsed \
  --export /tmp/occlusion-repeat \
  .venv/bin/python scripts/eval/mot17.py --preset mamba_whole_graph_m --detector SDP \
  --double-buffer --sequences MOT17-04-SDP --max-frames 160 --latency-only \
  --output /tmp/occlusion-repeat-out

ncu --import /tmp/occlusion-repeat.ncu-rep --page raw --csv --print-units base \
  > /tmp/occlusion-repeat.csv

# Whole-conversion counter probe; substitute current/inplace/fused and all/none.
timeout 90s ncu --replay-mode range --clock-control none --cache-control all \
  --metrics gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_sector_hit_rate.pct \
  --export /tmp/ingest-fused-repeat \
  .venv/bin/python /home/ray/.local/state/saccade/perf/ncu-20260905/ingest_probe.py fused --profile
```

The former `gpu__dram_throughput`, `dram__bytes_read` and `dram__bytes_write` names
were replaced with this version's `dram__throughput`, `dram__bytes_op_read` and
`dram__bytes_op_write`. Use demangled names to match template functors such as
`direct_copy_kernel_cuda`, which can be hidden by the default function base-name filter.
Check logs for actual profiled launches and verify CSV row counts: a zero exit code
with “No kernels were profiled” is not a successful collection. See the
[NVIDIA CLI reference](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#command-line-options)
for skip, filtering, replay and export options.

The permission gap is closed, launch underutilization is measured, and the isolated
F3 traffic reduction is supported. **Production L2 hit rates, an exact per-frame DRAM
ledger and production FPS gains remain unestablished.**

---

Sections 1–5 retain the supplied nsys/host/microbenchmark figures, except for the
explicit F2/F3 corrections linked to this follow-up. They were restored from the text
read in this conversation after `/tmp` was cleared; the original nsys-rep, SQLite,
frame ledger and bandwidth benchmark were not available for revalidation. The fresh
ncu results above have retained raw artefacts and do not depend on the lost first attempt.

Claims remain scoped to this host and preset. Nothing has been promoted to
`evidence_ledger` or `report_data`; the attribution is exploratory and unreviewed.
