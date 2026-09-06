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

> **Superseded attribution (2026-09-06).** These are three *different* copies with three
> different owners, not one expression repeated: the ingest expression, the detector
> CUDA graph's static input, and the GMC CUDA graph's static input. §7 separates them by
> stream, records which one F3a removed, and measures what removing a second one was
> worth.

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

> **Status (2026-09-06).** F3 was split into F3a (this expression), F3c (the detector
> CUDA graph's static input) and F3b (the representation change). **F3a is merged**
> (#339, paired +2.15%). **F3c was implemented and measured but not landed** (#338,
> closed as not planned): the copy provably disappears and whole-pipeline D2D volume
> halves, with no detectable throughput change, which did not justify the ownership
> protocol it required. §7 has the numbers. F3d (#341) is the same defect at the GMC
> graph's input and is **not** ranked by its byte count — see §7. **F3b is closed by
> structural disposition** (2026-09-06) — its premise fails, not its payoff; see §5.1.

#### §5.1 F3b — closed, structural premise failure (2026-09-06)

F3b was "replace the ingest representation so the full-resolution fp32 RGB frame is
never materialized". The gating question was never its payoff; it was whether the
materialization can be removed at all. It cannot.

**`pool.frame_buffer` has two consumers on the headline active path** (`reid_mode: off`,
`tiling: native_640`, `preprocess: none`, `gmc: true`):

| consumer | site | full-res fp32 required? |
|:--|:--|:--|
| detector | `detection.py:1043` — `detect_raw(pool.frame_buffer.unsqueeze(0))` | no — only as the resize source |
| GMC | `stages.py:3317` → `_gmc_frame_buf.copy_(_frame_gmc)` (`stages.py:293`/`:322`/`:323`), buffer `(3, h_orig, w_orig)` fp32 at `pipeline.py:1094` | **yes, unconditionally** |

The four other `as_rgb_chw()` sites (`stages.py:2077`/`:2233`/`:3127`/`:3229`) are ReID
crop paths, dead with `reid_mode: off`; the `evaluator.py` sites are the workbench
runner, which this preset never enters (see the call-site correction above).

GMC is unconditional in the preset, so any candidate can remove the **detector's read**
of the buffer, never the buffer. That reduces the opportunity to:

| item (1920×1080, per frame) | bytes | removable |
|:--|--:|:--|
| ingest write of `pool.frame_buffer` | 24.88 MB W | no — GMC |
| detector interpolate read (640·640·3 × 4 taps × 4 B) | 19.66 MB R | yes → 4.92 MB uint8 |
| GMC read of `pool.frame_buffer` | 24.88 MB R | no |
| GMC write of `_gmc_frame_buf` | 24.88 MB W | no |
| 640 canvas write | 4.92 MB W | no — unchanged either way |

**≈ −14.7 MB of ≈ 94 MB**, and that number is a *logical / raw-cost reduction bound*,
not a DRAM saving: cache reuse and locality can make the realised gain smaller. At
640×480 the source buffer is 3.69 MB and largely L2-resident, so the stratum-level
saving there is near zero. Separately, `frame_gpu` is confined to the ingest stage
(`stages.py:1874`/`:1879`); carrying it to a fused preprocessing site is refcount-safe
for the torchvision decoder but not guaranteed for DALI, whose iterator returns tensors
backed by its own output buffers under `prefetch_queue_depth=2` — keeping it safely may
cost an extra 6.22 MB uint8 clone, a third of the remaining opportunity.

A new fused kernel is not justified on an opportunity of that size and shape.

**Not the reason for closure.** The rejected NV12 candidate's semantic delta was measured
first, at the tensor `backbone.infer_graph` receives, 28 paired frames across the 7-seq
(8-bit levels): MAE 3.61, p99 38.1, 20.8% of pixels beyond 4 levels, 5.1% beyond 16.
Controls separate the mechanism cleanly — constant/low-frequency colour patches give
MAE ≤ 0.44 (max 2.2), so the NV12 4:2:0 + BT.601 + quantization round-trip is nearly
inert, while achromatic synthetics at 1080p give checkerboard MAE 51–73. The delta is
spatial resampling, plus a systematic sampling-phase offset (constant −1.000 source px
on the 1080p x axis, mean −0.812 px on y). That evidence stands as **rejected-candidate
evidence**: it shows the orphaned `nv12_to_chw_resize` kernel must not simply be wired
up as-is, and it is not what closed F3b.

**Closure characterization (not a gate).** A dose ladder was run at the detector's
in-graph resize site (a temporary probe patch, not landed: k extra copies of the same
`F.interpolate(frame, (640,640), bilinear, align_corners=False)` inserted immediately
before the real one inside `_whole_graph_fn`, so the dose is captured into the same graph;
k ∈ {0,1,2,4} × 3 mirrored reps,
7-seq, metrics identical across all 12 runs). It **does not resolve**: per-rep signs at
k=1 are (−,+,+), the k=0 anchor spread across reps is 0.192 ms against a largest mean
dose effect of 0.054 ms, and the 640×480 stratum is negative at every dose. Per the F3d
reading protocol this means no breakpoint may be named. What it supports is an upper
bound only — one detector-side resize costs less than the ≈0.05 ms paired spread, i.e.
under ~1.6% of the 3.1 ms frame period, with a point estimate of +0.007 ms (+0.2%) that
is not separable from zero. Note the contrast with F3d, whose GMC site was 3/3 positive
at k=1 (S < 1, no slack); under the same pre-registered falsifier this site is
*suggestive of slack*, but n=3 with inconsistent signs does not establish it.

**Recorded, not opened.** GMC's first operation on the frame is `launch_grayscale_downscale`
(`gmc.cpp:11`) → a single-channel `W/4 × H/4` buffer, and the NV12 configuration feeds it
`_luma.repeat(3, 1, 1)` (`stages.py:3314`) — three identical channels, so GMC extracts no
independent per-channel information. Its information requirement at 1080p is 0.52 MB
against ~74.6 MB moved to deliver it. This is a **candidate mechanism only**: F3d already
bounds removing one such copy, and GMC has a measured input-representation sensitivity of
about 0.5 pp IDF1 (`pool.get_frame_luma()`), so it would be an independent admission
problem, not a free optimization.

**Defects found while evaluating F3b, tracked separately from it.**

1. `src/perception/letterbox_kernel.cu:6` and the bindings at `tracker_gpu_python.cpp:5627`
   and `:5646` all describe the kernel as "bilinear resize"; it is nearest-neighbour
   (`letterbox_kernel.cu:32`).
2. `_prepare_canvas_960p` and `detect_single_patch_640`'s letterbox branch select nearest
   vs bilinear on `if cpp_letterbox_gpu is not None` — the same preset yields different
   detector input depending on build state, with nothing recorded.
3. `src/perception/nv12_kernel.cu` and `src/perception/rgb_to_nv12_kernel.cu` have been
   present since the initial commit (`f6d6dd59`), are absent from `CMakeLists.txt`, and are
   bound by no extension. `_import_nv12_ops()` swallows the `ImportError` and substitutes
   Python fallbacks with different semantics and no performance meaning, so
   `SACCADE_NV12_BUFFER=1` has never enabled the fast path its name implies. This is a
   feature-availability / provenance-correctness bug, not a performance item.

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

## §7 The three 24.9 MB copies, separated — and what removing one was worth (2026-09-06)

> Measured the day after §1–§6 on the same GPU but a **different driver (616.56**, not
> 610.57), with F3a (#339) already merged. Absolute throughput here is **not** comparable
> to §1's 349.64 fps; only the paired within-session comparison below is.
> Evidence: `~/.local/state/saccade/perf/f3c-20260906/` (nsys reports, per-run JSONL,
> harnesses, analysis) — local retained evidence, not a repository artefact.

§4 recorded `24 300 KiB × 3.02 per frame` as a single size class and attributed it to
`pool.frame_buffer`; §6 said explicitly that the three copies were **not** independently
attributed to the ingest expression. An nsys node-granularity trace over 400 frames of
MOT17-04-SDP separates them by stream. There are three, and they are three *different*
copies with three different owners:

| The copy | Stream | Status on `main` |
|:--|:--|:--|
| ingest `pool.frame_buffer.copy_(…)` | detect side | **removed** by F3a (#339) |
| detector CUDA-graph static input (`make_graphed_callables` copies the runtime input into the captured surface whenever the pointers differ, `torch/cuda/graphs.py:533`) | detect side `24` | **still present** — investigated as F3c (#338), not landed |
| GMC CUDA-graph static input, `_gmc_frame_buf.copy_(_frame_gmc)` (`stages.py:260`, `:283`) | post `7` | **still present** — filed as F3d (#341) |

Current `main` therefore runs **two** of these per frame, not three: 802 copies of
24 300 KiB per 400 frames, 402 on the detect stream and 400 on the post stream. (402
rather than 400 because a capture also clones the frame twice, once in
`_whole_graph_warmup` and once as the capture sample.)

### Removing one of them changed nothing measurable

An unmerged experiment — branch `perf/f3c-graph-input-ownership`, issue #338 — made the
frame pool lease the detector graph's static input surface so the producer writes it in
place. The copy provably disappears:

| Trace quantity, 400 frames | `main` | experiment |
|:--|--:|--:|
| 24.9 MB D2D on the detect side stream | 402 copies · 22.19 ms | 1 copy · 0.04 ms |
| all D2D operations | 7 626 · 20 201 MB · 60.6 ms | 7 225 · 10 223 MB · 37.4 ms |

That is **−55.4 µs/frame of copy-engine time and half the entire pipeline's
device-to-device byte volume**. The paired production A/B — `mot17.py --preset
mamba_whole_graph_m --detector SDP --double-buffer`, all 7 SDP sequences, 48 runs
interleaved as 12 ABBA blocks, quiet host — found:

| | `main` | experiment |
|:--|--:|--:|
| mean fps | 328.63 | 328.26 |
| median | 329.18 | 335.66 |
| sd | 10.79 | 18.06 |

Block-paired delta **−0.376 fps (−0.11%)**, 95% bootstrap CI **[−2.59%, +2.07%]**,
block-restricted permutation **p = 0.91**, sign test p = 0.77, 7/12 blocks positive. All
48 runs produced a single sha256 over the concatenated MOT output, with
IDF1/MOTA/HOTA/DetA/AssA/IDs/FP/FN identical in every run (80.3 / 81.8 / 74.3 / 76.1 /
72.8 / 358 / 2005 / 18053), so the null is not hiding a behaviour change.

### Raw copy cost is not critical-path cost

This is the transferable result, and it is a constraint on how the remaining items in §5
may be ranked.

Half of the pipeline's device-to-device traffic was removed, verified in the trace, with
**no detectable effect on the frame period**. That is consistent rather than
contradictory: the copy sat on the ingest lane, which is 7% of the frame period (§2) and
overlaps compute, so eliminating it frees bandwidth, not wall clock. F3a's own commit
recorded the same caveat about stream-overlapped savings; this measures it.

Three of the twelve blocks show *both* arms slowing together (run walls 29 s → 33–35 s),
a host disturbance that ABBA blocking absorbs imperfectly; it widens the interval rather
than shifting it. Resolving an effect the size of the plausible one would need roughly an
order of magnitude more runs, which was not spent.

Do not read the null as "the copy did not exist", or as "removing D2D traffic cannot
help". It is one negative result about one lane on one host. What it does establish is
that **a byte count is not an ROI estimate here**: any remaining copy of this shape,
F3d (#341) included, needs its critical-path exposure measured before it is scheduled,
not after.

### A capture race the same measurements exposed

CUDA graph capture runs in `cudaStreamCaptureModeGlobal` and races the frame streamer's
decode thread; each sequence captures four to five graphs at its start.
`streaming.py:154` documents this and mitigates it only partially — it covers
`cudaErrorStreamCaptureInvalidated` raised by `TorchvisionGpuStreamer`'s own worker, not
the `Implicit` variant, and cannot reach DALI's threads. A collision kills the sequence
and truncates the output; it is never silent.

Over full 7-sequence runs on this host, `main` fails **1 in 35** on `--no-gpu-decode`
(DALI) and **0 in 24** on the production GPU-decode path. This is tracked as **#340**,
which carries the failure shapes, the per-arm rates and the direction (capture-mode /
capture-site semantics). It is a pre-existing defect, not a consequence of anything in
this section.


---

Sections 1–5 retain the supplied nsys/host/microbenchmark figures, except for the
explicit F2/F3 corrections linked to this follow-up. They were restored from the text
read in this conversation after `/tmp` was cleared; the original nsys-rep, SQLite,
frame ledger and bandwidth benchmark were not available for revalidation. The fresh
ncu results above have retained raw artefacts and do not depend on the lost first attempt.

Claims remain scoped to this host and preset. Nothing has been promoted to
`evidence_ledger` or `report_data`; the attribution is exploratory and unreviewed.
