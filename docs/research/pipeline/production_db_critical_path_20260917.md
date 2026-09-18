# Production double-buffer critical-path attribution (2026-09-17)

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-09-17 -->
<!-- doc-module: pipeline -->

Contract: [production_db_critical_path_contract.md](production_db_critical_path_contract.md).
Machine-readable derived evidence:
[production_db_critical_path_20260917.json](../../reference/benchmarks/production_db_critical_path_20260917.json).
This executes [production_pipeline_profiling_todo.md](production_pipeline_profiling_todo.md)
P1–P6. It does not re-sweep GPU routing (#419 / #431–#433).

GPU: NVIDIA GeForce RTX 5070 Ti Laptop. SM clock during the run is
`UNRESOLVED` (not sampled). Command:

```bash
uv run scripts/eval/mot17.py \
  --preset mamba_whole_graph_m --detector SDP --double-buffer
```

`--latency-only` was used for timing runs. TrackEval is post-inference and
does not change the production scheduler.

---

## Answers (evidence, not guesses)

| Question | Answer |
|:--|:--|
| What is the production DB critical path? | The detect whole-graph. Scan-anchored detect span is **2.65 ms** on MOT17-04; warmed production period is **2.88 ms** (347.83 FPS). Detector work is fully inside that span. |
| How much do `detect(N+1)` and `tracker(N)` overlap? | Substantially. Kernel sum/union = **1.26×** on MOT17-04. Tracker graph 0.38 ms busy on stream 171 while detect occupies stream 164 for 2.49 ms. NMS is almost fully hidden (0.307 ms duration, 0.018 ms exposed). |
| Which large stages are hidden? | TRT gemm/conv 1.82 ms, TRT head 0.43 ms, selective_scan 0.38 ms, most of NMS 0.29/0.31 ms, GMC FFT 0.040/0.046 ms, private append 0.019/0.020 ms. |
| Which small costs cause scheduling loss? | 1080p DtoD memcpy **48 MB/frame, 0.11 ms exposed**. Host `cudaStreamSynchronize` API is 23 µs; the production tail leftover is **~0.23 ms**, i.e. opportunity loss ≫ API duration. |
| Does the tracker have a fixed-capacity latency floor? | **Yes** for association compute. Auction 34–35 µs/frame and sinkhorn 75–77 µs/frame are identical on 11 vs 44 active tracks. Occlusion 148–165 µs is almost flat. NMS select **does** scale (39 µs @ 7 dets vs 243 µs @ 43 dets). |
| GMC exposed cost? | **~6 µs** FFT+downscale exposed on MOT17-04. Compute is hidden. Full-frame staging is 24.9 MB × ~2 DtoD on 1080p and shows up in the memcpy line, not in the GMC kernels. |
| Dead association passes? | **S2 is 常跑 + 幾乎沒工作** (0.00–0.04 assignments/frame). S1b/S1c run every frame with 0.2–1.5 assignments. S0/S1 do the real work. Per-stage counts are branch-only evidence (one run at `e03f7d81`; not reproducible from `main`, see provenance split below). Private continuation always launches; adds 0.17–3.1 boxes/frame (reproducible from `main`). |
| Next optimization target? | **Detector compute**, specifically selective_scan (0.38 ms, 3 launches, fully on the detect span). Secondary: 1080p DtoD staging if a later PR proves it *causes* the 0.11 ms exposed memcpy. Not GMC FFT, not S2 removal, not another routing sweep. |
| Headroom if that target vanished? | Scan-only upper bound ≈ **400 FPS** (`1000/(2.875−0.38)`). Entire 0.23 ms tail vanishing ≈ **378 FPS**. These are ceilings, not predictions. |

---

## Layer P — production throughput

| Run | FPS | Period | Notes |
|:--|--:|--:|:--|
| P0 cold (first process) | 324.11 | 3.09 ms | First run after idle |
| **P0 warm (owner)** | **347.83** | **2.875 ms** | Clean, no nsys, no assoc stats |
| P1 `--profile-frame-csv` | 352.52 | 2.84 ms | Allowed P-host observer; within run-to-run noise of warm P0 |
| D2 `SACCADE_ASSOC_STATS=1` | 316.59 | 3.16 ms | Diagnostic. −31 FPS vs warm P0. Do not quote as production. |
| D1 nsys MOT17-04 | 277.00 | 3.61 ms | Host-inflated. Kernel spans only. |

Warm P0 per sequence (production):

| Sequence | FPS | mean ms | p95 | p99 | n_dets_final |
|:--|--:|--:|--:|--:|--:|
| MOT17-02-SDP | 347.26 | 6.42 | 7.31 | 8.19 | 20.0 |
| MOT17-04-SDP | 352.34 | 6.38 | 7.10 | 9.05 | 42.9 |
| MOT17-05-SDP | 378.30 | 5.87 | 6.76 | 7.54 | 6.8 |
| MOT17-09-SDP | 352.02 | 6.30 | 7.09 | 7.46 | 8.3 |
| MOT17-10-SDP | 347.54 | 6.39 | 7.30 | 7.71 | 20.0 |
| MOT17-11-SDP | 349.03 | 6.36 | 7.12 | 7.70 | 9.3 |
| MOT17-13-SDP | 310.81 | 7.19 | 10.79 | 14.19 | 18.0 |

Latency is decode-to-output (DB adds one-frame delay). Throughput period is ~half of mean latency, as expected.

Double-buffer eligibility was on for every P/D run (`detect(N+1) overlaps tracker(N) on a side stream`).

---

## Layer D — unconstrained DB timeline (MOT17-04)

Scan-anchored (3 `selective_scan` launches / frame), 948 steady frames.

| Quantity | Value |
|:--|--:|
| Detect graph union | 2.65 ms |
| GPU union busy | 3.08 ms |
| Kernel-sum / union | 1.26× |
| nsys scan-to-scan period | 3.62 ms (host-inflated) |
| Production period (warm 04) | 2.84 ms |

Streams (busy ms/frame): detect 164 = 2.49; tracker 171 = 0.38; detect-side 166 = 0.22; NMS-private 7 = 0.20; NMS-main 169 = 0.18; ingest 24 = 0.07; GMC 170 = 0.07.

Graphs: detect graphId 2 = 2.76 ms; tracker 11 = 0.38; eager 0 = 0.30; main NMS 5 = 0.18; GMC 8 = 0.07.

`group_runs` on the detect graph is **not** a frame clock under DB: overlapping `detect(N)` and `detect(N+1)` merge (581 blobs vs 948 frames). Overlap numbers below use scan-anchored windows against the detect-graph union.

### Exposed vs hidden (MOT17-04)

| Stage | duration | hidden | exposed | class |
|:--|--:|--:|--:|:--|
| gemm_conv | 1.822 | 1.822 | 0 | fully hidden (is detect) |
| trt | 0.426 | 0.426 | 0 | fully hidden |
| scan | 0.377 | 0.377 | 0 | fully hidden |
| nms | 0.307 | 0.289 | 0.018 | almost hidden |
| memcpy | 0.164 | 0.058 | **0.107** | partially exposed |
| tracker_occlusion | 0.165 | 0.102 | 0.063 | partially exposed |
| tracker_sinkhorn | 0.077 | 0.014 | 0.063 | partially exposed |
| pointwise | 0.114 | 0.044 | 0.071 | partially exposed |
| tracker_auction | 0.039 | 0.018 | 0.021 | partially exposed |
| tracker_cost | 0.034 | 0.010 | 0.024 | partially exposed |
| gmc_fft | 0.046 | 0.040 | 0.006 | almost hidden |
| gmc_downscale | 0.021 | 0.015 | 0.006 | almost hidden |
| private_append | 0.020 | 0.019 | 0.001 | fully hidden |
| decode_pre | 0.030 | 0.009 | 0.021 | partially exposed |

Period decomposition (production clocks):

```text
2.875 ms  ≈  2.65 ms detect span
          +  0.23 ms outside_detect_remainder
```

`outside_detect_remainder` is P-layer period minus D-layer detect span,
not production period minus nsys GPU-union busy (that difference is
negative and is not idle). nsys `cudaStreamSynchronize` in the tail is
**23 µs** (cost A). The 0.23 ms remainder is cost B (opportunity, with
cross-run uncertainty). B dominates A.

---

## Fixed-capacity tracker scaling

| Seq | role | dets | active tracks | auction µs | sinkhorn µs | occlusion µs | NMS select µs | host track_ms |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| MOT17-05 | low | 6.8 | 10.8 | 34.5 | 75 | 155 | **39** | 0.263 |
| MOT17-13 | medium+motion | 18.0 | 19.7 | 33.0 | 74 | 148 | 93 | 0.273 |
| MOT17-04 | high | 42.9 | 43.9 | 34.6 | 77 | 165 | **243** | 0.326 |

Association compute is a **Tcap=2048 / Dcap=1024 floor**. NMS select is the occupancy-sensitive GPU piece. Host `track_ms` only moves 0.263 → 0.326.

`SACCADE_ASSOC_STATS` (branch-only tracker counters, commit `e03f7d81`) reports `sum_num_dets = 1024` every frame: that is Dcap, not the live count. Live detections are `dets_hi + dets_mid + dets_lo`.

---

## GMC breakdown

Production path: `_gmc_frame_buf.copy_(_frame_gmc)` then cuFFT graph replay
(`estimate_into_direct`: gray-downscale → phase correlation → small prev-gray D2D).
`gmc_fg_mask` is off.

| Piece | MOT17-04 | MOT17-05 | On critical path? |
|:--|--:|--:|:--|
| Full-frame DtoD staging | 47.4 MB (two ~24.9 MB copies) | 0 MB above 8 MB threshold | contributes to exposed memcpy 0.11 ms on 1080p |
| Downscale kernel | 21 µs | 6 µs | 6 µs exposed |
| FFT / cross-power / peak | 46 µs | 26 µs | 6 µs exposed |
| Result warp | inside graph, device-side | same | hidden |
| Sync | graph replay, no extra device sync | same | not a measured stall |

GMC is a **memory-traffic** issue on 1080p if and only if the staging copies are the 0.11 ms exposed memcpy. GMC **compute** is not a production-period problem. Do not start #341 from byte count alone; the copies are mostly overlapped, and only the exposed 0.11 ms (shared with ingest DtoD) is in the tail.

---

## Association / private continuation workload

Diagnostic (`SACCADE_ASSOC_STATS=1`). Extra kernels captured into the tracker/NMS graphs. FPS is not production.

**Provenance split (read before quoting this section).** The per-stage
association counters (S0/S1/S1b/S1c/S2 assignments, `dets_hi/mid/lo`,
active/confirmed/tentative, occ-state) were produced by tracker-side
instrumentation that is **not on `main`**. `include/tracking/tracker_gpu.hpp`
and `src/tracking/tracker_gpu.cu` are strict path+sha256 frozen inputs of the
closed H0 / GCTM packets (`h0_gctm_interface_static_feasibility_20260723`,
`gctm_runtime_native_candidate_universe_20260724`); the instrumentation was
therefore kept out of the landed code rather than re-freezing those packets.
It exists only at commit `e03f7d81` (blobs `tracker_gpu.hpp@a43f7069`,
`tracker_gpu.cu@e094d4f1`) and the D2 dumps under
`runs/production_db_critical_path_20260917/d2_assoc_stats/` were written by
that build on 2026-09-17 22:44. Those columns are **branch-only evidence**:
one run, not reproducible from `main`, and they do not enter the bottleneck
ranking below. What `main` carries and can reproduce: the
`PerceptionPipeline` private-continuation counter (`private added`) and the
bridge counters (`bridge accept`).

Per-frame means:

| Seq | S0 assign | S1 assign | S1b | S1c | S2 | private added | bridge accept |
|:--|--:|--:|--:|--:|--:|--:|--:|
| 02 | 7.4 | 9.1 | 0.58 | 0.69 | 0.02 | 1.30 | 24 |
| 04 | 24.5 | 15.1 | 1.46 | 0.90 | **0.00** | 1.61 | 9 |
| 05 | 1.7 | 3.5 | 0.26 | 0.58 | 0.01 | 0.17 | 45 |
| 09 | 3.1 | 3.6 | 0.28 | 0.54 | 0.01 | 0.38 | 11 |
| 10 | 4.8 | 9.7 | 1.03 | 1.04 | 0.04 | 2.52 | 57 |
| 11 | 5.0 | 3.4 | 0.21 | 0.26 | 0.01 | 0.38 | 11 |
| 13 | 3.3 | 8.9 | 0.99 | 0.97 | 0.04 | 3.13 | 37 |

Classification:

| Pass | Pattern |
|:--|:--|
| S0, S1 | 常跑 + 有效工作 |
| S1b, S1c | 常跑 + little work (0.2–1.5 assigns) |
| S2 | 常跑 + 幾乎沒工作 (FP filter empties `[0.05, 0.10)`) — branch-only evidence, see provenance split |
| Private NMS + `<<<1,1>>>` append | 常跑; 0.17–3.1 added boxes; 2048-slot prior buffer always passed |
| Bridge | 少跑 relative to frames (43–169 attempts / sequence), cheap |
| Occlusion | 常跑 + fixed Tcap scan |

S2 still pays one auction+commit launch (~7 µs). Observation only; no module removed.

Occ-state: 0.4–3.1 tracks with `front_ttl > 0` per frame. Not a duration driver.

---

## Bottleneck closure

Ranked by **exposed cost to production period**, not kernel duration.

### Primary — detector whole-graph (class D)

- Observed: 2.65 ms detect span (gemm 1.82 + scan 0.38 + TRT 0.43 + upsample/pointwise inside the graph).
- Exposed: 2.65 ms (the period).
- Every frame. Occupancy-insensitive; mildly resolution-sensitive (05 span 2.54 vs 04 2.65).
- Largest attackable slice **inside** the path: selective_scan 0.38 ms, 3 launches.
- If scan vanished and nothing else moved: ceiling ≈ 400 FPS. Not a commitment.

### Secondary — detect-to-detect tail (class B + C)

- Observed leftover on production clocks: 2.875 − 2.65 = **0.23 ms**.
- Exposed memcpy 0.11 ms of 48 MB DtoD on 1080p; 0.016 ms on 640×480.
- `cudaStreamSynchronize` self-cost 23 µs; opportunity loss is the rest.
- If the whole tail vanished: ceiling ≈ 378 FPS.

### Tertiary — fixed-capacity association (class A)

- Occlusion + sinkhorn + 5 auctions: ~0.28 ms duration, ~0.15 ms exposed on MOT17-04.
- Does not scale with 11 vs 44 tracks.
- If all exposed association vanished: ceiling ≈ 367 FPS.

**Not a next target:** GMC FFT, S2, Green Context routing.

**Class G (overlap already saturated) applies to tracker/GMC/NMS as a group.** Their duration is large relative to the 0.23 ms tail they actually expose. Optimizing a hidden 0.5 ms kernel will not move FPS.

---

## UNRESOLVED

- Exact SM clock / power during P0 (not sampled).
- Which of the two ~24.9 MB DtoD copies is GMC staging vs ingest/DB clone, at the call site, with production clocks. nsys shows 48 MB total k8; attribution to GMC vs ingest is by size, not by CUDA graph id of the copy.
- Per-pass GPU duration of S0 vs S1 vs S1b vs S1c vs S2 (same kernel name; order is 5 consecutive auctions at 7 µs each, so splitting them does not change the conclusion).
- Per-stage association workload on `main`: the S0–S2 counters above are a single branch-only run (commit `e03f7d81`). Re-measuring them requires either re-applying that instrumentation on a research branch or a governed supersession of the frozen `tracker_gpu.{hpp,cu}` identities; neither is done here.
- Whether the 0.11 ms exposed memcpy is *caused* by GMC staging or by detect-side buffer rotation. Needed before a copy-elimination PR.

---

## Reproduce

See the contract. Raw traces live under
`runs/production_db_critical_path_20260917/` (not versioned). Derived JSON is
versioned.
