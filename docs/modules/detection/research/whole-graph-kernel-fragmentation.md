# Whole-Graph Kernel Fragmentation Analysis

**Date:** 2026-06-08
**Config:** `mamba_whole_graph` preset, SDP detector, relink-bridge on, MOT17-02-SDP (600 frames)
**GPU:** RTX 5070 Ti Laptop (Blackwell GB205), FP16 = FP32 = 17.04 TFLOPS (shader 1:1)
**Tools:** `nsys --cuda-graph-trace=node`, `nsys --gpu-metrics-set=gb20x`, `ncu`

---

## TL;DR

- The detect path is **GPU-busy-bound**, not host/launch-bound. whole_graph = **2.82 ms GPU busy / frame**, bubble (idle gaps) only 0.13 ms.
- But the GPU is **NOT compute- or memory-saturated**: during kernel execution **SM Issue 22%, Tensor 19%, DRAM <17%**. It is **latency / occupancy / fragmentation-bound** — ~78% of issue slots are empty.
- Root cause is **kernel fragmentation**: **~372 kernels per frame** inside one detection forward, of which **192 are tiny pointwise** kernels (~2 µs each).
- Of those 192: **88 are TRT-internal (`__myl`, not ours)**, **104 are torch elementwise we can fuse**.
- The single biggest *fixable* knobs:
  1. **Fuse the 104 torch pointwise kernels** (torch.compile / custom fused kernels) — attacks fragmentation directly.
  2. **selective_scan kernel redesign** (block size 16 → ≥32; optionally fuse the 3 FPN scales) — 342 µs/frame at 30% SM.
- Already tested and **rejected**: fp16 head (−0.7 pp IDF1), channels_last (zero speedup). See [fp16](#appendix-rejected-approaches) / [channels_last](#appendix-rejected-approaches).

---

## 1. Measurement caveat that misled the first pass

`nsys`'s default `cuda_gpu_kern_sum` (graph trace = "graph" granularity) **does not see kernels inside a captured CUDA graph** — it only records one span per graph launch. Reading that table alone shows GPU ~3% busy and screams "host-bound", which is **wrong**.

The fix is `--cuda-graph-trace=node`: every kernel node inside the graph then lands in `CUPTI_ACTIVITY_KIND_KERNEL` with a `graphId`. All numbers below come from node-mode.

---

## 2. Per-frame kernel anatomy of whole_graph

whole_graph (one detection forward, captured + replayed every frame):

| # | Category | kernels/frame | µs/frame | % time | Ours to change? |
|---|---|---:|---:|---:|---|
| 1 | TRT backbone conv (fp16, `*_trt`) | 84 | 659 | 23% | No (TRT engine) |
| 2 | head conv cutlass **TF32** | 9 | 684 | 24% | Yes (precision/fusion) |
| 3 | head GEMM `simt_sgemm` (**fp32**) | 13 | 234 | 8% | Yes |
| 4 | other conv/gemm | 11 | 161 | 6% | Partly |
| 5 | **selective_scan** | 3 | 342 | 12% | **Yes (kernel)** |
| 6 | layout transpose (nchwToNhwc/…) | 33 | 182 | 6% | Architectural* |
| 7 | cat/concat | 13 | 32 | 1% | Partly |
| 8 | **elementwise/pointwise** | **192** | 379 | 13% | **104 of them, yes** |
| 9 | index/gather/topk | 3 | 32 | 1% | Partly |
| 0 | other | 11 | 118 | 4% | — |
| | **TOTAL** | **~372** | **2823** | 100% | |

\* layout transpose is *not* removable by `channels_last` — see appendix.

### The fragmentation is the pointwise swarm

Category 8 is **192 kernels = >50% of all kernels**, but only 13% of time → a swarm of ~2 µs ops. Split:

| pointwise source | kernels/frame | µs/frame |
|---|---:|---:|
| TRT myelin (`__myl_*`, inside backbone) | 88 | 182 | not ours |
| **torch elementwise (head + postprocess)** | **104** | **197** | **fuseable** |

The 104 torch ops are silu / mul / add / cast / sigmoid / DFL decode / index ops scattered across the 3 FPN scales and the postprocess.

---

## 3. Where the concrete overhead is

Time (`busy`) is dominated by real conv/gemm work; **kernel *count* is dominated by pointwise fragmentation**:

```
busy (µs/frame):   conv/gemm 1738 | selective_scan 342 | pointwise 379 | transpose 182 | other 182
count (k/frame):   pointwise 192  | conv/gemm 117      | transpose 33  | cat 13 | other 17
```

GPU-metric proof that this is fragmentation/latency-bound, not saturation (samples taken while SMs active >50%):

| metric | value | meaning |
|---|---:|---|
| SMs Active | 78% | blocks resident on SMs |
| **SM Issue** | **22%** | instructions issued only 22% of cycles → stalling 78% |
| Tensor Active | 19% | tensor cores mostly idle |
| Compute Warps in Flight | 37% | low occupancy |
| DRAM R / W | 7% / 17% | **not bandwidth-bound** |

Interpretation: the SMs hold blocks but spend most cycles **stalled** — waiting on dependencies, kernel-launch latency between dependent tiny kernels, and low occupancy (selective_scan runs 16 threads/block). There is large idle compute capacity; the limiter is how the work is **shaped and chopped up**, not raw FLOPs or bytes.

---

## 4. Why fragmentation still hurts inside a CUDA graph

CUDA graph capture already removed the **host-side** per-launch cost (replay issues the whole graph with ~one host call; bubble is only 0.13 ms). But it does **not** remove the **device-side** costs of having 372 kernels:

1. **Grid-launch latency on the GPU** — each kernel still has a fixed front/back latency the scheduler pays, even back-to-back.
2. **Global-memory round-trips** — each pointwise kernel reads its input from and writes its output to global memory. Fusing N pointwise ops into 1 keeps intermediates in registers/shared, cutting (N−1) round-trips.
3. **Serial dependency stalls** — a 2 µs kernel that depends on the previous one can't start until the prior finishes draining; with 192 of them the stalls dominate (hence SM Issue 22%).
4. **Low occupancy per kernel** — tiny grids don't fill the SM, so even when "active" the issue rate is low.

Fewer, larger kernels raise occupancy, cut memory traffic, and remove stall points → directly lifts the 22% issue rate.

---

## 5. How to reduce kernel count (ranked)

### A. Fuse the 104 torch pointwise kernels — *highest leverage on fragmentation*
- **`torch.compile`** on `mamba_head._forward_eager` (or the per-scale block) lets Inductor fuse pointwise chains (silu, mul, add, sigmoid, cast) into a handful of kernels. Biggest count reduction for least code.
  - Risk: `torch.compile` + `make_graphed_callables` interaction; may need `mode="reduce-overhead"` (which itself uses CUDA graphs) instead of the manual whole_graph. Validate capture still works and output bit-exactness.
- **Hand-written fused kernels** for the hot chains (e.g. the mamba block's `silu(conv1d) → x_proj` and `y*silu(z) → out_proj` boundaries, and the postprocess DFL→sigmoid→topk). More work, more control, graph-safe.
- Expected: 104 → ~20–30 kernels; the 197 µs is mostly latency so wall savings can exceed the nominal busy time by cutting stalls.

### B. selective_scan redesign — *no accuracy cost, single kernel*
- 342 µs/frame, 3 launches, **block size 16** (= d_state N, half a warp), SM 30%, occupancy 50% (ncu).
- Fix: parallelize over `D_dim` within the block (use ≥32, ideally 128–256 threads) so each block fills a warp+; raises occupancy and issue rate. Pure kernel change in `src/tracking/mamba_scan.cu`, numerics unchanged.
- Optional follow-on: the 3 FPN scales are independent until postprocess — they currently run serially on one stream. Could be merged into a **batched** scan (one launch over all scales) → 3 → 1 launch and better SM fill. Bigger change.

### C. Eliminate cast/transpose kernels — *only if precision path changes*
- 33 layout transposes (182 µs) + the fp16 cast swarm are seams between precisions/layouts. `channels_last` does **not** remove the transposes (appendix). They are only removable by changing the architecture so the mamba block doesn't impose an NCHW boundary, or by a unified-precision rebuild (rejected — see appendix).

### D. Multi-stream overlap (the "manual scheduling" idea)
- Orthogonal to reducing count: overlaps the independent 3 scales to fill the idle 78% issue. Requires manual graph capture (fork/join streams; `make_graphed_callables` is single-stream). Engineering-heavy. Prefer A+B first — they reduce the work itself rather than packing it tighter.

---

## 6. What is NOT worth touching

- **TRT backbone**: 84 conv + 88 myelin pointwise = ~841 µs/frame, all inside the TRT engine. Closed kernels, already fused by TRT's myelin. Not ours to fragment-reduce.
- **Bubble**: 0.13 ms/frame. Already negligible — "better kernel connection/scheduling" of the *serial* chain buys nothing here.

---

## Appendix: rejected approaches (measured, 2026-06-08)

| Approach | Speed | Accuracy (full MOT17-SDP) | Verdict |
|---|---|---|---|
| **fp16 head** (autocast, scan kept fp32) | +17% FPS | **−0.7 pp IDF1 / −0.7 pp AssA**, FP +190 | NO-GO (accuracy; assoc hurt by fp16 feature jitter). GPU is shader 1:1 so only tensor-core convs gain; cast overhead ate ~17% of gross gain (~126 µs/frame). |
| **channels_last head** | 0% (net wash) | bit-exact | NO-GO. Transposes 100% removed (22→0) but equal cost reappears as copies — mamba SSM `flatten/transpose/reshape` forces an NCHW boundary; cuDNN relocates the transpose, doesn't remove it. |
| **unify precision (fp16 IO engine + .half head)** | would recover the ~126 µs cast overhead | unchanged −0.7 pp | not pursued — removes cast cost but not the accuracy regression; needs backbone engine rebuild. |

Backbone engine `yolo26s_backbone_640_best.engine`: **IO bindings FP32, compute FP16** (standard TRT fp16 mode) — there are fp32↔fp16 reformats at the TRT boundary already.

The only way to run the head in fp16 without the −0.7 pp hit is to **fine-tune / QAT the head in fp16**.

---

## Reproduce

```bash
# node-level graph trace (per-kernel inside the graph)
nsys profile --trace=cuda --cuda-graph-trace=node --force-overwrite=true \
  -o /tmp/nsys_node .venv/bin/python scripts/eval/mot17.py \
  --preset mamba_whole_graph --detector SDP --relink-bridge-enabled \
  --sequences MOT17-02-SDP --output /tmp/node_out

# GPU utilization sampling (SM Issue / Tensor / DRAM)
nsys profile --trace=cuda --gpu-metrics-devices=0 --gpu-metrics-set=gb20x ... 

# per-kernel roofline (e.g. selective_scan)
ncu --launch-count 3 -k "regex:selective_scan" \
  --section SpeedOfLight --section Occupancy ... 
```

Categorize per-frame kernels (node sqlite, graphId of whole_graph = the largest):
```sql
SELECT <category CASE>, COUNT(*)/600, ROUND(SUM(end-start)/1e3/600,1)
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.shortName=s.id JOIN StringIds d ON k.demangledName=d.id
WHERE k.graphId=<id> GROUP BY category;
```
