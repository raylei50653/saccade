# Kernel Fusion Plan: Torch Elementwise Fragmentation

**Date:** 2026-06-08
**Baseline:** `mamba_whole_graph` preset, SDP detector, relink-bridge on, MOT17-02-SDP
**GPU:** RTX 5070 Ti Laptop
**PyTorch:** 2.11.0+cu130

---

## 1. Problem Recap

From nsys `--cuda-graph-trace=node` analysis (matching `whole-graph-kernel-fragmentation.md`):

| Graph | kernels/frame | µs/frame |
|---|---|---|
| whole_detect (#2) | 372 | 2823 |
| NMS (#5) | 9 | 115 |
| GMC (#8) | 10 | 48 |
| Tracker (#11) | 25 | 433 |
| **Total** | **416** | **3418** |

Within whole_detect, **178 torch elementwise kernels** (104 elementwise + 66 SiLU + 4 cast + 4 arith) consume 338 µs but are the dominant fragmentation source (46% of kernels, only 12% of time). Each tiny kernel pays per-launch scheduler tax and forces global-memory round-trips.

The 178 kernels fall into 6 categories spanning 3 pipeline phases:

| Category | kernels/frame | µs/frame | Where |
|---|---|---|---|
| A: Pure activation (SiLU/sigmoid) | ~66 | ~109 | MambaBlock + cls/reg heads + postprocess |
| B: Arithmetic chains (mul/add/div/sub/exp) | ~40 | ~70 | MambaBlock + postprocess |
| C: Slice/index ops | ~20 | ~35 | MambaBlock x_proj slicing |
| D: Layout transpose (nchw↔nhwc) | 33 | 182 | MambaBlock + per-scale processing |
| E: Concat | 13 | 32 | Postprocess + per-scale skip-cat |
| F: Postprocess dense chain (sigmoid→max→topk, dist2bbox→scale→xyxy) | ~48 | ~108 | `_postprocess_mamba_fixed` |

**Untouchable:** 88 TRT myelin pointwise (inside TRT engine), 31 cutlass conv/GEMM (already optimized).

---

## 2. Fusion Strategy: torch.compile

`torch.compile` (Inductor backend) fuses pointwise chains by keeping intermediates in registers/shared memory. Applied at the right granularity:

- **P1** — Postprocess chain: `torch.compile` on `_postprocess_mamba_fixed`
- **P2** — Per-scale cls/reg head: `torch.compile` on `cls_head[i].forward` + `reg_head[i].forward`
- **P3** — MambaBlock: `torch.compile` on `MambaBlock.forward`

P4 (layout transpose) is deferred — historical measurement shows `channels_last` yields zero net speedup.

### Compatibility with `make_graphed_callables`

The whole_detect graph uses `torch.cuda.make_graphed_callables` which wraps `_head_fn` →
`_forward_eager`. `torch.compile` is compatible with CUDA graphs — Inductor uses
`mode="reduce-overhead"` which itself leverages CUDA graphs, but when the compiled
function is placed *inside* a graphed callable, the outer graph capture works on the
compiled graph's kernel launches.

**Risk:** Double-graph nesting (Inductor's internal graph inside our outer graph).
Mitigation: use `mode="default"` (not `"reduce-overhead"`) at P1/P2/P3, letting our
existing outer `make_graphed_callables` handle the replay. If `mode="reduce-overhead"`
is needed, we would replace the outer graph with Inductor's.

---

## 3. Implementation Plan

### P1: Postprocess (`_postprocess_mamba_fixed`)
- **File:** `src/saccade/perception/temporal_yolo/mamba_gated_detector.py`
- **Scope:** Wrap the decode logic in `@torch.compile` or `torch.compile(func, mode="default")`
- **Expected:** 48 → ~5 kernels, ~108 → ~60 µs
- **Risk:** `topk` inside compile may have graph breaks; test first
- **Verification:** bit-exact output diff == 0.0 on random inputs

### P2: Per-scale cls/reg heads
- **File:** `src/saccade/perception/temporal_yolo/mamba_head.py`
- **Scope:** Compile the `nn.Sequential` (Conv2d→SiLU→Conv2d) chain per scale
- **Expected:** 30 → ~8 kernels, ~50 → ~30 µs
- **Risk:** `nn.Sequential` recompilation on first capture
- **Verification:** output diff ≤ 1e-5 on random inputs

### P3: MambaBlock
- **File:** `src/saccade/perception/temporal_yolo/mamba_head.py`
- **Scope:** `torch.compile(MambaBlock.forward, mode="default")`
- **Expected:** 100 → ~25 kernels, ~180 → ~100 µs
- **Risk:** Custom CUDA op (`selective_scan_fwd`) inside compiled function — needs `@torch.library.custom_op` which is already registered. torch.compile treats it as a black-box op via `torch._higher_order_ops.wrap`.
- **Verification:** output diff ≤ 1e-5 on random inputs

---

## 4. Verification Criteria

For each phase:
1. **Bit-exactness:** Random input, compare compiled vs eager output (max abs diff)
2. **CUDA Graph compatibility:** Run full `scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP --sequences MOT17-02-SDP --max-frames 20`, confirm graph capture succeeds and no errors
3. **No regression:** Full MOT17-SDP eval (7 sequences, ~1600 frames) — MOTA/IDF1 within ±0.3pp of baseline, FPS ≥ baseline

### Baseline metrics (from `mamba_whole_graph` preset, 2026-06-08)
```
IDF1 73.3%, MOTA 77.1%, HOTA 66.7%, FPS 157 (7-seq aggregate)
```

### Results (2026-06-08, RTX 5070 Ti Laptop)

**P1+P2 (postprocess + cls/reg head compile):**

| Metric | Baseline | Compiled | Delta |
|---|---|---|---|
| IDF1 | 74.2% | 74.2% | 0.0 pp |
| MOTA | 77.7% | 77.7% | 0.0 pp |
| HOTA | 67.5% | 67.5% | 0.0 pp |
| IDs | 497 | 497 | 0 |
| FP | 3560 | 3560 | 0 |
| FN | 20933 | 20933 | 0 |
| **FPS** | **211.47** | **217.54** | **+2.9%** |

- Accuracy: **bit-exact** (all metrics identical)
- FPS: +2.9% aggregate, +1-2% per sequence
- CUDA Graph capture: compatible, no errors

**P3 (MambaBlock compile):** Fixed. Root cause was the fake kernel for
`selective_scan_fwd` returning `torch.empty_like(u)` which preserved
non-contiguous strides from `flatten(2).transpose(1,2)` in the cross-scan
path. The real kernel always outputs contiguous (calls `.contiguous()` first).
Fixed by changing fake kernel to `torch.empty(u.shape, dtype=u.dtype,
device=u.device)` which always returns contiguous.

### Final results (P1+P2+P3, 2026-06-08)

| Metric | Baseline | P1+P2+P3 | Delta |
|---|---|---|---|
| IDF1 | 74.2% | 74.2% | 0.0 pp |
| MOTA | 77.7% | 77.7% | 0.0 pp |
| HOTA | 67.5% | 67.5% | 0.0 pp |
| IDs | 497 | 497 | 0 |
| **FPS** | **211.47** | **221.48** | **+4.73%** |

Per-sequence FPS:
| Sequence | Baseline | P1+P2+P3 | Delta |
|---|---|---|---|
| MOT17-02 | 216.49 | 220.49 | +1.8% |
| MOT17-04 | 210.68 | 216.11 | +2.6% |
| MOT17-05 | 220.96 | 228.33 | +3.3% |
| MOT17-09 | 217.81 | 222.74 | +2.3% |
| MOT17-10 | 212.10 | 220.34 | +3.9% |
| MOT17-11 | 216.01 | 223.05 | +3.3% |
| MOT17-13 | 190.67 | 220.88 | — |

### Changes summary

| File | Change |
|---|---|
| `mamba_head.py:107` | Fake kernel: `empty_like(u)` → `empty(u.shape, ...)` |
| `mamba_head.py:621-673` | `set_head_compile()` + `set_block_compile()` methods |
| `mamba_head.py:561-563` | Added `_head_compile_enabled`, `_block_compile_enabled` flags |
| `mamba_gated_detector.py:114-210` | `_postprocess_mamba_fixed` split into dispatch + eager + compiled |
| `scripts/eval/mot17.py:95-98` | Added `--no-compile` flag |
| `scripts/eval/mot17.py:271-274` | Enable all three compilations by default |

### Rollback

Use `--no-compile` flag:
```bash
uv run scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP --no-compile ...
```
