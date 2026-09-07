# Tracker-lane dose response — added work is shape-sensitive, not duration-sensitive (2026-09-07)

> **本文回答的問題**:headline path 已經把 detect(N+1) 疊在 tracker(N) 上,那麼**在 tracker lane
> 上多加工作,會有多少變成 frame period**?這是 [frame_budget_20260905](frame_budget_20260905.md)
> 的**實驗性補充**,不是新的 optimization claim —— 本文不主張任何優化,只量一條 lane 的
> absorption envelope。
>
> **核心結果先講。** 兩種 dose 的 isolated wall time 幾乎相同(169.0 µs vs 171.0 µs,差 1.2%),
> 但 dose-response 完全不同:
>
> > Tracker-lane added work is shape-sensitive rather than merely duration-sensitive. A
> > wall-clock-matched high-occupancy GEMM dose transfers to frame period at S ≈ 0.95–1.00,
> > while a `<<<1,1>>>` latency-shaped dose exhibits approximately 0.5 ms/frame of absorption
> > before becoming increasingly exposed. The result establishes a limited low-resource
> > overlap envelope, not a general-purpose compute budget.
>
> 這個 **matched-cost shape control** 才是本文的主要產出:「能不能藏」主要取決於
> **resource shape / concurrency compatibility**,不是這份工作花幾微秒。
>
> 全部數字出自 2026-09-07、`mamba_whole_graph_m` preset、MOT17 train / SDP 七序列、
> RTX 5070 Ti Laptop GPU、main `745be394`(clean)。未變更 driver、顯示設定或主機設定。
>
> ⚠️ **量測用的是 temporary probe patch,未 land、不在版控**(§2.3 有完整 patch 內容與
> 還原步驟)。measurement 結束後 working tree 已還原為 clean。
>
> ⚠️ **本文不 promote 到 `evidence_ledger` 或 `report_data`**,scope 限本機、本 preset、
> single-stream。

---

## §1 The question

`frame_budget_20260905` §2 記錄 headline path 的 lane 結構:detect lane 佔 2019.5 µs
(68% of a 3022 µs period),其餘 ~1.0 ms 是 postproc(tracker association 314.6 µs、
兩趟 NMS 410.8 µs、frame ingest 206.3 µs、GMC 54.3 µs),而 GPU busy 93.6%、
lane busy 總和只有 period 的 1.18×。

那份文件量的是**現況的分佈**。本文問的是**邊際**:在 tracker lane 上再加 Δ 的工作,
frame period 會長多少?記

```
S = Δ(frame period) / (k · T_iso)
```

`S = 1` ⇒ 加的工作全額曝露在 critical path;`S = 0` ⇒ 完全被既有排程吸收。

這個問題**不能**從 §3 的 µs/frame 表推出來,因為那張表是 cost 不是 wall-time partition。

---

## §2 Substrate and protocol

### §2.1 Baseline

| | |
|:--|:--|
| command | `scripts/eval/mot17.py --preset mamba_whole_graph_m --detector SDP --double-buffer` |
| data | MOT17 train / SDP, 7 sequences, 4 966 frames |
| gpu | RTX 5070 Ti Laptop GPU · GB205 · 46 SM · 12 GiB |
| driver | 616.56 (Windows KMD) · WSL2 |
| toolchain | torch 2.11.0+cu130 · CUDA 13.0 runtime · nvcc 13.3 for the probe library |
| repo | main `745be394`, clean apart from the probe patch |
| tracker path | `GraphedTrackerUpdate` captured (preset sets `use_tracker_graph: true`, `reid_mode: off`) |
| decode | left at preset default (no `--gpu-decode` / `--no-gpu-decode` on the command line) |

Baseline throughput over the session ranged 334–362 FPS (period 2.76–2.99 ms). That drift
is why **every dose point is paired against its own rep's `k=0` anchor** and never against a
session-wide mean.

### §2.2 Injection site

Dose is injected in `_run_track` (`src/saccade/perception/eval/stages.py:518`), immediately
after `_apply_score_jitter` and **before** `gtu.copy_inputs` / `gtu.replay()`:

- **same stream** — the injected launches and `gtu.replay()` both go to the current stream,
  which is the tracker lane (detect runs on `double_buffer_stream`);
- **same dependency boundary** — after detect(N) has landed, before the tracker output is read;
- **outside the captured tracker graph** — the graph wraps the C++ `update_into`, so the dose
  cannot be captured into it without a native rebuild. The dose is therefore an *adjacent*
  launch on the same lane, not an in-graph node. This differs from the F3b/F3d probes, which
  injected inside the detector and GMC graphs.

### §2.3 The probe patch (not landed)

Env-gated so `k=0` is one integer comparison. `_dose_inject()` was added at the site above.

```python
_DOSE_K     = int(os.environ.get("SACCADE_TRACKER_DOSE_K", "0") or "0")
_DOSE_TYPE  = os.environ.get("SACCADE_TRACKER_DOSE_TYPE", "gemm")     # gemm | spin
_DOSE_N     = int(os.environ.get("SACCADE_TRACKER_DOSE_N", "1024"))    # gemm side
_DOSE_ITERS = int(os.environ.get("SACCADE_TRACKER_DOSE_ITERS", "104000"))  # spin side

def _dose_inject() -> None:
    if _DOSE_K <= 0:
        return
    if not _dose_state:
        _dose_init()
    if _DOSE_TYPE == "gemm":
        a, b, c = _dose_state["bufs"]
        for _ in range(_DOSE_K):
            torch.mm(a, b, out=c)          # 1024³ fp32, high occupancy
    else:
        for _ in range(_DOSE_K):
            lib.spin_launch(_DOSE_ITERS, sink_ptr, stream)   # <<<1,1>>>, low occupancy
```

The spin library is a standalone `.so` (no torch headers, loaded by `ctypes`, launched onto
the stream handle passed in from Python) so that no dependency had to be added to `.venv`:

```cuda
__global__ void spin_kernel(long long iters, float* sink) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float acc = 0.0f;
    for (long long i = 0; i < iters; ++i) acc = fmaf(acc, 1.0000001f, 1.0f);
    *sink = acc;
}
extern "C" void spin_launch(long long iters, float* sink, void* stream) {
    spin_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(iters, sink);
}
```

```bash
nvcc -O3 -shared -Xcompiler -fPIC -arch=native -o libspindose.so spin_dose.cu
```

Reverted with `git checkout -- src/saccade/perception/eval/stages.py`.

### §2.4 Dose calibration — the matched-cost control

Isolated cost, measured with CUDA events on an otherwise idle GPU (50 warmup, then 200 (GEMM)
/ 100 (spin) back-to-back launches):

| Dose A — `torch.mm` fp32 | µs | | Dose B — `<<<1,1>>>` spin | µs |
|:--|--:|:--|:--|--:|
| n = 512 | 36.4 | | 100 000 iters | 161.8 |
| n = 768 | 85.7 | | **104 000 iters** | **171.0** |
| **n = 1024** | **169.0** | | 200 000 iters | 321.1 |
| n = 1280 | 347.0 | | 400 000 iters | 636.1 |
| n = 1536 | 569.9 | | 500 000 iters | 793.0 |

The two chosen units differ by **1.2% in isolated wall time**. The 1024³ GEMM saturates the
device; the spin kernel occupies **one thread** of the 70 656 the device can hold resident, and
touches essentially no memory bandwidth, L2, atomics or registers. They therefore differ along
*several* resource axes simultaneously — which is what makes the pair a control on
"shape versus duration" and what stops it from isolating any single mechanism (§5).

### §2.5 Reading protocol

Pre-registered before the ladders were run, following the F3d protocol:

1. **Instrument positive control first** at large `k`, to prove the injection lands at all.
2. `k ∈ {0, 1, 2, 4}`, four reps, **order mirrored** per rep (asc / desc / asc / desc) so
   monotone session drift cancels.
3. Read **each rep's delta against its own `k=0`**, never a pooled anchor.
4. A dose level counts as **resolved** only if all four per-rep signs agree; otherwise it is
   reported as unresolved and **no breakpoint may be named as a real number** — only an interval.
5. **Dose inertness check**: IDF1 / MOTA / IDs must be identical across all runs.

**Run accounting.** 35 runs were executed under the probe patch:

| group | runs | anchor |
|:--|--:|:--|
| Dose A ladder — 4 levels × 4 reps | 16 | each rep's own `k=0` |
| Dose A positive control — `k=0` and `k=16` | 2 | its **own paired `k=0`** (2.9558 ms), run immediately before it and distinct from all four ladder anchors |
| Dose B ladder — 4 levels × 4 reps | 16 | each rep's own `k=0` |
| Dose B positive control — `k=16` | 1 | **none** — read in §4 against the pooled ladder `k=0` mean |
| **total under the probe patch** | **35** | |

One further baseline run was executed *before* the patch was applied (334.26 FPS) and is not
part of the 35.

**Dose inertness.** All 36 runs returned identical **IDF1 80.3 / MOTA 81.8 / IDs 358**. HOTA
was captured only on the pre-patch baseline run (74.3); the ladder harness extracted IDF1,
MOTA and IDs only, so **no HOTA invariant is claimed across the 35**. On the three metrics that
were recorded every time, the dose is inert and the pipeline is deterministic at this preset.

---

## §3 Dose A — high occupancy (T_iso = 169.0 µs)

Frame period, ms:

| rep | k=0 | k=1 | k=2 | k=4 |
|:--|--:|--:|--:|--:|
| 1 *(cold start)* | 2.9315 | 3.1395 | 3.2403 | 3.4740 |
| 2 | 2.7624 | 2.9273 | 3.0862 | 3.4186 |
| 3 | 2.7598 | 2.9188 | 3.0799 | 3.4077 |
| 4 | 2.7651 | 2.9301 | 3.0838 | 3.4082 |

Rep 1's own anchor sits 0.167 ms above reps 2–4, which agree to within 0.005 ms. Reps 2–4 are
reported as the primary reading and rep 1 as a documented cold-start transient; **both readings
are given, and they do not differ in conclusion.**

| k | injected | Δperiod per rep (reps 2–4) | µs / unit | **S** | signs |
|--:|--:|:--|--:|--:|:--|
| 1 | 169 µs | +0.1650 / +0.1589 / +0.1650 | 162.9 (sd 3.5) | **0.964** | 3/3 + |
| 2 | 338 µs | +0.3239 / +0.3200 / +0.3187 | 160.4 (sd 1.3) | **0.949** | 3/3 + |
| 4 | 676 µs | +0.6562 / +0.6479 / +0.6431 | 162.3 (sd 1.7) | **0.960** | 3/3 + |

Including rep 1: S = 1.031 / 0.940 / 0.921 at k = 1 / 2 / 4.

**Positive control**, k=16 (2 704 µs injected, ≈ 90% of a frame period):

| k | FPS | period | Δ | µs / unit | S |
|--:|--:|--:|--:|--:|--:|
| 0 | 338.32 | 2.9558 ms | — | — | — |
| 16 | 176.64 | 5.6613 ms | +2.7055 ms | 169.1 | **1.001** |

12/12 per-rep signs positive across the ladder, plus a control that reproduces the isolated
cost to within 0.1%. **High-occupancy work on the tracker lane is paid in full.**

---

## §4 Dose B — low occupancy (T_iso = 171.0 µs)

Same protocol, same site, same lane, wall-clock-matched dose.

Frame period, ms:

| rep | k=0 | k=1 | k=2 | k=4 |
|:--|--:|--:|--:|--:|
| 1 | 2.7757 | 2.8469 | 2.8148 | 2.9752 |
| 2 | 2.8243 | 2.7780 | 2.7984 | 3.0688 |
| 3 | 2.8394 | 2.7996 | 2.9390 | 2.9865 |
| 4 | 2.7746 | 2.7659 | 2.7937 | 2.9747 |

| k | injected | Δperiod per rep | mean Δ | **absorbed** | S | signs | verdict |
|--:|--:|:--|--:|--:|--:|:--|:--|
| 1 | 171 µs | +0.0712 / −0.0463 / −0.0397 / −0.0087 | −0.0059 ms | 177 µs | −0.03 | `+ − − −` | **unresolved** |
| 2 | 342 µs | +0.0391 / −0.0259 / +0.0996 / +0.0191 | +0.0330 ms | 309 µs | 0.10 | `+ − + +` | **unresolved** |
| 4 | 684 µs | +0.1995 / +0.2445 / +0.1471 / +0.2001 | +0.1978 ms | 486 µs | 0.29 | `+ + + +` | **resolved** |

**Positive control**, k=16 (2 736 µs injected), single run against the pooled `k=0` mean
(2.8035 ms) because no paired anchor was run for it:

| k | FPS | period | Δ | absorbed | S |
|--:|--:|--:|--:|--:|--:|
| 16 | 201.37 | 4.9660 ms | +2.1625 ms | 574 µs | **0.79** |

### What is and is not established here

The `k=0` anchor spread in this ladder is **0.065 ms**, larger than the mean dose effect at
both k=1 and k=2. By §2.5 rule 4 those two levels are **not resolved**: they are *consistent
with* full absorption but do not establish it, and no per-unit cost may be quoted from them.
This is the same resolution wall that closed the F3b detector-resize ladder.

The two readings that yield a numeric absorbed quantity agree with each other — noting that
only the first of them is a paired, resolved ladder point:

| | injected | absorbed |
|:--|--:|--:|
| k=4 (4/4 positive, paired) | 684 µs | **486 µs** |
| k=16 (control, pooled anchor) | 2 736 µs | **574 µs** |

**The resolved readings are consistent with a saturation plateau of roughly 0.5 ms/frame.**
Two dose levels fourfold apart in injected work return closely similar absorbed quantities.
This is an interpretation the two readings support, not an established scaling law: only k=4
is paired and resolved, while k=16 is a single run against a pooled anchor and is not a ladder
point (§2.5, §6.2). The implied plateau falls in the interval **342–684 µs of injected low-occupancy work per frame**
(above the largest unresolved level, at or below the smallest resolved one). Per §2.5 rule 4
it is reported as that interval and not fitted to a number; "approximately 0.5 ms" in the
summary is the absorbed plateau, read off the two resolved points.

---

## §5 Interpretation

Two doses, isolated wall times differing by 1.2%, produce qualitatively different responses:

| | Dose A (high occupancy) | Dose B (low occupancy) |
|:--|:--|:--|
| isolated cost | 169.0 µs | 171.0 µs |
| S at the low end | 0.96 (resolved, 3/3) | not resolved |
| S at k=4 | 0.96 | 0.29 |
| S at k=16 | 1.00 | 0.79 |
| absorbed | ~0 | plateau ≈ 0.5 ms/frame (two resolved readings, §4) |

**Isolated duration alone does not predict the response; the response is
workload-shape-sensitive. This matched-cost control does not identify which resource dimension
or scheduling mechanism causes the difference.** Two points cannot separate occupancy from
power, scheduler behaviour, register pressure, the memory subsystem, or anything else — and
the doses differ along all of those axes at once, not only in thread count.

Two facts already in `frame_budget_20260905` are *compatible* with the observed difference and
are recorded here as candidate mechanisms, **not** as attributions this experiment can support:
the GPU is 93.6% busy (Dose A's ~4% residual absorption is the same order as that 6.4%
timeline idle), and it holds 135 W against a 140 W cap with SM clock pinned at 2 497 MHz
against a 3 090 MHz boost. Discriminating among them needs an experiment that varies one axis
at a time, which this one does not.

**What this does not say.** The envelope is measured for a dose that touches essentially no
memory bandwidth, no L2, no atomics, and holds a trivial register footprint. It is an
envelope for that shape only.

---

## §6 Limits

1. **`k=1` and `k=2` on Dose B are unresolved,** not zero. The absorbed plateau rests on two
   points (k=4 paired, k=16 unpaired).
2. **Dose B's k=16 control has no paired anchor.** It is a control for instrument response,
   not a ladder point, and §4 reads it against a pooled mean. (Dose A's control *does* carry
   its own paired `k=0`; the two controls are not equivalent in strength.)
3. **The two ladders were run in sequence, not interleaved.** Dose A's anchors were tight
   (0.005 ms across reps 2–4), Dose B's loose (0.065 ms). Order effects *between* the two dose
   types are not controlled; each ladder is internally paired, and the cross-ladder comparison
   in §5 inherits that limitation.
4. **The spin dose is a lower bound on resource footprint,** so ~0.5 ms is an **upper bound**
   on the absorbable amount of any real work. A real postproc addition also consumes bandwidth,
   L2, atomics, registers and power, and may carry dependencies the spin kernel does not — any
   of which can move the response toward Dose A.
5. **The dose is adjacent to, not inside, the captured tracker graph** (§2.2). Work added
   *inside* the graph may schedule differently.
6. **Single host, single GPU, single preset, single-stream, MOT17 train / SDP.** No claim of
   transfer to other presets, resolutions, datasets or multi-stream operation.
7. **`S` is defined against an isolated cost measured at full clocks on an idle GPU.** It is a
   translation ratio, not a fraction of anything, and is not bounded above by 1.

---

## §7 Actionable consequences

Stated as what the measurement licenses, and no more.

1. **There is a limited low-resource overlap envelope on the tracker lane** — approximately
   0.5 ms/frame of absorption for extremely low resource-footprint, latency-shaped work. This
   is **not** a general-purpose 0.5 ms compute budget and must not be quoted as one.

2. **Additional association work is worth testing, at matched resource footprint.** Tracker
   association currently costs 386.5 µs/frame (`frame_budget_20260905` §3), which is the same
   order of magnitude as the observed absorption capacity. That makes additional association
   work with a comparably low resource footprint a **candidate worth measuring** — it does not
   make it free. Small grid alone is not sufficient: memory bandwidth, L2 / atomic traffic,
   register pressure, power and dependency structure can each break the analogy with the spin
   dose. Any candidate needs its own A/B.

3. **High-occupancy postproc is full price.** Anything GEMM-shaped — a ReID embedding pass
   being the obvious case — transfers at S ≈ 0.96 and should be costed at its isolated time.
   This is a timing-side observation consistent in direction with the accuracy-side result in
   [reid_handover_ablation_20260808](reid_handover_ablation_20260808.md); the two are
   independent measurements and neither is evidence for the other.

4. **Unverified inference, recorded as such:** F1 (`nms_select_counted_kernel<<<1,1>>>`,
   251.9 µs/frame) and F2 (occlusion, 169.0 µs/frame, 0.03 waves/SM) are exactly the shape this
   ladder finds absorbable, and they sit inside the measured envelope. It is *plausible* that
   removing them returns headroom to that envelope. **It is not measured, and the 251.9 µs must
   not be treated as a recoverable budget:** what the ladder measured is an absorption envelope
   *under the current schedule*, and removing F1/F2 changes the overlap topology that produced
   it. Re-running this ladder after any F1/F2 change is the way to find out; predicting the
   result from these numbers is not.

---

## §8 Reproduction

```bash
# 1. apply the §2.3 probe patch to src/saccade/perception/eval/stages.py
# 2. build the spin library
nvcc -O3 -shared -Xcompiler -fPIC -arch=native -o libspindose.so spin_dose.cu

# 3. one ladder point
export SACCADE_TRACKER_DOSE_TYPE=gemm   # or: spin
export SACCADE_TRACKER_DOSE_K=4
export SACCADE_TRACKER_DOSE_LIB=$PWD/libspindose.so
scripts/eval/mot17.py --preset mamba_whole_graph_m --detector SDP --double-buffer

# 4. revert
git checkout -- src/saccade/perception/eval/stages.py
```

**Retention.** The run logs were written to a session scratch directory and **were not
retained**. The per-run FPS in the tables above is the surviving record; every derived quantity
in this document is recomputed from those numbers.
