# Mamba head recall bottleneck — architecture & solution space

_Investigation 2026-06-29. Recall-first MOT. Records why head-side levers can't
move recall, the measured evidence, the core resolution↔scan-cost tension, and
the open solution space. Written to be reasoned over before committing to a path._

## 1. Goal
Real-time MOT, **recall-first** (FP not trusted — test has no GT, precision
already 97%+). Primary metric = recall, especially **small / distant people**.
Real-time budget ≈ 33 ms/frame (~30 fps video). d128 deploy = 269 FPS / ~3.7 ms
→ ~89% per-frame GPU idle = an *accuracy budget*, not a throughput trophy.

## 2. Current architecture
```
frozen yolo26s backbone ─► FPN P3/P4/P5 (128/256/512 ch)
                              │  (per level, strictly serial / data-dependent)
 input_proj(full-res) ─► downsample ÷4 ─► [Mamba scan] ─► upsample ×4 ─► heads(full-res)
                          feeds scan      coarse grid    restores res    cls/reg
```
Param breakdown (d128 head, 11.37M total):

| module | params | % | role |
|---|---:|---:|---|
| upsample (pixel-shuffle, conv d_model→16·d_model) | 7.08M | **62%** | restore the ÷4-discarded resolution |
| downsample | 0.79M | 7% | ÷4 so the scan is cheap |
| cls/reg/gate heads | ~2.7M | 24% | detection output (full-res) |
| **Mamba (the actual core)** | **0.69M** | **6%** | global context |

→ This "Mamba head" is **69% conventional FPN neck, only 6% real Mamba.**

## 3. Core tension (one line)
> Small-object recall needs **full spatial resolution** into detection; the Mamba
> scan cost scales **∝ sequence length L = H×W**.

The head resolves this by `downsample ÷4` (L 6400→400 at P3): scan becomes
cheap (launch-bound, ~free) **but the ÷4 discards small-object resolution → recall
ceiling**, and costs 7.87M params on the round-trip. And it **cannot just be
removed** — deleting the downsample makes the scan compute-bound at full res =
**14× slower** (measured).

## 4. Measured evidence (this session, all benchmarked)

**A. Widening the head (d256) = 堆料, does not move recall — CLOSED.**
Params 4.4× (all into the neck; d256 still 63% upsample). In-sample recall only
+1.2. Latency 154 FPS w/ double-buffer (still real-time). → head capacity can't
move recall.

**B. GT1 augmentation, held-out LOO (MOT17-10, d128) = recall NO-GO; assoc GO.**

| | control | augment | Δ |
|---|---:|---:|---:|
| Rcll | 74.5 | 74.0 | −0.5 |
| IDF1 | 56.9 | 57.4 | +0.5 |
| MOTA | 61.4 | 63.8 | +2.4 |
| IDs | 143 | 108 | −35 |

Held-out recall flat (NO-GO for recall); IDs/MOTA clearly better → augmentation is
an **association/stability regularizer**, parked for the recall goal, kept in
pocket if IDs/association becomes the target. (Single fold — directional.)

**C. spatial_reduction latency sweep (eager fp32, head-only):**

| config | P3 scan grid | ms | vs sr=4 |
|---|---:|---:|---:|
| sr=4 (current) | 20×20=400 | 16.6 | 1× |
| sr=2 | 40×40=1600 | 48.6 | **3×** |
| sr=1 | 80×80=6400 | 189.8 | 11.5× |
| delete down/up | 80×80=6400 | 197.7 | **14×** |

Scan is **launch-bound at sr=4** (44% launch-bound, idle SMs), **compute-bound at
full res**. → delete-down/up (sr=1) CLOSED. sr=2 borderline (~3× head latency;
after eager→production deflation ~13 ms head, likely still real-time but eats half
the headroom; recovers half the resolution).

**D. Backbone (from memory `project_yolo26m_capacity`, not re-run):** yolo26m gives
**small-object recall +7pp** (02 min_4to8 0.900 vs s 0.826), MOTA +0.8, FP −14%,
but AssA −3.0; m-best (relax relink bridge gate) = IDF1 parity, strictly better
detection/MOTA/FP/IDs = "recall/MOTA win at IDF1 parity".

## 5. Attribution — why head-side levers can't move recall
Three head levers (d256 widen, aug regularize, delete down/up) all hit a wall
because **the recall information bottleneck is upstream**: the frozen yolo26s P3
features + the head's own ÷4 downsample both discard small-object detail *before*
detection. No amount of head widening / regularizing / re-ordering invents detail
the features don't contain.

## 6. Key asymmetry (the opportunity)
down/up are **serial around the scan** (down→scan→up, data-dependent), so the
scan's launch-bound **idle SMs are currently wasted — the parallel slot is empty**:
```
time:  [input_proj][downsample][===scan (SMs mostly idle)===][upsample][heads]
slot:                          ↑ empty, nothing fills it ↑
```
The current down/up "handle the cost need" by **shrinking** the scan, NOT by
parallelizing — which is exactly why a parallel detail branch could slot in ~free.

## 7. Solution space

| option | what | cost | risk / precondition |
|---|---|---|---|
| **A. backbone s→m** | add small-object features upstream, doesn't touch head scan length | **heavy**: engine + teacher + full curriculum retrain | AssA −3 needs relink re-tune; memory already shows recall/MOTA win at IDF1 parity |
| **B. ① full-res skip + side-stream** | keep coarse Mamba; skip raw full-res FPN detail to the heads; overlap it into the scan's idle SMs via a side CUDA stream captured in the CUDA-graph | **light**: head-only retrain, reuse cache | only recovers detail that IS in s's P3 (else still need m). **B doubles as the probe for that question.** Overlap must be measured (multi-stream history was GIL-bound at the *pipeline* level — does not directly apply intra-graph) |
| **C. sr=2** | ÷4 → ÷2 | **lightest**: one constructor arg | head latency 3× (~borderline real-time after deflation), eats half the headroom, recovers half the resolution |
| **D. windowed/local Mamba** | full res, bounded per-window scan length | medium: scan change | cuts long-range (but small-object recall wants local detail — acceptable) |

**CLOSED (NO-GO):** d256 widen, delete down/up (sr=1), augmentation-for-recall.

## 8. Open decision points
1. **Is the detail in s's P3 or not?** Watershed: in → B recovers it ~free; not in
   → only A (m) helps. **B is cheap AND answers this.**
2. **Retrain appetite:** B/C are head-only (cheap, reuse cache); A is
   backbone+teacher+head (heavy).
3. **recall vs IDF1:** A reintroduces the AssA tension (needs relink re-tune for parity).
4. **Bet on parallelism?** B's "~free" rides side-stream + CUDA-graph overlap —
   yellow flag, must measure the overlap fraction.

**Provisional lean:** do **B first** — cheap, and one-stone-two-birds: gain recall
if the detail is there, else cleanly prove "detail not in s's P3" and hand the
problem to A (m).

## Artifacts
- `scripts/eval/bench_reduction_bypass.py` — sr / no-down-up latency benchmark.
- `scripts/train/temporal_yolo/run_aug_loo.sh` — parametric LOO (width/holdout/aug).
- `scripts/eval/select_ckpt_by_recall.py` — post-train knee selection by held-out recall.
- `mamba_head.py` `_bypass_reduction` flag (default off, bit-exact) — no-down/up hook.
- d128 LOO runs: `runs/aug_loo/{t1_d128_hoMOT17-10_noaug,_augment}/best_recall.ckpt`.
