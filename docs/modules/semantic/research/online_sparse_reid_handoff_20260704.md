# Online Sparse ReID Handoff — 新對話接續摘要（2026-07-04）

## 目標

把 ReID 從「每 frame / 每 detection」改成線上可承受的 sparse key-crop job system：

```text
少量 clean/key embeddings
+ per-track FIFO memory
+ async/idle-budget ReID
+ Cheb-GR handover/relink 主導
```

核心問題不是 offline IDF1 再多 +0.01，而是：

```text
ReID jobs 壓到 online 可跑
同時保留 dense bank handover/relink 的大部分效果
```

## 目前結論

### 1. 只用 key embeddings 近似 dense bank：可用

在 frozen no-ReID substrate 上，dense reference 是 `spread-50` bank。

| strategy | m IDF1 | m Δ vs dense | s IDF1 | s Δ vs dense | 判斷 |
|---|---:|---:|---:|---:|---|
| dense ref / spread-50 | 80.219 | - | 78.825 | - | reference |
| recent-20 | 80.286 | +0.067 | 78.877 | +0.051 | 最穩 |
| stridefifo-3-20 | 80.173 | -0.046 | 78.792 | -0.034 | 線上初版可用 |
| stridefifo-5-20 | 80.154 | -0.064 | 78.614 | -0.211 | s 開始不穩 |
| stridefifo-10-20 | 80.234 | +0.015 | 78.575 | -0.250 | 不穩 |

結論：

```text
線上初版：clean FIFO-20 + stride=3 + birth head dense window
不要 mean prototype
不要 EMA 全部樣本
bank 存 raw unique embeddings
```

### 2. ReID job 壓縮量

以「每個 detection 都跑 ReID」為 100%：

| schedule | m jobs | m reduction | s jobs | s reduction |
|---|---:|---:|---:|---:|
| all detections | 96,238 | 1.0x | 94,913 | 1.0x |
| stride-3 + birth head | 26,766 | 3.6x less | 26,490 | 3.6x less |
| stride-5 + birth head | 16,979 | 5.7x less | 16,915 | 5.6x less |
| stride-10 + birth head | 9,635 | 10.0x less | 9,749 | 9.7x less |

Interpretation:

```text
stride-3 ≈ 28% of per-detection ReID cost, effect near dense
stride-5 ≈ 18% cost, but s drops -0.21 IDF1
```

所以若 online budget 允許，先上 `stride=3`。若必須低於 20%，測：

```text
base stride=5
+ event override:
   pre-occlusion snapshot
   handover/newborn head dense 5 frames
   ambiguous top1/top2 candidate query
```

### 3. Direct key gate 可當 veto，但不是主收益

已接入正式管線但 default off：

```text
key_sim_min
key_sim_cost_floor
key_margin_min
```

實測：

| policy | m IDF1 | s IDF1 | 判斷 |
|---|---:|---:|---|
| control | 80.22 | 78.83 | GO |
| `key_sim_min=0.54` | 80.24 | 78.84 | 小幅正向 |
| `key_margin_min=0.03` | 80.21 | 78.82 | 幾乎無效 |
| `key_sim_min=0.54 + center_dist_veto=0.756` | 80.01 | 78.77 | 不可用，傷 MOT17-10 |
| `key_sim_min=0.54 + key_sim_cost_floor=0.25/0.30` | 80.24 | 78.84 | 最合理候選，但收益小 |

結論：

```text
key_best_sim / key_margin 適合 log、veto、abstain
不適合目前當 accept 主邏輯
confirm/rescue 方向暫不成立
```

## 已完成實作入口

### Sparse bank / key gate

- `src/saccade/perception/eval/clean_fifo_bank.py`
  - `CleanFifoBank`
  - metadata/query/query_all
  - FIFO + stride planning helpers

- `src/saccade/perception/eval/cheb_gr_online.py`
  - `causal_handover_lines(...)`
  - direct key-bank metrics:
    - `key_best_sim`
    - `key_mean_topk_sim`
    - `key_margin`
    - support counts
  - gates:
    - `key_sim_min`
    - `key_sim_cost_floor`
    - `key_margin_min`

- `scripts/eval/run_offline_handover_ablation.py`
  - frozen substrate handover replay
  - variants:
    - `bank_mode=recent`
    - `bank_n=20`
    - `key_sim_min=...`
    - `key_sim_cost_floor=...`
    - `key_margin_min=...`
  - runner optimization:
    - GT cache
    - in-memory scoring
    - `--no-write-results`
    - `--no-score`

- `scripts/eval/diagnostics/probe_sparse_bank_equivalence.py`
  - sparse bank equivalence probe
  - strategies:
    - `recent-N`
    - `stridefifo-K-N`
    - `spread-N`
    - `mean1`
    - `segmean-K`
    - `topscore-N`
    - `lowpol-N`

### Online telemetry

- `src/saccade/perception/online_telemetry.py`
  - CPU/GPU sampler
  - rolling stage latency summary

- `src/saccade/perception/dispatcher.py`
  - WorkbenchPool telemetry:
    - queue wait
    - preprocess
    - workbench process
    - worker total
  - AsyncDispatcher telemetry:
    - batch build
    - infer
    - track

- `scripts/ops/run_8stream_perception.py`
  - `--online-telemetry`

Run:

```bash
uv run python scripts/ops/run_8stream_perception.py \
  --workbench \
  --streams 8 \
  --online-telemetry
```

## Important caveats

1. Existing sparse evidence is from offline/frozen substrate handover replay.
2. It validates appearance bank sparsity, not yet a production async ReID worker.
3. `stridefifo-3-20` is the best validated online-like schedule.
4. `stride-5` needs event override before it is safe.
5. `center_dist_veto=0.756` is not safe despite looking good in accepted-known search.
6. Workbench telemetry is Python-level. C++ internal split of detect wait / NMS / tracker needs extension-side profile hooks.

## Experiment B result (2026-07-04, same day): stride-5 + event overrides = GO

Ran on both frozen substrates (`diag_m/s_no_reid_current_20260704`, 7-seq) with
`probe_sparse_bank_equivalence.py` extended with production-scheduler and
event-override strategies:

- `schedfifo-K-N`: production `CleanFifoBank` scheduler (dense birth window
  **into the FIFO** + stride-K after), no overrides.
- `evfifo-K-N[-ambP]`: schedfifo + pre-occlusion/death snapshot (last clean
  crop before a dirty run / frame gap / death is force-extracted); `-ambP`
  adds the ambiguity trigger (force-extract clean crops with same-frame
  neighbor IoU >= P/100).

| strategy | m dIDF1 | m jobs% | s dIDF1 | s jobs% | s jaccard |
|---|---:|---:|---:|---:|---:|
| recent-20 (anchor) | +0.07 | 78.7% | +0.05 | 78.3% | 0.724 |
| stridefifo-3-20 | -0.05 | 27.7% | -0.03 | 27.7% | 0.609 |
| stridefifo-5-20 | -0.06 | 17.6% | **-0.21** | 17.7% | 0.456 |
| schedfifo-5-20 | -0.03 | 17.6% | -0.07 | 17.7% | 0.655 |
| **evfifo-5-20** | **-0.04** | **18.6%** | **+0.03** | **18.9%** | 0.671 |
| evfifo-5-20-amb30 | -0.05 | 25.6% | +0.04 | 26.2% | 0.667 |
| evfifo-3-20 | +0.01 | 28.5% | +0.00 | 28.7% | 0.689 |

Findings:

1. **GO: `evfifo-5-20` hits the <20% target on both substrates** (18.6% / 18.9%
   of per-detection jobs) with aggregate IDF1 within ±0.04 of the dense
   reference. The pre-occlusion/death snapshot costs only ~1pp of jobs and
   repairs the stride-5 accept set (s jaccard 0.456 → 0.671; s-05 −2.37 →
   +0.28).
2. **Half of the published stride-5 drop was probe-slicing artifact.** The
   production scheduler alone (`schedfifo-5-20`, birth window dense into the
   FIFO) already recovers s −0.21 → −0.07; the old `stridefifo` strategy
   sliced `items[::5][-20:]` post-hoc and dropped birth-window rows for short
   tracks.
3. **Ambiguity override is dead weight for the bank**: +7pp jobs, no IDF1 or
   accept-set gain on either substrate (consistent with the earlier findings
   that crossing-adjacent crops carry no extra bank value). Drop it.
4. **Residual: s-09 loses its single handover under any stride-5 variant**
   (−0.96 on that short seq; aggregate impact already inside the +0.03).
   Diagnosis: one borderline decision whose Cheb-GR cost lands ~0.004 above
   the gate with a stride-5 bank; FIFO-30 does not recover it (not a bank
   capacity issue), stride-4 does (`evfifo-4-20`, but ~22% jobs). This is the
   known condition-sensitive-accept fragility, not an event-override failure.

Operating recommendation:

```text
online sparse ReID schedule = evfifo-5-20
  (CleanFifoBank fifo_n=20 stride=5 decide_n=5
   + preocc/death snapshot force-extraction; no ambiguity trigger)
~19% of per-detection ReID cost, m/s aggregate parity with dense bank.
Fallback if the s-09-style borderline loss matters: stride=4 (~22% jobs)
or stride=3 (~28% jobs, +0.00/+0.01).
```

Implementation: `CleanFifoBank.should_extract(..., force=)` +
`plan_clean_fifo_crops(preocc_snapshot=, ambiguity_iou=)` + `job_crops` in
`src/saccade/perception/eval/clean_fifo_bank.py`; probe strategies + per-
strategy job accounting in `probe_sparse_bank_equivalence.py`. Results JSON:
`results/probe_sparse_bank_eventfifo_{m,s}_20260704.json`.

Online causality note: the snapshot trigger is 1-frame-delayed — keep the last
clean crop per track buffered; when the next frame turns dirty/lost (or the
track dies), submit the buffered crop as the ReID job.

## Experiment C result (2026-07-04, same day): death-window w3 upgrade, then the wall

Follow-up sweep on the same substrates. New planner knobs: `preocc_window=W`
(last W clean crops before each dirty-run/gap/death, online = W-slot ring
buffer flushed on the event) and `postocc_snapshot` (first clean crop after a
dirty run / gap). Strategy tokens: `evfifo-K-N[-wW][-po]`.

| strategy | m dIDF1 | s dIDF1 | jobs% (m/s) | note |
|---|---:|---:|---|---|
| **evfifo-5-20-w3** | **+0.03** | **+0.07** | 20.4 / 20.9 | s per-seq 0/7 negative, s-09 fixed |
| evfifo-5-20-po | -0.04 | +0.04 | 19.2 / 19.5 | po alone ~no effect |
| evfifo-5-20-w5 | +0.07 | +0.07 | 21.9 / 22.5 | s accept jaccard drops to 0.645 |
| evfifo-5-30-w3 | -0.09 | +0.11 | 20.4 / 20.9 | FIFO depth diverges m vs s |
| evfifo-5-40-w3 | -0.12 | +0.09 | 20.4 / 20.9 | 〃 |
| evfifo-7-20-w3-po | -0.30 | +0.08 | 16.6 / 17.2 | m-04 -0.59, m-11 -0.87 |
| evfifo-10-20-w3-po | -0.07 | +0.08 | 13.4 / 14.1 | m-11 -0.87 |

Findings:

1. **Death-window w3 is the last transferable upgrade.** `evfifo-5-20-w3` is
   aggregate-positive on both substrates (+0.03 / +0.07) at ~20.5% jobs,
   fixes the s-09 borderline loss, and is 0/7-negative per-seq on s
   (m worst case −0.25). It costs ~1.7pp of jobs over w1.
2. **Below stride-5 the m substrate breaks non-monotonically** (stride-7 worse
   than stride-10; the damage is 2 borderline flips on m-04/m-11), while s
   holds parity down to stride-10 at 14.1% jobs. Aggressive strides are not
   transferable — do not chase the s-only 14% number.
3. **FIFO depth (30/40) and w5 diverge between m and s** — each helps one
   substrate and hurts the other (or degrades accept fidelity). At parity
   level the residual deltas are individual borderline Cheb-GR accepts whose
   cost jitters ~±0.005 with bank composition: the same
   condition-sensitive-accept wall as the applicability map. Stop tuning
   schedule knobs on this replay; further gains need decision-side work,
   not bank-side.

### Attribution: why m breaks where s holds (decision-log analysis)

Hypothesis tested (probe `--dump-decisions`,
`results/probe_sparse_decisions_{m,s}_20260704.json`): the m tracker is
stronger, so its *remaining* handover events are harder. Verdict: right
direction, sharper mechanism —

- **Distribution**: m's accepts are not wholesale harder (headroom median
  0.122 vs s 0.101) but have a fatter tight tail (p25 0.032 vs 0.047;
  headroom<0.05 = 35% vs 27%, n.s. at these counts).
- **Flip frequency**: under identical sparsity, s flips 2-3x MORE decisions
  than m (lost 14-15/new 8-12 vs lost 5-7/new 4-8) yet stays +0.08 — s's
  many borderline accepts are individually low-value and flips cancel.
- **Damage structure**: all of m's aggregate damage is 1-2 named high-payoff
  flips. m-11 −0.87 = one false NEW accept (nb=566, ref rejected at
  headroom −0.043) admitted by sparse jitter; w3 happens to keep it out
  (+0.52), stride-7/10 re-admit it. m-04 −0.59 = one ref accept at
  headroom +0.008 dropped by stride-7 phase.

So m's pool is **fewer, higher-stakes decisions** (51 accepts carrying an
80.2 substrate vs s's 67 on 78.8): the law of large numbers that protects
s's aggregate does not operate on m. "Not transferable" means "no
statistical averaging exists on m", not "the knobs act differently".

Actionable corollary (new decision-side line, does not touch the
hardened-accept NO-GO): flips live exclusively in the |cost−0.45| ≲ 0.05
band. A **borderline dense re-query** — when a handover decision lands in
that band, trigger one dense re-extraction of the candidate bank before
deciding — would cost ~0 jobs (the band is a handful of events per
sequence) and removes the phase-luck exposure entirely.

## Experiment D result (2026-07-04, same day): borderline re-query = GO, enables stride-10

Implemented in `causal_handover_lines(requery_bank_embs=, requery_band=,
requery_top=)`: a decision is *flippable* iff a band-sized perturbation could
change a gate outcome (|best_cost − max_cost| ≤ band, or cost (nearly)
passing and |margin − required| ≤ band). Flippable decisions are re-scored
with a dense fallback bank before gating. Probe flags: `--requery-band`,
`--requery-top`, `--requery-source {ref,recent}`.

| config (band 0.05) | m dIDF1 | s dIDF1 | requeries m/s | note |
|---|---:|---:|---|---|
| full-archive ref fallback | +0.01…+0.02 | −0.01…+0.03 | 56 / 85 | every stride 5/7/10 → ref parity |
| top-2 ref fallback | +0.01…+0.02 | −0.02…+0.04 | 〃 | conservative (drops IDF1-neutral borderline accepts) |
| **top-2 recent-tail fallback** | **+0.06** | **+0.07** | 〃 | online-feasible form; best of all |

Key facts:

1. **Stride-phase sensitivity is eliminated**: with re-query, m's per-seq
   deltas are *identical* at stride-5 and stride-10 — decisions no longer
   depend on bank-composition luck. The m-04/-11 breakers are gone.
2. **The online-feasible fallback (recent-20 tail, what a raw-crop ring
   buffer can reconstruct) beats the offline spread-50 fallback** —
   consistent with recent-N being the strongest bank form throughout.
3. Trigger discipline matters: a naive band test (margin without the
   cost-passing precondition) fires on 80% of events; the flippable-only
   trigger fires on ~18% of *scored decisions* = **~8-12 events/seq**,
   re-extracting ≤ 2 candidate tails each (~60 crops/event ⇒ ~3.4% m /
   ~5.3% s of detections, memoizable per dead track).

### Final operating recommendation (supersedes Experiment C's)

```text
schedule  = evfifo-10-20-w3-po   (stride=10, fifo 20, birth-window dense,
                                  3-crop pre-occ/death ring buffer, post-occ snapshot)
decision  = borderline re-query  (band 0.05, top-2 candidates,
                                  recent-tail fallback from raw-crop ring buffer)
cost      ≈ 13.4% + 3.4% (m) / 14.1% + 5.3% (s) ≈ 17-19% of per-detection ReID
accuracy  = m +0.06 / s +0.07 vs dense bank; per-seq m 1/7 neg (−0.25), s 0/7
```

If the raw-crop ring buffer is unwanted, fall back to Experiment C's
`evfifo-5-20-w3` (~20.5% jobs, +0.03/+0.07, no re-query machinery).

Results JSON: `results/probe_sparse_bank_requery{,_top2,_recent}_{m,s}_20260704.json`.

Final operating recommendation:

```text
evfifo-5-20-w3  (fifo_n=20, stride=5, decide_n=5,
                 preocc_window=3 death/occlusion ring buffer,
                 no postocc, no ambiguity trigger)
~20.5% of per-detection ReID jobs, m +0.03 / s +0.07 vs dense bank.
Hard <19% budget: drop to w1 (evfifo-5-20, −0.04 / +0.03).
```

Results JSON: `results/probe_sparse_bank_event{fifo,win}_{m,s}_20260704.json`,
`results/probe_sparse_bank_fifocap_{m,s}_20260704.json`.

## Suggested next experiment (original plan, superseded by the result above)

Do not tune key thresholds first. Test whether stricter sparse ReID budget can stay usable:

```text
Experiment A:
  stridefifo-3-20
  baseline online-like schedule

Experiment B:
  stridefifo-5-20
  + dense birth head window
  + pre-occlusion snapshot override
  + ambiguous candidate override

Measure:
  ReID jobs / frame
  memory coverage per active track
  handover candidate with key support %
  IDF1 / IDs vs dense ref
  online p95 latency with --online-telemetry
```

Expected:

```text
stride=3: likely safe, about 28% ReID jobs
stride=5 + event override: target <20% jobs, unknown accuracy
```

## Reproduce key commands

Sparse bank probe:

```bash
uv run python scripts/eval/diagnostics/probe_sparse_bank_equivalence.py \
  --substrate results/diag_m_no_reid_current_20260704 \
  --out-json results/probe_sparse_bank_stridefifo_m_20260704.json \
  --strategies ref recent-20 stridefifo-3-20 stridefifo-5-20 stridefifo-10-20
```

Handover ablation:

```bash
uv run python scripts/eval/run_offline_handover_ablation.py \
  --substrate results/diag_m_no_reid_current_20260704 \
  --variant key054_cf030 key_sim_min=0.54 key_sim_cost_floor=0.30 \
  --no-write-results
```

MOT17 eval entry:

```bash
uv run python scripts/eval/mot17.py \
  --preset mamba_whole_graph \
  --detector SDP \
  --output results/mot17_run
```

Online telemetry:

```bash
uv run python scripts/ops/run_8stream_perception.py \
  --workbench \
  --streams 8 \
  --online-telemetry
```

## One-line handoff

~~Use `clean FIFO-20 + stride=3 + dense birth head` as the first online sparse ReID target (~28% jobs).~~ **Answered 2026-07-04 (Experiments B/C/D): the online sparse ReID form is `evfifo-10-20-w3-po` + borderline re-query (band 0.05, top-2 candidates, recent-tail fallback) — ~17-19% of per-detection ReID jobs total, m +0.06 / s +0.07 vs the dense bank, stride-phase sensitivity eliminated (m per-seq identical at stride 5 and 10). Simpler no-re-query fallback: `evfifo-5-20-w3` (~20.5% jobs, +0.03/+0.07). Next step is porting into the production online path (async ReID worker + raw-crop ring buffer + C++ event hooks).**
