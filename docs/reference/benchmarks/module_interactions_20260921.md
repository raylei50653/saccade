# #418 — full-stack leave-one-out subtraction of the production configuration (2026-09-21)

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-09-21 -->
<!-- doc-module: cross -->

## Scope and interpretation

Evidence record for [#418](https://github.com/raylei50653/saccade/issues/418).
It measures, for every mechanism that is on in the production configuration
(`mamba_whole_graph`, ReID off), what happens to the same eval when that one
mechanism is switched off **with everything else left in place**
(`FULL-A`), plus a small pre-declared set of pair removals (`FULL-A-B`),
two reference operating points that name a confound, and the runtime-only
toggles (which must leave the output bit-identical).

It answers only "does removing X from today's stack change the output, and
how". It does not attribute a mechanism to a cause, does not re-tune anything,
does not add a heuristic, and does not use `IDF1 ≥ 80` or any headline score
as a completion bar. Every row is an **observed difference** between two runs
of the same identity; no row is an effect claim. The reading of these rows
into roles lives in
[`docs/architecture/module_interactions.md`](../../architecture/module_interactions.md).

**Not a new number owner.** `FULL` here is the #421 production formal row
`wg:mamba_whole_graph:s.t3t1_phase_b` (commit `50f93a505cc4`,
[training_eval_baselines](../../research/training/training_eval_baselines.md));
this run reproduces its seven MOT files md5-for-md5 at `HEAD`, so #423 may cite
this record for the production row instead of re-running it. The headline
baseline remains [docs/TODO.md](../../TODO.md); the 2026-06-21 `frozen_v2`
numbers there are a different run (older checkpoint identity, double-buffer)
and are not compared against here.

## Pre-declaration

Fixed in `scripts/benchmarks/module_interactions/subtraction.py` (`RULES`,
`VARIANTS`) before the first run; the `report` JSON carries a copy.

- **Identity.** `mamba_whole_graph`, MOT17 train / SDP, seven sequences,
  serial scheduling (no `--double-buffer`), GPU decode, `machine-bench` lease.
  `FULL` ×3; each subtraction, pair and reference ×2; each runtime toggle ×3.
- **Delta direction.** variant − FULL; a negative quality delta means the
  removal hurt. IDs/FP/FN are counts (lower is better).
- **Resolution.** `=` no difference at print precision (0.1 for percent
  metrics, 1 for counts); `≤range` |Δ| within the larger of the two sides'
  same-session repeat ranges; `>range` above it. A repeat range of 0.0 with
  k = 1 distinct output bounds nothing and is not a determinism claim
  ([nogpudecode_reproducibility](../../research/eval/nogpudecode_reproducibility_20260907.md)).
- **Materiality.** |Δ| > 0.2 on IDF1 / HOTA / AssA / MOTA or |ΔIDs| > 10 and
  `>range`. 0.2 is the evidence-ledger decision-knob guidance, not a test.
- **Single-row reading.** `essential`: material loss on IDF1 or HOTA with no
  material gain on IDF1/HOTA/MOTA. `conflict`: the mirror image. `trade_off`:
  material loss on one axis and gain on another. `no_material_quality_role`:
  nothing material (the role then rests on pairs, runtime, safety or code
  dependency).
- **Pair reading.** `I(A,B) = M(FULL) − M(FULL−A) − M(FULL−B) + M(FULL−A−B)`
  on IDF1 and HOTA. `I < −0.2`: substitute / backup pair (removing both loses
  more than the sum). `I > +0.2`: complement / dependency (removing both loses
  less than the sum). Otherwise additive at this resolution.
- **Runtime reading.** `runtime-only` requires all seven MOT md5s equal to
  FULL's and a serial fps delta > 2 % with non-overlapping fps ranges. fps is
  the serial whole-graph profile, not the double-buffer headline throughput.
- **Confounds are carried, not tuned away.** `FULL-OAORamp` keeps the ramped
  `tau = 0.50` (plain OAO shipped at 0.25 → `REF-OAOPlain025` names it);
  `FULL-MultCost` keeps knob values tuned for the multiplicative form, and the
  stability cost has no additive-form code path, so that row removes both.
- **Pairs.** Chosen a priori from suspected overlap, not from the single
  results: Bridge×Interp (both gap recovery), OccState×OAO (both occlusion cost
  terms), StabCost×StabBid (dual stability, re-reads the 2026-07-09 4-way on
  the current identity), GMC×KalmanR (motion model), Bridge×PrivCont
  (identity recovery vs continuation recall, expected orthogonal).

**Amendments made during the campaign** (after the pre-declared rows had
run; listed so they are not mistaken for pre-declared):

1. `RT-NoWholeGraph` / `RT-NoDetectGraph` changed the output (the checkpoint's
   temporal path activated without the whole-graph forward). Two rows with
   `--no-temporal` (`RT-NoWholeGraph-T1`, `RT-NoDetectGraph-T1`) were added as
   the pure scheduling subtraction; the original rows are kept as evidence of
   the coupling.
2. Serial fps drifted downward over the session, so a late same-session `FULL`
   control (`CTRL-FULL-late`, ×3) was added and the runtime reading now also
   requires the row to clear the control's fps range. `FULL-DDA` was
   output-identical, so it was re-run as a runtime row (`RT-NoDDA`, ×3).
3. The `-T1` rows were metric-identical but not md5-identical (≤2 lines per
   sequence differ by 1e-4 in score, graph-vs-eager numerics). A
   `metric-identical` runtime reading was added for that case; it is reported
   as such, not as bit-identity.

The `(dirty)` mark on the commit line below refers to this PR's untracked
docs/harness files in the working tree; no decision-relevant source
(`src/`, `scripts/eval/`, `include/`, `configs/`) differs from `50f93a505cc4`,
which is why the md5 identity check against #421 passes.

## Results

<!-- generated by scripts/benchmarks/module_interactions/subtraction.py report at 2026-09-21T15:39:37+00:00; regenerate rather than edit -->

FULL = `mamba_whole_graph` SDP 7-seq serial at commit `df419b0b7904` (dirty), host `DESKTOP-0FLA6SQ`, GPU `NVIDIA GeForce RTX 5070 Ti Laptop GPU, 616.92`; n=3 runs, k=1 distinct output(s).

FULL readings: HOTA 70.0 · IDF1 78.3 · AssA 69.9 · DetA 70.2 · MOTA 77.9 · IDs 429 · FP 3471 · FN 20940 · fps 191.9 (range 1.09) · mean latency 4.83 ms.

Identity vs #421 production formal run `results/training_eval_contract/wg_mamba_whole_graph_s.t3t1_phase_b/formal-20260919T092203Z-r01` (recipe `wg:mamba_whole_graph:s.t3t1_phase_b`, commit `50f93a505cc4`): MOT md5 equal = **True**.

### Single subtractions (variant − FULL)

| variant | removes | n/k | HOTA | IDF1 | AssA | DetA | MOTA | IDs | FP | FN | fps Δ% | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `FULL-GMC` | GMC | 2/1 | -3.3 (>range) | -4.9 (>range) | -4.6 (>range) | -1.8 (>range) | -2.3 (>range) | +358 (>range) | -208 (>range) | +2369 (>range) | +2.4 | essential |
| `FULL-KalmanR` | KalmanR | 2/1 | -0.6 (>range) | -1.0 (>range) | -1.0 (>range) | -0.1 (>range) | -0.4 (>range) | +51 (>range) | +165 (>range) | +220 (>range) | +1.4 | essential |
| `FULL-OccState` | OccState | 2/1 | -1.1 (>range) | -1.4 (>range) | -2.2 (>range) | +0.1 (>range) | -0.2 (>range) | +19 (>range) | +248 (>range) | -34 (>range) | +0.4 | essential |
| `FULL-OAO` | OAO | 2/1 | -1.3 (>range) | -1.6 (>range) | -2.2 (>range) | -0.3 (>range) | -0.3 (>range) | +9 (>range) | +936 (>range) | -679 (>range) | +0.1 | essential |
| `FULL-OAORamp` | OAORamp | 2/1 | -0.5 (>range) | -0.8 (>range) | -1.1 (>range) | +0.3 (>range) | +0.1 (>range) | +37 (>range) | -389 (>range) | +208 (>range) | +0.5 | essential |
| `FULL-Bridge` | Bridge | 2/1 | -1.4 (>range) | -2.7 (>range) | -2.0 (>range) | -0.6 (>range) | -1.3 (>range) | +94 (>range) | +1292 (>range) | +60 (>range) | -8.0 | essential |
| `FULL-BridgeMargin` | BridgeMargin | 2/1 | +0.0 (=) | +0.1 (>range) | +0.0 (=) | +0.2 (>range) | +0.1 (>range) | -3 (>range) | -81 (>range) | -14 (>range) | -7.0 | no_material_quality_role |
| `FULL-BridgeDirBonus` | BridgeDirBonus | 2/1 | -0.3 (>range) | -0.5 (>range) | -0.5 (>range) | +0.0 (=) | -0.2 (>range) | +11 (>range) | +89 (>range) | +70 (>range) | -1.8 | essential |
| `FULL-Interp` | Interp | 2/1 | -1.0 (>range) | -0.7 (>range) | -0.8 (>range) | -1.1 (>range) | -1.4 (>range) | +380 (>range) | -2318 (>range) | +3461 (>range) | -5.4 | essential |
| `FULL-PrivCont` | PrivCont | 2/1 | +0.0 (=) | -0.3 (>range) | -0.4 (>range) | +0.5 (>range) | +0.2 (>range) | -10 (>range) | -789 (>range) | +516 (>range) | -3.4 | essential |
| `FULL-StabCost` | StabCost | 2/1 | -0.4 (>range) | -0.7 (>range) | -0.9 (>range) | +0.2 (>range) | +0.2 (>range) | -1 (>range) | -199 (>range) | +1 (>range) | -9.9 | essential |
| `FULL-StabBid` | StabBid | 2/1 | -0.5 (>range) | -1.0 (>range) | -1.0 (>range) | +0.2 (>range) | +0.1 (>range) | +71 (>range) | -198 (>range) | +33 (>range) | -6.8 | essential |
| `FULL-DDA` | DDA | 2/1 | +0.0 (=) | +0.0 (=) | +0.0 (=) | +0.0 (=) | +0.0 (=) | +0 (=) | +0 (=) | +0 (=) | -5.4 | no_material_quality_role |
| `FULL-MultCost` | MultCost | 2/1 | -2.2 (>range) | -4.4 (>range) | -4.2 (>range) | -0.2 (>range) | -0.7 (>range) | +126 (>range) | -1288 (>range) | +1915 (>range) | -9.2 | essential |
| `REF-OAOPlain025` | OAORamp | 2/1 | -1.8 (>range) | -2.2 (>range) | -3.1 (>range) | -0.4 (>range) | -1.2 (>range) | +31 (>range) | +1720 (>range) | -406 (>range) | -5.9 | reference |

### Pair subtractions

| variant | n/k | HOTA | IDF1 | AssA | MOTA | IDs | I(IDF1) | I(HOTA) | reading |
|---|---|---|---|---|---|---|---|---|---|
| `FULL-Bridge-Interp` | 2/1 | -1.9 (>range) | -2.9 (>range) | -2.9 (>range) | -1.5 (>range) | +540 (>range) | +0.5 | +0.5 | IDF1 complement_dependency · HOTA complement_dependency |
| `FULL-OccState-OAO` | 2/1 | -1.2 (>range) | -1.5 (>range) | -2.0 (>range) | +0.0 (=) | -5 (>range) | +1.5 | +1.2 | IDF1 complement_dependency · HOTA complement_dependency |
| `FULL-StabCost-StabBid` | 2/1 | -0.3 (>range) | -1.0 (>range) | -0.7 (>range) | +0.1 (>range) | +34 (>range) | +0.7 | +0.6 | IDF1 complement_dependency · HOTA complement_dependency |
| `FULL-GMC-KalmanR` | 2/1 | -2.7 (>range) | -5.0 (>range) | -4.0 (>range) | -2.3 (>range) | +339 (>range) | +0.9 | +1.2 | IDF1 complement_dependency · HOTA complement_dependency |
| `FULL-Bridge-PrivCont` | 2/1 | -2.1 (>range) | -3.7 (>range) | -4.0 (>range) | -0.7 (>range) | +90 (>range) | -0.7 | -0.7 | IDF1 substitute_backup · HOTA substitute_backup |

### Runtime toggles (serial profile unless noted)

| variant | n/k | output identical | fps | fps Δ% vs FULL | fps Δ% vs late control | mean latency ms | p99 ms | reading |
|---|---|---|---|---|---|---|---|---|
| `RT-NoWholeGraph` | 3/1 | False | 51.5 (range 0.20) | -73.2 | -73.7 | 19.02 | 22.468 | OUTPUT DIFFERS (not a pure runtime toggle) |
| `RT-NoDetectGraph` | 3/1 | False | 51.1 (range 3.26) | -73.4 | -73.9 | 19.20 | 22.245 | OUTPUT DIFFERS (not a pure runtime toggle) |
| `RT-NoTrackerGraph` | 3/1 | True | 164.2 (range 5.01) | -14.4 | -16.0 | 5.63 | 7.726 | runtime-only |
| `RT-NoNMSGraph` | 3/1 | True | 160.7 (range 29.55) | -16.3 | -17.8 | 5.88 | 9.952 | runtime-only |
| `RT-DoubleBuffer` | 3/1 | True | 297.1 (range 23.36) | +54.8 | +52.0 | 7.56 | 11.788 | runtime-only |
| `RT-NoWholeGraph-T1` | 3/1 | False | 119.9 (range 4.90) | -37.5 | -38.6 | 7.95 | 10.348 | runtime-only (metric-identical; md5 differs by sub-precision score rounding) |
| `RT-NoDetectGraph-T1` | 3/1 | False | 110.5 (range 3.97) | -42.4 | -43.5 | 8.67 | 11.403 | runtime-only (metric-identical; md5 differs by sub-precision score rounding) |
| `RT-NoDDA` | 3/1 | True | 192.8 (range 4.07) | +0.4 | -1.4 | 4.81 | 5.928 | output-identical, fps unresolved |
| `CTRL-FULL-late` | 3/1 | True | 195.5 (range 0.65) | +1.8 | +0.0 | 4.74 | 5.935 | output-identical, fps unresolved |

fps on the quality rows is **not read**: serial fps drifted during the
campaign (FULL 191–193 early, single rows 167–197, late control 195); only the
runtime rows, which clear both FULL and the late control ranges, carry an fps
reading.

## Reading

### Identity

`FULL` reproduces the #421 production formal run md5-for-md5 (7/7 MOT files)
at `HEAD`; three early runs and three late control runs give one distinct
output (k = 1) and the same metrics as the #421 record (HOTA 70.0 / IDF1 78.3 /
MOTA 77.9 / IDs 429). #423 can therefore cite this record for the production
row. Every variant produced one distinct output over its runs, so every
quality delta below is resolved by the instrument; that says nothing about
determinism in general.

### Single removals

Eleven of fourteen removals are material losses on IDF1 and HOTA with no
material gain elsewhere (`essential` under the pre-declared rule): GMC (−4.9
IDF1, IDs +358), multiplicative cost (−4.4, but see confound), bridge (−2.7,
FP +1292), OAO (−1.6, FP +936), occ_state (−1.4), Kalman R scale (−1.0),
stability bid (−1.0, IDs +71), OAO ramp (−0.8), interpolation (−0.7 IDF1 but
−1.0 HOTA and IDs +380), stability cost (−0.7), bridge direction bonus (−0.5).
Private continuation sits at the materiality edge (−0.3 IDF1 / −0.4 AssA,
IDs −10, FP −789 / FN +516 — a recall-for-precision trade). Two removals have
**no material quality role**: the DDA S0 stage (output bit-identical, all seven
md5s) and the bridge reciprocal margin (+0.1 IDF1 / −3 IDs when off, below
materiality and in the *wrong* direction for a guard).

Confounds carried on rows, not tuned away: `FULL-OAORamp` keeps `tau = 0.50`;
the historical plain-OAO point (`REF-OAOPlain025`, tau 0.25 no ramp) is −2.2
IDF1, so the −0.8 is the conservative reading of the ramp. `FULL-MultCost`
reverts to the additive chain with knobs tuned for the multiplicative form and
necessarily drops the stability cost too; the −4.4 is a reversion cost, not a
fair marginal of the form.

### Pairs

Four of five pre-declared pairs read as **complement / dependency** (positive
`I`): removing the second member after the first is nearly free, so each
member's marginal value exists only while its partner is present —
occ_state × OAO (`I` +1.5 IDF1: 76.9 / 76.7 / 76.8), stability cost × stability
bid (+0.7: the cost contributes 0 once the bid is off, matching the direction
of the 2026-07-09 4-way), GMC × Kalman R (+0.9: R scaling contributes ~0 once
GMC is off), bridge × interpolation (+0.5 on IDF1/HOTA; on IDs the two are
closer to additive: +94 / +380 / +540). Bridge × private continuation is the
one **substitute / backup** pair (`I` −0.7): private continuation is worth 1.0
IDF1 when the bridge is off and 0.3 with it on. No pair showed the
"both harmless alone, harmful together" hidden-dependency pattern.

### Runtime

Tracker graph (−16 % serial fps when off), graphed main NMS (−18 %) and
double-buffer (+52 %) are output-identical and clear both FULL and the late
control: `runtime-only`. The whole-detect graph is **not a pure runtime
toggle in this preset**: with `use_whole_graph: false` the checkpoint's
temporal blocks activate (the whole-graph forward is what bypasses them) and
the output changes wholesale (IDF1 68.7). Pinning `--no-temporal` restores a
metric-identical output (md5 differs on ≤2 lines per sequence by 1e-4 score
rounding, graph-vs-eager numerics) at −39 % serial fps (−44 % fully eager).
`RT-NoDDA` is output-identical and its fps (192.8, range 4.1) overlaps both
FULL and the late control: the DDA stage has neither a quality nor a resolved
runtime role in `FULL`.

### What this does not say

No row is an effect size or a mechanism attribution; every delta is the
observed difference of two runs of one identity on one host. The single-row
losses are not additive (the pair rows show why) and must not be summed into a
"contribution ledger". The fps readings are serial-profile deltas at one
operating point; the double-buffer headline throughput and its resource
frontier stay with the #419 closure record. Nothing here changes a preset.

## Reproduction

```bash
.venv/bin/python tools/resctl.py run --wait machine-bench -- \
  .venv/bin/python scripts/benchmarks/module_interactions/subtraction.py run \
    --root results/module_interactions_418/20260921
.venv/bin/python scripts/benchmarks/module_interactions/subtraction.py report \
  --root results/module_interactions_418/20260921 \
  --json-output docs/reference/benchmarks/module_interactions_20260921.json
```

Raw runs (MOT files, stdout/stderr, per-run records, latency profiles) stay
under `results/module_interactions_418/20260921/` on the measuring host
(not tracked); the tracked artifact is the report JSON beside this file, which
carries every per-run metric, md5 set, fps range and the rules used.
