# D0 — runtime shadow bridge fidelity: results

<!-- doc-status: active -->
<!-- doc-promotion: evidence packet; executed under a sealed §20.2 declaration -->
<!-- doc-date: 2026-07-12 -->
<!-- doc-module: semantic -->

> **Terminal: T2 — PROXY_UNFAITHFUL.** All three predeclared boxes failed, with
> every validity gate passing. The offline proxy `score_m_bridge` is **not** a
> faithful stand-in for the live CUDA `bdist`. Issue #112 is answered — in the
> negative.

Declaration (sealed, boxes owner-confirmed 2026-07-12 before any metric was
computed): [d0_runtime_shadow_fidelity_declaration_20260712.md](d0_runtime_shadow_fidelity_declaration_20260712.md)

---

## 1. Validity gate — all pass (this is a real verdict, not UNRESOLVED)

| Gate | Result |
| --- | --- |
| V1 shadow provenance + Run B byte-identical to Run A (7 seq) | **PASS** (3rd independent run) |
| V2 capture complete, `overflow_events == 0` | **PASS** |
| V3 id-map injective per seq; zero keyed rows with global id −1 | **PASS** |
| V4 partition conserved `1684 + 539 + 354 == 2577` | **PASS** |
| V5 matched N = 1,684 (≥ 1,000); 0 NaN in required columns | **PASS** |
| V6 all four frozen input hashes reproduce bit-for-bit | **PASS** |

## 2. Fidelity verdict — on the 1,684 `matched` pairs only

| Box | Bar | Observed | Result |
| --- | --- | --- | --- |
| **B1** decision agreement @ `≤ 0.4` | ≥ 99 % | **95.07 %** | **FAIL** |
| **B2** \|Δ\| q95 | ≤ 0.05 | **1.4171** | **FAIL** |
| **B3** Spearman ρ | ≥ 0.98 | **0.9558** | **FAIL** |

All three fail. The boxes are non-compensatory, so any one of these alone is
terminal; all three failing removes any ambiguity.

**B1 — threshold transfer.** 83 of 1,684 decisions flip; the 99 % bar allowed at
most 16. The disagreement is roughly symmetric and must not be netted:

| | kernel accepts | kernel rejects |
| --- | ---: | ---: |
| **proxy accepts** | 161 | **46** |
| **proxy rejects** | **37** | 1,440 |

46 pairs the proxy would accept, the kernel rejects; 37 the kernel accepts, the
proxy rejects. Against only 161 true joint accepts, the off-diagonal is large:
**the proxy misclassifies about a third as many pairs as it correctly accepts.**

**B2 — numeric calibration.** |Δ| q95 = 1.417 is **3.54× the entire 0.4
threshold**, and even the median |Δ| is 0.148 — over a third of the threshold.
Δ median = −0.035, IQR [−0.261, +0.079]: the error is not a small bias, it is a
wide dispersion.

**B3 — rank transfer.** ρ = 0.9558 < 0.98. The proxy does not preserve the
kernel's ordering to the standard required for existing ranking-class studies to
survive intact.

## 3. Mechanism (component attribution — diagnostic, did not move the terminal)

| Component | \|d\| median | \|d\| q95 | ρ |
| --- | ---: | ---: | ---: |
| `dist_h` (static geometry) | 0.104 | 0.745 | **0.9909** |
| `bwd_r` | 0.203 | 1.134 | 0.9551 |
| `fwd_r` | 0.183 | **2.020** | 0.9292 |
| `w` (speed weight) | 0.034 | 0.356 | **0.8656** |

Fidelity degrades **monotonically with how much temporal reduction the term
requires**: the 0th-order distance term transfers best, first-order
(velocity-extrapolated) terms are worse, and the pure speed weight — which has no
positional anchor to stabilize it — is worst.

### 3.1 The two quantities are not the same random variable

Both sides evaluate the *same* function
`f(state) = w·½(fwd_r + bwd_r) + (1 − w)·dist_h`. But `state = R(trajectory)`,
and the **temporal-reduction operator `R` differs**:

| | kernel `R` | offline `R` |
| --- | --- | --- |
| samples | foot-ring, stride-3, last 4 | per-frame MOT rows |
| velocity | `bridge_vel4` OLS slope | window-mean difference |
| anchor | adaptive edge-weighted (top/bottom edge trajectories weighted by inverse residual) | raw endpoint |
| scale | EMA height (α = 0.05, causal) | raw endpoint height |
| horizon | `la` | offline frame gap (`gap = la − bridge_at + 1`) |

So `s0 = f(R_off(x))` and `bdist = f(R_ker(x))`. **The shared `f` creates the
illusion of one quantity; the differing `R` makes them two.** This is the general
lesson, not a bridge-specific quirk — see the
[runtime-quantity fidelity protocol](../../../research/eval/runtime_quantity_fidelity_protocol.md).

### 3.2 Two reduction errors, with different signatures

Binning the matched events by extrapolation horizon `la` separates them:

| `la` bin | n | median \|Δ `dist_h`\| | median \|Δ `fwd_r`\| |
| --- | ---: | ---: | ---: |
| [0, 6) | 305 | 0.1003 | 0.1398 |
| [6, 11) | 326 | 0.1096 | 0.1659 |
| [11, 16) | 310 | 0.1010 | 0.2038 |
| [16, 22) | 346 | 0.1061 | 0.2144 |
| [22, ∞) | 397 | 0.1002 | 0.1997 |
| | | ρ(`la`, \|Δ\|) = **−0.012** | ρ(`la`, \|Δ\|) = **+0.099** |

**① Scale operator (EMA vs raw height) — a horizon-independent floor.**
`dist_h` is 0th-order and needs no extrapolation, so it *should* transfer
perfectly. It does not: its error sits at ≈ 0.10 in **every** horizon bin, flat
(ρ = −0.012). The residual comes from the `h_ref` normalizer. The consequence is
sharp — **the two quantities do not converge even as `gap → 0`.** There is no
region where they agree. Against the 0.4 threshold, that floor alone is 25 %.

**② Velocity operator (foot-ring OLS vs window-mean) — amplified by horizon.**
`fwd_r`'s error grows with `la` (0.140 → 0.214, ρ = +0.099), exactly the
signature of `fwd = lost + v·horizon`: a velocity error multiplied by ~14 frames.
This is why `fwd_r` carries the worst tail (q95 = 2.02).

### 3.3 Corollary — why re-fitting is the wrong repair

The divergence is **systematic, not noise**: it does not shrink with more data,
and it has no vanishing limit. Fitting a new proxy to agree with `bdist` would
merely produce a *third* quantity `f(R_fit(x))` — well-correlated, perhaps, but
still not production's reduction. **Only reproducing `R` itself works**, which is
why a faithful replay must share or line-by-line replicate the estimator rather
than approximate it.

## 4. Join validity (checked, because a bad join would fake this result)

The two sides' `gap` columns disagree on 100 % of matched rows, which would be
alarming if it were not a pure convention difference. It is:

* kernel: `gap = la − bridge_at + 1` (`bridge_at = 4`) — verified against the
  raw rows (`16 = 19−4+1`, `26 = 29−4+1`, `5 = 8−4+1`).
* offline: `gap = cand_first_frame − lost_last_frame`.

The join-validity statistic is therefore runtime `la` vs the offline frame gap:

* `|la − offline_gap| ≤ 1` on **93.7 %** of matched events; Spearman **0.9651**.

The join addresses the **same tracks at the same times**. The verdict is not a
join artifact.

## 5. Coverage verdict — would have passed, which *strengthens* T2

| Criterion | Result |
| --- | --- |
| C1 matched share | 65.35 % of all captured; 75.75 % of mappable |
| C2 bias: matched vs `cohort_gap` (`bdist`) | KS = 0.075 — nearly identical (medians 1.674 vs 1.667) |
| C2 bias: matched vs `unemitted` (`bdist`) | KS = 0.271 — `unemitted` are far-apart, high-`bdist` proposals (median 4.011) the tracker never emitted |
| **C3 accept region (`bdist ≤ 0.4`)** | **matched 63.67 % vs 65.35 % overall — only −1.68 pp** |

C3 is the criterion that mattered, and it is clean: of the 311 runtime proposals
in the production-accept region, the matched set's share (63.67 %) is
essentially its overall share (65.35 %). **The offline cohort is representative
of the region where the bridge actually fires.**

This does not rescue the fidelity result — the boxes are non-compensatory — but
it removes the most attractive escape from it. The proxy did not fail because it
was measured on an unrepresentative slice. It failed on the slice that counts.

## 6. Terminal and mainline transition (§20.7)

**T2 — PROXY_UNFAITHFUL.** `score_m_bridge` is not a valid stand-in for
consumer-A `bdist`. Per the sealed declaration, every prior study that used it
as a proxy inherits an explicit, recorded validity limit.

**Scope caveat (unconditional, per declaration §9):** the study's scope is the
7-sequence MOT17-SDP `mamba_whole_graph_m` bridge-off substrate. This terminal
establishes nothing about a bridge that *commits* (shadow suppresses the commit
by construction), nor about any other detector, preset, or sequence set.

### Consequences for already-accepted evidence

The prior docs already carried an *unquantified* "estimator-shifted; no
deployment-facing numeric claim" caveat. #112 converts that caveat into a
measurement, and the measurement is worse than the caveat implied:

1. **Door-0 ranking probe / step ⑤ class closure (#136).** Its baseline ordering
   was `s0 = score_m_bridge`. With ρ(s0, bdist) = 0.956 < 0.98, the closure
   `NO_USABLE_RANKING_POWER_IN_CLASS` is established **against the offline
   ordering, not production's ordering**. It therefore does not establish the
   absence of ranking power over the production `bdist` ordering. This is a
   **scoping limit, not a reversal** — the 12-member class remains closed *with
   respect to s0* — but the closure may no longer be read as a statement about
   the production score. **Owner decision required.**
2. **`m_b1` gate-coverage claims.** Any production-shaped gate coverage computed
   on `s0` mislabels ≈ 5 % of accept/reject decisions at the 0.4 threshold, with
   the error concentrated in exactly the marginal band a gate is about.
3. **The sealed 2026-07-11 reconstruction packet** is unaffected and stays
   frozen: it already fails closed and never claimed runtime capture.

### What is now unblocked

The shadow bridge is the first mechanism that separates *observing* a bridge
proposal from *changing* the tracking result. Any future consumer-A study can
now be run against real float32 kernel values instead of an offline
reconstruction — which is what #112 existed to provide.

## 7. Artifacts

```
out/signal_study/d0_runtime_shadow_fidelity_20260712T085642Z/
  pairs.csv                     ee2898a25ef7f01e…   (24,346 pairs, 339 GT)
  capture.csv.gz                96093b9b723ed450…   (2,577 events, v2 global key)
  capture.csv.gz.manifest.json                      (partition + provenance)
  fidelity_metrics.json                             (F1–F3, coverage)
results/MOT17_eval_d0_shadow_substrate_20260712T085642Z/
  MOT17-*.txt                   4c5e322a3b8c026d…   (7-seq, bridge-off)
  _global_id_map.txt            ae3b6441d1712bcc…
```
