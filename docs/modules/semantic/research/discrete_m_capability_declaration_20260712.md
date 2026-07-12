<!-- doc-status: draft -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-12 -->
<!-- doc-module: semantic -->

# Discrete-\(M\) anchor-propagation capability — **draft** preflight declaration

> **One-line:** `m0_state_capture_v1` records the runtime **lost-side local
> reduction** at every observed frame, and a frozen affine family
> (\(z_{t+1}\approx Mz_t+c\)) is asked one narrow question — **can that state's
> anchor be propagated over 1–8 frames better than the constant-velocity rule
> production already uses?** It is not a score model, not a `bdist` claim, not a
> gate study, not \(e^{A\Delta t}\), and not a production change.

> **⚠️ NOT SEALED.** This document is a **draft**. It becomes binding only at the
> seal event defined in § 12. Until then, **no capture, export, fit, or metric is
> authorized**, and no number in this file may be read from data.

Parent: [score temporal-to-stable-domain charter § 5](score_temporal_to_stable_domain_20260712.md) ·
Navigation: [active thread](../../../research/threads/score_temporal_to_stable_domain_20260712.md) ·
Prior terminal: [R1 results — `R1_FAITHFUL`](r1_temporal_reduction_capture_results_20260712.md) ·
Inherited contract: [R1 capture declaration](r1_temporal_reduction_capture_declaration_20260712.md) ·
Fidelity vocabulary: [runtime-quantity protocol](../../../research/eval/runtime_quantity_fidelity_protocol.md)

## 1. Scope (binding at seal)

| Field | Value |
|---|---|
| Study unit | discrete-\(M\) **anchor-propagation** capability gate (charter § 5) |
| Study id | `discrete_m_capability_20260712` |
| Question | Can the R1 lost-side reduced state \(z^{R}_t\) propagate its **anchor** to \(t+k\), \(k\in\{1,2,4,8\}\), better than the constant-velocity rule, stably and without concentration? |
| Capture contract | `m0_state_capture_v1` (new; per-frame, per-track) |
| Payload schema | `m0_state_capture_payload_v1` |
| Capture mode | native CUDA observation, **read-only** on the unmodified headline production path |
| Default | disabled unless `SACCADE_RESEARCH_M0_STATE_CAPTURE_DIR` is set; no allocation when unset |
| Reduction authority | R1's fail-closed device backend `device_bridge_anchor4` (`--require-device`); host fallback forbidden for the authority packet |
| Configuration | sealed headline Consumer-A config, identical to R1: adaptive anchor (mode 2), rate 0.03, `bridge_at=4`, production threshold 0.4 |
| Support | MOT17 train SDP baseline seven sequences (same support as R1) |
| Labels / GT | **not read**; no GT join exists in this study |
| `bdist` / score / gate / preset / ledger | **not computed**, unchanged, unauthorized |
| \(e^{A\Delta t}\) / \(\log M\) | **forbidden in this unit** (charter § 5) |

The `m0` capture directory is **mutually exclusive** with the R1 and D0 capture
directories in one run: a new payload must never be emitted through an older
sealed contract's field list.

### 1.1 Why a new capture contract (binding precondition)

The sealed R1 packet cannot answer this question and must not be stretched to
try. It exports only the **effective four-sample window** consumed by
`bridge_anchor4` at 2,577 bridge decision events:

| Requirement (charter § 5) | R1 packet |
|---|---|
| horizons 1, 2, 4, 8 frames | only ≤ 3 (four consecutive samples) |
| \(M^k\) power stability over 8 frames | not measurable |
| per-frame \(z_t\) trajectory over all tracks | absent (bridge events only) |
| leave-one-sequence-out on a general track population | biased to bridge-event tracks |

Reading a terminal from the R1 packet would require relaxing charter § 5's
already-frozen readouts after the fact. Hence a new, separately versioned
capture contract with its own validity gates.

## 2. Capture contract `m0_state_capture_v1`

### 2.1 Hook point (exact, non-negotiable)

The runtime writes the per-track foot ring and EMA height in
`update_foot_history_kernel`, **after** the frame's Kalman posterior is available
and **before** `relink_bidir_propose_kernel` reads them in the same frame. The
capture is taken at exactly that point, so a record is **the state the
production bridge would read at frame \(t\)** — not a reconstruction of it.

That kernel updates only slots that are `active`, non-`TRACK_EMPTY`, and
`age == 0` (matched this frame, or freshly spawned); coasting lost tracks keep a
frozen history and are not re-emitted. That runtime rule **defines** this
study's observation universe. The study invents no sample for a frame the
runtime did not observe.

### 2.2 Serialized record

One JSONL record per **observed track-frame**:

| Source | Serialized field | Rule |
|---|---|---|
| identity | `seq`, `frame`, `slot`, `track_uid`, `local_id` | event key = `(seq, track_uid, frame)`; `track_uid` (uint64) is slot-reuse-safe; `slot`/`local_id` are provenance only |
| Kalman posterior | `state8` = `[cx, cy, a, h, vx, vy, va, vh]` | verbatim `states[slot*8 + i]`, float32, post-update |
| causal window | `foot_ring` = up to 8 chronological `(cx, cy, h)`, `foot_len` | verbatim ring **after** this frame's append; unused tail zero-filled, bounded by `foot_len` |
| causal normalizer | `ema_h` | verbatim post-update EMA (alpha 0.05, seeded by first observed `h`) |
| lifecycle | `track_state`, `age`, `time_since_update` | `age == 0` by construction; recorded so the invariant is auditable |
| configuration | anchor mode/rate, `bridge_at`, production threshold, tracker/preset provenance | must equal the sealed R1 configuration under float32 representation |
| provenance | contract id, payload schema, `tracker_gpu.cu` SHA256, preset SHA256, per-file complete/overflow counters | mixed provenance, overflow, or duplicate keys **fail closed** |

No GT, no detection score, no assignment outcome, no post-commit state.

### 2.3 Derived state — the two variants (both frozen before any data are read)

| Variant | Definition | Role |
|---|---|---|
| **\(z^{R}\) (primary; terminal-bearing)** | the **sealed R1 reduction operator** applied to the last four ring samples at frame \(t\) (`bridge_anchor4`, `endpoint_idx = 3`, adaptive mode 2, rate 0.03) → \(z^{R}_t = [a_x,\ a_y,\ v_x,\ v_y,\ e]\): anchor endpoint, OLS per-frame velocity, \(e=\texttt{ema\_h}\) | **the terminal is decided on \(z^{R}\) alone** |
| \(z^{K}\) (companion; diagnostic only) | Kalman posterior motion state \([c_x, c_y, v_x, v_y, h, v_h]\) from `state8` | reported for context; **may not** produce, overturn, or rescue a terminal |

Declaring both now — and binding the terminal to \(z^{R}\) only — removes the
freedom to pick whichever state looks better after the fit.

**What \(z^{R}\) is, precisely.** It is the **lost-side** local reduction: the
exact branch (`bridge_anchor4_last4`, `endpoint_idx = 3`) that production applies
to a *lost* track before scoring. It is **not** "the state `bdist` consumes":
`bdist` is a **pair-level** quantity that additionally consumes the candidate's
head-four reduction (`endpoint_idx = 0`), a two-sided normalizer
\(h_{\mathrm{ref}} = \max((e_{\mathrm{lost}} + e_{\mathrm{cand}})/2,\ 1)\), a
speed weight \(w\), the direct distance `dist_h`, and the real lost age `la`.
None of those enter this study. See § 5.1.

**Fidelity inheritance, stated precisely.** \(z^{R}\) uses the same operator, the
same window shape, the same configuration, and the same fail-closed device
calculator R1 certified. That is **operator reuse under the declared input
shape** — *not* a claim that `R1_FAITHFUL` generalizes to a broader event
universe. No `bdist`, accept predicate, or deployment-facing quantity is computed
or claimed in this unit.

## 3. Frozen pair universe

For a track \(u\) in sequence \(s\), a candidate pair at horizon \(k\) is
\((z_t, z_{t+k})\) with \(t\) and \(t+k\) both **observed frames** of \(u\).
Horizons are frozen at \(k \in \{1,2,4,8\}\) frames.

Every candidate pair is assigned to exactly one bucket by this **frozen
precedence ladder** (first matching rule wins; the ladder is what makes "exactly
one bucket" reproducible):

| # | Bucket | Rule |
|---|---|---|
| 1 | `excluded_not_confirmed` | `track_state != TRACK_CONFIRMED` at \(t\) **or** \(t+k\) |
| 2 | `excluded_short_window` | `foot_len < 4` at \(t\) **or** \(t+k\) (the short-lost fallback has no velocity) |
| 3 | `excluded_non_contiguous` | any intermediate frame \(t+1,\dots,t+k-1\) is not an observed frame of \(u\) |
| 4 | `primary` | everything else |

Rule 3 is not fastidiousness: if the runtime coasted through a gap, the ring at
\(t+k\) still contains pre-\(t\) samples, so the pair is not a \(k\)-step
evolution of the same window; pooling it would silently redefine the horizon.

Bucket counts must sum to the candidate-pair total (partition conservation, gate
V4). Excluded buckets are **reported, never pooled into, and never able to
change** the terminal.

Future state \(z_{t+k}\) is used **only as the prediction target** — never as an
input feature. Folds are split by sequence, so no target leaks across folds.

## 4. Frozen model family (no member may be added after data are read)

All fitted models are affine, \(\hat z_{t+1} = M z_t + c\), and propagate to
horizon \(k\) by **repeated powers**:

\[
\hat z_{t+k} \;=\; M^{k} z_t \;+\; \Big(\textstyle\sum_{j=0}^{k-1} M^{j}\Big) c .
\]

| Family | Parameters (matrix + intercept) | Definition |
|---|---|---|
| `identity` | 0 | \(\hat z_{t+k} = z_t\) — no-dynamics lower bound |
| `const_velocity` | 0 | \(\hat a_{t+k} = a_t + k\,v_t\); \(\hat v_{t+k}=v_t\); \(\hat e_{t+k}=e_t\) — **the baseline the fitted families must beat**; it is also the rule production's `fwd_r` term already applies to this state (§ 5.1) |
| `diag_M` | 5 + 5 = **10** | \(M\) diagonal, plus intercept |
| `full_M` | 25 + 5 = **30** | unrestricted \(M\), plus intercept — tests whether cross-state coupling is real |
| `regime_M` | 3 × 30 = **90** | independent \(M_r, c_r\) per scale regime (§ 4.3) |

### 4.1 Nondimensionalization (fixed, and it is what makes norms meaningful)

\(z\) mixes units (px, px·frame⁻¹, px). Both the solver and the stability gate
operate on a **nondimensionalized** state

\[
\tilde z = D z,\qquad
D = \operatorname{diag}\!\big(1/h_0,\ 1/h_0,\ \Delta t/h_0,\ \Delta t/h_0,\ 1/h_0\big),
\qquad \Delta t = 1\ \text{frame},
\]

where \(h_0\) is the **median of \(e_t\) over that fold's training pairs**
(frozen rule; never computed on held-out data). Because inputs and targets are
scaled by the same \(D\) and there is no regularizer, this is an exact
reparameterization: \(\tilde M = D M D^{-1}\), \(\tilde c = D c\), and
predictions map back exactly. It only fixes conditioning and gives
\(\|\tilde M^k\|_2\) a well-defined meaning.

### 4.2 Solver contract (frozen; a packet must yield a unique terminal)

| Item | Frozen value |
|---|---|
| Precision | **float64** throughout (capture is float32; the fit is not) |
| Design matrix | \([\tilde z_t^\top,\ 1]\) — intercept as an explicit ones column, never centered separately |
| Solver | **SVD least squares** (`numpy.linalg.lstsq`, LAPACK `gelsd`); normal equations and QR are forbidden |
| `rcond` | **`1e-10`** (relative to the largest singular value), passed explicitly — never `None`/library default |
| Rank deficiency | any singular value \(< \texttt{rcond}\cdot\sigma_{\max}\) ⇒ **that family fails that fold** (not a licence to add ridge, drop a column, or re-parameterize) |
| Conditioning | \(\kappa_2 = \sigma_{\max}/\sigma_{\min} > 10^{8}\) ⇒ **that family fails that fold**; \(\kappa_2\) is recorded for every fit |
| Regularization / scaling / outlier rejection | **none** (\(\lambda = 0\)); no robust loss, no iteration budget |
| Minimum training pairs | a fit (per fold; per regime for `regime_M`) requires **≥ 500** one-step training pairs, else that family fails that fold |
| Fit data | one-step (\(k=1\)) **primary** pairs of the training folds only |
| Horizon prediction | powering the **one-step** \(M\) (§ 4). Direct per-horizon refits are **diagnostic only** and cannot bear the terminal |
| Reproducibility | the calculator records numpy/LAPACK/BLAS provenance and must emit byte-identical JSON on a rerun of the same payload (same rule R1 used) |

A family that fails in **any** fold fails as a family: no fold-level substitution.

### 4.3 Regime partition (frozen rule, refit per fold)

Regimes are assigned **causally, from \(z_t\) only**: scale terciles of
\(e = \texttt{ema\_h}\). Boundaries are the empirical 1/3 and 2/3 quantiles
(linear interpolation, i.e. numpy `method="linear"`) of \(e_t\) over **that
fold's training pairs only**; a value exactly on a boundary takes the **lower**
regime index. Boundaries are therefore refit per fold by a fixed rule, never
hand-picked, never computed on held-out data.

## 5. Frozen metric

For horizon \(k\) and primary pair \(i\), the error is the scale-normalized
**anchor endpoint** error:

\[
e^{(i)}_k \;=\; \frac{\big\|\,(\hat a_x,\hat a_y) - (a_x,a_y)\,\big\|_2}{\max(e_t,\ 10^{-3})}.
\]

- Per-fold statistic: **median** over that held-out sequence's primary pairs.
- Aggregate \(\bar e_k\): **median of the seven per-fold medians** (so one long
  sequence cannot dominate).
- Velocity error and scale error \(|\hat e - e|\) are **reported per horizon and
  per component**, and they do **not** enter the decision rule.

Validation is **leave-one-sequence-out**: 7 folds, fit on 6, evaluate on the
held-out sequence.

### 5.1 Claim boundary of the metric and of the 0.40 ceiling (binding)

This metric is a **single-track, lost-side, anchor-position** quantity. Production
`bdist` is a **pair-level** quantity:

\[
\texttt{bdist} = w\cdot\tfrac12\,(\texttt{fwd\_r}+\texttt{bwd\_r}) + (1-w)\cdot\texttt{dist\_h},
\qquad
h_{\mathrm{ref}} = \max\!\Big(\tfrac{e_{\mathrm{lost}}+e_{\mathrm{cand}}}{2},\,1\Big),
\qquad
w=\sqrt{\operatorname{clamp}(s_{\mathrm{lost}}/0.12,0,1)}.
\]

| Production ingredient | In this study? |
|---|---|
| lost-side reduction (last-4, `endpoint_idx=3`) | **yes** — this is \(z^{R}\) |
| candidate-side reduction (head-4, `endpoint_idx=0`) | **no** |
| two-sided normalizer \(h_{\mathrm{ref}}\) | **no** (we normalize by the single-sided \(e_t\)) |
| speed weight \(w\), `dist_h` blend, direction bonus | **no** |
| real horizon `la` (lost age), gap propagation | **no** (we use fixed \(k\in\{1,2,4,8\}\)) |

Therefore **0.40 is declared as a `production_inspired_heuristic_ceiling`, not as
a quantity-equivalent accept margin.** It is the sealed production `bdist`
threshold reused as an *order-of-magnitude* bar in a comparable dimensionless
scale, chosen a priori and never tuned. No terminal of this study may be
described as reaching, missing, or predicting the production accept boundary. A
pair-level, two-sided, `la`-conditioned metric would be required for that, and it
is **not** declared here.

**What the study *is* legitimately relevant to, stated exactly:** production's
`fwd_r` term propagates this very state by the **constant-velocity rule**
(\(\text{anchor} + v\cdot la\)). This study asks whether a stable linear operator
propagates that same state better than that same rule. That is a
representation-capability question about an *ingredient* of the score — not a
claim about the score.

## 6. Frozen validity gates (non-compensatory)

Any failure ⇒ terminal `M0_INVALID`; no capability conclusion may be drawn.

| Gate | Must pass |
|---|---|
| **V1 production neutrality** | the capture-on run's MOT output is **byte-identical** to the capture-off run on all seven sequences |
| **V2 bounded complete capture** | every per-sequence native file reports `complete=true`, `overflow_events=0`, `total_records == len(records)` |
| **V3 version / provenance** | all files declare exactly `m0_state_capture_v1` / `m0_state_capture_payload_v1` with identical provenance; `tracker_gpu.cu` + preset + device-backend + numpy/LAPACK hashes recorded; configuration equals the sealed R1 configuration |
| **V4 partition conservation** | for every `(seq, frame)`, the record count equals the independently counted number of tracks with `active && state != EMPTY && age == 0`; all `(seq, track_uid, frame)` keys unique; the four buckets of § 3 sum exactly to the candidate-pair total |
| **V5 causal completeness** | every record has finite `state8`, `foot_len ∈ [1,8]`, `ema_h > 0`, `age == 0`; every \(z^{R}\) is produced by the fail-closed device backend (`--require-device`); a host-fallback reduction invalidates the authority packet |
| **V6 declared support** | all seven sequences present; **each fold** contributes **≥ 200 primary pairs at every horizon**; every fitted family has **≥ 500** one-step training pairs per fold (per regime where applicable). A thinner packet **fails closed** rather than reporting a fragile median |

## 7. Frozen decision rule (ordered; mechanically decidable)

Terminology: **baselines** \(B=\{\texttt{identity},\ \texttt{const\_velocity}\}\)
(closed-form, always eligible); **fitted** \(\{\texttt{diag\_M}, \texttt{full\_M},
\texttt{regime\_M}\}\).

**G0 — validity.** V1–V6 all pass, else `M0_INVALID`.

**G1 — stability eligibility (fitted families only; evaluated *before* any
ceiling).** A fitted family is **eligible** only if, in **every** LOO fold (and,
for `regime_M`, **every** regime), all of the following hold on the
nondimensionalized operator \(\tilde M\) of § 4.1:

| Sub-gate | Frozen criterion | Why |
|---|---|---|
| G1a asymptotic | \(\rho(\tilde M) \le 1.001\) | no long-run explosion |
| G1b **finite-horizon transient** | \(\|\tilde M^{k}\|_2 \le 2.0\) for every \(k\in\{1,2,4,8\}\) | a non-normal \(\tilde M\) can have \(\rho<1\) and still amplify hugely within 8 steps; charter § 5 asks for the stability of the **repeated powers**, not of the eigenvalues |
| G1c **affine drift** | \(\big\|\big(\sum_{j=0}^{k-1}\tilde M^{j}\big)\tilde c\big\|_2 \le 0.40\) for every \(k\) | the intercept accumulates even when \(\tilde M\) is contractive |
| G1d finiteness | every held-out prediction is finite | — |

Ineligible families are **reported and excluded from G2/G3** — a matrix whose
powers blow up cannot buy a low one-step error and call itself a stable
short-horizon transition. Evaluating eligibility *first* is what stops an
unstable fitted family from carrying the ceiling for everybody else.

**G2 — ceiling, over all four horizons, over eligible families only.** For each
family \(F \in B \cup \{\text{eligible fitted}\}\) define
\(K_{\text{pass}}(F) = \{\,k : \bar e_k^{F} \le 0.40\,\}\) (§ 5.1: heuristic
ceiling, not a production-equivalence claim). Call \(F\) **ceiling-complete** iff
\(K_{\text{pass}}(F) = \{1,2,4,8\}\), and let \(A\) be the set of ceiling-complete
families.

**G3 — real improvement over constant velocity.** An **eligible, ceiling-complete**
fitted family \(F \in A\) is **useful** iff **all three** hold at **every**
\(k \in \{1,2,4,8\}\):

| Sub-gate | Frozen criterion |
|---|---|
| G3a magnitude | \(\big(\bar e_k^{\mathrm{CV}} - \bar e_k^{F}\big)\big/\bar e_k^{\mathrm{CV}} \;\ge\; 0.10\) |
| G3b retention | \(F\) beats `const_velocity` in **≥ 6 of 7** LOO folds (strictly lower per-fold median) |
| G3c concentration | defined exactly below; **≥ 50 %** of the relative gain survives at every \(k\) |

**G3c, in full (no post-hoc freedom):** within each fold \(f\) and horizon \(k\),
let \(n_{f,k}\) be its primary-pair count and
\(\Delta_i = e^{(i),\mathrm{CV}}_k - e^{(i),F}_k\) the per-pair improvement. Drop
the \(d_{f,k} = \lceil 0.10 \cdot n_{f,k}\rceil\) pairs with the **largest**
\(\Delta_i\); ties at the cut are broken by ascending
`(seq, track_uid, t)` and the first \(d_{f,k}\) are dropped. On the **same
retained subset**, recompute the per-fold medians of **both** CV and \(F\), then
the aggregates \(\bar e_k^{\mathrm{CV,drop}}, \bar e_k^{F,\mathrm{drop}}\) by the
same median-of-folds rule. With
\(g_k = (\bar e_k^{\mathrm{CV}} - \bar e_k^{F})/\bar e_k^{\mathrm{CV}}\) and
\(g_k^{\mathrm{drop}}\) defined identically on the dropped-subset aggregates,
require \(g_k^{\mathrm{drop}} \big/ g_k \ge 0.50\) (with \(g_k>0\) guaranteed by
G3a).

G3a's 10 % is declared pre-data as the **decision-relevant** margin: below it, a
30- or 90-parameter operator does not earn its interpretation and deployment cost
over a zero-parameter physical baseline. G3c exists because a gain living
entirely in a small special-scene subset is a scene finding, not a representation
capability.

**Winner selection (only if ≥ 1 family passes G3):** fewest parameters wins; ties
broken in the frozen order `diag_M` → `full_M` → `regime_M`. No other tiebreak.

## 8. Terminal mapping (ordered, disjoint, exhaustive)

Evaluated top-down; the first matching row is the terminal. Every name is scoped
to **anchor propagation**, because that is the only thing the metric measures
(§ 5): no terminal claims velocity or scale predictive sufficiency, and none
claims anything about `bdist`.

| # | Terminal | Condition | Consequence |
|---|---|---|---|
| 0 | `M0_INVALID` | any V1–V6 gate fails | instrumentation repair only; **no** capability conclusion |
| 1 | `ANCHOR_STATE_INSUFFICIENT` | \(1 \notin K_{\text{pass}}(F)\) for **every** \(F \in B \cup \{\text{eligible fitted}\}\) — nothing meets the ceiling even one step ahead | the declared state cannot propagate its own anchor at the declared scale; **no** \(e^{A\Delta t}\), **no** score model on this state; revisit the representation \(R\) |
| 2 | `ANCHOR_SHORT_HORIZON_ONLY` | some family meets the ceiling at \(k=1\), but \(A = \varnothing\) (no family is ceiling-complete over 1/2/4/8) | capability exists only up to the reported \(h_{\max}\); **no** \(e^{A\Delta t}\); any follow-on must be separately declared **and horizon-restricted**. This row is what prevents a good one-step number from being narrated as short-horizon sufficiency |
| 3 | `ANCHOR_M_SUFFICIENT` | \(A \neq \varnothing\) and ≥ 1 eligible fitted family passes G3 | anchor propagation is stable, ceiling-complete, and linear structure beyond CV is real and non-concentrated → **may request** a separately declared \(e^{A\Delta t}\) interpretation study **and/or** a separately declared real-`bdist` score-ranking study (which may **not** assume velocity/scale sufficiency) |
| 4 | `ANCHOR_BASELINE_DOMINANT` | \(A \neq \varnothing\), no fitted family passes G3, and a **baseline** (`const_velocity` or `identity`) is itself ceiling-complete | the state propagates its anchor at the declared scale, but **no declared linear family beats the baseline** → the declared linear class is **closed** on this state; **no** \(e^{A\Delta t}\); a score-ranking study may still be separately declared but **may not** use \(M\). The report must name which baseline dominated |
| 5 | `ANCHOR_M_REQUIRED_MARGIN_INSUFFICIENT` | \(A \neq \varnothing\), no fitted family passes G3, and **no baseline** is ceiling-complete (only fitted operators reach the ceiling, but none clears the 10 % margin) | a fitted operator is *necessary* to reach the ceiling yet *not* decision-relevantly better than CV → **no** \(e^{A\Delta t}\), **no** score model; the gap between "needed" and "not worth it" must be resolved by a **new** declaration, not by relaxing this one |

No terminal authorizes a score fit, a gate or threshold sweep, a preset change, a
ledger entry, a no-go entry, or an online policy evaluation. Research acceptance
remains owner-side after the packet is reviewed.

## 9. Closure class (what a negative terminal does and does **not** exhaust)

A negative terminal (rows 1, 2, 4, 5) closes **only** this class:

```text
affine one-step transitions  {identity, const_velocity, diag_M, full_M, regime_M(ema_h terciles)}
  fitted by the frozen float64 SVD solver contract (§4.2)
  on  z^R  — the LOST-SIDE R1 reduction (adaptive anchor mode 2, rate 0.03, bridge_at=4)
  judged by ANCHOR-POSITION error only
  over MOT17 train SDP 7-seq, observed-contiguous confirmed pairs
  at horizons k ∈ {1, 2, 4, 8}, powered from the one-step fit
```

It **does not** exhaust: non-linear or learned dynamics; other state definitions
(including the \(z^{K}\) companion); velocity or scale predictability; other
regime variables; gap-spanning pairs; longer or `la`-distributed horizons; the
candidate-side or pair-level geometry; other anchors, presets, detectors, or
consumers; anything about `bdist` itself; or the continuous-time generator
(unreachable from a negative terminal by charter § 5).

## 10. Must not

- Fit or read **any** score, weight, gate, or threshold in this unit; do not
  compute `bdist`.
- Describe any terminal as reaching, missing, or predicting the production accept
  boundary (§ 5.1).
- Introduce \(e^{A\Delta t}\), \(\log M\), or a Jacobian study before a
  `ANCHOR_M_SUFFICIENT` terminal.
- Add a family member, a regime variable, a horizon, a tiebreak, a solver option,
  or a regularizer after data are read.
- Swap the terminal onto the \(z^{K}\) companion, or onto velocity/scale error,
  because anchor position disappointed.
- Treat operator reuse as a generalization of `R1_FAITHFUL`.
- Use GT, labels, detection scores, post-commit state, future frames as inputs, or
  sequence-global normalization.
- Change production, default presets, the evidence ledger, or no-go status.

## 11. Planned artifacts (implementation follows the seal, in a separate PR)

```bash
# Capture (default-off; unset ⇒ no allocation, no writes, production path unchanged)
SACCADE_RESEARCH_M0_STATE_CAPTURE_DIR=out/m0-native \
uv run scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP --double-buffer

uv run python scripts/tools/export_m0_state_capture.py \
  --capture-dir out/m0-native --output out/m0-discrete/records.jsonl

uv run python scripts/tools/verify_m0_state_capture.py \      # V1–V6, fail-closed
  --payload out/m0-discrete/records.jsonl \
  --byte-identity-baseline results/<capture-off-run> \
  --output out/m0-discrete/validity.json

uv run python scripts/tools/run_m0_discrete_transition.py \   # frozen solver + terminal
  --payload out/m0-discrete/records.jsonl --require-device \
  --output out/m0-discrete/terminal.json
```

This declaration contains **no results** and is **not** evidence promotion.

## 12. Seal transition (the single authoritative event)

This document is **`doc-status: draft`** and carries **no authority** until all
three of the following exist:

```text
1. owner review comment on the PR containing the literal token:  SEALED
2. an append-only "Seal record" appended to this section (date, PR, head SHA,
   and doc-status flipped draft → active)
3. thread + module TODO transitioned to "sealed; execution authorized"
```

Before that event: **draft**; no capture, export, fit, or metric may be run, and
the study id may not appear in any evidence packet. After it: **append-only** —
gates, families, solver, metric, ceiling, and terminals may not be edited, only
amended by appended record. PR merge alone is **not** the seal (merge ≠ research
acceptance).

### Seal record

*(none — not sealed)*
