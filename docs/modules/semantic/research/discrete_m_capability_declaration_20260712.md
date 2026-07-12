<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-12 -->
<!-- doc-module: semantic -->

# Discrete-\(M\) representation-capability — sealed preflight declaration

> **One-line:** `m0_state_capture_v1` records the runtime Consumer-A local state
> at every observed frame, and a frozen affine family
> (\(z_{t+1}\approx Mz_t+c\)) is then asked one question — **is that state
> sufficient to predict its own short-term future?** — under predeclared
> horizons, held-out folds, and a stability/concentration rule. It is not a
> score model, not a gate study, not \(e^{A\Delta t}\), and not a production
> change.

Parent: [score temporal-to-stable-domain charter § 5](score_temporal_to_stable_domain_20260712.md) ·
Navigation: [active thread](../../../research/threads/score_temporal_to_stable_domain_20260712.md) ·
Prior terminal: [R1 results — `R1_FAITHFUL`](r1_temporal_reduction_capture_results_20260712.md) ·
Inherited contract: [R1 capture declaration](r1_temporal_reduction_capture_declaration_20260712.md) ·
Fidelity vocabulary: [runtime-quantity protocol](../../../research/eval/runtime_quantity_fidelity_protocol.md)

## 1. Seal and scope

**Seal time:** 2026-07-12, before any per-frame state capture is exported, any
transition matrix is fitted, and any error/terminal number is read. This
declaration may only be amended **append-only**. The R1 packet and the D0 CSV
are frozen inputs; neither may be rewritten by this study.

| Field | Sealed value |
|---|---|
| Study unit | discrete-\(M\) representation-capability gate (charter § 5) |
| Study id | `discrete_m_capability_20260712` |
| Question | Is the R1-faithful Consumer-A local state \(z_t\) sufficient to predict \(z_{t+k}\) at \(k\in\{1,2,4,8\}\)? |
| Capture contract | `m0_state_capture_v1` (new; per-frame, per-track) |
| Payload schema | `m0_state_capture_payload_v1` |
| Capture mode | native CUDA observation, **read-only** on the unmodified headline production path (no shadow bridge, no suppressed commit) |
| Default | disabled unless `SACCADE_RESEARCH_M0_STATE_CAPTURE_DIR` is set; no allocation when unset |
| Reduction authority | R1's fail-closed device backend `device_bridge_anchor4` (`--require-device`); host fallback forbidden for the authority packet |
| Configuration | sealed headline Consumer-A config, identical to R1: adaptive anchor (mode 2), rate 0.03, `bridge_at=4`, production threshold 0.4 |
| Support | MOT17 train SDP baseline seven sequences (same support as R1) |
| Labels / GT | **not read**; no GT join exists in this study |
| Score / gate / preset / ledger | unchanged and unauthorized |
| \(e^{A\Delta t}\) / \(\log M\) | **forbidden in this unit** (charter § 5: only after this gate passes, separately declared) |

The `m0` capture directory is **mutually exclusive** with the R1 and D0 capture
directories in one run, for the same reason R1 was made exclusive of D0: a new
payload must never be emitted through an older sealed contract's field list.

### 1.1 Why a new capture contract (binding precondition)

The sealed R1 packet cannot answer this question and must not be stretched to
try. R1 exports only the **effective four-sample window** consumed by
`bridge_anchor4` at 2,577 bridge decision events. It therefore supplies:

| Requirement (charter § 5) | R1 packet |
|---|---|
| horizons 1, 2, 4, 8 frames | only ≤ 3 (four consecutive samples) |
| \(M^k\) power stability over 8 frames | not measurable |
| per-frame \(z_t\) trajectory over all tracks | absent (bridge events only) |
| leave-one-sequence-out on a general track population | biased to bridge-event tracks |

Reading a terminal from the R1 packet would require relaxing charter § 5's
already-frozen readouts after the fact. That is exactly the post-hoc weakening
the predeclaration bar exists to prevent. Hence a new, separately versioned
capture contract, carrying its own validity gates.

## 2. Capture contract `m0_state_capture_v1`

### 2.1 Hook point (exact, non-negotiable)

The runtime writes the per-track foot ring and EMA height in
`update_foot_history_kernel`, **after** the frame's Kalman posterior is
available and **before** `relink_bidir_propose_kernel` reads them in the same
frame. The capture is taken at exactly that point, so a captured record is
**the state the production bridge would see at frame \(t\)** — not a
reconstruction of it.

The kernel updates only slots that are `active`, non-`TRACK_EMPTY`, and
`age == 0` (matched this frame, or freshly spawned). Coasting lost tracks keep
a frozen history and are **not** re-emitted. That runtime rule defines the
observation universe of this study; the study does not invent samples for
frames the runtime did not observe.

### 2.2 Serialized record

One JSONL record per **observed track-frame**:

| Source | Serialized field | Rule |
|---|---|---|
| identity | `seq`, `frame`, `slot`, `track_uid`, `local_id` | event key = `(seq, track_uid, frame)`; `track_uid` (uint64) is the slot-reuse-safe identity; `slot`/`local_id` are provenance only |
| Kalman posterior | `state8` = `[cx, cy, a, h, vx, vy, va, vh]` | verbatim `states[slot*8 + i]`, float32, post-update |
| causal window | `foot_ring` = up to 8 chronological `(cx, cy, h)`, `foot_len` | verbatim ring **after** this frame's append; unused tail zero-filled and excluded by `foot_len` |
| causal normalizer | `ema_h` | verbatim post-update EMA (alpha 0.05, seeded by the first observed `h`) |
| lifecycle | `track_state` (`TENTATIVE`/`CONFIRMED`), `age`, `time_since_update` | `age == 0` by construction; recorded to make that invariant auditable |
| configuration | anchor mode/rate, `bridge_at`, production threshold, tracker/preset provenance | must equal the sealed R1 configuration under float32 representation |
| provenance | capture-contract id, payload schema, `tracker_gpu.cu` SHA256, preset SHA256, per-file complete/overflow counters | mixed provenance, overflow, or duplicate keys **fail closed** |

The record carries **no** GT, no detection score, no assignment outcome, and no
post-commit state. It is a causal snapshot of runtime local state, nothing else.

### 2.3 Derived state — the two \(z\) variants (both frozen now)

Two state definitions are declared **before** any data are read. Only one of
them can produce a terminal.

| Variant | Definition | Role |
|---|---|---|
| **\(z^{R}\) (primary; terminal-bearing)** | apply the **sealed R1 reduction operator** to the last four ring samples at frame \(t\) (`bridge_anchor4`, `endpoint_idx = 3`, adaptive mode 2, rate 0.03) → \(z^{R}_t = [a_x,\ a_y,\ v_x,\ v_y,\ e]\), where \((a_x,a_y)\) is the anchor endpoint, \((v_x,v_y)\) the OLS per-frame velocity, and \(e=\texttt{ema\_h}\) | the state the production score `bdist` actually consumes; **the terminal is decided on \(z^{R}\) alone** |
| \(z^{K}\) (companion; diagnostic only) | Kalman posterior motion state \([c_x,\ c_y,\ v_x,\ v_y,\ h,\ v_h]\) from `state8` | reported for context; **may not** produce, overturn, or rescue a terminal |

Declaring both up front — and binding the terminal to \(z^{R}\) only — removes
the freedom to pick whichever state happens to look better after the fit.

**Fidelity inheritance, stated precisely.** \(z^{R}\) is computed with the same
operator, the same window shape (four chronological ring samples), the same
configuration, and the same fail-closed device calculator that R1 certified.
This is *operator reuse under the declared input shape*, and nothing more. It
is **not** a claim that `R1_FAITHFUL` generalizes to a broader event universe.
No `bdist`, accept predicate, or deployment-facing quantity is claimed in this
unit; only the reduced local state is used, as an input to a prediction study.

## 3. Frozen pair universe

For a track \(u\) in sequence \(s\), a candidate pair at horizon \(k\) is
\((z_t, z_{t+k})\) with \(t\) and \(t+k\) both **observed frames** of \(u\).
Horizons are frozen at \(k \in \{1,2,4,8\}\) frames.

A candidate pair enters the **primary set** iff all of:

1. **Observed-contiguous:** every intermediate frame \(t+1,\dots,t+k-1\) is also
   an observed frame of \(u\). *(If the runtime coasted through a gap, the ring
   at \(t+k\) still contains pre-\(t\) samples, so the pair is not a \(k\)-step
   evolution of the same window; including it would silently redefine the
   horizon.)*
2. **Full window at both endpoints:** `foot_len >= 4` at \(t\) and \(t+k\), so
   the R1 `bridge_anchor4_last4` branch applies. The short-lost one-point
   fallback has no velocity and is excluded.
3. **Confirmed at both endpoints:** `track_state == TRACK_CONFIRMED`, matching
   the population Consumer-A bridges over.

Every candidate pair lands in exactly one bucket — `primary`,
`excluded_non_contiguous`, `excluded_short_window`, `excluded_not_confirmed` —
and the counts must sum to the total candidate pairs (partition conservation,
gate V4). Excluded buckets are **reported, never pooled into, and never able to
change** the terminal.

Future state \(z_{t+k}\) is used **only as the prediction target**. It is never
an input feature, and folds are split by sequence so no target leaks across
folds.

## 4. Frozen model family (no member may be added after data are read)

All fitted models are affine, \(\hat z_{t+1} = M z_t + c\), and propagate to
horizon \(k\) by **repeated powers**:

\[
\hat z_{t+k} \;=\; M^{k} z_t \;+\; \Big(\textstyle\sum_{j=0}^{k-1} M^{j}\Big) c .
\]

| Family | Parameters | Definition |
|---|---|---|
| `identity` | none | \(\hat z_{t+k} = z_t\) — no-dynamics lower bound |
| `const_velocity` | none | \(\hat a_{t+k} = a_t + k\,v_t\); \(\hat v_{t+k}=v_t\); \(\hat e_{t+k}=e_t\) — interpretable physical baseline, **the baseline the fitted families must beat** |
| `diag_M` | 5 + 5 | \(M\) restricted to diagonal, plus intercept |
| `full_M` | 25 + 5 | unrestricted \(M\), plus intercept — tests whether cross-state coupling is real |
| `regime_M` | (25 + 5) × 3 | independent \(M_r, c_r\) per scale regime \(r\) (§ 4.2) — tests observable context dependence without hiding it in a pooled matrix |

### 4.1 Estimator (no free knobs)

Ordinary least squares on the **one-step (\(k=1\)) primary pairs of the
training folds**, closed form, **no regularization** (\(\lambda = 0\)), no
feature scaling, no outlier rejection, no iteration budget. If a normal matrix
is rank-deficient, that family **fails** for that fold; it is not a licence to
add ridge, drop a column, or re-parameterize.

Horizon-\(k\) predictions come from powering the **one-step** \(M\), as above.
Direct per-horizon refits are **diagnostic only** and cannot bear the terminal.

### 4.2 Regime partition (frozen rule, refit per fold)

Regimes are assigned **causally, from \(z_t\) only**: scale terciles of
\(e = \texttt{ema\_h}\). Boundaries are the empirical 1/3 and 2/3 quantiles
(linear interpolation) of \(e_t\) **computed on that fold's training pairs
only**; a value exactly on a boundary takes the **lower** regime index. The
boundaries are therefore refit per fold by a fixed rule, never hand-picked, and
never computed on held-out data.

## 5. Frozen metric

For horizon \(k\) and primary pair \(i\), the error is the **scale-normalized
anchor endpoint error**, in the same dimensionless units the production score
normalizes by:

\[
e^{(i)}_k \;=\; \frac{\big\|\,(\hat a_x,\hat a_y) - (a_x,a_y)\,\big\|_2}{\max(e_t,\ 10^{-3})}.
\]

- Per-fold statistic: **median** of \(e^{(i)}_k\) over that held-out sequence's
  primary pairs.
- Aggregate statistic \(\bar e_k\): **median of the seven per-fold medians**
  (so one long sequence cannot dominate).
- Scale error \(|\hat e - e|\) and velocity error are **reported separately**
  and do **not** enter the decision rule.

Validation is **leave-one-sequence-out**: 7 folds, fit on 6 sequences, evaluate
on the held-out one.

## 6. Frozen validity gates (non-compensatory)

Any failure ⇒ terminal `M0_INVALID`; no capability conclusion may be drawn.

| Gate | Must pass |
|---|---|
| **V1 production neutrality** | the capture-on run's MOT output is **byte-identical** to the capture-off run on all seven sequences (the capture is read-only; it must not perturb a single tracker decision) |
| **V2 bounded complete capture** | every per-sequence native file reports `complete=true`, `overflow_events=0`, `total_records == len(records)` |
| **V3 version / provenance** | all files declare exactly `m0_state_capture_v1` / `m0_state_capture_payload_v1` with identical provenance; `tracker_gpu.cu` + preset + device-replay-backend hashes recorded; configuration equals the sealed R1 configuration |
| **V4 partition conservation** | for every `(seq, frame)`, the record count equals the independently counted number of tracks with `active && state != EMPTY && age == 0`; all `(seq, track_uid, frame)` keys unique; and the four pair buckets of § 3 sum exactly to the candidate-pair total |
| **V5 causal completeness** | every record has finite `state8`, `foot_len ∈ [1, 8]`, `ema_h > 0`, and `age == 0`; every \(z^{R}\) is produced by the fail-closed device backend (`--require-device`); a host-fallback reduction invalidates the authority packet |
| **V6 declared support** | all seven MOT17 SDP baseline sequences present; **each fold** contributes **≥ 200 primary pairs at every horizon** \(k\in\{1,2,4,8\}\). A thinner packet cannot produce a terminal — it fails closed rather than reporting a fragile median. |

## 7. Frozen decision rule (ordered; mechanically decidable)

Evaluated **in this order**, so the terminals are disjoint and no judgement call
remains after the numbers land.

**G0 — validity.** V1–V6 all pass, else `M0_INVALID`.

**G1 — usefulness ceiling (production-anchored, a priori).** Let \(\bar e_1^{\*}\)
be the best (lowest) aggregate one-step error over **all five families**. Require

\[
\bar e_1^{\*} \;\le\; 0.40 .
\]

`0.40` is not a tuned number: it is the **sealed production `bdist` accept
threshold** in the same \(h\)-normalized units. A state whose own next-frame
normalized position error is as large as the entire production accept margin
cannot support a score at that threshold. If G1 fails ⇒ **`STATE_INSUFFICIENT`**.

**G2 — stability eligibility (fitted families only).** A fitted family is
eligible only if, in **every** LOO fold (and for `regime_M`, in **every**
regime), the spectral radius satisfies

\[
\rho(M) \;\le\; 1.001 .
\]

An ineligible family is reported but excluded from G3 — a matrix whose powers
grow cannot be called a stable short-horizon transition just because its 1-step
error is low.

**G3 — real improvement over constant velocity.** An eligible fitted family
\(F\) is **useful** iff **all three** hold at **every** \(k \in \{1,2,4,8\}\):

| Sub-gate | Frozen criterion |
|---|---|
| G3a magnitude | \(\big(\bar e_k^{\mathrm{CV}} - \bar e_k^{F}\big)\big/\bar e_k^{\mathrm{CV}} \;\ge\; 0.10\) |
| G3b retention | \(F\) beats `const_velocity` in **≥ 6 of 7** LOO folds |
| G3c concentration | after dropping, within each fold, the **top 10 % of pairs by per-pair improvement**, **≥ 50 %** of the relative gain survives at every \(k\) |

G3a's 10 % is declared pre-data as the **decision-relevant** margin: below it,
a 25-parameter (or 90-parameter) matrix does not earn its deployment and
interpretation cost over a zero-parameter physical baseline. G3c exists because
a gain that lives entirely in a small special-scene subset is a scene finding,
not a representation capability.

**Winner selection (only if ≥ 1 family passes G3):** fewest parameters wins;
ties broken in the frozen order `diag_M` → `full_M` → `regime_M`. No other
tiebreak is permitted.

## 8. Terminal mapping

| Terminal | Condition | Consequence |
|---|---|---|
| `M0_INVALID` | any V1–V6 gate fails | instrumentation repair only; **no** capability, score, or dynamics conclusion |
| `M_STATE_SUFFICIENT` | G1 passes **and** ≥ 1 eligible fitted family passes G3 | the declared state predicts its short-term future, and linear structure beyond CV is real, stable, and non-concentrated → **may request** a separately declared \(e^{A\Delta t}\) interpretation study **and/or** a separately declared real-`bdist` score-ranking study |
| `CV_DOMINANT` | G1 passes, but **no** eligible fitted family passes G3 | the state does predict its short future at production-relevant scale, but **no declared linear family beats constant velocity** → the declared linear class is **closed** on this state; **no** \(e^{A\Delta t}\); a score-ranking study may still be separately declared but **may not** use \(M\) |
| `STATE_INSUFFICIENT` | G1 fails (no family, including CV/identity, reaches the ceiling) | the declared \(z^{R}\) is **not sufficient** to predict its own short-term future → any score built on it inherits that limit; **no** \(e^{A\Delta t}\), **no** score model on this state; the representation \(R\) must be revisited |

No terminal authorizes a score fit, a gate or threshold sweep, a preset change,
a ledger entry, a no-go entry, or an online policy evaluation. Research
acceptance remains owner-side after the packet is reviewed.

## 9. Closure class (what a negative terminal does and does **not** exhaust)

A `CV_DOMINANT` or `STATE_INSUFFICIENT` terminal closes **only** this class:

```text
affine one-step transitions  {identity, const_velocity, diag_M, full_M, regime_M(ema_h terciles)}
  on  z^R  (R1 reduction, adaptive anchor mode 2, rate 0.03, bridge_at=4)
  over MOT17 train SDP 7-seq, observed-contiguous confirmed pairs
  at horizons k ∈ {1, 2, 4, 8}, powered from the one-step fit
```

It **does not** exhaust: non-linear or learned dynamics; other state definitions
(including the \(z^{K}\) companion); other regime variables; gap-spanning pairs;
longer horizons; other anchors, presets, detectors, or consumers; or the
continuous-time generator (which is simply **not reachable** from a negative
terminal, by charter § 5).

## 10. Must not

- Fit or read **any** score, weight, gate, or threshold in this unit.
- Introduce \(e^{A\Delta t}\), \(\log M\), or a Jacobian study before this gate
  passes — charter § 5 makes the exponential an *interpretation* layer, never a
  source of prediction capacity.
- Add a family member, a regime variable, a horizon, a tiebreak, or a
  regularizer after data are read.
- Swap the terminal onto the \(z^{K}\) companion because it scored better.
- Treat operator reuse as a generalization of `R1_FAITHFUL`.
- Use GT, labels, detection scores, post-commit state, future frames as inputs,
  or sequence-global normalization.
- Change production, default presets, the evidence ledger, or no-go status from
  this unit.

## 11. Planned artifacts (implementation follows the seal, in a separate PR)

```bash
# Capture (default-off; unset ⇒ no allocation, no writes, production path unchanged)
SACCADE_RESEARCH_M0_STATE_CAPTURE_DIR=out/m0-native \
uv run scripts/eval/mot17.py --preset mamba_whole_graph --detector SDP --double-buffer

uv run python scripts/tools/export_m0_state_capture.py \
  --capture-dir out/m0-native \
  --output out/m0-discrete/records.jsonl

uv run python scripts/tools/verify_m0_state_capture.py \      # V1–V6, fail-closed
  --payload out/m0-discrete/records.jsonl \
  --byte-identity-baseline results/<capture-off-run> \
  --output out/m0-discrete/validity.json

uv run python scripts/tools/run_m0_discrete_transition.py \   # frozen fit + terminal
  --payload out/m0-discrete/records.jsonl \
  --require-device \
  --output out/m0-discrete/terminal.json
```

The exporter manifest is the pre-outcome evidence index. This declaration
contains **no results** and is **not** evidence promotion. The terminal JSON is
produced by the frozen calculator above and reviewed owner-side; a packet
directory with frozen hashes is sealed under
`docs/modules/semantic/research/evidence/discrete_m_capability_20260712/` when
the run completes.
