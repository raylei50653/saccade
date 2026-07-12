<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-12 -->
<!-- doc-module: semantic -->

# Score temporal-to-stable-domain modeling — discussion charter

> **One-line:** score 建模先不找權重或新 gate；先把 runtime 的時域軌跡以可重播、因果且可比較的化約算子 \(R\) 映射為穩定表示，再以真實 kernel `bdist` 驗證該映射，通過後才討論 retained pairs 的排序模型。

Thread: [score temporal-to-stable-domain](../../../research/threads/score_temporal_to_stable_domain_20260712.md) ·
Runtime fact: [D0 runtime-shadow fidelity](d0_runtime_shadow_fidelity_results_20260712.md) ·
Scope amendment: [`s0` is offline-only](s0_proxy_validity_amendment_20260712.md) ·
Layer contract: [gate vs score](../../../research/eval/signal_table_schema.md#05-gate-vs-score-support--calibration--policy) ·
Fidelity protocol: [runtime-quantity fidelity](../../../research/eval/runtime_quantity_fidelity_protocol.md)

## 1. Why this is a separate score problem

`gate` and `score` answer different questions:

| Layer | Question | Primary evidence | This line's status |
|---|---|---|---|
| L0 support gate | Can this candidate remain in the decision set? | GT coverage / hurt / safe prune | Out of scope |
| L1 calibration | Are score terms on a comparable scale? | conditional distribution / overlap / failure slice | Depends on a valid \(R\) |
| L2 score-ranking | Among retained candidates, which should win? | event-local rank, margin, top-1 | Deferred until \(R\) is accepted |

A hard threshold can be evaluated as a support decision. A score must preserve
**relative order within one event**, so it cannot inherit a gate conclusion or
be judged by full-pool threshold precision. The target consumer is the active
tracker-core bridge (consumer A), whose live score is `bdist`; it is not the
optional Boolean semantic-relink gate chain.

The D0 runtime-shadow study established why this needs its own line. The
offline proxy and kernel use the same outer function \(f\), but evaluate it on
different reductions of a trajectory:

\[
s_0 = f(R_{\mathrm{offline}}(x)),\qquad
\mathrm{bdist} = f(R_{\mathrm{kernel}}(x)).
\]

The proxy failed all fidelity boxes. Its scale mismatch leaves a
horizon-independent error floor, while its velocity mismatch grows with
extrapolation horizon. Therefore fitting weights on \(s_0\), or simply
recalibrating \(s_0\), would model a third quantity rather than production
`bdist`.

## 2. Object to model

For one runtime bridge decision event \(e\), let the causal observations before
the decision be

\[
X_e = \{(t_k, p_k, h_k, q_k)\}_{k \leq K_e},
\]

where \(p_k\) is the tracked foot position, \(h_k\) the tracked height, and
\(q_k\) only records runtime-valid metadata (ring position, stride, horizon,
and missingness). The proposed modeling boundary is

\[
X_e \xrightarrow{\ R\ } z_e
\xrightarrow{\ f\ } b_e
\xrightarrow{\ g\ } s_e .
\]

- \(R\): temporal reduction — sampling, anchor choice, causal scale update,
  velocity estimate, horizon convention, and missing-data behavior.
- \(z_e\): stable-domain representation, e.g. normalized endpoint distance,
  forward/backward residuals, speed weight, scale state, and representation
  provenance. “Stable” means *the same causal event has the same defined
  representation across a faithful replay*, not “time has been discarded”.
- \(f\): current production aggregation (`bdist`) on the kernel's state.
- \(g\): a future score-layer calibration or ranking model. This line does not
  choose \(g\) yet.

The key rule is:

\[
\boxed{\text{A score feature is not defined until its }R\text{ is defined.}}
\]

Thus a stateful EMA normalizer, a stride-3 foot ring, and a four-sample OLS
velocity are part of the quantity definition, not implementation detail to be
approximated after fitting.

## 3. Candidate conversion strategy

The first study compares representations, not score weights.

| Representation | Definition | Role in first study |
|---|---|---|
| \(R_0\): kernel-faithful replay | Reproduce the CUDA ring, OLS, adaptive anchor, EMA height, horizon, float precision, and missing-value path exactly from shadow-captured state. | Required reference; only source allowed to make a deployment-facing `bdist` claim. |
| \(R_1\): explicit canonical state | Serialize all causal state required by \(R_0\) into a versioned event record; derive terms from that record. | Preferred stable-domain interface, provided it reproduces \(R_0\). |
| \(R_2\): reduced temporal summaries | Fixed causal multi-horizon summaries derived from \(R_1\), with every aggregation window and normalizer declared. | Diagnostic candidate only; cannot replace \(R_0\) until it passes a new fidelity contract. |

`R0` is not an optional baseline. It separates an estimator-preserving
conversion from another offline approximation. `R1` makes the state explicit
and auditable; `R2` may later trade information for a simpler ranking model,
but that is a hypothesis rather than a free normalization step.

## 4. First research unit: temporal-reduction contract

Before any score-family or weight search, create one read-only runtime-shadow
study with bridge commit suppressed. Its output is a versioned
`TemporalReductionContract`, not a model checkpoint.

### 4.1 Capture contract

Capture, on the exact consumer-A event key, both the raw causal state and the
kernel output:

- ring samples and their frame/time indices;
- stride and selected sample indices;
- OLS inputs and fitted velocity;
- adaptive anchor inputs and selected anchor;
- EMA height state before/after the update and its coefficient;
- `la`, bridge-at convention, missingness flags, and float32 kernel terms;
- `dist_h`, `fwd_r`, `bwd_r`, `w`, and final `bdist`.

The capture must preserve the existing shadow guarantees: no bridge commit,
partition conservation, deterministic replay, and explicit coverage of the
production accept region. A MOT-row reconstruction alone is insufficient.

### 4.2 Conversion tests

The future sealed declaration must predeclare thresholds, but its tests should
be organized in this order:

1. **Replay fidelity:** \(R_1(X_e)\) reproduces every captured \(R_0\) term,
   final `bdist`, accept predicate, and within-event ordering at the declared
   precision.
2. **Temporal stability:** controlled causal truncation, ring-position shift,
   and equivalent serialization/replay do not create unexplained score drift.
   Sensitivity is reported by horizon and by component, never pooled only.
3. **Boundary preservation:** error and rank agreement are reported separately
   for accept/reject boundary, GT/FP conditionals, and the production support.
4. **Provenance completeness:** every field consumed by \(R\) has a source,
   update order, unit, and missing-data rule. An unobserved kernel state is a
   validity failure, not an imputation opportunity.

### 4.3 Terminals

| Terminal | Meaning | Consequence |
|---|---|---|
| `R0_INVALID` | capture/replay/coverage validity fails | no score claim; repair instrumentation only |
| `R1_FAITHFUL` | explicit causal state reproduces the kernel and survives the declared stability checks | authorizes a new, separately declared score-ranking capability study on real `bdist` |
| `R2_UNFAITHFUL` | a reduced summary changes production quantity or decision surface | keep it diagnostic; do not fit or deploy it as a score substitute |
| `R_STABILITY_UNRESOLVED` | fidelity holds but controlled temporal sensitivity has no defensible interpretation | no score-model selection; refine the representation contract |

No terminal authorizes a gate change, a preset change, or weight fitting in the
same unit.

### 4.4 Local propagation after the state is faithful

The temporal-reduction operator \(R\) and a state-transition operator answer
different questions. \(R\) reconstructs/serializes the causal state that the
runtime actually used; it cannot be replaced by a matrix exponential. Once
`R1_FAITHFUL` supplies a complete local state \(z\), a short-horizon dynamics
model can describe how that state or a small error evolves:

\[
\dot z = A z,
\qquad
z(t+\Delta t) = e^{A\Delta t}z(t).
\]

Here \(A\) is a local instantaneous rule—for example position driven by
velocity, velocity damping, or coupled scale/motion terms—whereas
\(e^{A\Delta t}\) is the accumulated finite-time transition. Its series

\[
e^{A\Delta t}=I+A\Delta t+
\frac{(A\Delta t)^2}{2!}+\frac{(A\Delta t)^3}{3!}+\cdots
\]

is a sum of direct and repeatedly propagated cross-state effects; it is not an
elementwise exponential of a tensor.

For non-linear local dynamics \(\dot z=f(z)\), use the Jacobian at the
captured state, \(A=J_f(z_t)\), only as a short-horizon approximation:

\[
\delta z(t+\Delta t)\approx e^{J_f(z_t)\Delta t}\delta z(t).
\]

If a future score or safety quantity is \(b=g(z)\), the sensitivity chain is

\[
\delta z_t
\xrightarrow{\ e^{A\Delta t}\ }
\delta z_{t+\Delta t}
\xrightarrow{\ J_g\ }
\delta b_{t+\Delta t},
\qquad
\delta b_{t+\Delta t}\approx J_g e^{A\Delta t}\delta z_t.
\]

This is the appropriate use of the stable domain: explain which local errors
are attenuated, rotated, mixed, or amplified before they affect a score/safety
margin. It is **not** a way to recover a missing foot-ring sample, OLS
residual, EMA state, or adaptive-anchor input. If those were not captured,
there is no complete \(z\) from which \(A\) can restore the lost history.

Any \(A\) or \(J_f\) study is therefore downstream of `R1_FAITHFUL` and must
be separately declared. It is also **not** the first dynamics test. First
establish that the selected state has a stable, useful *discrete* short-horizon
transition; only then decide whether a continuous-time generator adds meaning.

## 5. Discrete short-horizon transition gate (before any matrix exponential)

Given a complete captured state

\[
z_t=[x,y,v_x,v_y,h,\ldots]^\top,
\]

fit and validate the smallest useful discrete transition first:

\[
z_{t+1}\approx Mz_t+c,
\qquad
z_{t+k}\approx M^kz_t.
\]

This is a representation-capability study, not score fitting. Its first
question is:

\[
\boxed{\text{Is the selected }z_t\text{ sufficient to predict its short-term future?}}
\]

The predeclared comparison family must include, on identical events and
horizons, at least:

| Family | Purpose |
|---|---|
| identity | no-dynamics lower bound |
| constant velocity | interpretable physical baseline |
| diagonal \(M\) | independent per-state persistence/decay |
| full \(M\) | tests whether cross-state coupling adds real value |
| regime-conditioned \(M_r\) | tests observable context dependence without hiding it in a pooled matrix |

Required readouts are horizon-specific error at 1, 2, 4, and 8 frames;
stability of repeated powers \(M^k\); per-sequence leave-one-out retention;
and concentration analysis showing whether any apparent gain comes from a
small special-scene subset. A full or regime-conditioned matrix is not useful
unless it beats constant velocity under those held-out checks without unstable
growth.

Only after this gate passes may a separate unit ask whether a continuous-time
generator is useful:

\[
M=e^{A\Delta t}
\qquad\text{or, where a real logarithm is well-defined,}\qquad
A=\frac{\log M}{\Delta t}.
\]

The matrix exponential is therefore an optional **interpretation / irregular-
horizon propagation** layer, not a source of prediction capacity. It cannot
repair an insufficient state, a missing latent variable, or omitted temporal
history.

## 6. What score modeling looks like only after the transition gate

The second unit must be separately predeclared at `target decision layer =
score-ranking`. It works only inside the runtime-retained candidate set and
uses event-local comparisons:

\[
s_e = g(z_e),\qquad
\text{evaluate }\Delta\operatorname{rank}_{\mathrm{GT,FP}},\
\Delta\operatorname{margin},\ \Delta\operatorname{top1}.
\]

Recommended progression:

1. Calibrate individual, dimensionless terms from \(z_e\); retain context
   labels such as horizon rather than hiding them inside an unreported global
   average.
2. Establish whether any term has stable ranking headroom over **real**
   `bdist` in the same event.
3. Only then compare a small, predeclared family of monotone/additive models
   \(g\), with a fixed baseline and no gate threshold tuning.
4. Evaluate an accepted score model online as a distinct decision-policy study;
   offline rank improvement is not an identity or IDF1 claim.

This keeps normalization, ranking, and decision policy separate:

```text
temporal observations → R (quantity definition) → stable representation
                     → calibration → ranking score → assignment policy
                     ↘ support gate remains a separate branch
```

## 7. Non-goals and guardrails

- Do not reopen the closed `s0` 12-member probe as if it were a production
  `bdist` result; it remains a proxy-space capability closure only.
- Do not turn the new score into a gate by optimizing a threshold, coverage, or
  FP-pruning headline.
- Do not fit a proxy to `bdist` and call correlation a conversion; reproducing
  \(R\), not regressing onto its output, is the prerequisite.
- Do not introduce \(e^{A\Delta t}\) before the discrete \(M\) family has
  demonstrated stable held-out short-horizon prediction.
- Do not use future frames, GT labels, post-commit state, or sequence-global
  normalization in a causal production representation.
- Do not change production, default presets, evidence ledger, or no-go status
  from this discussion.

## 8. Open design questions for the discussion

1. What is the smallest serialized causal state that still reproduces `bdist`
   exactly enough for its predicate and event-local rank?
2. Which time transformations are physically equivalent and therefore valid
   stability tests, versus actual changes to the observed event?
3. Should horizon remain an explicit context variable, or can a faithful
   representation make a conditional score comparable across horizons without
   hiding residual drift?
4. Is `R2` useful only as an explanatory view, or can a bounded reduced
   representation become an online interface after a separate fidelity pass?

## 9. Initial code audit — 2026-07-12

The first inspection establishes the exact starting boundary:

| Surface | Present in existing shadow event | Enough for independent \(R\) replay? |
|---|---:|---:|
| final `bdist`, terms, anchors, OLS velocities, EMA heights, `la` | yes | no — these are already-reduced outputs |
| four chronological `(cx, cy, h)` samples consumed by `bridge_anchor4` for each endpoint | no | no |
| short-lost fallback sample/count | no | no |
| anchor mode/rate and output threshold | yes | only as configuration provenance |

`relink_bidir_propose_kernel` consumes the candidate head-four and the lost
last-four (or the short-lost fallback) before it emits the currently captured
anchors and velocities. The host helper
`consumer_a_estimate_from_rings` already encodes the intended replay shape,
but current capture rows cannot supply its rings. Therefore the existing D0
packet remains valid for proxy fidelity; it cannot answer this line's
representation-fidelity question.

**Instrumentation started:** the default-off native observation payload now
copies those two effective windows, their consumed lengths, and
`bridge_dir_bonus` into `BridgeFidelityEvent`; the Python binding exports them
as chronological sample lists. D0's fixed v2 CSV field list is deliberately
unchanged, so this does not reinterpret or modify its sealed packet. The new
payload is not yet a versioned R1 artifact and no capture has been run.

**Active first unit:** define capture contract `R1` and promote this
default-off observation state into a versioned payload rather than mutating
D0's sealed v2 packet. Before capture data are read, it will seal: source/payload
hashes, shadow byte-identity, zero overflow, all source-field completeness,
component replay tolerance, predicate agreement, and event-local rank policy.
Only this instrumentation-and-replay unit is active; it does not fit \(M\),
\(A\), \(J_f\), or a score model yet.

## Status

**ACTIVE, sole semantic WIP.** The modeling order and claim boundaries above
are now authorized. This is not yet a sealed execution declaration: no runtime
capture result, score model, gate change, or production change has been made.
