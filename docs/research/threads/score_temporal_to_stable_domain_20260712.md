---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
work-class: mainline-study
wip-role: parked
created: 2026-07-12
---

# Score temporal-to-stable-domain modeling

## Status

**PARKED** (2026-07-12). R1 capture/replay is **closed** at terminal
`R1_FAITHFUL` — that result stands and is what made runtime coordinates
auditable. The WIP lock has moved to the gate-shaped line,
[runtime-faithful safe domain](runtime_faithful_safe_domain_20260712.md).

The discrete-\(M\) follow-on was **reclassified as a score-ranking feature
question, not a gate**, and is parked unsealed
([declaration § 0](../../modules/semantic/research/discrete_m_capability_declaration_20260712.md)).
Under the owner's architecture — *gate builds a safe domain; score ranking
separates GT* — the score layer may not be opened before the retained domain it
is supposed to rank inside has been established on runtime-faithful coordinates.
This line resumes there.

## Current boundary

The target remains consumer-A runtime bridge score `bdist`. The first unit has
established a faithful, auditable temporal reduction \(R\) under the sealed
headline configuration and seven-sequence support. Score ranking, gate policy,
and continuous-time \(e^{A\Delta t}\) remain out of scope until discrete-\(M\)
is declared and accepted on its own contract.

## Read first

- [discussion charter](../../modules/semantic/research/score_temporal_to_stable_domain_20260712.md)
- [R1 results — terminal `R1_FAITHFUL`](../../modules/semantic/research/r1_temporal_reduction_capture_results_20260712.md)
- [R1 capture declaration](../../modules/semantic/research/r1_temporal_reduction_capture_declaration_20260712.md)
- [D0 runtime-shadow fidelity result](../../modules/semantic/research/d0_runtime_shadow_fidelity_results_20260712.md)
- [`s0` validity amendment](../../modules/semantic/research/s0_proxy_validity_amendment_20260712.md)
- [runtime-quantity fidelity protocol](../contracts/runtime_quantity_fidelity_protocol.md)

## Artifacts

- [R1 results](../../modules/semantic/research/r1_temporal_reduction_capture_results_20260712.md)
  — owner-accepted `R1_FAITHFUL` with scoped configuration/support.
- [evidence packet](../../modules/semantic/research/evidence/r1_temporal_reduction_capture_20260712/manifest.json)
  — sealed audit record + replay JSON + frozen hashes.
- [R1 declaration](../../modules/semantic/research/r1_temporal_reduction_capture_declaration_20260712.md)
  — sealed pre-outcome contract (unchanged gates).
- `scripts/tools/export_r1_temporal_reduction_capture.py` /
  `verify_r1_temporal_reduction_replay.py` — fail-closed export + authority
  `--require-device` calculator.

## Current step

**None — parked.** Resume only after the safe domain exists on runtime
coordinates. Preserved state of the (unsealed) discrete-\(M\) design, so it is not
re-derived:

```text
reclassified: score-ranking feature (L2), NOT a support gate (L0)
mechanism prior: velocity shrinkage (gain grows with horizon); k=1 is noise-dominated
horizon defect: declared {1,2,4,8} vs production la (median 12, p90 26; only 6.9% have la<=2)
authorization: none. capture/fit/metric all unauthorized.
```

<details>
<summary>Superseded pre-park framing (kept for audit)</summary>

**Discrete-\(M\) declaration in `draft`; owner seal pending (rev. 2 after review).**

Do **not** open score fitting, gate sweeps, or preset changes from R1. The
[discrete-\(M\) declaration](../../modules/semantic/research/discrete_m_capability_declaration_20260712.md)
freezes, before any data are read: the per-frame capture contract
`m0_state_capture_v1`; the terminal-bearing state \(z^{R}\) (the **lost-side** R1
reduction) with \(z^{K}\) demoted to diagnostic; the pair universe with a frozen
exclusion **precedence ladder** and partition conservation; the five-member
affine family with a fully pinned **float64 SVD solver contract** (explicit
`rcond`, condition-number and minimum-sample failure rules); the
\(h\)-normalized **anchor-position** metric under leave-one-sequence-out; and an
ordered decision rule — stability eligibility **first**
(\(\rho\le1.001\) **plus** finite-horizon \(\|\tilde M^k\|_2\le2.0\) and affine
drift bounds), **then** a ceiling required across **all four** horizons, then
≥10 % over constant velocity with ≥6/7 folds and ≥50 % gain surviving
de-concentration.

**Two claim boundaries are binding.** (1) The scope is **anchor propagation**,
not "state sufficiency": terminals are named `ANCHOR_*` and no terminal claims
velocity or scale predictability. (2) The 0.40 bar is a
**production-inspired heuristic ceiling**, *not* a quantity-equivalent accept
margin — production `bdist` is pair-level (candidate head-4, two-sided
\(h_{\mathrm{ref}}\), speed weight, `dist_h` blend, real `la`), and none of that
enters this metric. What the study *is* relevant to: production's `fwd_r` already
propagates this very state by the **constant-velocity rule**, and this study asks
whether a stable linear operator beats that rule.

**Blocking precondition recorded:** the sealed R1 packet **cannot** carry this
study — it holds only the effective four-sample window at bridge events, so
horizons 4 and 8 and \(M^k\) stability are unmeasurable on it. Hence a new,
separately versioned capture rather than a post-hoc relaxation of charter § 5.

Nothing may be captured, exported, or fitted until the seal event in
[declaration § 12](../../modules/semantic/research/discrete_m_capability_declaration_20260712.md#12-seal-transition-the-single-authoritative-event)
occurs. PR merge alone is **not** the seal.

</details>

## Acceptance

```text
R1_FAITHFUL (closed under sealed config + seven-seq support)
→ separately declared discrete-M short-horizon capability study
→ separately declared real-bdist score-ranking capability study (if justified)
→ separately declared online policy evaluation (if justified)
```

## Must not

- Treat `R1_FAITHFUL` as universal bridge-reduction fidelity.
- Treat a gate coverage result as a score-ranking result, or vice versa.
- Use `s0` as a production `bdist` stand-in.
- Fit \(M\), \(A\), or a score before a discrete-\(M\) declaration exists.
- Use future information or GT labels in the causal conversion.
- Change the preset, production path, ledger, or no-go registry from this line.

## History

- 2026-07-12: Proposed after runtime-shadow D0 established that the temporal
  reduction operator, rather than the shared outer score formula, is the
  blocking source of proxy unfaithfulness. Opened as non-WIP discussion; no
  execution authorization.
- 2026-07-12: Owner activated the line. Initial source audit found that D0
  captures reduced anchor/velocity outputs but not the effective input windows;
  first unit narrowed to a versioned R1 capture-contract preflight.
- 2026-07-12: Sealed `r1_temporal_reduction_capture_v1` before data collection
  (PR #142).
- 2026-07-12: Non-terminal MOT17-04-SDP smoke exceeded presealed `1e-5` host
  replay; instrumentation repair only.
- 2026-07-12: Host R0 repaired (device `bridge_anchor4` + FMA fallback; PR
  #143). Seven-seq packet collected under stamp `20260712T115004Z`.
- 2026-07-12: **Owner terminal `R1_FAITHFUL` accepted** for sealed Consumer-A
  headline adaptive-anchor configuration and declared seven-sequence support.
  R1 capture/replay unit closed. Discrete-\(M\) study authorized only as a
  separately declared follow-on (not opened here).
- 2026-07-12: Substrate audit for the discrete-\(M\) follow-on. The runtime foot
  ring (`FOOT_RING_CAP = 8`, written only on observed frames, frozen while
  coasting) and the Kalman posterior are the state the bridge actually reads;
  the sealed R1 packet exports only the four-sample effective window, so it
  cannot supply horizons 4/8 or \(M^k\) stability. Owner chose a **new per-frame
  state capture** over relaxing charter § 5.
- 2026-07-12: Discrete-\(M\) declaration **drafted** (`m0_state_capture_v1`,
  study `discrete_m_capability_20260712`); **owner seal pending**. No capture,
  fit, or metric authorized yet.
- 2026-07-12: **Line parked.** Owner set the layer architecture — *gate builds a
  safe domain; score ranking separates GT* — under which anchor-propagation
  accuracy is a **score-ranking feature**, not a gate: it changes which retained
  candidate wins, not which candidates remain. The discrete-\(M\) unit is parked
  unsealed and the WIP lock moves to
  [runtime-faithful safe domain](runtime_faithful_safe_domain_20260712.md).
  `R1_FAITHFUL` stands and is the enabler there: it is what makes the runtime
  coordinates auditable.
- 2026-07-12: Owner review returned `REQUEST_CHANGES` on the draft (research
  layer; engineering layer clean). Four defects were real and are now repaired:
  (a) the ceiling was evaluated **before** stability eligibility, so an unstable
  fitted family could carry it and leave a mislabelled `CV_DOMINANT`; (b) a
  one-step-only ceiling was being narrated as 1/2/4/8 short-horizon sufficiency;
  (c) 0.40 was described as the same production accept margin although the metric
  is single-track/lost-side/anchor-only while `bdist` is pair-level; (d) the
  terminal read anchor error yet claimed full-state sufficiency. Also repaired:
  spectral radius alone cannot certify \(M^k\) stability (non-normal transient
  growth) → finite-horizon norm and affine-drift gates added; the OLS calculator
  was underspecified → full float64 SVD solver contract pinned; exclusion-bucket
  precedence, G3c drop/tie/recompute rule, and parameter counts made explicit;
  and the document's self-declared "sealed" status was contradicting the pending
  seal → now `draft` with a single authoritative seal event (§ 12).
