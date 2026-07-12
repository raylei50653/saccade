---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
work-class: mainline-study
wip-role: sole-active
created: 2026-07-12
---

# Score temporal-to-stable-domain modeling

## Status

**ACTIVE · mainline-study · sole-active.** The owner authorized the first
research unit on 2026-07-12. The semantic module TODO is the WIP authority;
this card navigates its scope and handoff only.

## Current boundary

The target is consumer-A runtime bridge score `bdist`, not the optional
Boolean gate chain and not offline `score_m_bridge` / `s0`. The known blocker
is representation: the same outer formula produces a different quantity when
the temporal reduction operator differs. First establish a faithful,
auditable time-domain → stable-domain conversion; only then ask whether a
score improves ranking among candidates retained by the existing gate. After
that conversion, first test discrete \(z_{t+1}\approx Mz_t+c\) and
\(z_{t+k}\approx M^kz_t\) against identity / constant-velocity / diagonal /
full / regime-conditioned baselines. \(e^{A\Delta t}\) is only a later,
optional continuous-time interpretation; it is not a replacement for missing
temporal state or a new scoring formula.

## Read first

- [discussion charter](../../modules/semantic/research/score_temporal_to_stable_domain_20260712.md)
- [D0 runtime-shadow fidelity result](../../modules/semantic/research/d0_runtime_shadow_fidelity_results_20260712.md)
- [`s0` validity amendment](../../modules/semantic/research/s0_proxy_validity_amendment_20260712.md)
- [runtime-quantity fidelity protocol](../eval/runtime_quantity_fidelity_protocol.md)
- [signal-table gate vs score contract](../eval/signal_table_schema.md#05-gate-vs-score-support--calibration--policy)

## Artifacts

- [R1 temporal-reduction capture declaration](../../modules/semantic/research/r1_temporal_reduction_capture_declaration_20260712.md)
  — sealed pre-outcome contract for the nested runtime payload, replay, and
  temporal-sensitivity readout; it is not evidence and contains no outcomes.
- `scripts/tools/export_r1_temporal_reduction_capture.py` — fail-closed export
  of native shadow observations into `r1_temporal_reduction_payload_v1`.
- `scripts/tools/verify_r1_temporal_reduction_replay.py` — label-free R0 replay
  and declared causal-sensitivity calculator.

## Current step

**R1 capture-contract preflight — packet ready for owner review.** Host R0
replay was repaired (device-bit-exact `bridge_anchor4` / FMA `bridge_vel4`;
presealed `1e-5` unchanged). The declared seven-sequence shadow packet is at
`out/r1_temporal_20260712T115004Z/` (2577 events; V1 MOT byte-identity;
R0 terms within tolerance; predicate/order clean;
`causal_sensitivity_interpretable=true`). **No research terminal is accepted
until owner review** — do not promote `R1_FAITHFUL` or open score fitting /
gate / preset work from this handoff alone.

## Acceptance

The active first unit is limited to:

```text
versioned runtime shadow capture → R0/R1 replay fidelity + temporal stability
→ separately declared discrete-M short-horizon transition gate
→ separately declared real-`bdist` score-ranking capability study
→ separately declared online policy evaluation (if justified)
```

The first unit must have a sealed declaration with validity, fidelity,
boundary-preservation, and coverage criteria before outcome metrics are read.

## Must not

- Treat a gate coverage result as a score-ranking result, or vice versa.
- Use `s0` as a production `bdist` stand-in.
- Fit a learned/calibrated proxy before specifying and validating \(R\).
- Use future information or GT labels in the causal conversion.
- Change the preset, production path, ledger, or no-go registry.

## History

- 2026-07-12: Proposed after runtime-shadow D0 established that the temporal
  reduction operator, rather than the shared outer score formula, is the
  blocking source of proxy unfaithfulness. Opened as non-WIP discussion; no
  execution authorization.
- 2026-07-12: Owner activated the line. Initial source audit found that D0
  captures reduced anchor/velocity outputs but not the effective input windows;
  first unit narrowed to a versioned R1 capture-contract preflight.
- 2026-07-12: Sealed `r1_temporal_reduction_capture_v1` before data collection:
  independent nested-window export, strict source/provenance gates, label-free
  R0 replay, predicate/order checks, and causal-sensitivity reporting. D0's
  fixed evidence packet remains unchanged.
- 2026-07-12: Non-terminal MOT17-04-SDP engineering smoke: 121 native rows,
  zero overflow, nested export complete; 24 rows had no output-layer global id
  and were retained under native identity. Predicate and comparable-order
  replay agreed, but component replay exceeded the presealed `1e-5` tolerance.
  This is instrumentation/replay repair only (not `R1_FAITHFUL`, no full-support
  packet, no score/gate/preset conclusion).
- 2026-07-12: Host R0 repair: plain float64 / naive float32 missed CUDA FMA
  contraction on `bridge_vel4` (~6e-5) and adaptive residual weights on
  `ay`/`cy0`. Device-backed `libr1_bridge_replay.so` (+ host FMA fallback)
  restores bit-exact anchors/velocities; score terms remain within `1e-5`.
  Full seven-seq packet collected under stamp `20260712T115004Z` for owner
  review (see `out/r1_temporal_20260712T115004Z/packet_summary.json`).
