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

None yet. A future activated unit must create a versioned temporal-reduction
contract and a runtime-shadow evidence packet; it must not reuse an offline
MOT-row reconstruction as production score data.

## Current step

**R1 capture-contract preflight:** existing D0 shadow rows contain reduced
terms but not the two effective four-sample windows consumed by
`bridge_anchor4`; they cannot independently replay \(R\). Define and add a
versioned, default-off capture of those windows and the short-lost fallback,
then seal replay/stability criteria before collecting outcomes. Score fitting,
gate sweep, and production change remain unauthorized.

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
