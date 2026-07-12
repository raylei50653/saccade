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

**ACTIVE · mainline-study · sole-active.** R1 capture/replay is **closed** at
terminal `R1_FAITHFUL`. The discrete-\(M\) representation-capability study is
**drafted and pending owner seal** —
[declaration](../../modules/semantic/research/discrete_m_capability_declaration_20260712.md).
No capture, fit, or metric may be run until that seal.

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
- [runtime-quantity fidelity protocol](../eval/runtime_quantity_fidelity_protocol.md)

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

**Discrete-\(M\) declaration drafted; owner seal pending.**

Do **not** open score fitting, gate sweeps, or preset changes from R1. The
[discrete-\(M\) declaration](../../modules/semantic/research/discrete_m_capability_declaration_20260712.md)
freezes, before any data are read: the new per-frame capture contract
`m0_state_capture_v1`; the terminal-bearing state \(z^{R}\) (R1 reduction) with
\(z^{K}\) demoted to diagnostic; the pair universe (observed-contiguous,
confirmed, full-window) with partition conservation; the five-member affine
family powered from a one-step OLS fit; the \(h\)-normalized anchor-error metric
under leave-one-sequence-out; and the ordered decision rule
(ceiling 0.40 = production threshold → \(\rho(M)\le 1.001\) → ≥10 % over
constant velocity, ≥6/7 folds, ≥50 % gain surviving de-concentration).

**Blocking precondition recorded:** the sealed R1 packet **cannot** carry this
study — it holds only the effective four-sample window at bridge events, so
horizons 4 and 8 and \(M^k\) stability are unmeasurable on it. Hence a new,
separately versioned capture rather than a post-hoc relaxation of charter § 5.

Nothing may be captured, exported, or fitted until the owner seals the
declaration.

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
