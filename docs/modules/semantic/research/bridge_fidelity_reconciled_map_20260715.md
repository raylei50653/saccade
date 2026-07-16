<!-- doc-status: draft -->
<!-- doc-promotion: navigation-only; not evidence -->
<!-- doc-module: semantic -->
<!-- doc-date: 2026-07-15 -->

# Bridge-fidelity / commutativity — reconciled flagship map (draft)

> **非規範 / navigation-only。** 這是對 bridge-fidelity 線現況的**重建地圖**(2026-07-15;codex read-only 調查產出、抽驗數字逐字命中源檔),以 ADR 020 的 typed-terminal schema 表達。**所有數字由下方 cite 的 owner doc 擁有**,本檔不擁有任何 verdict;owner 一改,本圖須**重生而非手改**。是否升格進 charter preamble,待 owner review。本圖校正了先前一版口耳流傳的旗艦圖(見文末「Discrepancies」),最大校正:**D0 proxy-fidelity 是 FALSIFIED,非 bit-exact**。

---

Schema basis: [`docs/decisions/020-doc-lifecycle-new-nogo.md`](../../../decisions/020-doc-lifecycle-new-nogo.md). Numbers carry an owner tag:

- `[D0]` `docs/modules/semantic/research/d0_runtime_shadow_fidelity_results_20260712.md`
- `[R1]` `docs/modules/semantic/research/r1_temporal_reduction_capture_results_20260712.md`
- `[S0]` `docs/modules/semantic/research/closed/safe_domain_runtime_transfer_results_20260713.md`
- `[EK0]` `docs/modules/semantic/research/frozen_packet_exact_key_recoverability_results_20260713.md`
- `[P0]` `docs/modules/semantic/research/runtime_bridge_decision_path_identifiability_declaration_20260713.md`
- `[T2]` `docs/modules/semantic/research/door0_ranking_probe_results_20260712.md`
- `[State]` `docs/research/contracts/claim_state_registry.md`
- `[Work]` `docs/modules/semantic/TODO.md`

```text
Captured runtime event universe U^evt = 2,577 [D0; unchanged]
├─ Matched / joined pairs M^evt = 1,684 [D0; unchanged]
│  │
│  ├─ κ_D0 proxy-fidelity
│  │    s0 = f(R_offline(x))  ?≈  bdist = f(R_kernel(x))
│  │    Quantification: matched pairs only.
│  │    Result: FALSIFIED, not bit-exact:
│  │      agreement = 95.07% [D0]
│  │      |Δ| q95 = 1.4171 [D0]
│  │      Spearman ρ = 0.9558 [D0]
│  │
│  ├─ κ_R1 runtime replay
│  │    captured native causal state ──replay──> runtime bdist / predicate
│  │    Quantification: captured native events, not offline pairs.
│  │    Result: R1_FAITHFUL within the sealed capture contract; this is a
│  │    different fidelity assertion from κ_D0. [R1]
│  │
│  └─ ρ_S0: matched GT-bearing rows → lost-track trial unit
│       Statistical exposure: 116 GT lost tracks [S0]
│       Best offline grid: 3 hurts / 116; one-sided 95% CP UCB = 0.0654833,
│       above ε = 0.05 [S0]
│       ⇒ no active offline-safe grid; S0_UNDECIDABLE. [S0]
│
├─ cohort_gap G^evt = 539 [D0; unchanged]
│  └─ 169 identified unique (seq, lost_global_id) tracks [EK0; unchanged]
│     All are absent from the frozen offline pair universe; none is
│     exact-key recoverable. They are not statistical trial units. [EK0]
│
└─ unemitted E^evt = 354 [D0; unchanged]
   └─ no valid global lost-track identity; no offline key and no trial-unit
      reduction. They are not statistical trial units. [EK0]

U^evt = M^evt ⊍ G^evt ⊍ E^evt
2,577 = 1,684 + 539 + 354 [D0; unchanged]
```

`ρ_v: event → (seq, lost_global_id)` should not be attributed to R1 as in the pasted graph. R1’s required native identity is `(seq, native_capture_ordinal, lost_slot, cand_slot)`; output-layer global IDs are optional provenance and missing IDs remain reported. `[R1]` The global-ID reduction is EK0’s descriptive bookkeeping for `cohort_gap`, while S0’s statistical reduction is specifically to lost-track units. `[EK0] [S0]`

```text
Offline/runtime comparison square — only e ∈ M^evt has an offline counterpart

offline trajectory ─R_offline─> s0
       │ J_v (partial exact-key join)
       ▼
runtime trajectory ─R_kernel──> bdist

s0 ≉[κ_D0] bdist on M^evt: κ_D0 is FALSIFIED. [D0]

Separate runtime-only replay square

captured causal state ─C_R1─> replayed bdist
       │                         │
       └──── native bdist ───────┘

replay agrees under the sealed R1 contract. [R1]
```

Thus the current ceiling is not “commutes on the representable subdomain.” Offline-to-runtime commutativity fails on the representable matched domain itself. The only positive fidelity result is runtime capture-to-replay fidelity under R1’s scoped contract. `[D0] [R1]` The matched accept-region coverage was not the escape hatch: it was 63.67% versus 65.35% overall, so D0’s negative result was not attributed to accept-region undercoverage. `[D0]`

## Typed terminals

`model_ref / model_version` is intentionally “no model yet” below: I found versioned capture contracts and the framework contract, but no line-specific, versioned master model to which these verdicts are pinned. A capture-contract version is not silently promoted into a model version.

```yaml
layer: κ_D0 — offline s0 ↔ runtime bdist fidelity
epistemic_verdict: FALSIFIED
lifecycle_disposition: SEALED
verdict_locus:
  model_ref / model_version: no line-specific versioned model yet
  assumptions: shadow, non-committing Consumer-A runtime capture; sealed headline-m configuration; predeclared D0 threshold, numeric, and rank boxes
  domain: exact-key matched D0-v2 pairs only; excludes cohort_gap, unemitted, other presets, detectors, and committing-bridge behavior
evidence_owner: docs/modules/semantic/research/d0_runtime_shadow_fidelity_results_20260712.md
```

```yaml
layer: κ_R1 — captured-runtime temporal-reduction replay
epistemic_verdict: VERIFIED
lifecycle_disposition: CLOSED
verdict_locus:
  model_ref / model_version: no line-specific versioned model yet
  assumptions: r1_temporal_reduction_capture_v1; device replay backend; sealed adaptive-anchor Consumer-A configuration
  domain: captured events on the declared seven-sequence MOT17-SDP support
evidence_owner: docs/modules/semantic/research/r1_temporal_reduction_capture_results_20260712.md
```

> `VERIFIED` reflects the owner-accepted `R1_FAITHFUL`. (Earlier draft of this map was forced to `INCONCLUSIVE` because ADR 020 lacked an affirmative epistemic value; the value was added on 2026-07-15 precisely from this gap.)

```yaml
layer: κ_claim / S0 — safe-axis transfer and ε-level hurt bound
epistemic_verdict: NOT_IDENTIFIABLE
lifecycle_disposition: CLOSED
verdict_locus:
  model_ref / model_version: no line-specific versioned model yet
  assumptions: frozen coarse grid, one-sided 95% Clopper–Pearson rule, ε = 0.05, lost-track independence unit
  domain: D0 matched rows with valid GT lost-track identities; unjoined events are coverage inputs, never CP trials
evidence_owner: docs/modules/semantic/research/closed/safe_domain_runtime_transfer_results_20260713.md
```

```yaml
layer: EK0 — exact-key recoverability of frozen unjoined events (RJ0 replacement)
epistemic_verdict: NET_NEGATIVE
lifecycle_disposition: SEALED
verdict_locus:
  model_ref / model_version: no line-specific versioned model yet
  assumptions: outcome-blind audit of immutable D0-v2 artifacts and their hashes; exact v2 key only
  domain: frozen cohort_gap and unemitted events, not prospective expanded captures or joins
evidence_owner: docs/modules/semantic/research/frozen_packet_exact_key_recoverability_results_20260713.md
```

```yaml
layer: P0 — capture provenance and decision-path observability
epistemic_verdict: NOT_IDENTIFIABLE
lifecycle_disposition: CLOSED
verdict_locus:
  model_ref / model_version: no model yet
  assumptions: scope-corrected target is headline-m; provenance lacks h_lo, h_hi, spatial_gate, max_speed, and a capture-time kernel hash
  domain: the frozen D0/R1/S0 packets; not an assertion that their capture semantics are invalid
evidence_owner: docs/research/contracts/claim_state_registry.md
```

```yaml
layer: Door 0 / T2 — ranking power of the declared proxy-space class
epistemic_verdict: NET_NEGATIVE
lifecycle_disposition: CLOSED
verdict_locus:
  model_ref / model_version: no line-specific versioned model yet
  assumptions: gate-retained ambiguous band; s0 ordering; sealed ranking metrics and boxes
  domain: exactly the 12 enumerated candidates; excludes the 9 untested AND pairs, finite-λ, continuous, learned, and runtime-bdist score families
evidence_owner: docs/modules/semantic/research/door0_ranking_probe_results_20260712.md
```

The class closure is only about `s0`, not production `bdist`; it remains a valid proxy-space closure, but its mainline-transition status was revoked. `[T2] [State]`

```yaml
layer: discrete-M anchor propagation
epistemic_verdict: NOT_EVALUATED
lifecycle_disposition: PARKED
verdict_locus:
  model_ref / model_version: no model yet
  assumptions: unsealed m0_state_capture_v1 design; no capture, fit, or metric authorized
  domain: none observed
evidence_owner: docs/modules/semantic/research/discrete_m_capability_declaration_20260712.md
```

H0 has no terminal slot: it is proposed/draft-unsealed (`lifecycle_disposition: PROPOSED`), has no results document, and execution remains prohibited pending a complete pre-seal freeze artifact and owner seal. Its corrected target is `m`, not `s`. `[Work]` Its pre-terminal state should not be fabricated as a sealed terminal.

## Discrepancies vs the pasted reference

- The event partition is unchanged: `2,577 = 1,684 + 539 + 354`; `cohort_gap` still reduces to 169 identified lost tracks and `unemitted` still has no such identity. `[D0] [EK0]`

- “Pair-level bit-exact fidelity on 1,684” is wrong. D0’s pair-level test is a predeclared threshold/numeric/rank fidelity test, and it falsified proxy fidelity on all three boxes. Exact/tolerance-bounded replay belongs to R1 and quantifies captured runtime events, not the joined D0 pair set. `[D0] [R1]`

- Consequently, “commutes only on the representable subdomain” is too strong and directionally wrong. The offline-to-runtime square fails on the representable matched subdomain. `[D0]`

- The pasted R1 reduction is misattributed. R1 retains native event identity; S0 supplies the lost-track statistical unit, and EK0 supplies the `(seq, lost_global_id)` bookkeeping reduction for `cohort_gap`. `[R1] [S0] [EK0]`

- `N_max = 116 < 153` is not a current support verdict. Current S0 owns 116 exposed GT lost tracks and the observed `3/116` CP result, but no current fact owner declares 153 as the required support for this case. EK0 explicitly removed the old `(N,k)`/UCB feasibility envelope. `[S0] [EK0]`

- `RJ0_EXPANSION_FUTILE` was rescinded. It is replaced by EK0’s narrow `EK0_NO_RECOVERABLE_SUPPORT`: an internal-consistency result for the frozen packet, not evidence that broader joins, new identity observability, or recapture are futile. `[EK0]`

- P0’s foreign-capture cause is withdrawn for this line. P0 originally audited `s`; the frozen D0/R1/S0 evidence is `m`. Under `m`, no stamped field contradicts policy; the current terminal is `P0_CAPTURE_SEMANTICS_UNVERIFIABLE` because four policy knobs and the capture-time kernel hash are unstamped. `[P0] [State]`

- H0 was retargeted to `m`; its height gate is `[0.6, 1.7]`, not `s`’s `[0.75, 1.33]`. It remains unsealed and unexecuted. `[H0]`

- Door 0’s `NO_USABLE_RANKING_POWER_IN_CLASS` still closes exactly 12 members, but only over offline `s0`. It does not establish no ranking power for runtime `bdist`; current owner state calls it a proxy-space capability closure, not a production/mainline transition. `[T2] [State]`

- “Informative censoring correlated with offline/runtime difference” is not the current owner phrasing. The owners call `cohort_gap` and `unemitted` non-missing structural partitions, excluded from agreement denominators and CP trials; they do not support a general causal missingness claim. `[D0] [S0] [EK0]`

## Open questions / unverifiable items

- ADR 020 is proposed and no versioned master model or terminal-slot owner exists for this line. The graph therefore cannot honestly pin verdicts to a `model_ref@version`.

- The closed Door 0 thread still describes its result as a step-⑤ mainline closure, while the later D0 validity amendment and claim-state registry revoke that mainline interpretation. The proxy-space closure itself is not in dispute. `[T2] [State]`
