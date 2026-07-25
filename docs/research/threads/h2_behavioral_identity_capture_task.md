---
doc-status: proposed
doc-promotion: navigation-only; not evidence
doc-date: 2026-07-25
doc-module: semantic
owner-module: semantic
work-class: mainline-study
wip-role: non-wip
activation-gate: "owner acceptance of the §7 terminal partition + registry captured_under field decision + slot/prefix naming + identity-fixture selection; then a Layer-P pass certificate before any seal"
target-decision-layer: none
primary-intent: boundary-diagnostic
output-class: "diagnostic result | substrate-fidelity edge proposal"
mainline-transition: "none from this charter; per-terminal transitions listed in the declaration §7"
created: 2026-07-25
---

# H2 — bridge-decision capture under behavioral runtime identity — proposed task charter

## Status and authority

**PROPOSED / non-WIP.** Not active, not sealed, not sole-active; occupies no
semantic WIP. It authorizes no capture, selects no `I`, creates no `F`/`S`, grants
no exactly-once authorization, establishes no runtime substrate, and activates
nothing downstream.

Authority split:

- the **[H2 declaration](../../modules/semantic/research/headline_bridge_behavioral_identity_capture_declaration_20260725.md)**
  owns the identity mechanism, the two-layer budget, κ typing, frozen degrees of
  freedom, and the ordered terminal partition — this charter restates none of it;
- the **[H0 declaration](../../modules/semantic/research/headline_bridge_full_decision_capture_declaration_20260713.md)**
  remains authoritative for its own closed history and for everything the
  successor consumes unchanged (capture ABI, A3, A5, **A7.6**, packet verifier);
- the **[claim-state registry](../contracts/claim_state_registry.md)** remains the
  sole writer of `quantity.bridge_capture_provenance` state (C5.1);
- this charter owns only navigation, the staging order, and the open-decision list.

## Why this unit exists

Five sealed H0 invocations produced five `H0_PROVENANCE_INVALID` and zero
faithful capture. The owner's
[R5 parity audit](../../modules/semantic/research/evidence/h0_r5_qualification_authoritative_parity_audit_20260725/)
records that the membership predicate was byte-identical between qualification
and the authoritative run while the artifacts under test were not — the same
source built in a different directory yields a different `sha256` and ELF
build-id. Physical artifact identity is therefore not a function of source, and
the loaded closure (4518 `tool_runtime` members of a 4664-member plan) is
loader-emergent. H2 replaces that identity notion with a derivational +
behavioral one and moves plumbing verification out of the exactly-once budget.

Full argument: declaration §0. Design record:
`~/.claude/plans/harmonic-jingling-lemur.md` is a working plan, not authority.

## Staging

| Stage | Content | Touches production behavior |
|:--|:--|:--|
| **S0** | ✅ landed — this charter + the H2 declaration + navigation + registry field proposal | no — docs only |
| **S1** | ✅ landed — behavioral-identity module, four-axis builder, path-partition firewall; **G1 and G2 both pass** (declaration § 5.1.2–5.1.3) | no — default-off research tooling |
| **S2** | ✅ landed — published `runtime_identity`, `captured_under` sidecar, staleness guard in `pre_push`, self-hosted CI re-attestation (declaration § 8.3.1) | no |
| **S3** | ✅ landed — `run_h2_layer_p.py`: retry admissibility, launch hygiene, build, build-tool witness, load-consumption proof, mutation monitor, behavior comparison, append-only retry log | no |
| **S4** | ⚠️ **partial** — terminal partition landed as an executable module (`h2_terminal_partition.py`, 28 tests); the measurement run/packet plumbing is **not implemented** (below) | no |

### S4 — what is and is not implemented

**Landed.** `scripts/tools/h2_terminal_partition.py` is the ordered partition of
declaration § 7 as executable, tested code: first-applicable selection,
exhaustive result mapping with the mandatory execution catch-all, fail-closed on
missing or non-boolean predicates, and a test proving **witness fields cannot
select a terminal**. This is the epistemic core and the one load-bearing owner
decision; § 20.8's two-implementer test is not meetable from a table alone.

**Not implemented.** `run_h2_measurement.py` and its verifier: the four ordered
runs, three capture-on packets, the A7.6 comparison wiring, the packet-verifier
invocation, the evidence root / manifest / checksum inventory, the independent
verifier, and an archive checker for the new corpus. H0's equivalents are
`run_h0_phase_a.py` (~4360 lines) and `verify_h0_phase_a.py` (~2500), hardwired
to H0's invocation contract, sanitized environment, and exactly-once launch;
reusing them means reimplementing that parent contract, which is its own build.

Nothing is blocked by this today: a seal requires the four open owner decisions
below, and none of them has been made. Layer P is independently useful without it
— it is what resolves plumbing coordinates without spending authorizations.

### Falsification gates (S1, before anything downstream)

- **G1** — the same source built in two different directories must produce an
  **equal `behavior` axis** (physical digests will differ and are witness only).
  G1 failing kills the design; stop and report.
- **G2** — three repeats under the A5 policy target must be byte-identical on the
  A7.6 inventory. G2 is a **cheap pre-seal probe of a risk independent of
  provenance**: if production is nondeterministic (GPU-decode race, relink
  threading), a sealed Layer-M would die at `H2_CAPTURE_PERTURBS_POLICY` /
  `H2_PACKET_INVALID` regardless of identity mechanism. H0's structure never
  allowed this to be tested cheaply — its five invocations never reached the runs
  stage.

## Open owner decisions

**Decision surface:** [#286](https://github.com/raylei50653/saccade/issues/286) —
one surface for all four; it authorizes nothing. S4's remaining plumbing is
blocked on decision 1 and deliberately has **no** issue of its own: the
`S4 — what is and is not implemented` section above is its fact-owner, and a
second copy would be a second truth (C5).

1. **The §7 terminal partition** — `provenance_invalid` removed as a predicate;
   pre-seal plumbing failures are not terminals, while a post-launch execution
   failure keeps a fail-closed terminal (§20.8 item 3). **Load-bearing.**
2. **Registry field `captured_under`** (below) — C5.1 owner write.
3. Slot name `H2`, terminal prefix `H2_*`, evidence prefix `h2_measure_<I40>`.
4. Identity fixture (default MOT17-09-SDP, 525 frames).

## Proposed registry diff (not applied — owner writes state)

Against `quantity.bridge_capture_provenance`
([registry](../contracts/claim_state_registry.md)); **state, terminal, cause, and
`open_limits` unchanged** — this adds a substrate-version coordinate and one
dependency, nothing else:

```yaml
captured_under:                  # NEW — which published runtime_identity the
  decision_surface: <sha256>     #       evidence was captured under; absent for
  implementation:   <sha256>     #       all existing rows (no retroactive claim)
  environment:      <sha256>
  behavior:         <sha256>
dependencies:                    # ADD one entry
  - H2 Layer-M measurement (proposed; declaration 2026-07-25) — a passing
    terminal 5 is the precondition, not the activation, of a runtime-fidelity edge
```

Consumption rule proposed with it: a row whose `captured_under.decision_surface`
or `captured_under.behavior` differs from the published `runtime_identity` is
**stale** and its consumers are inadmissible until re-attested; a difference
confined to `implementation` / `environment` with `behavior` equal is
behavior-preserving and changes nothing. Version-lag accounting in the sense of
[ADR 020 §S2](../../decisions/020-doc-lifecycle-new-nogo.md).

## Boundary

Does **not**: change any preset default, kernel constant, or capture ABI; enable
any capture by default; unblock `H0_ROUTE5_B1`, `GCTM_B1`, or O1; touch any
`h0_phase_a_*` / `h0_preseal_freeze_*` evidence root; alter H0's declaration
bytes (declaration §0.2 explains why an Amendment 11 is mechanically
inadmissible); or establish any guarantee retroactively.
