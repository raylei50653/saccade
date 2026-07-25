---
doc-status: proposed
doc-promotion: navigation-only; not evidence
doc-date: 2026-07-25
doc-module: semantic
owner-module: semantic
work-class: mainline-study
wip-role: non-wip
activation-gate: "四項 owner decision 已於 2026-07-25 accepted（#286）；剩餘 gate = Phase-B chain form 先成文 + S4 Layer-M plumbing 實作 + 一份 Layer-P pass certificate，然後才談 seal"
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

The four owner decisions of [#286](https://github.com/raylei50653/saccade/issues/286)
were **accepted on 2026-07-25** (below). Acceptance moved no state: it fixes the
partition, the naming, the fixture, and the registry field *schema*, and it
authorizes no execution.

Authority split:

- the **[H2 declaration](../../modules/semantic/research/headline_bridge_behavioral_identity_capture_declaration_20260725.md)**
  owns the identity mechanism, the two-layer budget, κ typing, frozen degrees of
  freedom, and the ordered terminal partition — this charter restates none of it;
- the **[H0 declaration](../../modules/semantic/research/headline_bridge_full_decision_capture_declaration_20260713.md)**
  remains authoritative for its own closed history and for everything the
  successor consumes unchanged (capture ABI, A3, A5, **A7.6**, packet verifier);
- the **[claim-state registry](../contracts/claim_state_registry.md)** remains the
  sole writer of `quantity.bridge_capture_provenance` state (C5.1);
- this charter owns only navigation, the staging order, and **the decision list**
  — including the record of its resolution.

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
| **S1** | ✅ landed — bounded behavior probe, coordinate/probe publisher, runtime-input manifest and path-partition firewall; G1/G2 remain probes, not equivalence evidence | no — default-off research tooling |
| **S2** | ✅ landed — published coordinate/probe/equivalence split, `captured_under` sidecar, fail-closed static guard, same-repository PR-head + main self-hosted input/probe re-attestation | no |
| **S3** | ✅ landed — `run_h2_layer_p.py`: required base, retry verdict, build/load proof, monitor-before-hash runtime-input binding, post-run content/membership/symlink revalidation, bounded probe, v2 certificate, append-only retry log | no |
| **S4** | ⚠️ **partial** — terminal partition landed as an executable module (`h2_terminal_partition.py`, 28 tests) and is now owner-accepted; the measurement run/packet plumbing is **not implemented** (scope below) | no |

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
verifier, and an archive checker for the new corpus. The scope of that build is
fixed in `Current step` below.

Nothing is blocked by this today. Layer P is independently useful without it — it
is what resolves plumbing coordinates without spending authorizations.

### Falsification gates (S1, before anything downstream)

- **G1** — the same source built in two different directories should produce an
  equal bounded behavior probe. Inequality falsifies the intended stability;
  equality says only that MOT17-09 observed no difference and is not an
  equivalence proof. Exact extension/plugin bytes remain certificate/F inputs.
  **Result:** four physically distinct builds — `build/`, `build/ci_arch214/`,
  `build/h2_layer_p/` (declaration § 5.1.2) and the controlled-host CI rebuild at
  merge commit `feb99732` — all produced probe `2dabed0bc05e3bc7…`. Forward
  direction only; the reverse implication is rejected.
- **G2** — three repeats under the A5 policy target must be byte-identical on the
  A7.6 inventory. G2 is a **cheap pre-seal probe of a risk independent of
  provenance**: if production is nondeterministic (GPU-decode race, relink
  threading), a sealed Layer-M would die at `H2_CAPTURE_PERTURBS_POLICY` /
  `H2_PACKET_INVALID` regardless of identity mechanism. H0's structure never
  allowed this to be tested cheaply — its five invocations never reached the runs
  stage. **Result:** `bd88260a76ceb395…` ×3 (declaration § 5.1.3).

## Owner decisions — accepted 2026-07-25

Decision surface [#286](https://github.com/raylei50653/saccade/issues/286).
Acceptance authorizes nothing: no `I`, no `F`/`S`, no capture, no exactly-once
grant, no registry state write.

| # | Decision | Verdict |
|--:|:--|:--|
| 1 | **The §7 terminal partition** — `provenance_invalid` removed as a predicate; pre-seal plumbing failures are not terminals, while a post-launch execution failure keeps a fail-closed terminal (§20.8 item 3) | **ACCEPTED, with one precondition** (below) |
| 2 | **Registry field `captured_under`** — coordinate/probe sidecar on `quantity.bridge_capture_provenance` | **ACCEPTED as schema**; written when the first H2 evidence exists, never retroactively |
| 3 | Slot name `H2`, terminal prefix `H2_*`, evidence prefix `h2_measure_<I40>` | **ACCEPTED** |
| 4 | Identity fixture MOT17-09-SDP (525 frames), identity-mode with `--no-gpu-decode` + single-thread relink | **ACCEPTED** |

### Decision 1's precondition — the Phase-A success path

The partition is accepted as written, **and** the following gap must be closed in
the declaration before any Phase-A seal is issued.

A fully successful Phase A selects no terminal. `h2_terminal_partition.py` maps
`measurement_pass` to `None` and says so explicitly: terminal 5 additionally
requires the frozen seven-sequence Phase B. But declaration §2 states the seal
authorizes "no Phase B", §7 states "Phase A never starts Phase B", and §5.2
forbids retry, resume, and repaired re-run under the same `S`. There is no
Phase-B chain defined anywhere in the declaration — Phase B appears only as *not
authorized*, *required for terminal 5*, and *never started by Phase A*.

The consequence, stated exactly: **the best possible outcome of the one
authorized Layer-M invocation is a spent authorization, an unclosed unit, and
zero mainline transition** (§20.7).

This is **not** inherited from H0. H0's seal covered "sealed Phase A, then (only
if admitted) Phase B" in one chain
([H0 declaration](../../modules/semantic/research/headline_bridge_full_decision_capture_declaration_20260713.md),
pre-seal engineering block). H2 narrowed the seal to Phase A only without
supplying the Phase-B chain that terminal 5 depends on.

Accepted resolution: **the Phase-B chain form — its own `I → F → S`, its
authorization form, and its precondition on a passing Phase A — must be declared
before the Phase-A seal.** The seal scope stays Phase-A-only, so H2's
minimal-epistemic-budget thesis is preserved; what changes is that the success
path exists on paper before an authorization can be spent reaching it.

This adds a precondition to declaration §5.2 and resolves §9. Recording it there
requires an append-only correction plus a `sealed_prefix` re-pin in
`…_declaration_20260725.policy.yaml` (which pins the *entire* body) and a
republish of `docs/reference/runtime_identity.generated.json`, because the policy
file is an `identity_semantics` member. That is a separate change and is **not**
done by this charter; it is listed in `Acceptance` as a seal gate.

## Registry field `captured_under`

Accepted as schema against `quantity.bridge_capture_provenance`
([registry](../contracts/claim_state_registry.md)); **state, terminal, cause, and
`open_limits` unchanged** — this adds a substrate-version coordinate and one
dependency, nothing else:

```yaml
captured_under:                  # exact coordinate/probe used by evidence
  coordinate:
    decision_surface:   <sha256>
    implementation:     <sha256>
    environment:        <sha256>
    identity_semantics: <sha256>
    runtime_inputs:     <sha256>
  probe: <sha256>               # bounded MOT17-09 observation, not equivalence
dependencies:                    # ADD one entry
  - H2 Layer-M measurement (proposed; declaration 2026-07-25) — a passing
    terminal 5 is the precondition, not the activation, of a runtime-fidelity edge
```

**Not written yet, deliberately.** The field records the coordinate *under which
evidence was captured*, and no H2 evidence exists. It is absent for all existing
rows and no retroactive claim is made anywhere; the owner writes it (C5.1) in the
same change as the first H2 evidence root. The digests live in the sidecar
`../contracts/runtime_identity_bindings_v1.json`, not parsed out of registry
Markdown.

Consumption rule: decision-surface, identity-semantics or probe drift is stale.
Implementation, environment or runtime-input drift with an equal probe is
`re_attestation_required`, never behavior-preserving. V1 defines no equivalence
upgrade. Version-lag accounting follows
[ADR 020 §S2](../../decisions/020-doc-lifecycle-new-nogo.md).

## Expected state (lease)

*Planning target only; not accepted state; replaceable inside this charter
without a registry write.*

By the next commit point, the Layer-M build is expected to be **scoped and
reviewable but not sealed**: the controller scope below either survives a review
pass or is replaced. No `I`/`F`/`S`, no authorization, no capture, and no
registry state change is expected to occur.

## Commit point

Owner reviews whether state changed when **either** the Layer-M controller scope
below is implemented and its verifier runs end-to-end on a Layer-P pass
certificate, **or** the scope is disconfirmed and the charter is re-planned.

## Discard when

- the Phase-B chain form cannot be declared on terms the owner accepts (decision
  1's precondition fails) — then the Phase-A seal has no reachable success path
  and the unit is re-planned or parked before any authorization is spent; or
- a G1 probe inequality appears across builds of identical decision-relevant
  source — the design premise is falsified and the coordinate/probe split needs
  rework before anything downstream; or
- an owner decision elsewhere retires observational shadow capture at this ABI,
  which is terminal 2's transition reached without spending an invocation.

## Read first

1. [H2 declaration](../../modules/semantic/research/headline_bridge_behavioral_identity_capture_declaration_20260725.md)
   — §0.1 supersession boundary, §4 coordinate/probe/equivalence, §5 two-layer
   budget, §7 partition, §9 decisions, Review Correction 1.
2. [H0 declaration](../../modules/semantic/research/headline_bridge_full_decision_capture_declaration_20260713.md)
   — A7.6 non-perturbation inventory and the packet verifier, consumed verbatim.
3. [R5 parity audit](../../modules/semantic/research/evidence/h0_r5_qualification_authoritative_parity_audit_20260725/)
   — why H0's identity layer could not transfer.
4. [claim-state registry](../contracts/claim_state_registry.md) — the sole writer
   of `quantity.bridge_capture_provenance`.

## Artifacts

| Artifact | Role |
|:--|:--|
| `scripts/tools/h2_terminal_partition.py` | the §7 partition, executable; the only terminal authority |
| `scripts/tools/run_h2_layer_p.py` | Layer-P controller; emits `h2_layer_p_certificate_v2` |
| `scripts/tools/h2_behavioral_identity.py` | bounded MOT17-09 probe (four A7.6 members, capture-off) |
| `scripts/tools/h2_runtime_inputs.py` | `h2_runtime_input_manifest_v1` content binding |
| `scripts/tools/h2_path_partition.py` | the retry firewall; total, mutually exclusive, fail-closed |
| `scripts/tools/build_runtime_identity.py` · `check_runtime_identity_staleness.py` | coordinate/probe publisher and staleness verdicts |
| `docs/reference/runtime_identity.generated.json` | published coordinate + probe |
| `../contracts/runtime_identity_bindings_v1.json` | `captured_under` sidecar home |
| `.github/workflows/runtime_identity.yml` | controlled-host re-attestation (same-repo PR head + main) |

## Current step

**Scope the Layer-M controller. Do not implement it yet, and do not seal.**

The reuse pattern is already established and should not be re-litigated:
`run_h2_layer_p.py:351` imports `run_h0_phase_a` as a library and uses exactly
`BoundInputMonitor` + `DriftError`, never modifying it. H0's controller is
~4360 lines and its verifier ~2477, but the majority of that is the enumerative
provenance apparatus H2 **deleted**, so the size is not the thing being inherited.

**Reuse unmodified (import, never edit — the frozen files stay hash-pinned):**

- `run_h0_phase_a.BoundInputMonitor` / `DriftError` — mutation detection, the
  `bound_input_mutated` predicate;
- `run_h0_phase_a` canonical-JSON / digest helpers — the §8.1 frozen convention;
- `run_h0_phase_a` child/evaluator argv, launch and deadline helpers — the four
  ordered runs;
- `h2_runtime_inputs` — manifest build, validate, post-run revalidate;
- `h2_behavioral_identity` — the launch-time probe, the
  `behavior_probe_equals_freeze` predicate;
- `h2_terminal_partition.select_terminal` — the only terminal authority;
- the A7.6 comparison inventory and the packet verifier, consumed verbatim (§6);
  no tolerance, proxy, or extra counter may be substituted for either.

**Deliberately not ported** (this is the size reduction, and each omission is a
declaration consequence, not a shortcut):

- loaded-closure enumeration — `discover_python_interpreter_runtime_paths`,
  `_dynamic_dependencies`, `_runtime_maps_dependencies`: `provenance_invalid` is
  gone as a predicate, so closure membership is not computed at all;
- `repository_inventory` / `bound_inventory_digest` / `validate_bound_inventory`
  — replaced by `h2_runtime_inputs` content binding;
- `owner_authority_overlay` / `verify_owner_authority_overlay_s_bytes` /
  `_discover_controller_input` / `preflight_controller_input` — the A10 overlay
  landing machinery; H2's seal binds the Layer-P certificate instead;
- `_elf_build_id`, `_tool_version`, `nvml_gpu_inventory`,
  `_collect_runtime_attestation` — witness-only under §4.1: recorded, never
  predicates, never terminal-selecting;
- `_checkpoint_inventory_verdict` and H0's A2 three-attempt coverage budget — H2
  has no coverage attempts.

**Must be newly written:**

1. the four ordered runs `00_capture_off`, `01/02/03_capture_on` on MOT17-04-SDP
   under the unmodified A5 target with `SACCADE_GPU_DECODE=1` (§3.3);
2. the A7.6 seven-member capture-off/on comparison → `capture_off_on_equal`;
3. three capture-on packets + packet-verifier invocation → `packets_valid`;
4. the recorder normalization of §5.1.1 — write `sorted(raw, key=slot)` and
   assert *no duplicate slot*, never the native `unordered_map` iteration order;
   the frozen `run_h0_phase_a_child.py` is **not** edited;
5. evidence root `h2_measure_<I40>/` with manifest and checksum inventory;
6. `verify_h2_measurement.py` — independent verifier;
7. `check_h2_measure_archives.py` — a **new** corpus checker;
   `check_h0_phase_a_archives.py` is untouched and keeps verifying the frozen v1
   corpus under the v1 schema;
8. an observation emitter producing exactly `ORDERED_PREDICATES` plus an optional
   `execution_result`, so the controller cannot express a terminal the partition
   does not define.

## Acceptance

A Layer-M seal may be proposed only when **all** of these hold:

1. the Phase-B chain form is declared (decision 1's precondition) — an
   append-only declaration correction, a `sealed_prefix` re-pin, and a
   `runtime_identity.generated.json` republish;
2. the S4 items above are implemented, tested, and reviewed;
3. a Layer-P pass certificate (`h2_layer_p_certificate_v2`) exists for the exact
   head under seal, with `--base` given and the full changed-path verdict clean;
4. the published coordinate is current and the controlled-host workflow is green
   at that head;
5. the owner issues a separate exactly-once authorization — this charter is not
   one and cannot become one.

Terminal acceptance, when a terminal is eventually selected, is recorded by the
[claim-state registry](../contracts/claim_state_registry.md), not here.

## Must not

Does **not**: change any preset default, kernel constant, or capture ABI; enable
any capture by default; unblock `H0_ROUTE5_B1`, `GCTM_B1`, or O1; touch any
`h0_phase_a_*` / `h0_preseal_freeze_*` evidence root; alter H0's declaration
bytes (declaration §0.2 explains why an Amendment 11 is mechanically
inadmissible); or establish any guarantee retroactively.

Additionally, and specific to the accepted decisions:

- a probe equality may never be reported as behavior preservation, measurement-
  domain equivalence, or an equivalence upgrade — `equivalence.state` is
  `unproven` in v1 and there is no upgrade path without a versioned verifier;
- `captured_under` may not be written to any existing registry row, and may not
  be written at all until H2 evidence exists;
- no witness field (ELF build-id, host `tool_runtime`, observed file closure,
  NVML identity) may select a terminal or block a Layer-P retry;
- a successful Phase A may not be reported as terminal 5, as partial capture, or
  as any mainline transition.

## History

- **2026-07-25** — S0 landed: this charter, the H2 declaration, navigation, and
  the registry field proposal.
- **2026-07-25** — [PR #287](https://github.com/raylei50653/saccade/pull/287)
  merged (`9edfec57`): S1–S3 plus the S4 terminal-partition module. The review
  reset split runtime identity into `coordinate` / `probe` / `equivalence`,
  removed the behavior-preserving shortcut, reclassified `datasets/`, `models/`,
  `runs/`, `build/`, `third_party/` as execution inputs, made probe intake
  fail-closed, bound the ruler and the Layer-P certificate, and closed the
  manifest/monitor TOCTOU (declaration Review Correction 1).
- **2026-07-25** — [PR #288](https://github.com/raylei50653/saccade/pull/288)
  merged (`feb99732`): the controlled-host workflow no longer hardcodes an
  absolute resource root; runtime inputs bind via `ln -s --relative`; runtime
  identity republished.
- **2026-07-25** — controlled-host re-attestation green on `main` at `feb99732`
  (workflow run `30156797208`): fresh extension build, bound runtime inputs,
  MOT17-09 probe = `2dabed0bc05e3bc7…`, `--strict` staleness pass. This is the
  fourth physically distinct build to reproduce the G1 probe and it closes PR
  #288's open test-plan item.
- **2026-07-25** — owner accepted all four decisions on
  [#286](https://github.com/raylei50653/saccade/issues/286); decision 1 accepted
  with the Phase-B-chain precondition recorded above. Next concrete work: scope
  the Layer-M controller (`Current step`), not implement or seal it.
