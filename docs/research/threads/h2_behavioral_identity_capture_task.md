---
doc-status: proposed
doc-promotion: navigation-only; not evidence
doc-date: 2026-07-25
doc-module: semantic
owner-module: semantic
work-class: mainline-study
wip-role: non-wip
activation-gate: "四項 owner decision 已於 2026-07-25 accepted（#286）；Phase-B chain form 已於 2026-07-25 accepted（#290）並成文為 declaration Review Correction 3；C3.9 pre-seal ruler 編輯（七序列 manifest + phase_a_evidence schema + phase-aware partition）已 landed、republish，並於 `7c348f4e` 通過 controlled-host re-attestation（run 30164262580）；Layer-M review 又把語義常數搬回 ruler ⇒ identity_semantics 再動一次（`3edf6953`）已 republish，並於 `f2c2510a`（run 30243657973）重新通過 controlled-host re-attestation；PR #295 第二輪修補再次修改 packet-invalid exception boundary 與 terminal-2 surviving-evidence ruler，identity_semantics `93f87a83` 已 republish，並於 `32935d5d`（run 30254189532）重新通過 controlled-host re-attestation ⇒ gate 2 再度關閉；S4 Phase-A controller（items 0–4）已 implemented/tested，**review 已完成、PR #295 已 merge ⇒ S4 code-review gate closed**（reviewed implementation head `4c78b962…`，landed merge commit `b2f3c23f…`）；Acceptance items 4→5→6 已於 2026-07-27 在 head `0a5dffe9` 全部達成（controlled-host run 30276844285 綠、Layer-P certificate `d95859cb…` 37/37 獨立驗證、F64 `a03fc459…` 22/22、owner single-invocation authorization），並已執行一次 ⇒ **authorization spent、controller terminal `H2_INPUT_MUTATED_DURING_MEASUREMENT`、0/4 ordered runs started、no capture、seal 未完成**；adjudicated root cause = controller self-mutation（在 repo 內建 evidence root 後又要求 checkout 乾淨），非 head/binding/ruler 問題 ⇒ **items 4–6 satisfied-then-void，下一步是 successor head 上的 controller repair，之後整個 acceptance + authorization 週期全部重建**；失敗證據見 evidence/h2_phase_a_failed_attempt_0a5dffe9_20260727/"
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
- **declaration Review Correction 2** owns the record of the §9 decisions'
  resolution — it is the only document that can supersede §9 inside its own
  authority — so everything this charter says about that resolution is a
  navigation projection of it;
- this charter owns only navigation and the staging order.

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
| **S4** | ⛔ **implemented / reviewed / merged — executed once and defective** — the one authorized Phase-A invocation reached terminal 1 with 0/4 ordered runs started; repair required before any further attempt — terminal partition, evidence-root contract, independent verifier, corpus checker, observation emitter, and the Phase-A controller with its four ordered runs. This implementation creates no `I`/`F`/`S` and performs no authorized measurement | no |

### S4 — what is and is not implemented

**Landed.** `scripts/tools/h2_terminal_partition.py` is the ordered partition of
declaration § 7 as executable, tested code: first-applicable selection,
exhaustive result mapping with the mandatory execution catch-all, fail-closed on
missing or non-boolean predicates, and a test proving **witness fields cannot
select a terminal**. This is the epistemic core and the one load-bearing owner
decision; § 20.8's two-implementer test is not meetable from a table alone.

**Also landed — the artifact side of the build** (items 5–8 of `Current step`):

- `h2_measurement_evidence.py` — the evidence-root contract: the § 9 item 3 /
  § C3.1 root names with the complete `F64` recomputed rather than truncated, the
  record set, the checksum inventory, and the **observation emitter**, which
  carries exactly `ORDERED_PREDICATES` plus an optional `execution_result` so a
  controller can express neither more nor less than the partition decides;
- `verify_h2_measurement.py` — the independent verifier. Independent means it
  recomputes: the terminal is re-selected from the archived observation, the A7.6
  comparison is rebuilt from the archived policy inventories, and every capture-on
  packet is re-verified through the packet verifier. § 6 is honoured by import —
  the A7.6 member set, the policy-inventory shape and the packet predicates come
  from H0's frozen implementation, not from a re-typed copy. It also implements
  § C3.5.1's three verify classes and the kill-switch;
- `check_h2_measure_archives.py` — the new corpus checker: classification into
  `complete` / `envelope` / `unterminated` / `inadmissible`, the § C3.1 root-name
  digest recomputation, `prior_attempts` completeness and ordering, and the § C3.5
  ban including the "changed surface is necessary, never sufficient" guard against
  H0 § 6's own repair vocabulary. An empty corpus passes: no Layer-M
  authorization has ever been issued.

All three are `plumbing_only` and a test scans **all three** for restated ruler
facts — C3.9's trap, made mechanical rather than remembered. The first draft
failed that scan: the A7.6 equality and projection members, the overflow zero
vector, the policy-inventory schema, the surface-ban terminals and H0 § 6's
repair vocabulary had all been typed out in the verifier and the corpus checker,
where an edit would have changed what re-attempt is admissible while
`identity_semantics` stood still. They now live in the ruler
(`h2_behavioral_identity.py` for the A7.6 member definitions,
`h2_terminal_partition.py` for the verify classes, surface-ban terminals, repair
vocabulary and completion keys) and are published in `as_payload()` so both
consumption paths agree.

Three properties the verifier owes its name:

- **the Phase-A terminal-1 inputs are cross-checked, not merely inventoried.**
  The archived freeze/certificate/content bindings, checkout-identity witness,
  launch probe, mutation record and controller result must agree with the
  observation and the independently selected terminal. The verifier rebuilds
  `source_tree` and the decision-relevant, identity-semantics and plumbing file
  sets from the bound Git commit; it does not import the controller's certificate
  evaluator. The recorded predicate must equal that recomputation in both
  directions. Table-driven tamper tests pin every certificate condition;
- **the admission gate is recomputed, not read.** § C3.6's five conditions are
  rebuilt from the bound Phase-A root, both freeze records, the archived Layer-P
  certificate and the prior-attempt chain, and must equal the record condition
  for condition. A verifier that recomputes the terminal while believing
  `admission.json` lets the archive attest its own right to have spent `S_B`.
  Condition (e) is **discovered, not walked**: the corpus is scanned for the
  consumed attempts of this Phase-A result, because a `prior_attempts` list
  cannot establish its own completeness, and an omitted predecessor or an
  `inadmissible` root inside the chain is exactly what makes an attempt
  ineligible. The chain rule lives in one place (`verify_prior_chain`) and the
  corpus checker applies it rather than restating it;
- **surviving evidence accumulates monotonically.** Replay is per run, and a
  later missing artifact can no longer erase an inequality or an invalid packet
  already found — which is the only way § C3.5.1's kill-switch actually bans
  anything.

**Implemented, reviewed, and merged in [PR #295](https://github.com/raylei50653/saccade/pull/295).** `run_h2_measurement.py`
is a Phase-A-only controller and `run_h2_measurement_child.py` is its dedicated
child/recorder. Together they implement the four ordered runs, § 5.1.1 recorder
normalization, live A7.6 comparison and packet-verifier wiring, and emission of
the artifacts the three tools above verify. The controller consumes an external
freeze, exact-head Layer-P certificate, reference probe, runtime-input manifest
and published identity; it does not create or seal `I`, `F`, or `S`, and it has
no Phase-B launch path.

The raw MOT output is atomically committed and directory-fsynced first; that
final path is the independent commit point for the `mot_output` A7.6 member.
`policy_base_inventory.json` is then committed by same-directory temporary
write, file fsync, atomic replace and directory fsync; that replace is the
commit point for all four base members. Every other surviving JSON record uses
the same atomic protocol. If execution stops between the two base commit
points, controller and verifier replay the MOT-only survivor rather than erase
its equality relation. The raw capture is written before canonicalization,
projection or replay. Packet-data shape, numeric and binary-layout exceptions
declared by `h2_behavioral_identity.py` record `packet_invalid`; unclassified
implementation exceptions remain terminal-4 execution failures. A packet
failure never fabricates projection or overflow fields and never deletes the
already-known base equality evidence. After any child outcome, including
nonzero, the controller replays every surviving base/full inventory and packet
before applying terminal priority.

The H2 measurement invocation ends at a clean final monitor drain. With the
monitor still active, the controller revalidates every bound launch record and
the checkout head/tree/cleanliness; it then performs the final nonblocking drain.
A drain with no mutation event is the common linearization point. The monitor is
closed only afterward, and all later writes are evidence-only. Thus a mutation
during the sequential scan is caught by the monitor, while a mutation after the
clean drain is explicitly outside the invocation. Failure, mismatch or an event
during this stop protocol has terminal-1 priority.

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

## Owner decisions — resolved 2026-07-25

**Navigation projection.** The fact-owner is declaration
[Review Correction 2](../../modules/semantic/research/headline_bridge_behavioral_identity_capture_declaration_20260725.md),
which closes §9 inside the declaration's own authority; the decision surface was
[#286](https://github.com/raylei50653/saccade/issues/286), now resolved and
closed. **If anything below conflicts with Correction 2, Correction 2 wins** —
this section carries no verdict of its own and must be updated in the same change
as any correction that supersedes it.

All four §9 decisions were accepted on 2026-07-25. Acceptance authorizes nothing:
no `I`, no `F`/`S`, no capture, no exactly-once grant, no registry state write.
Correction 2 carries the verdict table, the §5.2 precondition it adds, and the
§8.4 write condition it fixes.

The two consequences a reader of this charter needs in order to navigate the
staging order are projected below.

### Decision 1's precondition — the Phase-A success path

*Projection of Correction 2's §5.2 precondition 6 and of **Correction 3**, which
satisfies it. Reasoning and normative text live there.*

A fully successful Phase A selects no terminal — `h2_terminal_partition.py` maps
`measurement_pass` to no terminal, and terminal 5 additionally requires the
frozen seven-sequence Phase B, which §2 does not authorize and §7 forbids Phase A
from starting. Without a Phase-B chain, the best available outcome of the one
authorized invocation is a spent authorization, an unclosed unit, and no mainline
transition (§20.7).

So the staging order gained a gate: **the Phase-B chain — its own `I → F → S`, its
authorization form, and its precondition on a passing Phase A — must be published
before the Phase-A seal.** Seal scope is unchanged; the success path simply has
to exist before an authorization can be spent reaching it.

**Status: published.** The chain form was accepted on
[#290](https://github.com/raylei50653/saccade/issues/290) (2026-07-25, with three
narrowings; the issue stays open as the standing surface) and is written as
declaration **Review Correction 3** — `I_B = (I40_B, F_B)`, the `F_B` freeze list,
`S_B`, the result mapping, the measurement-surface re-attempt rule, the
pre-terminal admission gate, and the C3.9 pre-seal edit list. What precondition 6
asks for now exists, and the ruler work C3.9 schedules has since landed; what
remains is review of the implemented Phase-A Layer-M plumbing plus the
exact-head seal gates below.

Two consequences a reader of this charter must not get wrong:

- **terminal 2 is reachable in Phase B.** All seven sequences run
  `00_capture_off` and the A7.6 comparison is live on each, because H0's
  terminal-5 semantics require the non-perturbation bars to pass for every
  sequence (C3.4);
- **`identity_semantics` is frozen from the Phase-A seal to the Phase-B seal**
  (C3.1(b) requires coordinate equality across both phases), so every ruler edit —
  including any further re-pin of the declaration's `.policy.yaml` — must land
  before Phase A seals. Editing the ruler after that seal invalidates the Phase-A
  result the chain is built to consume (C3.9).

## Registry field `captured_under`

*Projection of Correction 2's §8.4 write condition. The mechanical authority for
the rule is the sidecar `../contracts/runtime_identity_bindings_v1.json`, not
this charter and not registry Markdown.*

Accepted as **schema only** against `quantity.bridge_capture_provenance`
([registry](../contracts/claim_state_registry.md)); **state, terminal, cause, and
`open_limits` unchanged**:

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

One write condition, and nothing may be inferred beside it:

```text
a binding appears only if an H2 Layer-M measurement reaches terminal 5
and an owner accepts it
```

The sidecar row already exists with `captured_under: null`, which means *no
substrate-version claim* and is never read as agreement. Terminals 1–4 record a
capture coordinate inside their own packet; that is never promoted to an
object-level binding, and no retroactive binding is written.

Consumption rule: decision-surface, identity-semantics or probe drift is stale.
Implementation, environment or runtime-input drift with an equal probe is
`re_attestation_required`, never behavior-preserving. V1 defines no equivalence
upgrade. Version-lag accounting follows
[ADR 020 §S2](../../decisions/020-doc-lifecycle-new-nogo.md).

## Expected state (lease)

*Planning target only; not accepted state; replaceable inside this charter
without a registry write.*

The previous lease — the implemented Phase-A Layer-M build is **reviewable but
not sealed**, surviving review or being replaced — was **met on 2026-07-27**: it
survived review and landed (`Acceptance` item 3), with no `I`/`F`/`S`, no
authorization, no capture, and no registry state change.

That lease was in turn **met and then spent on 2026-07-27**: the head was
attested, certified and independently verified, authorized once, and executed —
and the invocation reached terminal 1 with no capture. See `History` and the
[failure evidence](../../modules/semantic/research/evidence/h2_phase_a_failed_attempt_0a5dffe9_20260727/).

By the next commit point, the controller is expected to be **repaired on a
successor head, unbound and unauthorized**: the evidence-root placement and the
predicate ownership fixed, the launch → stop-boundary ordering re-reviewed as a
sequence rather than as isolated predicates, and no acceptance, certificate,
freeze or authorization carried over from `0a5dffe9`. That state is still not an
authorization and still writes no registry state.

## Commit point

Owner reviews whether state changed when **either** a repaired controller
demonstrates a reachable success path through its own launch → stop-boundary
ordering at a successor head — a demonstration that is not an execution and
spends nothing — **or** the repair shows the two-layer design cannot hold the
invariants it declares, and the charter is re-planned. The spent `0a5dffe9`
attempt (2026-07-27) was the previous commit point: it changed no claim-state
object and produced no measurement. Repair is not a seal gate reopening; a new
seal requires the whole acceptance chain rebuilt at the repaired head, and the
separate owner exactly-once authorization remains a distinct step that no
artifact in this repository can supply.

## Discard when

- ~~the Phase-B chain form cannot be declared on terms the owner accepts~~ —
  **resolved 2026-07-25**: accepted on #290 and published as Correction 3. Its
  successor condition — that the C3.9 pre-seal ruler edits cannot be completed
  before the Phase-A seal — is **also spent**: those edits landed on 2026-07-25.
  Neither can discard this unit again; or
- a G1 probe inequality appears across builds of identical decision-relevant
  source — the design premise is falsified and the coordinate/probe split needs
  rework before anything downstream; or
- an owner decision elsewhere retires observational shadow capture at this ABI,
  which is terminal 2's transition reached without spending an invocation.

## Read first

1. [H2 declaration](../../modules/semantic/research/headline_bridge_behavioral_identity_capture_declaration_20260725.md)
   — §0.1 supersession boundary, §4 coordinate/probe/equivalence, §5 two-layer
   budget, §7 partition, Review Correction 1, **Review Correction 2**, which
   closes §9, and **Review Correction 3**, the Phase-B chain. Read Correction 2
   before §9: §9 is retained as the historical statement of the question and is no
   longer an open-decision authority. Read Correction 3 before §7: §7 alone does
   not describe how Phase B is entered, what a Phase-B terminal closes, or which
   pre-launch failures cost no authorization.
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
| `scripts/tools/h2_measurement_evidence.py` | evidence-root contract + the observation emitter |
| `scripts/tools/run_h2_measurement.py` | Phase-A Layer-M controller; validates externally supplied inputs and emits one evidence root |
| `scripts/tools/run_h2_measurement_child.py` | H2-specific child/recorder; normalized active-pair inventory and frozen A5 execution |
| `scripts/tools/verify_h2_measurement.py` | independent verifier; the three § C3.5.1 verify classes |
| `scripts/tools/check_h2_measure_archives.py` | corpus checker; `prior_attempts` and the § C3.5 ban |
| `scripts/tools/run_h2_layer_p.py` | Layer-P controller; emits `h2_layer_p_certificate_v2` |
| `scripts/tools/h2_behavioral_identity.py` | bounded MOT17-09 probe (four A7.6 members, capture-off) |
| `scripts/tools/h2_runtime_inputs.py` | `h2_runtime_input_manifest_v1` content binding |
| `scripts/tools/h2_path_partition.py` | the retry firewall; total, mutually exclusive, fail-closed |
| `scripts/tools/build_runtime_identity.py` · `check_runtime_identity_staleness.py` | coordinate/probe publisher and staleness verdicts |
| `docs/reference/runtime_identity.generated.json` | published coordinate + probe |
| `../contracts/runtime_identity_bindings_v1.json` | `captured_under` sidecar home |
| `.github/workflows/runtime_identity.yml` | controlled-host re-attestation (same-repo PR head + main) |

## Current step

**Repair the controller on a successor head. Do not re-run, do not seal, and do
not treat any `0a5dffe9` binding as carried over.** The one authorized Phase-A
invocation was spent on 2026-07-27 and reached terminal 1 with zero ordered runs
started; the cause was the controller's own launch sequence, not the head, the
bindings or the ruler. What is left is code-bound again — see `History` and the
[failure evidence](../../modules/semantic/research/evidence/h2_phase_a_failed_attempt_0a5dffe9_20260727/)
for the two defect sites and the ordering rule they establish. The head-bound
gates below (`Acceptance` items 4 and 5) were satisfied at `0a5dffe9` and must be
rebuilt from scratch at whatever head the repair lands on.

Correction 3 puts an ordering constraint on everything below: `identity_semantics`
must be frozen from the Phase-A seal to the Phase-B seal, so the ruler work comes
first and the controller work may follow (it is `plumbing_only` and moves no
axis).

**The ruler work has landed.** `h2_runtime_inputs.py` now binds all seven
`MEASUREMENT_SEQUENCES` in both phases and carries the `phase_a_evidence`
section, which is bound and watched while belonging to neither digest — the two
member sets are named once, in `COORDINATE_SECTIONS` / `FULL_DIGEST_SECTIONS`, so
a later section cannot land in the wrong digest by omission.
`h2_terminal_partition.py` now requires an explicit `phase`, carries per-phase
terminal conditions and `PHASE_COMPLETION` counts (Phase A: 1 sequence / 3
capture-on packets / 1 capture-off run; Phase B: 7 / 21 / 7), and implements the
C3.6 gate as `evaluate_admission` — a pre-terminal object whose refusal selects
no terminal and which `select_terminal` requires before any Phase-B selection.
`.github/workflows/runtime_identity.yml` binds all seven sequences on the
controlled host. The coordinate is republished and the controlled-host
re-attestation is **green at that head** — `7c348f4e`, workflow run
[`30164262580`](https://github.com/raylei50653/saccade/actions/runs/30164262580) —
so the ruler gate is closed for that head, and no ruler path has changed since.

C3.9's trap applied to that work directly and was honoured: a *new*
`scripts/tools/h2_*.py` file classifies as `plumbing_only`, so the admission and
phase logic went inside `h2_terminal_partition.py` rather than into a new module
that would have moved the ruler inside the frozen window with nothing to catch
it. The same rule binds the Layer-M work below.

The reuse pattern is already established and should not be re-litigated:
`run_h2_layer_p.py:351` imports `run_h0_phase_a` as a library and uses exactly
`BoundInputMonitor` + `DriftError`, never modifying it. H0's controller is
~4360 lines and its verifier ~2477, but the majority of that is the enumerative
provenance apparatus H2 **deleted**, so the size is not the thing being inherited.

**Reuse unmodified (import, never edit — the frozen files stay hash-pinned):**

- `run_h0_phase_a.BoundInputMonitor` / `DriftError` — mutation detection, the
  `bound_input_mutated` predicate;
- `run_h0_phase_a` canonical-JSON / digest helpers — the §8.1 frozen convention;
- `run_h0_phase_a`'s **fixture-agnostic** process primitives —
  `_bounded_remaining`, `_deadline_checked_call`, `_wait_with_monitor`,
  `_terminate_process_group`;
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
  has no coverage attempts;
- **`child_argv` and `evaluator_argv`** — these are not generic launchers.
  `child_argv` hardcodes `scripts/tools/run_h0_phase_a_child.py`
  (`run_h0_phase_a.py:1207`) and `evaluator_argv` is bound to
  `EVALUATOR_ARGV_PREFIX` and H0's run-directory layout. H2 writes its own child
  argv and its own child/recorder.

  This is forced, not preferred. The frozen child takes the native pairs and
  raises inside its **own** recorder if they are not already slot-sorted
  (`run_h0_phase_a_child.py:444`), so no outer layer can supply §5.1.1's
  normalization — reusing that argv and normalizing the order are mutually
  exclusive. Writing an H2 child is what makes the normalization reachable while
  leaving the frozen child untouched.

**Newly written, reviewed, and merged in [PR #295](https://github.com/raylei50653/saccade/pull/295):**

0. ✅ an H2 child entrypoint and its argv builder, replacing the two H0 argv
   helpers above;
1. ✅ the four ordered runs `00_capture_off`, `01/02/03_capture_on` on MOT17-04-SDP
   under the unmodified A5 target with `SACCADE_GPU_DECODE=1` (§3.3);
2. ✅ the A7.6 seven-member capture-off/on comparison → `capture_off_on_equal`;
3. ✅ three capture-on packets + packet-verifier invocation → `packets_valid`;
4. ✅ the recorder normalization of §5.1.1 — write `sorted(raw, key=slot)` and
   assert *no duplicate slot*, never the native `unordered_map` iteration order;
   the frozen `run_h0_phase_a_child.py` is **not** edited;
5. ✅ evidence root `h2_measure_<I40>/` with manifest and checksum inventory —
   landed as `h2_measurement_evidence.py`, with § C3.1's Phase-B root name and
   its complete-`F64` rule;
6. ✅ `verify_h2_measurement.py` — independent verifier, plus § C3.5.1's three
   verify classes and the kill-switch;
7. ✅ `check_h2_measure_archives.py` — the **new** corpus checker;
   `check_h0_phase_a_archives.py` is untouched and keeps verifying the frozen v1
   corpus under the v1 schema;
8. ✅ an observation emitter producing exactly `ORDERED_PREDICATES` plus an
   optional `execution_result`, so the controller cannot express a terminal the
   partition does not define.

Items 5–8 landed before items 0–4 by design. The artifact contract was fixed
before the controller, so the controller is reviewed against that target rather
than the target being adjusted to whatever the controller happened to emit —
the § 5.3 circular-oracle hazard in its most ordinary form.

### The remaining work is head-bound, and the head is not the reviewed head

> **Spent at `0a5dffe9` on 2026-07-27.** Steps 1–5 below were all performed and
> independently verified at that head, and step 6 executed once — reaching
> terminal 1 with zero ordered runs started. The sequence itself was sound; it
> ran into a defect *inside* the controller (`History`, and §4 of the
> [failure evidence](../../modules/semantic/research/evidence/h2_phase_a_failed_attempt_0a5dffe9_20260727/)).
> Nothing in this ordering carries over: after the repair, every step below must
> be redone at the successor head, against a controller that has first shown a
> reachable success path through its own launch → stop-boundary ordering.

Every gate below binds **one exact commit**, so the order is fixed and no step may
be carried over from an earlier head:

1. stop modifying `main`;
2. the published coordinate/probe current and controlled-host re-attestation green
   at that final head (`Acceptance` item 4);
3. a Layer-P pass certificate produced **at that same head**, `--base` given, full
   changed-path verdict clean (`Acceptance` item 5);
4. independent verification of every certificate binding (also item 5);
5. only then a separate owner exactly-once authorization (`Acceptance` item 6);
6. only after that authorization, `F`/`S` and the Phase-A execution — not an
   `Acceptance` item, because that list ends at proposing a seal.

The reviewed implementation head `4c78b962…` is **not** a seal candidate, and
neither is the merge commit that landed it. Content equality is not head equality:
PR #295 landed as `b2f3c23f…` with a tree byte-identical to `4c78b962…`, and this
governance closeout produces yet another commit over that same code. The
certificate records `source_head` *and* `source_tree` (`run_h2_layer_p.py:472`),
the controller requires `certificate["source_head"] == bundle.head == current_head`
(`run_h2_measurement.py:249`), and the evidence root is keyed by that head's `I40`
(`verify_h2_measurement.py:342`) — so a certificate issued at any predecessor is
mechanically unusable at the head that is actually sealed, identical trees or not.
The seal candidate is therefore whatever `main` head is final after documentation
stops moving, which is the head step 2 must attest.

## Acceptance

A Layer-M seal may be proposed only when **all** of these hold:

1. ✅ the Phase-B chain form is **published** — declaration §5.2 precondition 6
   (Correction 2) is satisfied by **Review Correction 3**, accepted on
   [#290](https://github.com/raylei50653/saccade/issues/290) on 2026-07-25;
2. ✅ the **C3.9 pre-seal ruler edits** are landed, republished, and re-attested
   green on the controlled host — `h2_runtime_inputs.py` (seven sequences plus the
   `phase_a_evidence` schema/producer, which by C3.8 must move no published axis)
   and `h2_terminal_partition.py` (phase-scoped terminal-1 metadata, phase-aware
   terminal-5 metadata, the `phase` argument, updated tests). These are ruler
   files: after the Phase-A seal they cannot be touched at all. **Landed,
   republished, and re-attested green on 2026-07-25 at `7c348f4e` (run
   `30164262580`: runtime inputs re-bound on the controlled host,
   `coordinate_digest 0b839df0…`, probe recomputed, `--strict` staleness pass).
   The gate is closed for that head only — any later ruler edit reopens it, and
   one did: the Layer-M review moved the A7.6 member definitions, the verify
   classes, the surface-ban terminals and H0 § 6's repair vocabulary out of the
   `plumbing_only` files and into the ruler, so `identity_semantics` moved
   `67b35d8f` → `3edf6953` and was republished. Only that axis moved and the
   probe is unchanged. Re-attested green at `f2c2510a` (run `30243657973`,
   attesting the merge tree `f04d4799`): runtime inputs re-bound, probe
   recomputed to `2dabed0bc05e3bc7…` — the seventh physically distinct build to
   reproduce it — and `--strict` staleness pass;**
3. ✅ the S4 items above are implemented and contract-tested, and the **S4
   code-review gate is closed**: reviewed implementation head `4c78b962…`, landed
   as merge commit `b2f3c23f…` on `main` via
   [PR #295](https://github.com/raylei50653/saccade/pull/295) on 2026-07-27. This
   closes review of the *code*; it certifies no head and authorizes nothing;
4. the published coordinate and bounded probe are current and the controlled-host
   workflow is green at the exact final seal-candidate head. **Satisfied at
   `0a5dffe9` (run `30276844285`) and now void**: it certifies that head, and the
   repair lands on another one;
5. a Layer-P pass certificate (`h2_layer_p_certificate_v2`) exists for that **same**
   head, with `--base` given and the full changed-path verdict clean — and its
   bindings independently verified. Neither `4c78b962…` nor `b2f3c23f…` can supply
   this for a later head: `source_head` is part of the certificate and is required
   to equal the executing head, so identical content does not transfer (see
   `Current step`). **Satisfied at `0a5dffe9` — certificate `d95859cb…`,
   `selected_base b2f3c23f419cb03c…`, 37/37 bindings independently verified — and
   void for the same reason as item 4**;
6. the owner issues a separate exactly-once authorization — this charter is not
   one and cannot become one. **Issued and consumed on 2026-07-27.** It is spent:
   it authorized one invocation at `0a5dffe9`, that invocation happened, and no
   part of it survives to the repaired head.

These are numbered in the order they must be performed: 4 → 5 → 6 is the same
sequence as `Current step` steps 2 → 3–4 → 5, and no item may be satisfied at an
earlier head than its predecessor. The Phase-A execution itself is not on this
list — `Acceptance` governs only when a seal may be *proposed*; execution follows
the authorization of item 6.

**Items 4–6 are satisfied-then-void, not open.** A repaired controller moves
execution-relevant code, so the head changes and all three must be rebuilt from
scratch there; a repair PR may not present itself as continuing the `0a5dffe9`
attempt. Item 3 is unaffected — it closed review of code that the repair will now
modify, and the repair carries its own review.

A Phase-B seal has its own gates, and they are Correction 3's, not this list's:
`I_B`, `F_B` and `S_B` per C3.1–C3.3, the C3.6 admission gate passing before
`S_B` is consumed, and the C3.5 re-attempt rule if a prior Phase-B terminal
exists.

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
- `captured_under` may not be written for any terminal other than an
  owner-accepted terminal 5, and no retroactive binding may be written for
  evidence predating published identities; a terminal-1–4 packet's own capture
  coordinate may not be promoted into the object-level sidecar;
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
  with the Phase-B-chain precondition. Recorded in the fact-owner as declaration
  **Review Correction 2**: §9 closed, §5.2 precondition 6 added, §8.4's write
  condition fixed to owner-accepted terminal 5. Next concrete work: scope the
  Layer-M controller (`Current step`), not implement or seal it.
- **2026-07-25** — PR boundary review found three P1 defects in the first draft
  of this charter, all fixed before merge: the Layer-M reuse list was
  unimplementable (`child_argv` hardcodes the frozen child, whose recorder
  raises on unsorted pairs before any outer normalization could run);
  `captured_under` carried three conflicting write rules; and the decision
  status had two live answers because the declaration still presented §9 as
  open.
- **2026-07-25** — the Phase-B chain form was proposed on
  [#290](https://github.com/raylei50653/saccade/issues/290), returned
  *changes required*, revised, and **accepted** with three narrowings: the
  evidence-root key uses the complete `F64` digest rather than a truncation;
  admission is an independent pre-terminal gate, so a failure spends no `S_B` and
  selects no terminal, and Phase-B terminal 1 means bound-input mutation only;
  and `phase_a_evidence` binds in `F_B` and the monitor watch set but never in the
  published `runtime_inputs` axis, since only its schema and producer can be
  frozen before Phase A has run. The review also corrected two errors of mine:
  keying the terminal-2/3 retry ban to the *coordinate* would have forbidden the
  capture-ABI-delta route terminal 3 exists to select (the key is the sealed `F_B`
  measurement surface, which `plumbing_only` code can move without moving an
  axis), and "the partition file needs no change" was false — terminal 5's
  mechanical condition still said *three* capture-on packets against Phase B's 21.
  Landed as declaration **Review Correction 3**, `sealed_prefix` re-pinned to
  61 605 bytes, runtime identity republished. The issue stays open as the standing
  surface for this chain.
- **2026-07-25** — PR #291 review returned *changes required* on the identity
  boundary, and Correction 3 was revised in place before merge. **P0:** C3.5's
  measurement surface named neither the executed extension nor the TensorRT
  plugin. The published `runtime_inputs` axis is the manifest's
  `coordinate_digest`, which excludes `build_artifacts` by design so one
  coordinate can span builds, and the only other binding — the Layer-P
  certificate's `runtime_input_full_digest` — was excluded from the surface as
  attempt-local. Terminals 2 and 3 were therefore declared permanent properties of
  a digest that could not see the binary that produced them, while §4.2 already
  says different native bytes can hold the bounded probe equal. The surface now
  names `full_digest`, and C3.8's axis membership — which had listed build
  artifacts as published — is corrected to match §4 and the implementation.
  **P1:** terminal 4 covers failures that can occur before an archive exists, so
  `prior_attempts` could be unformable after `S_B` was spent. New **C3.5.1** binds
  consumed-attempt records with a normative write-before-consume ordering and
  three verify classes (`complete` / `envelope` / `unterminated`), and closes the
  kill-switch: an unterminated attempt whose surviving artifacts already show
  perturbation or an invalid packet inherits the terminal-2/3 ban. Neither fix
  adds a pre-seal ruler edit. Re-pinned to 69 228 bytes, republished.
- **2026-07-25** — the same review's second round accepted both repairs and found
  one defect they had introduced: `S_B` had **two consumption points**. C3.3 said
  process launch; C3.5.1's ordering wrote the `authorization_consumed` record
  before it. The fail-closed direction was right but the authority state was not
  single-valued — a crash in that window was spent, unspent, or governed-as-spent
  depending on which clause an implementer read. Collapsed to one durable
  linearization point: the write-and-flush **is** the consumption, the launch
  follows it, and the one-authorization cost of a crash in that window is recorded
  as a deliberate loss rather than described away as a record of nothing. Re-pinned
  to 70 274 bytes, republished.
- **2026-07-25** — a second review round closed two governance defects. The
  decision resolution had **two fact-owners** — this charter claimed to own the
  decision list while Correction 2 claimed to be §9's single resolution pointer,
  and both carried a full verdict table; the charter is now a projection only.
  And the `sealed_prefix` lifecycle had two rules: the binding declares that,
  pre-seal, the pin is the entire document and every correction forces a
  conscious re-pin, while an intermediate draft of this charter claimed
  Correction 2 was free because it landed below the byte boundary. The declared
  rule wins. Correction 2 was re-pinned to the full body and
  `runtime_identity.generated.json` republished; the re-pin also brought
  `Review Correction 1` — appended in PR #287 without a re-pin — back inside the
  pin. A green suite never established the append was permitted: the byte test
  tolerates trailing content by construction, so only the binding's re-pin log
  can record that the pre-seal rule was honoured.
- **2026-07-25** — [PR #292](https://github.com/raylei50653/saccade/pull/292)
  merged (`7c348f4e`): the C3.9 pre-seal ruler edits — seven `MEASUREMENT_SEQUENCES`
  bound in both phases, the `phase_a_evidence` section bound and watched while
  belonging to neither digest, a required `phase` argument with per-phase terminal
  conditions and `PHASE_COMPLETION` counts, and `evaluate_admission` as the C3.6
  pre-terminal gate. Review found two defects, both fixed before merge: a clean
  Phase B could return no terminal at all after `S_B` was spent, and `as_payload()`
  did not publish the Phase-B narrowing, so two implementers reading the payload
  and the function would disagree. Coordinate republished (`identity_semantics`
  → `67b35d8f`).
- **2026-07-25** — controlled-host re-attestation green on `main` at `7c348f4e`
  (workflow run `30164262580`): runtime inputs re-bound with all seven sequences,
  `coordinate_digest 0b839df0…`, probe recomputed, `--strict` staleness pass. This
  closes the open half of acceptance gate 2 for that head. The gate is head-scoped:
  any later edit to a ruler file reopens it, and no ruler path has changed since.
  What remains before a seal may be proposed is the S4 Layer-M plumbing and a
  Layer-P pass certificate for the head under seal.
- **2026-07-27** — S4's artifact side landed: `h2_measurement_evidence.py`
  (evidence-root contract + observation emitter), `verify_h2_measurement.py`
  (independent verifier, § C3.5.1's three classes, the kill-switch) and
  `check_h2_measure_archives.py` (corpus checker, `prior_attempts`, the § C3.5
  ban), with 26 contract tests. Written before the controller deliberately: the
  artifact contract is what the controller is reviewed against, and deriving it
  from whatever a controller emitted is the § 5.3 hazard in its most ordinary
  form. Nothing here is a ruler edit — the three files classify as
  `plumbing_only` and a test pins both that classification and the absence of any
  phase, admission or terminal fact of their own (C3.9's trap). No `I`/`F`/`S`,
  no authorization, no capture, no registry write.
- **2026-07-27** — Layer-M review returned *changes required* on the artifact
  side, and all four findings were structural rather than missing coverage.
  **The § C3.6 gate was self-attested:** the verifier recomputed the terminal but
  read `admission.json`, so an archive could name a non-existent Phase-A root,
  record the condition as true, and verify — the one claim the gate exists to
  deny. Admission is now rebuilt from the bound Phase-A root, both freeze records,
  the archived Layer-P certificate and the prior-attempt chain, and must equal the
  record condition for condition. **Surviving evidence was not monotone:** replay
  discarded an already-found `fail` at the first missing later packet, and skipped
  a whole sequence for one missing inventory, so killing a process right after a
  perturbation laundered a banned terminal 2/3 into a re-attemptable 4 — the exact
  hole § C3.5.1 closes. Replay is per run now, and every comparison the survivors
  allow is made. **Semantic rules were sitting in `plumbing_only` files:** the
  A7.6 members, the surface-ban terminals and H0 § 6's repair vocabulary were
  typed out in the verifier and the checker, where an edit moves no axis; they
  moved to the ruler and the C3.9 scan now covers all three files, which caught
  two more while being written. **`inadmissible` was classified but never
  verified**, admitting a symlinked root, a Phase-A root carrying a Phase-B-only
  gate, or a name disagreeing with its freeze record; it is a verified class now.
  The ruler edits moved `identity_semantics` `67b35d8f` → `3edf6953`, republished
  with the probe unchanged, which **reopened acceptance gate 2**; it closed again
  when the controlled-host workflow went green at `f2c2510a` (run `30243657973`),
  reproducing `2dabed0bc05e3bc7…` for the seventh time from a physically distinct
  build.
  ⚠️ The general lesson is the one the charter already states and this PR still
  got wrong on the first pass: a file that moves no axis may hold no rule. Writing
  "this module holds no ruler" in a docstring is not the check; the scan is.

- **2026-07-27** — second review round closed the last admission gap: § C3.6(e)
  was still decided by walking the `prior_attempts` list `F_B` supplied, which
  cannot detect an omitted predecessor or an `inadmissible` root inside the chain
  — both were caught only later, by the corpus checker. `verify_prior_chain()`
  now discovers the chain by scanning the corpus for the consumed attempts of the
  bound Phase-A result, and the corpus checker calls that one rule instead of
  carrying a second implementation of "complete". ⚠️ The general lesson, and the
  one worth carrying into the controller: **a completeness check may not take its
  input from the object being checked.** Both defects in this round were the same
  shape — trusting a list to describe its own gaps.

- **2026-07-27** — implemented the Phase-A Layer-M controller without sealing or
  executing a measurement. `run_h2_measurement.py` consumes externally supplied
  freeze/certificate/identity inputs, launches exactly the four ordered runs,
  rechecks the A7.6 comparison and packet verdicts controller-side, and emits the
  fixed evidence contract. Its dedicated child performs § 5.1.1 slot
  normalization and duplicate rejection without editing H0's frozen child.
  CPU-injected contract tests cover the clean path and terminals 1–4, including
  mutation priority; the complete contract suite passed (822 passed, 4 skipped,
  3 xfailed). The tests also exposed and fixed a verifier replay defect:
  a present-but-invalid packet is complete evidence for terminal 3, not a
  missing packet. Self-review then closed the initial-intake/monitor TOCTOU
  window, added post-run runtime-input revalidation, and made the independent
  verifier cross-check the archived terminal-1 inputs rather than treating them
  as checksum-only files. Review, exact-head Layer-P certification,
  controlled-host re-attestation and separate owner authorization remain
  outstanding.

- **2026-07-27** — PR review remediation closed three producer/verifier gaps.
  The verifier now independently rebuilds the complete terminal-1 certificate
  predicate from an archived checkout witness plus the bound Git tree and
  requires exact bidirectional predicate equality. The child now persists raw
  capture before one total packet-processing operation, so structural
  canonicalization/projection/replay failures select terminal 3. The controller
  initially attempted a post-close scan as an invocation-end boundary. Second
  review showed that a sequential scan without the monitor cannot establish one
  common state, and that dropping the full inventory also dropped higher-priority
  base inequality evidence.

- **2026-07-27** — second-round remediation separates the four durable base
  A7.6 members from packet-derived projection/overflow, replays all surviving
  evidence after every child outcome, and preserves terminal 2 over terminal 3
  and 4. The stop protocol now revalidates while the monitor remains active and
  uses the clean final drain as its linearization point; the verifier enforces
  the exact v2 stop schema and `checkout_clean=true` on a clean progression.
  The exception/terminal-priority distinction moved `identity_semantics` to
  `93f87a83`; the publication was regenerated with the runtime-input coordinate
  and bounded probe digest unchanged, while equivalence remains `unproven`.
  At this point acceptance gate 2 was reopened. No Layer-P certificate or seal
  could be issued until the repaired head was republished and re-attested.

- **2026-07-27** — controlled-host re-attestation passed on the repaired code
  head `32935d5d` (run 30254189532), with the bounded probe and runtime-input
  coordinate equal to the republished identity. This closes acceptance gate 2
  for the repaired surface; S4 review, an exact seal-head Layer-P certificate
  and separate owner authorization remain outstanding. No certificate or seal
  was created by this run.

- **2026-07-27** — **the S4 code-review gate is closed.**
  [PR #295](https://github.com/raylei50653/saccade/pull/295) was reviewed at
  implementation head `4c78b962…` and merged to `main` as `b2f3c23f…`. This
  governance closeout records that state and nothing else: it edits no ruler, no
  controller and no verifier, moves no published axis, issues no certificate,
  creates no `I`/`F`/`S`, and grants no authorization. The remaining gates are
  `Acceptance` items 4 → 5 → 6 — coordinate/probe current plus controlled-host
  green, then the exact-head certificate and its independent verification, then the
  separate owner authorization — renumbered here to match that order, since the
  previous numbering listed the certificate before the attestation it depends on.
  All three bind **one** final seal-candidate head — which is not `4c78b962…` and not `b2f3c23f…`, because a
  certificate binds `source_head`, not content (see `Current step`). Until that
  head exists and is certified, verified, and separately authorized, no `F`/`S`
  may be created and Phase A may not run.
- **2026-07-27** — **the one authorized Phase-A invocation was spent and produced
  no capture.** `Acceptance` items 4 → 5 → 6 were all satisfied at head
  `0a5dffe921d78fce8e525baf8b4b624fc9ab957c`: controlled-host re-attestation green
  (run `30276844285`, coordinate `0b839df0…`, probe `2dabed0b…` reproduced by a
  ninth distinct build), a Layer-P pass certificate `d95859cb…` at that exact head
  with `selected_base b2f3c23f419cb03c…` and 37/37 bindings independently
  re-derived, an untracked freeze `F64 a03fc459…` verified 22/22 — including the
  controller's own terminal-1 predicate returning zero mismatch reasons — and then
  a separate owner single-invocation authorization, consumed at launch. The
  controller ran once and selected terminal 1
  `H2_INPUT_MUTATED_DURING_MEASUREMENT`: **0/4 ordered runs started, no `runs/`
  directory, zero faithful capture, `equivalence` untouched at `unproven`, no
  seal.** Exit code 2 came from the independent verifier refusing the archive, so
  there is no verifier report either. The adjudicated root cause is **not** the
  recorded label: no external input was mutated. The controller creates its own
  evidence root inside the working tree at a path that is not gitignored
  (`run_h2_measurement.py:1054-1057`, `h2_measurement_evidence.py:103`) and then
  requires that same tree to be clean (`:1140-1143`, `:1291-1293`), so its own
  artifact violates the invariant it enforces — at every head, independently of
  freeze, certificate, probe or host. A second defect folds that hygiene reason
  into `layer_p_certificate_matches_freeze` (`:1140-1145`), so the archive records
  a certificate mismatch that does not exist and contradicts the verifier's
  recomputation. Items 4–6 are therefore **satisfied-then-void**, the authorization
  is permanently spent, and the next step is a controller repair on a successor
  head followed by a completely new acceptance and authorization cycle. Failure
  evidence, including a disclosed chain-of-custody incident during its
  registration:
  [h2_phase_a_failed_attempt_0a5dffe9_20260727](../../modules/semantic/research/evidence/h2_phase_a_failed_attempt_0a5dffe9_20260727/).
