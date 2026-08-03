---
doc-status: proposed
doc-promotion: navigation-only; not evidence
doc-date: 2026-07-25
doc-module: semantic
owner-module: semantic
work-class: mainline-study
wip-role: non-wip
activation-gate: "四項 owner decision 已於 2026-07-25 accepted（#286）；Phase-B chain form 已於 2026-07-25 accepted（#290）並成文為 declaration Review Correction 3；C3.9 pre-seal ruler 編輯（七序列 manifest + phase_a_evidence schema + phase-aware partition）已 landed、republish，並於 `7c348f4e` 通過 controlled-host re-attestation（run 30164262580）；Layer-M review 又把語義常數搬回 ruler ⇒ identity_semantics 再動一次（`3edf6953`）已 republish，並於 `f2c2510a`（run 30243657973）重新通過 controlled-host re-attestation；PR #295 第二輪修補再次修改 packet-invalid exception boundary 與 terminal-2 surviving-evidence ruler，identity_semantics `93f87a83` 已 republish，並於 `32935d5d`（run 30254189532）重新通過 controlled-host re-attestation ⇒ gate 2 再度關閉；S4 Phase-A controller（items 0–4）已 implemented/tested，**review 已完成、PR #295 已 merge ⇒ S4 code-review gate closed**（reviewed implementation head `4c78b962…`，landed merge commit `b2f3c23f…`）；Acceptance items 4→5→6 已於 2026-07-27 在 head `0a5dffe9` 全部達成（controlled-host run 30276844285 綠、Layer-P certificate `d95859cb…` 37/37 獨立驗證、F64 `a03fc459…` 22/22、owner single-invocation authorization），並已執行一次 ⇒ **authorization spent、controller terminal `H2_INPUT_MUTATED_DURING_MEASUREMENT`、0/4 ordered runs started、no capture、seal 未完成**；adjudicated root cause = controller self-mutation（在 repo 內建 evidence root 後又要求 checkout 乾淨），非 head/binding/ruler 問題 ⇒ **items 4–5 satisfied-then-void（對 successor head 失效，須重建）；item 6 已 consumed 且永久 spent，不是重建而是由 owner 另行簽發新授權；下一步是 successor head 上的 controller repair，之後 acceptance 週期在該 head 全部重建、authorization 由 owner 另行簽發**；失敗證據見 evidence/h2_phase_a_failed_attempt_0a5dffe9_20260727/；**2026-07-28 在 successor head `7646f421` 上重建了 items 4–5（controlled-host run 30334080842 綠、Layer-P certificate `266f4b4c…` 65/65 獨立驗證、F64 `f0d1b02e…` 51/51），owner 另行簽發第二份授權並再度執行一次 ⇒ 第二份授權亦 spent、controller terminal `H2_MEASUREMENT_EXECUTION_INVALID`（order 4）、1/4 ordered runs started、zero faithful capture、seal 未完成。這次 archive 完整並通過 independent verifier（valid=true、complete），2026-07-27 登記的四項 controller 缺陷在該 head 全部關閉；新根因＝child 在 `_import_eval_stack()` 之後重新套用自己的 ingress environment contract，而 cv2 4.11.0 於 import 時改寫環境 ⇒ 任何裝 OpenCV 的 host 皆必然失敗，與 head/binding/ruler 無關；owner 已裁定修法（保留 ingress gate、撤掉 import 之後對它的重複套用、pre_import→post_import delta 僅作 diagnostic 不得再參與授權判定）。第二次失敗證據見 evidence/h2_phase_a_failed_attempt_7646f421_20260728/，controller archive 見 evidence/h2_measure_7646f421a85a580e37e457def5e8ddc7c4bfa0ab/**；**2026-07-29 在 `ba40b3f8`（PR #302 harness + PR #303 分類修復之後的 head）上重建了 items 4–5 與 F（controlled-host run 30454462387 綠、certificate 檔 sha256 `d11327752ac092f2…` 71/71 獨立驗證、`F64 70001e5449b8ad26…` 69/69），並**首次執行 rehearsal ⇒ FAILED**：terminal `H2_MEASUREMENT_EXECUTION_INVALID`、`runner_nonzero`、`00_capture_off` 的 child 失敗、01/02/03 未起、**未耗任何 owner authorization**。harness 的隔離／receipt／archive binding／checkout hygiene／corpus refusal 全部成立 ⇒ 這是 gate failure 而非 harness failure。根因＝H2 fixed-A5 invocation adapter：child 只傳 `--sequences`／`--output`，而 A5 preset 沒有 `double_buffer`／`detect_barrier` ⇒ `configure_runtime_env` 依其宣告的 args authority 把凍結環境改寫成 `full`／`0`；H0 早以 `EVALUATOR_ARGV_PREFIX` 解決，H2 未沿用。owner 已裁決 7 項 argv 差異（`detect_barrier`／`double_buffer`／`latency_only` 為缺陷須修，其餘四項接受 H2 現值），並要求 harness 的 run-completion predicate 改由 `invocation.json` 的 lifecycle state 導出且必須先於 child 修復落地。items 4–5 與 F 在 `ba40b3f8` 上 historically valid、對每個 descendant head stale、不追溯無效**"
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
| **S4** | ⛔ **implemented / reviewed / merged — executed twice, both defective** — the 2026-07-27 invocation reached terminal 1 with 0/4 ordered runs started (controller defect, since repaired); the 2026-07-28 invocation at `7646f421` reached terminal 4 with 1/4 started and still zero capture (child environment-validation ordering defect). Two authorizations are spent; an execution-and-archive-verifier repair is required before any further attempt — terminal partition, evidence-root contract, independent verifier, corpus checker, observation emitter, and the Phase-A controller with its four ordered runs. This implementation creates no `I`/`F`/`S` and performs no authorized measurement | no |

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

That repair landed, the whole chain was rebuilt on it, and the successor lease
was **met and spent in turn on 2026-07-28**: `7646f421` was attested, certified
and independently verified, authorized once more, and executed — and the
invocation reached terminal 4 with `00_capture_off` exiting non-zero and still no
capture. See `History` and the
[second failure evidence](../../modules/semantic/research/evidence/h2_phase_a_failed_attempt_7646f421_20260728/).

The first half of the successor lease is **met on 2026-07-29**: the
execution-and-archive-verifier repair landed on a further successor head,
unbound and unauthorized — the post-import re-application of the ingress
environment contract removed under the owner-adopted shape in `Current step`,
and archive verification freed from the verifying host. No `I`/`F`/`S`, no
authorization, no capture, no registry state change.

The rehearsal harness that run needed **landed on 2026-07-29**, with the corpus
admission guard that has to precede it and an issuer normalization that gives
the guard something to bind to. It authorizes nothing and it has not been run.

What the lease still expects, unmet: a real non-evidence run through
controller → child → eval-stack import → environment validation → capture
initialisation → first stop boundary, spending none of the owner's authority and
writing no evidence root. It has an entry point now; it has not been walked. No
acceptance, certificate, freeze or authorization carries over from `0a5dffe9` or
`7646f421`. That state is still not an authorization and still writes no
registry state.

An ordering trap the two spent attempts have already taught: the rehearsal
witness may **not** be committed before Phase A. Every commit moves the head,
and `F` and the Layer-P certificate bind `source_head`, so committing the
witness would staleness-kill the very chain the rehearsal was run to protect.
It stays in read-only custody outside the repository and is registered together
with the Phase-A outcome. If the owner ultimately issues no third grant, that
custody is closed out by a rehearsal-only registration or an abandonment rather
than left open.

## Commit point

Owner reviews whether state changed when **either** a repaired controller and
child together demonstrate a reachable success path from launch to the first
stop boundary at a successor head, as a real run under the real constructed
environment — a demonstration that is not an execution and spends nothing —
**or** the repair shows the two-layer design cannot hold the invariants it
declares, and the charter is re-planned. The spent `7646f421` attempt
(2026-07-28) was the previous commit point, as the spent `0a5dffe9` attempt
(2026-07-27) was the one before it: neither changed a claim-state object and
neither produced a measurement. Repair is not a seal gate reopening; a new
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
| `.github/workflows/runtime_identity.yml` | manual, non-qualifying controlled-host runtime diagnostic |
| `../../modules/semantic/research/evidence/h2_measure_7646f421a85a580e37e457def5e8ddc7c4bfa0ab/` | the 2026-07-28 controller archive — a spent attempt with a negative terminal and zero capture, committed at its canonical corpus position; CI enforces its inventory, and full re-verification waits on the host-binding defect of §4.2 |
| `../../modules/semantic/research/evidence/h2_phase_a_failed_attempt_7646f421_20260728/` | that attempt's adjudication, root-cause reproduction and custody record |
| `../../modules/semantic/research/evidence/h2_phase_a_failed_attempt_0a5dffe9_20260727/` | the 2026-07-27 attempt's failure evidence, including its own controller archive |

## Current step

### 2026-07-30 requirement narrowing — declared, not implemented

Declaration
[Review Correction 5](../../modules/semantic/research/headline_bridge_behavioral_identity_capture_declaration_20260725.md#review-correction-5--execution-integrity-is-the-requirement-2026-07-30-pre-seal)
supersedes the future acceptance target described below: this H2 unit now
requires **execution integrity**, not environment reproducibility. *Historical
account — the head-bound chain* below remains the record of how the existing
head-bound chain was built, failed and repaired; it is not authority to rebuild
that chain again.

The migration target is content-bound:

- one complete resolved RunSpec is the sole configuration authority; preset
  defaults, parser defaults, env and argv are derived transports;
- diagnostic and measurement share `resolved_run_spec_digest` and
  `execution_semantics_projection_digest`;
- the producer emits `run_spec.json`, `runtime_binding.json` and `result.json`;
  only a separate verifier process writes `verification.json`;
- the projection is digest equality over its declared content set and does not
  consult `h2_path_partition`;
- the verifier remains foreign-host independent and fails closed without
  invoking the producer.

The six Layer-P stages stay in order, but the successor path retires the
independent Layer-P certificate, `freeze.json` / `F`, published
coordinate/probe equality as a measurement gate, and commit/tree identity as a
validity gate. The bounded probe remains a recorded observation and
`equivalence` remains `unproven`. The two historical archives retain their
original schema and full verifier path.

Declaration Review Correction 8 applies that boundary to CI:
`.github/workflows/runtime_identity.yml` is now a `workflow_dispatch`-only,
non-qualifying controlled-host diagnostic. It no longer runs automatically for
pull requests or `main`, and its result cannot qualify or invalidate a successor
execution. The successor producer still performs the retained build, binding,
extension-load and identity-run stages inside the execution whose bytes it
records; independent archive verification remains foreign-host independent and
does not rebuild the execution.

At the declaration base head
`1a74276527b682587c7f23babd03eb6fb4bcf6b7`, Item 4 (workflow run
`30510440753`) and Item 5 (certificate file `6f338926…`, object digest
`8996ee13…`) were valid and held in read-only external custody. `F` was never
built at that head. The live `build/h2_layer_p` belongs to Item 5 and must not be
rebuilt or removed while that historical certificate is live.

This declaration does not itself implement the migration or authorize any
execution. In particular it does not authorize Phase A, a third grant, `S`,
seal, canonical-corpus admission or an equivalence update. Authoring the
full-namespace RunSpec requires a new explicit decision on `detector`,
`max_frames`, `preset` and `warmup_frames`; the prior adapter ruling remains
historically true but does not silently author the new sole authority.

### 2026-07-30 successor implementation order — planned, not implemented

Confirmed mechanically at `290fd0c10b3bff070b50fc1ebcd760c98ee36507` (PR #306
merge; clean worktree, `HEAD == origin/main`, no open pull request): the
migration's **contracts and configuration authority** have landed and nothing
executable consumes them yet. `tests/contract` is 1158 passed / 4 skipped /
3 xfailed, the corpus checker is `PASS (1 roots; complete=1)`, and the staleness
checker exits 0 with the three legacy unresolved-axis warnings — no host
environment, runtime-input content or probe was recomputed, and no equality was
claimed.

Two constraints are gone: with `source_head` / `source_tree` / `F` / the Layer-P
certificate / published coordinate-probe equality retired as successor gates,
`main` no longer has to stand still and Items 4–5 no longer have to be rebuilt
for every commit. This is what makes ordinary staged review possible again.

What blocks the successor path is a **verdict-algebra gap**: the successor
artifact vocabulary and the ruler's executable partition disagree, and no code
reads the successor vocabulary at all.

| gap | evidence at this head |
|:--|:--|
| three predicate names diverge, one with inverted polarity | `h2_execution_result_v1` requires `behavior_probe_equals_spec`, `runtime_binding_matches_spec` and `bound_input_unchanged`; `h2_terminal_partition.ORDERED_PREDICATES` carries `behavior_probe_equals_freeze`, `layer_p_certificate_matches_freeze` and `bound_input_mutated` (the last in `_TRUE_IS_FAILURE`) |
| the predicate state space widened from two values to four | the schema's `predicate_result.state` is `pass` / `fail` / `error` / `not_run`; `select_terminal` raises on any non-`bool` predicate |
| three ruler results have no successor token | `build_failed`, `extension_load_failed` and `certificate_mismatch` — while Correction 5 explicitly **retains** the `build` and `extension_load` stages, so their failures must still land somewhere by name |
| two successor results have no ruler mapping | `runtime_binding_mismatch` and `diagnostic_complete` |
| measurement verdicts are jointly unconstrained | `h2_execution_result_v1.allOf` constrains only the `non_qualifying_diagnostic` branch; under `exactly_once_measurement` a failure `result` with `terminal: null` validates — the successor shape of the "authorization spent, no terminal" state § C3.5.1 exists to make unformable |
| `valid` is unconstrained | `h2_execution_verification_v1` has no cross-constraint at all: `valid: true` with every `checks.*: false` and a non-empty `reasons` validates |
| nothing consumes the four artifacts | there is no `verify_h2_execution.py`; every `result.json` reference under `scripts/` belongs to H0's legacy schema |

The staged order this charter fixes, each stage a separate reviewable landing:

| | stage | needs a build / GPU |
|:--|:--|:--|
| **W1** | this governance landing — confirmed state, the gap table, the order. Docs only: no ruler edit, no re-pin, no republish | no |
| **W2** | ✅ **landed — verdict algebra.** Declaration Review Correction 9 with the joint constraints in the two already-ruler schemas, and the successor↔legacy mapping, four-state handling and `select_successor_result` in `h2_terminal_partition.py`, published through `as_payload()` (§ 20.8). `build_failed` / `extension_load_failed` were **re-admitted as named tokens** rather than folded, because Correction 5 retains the stages that produce them. `sealed_prefix` re-pinned to 94893 bytes; `identity_semantics` republished from the existing read-only records at each ruler edit; probe and the other four axes did not move. **Owner review found five blocking defects across four rounds, all fixed in place**: a named finding admitted any non-pass state, so `error`/`not_run` could select a finding by name; and the runtime binding made every stage failure — terminal 1 included — unformable, so it is now stage-aware through `failed_stage`; and the cross-artifact rules were unconditional, which deadlocked a build failure under a moved input and broke the diagnostic boundary, so they are now gated on `authority` and on the ruler's recomputed terminal, with a stage failure forbidden under the non-terminal progression and the subordinate-verdict set derived from Phase-A reachability, published as W3 verifier obligations. Reachability is explicit: a stage failure requires every ordered run unstarted and admits only the stage-independent findings, so terminals 2 and 3 cannot be claimed from an execution that started no run. Every defect sat in a cell no case had asked about, and three of the five came from enumerating over the wrong axis, so the rules are now closed by an exhaustive classification over named axes with a per-cell invariant | no |
| **W3** | ✅ **landed — archive-only verifier.** `scripts/tools/verify_h2_execution.py` writes `verification.json` and then closes the inventory over all four records, from archive bytes only. It **composes and holds nothing**: the shapes come from the four frozen schemas, the RunSpec's internal consistency from the resolver, the ordered verdict and the cross-artifact algebra from `h2_terminal_partition` — which is why it classifies `plumbing_only` and **needed no ruler edit, no re-pin and no republish**. `binding_agreement_reasons` receives the ruler's **recomputed** terminal, never the archive's, so a record cannot certify its own verdict. `verification_host_inputs_used: false` is enforced, not asserted: the one ruler function that hashes the local checkout is called only through `verify_projection=False`, and a test makes it raise while a faithful archive still verifies. Two rules together draw the failure classes — a defect inside a formable archive is a recorded `valid: false`, while an archive that cannot fill the record's own required fields, or whose root is not physically flat, writes nothing and exits non-zero. **Owner review found two more negative states, both fixed without touching the verdict algebra**: a formable but schema-invalid archive crashed instead of being recorded, because a list reached the selector and, more quietly, because a string satisfies `Sequence` and walked into the algebra one character at a time — so the container shapes are now checked at the plumbing boundary rather than by widening the ruler's tolerance; and the closure was not a state machine, so an archive half-written between the two commits verified as valid while `O_EXCL` refused to complete it — the two closing records must now appear together, and a stored verdict is compared against the recomputed one rather than merely counted, since re-deriving *a* verdict proves nothing about the one the archive carries. **A second round found two more cells on the same boundary**: that comparison covered only a closure-independent core, so `valid`, `reasons`, the closure check's own field and any additional property could all be rewritten while the archive still verified — every member excluded from the comparison is a member an editor may rewrite — so the stored record is now schema-validated and compared whole against an expectation built from the artifact checks and the physical closure alone; and two formable archives still escaped the verdict path, a non-string `result` token raising on an unhashable lookup key, and a `runtime_binding.json` that is not an object being misfiled as unformable even though the record's required fields all come from the other two artifacts and its own bytes. Fixtures are synthesised from the frozen schemas and the frozen authoring profile, never captured from producer output (§ 5.3) | no |
| **W4** | ✅ **landed — producer (code only).** `scripts/tools/run_h2_execution.py` assembles the three artifacts and closes nothing: no `verification.json`, no `checksums.sha256`, proven over the syntax tree. It decides no verdict — `result` and `terminal` come back from `select_successor_result` and are transcribed, and **whether it may even name terminal 4's cause is answered by asking the ruler for the unnamed selection first**, so precedence and the diagnostic boundary stay where they are owned. The owner adjudicated two questions before it was written: the four ordered runs reuse the Layer-M runner, so the seven `executed_surfaces` the schema names really are the code that ran and **no ruler edit, re-pin or republish was needed**; and W4 runs nothing, so item 5's `build/h2_layer_p` bytes are untouched. `Stages` and `Runs` are protocols, so the control flow this module owns is tested without a build, and the CLI refuses to execute until a driver is bound. The W3 verifier immediately caught a real producer bug — a stage failure that hid a recorded input mutation — which is exactly the value of having written the verifier first. **Owner review asked for one property to be pinned rather than left to statement order**: the two selections must read one frozen observation, so the runs' record is deep-copied and closed when it is handed over, and the invariant is asserted over every observation the producer can emit — the terminal never moves, and the result token moves only from `unclassified_execution_failure` to a cause mapping to that same terminal 4 | no |
| **W4b** | ✅ **ran — the first bound-driver execution, and it found something.** The producer's command line now constructs the two adapters against the real Layer P and the real Layer-M runner, and refuses three things it cannot honestly do: a measurement authority (its content is a spent grant, and none can be received here), a run tree inside the flat archive root, and a bundle-shaped stub with a freeze and a certificate left empty — `LaunchSite` answers only the two questions the launch path asks. **The owner adjudicated the fork the real binding exposed**: an execution that builds inside itself cannot watch its own outputs, because the monitor must start before the binding and they do not exist yet, so Layer P builds first and the diagnostic binds the result with `--skip-build`, keeping the build artifacts under the watch for the whole run phase as the legacy controller's were. The diagnostic came back `diagnostic_complete` with `terminal: null`, and the independent verifier records it `valid: true` on all seven checks — **including a truthful negative**: `00_capture_off` ran the complete 1050-frame sequence at 262.9 FPS and the child's own recorder rejected it, `raw binary32 emission rows differ from callback order`. The sequence callback receives `results_lines` after deferred alias remapping, the quality filter and `interpolate_tracklets` (777 frames added here), while the child's rows come from the per-frame hook that runs before all three — **the frozen child asserts an invariant the RunSpec's own configuration falsifies**, found before any authorization existed to spend, exactly as § 5.1.1 was. Runs 01–03 are `not_run` because the producer's fail-fast control flow left them so, never filled in by the driver. The rename-aside is evidence, not cleanup: 4355 files / 378 093 331 bytes inventoried, moved not deleted, and the inventory digest recomputed unchanged after both the new build and the run. One driver defect surfaced immediately — `bound_paths` read the path classes from the wrong module, and nothing had ever called it, because a helper only an entry point reaches is a helper only an entry point tests | yes |
| **W4b — the finding, adjudicated** | ✅ **the cross-boundary equality is retired, and the re-run is green.** The owner ruled it an unauthorised cross-member predicate rather than a production invariant: A7.6 already names `final_track_rows` and `mot_output` as two members, each compared capture-off ↔ capture-on on its own, and asks for no row projection between them — while the child was comparing a per-emission record against a callback that arrives after alias remapping, quality filtering and interpolation. Each surface keeps its own integrity check, now in two named functions instead of inside a closure only a GPU run can reach, which is precisely why the defect survived. None of the forbidden repairs were taken: no post-processing disabled, no binary32 reconstructed from decimal text, no fabricated bits, no id written back, and `final_track_rows` keeps its equality. Cost is exactly what the owner predicted — the RunSpec's declared content projection moves (`e81c3ee7…` → `478570c5…`) and nothing else: no ruler edit, no correction, no re-pin, no republish. The re-run completed all four ordered runs with every predicate passing **and still returned `diagnostic_complete` with `terminal: null`**, which is review point 3 demonstrated rather than asserted; the verifier records `valid: true` on all seven checks. The earlier failing diagnostic stands as truthful: it proved the old execution semantics could not complete under the frozen RunSpec | yes |
| **W5a** | ✅ **landed — the declared execution-code closure.** The declared content set was fourteen tooling files, so two records could carry one `execution_semantics_projection_digest` and have run different trackers — while Correction 5 requires that the bytes actually used are the bytes recorded. The closure is declared by **roots** (`include/`, `src/`, 198 files), not by names, and this unit already paid for the difference: `interpolate_tracklets` lives in `post_merge.py`, which is not one of the seven files the partition names as runtime paths or observation sites, and its behaviour decided the whole W4b finding. The selector admits present-but-uncommitted files, because those are files the interpreter can still import; it consults no path classification, since a path class is a verdict about what an edit *means* and this is a set of bytes. Two identifiers move because their domain did — `..._projection_v1` → `_v2` and a renamed algorithm — under Correction 7's rule that a changed meaning may not keep the old name. The closure needed its own member type: `src/saccade/__init__.py` is empty, ordinary Python and genuinely imported, and skipping empty files would have put the first exception into a rule that has none. **It does not ask whether the closure is complete**: that is unanswerable from these bytes and is checked in W5b against the observed import record, which this document did not produce. `identity_semantics` `dd354e13…` → `73d9a164…`, republished from existing read-only records; probe and the other four axes unchanged. The RunSpec projection moves `7b4db638…` → `57b0fe5a…` — measured on this branch's tree, since #314's child repair moved a named member of the same projection; the 198-member closure digest `f9e70894…` is unaffected, because #314 touched neither declared root | no |
| **W5b** | ✅ **landed (code) — the observed import witness, the wider roots, and namespace write-protection.** W5b was scoped as a witness and turned out to contain a declaration repair, because the witness's first act was to find one. Measuring the observed imports against the declared closure showed **seventeen loaded repository files bound by no content set**: `h2_runtime_inputs` (the canonicalization every digest in this unit rests on), `h2_behavioral_identity._import_eval_stack` (which chooses the evaluator stack), `resolved_bridge_policy_config`, `run_h0_phase_a`, `export_headline_bridge_decision_trace`, and all eleven of `scripts/eval/config`. `h2_path_partition` already ruled that they matter, so governance said they counted while nothing held their bytes — the `post_merge.py` finding one layer up, in the half W5a left alone. So the roots become **`include/`, `scripts/`, `src/`** (659 members): a root tuple declares which repository source *may* take part, and the witness proves one execution did not step outside it; neither may be derived from the other, which is why it is the whole of `scripts/` and not the two subdirectories the observed call path touched. The 459 extra members buy conservative failure — unrelated edits move the digest and force a republish — which is the sound direction. **Three boundaries the owner set, all load-bearing:** the recorder classifies everything repository-local rather than filtering `.venv`/`build/` at its entrance (a filter deletes evidence before anything judges it); it is installed by an independent bootstrap process, since a hook in the child's `main()` has already missed that module's top-level imports — exactly the ones deciding what runs; and the resolved namespace is frozen alias-free and recursively, rebuilt rather than wrapped, with the working copy re-derived, so mutate-then-restore has nowhere to happen. Domains are a **set, not a kind** — five named members now live under a root and carry both bindings — and admission asks only that the set be non-empty and every binding agree on the bytes. The measurement verifier re-derives containment from archive bytes (§ 20.8) and checks misclassification in both directions; the child writes the witness and does not judge it. Verified end to end: 1414 observations, **zero unbound**. `identity_semantics` `73d9a164…` → `32bee186…` (18 → 19 files); the other four axes and probe `2dabed0b…` unchanged, confirming the witness, bootstrap and verifier are all `plumbing_only`. Projection `57b0fe5a…` → `44ed14bf…`, closure `e5b1ef09…`, RunSpec `66c031fe…`. **GPU execution deliberately not yet run** — it must be on W5b's final tree | yes |
| **W5c** | ✅ **landed — successor corpus admission and rehearsal witness v3.** The canonical corpus discovers successor archives from the family-specific first artifact `run_spec.json` rather than a directory-name prefix, independently recomputes the successor verifier verdict, and admits only a fully closed (`verification.json` + `checksums.sha256`), valid archive whose recorded authority is `exactly_once_measurement`. PR CI caught the first discovery rule treating shared `result.json` / `verification.json` as family markers and therefore misclassifying an H0 archive; the regression now proves shared names do not select the successor path while producer-only successor roots stay visible. A producer-only directory is not evidence, and a green `non_qualifying_diagnostic` remains refused: archive validity and corpus admission are separate decisions. The authority spellings are now ruler-owned constants, not new tokens or a new grant. The rehearsal harness asks the real corpus owner and records its non-empty refusal in `h2_phase_a_rehearsal_witness_v3`; historical v1/v2 witnesses remain untouched. This is a structural provenance guard, **not** cryptographic authority proof; the repository has no signature mechanism. Contract evidence: successor measurement admission, diagnostic refusal, unclosed/invalid archive refusal, arbitrary-root discovery, cross-family isolation and corpus-owner rehearsal failure all pass. Final W5c closure digest `fab25fba…`, projection `34ad9734…`, RunSpec `5e648186…` (659 members) | no |
| **W5d** | ✅ **landed — declaration and governance closeout.** Review Correction 11 describes the code after it landed: W5a's root closure, W5b's independent observed-import containment and immutable namespace, and W5c's validity-versus-admission boundary. Its pre-merge revision records CI's cross-family discovery finding. The full pre-seal declaration is re-pinned at 115473 bytes / `717dde08…`; runtime identity is republished from the existing read-only W4b probe and runtime-input records, with no controlled-host probe or native rebuild. `identity_semantics` moves from W5b's `32bee186…` to final `89d4bb41…`; decision surface, environment, implementation, runtime inputs and probe `2dabed0b…` remain unchanged, and equivalence remains unproven. Governance projections now point only to the final non-qualifying diagnostic W5e; no authorization, measurement, `I`/`F`/`S`, seal, corpus admission or equivalence follows | no |
| **W5e** | ✅ **complete — the single final-projection diagnostic.** PR #317 merged W5c–W5d as `e94407f2…`; clean local `main` matched that merge and the retry base named the same head with zero changed paths. Exactly one `non_qualifying_diagnostic` then ran: all four ordered MOT17-04-SDP runs completed, all five predicates passed, the bound-input monitor observed zero changes with a clean final drain, and the retained identity stage recorded probe `2dabed0b…`. The producer still returned `diagnostic_complete`, `terminal: null`. The archive-only verifier independently records `valid: true`, all seven checks passing and `verification_host_inputs_used: false`; the canonical corpus owner independently recomputed it and refused it for exactly one reason — diagnostic authority is not exactly-once measurement authority. This is the intended W5c separation between internal validity and admission, not a near-measurement. External self-sealed custody: `/home/ray/h2_w5e_diagnostic_20260803T132933Z` (512 inventory members, 82 MB, inventory digest `c25baf48…`). The only post-run edits are these non-execution governance projections, mechanically classified as such; no final projection byte moved and no second diagnostic exists. W5 is closed with no automatic next: no authorization issued or consumed, no measurement/qualification, no `I`/`F`/`S`, no seal, no corpus admission and no equivalence claim | yes |

### W5 owner-review acceptance — 2026-08-03

At reviewed `main` `136b9eea337f6f7188902bea0dc17ddb632fb854`, the owner verdict is
**`ACCEPT — W5 implementation/evidence closure`**. The accepted scope is W5c's
implementation, W5d's governance and runtime-identity publication, W5e's exact
final-projection diagnostic evidence, and PR #318's non-execution registration.
The mechanical rederivation was performed by a reviewing agent under owner direction;
the owner reviewed and adopted it as the basis for the verdict. The rederivation covered
the load-bearing identities and relations rather than accepting their prose projections:
discovery and cross-family isolation;
closed/valid/measurement-authority corpus admission; witness-v3 delegation to the
corpus owner; policy pin and identity-axis movement; the 659-member closure,
projection and RunSpec; packet and archive closure; four run artifacts, output
equality and zero-unbound import witnesses; independent 7/7 verification; and the
single expected diagnostic-authority refusal.

The accepted external packet binding is the full packet-inventory digest:

    c25baf48b8425543a1f430b83b0381bcc2dc72386398d54acf22e41103bea268

Two observations are explicitly non-blocking. First,
`corpus_admission_witness` catches only `CorpusError`, whereas the canonical corpus
CLI also converts `EvidenceError`, `PartitionError`, `VerificationError` and
`OSError` into refusal exits. The narrower harness path remains fail-closed but may
abort without writing v3's `admitted: false` answer. That is a separate plumbing
follow-up, not repaired here: the file is inside the declared `scripts/` closure, so
any future edit moves the projection and needs a separate disposition; this
acceptance authorizes neither that edit nor a re-run. Second, the earlier 8-hex packet
digest was navigation only for external custody; this registration records the full
digest without changing packet bytes.

This acceptance closes only W5 implementation/evidence review. It is not measurement
acceptance or qualification, issues no authorization, creates no `I`/`F`/`S`, admits
nothing to the canonical measurement corpus, seals nothing, reopens no H0 path and
leaves equivalence `unproven`. There is no automatic next.

W3 precedes W4 for three reasons, two of them this unit's own scars. PR #294
wrote the evidence contract and the independent verifier **before** the
controller of PR #295, because the reverse order defines the contract from
whatever the producer happens to emit — § 5.3's everyday form. PRs #302 and #304
were both reordered so the guard landed before the entry point, because a
non-squash merge leaves every intermediate commit checkout-able: a head that can
emit self-consistent four-JSON archives with no verifier and no admission gate is
exactly that false-green combination. And W2–W3 need no build, so item 5's live
`build/h2_layer_p` bytes stay untouched and the rename-aside decision belongs to
W4, where a build is actually required.

This staging authorizes no execution. It selects no `I`, creates no `F`/`S`,
requests no third grant, seals nothing, admits nothing to the canonical corpus
and changes no equivalence claim. Working detail (not authority):
`~/.claude/plans/delightful-launching-gosling.md`.

### Historical account — the head-bound chain

**The rehearsal ran once, at `ba40b3f8`, and failed — and it cost no
authorization.** The controller reached terminal `H2_MEASUREMENT_EXECUTION_INVALID`
in the first ordered run. This is the third structural self-negation in this work
item and the first one found without spending an owner grant, which is precisely
what the harness exists for.

The cause is the H2 fixed-A5 invocation adapter, not the generic evaluator, not
the frozen A5 constants and not the mutation gate. The frozen environment
declares `SACCADE_DETECT_BARRIER=event` and `SACCADE_DOUBLE_BUFFER=1`; the child
must configure the evaluator from the one A5 preset, which carries neither knob;
so `configure_runtime_env` — whose declared authority is the parsed arguments —
rewrites them to `full` and `0`, and the gate refuses the run. H0 solved this
long ago by passing the fixed choices through the parser's authoritative surface
(`run_h0_phase_a.EVALUATOR_ARGV_PREFIX`). The H2 adapter never carried that over.

`Acceptance` items 4 and 5 and `F` were rebuilt at `ba40b3f8` and independently
verified (71/71 and 69/69). They are **historically valid at that head, stale for
every descendant head beginning with the registration commit below, and not
retroactively invalidated at `ba40b3f8`**. Any repair commit moves the head, and
item 4 must be green at the exact final head, the certificate binds `source_head`,
and `F` binds the certificate — identical trees do not transfer.

**The repair has landed. Rebuild the chain at this head, then rehearse once
more. Do not seal, do not ask for a third grant, and do not treat any
`ba40b3f8` binding as carried over.** The child now sends the frozen A5 choices
and the no-metrics boundary through the parser that is authoritative over them,
and the harness derives run completion from the child's lifecycle record rather
than from a directory existing. Nothing past the configuration gate has ever
executed, so the next rehearsal is a discovery run, not a formality. Two
authorized Phase-A invocations remain spent and neither produced any capture:

| | 2026-07-27 at `0a5dffe9` | 2026-07-28 at `7646f421` |
|:--|:--|:--|
| terminal | 1 `H2_INPUT_MUTATED_DURING_MEASUREMENT` | 4 `H2_MEASUREMENT_EXECUTION_INVALID` |
| ordered runs started | 0/4 | 1/4 |
| faithful capture | none | none |
| archive | refused by the independent verifier | **accepted** — `valid: true`, `complete` |
| defect | the controller's own launch sequence | the child's environment validation |
| label describes the cause | no | yes |

The second attempt is a real advance and must not be flattened into "failed
again": every defect registered against the first was closed, the controller
reached `child_launch` for the first time, checkout hygiene and predicate
ownership were clean at launch and at the stop boundary, and the evidence root
finalized at its canonical corpus position, where the independent verifier and
the corpus checker both accept it — on the execution host. Registering it
surfaced a third defect of the same shape: archive verification recomputes the
authorization execution domain from the *verifying* host, so a committed archive
verifies nowhere else. CI therefore enforces the host-independent inventory
contract, and full re-verification is part of the repair, not of this
registration.

What remains is code-bound again, one layer deeper. `cv2` 4.11.0 rewrites the
process environment when imported — two added keys and a mutated
`LD_LIBRARY_PATH` — and `run_h2_measurement_child.py:298` re-applies the child's
*ingress* environment contract after `_import_eval_stack()` at `:271`. The
ingress gate itself (`:683`) is correctly placed before any import and passed;
the controller's construction of that environment (`run_h2_measurement.py:602-629`)
is sound and measured correct. See
[the failure evidence](../../modules/semantic/research/evidence/h2_phase_a_failed_attempt_7646f421_20260728/)
for the reproduction, the defect sites, the latent H0 twin, and the review rule
this attempt establishes.

The owner-adopted repair shape, stated once so the repair PR does not re-derive
it:

> The authorized environment is the immutable launch snapshot captured before
> any third-party import; every evaluation of the ingress predicate takes that
> snapshot as its input; and the environment delta produced by importing `cv2`
> may be observed but may not retroactively negate an ingress authorization that
> has already passed.

That is declaration [Review Correction 4](../../modules/semantic/research/headline_bridge_behavioral_identity_capture_declaration_20260725.md#review-correction-4--which-environment-state-carries-authorization-authority-2026-07-28-pre-seal),
and it constrains the semantics only: the number of checks and their placement
are the repair's to choose.

**What the repair did.** The ingress gate stayed where it was; the launch
snapshot is consumed exactly once, in `execute_child`, and nothing downstream
compares against it or rebuilds the predicate from live state. The
`pre_import → post_import` delta is recorded as key names only — diagnostic,
outside every gate and the invocation digest — and the environment is not
restored after the import. The property that `configure_runtime_env` mutates
nothing outside its own declared keys survives as a separately named contract
with its own baseline, taken *after* the import: both traps registered here were
real, and taking the ingress snapshot as that baseline would have charged cv2's
injection to the repository and reproduced the failure under a new name. A
key-set-only fix was never attempted for the reason registered — the mutated
`LD_LIBRARY_PATH` fails the digest branch of the same predicate. The frozen H0
ruler is untouched: `EXPECTED_ENV_KEYS`, `STATIC_ENV` and
`run_h0_phase_a_child.py` are unchanged, and H0's twin at its `:372` stays
latent and out of scope.

**A third site belonged to the same repair, and landed with it.** Registering the
archive found that `verify_h2_measurement._authorization` recomputed the
authorization execution domain from the *verifying* host's `/etc/machine-id` and
`os.getuid()` and required equality with the archived record, so a committed
Phase-A archive verified only on the machine that produced it. Host-binding the
grant is correct at launch and stays — `run_h2_measurement` still derives the
domain live when an authorization is admitted and consumed, and still
fail-closes when the controlled host has no machine identity. Archive
verification now judges the archived record from the archive's own bytes: member
set, `host_identity` shape, `operator_uid` shape, a canonical absolute POSIX
`ledger_root` decided as a string rather than through `pathlib`, and the digest
binding that was always the honest assertion — the receipt's and grant's
`execution_domain` must equal the digest of the archived domain object. With the
coupling gone, `check_h2_measure_archives.py` is wired into CI over full git
history, which the verifier legitimately needs to rebuild each archived
attempt's content axes.

Before the next authorization is requested there is one further gate the two
spent attempts have earned: a **non-evidence full run** — controller → child
process → eval-stack import → environment validation → capture initialisation →
first valid stop boundary — consuming no *owner* authorization and writing no
evidence root. Source review, unit tests with synthetic environments, and a green
launch probe running under the *operator's* environment have now each failed to
predict an execution-time structural self-negation.

**That gate now has an entry point, and has not been walked.**
`run_h2_measurement.py` still has a single path: `--authorization` is required
and the ledger is the default one. It did not grow a rehearsal mode — branching
before admission would not exercise admission, and branching after it while
skipping consumption would change the production authorization invariant. So
walking admission at all logically requires *an* authorization, and "consumes no
authorization" means the owner's third grant is untouched, not that no
authorization artifact exists.

`scripts/tools/h2_rehearse_measurement.py` (2026-07-29) issues its own grant
against its own disposable ledger and runs the unmodified controller through the
real admission and consumption path. Its grant is owner-shaped and is not an
owner issuance: admission requires the authoritative issuer, so the record names
the research owner in bytes, and what separates it is the execution domain,
which binds the ledger root. That makes a rehearsal grant arithmetically
unusable against the owner ledger. Its outputs live outside the repository, its
archive is refused by the corpus admission guard that landed with it, and it can
never stand in for `Acceptance` items 4–5, `F` or `S`.

Two boundaries are stated rather than implied. The harness refuses launch-time
lexical, symlink and ancestor aliases of the repository and the owner ledger and
*detects* a destination substituted mid-run; it does not resist a hostile
concurrent process running as the same user, because the controller writes
through pathnames. And the corpus guard is a provenance guard, not an authority
proof: it refuses an attempt consumed under another ledger, including a
self-consistent one, but whoever can rewrite a digest chain can write the
anchor's content too.

**The harness landing is not the gate passing.** The rehearsal has not been run.

The head-bound gates below (`Acceptance` items 4 and 5) were satisfied at
`7646f421` and died when the repair landed: `F64 f0d1b02e…` and Layer-P
certificate `266f4b4c…` are stale from that commit, are not partially reused,
and must be rebuilt at whatever head the repair lands on. Item 6 is not rebuilt
either time: both authorizations were consumed at launch and stay permanently
spent, so a successor cycle needs a third, separately issued by the owner.

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

> **Spent twice — at `0a5dffe9` on 2026-07-27 and at `7646f421` on 2026-07-28.**
> Steps 1–5 below were performed and independently verified at each head in turn,
> and step 6 executed once at each. The ordering itself is sound and has now been
> walked end to end twice; both attempts ran into a defect *downstream* of it —
> first inside the controller, then inside the child
> (`History`, and §4 of the
> [second failure evidence](../../modules/semantic/research/evidence/h2_phase_a_failed_attempt_7646f421_20260728/)).
> Nothing in this ordering carries over: after the repair, every step below must
> be redone at the successor head, against a controller **and child** that have
> first shown a reachable success path through a real launch → stop-boundary run
> that consumes no authorization.

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

**Migration boundary (2026-07-30).** The numbered matrix below is retained as
the acceptance history of the environment-reproducibility design. Review
Correction 5 retires items 4–5, their certificate/freeze coupling and
head-equality gates for the successor execution-integrity path. It does not
retroactively invalidate any record produced under that matrix. A replacement
acceptance matrix must be reviewed against the new artifact contract before any
measurement; until then no seal may be proposed and no execution is authorized.

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
   workflow is green at the exact final seal-candidate head. **Satisfied twice and
   void twice** — at `0a5dffe9` (run `30276844285`) and, after the controller
   repair, at `7646f421` (run `30334080842`): each certifies its own head, and the
   repair lands on a third;
5. a Layer-P pass certificate (`h2_layer_p_certificate_v2`) exists for that **same**
   head, with `--base` given and the full changed-path verdict clean — and its
   bindings independently verified. Neither `4c78b962…` nor `b2f3c23f…` can supply
   this for a later head: `source_head` is part of the certificate and is required
   to equal the executing head, so identical content does not transfer (see
   `Current step`). **Satisfied twice and void twice** — at `0a5dffe9`
   (certificate `d95859cb…`, `selected_base b2f3c23f419cb03c…`, 37/37 bindings
   independently verified) and at `7646f421` (certificate `266f4b4c…`,
   `selected_base 7646f421a85a580e…` with `changed_count 0`, 65/65 bindings
   independently verified) — **void for the same reason as item 4**;
6. the owner issues a separate exactly-once authorization — this charter is not
   one and cannot become one. **Issued and consumed twice, on 2026-07-27 and on
   2026-07-28.** Both are spent: each authorized one invocation, at `0a5dffe9`
   and at `7646f421` respectively, both invocations happened, and no part of
   either survives to the repaired head.

These are numbered in the order they must be performed: 4 → 5 → 6 is the same
sequence as `Current step` steps 2 → 3–4 → 5, and no item may be satisfied at an
earlier head than its predecessor. The Phase-A execution itself is not on this
list — `Acceptance` governs only when a seal may be *proposed*; execution follows
the authorization of item 6.

**Items 4–5 were last satisfied at `7646f421` and are void for any successor
head. Both issued authorizations are consumed and remain permanently spent.**
The two kinds of exhaustion are not interchangeable: a repaired child moves
execution-relevant code, so items 4–5 lose their binding and must be rebuilt
from scratch at the new head, whereas item 6 is not rebuilt at all — each
invocation it authorized happened, each counts permanently, and a successor
cycle requires a third, separately issued authorization from the owner. The
rebuild is not cheap and has now been paid for twice: coordinate republication,
a controlled-host run, a Layer-P certificate with independent verification, and
a new `F`. A repair PR may not present itself as continuing either attempt.
Item 3 is unaffected — it closed review of code that the repair will now modify,
and the repair carries its own review.

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

- **2026-07-30 — requirement narrowed, implementation not started.** Declaration
  Review Correction 5 records that H2 requires execution integrity rather than
  environment reproducibility. It declares the sole-authority resolved RunSpec,
  content-keyed execution-semantics projection, three-artifact producer and
  independent fourth-artifact verifier; retains Layer-P's six stages; and
  retires the independent certificate, `F`, published probe equality and
  source-head/tree validity gates for the successor path. This governance
  change builds and executes nothing, restores no authorization, leaves
  `equivalence` unproven and preserves every historical record.

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
  recomputation. Items 4–5 are therefore **satisfied-then-void** for any successor
  head, item 6 was **consumed and remains permanently spent** — it is not rebuilt
  but replaced by a new, separately issued authorization — and the next step is a
  controller repair on a successor head followed by a completely new acceptance
  and authorization cycle. Failure
  evidence, including a disclosed chain-of-custody incident during its
  registration:
  [h2_phase_a_failed_attempt_0a5dffe9_20260727](../../modules/semantic/research/evidence/h2_phase_a_failed_attempt_0a5dffe9_20260727/).
- **2026-07-28** — the controller repair landed
  ([PR #299](https://github.com/raylei50653/saccade/pull/299), `7646f421`) and the
  acceptance cycle was rebuilt on it: controlled-host re-attestation green (run
  `30334080842`), a Layer-P pass certificate `266f4b4c…` at that exact head with
  `selected_base 7646f421a85a580e…`, `changed_count 0` and 65/65 bindings
  independently re-derived, and an untracked freeze `F64 f0d1b02e…` verified
  51/51. `identity_semantics` moved `93f87a83` → `08d2db6a` when the repair
  touched a ruler member, was republished, and gate 2 was re-closed by that same
  controlled-host run.
- **2026-07-28** — **a second authorized Phase-A invocation was spent and again
  produced no capture.** The owner issued a second single-invocation
  authorization, consumed at launch. The controller selected terminal 4
  `H2_MEASUREMENT_EXECUTION_INVALID`: `00_capture_off` launched and exited
  non-zero, `01/02/03_capture_on` were never reached, **zero faithful capture,
  `equivalence` untouched at `unproven`, no seal.**

  What changed for the better, and must not be lost inside a second failure:
  every defect registered against `0a5dffe9` is closed at this head. The
  controller reached `child_launch` for the first time; checkout hygiene passed at
  launch and at the stop boundary; `certificate_mismatch_reasons` is empty and
  `bound_input_mutated` false with no events; the stop boundary linearized as
  `clean_final_drain`; the evidence root finalized at its canonical corpus
  position with a complete 28-entry inventory; and the independent verifier
  **accepts** it (`valid: true`, `verify_class: complete`), as does the corpus
  checker. Unlike the first attempt, the recorded terminal is also a correct
  semantic description of the cause.

  The adjudicated root cause is one layer deeper. `cv2` 4.11.0 rewrites the
  process environment on import — it adds `QT_QPA_FONTDIR` and
  `QT_QPA_PLATFORM_PLUGIN_PATH` and prepends its own lib directory to
  `LD_LIBRARY_PATH` — and it is reached transitively by `_import_eval_stack()`.
  The child validates its environment twice with the *same* ingress predicate:
  at `run_h2_measurement_child.py:683`, before any import, where it passes, and
  again at `:298`, after the import at `:271`, where it cannot. Any host with
  OpenCV installed fails identically, independently of GPU, build, dataset and
  model state — structural self-negation for the second consecutive attempt. The
  controller's construction of that environment (`run_h2_measurement.py:602-629`)
  and the child's ingress gate are both sound and were measured correct, so the
  launch authorization decision itself was not at fault. Removing the two added
  keys would not be sufficient: the mutated `LD_LIBRARY_PATH` fails the digest
  branch of the same predicate. The frozen `run_h0_phase_a_child.py:372` carries
  the identical post-import re-application, latent and not to be edited.

  Pre-authorization review missed it because the 2026-07-27 rule was applied
  within the controller but not across the process boundary: the child was
  reviewed as source and exercised only through unit tests with synthetic
  environments, while the launch probe — which does import the same eval stack
  successfully — runs under the operator's inherited environment
  (`run_h2_measurement.py:654`), never the sanitized one, so its green result
  carried no information about the child's own contract.

  Items 4–5 are therefore satisfied-then-void a second time, both authorizations
  remain permanently spent, and the next step is the execution-and-archive-verifier
  repair on a successor
  head under the owner-adopted shape recorded in `Current step`, gated behind a
  non-evidence full run that consumes no authorization. Failure evidence:
  [h2_phase_a_failed_attempt_7646f421_20260728](../../modules/semantic/research/evidence/h2_phase_a_failed_attempt_7646f421_20260728/);
  controller archive, committed with its inventory enforced in CI:
  [h2_measure_7646f421…](../../modules/semantic/research/evidence/h2_measure_7646f421a85a580e37e457def5e8ddc7c4bfa0ab/).

- **2026-07-29** — the **execution-and-archive-verifier repair** landed as two
  independent commits, closing both registered executed-surface defects without
  moving any ruler, spending nothing, and binding no head.

  `cc02a0b0` — the child decides ingress authorization once, in `execute_child`,
  against the launch snapshot captured before any third-party import, and
  nothing downstream compares against that snapshot or rebuilds the predicate
  from live state. The import's environment side effect is recorded as key names
  only (`environment_import_delta.json`, `authority: diagnostic_only`): outside
  every gate, outside the invocation digest, and carrying no fingerprint of any
  environment value, because a non-authoritative document has no reason to hold
  one. `configure_runtime_env` keeps a mutation gate, but as separate subject
  matter with its own baseline taken *after* the import — the registered trap,
  had the ingress snapshot been used, was that cv2's injection would have been
  charged to the repository and reproduced the same failure. The declared
  repository-owned key set is bound to the producer's actual behaviour by
  equality, not subset: a subset assertion would have left the allowlist itself
  free to widen. `EXPECTED_ENV_KEYS`, `STATIC_ENV` and `run_h0_phase_a_child.py`
  are untouched, and H0's twin at `:372` stays latent.

  `7cae46d8` — archive verification judges the archived authorization execution
  domain from the archive's own bytes and no longer recomputes it from the
  verifying host's `/etc/machine-id` and `os.getuid()`. Shape predicates
  (`host_identity`, `operator_uid`, a canonical absolute POSIX `ledger_root`
  decided as a string, never through `pathlib`, whose round-trip would not reject
  `..` and whose answer would otherwise depend on the verifying host's OS) sit
  behind the unchanged digest binding. Launch-time host binding is not relaxed:
  the controller still derives the domain live at admission and consumption and
  still fail-closes without a machine identity. With the coupling gone the corpus
  checker is wired back into CI over full git history.

  Both defects are held closed by tests that fail unmodified at `c2d1c58f`: an
  AST reproducer for the child, which states the defect as source structure, and
  a live-derivation sentinel for the verifier, which fails there at
  `verify_h2_measurement.py:235`. The verifier's negatives come in two classes —
  breaking the digest chain, and recomputing it so an illegal record is
  internally consistent and must die on the shape predicate it violates — because
  only the second kind can prove a shape predicate exists at all.

  This repair authorizes nothing, seals nothing, and restores no authorization.
  `F64 f0d1b02e…` and certificate `266f4b4c…` are stale from `cc02a0b0`, and the
  rebuild of `Acceptance` items 4 and 5 belongs at the head this work merges to,
  never at either repair commit. The non-evidence full run still stands between
  the rebuild and any request for a third authorization, and it now has a named
  predecessor of its own: the rehearsal harness described in `Current step`.

- **2026-07-29** — the rehearsal harness landed, in four commits ordered
  `P → B → A → C`. The order is the point: after a non-squash merge the commit
  that introduces the entry point is a head someone can check out and run, so
  the guard cannot be a later commit in the same pull request.

  `P` — `issued_by == "research_owner"` was written out twice in production, in
  the controller's admission predicate and in the archive verifier, so who may
  authorize a measurement had two independent answers and § C3.9's trap applied
  to both. `AUTHORIZATION_ISSUER` now lives in `h2_measurement_evidence` with the
  grant schema, the member sets and the canonical digest. The value is unchanged;
  the tests move the authority and require both validators' verdicts to move with
  it, in both directions.

  `B` — a rehearsal produces an archive of exactly the canonical shape whose
  every internal binding holds, and after `7cae46d8` the archive verifier
  correctly judges a root from its own bytes, so it will call that archive valid
  anywhere. `archive_roots` globs by prefix. The corpus is therefore the only
  layer that can refuse it: `execution_domain_admission_reasons` compares the
  attempt's archived execution domain against a tracked anchor, two archived
  objects and never the running host, so the verdict is identical everywhere.
  The anchor is the domain both spent attempts were actually consumed under and
  a test holds it to that. Phase A only — § C3.5.1 step 5 makes the receipt the
  whole of a Phase-B consumption and that shape is not specified yet. The
  layering is now written where both readers can see it: `verify_evidence_root`
  decides internal validity and has no canonical-admission meaning alone,
  `check_corpus` owns admission, and `test_research_packet_schema`, which had
  been reading `valid is True` as acceptance, asserts the conjunction. It is a
  provenance guard, not an authority proof: it cannot refuse a forgery, because
  whoever can rewrite a digest chain can write the anchor's content too.

  `A` — `scripts/tools/h2_rehearse_measurement.py` adds no production file and
  uses only the seams that already existed. It has no `--authorization` and no
  `--invocation-id`, so there is no argument through which the owner's grant
  could be spent; it issues its own, reading every contract value from its
  authority at call time. Isolation is filesystem identity, not string
  comparison — destinations are resolved through existing symlinks, compared by
  path components rather than prefixes, required not to exist, and created
  exclusively at `0700` — and the threat model is bounded in the file: launch-
  time aliases are refused, a mid-run substitution is detected and not
  prevented, because the controller writes through pathnames. Success is a
  conjunction, not a `None` terminal: the archive must verify, the disposable
  ledger must hold exactly the one receipt that binds the synthetic grant, every
  ordered run must have completed — projected from the archive, never from a
  counter the harness kept — and the checkout must be clean afterwards. A
  harness invariant violation is recorded separately from a rehearsal that ran
  and reached a terminal. The witness is exclusive-created as `started` and
  atomically replaced, so a crash cannot leave an archive with no rehearsal
  marker, and a refusal that happens before a safe destination exists writes no
  witness at all.

  Nothing here authorizes, seals or restores anything, and the harness has not
  been run. The next work item is `h2_phase_a_rehearsal_execution`: rebuild
  `Acceptance` items 4 and 5 at the head this merges to, build `F`, rehearse
  once with both outputs outside the repository, and only then ask the owner for
  a third grant. The witness stays in custody outside the repository until Phase
  A is registered — committing it would move the head and staleness-kill the
  chain the rehearsal exists to protect.

- **2026-07-29 — the rehearsal ran once at `ba40b3f8` and failed, spending no
  authorization.** `Acceptance` item 4 (controlled-host run `30454462387`), item 5
  (certificate file `d1132775…ad908c99`, 71/71 bindings independently verified)
  and `F` (`F64 70001e54…d27b6b66`, 69/69) were rebuilt in order at that head, each
  with read-only custody outside the repository. The rehearsal then reached
  controller terminal `H2_MEASUREMENT_EXECUTION_INVALID`, `runner_nonzero`:
  `00_capture_off`'s child failed and `01/02/03_capture_on` never started.

  Every isolation guarantee the harness offered held. The owner's ledger is
  unchanged byte for byte; the synthetic grant binds the disposable execution
  domain and not the owner's; the disposable ledger holds exactly the one matching
  receipt; the witness's `evidence_root_digest`, `verifier_report_digest` and
  `receipt_digest` all recompute outside the harness; the checkout was clean
  before and after. Run on the real archive, the corpus guard did both halves of
  its job: `verify_evidence_root` accepted it and `check_corpus` refused it for the
  controlled-host execution domain. **This is a gate failure, not a harness
  failure.**

  The defect is the H2 fixed-A5 invocation adapter. `run_h2_measurement_child.py`
  passes only `--sequences` and `--output` and takes everything else from the A5
  preset, which declares neither `double_buffer` nor `detect_barrier`; the parser
  therefore resolves them to `False`/`None`, and `mot17_args.configure_runtime_env`
  — documented as the authority over these knobs — forces `SACCADE_DOUBLE_BUFFER=0`
  and `SACCADE_DETECT_BARRIER=full`, contradicting the frozen A5 environment. The
  reproduction is a pure-function chain with no GPU, dataset, authorization or
  child process, and is host-independent. Comparing the child's resolved arguments
  with H0's `EVALUATOR_ARGV_PREFIX` shows seven differences out of 454 attributes;
  the owner adjudicated all seven, leaving no open question: `detect_barrier`,
  `double_buffer` and `latency_only` are defects this repair fixes, and
  `warmup_frames`, `max_frames`, `detector` and `preset` are accepted as they are,
  each because a stronger H2 contract already fixes what H0 fixed through that
  flag. `latency_only` joins the repair rather than waiting, because the evaluator
  returns `{}` before metrics only under it (`evaluator.py:3336`) while the child
  requires exactly that empty result (`run_h2_measurement_child.py:541`) — a
  contradiction already decidable statically, and not worth another rebuild of the
  chain to discover.

  Registering the failure surfaced one harness defect: `ordered_run_summary`
  derives `present` from the run directory existing, so the failed
  `00_capture_off` was projected as complete. The verdict was still correct here
  because the terminal conjunct fired independently, but the predicate is weaker
  than the message it prints. The authoritative record already exists and the
  controller already reads it — `invocation.json`'s `state`. Tightening the harness
  lands **before** the child repair: after a non-squash merge each commit is a
  runnable head, and a head that both reaches further and judges more loosely is
  the one that could report a false green.

  The rehearsal is a sequential-discovery instrument, and this was the first time
  anything reached the child's execution body. Everything past the configuration
  gate — detector construction, capture initialisation, the four ordered runs and
  the stop boundary — has still never been executed. Custody:
  `/home/ray/h2_rehearsal_ba40b3f8_20260729T140437Z_424acca1/` (45 files,
  read-only, `SHA256SUMS` self-digest `ae965b99…eba358b4a`). The archive never
  enters the canonical corpus; git records only digests, inventory, the
  reproduction and the custody reference.

- **2026-07-29 — the rehearsal-terminal repair.** Four commits, `C_reg → B → A →
  C_close`, in that order and not squashed.

  `B` stops the harness reading "the run directory exists" as "the run
  completed", which is how the failed `00_capture_off` was projected as
  complete. Completion now comes from the child's own durable lifecycle record —
  the one the controller already reads for the same decision — and the summary
  keeps `materialized`, `lifecycle_present`, `lifecycle_state` and `completed`
  as four separate facts instead of one. A missing record, an invalid one or an
  undeclared state is not completion. The state names and the record's filename
  move into the child, which performs both transitions out of `running`, and the
  binding is tested by moving that authority and watching the harness's verdict
  follow. This lifecycle-derived `ordered_runs` shape begins with
  `h2_phase_a_rehearsal_witness_v2`; the first rehearsal's permanently custodial
  v1 witness keeps its original shape and is not modified, rewritten or renamed.

  `A` sends the already-frozen choices through the surface `mot17_args` declares
  authoritative over them. `FIXED_EXECUTION_ARGV` names all four A5-fixed
  environment knobs rather than the two that were wrong, because the other two
  only resolved correctly from a decision-relevant preset that can move without
  this file noticing; with all four named, repository-owned configuration becomes
  a no-op over the frozen environment rather than a mutation that merely stays
  inside its declared set. `--latency-only` is fixed in the same commit: the
  evaluator returns before metrics only under it, otherwise it reads ground truth
  this measurement may not read and the child refuses the result — a
  contradiction already decidable from the source, and not worth another rebuild
  of the head-bound chain to discover.

  Nothing else moved. The mutation gate, `STATIC_ENV`, the generic parser and the
  preset are untouched; the last two are decision-relevant, and editing them
  would have turned an adapter defect into a decision-surface change. Four of the
  seven divergences from H0's vector are accepted as they stand, each because a
  stronger H2 contract already fixes what H0 fixed through that flag.

  The regression runs the production `repository_runner` over the real parser,
  the real preset read from disk and the real configuration function, stubbing
  only the GPU-bearing objects, and its sentinel sits at `run_eval` and refuses
  anything but `latency_only=True` — clearing the environment gate must not read
  as clearing the boundary behind it. Removing `FIXED_EXECUTION_ARGV` fails it.

  This repair authorizes nothing, seals nothing and restores nothing. It does not
  rehearse, does not build `F`, and does not claim any execution boundary past
  the one it fixes. `Acceptance` items 4 and 5 and `F` are stale for this head
  and must be rebuilt from scratch before the next single rehearsal.

- **2026-07-30 — frozen RunSpec authoring profile (PR #305 review).** The
  complete 454-key
  `docs/research/contracts/h2_phase_a_authoring_profile_v1.json` replaces
  Python `OWNER_DECLARED_VALUES` and live preset/parser-default resolution as
  the authoring authority. It is owner-bound by
  `h2_phase_a_run_spec_authoring_decision_v1.json`; `detector=null`,
  `max_frames=null`, `preset=null`, and `warmup_frames=50`.
  `mamba_whole_graph_m.yaml` remains authoring lineage only, with no runtime
  preset load. The profile, profile schema and owner decision join the declared
  execution-semantics projection. This resolves only authoring authority; the
  execution-code closure, transient-mutation gap and cross-verdict constraints
  remain successor implementation obligations.

- **2026-07-30 — RunSpec canonical byte domains (PR #305 review).** The
  ambiguous single canonicalization identifier is replaced by two required
  identifiers: object digests cover compact finite canonical JSON with no
  trailing LF, while `run_spec.json` serialization adds exactly one trailing
  LF. The frozen 454-key authoring profile and its SHA-256 remain unchanged.

- **2026-07-30 — controlled-host workflow downgraded to manual diagnostic.**
  Review Correction 8 removes the pull-request and `main` triggers from
  `.github/workflows/runtime_identity.yml`. Correction 5 already retired
  cross-host reconstruction, published coordinate/probe equality and unrelated-
  commit re-attestation as successor validity gates; keeping the expensive
  rebuild automatic would preserve their cost without their authority. Manual
  runs remain available for CUDA/TensorRT/input/probe diagnosis only. The
  successor execution still owns its build binding, extension load and identity
  run, and its archive remains independently verifiable without reconstruction.
  The legacy publication's static axes were regenerated from the existing
  read-only probe/runtime-input records; no workflow was dispatched and no
  controlled-host or native artifact was rebuilt. No authorization, execution,
  seal, corpus admission or equivalence transition occurred.
