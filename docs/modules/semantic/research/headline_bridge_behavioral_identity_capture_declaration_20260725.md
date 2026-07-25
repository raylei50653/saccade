# H2 — headline-m bridge capture under a bound coordinate and bounded probe

<!-- doc-status: proposed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-25 -->
<!-- doc-module: semantic -->

> **Status: proposed / draft-unsealed.** This declaration is **not** an execution
> seal. It selects no `I`, creates no `F` or `S`, authorizes no capture, grants no
> execution authority, establishes no runtime substrate, registers no guarantee,
> and starts no downstream unit (`H0_ROUTE5_B1`, `GCTM_B1`, O1, Phase B). It
> replaces **only the runtime-identity mechanism** of the
> [2026-07-13 H0 declaration](headline_bridge_full_decision_capture_declaration_20260713.md);
> every sealed H0 artifact, digest, amendment text, and terminal stays
> byte-frozen, and the five spent `S` chains stay permanently spent.

Research-wide type and upstream/downstream routing live in the
[research control plane](../../../research/README.md#research-control-plane).
This declaration owns only the H2 identity mechanism, its two-layer budget, and
its ordered terminal partition in §7.

---

## 0. Why a successor unit rather than an eleventh amendment

H0's scientific premise is retained without change: a claim about production must
be runtime-grounded, because a shared formula does not transfer semantics
([fidelity protocol](../../../research/contracts/runtime_quantity_fidelity_protocol.md)
core lemma; #112 measured ρ = 0.9558 / q95 = 1.417 on the same algebraic form).

What is replaced is H0's **operationalization of identity**: that the physical
file closure of the executing process is declarable in advance, and that a
mismatch is an epistemic terminal consuming an exactly-once authorization. Five
sealed invocations produced five `H0_PROVENANCE_INVALID` and zero faithful
capture. The owner's own R5 parity audit
([evidence](evidence/h0_r5_qualification_authoritative_parity_audit_20260725/))
records the decisive facts:

| Recorded field | Value |
| --- | --- |
| `membership_predicate_byte_identical_qual_vs_auth` | `true` |
| `F_bound_inputs_digest == authoritative` | `true` |
| `qualification_build_artifact_sha256 == authoritative` | **`false`** |
| `qualification_build_artifact_paths == authoritative` | **`false`** |
| extension identity | both `length = 4065896`; `sha256` `f374a223…` vs `b064a700…`; distinct ELF build-ids |
| `tool_runtime_count` / `plan_files` | `4518` / `4664` |
| `extension_load_record` | `not_recorded` (abort before write) |

Two consequences are structural, not operational:

1. **Physical artifact identity is not a function of source.** The same source
   built in a different directory yields a different `sha256` and build-id, so a
   passing qualification carries no information about the authoritative run's
   artifact identity. Pre-verification of the physical layer is impossible in
   principle under this identity notion, not merely difficult.
2. **The loaded closure is loader-emergent.** 4661 of 4664 declared plan members
   are host tool runtime that cannot reach bridge arithmetic, while the actual
   membership outcome depends on `dlopen` siblings, symlink path forms,
   non-canonical `..` loader paths, and interpreter `base_prefix` layout —
   discovered only by executing. Combined with an exactly-once budget, each
   re-entry chain purchases at most one bit of closure information (R5 purchased
   zero: the record was lost before persistence).

The same defect class had already appeared twice on the *admission* side:
[A6 Correction 1](headline_bridge_full_decision_capture_declaration_20260713.md#amendment-6-correction-1--admitted-runtime-surface-after-the-frozen-cuda-substrate-20260719-pre-seal)
records that ordinary accepted `main` landings made every descendant unsealable,
and states the diagnosis in the owner's own words — *"the admission rule, not the
provenance record, is what must be repaired."* H2 repairs the rule.

### 0.1 Supersession boundary (exact)

**Superseded for H2 execution** (H0 text remains byte-frozen and remains
authoritative *for H0's own closed history*):

- §2's runtime-bound repository inventory as an identity predicate;
- A6 / A6 Correction 1 — `h0_admitted_runtime_paths_v1`, `h0_projection_path_class_v1`,
  `h0_admitted_runtime_blobs_v1` content-pinned admission;
- A7 Review Correction 1's input-immutability-as-provenance-terminal;
- A8 build-tool provenance closure; A9 re-entry; A10's enumerative
  `h0_bound_inputs_v1.repository` binding (A10's *authority-overlay separation
  principle* is retained — see §5.4);
- A2.4 terminal 1 `H0_PROVENANCE_INVALID`.

**Consumed unchanged, by reference, not restated** (link, don't relabel — C5.1):

- the capture ABI `scripts/tools/h0_bridge_decision_trace_schema_v2.json` and
  its static admission `check_h0_bridge_decision_trace_contract.py`
  (`h0_coverage_v2`);
- A3 / A3.1 fail-closed envelope, mechanical writer admission, exposure equalities;
- A5 policy-target identity — the sole policy target remains
  `configs/presets/mamba_whole_graph_m.yaml`;
- **A7.6 frozen policy-visible comparison inventory** — the seven members and
  their required comparisons are consumed verbatim as H2's behavioral vocabulary;
- the sealed packet verifier's observer ABI, native-universe equality,
  conservation, canonical determinism, and replay predicates;
- A2.4 terminals 3–5 semantics, renamed under the `H2_` prefix in §7;
- the exactly-once authorization model — narrowed in §5 to the measurement only.

### 0.2 Why no Amendment 11 is appended to the H0 declaration

The supersession is recorded **here and in the navigation layer only**. No
pointer is appended to the H0 declaration, because that file is now byte-pinned
by sealed downstream packages: `h0_gctm_guarantee_registration_v3_20260724` and
`gctm_runtime_native_candidate_universe` freeze it by `path + sha256`, and the
only tolerated drift is *pure trailing `SEALED` owner-event rows*
(`scripts/tools/h0_declaration_frozen_identity.py::matches_frozen_sha256_or_sealed_append`,
which fails closed on any other trailing line). A pointer-only append was
attempted and mechanically rejected — five sealed-package tests failed — so it
was reverted rather than accommodated: widening the tolerance to admit amendment
prose, or re-freezing the packages, would launder a mutation exactly as the
fidelity protocol §2.8 forbids (`b43772b7` precedent — restore the sealed bytes,
never regenerate the checksum).

This is itself a third instance of the pathology in §0: byte-level identity of a
moving artifact welded a document shut. It also settles the governance form —
an eleventh amendment was not merely less tidy than a successor declaration, it
was **inadmissible**. Discovery direction therefore stays downstream-to-upstream
(C5.1: link, don't relabel): this declaration cites H0, the charter and module
TODO carry the navigation, and H0's bytes stay untouched.

---

## 1. §20.2 required declarations

```text
Target decision layer   none (cross-layer substrate work)
Study intent            boundary diagnostic
                        (primary; secondary: capability map of the capture surface)
Design objective        n/a — no design-evaluation intent is claimed; §20.3 role-legal
                        objectives are not invoked and no design candidate may be emitted
Selection rule          n/a for candidate selection; the terminal is selected
                        mechanically by the ordered partition of §7 (first applicable)
Validity gate           (a) Layer-P v2 pass certificate present and matching (§5.2);
                        (b) content-bound fixtures/assets/build artifacts equal F;
                        (c) bounded behavior probe equal to the published reference
                            at measurement launch (change detection only);
                        (d) A7.6 capture-off/on comparison computable on all four runs;
                        (e) three capture-on packets present and schema-valid.
                        Failure of (a)–(e) routes to §7, never to a re-run.
Stop condition          sufficiency: one terminal in §7 is selected.
                        futility: no futility stop exists inside a sealed
                        measurement — the partition is total and the invocation
                        is single.
Output class            diagnostic result (§20.4); on T5 additionally a
                        substrate-fidelity edge proposal for owner acceptance.
                        No design candidate, no performance upper-bound candidate.
Mainline transition     per terminal, §7 — every terminal names one; none is
                        "describe more and continue".
Type κ                  §1.1
```

### 1.1 Typed κ per decidable unit

Each unit is `κ = (quantification space, comparison relation, decision rule)`.
Spaces use the owner symbols of contract §20.9.2.

| Unit | Quantification space | Comparison relation | Decision rule |
| --- | --- | --- | --- |
| `identity.probe` | the identity fixture's finite frame set (§3.2), all A7.6 probe members | **exact byte equality** of the canonical inventory digest | unequal ⇒ observed difference and Layer-P hard stop (§5.3) or T1 at launch; equal ⇒ only "probe observed no difference", never equivalence |
| `policy.nonperturbation` | `U^evt` restricted to the measurement fixture; A7.6 members `mot_output`, `final_track_rows`, `active_tid_slot_pairs`, `relink_debug_raw`, `overflow_vector` | **exact** equality, capture-off vs each capture-on | any difference ⇒ T2 |
| `pair` | joined candidate–lost pairs, exact-key join `J_v` | exact equality of packed claim inputs and canonical digest across capture-on runs 1–3 | any difference ⇒ T3 |
| `candidate` | capture-on canonical candidate keys with `proposal_emitted=pass` | exact key-set, count, and SHA-256 equality; exact join to native proposal keys | any difference ⇒ T3 |
| `claim` | claim records and claim-winner transitions | exact replay agreement (gate / ranking / margin / claim) | any disagreement ⇒ T3 |
| `commit` | commit records, pre/post `track_id` and `active` | exact replay agreement; agreement with final state and bridge-accept count | any disagreement ⇒ T3 |

`identity.probe` is a finite-fixture diagnostic κ over frame units. It carries no
claim-level ε statement and no implication outside that frame set. In particular,
its equality is not an implementation→measurement-domain injection and cannot
preserve an old claim. No unit here quantifies over trial units `T_v`; H2 makes no
ε-bounded claim and §20.9.6's bound interface is not invoked.

### 1.2 §20.9.1 four declaration coordinates

1. **target decision layer** — `none` (cross-layer substrate work);
2. **study intent** — `boundary diagnostic`;
3. **κ quantification space** — event/pair/frame units (§1.1); no `T_v`;
4. **substrate** — the production CUDA foot-bridge decision path at the
   published runtime coordinate (§4), measured on the fixture of §3.3.

Registered-object agreement: H2 consumes `quantity.bridge_capture_provenance`
whose accepted record is `substrate = D0/R1/S0 shadow capture provenance
(runtime CUDA)`, `target_substrate = same`, state
`P0_CAPTURE_SEMANTICS_UNVERIFIABLE`
([claim-state registry](../../../research/contracts/claim_state_registry.md)).
H2 does **not** write that record; the registry remains the single state writer
(C5.1). H2 proposes only the addition of a `captured_under` field (§8.4) for
owner acceptance.

---

## 2. What the seal will and will not authorize

The seal, when and if it is issued, authorizes exactly one Layer-M measurement
invocation (§5.2) of an observational capture on the fixture of §3.3, and
nothing else. It authorizes no GT/FP read, no threshold variation, no policy
choice, no preset change, no registry or ledger write, no Phase B, and no
downstream unit activation. A passing measurement is a *precondition* for owner
consideration of downstream units, never their activation.

---

## 3. Frozen configuration and the two fixtures

### 3.1 Policy base

The sole policy target is `configs/presets/mamba_whole_graph_m.yaml` (A5,
consumed unchanged). Its resolved-parameter fingerprint is the
`decision_surface` axis input of §4.

### 3.2 Identity fixture (Layer P, and the published reference)

```text
sequence     MOT17-09-SDP  (525 frames; shortest 7-seq member)
mode         identity-mode configuration
determinism  known nondeterminism sources pinned off:
             gpu_decode = 0, single-threaded relink
purpose      deterministic change detector for decision-relevant code
```

**Identity-mode configuration is not a production-equivalence claim.** It is a
bounded change detector: digest inequality proves a difference on this fixture;
digest equality says only that this fixture observed no difference. A branch that
fires only for MOT17-04 state, size, gap, or candidate combinations can change
Layer-M behavior while this digest stays equal. Any production statement comes
from §3.3 or a separately accepted sufficiency proof, never from §3.2 alone.

### 3.3 Measurement fixture (Layer M)

```text
sequence     MOT17-04-SDP
mode         the A5 policy target, unmodified, as A7.6 requires
runs         00_capture_off, 01/02/03_capture_on, in that order
```

The measurement fixture keeps H0's environment exactly (including
`SACCADE_GPU_DECODE=1`): the point of the measurement is what production does,
and pinning determinism there would substitute a different `R`.

---

## 4. Coordinate, probe, and equivalence are distinct

The published schema `h2_runtime_coordinate_probe_v1` has three explicit layers:
`coordinate` versions what ran, `probe` records one bounded observation, and
`equivalence.state` remains `unproven`. No consumer may collapse the latter two.

| Coordinate / probe | Definition | Derived from |
| --- | --- | --- |
| `decision_surface` | canonical digest of the fully resolved parameter snapshot of the A5 preset, plus the declared kernel decision constants | `scripts/eval/config/gen_golden_snapshot.py`; guarded by `check_headline_decision_contract.py` |
| `implementation` | content digest of the complete `src/` + `include/` production implementation and resolved eval surface, not only paths exercised by MOT17-09 | `h2_path_partition.py` |
| `environment` | build recipe digest + `uv.lock` digest + CUDA / TensorRT / driver versions (optionally the GHCR image digest) | `CMakeLists.txt`, `uv.lock`, `docker_build.yml` |
| `identity_semantics` | content digest of the digest producer, classifier, publisher, staleness checker, terminal partition, Layer-P certificate producer, workflow and schema policy | exact authority set in `h2_path_partition.py` |
| `runtime_inputs` | content digest of both sequence fixtures, every configured weight/checkpoint/engine, and executed third-party evaluator code | `h2_runtime_inputs.py` |
| `probe` | canonical digest of the four capture-off-computable A7.6 members on MOT17-09 | `h2_behavioral_identity.py`; sufficiency is `fixture_change_detector_only` |

### 4.0 The behavior-probe member set (exact)

The probe digests exactly these A7.6 members, in canonical key order:

```text
active_tid_slot_pairs      per frame, sorted by slot
final_track_rows           frame, row index, raw binary32 bits, class, final track ID
mot_output                 complete file bytes length + SHA-256
relink_debug_raw           the complete 13-integer vector
```

These are exactly the members A7.6 requires to be **equal between capture-off and
every capture-on run** — that is, the policy-visible state. The other three
members are deliberately excluded and the exclusion is part of the frozen
definition:

- `proposal_projection` and `winner_commit_projection` are **trace-only**
  projections that do not exist with capture disabled; A7.6 states outright that
  they may not be fabricated for capture-off. Including them would make the axis
  uncomputable in identity mode.
- `overflow_vector` is a capture-on predicate (every value exactly zero), not a
  policy-visible state, and its semantic counters are meaningless when the trace
  buffers are not allocated.

The identity run is therefore **capture-off**, which is also what makes the probe
publishable by the online track without enabling any research instrumentation.
`run_h0_phase_a_child.py` is not modified: it is hash-pinned inside historical
freeze artifacts, so H2 reimplements the technique parameterized rather than
editing frozen plumbing.

### 4.1 Bound runtime inputs versus witness

Content hashes of both fixtures, configured weights/checkpoints/engines,
extension, TensorRT plugin, sequence metadata, and executed TrackEval code are
**certificate/F inputs**. They define which measurement was authorized; they do
not prove equivalence to any other coordinate.

ELF build-id, broad host `tool_runtime`, observed regular-file closure, unrelated
dynamic dependencies, and NVML serial identity remain witness-only.

> **No terminal in §7 may be selected on a witness field.** A witness mismatch is
> recorded and reported; it never selects a terminal and never blocks a Layer-P
> retry.

### 4.2 Why this is tighter on the axis that matters

The threat model is (i) different decision-relevant code, (ii) mutated inputs,
(iii) capture perturbing decisions.

- (i) is versioned by `decision_surface`, broad `implementation`, and
  `identity_semantics`; the finite probe is an additional detector, not coverage
  proof.
- (ii) is content-bound by `h2_runtime_inputs.py` and watched during execution by
  `BoundInputMonitor`, including ignored datasets and model/build assets.
- (iii) is A7.6's capture-off/on equality, unchanged.

Retained physical predicates, both decidable without prior closure knowledge:

1. extension and plugin were **actually loaded and consumed** — the
   `CONSUMING_OPERATIONS` clause of `assert_build_artifact_observed`, with its
   pre-declared-membership clause removed;
2. no bound input was written during the run.

---

## 5. Two-layer budget

### 5.1 Layer P — plumbing, pre-seal, retryable

Scope: build, build-tool binding, extension/plugin load, confinement setup,
attestation recording, evidence serialization.

Layer P is **pre-seal engineering**. It produces no H2 terminal and consumes no
authorization; its results are `plumbing_pass` or
`plumbing_blocked(<coordinate>)`. This is not a new permission: Amendments 1–10
were all pre-seal repairs and none was a terminal. What changes is that H2's
pre-seal validation is *self-contained and repeatable*: each attempt records an
exact final source/input/build coordinate plus a bounded smoke probe before any
seal. No attempt claims equivalence to another coordinate. H0's pre-seal
validation could only be tested by spending an authorization because
qualification provably did not transfer (§0).

**Every rejection must persist its coordinate before aborting.** R5's
`extension_load_record: not_recorded` is inadmissible here: a retryable budget is
worthless without the coordinate to retry against.

#### 5.1.1 Worked instance — a defect Layer P found for free

The first identity run under this design aborted on
`run_h0_phase_a_child.py`'s frame-level assertion

```python
if pairs != sorted(pairs, key=lambda pair: pair[1]):
    raise ChildContractError("active tid/slot pairs are not sorted by slot")
```

The assertion is false in general. A7.6 defines the canonical value as the pairs
**sorted by slot** — a normalization the recorder owes — while the native
`get_active_tid_slot_pairs()` iterates
`std::unordered_map<int,int> h_tid_to_slot_` (`src/tracking/tracker_gpu.cu:5084`)
in track-**id** bucket order. Slot order therefore holds only by coincidence,
typically on early frames with small consecutive ids, which is why a prefix can
pass before a later frame fails.

Consequences, stated exactly:

- H2 records `sorted(raw, key=slot)` and asserts the invariant actually worth
  asserting — no duplicate slot — rather than asserting the native order.
- The frozen child is **not edited**: it is hash-pinned inside historical freeze
  artifacts, and this declaration only records the defect.
- Under H0's structure this defect was reachable **only inside a sealed
  measurement**, where it maps to `runner_nonzero` → `H0_EXECUTION_INVALID`
  (A7.7). A five-line recorder bug would have consumed a sixth exactly-once
  authorization. Under H2 it cost one retryable Layer-P run. This is the two-layer
  split earning its keep on the first attempt, and it is why §7 keeps a
  post-launch execution terminal while refusing to charge pre-seal plumbing to
  the epistemic budget.

#### 5.1.2 Gate G1 result (2026-07-25, non-authoritative)

The design's central premise is falsifiable in minutes and was tested before
anything downstream was built:

| Build directory | extension length | extension `sha256` | behavior-probe digest |
| --- | ---: | --- | --- |
| `build/` | 4 115 232 | `79ea49453402de47…` | `2dabed0bc05e3bc7…` |
| `build/ci_arch214/` | 4 065 896 | `8d3cdb61b7b9d1cd…` | `2dabed0bc05e3bc7…` |
| `build/h2_layer_p/` (built fresh by Layer P) | 4 065 896 | `1ad797d26bd4c8ab…` | `2dabed0bc05e3bc7…` |

Three physically distinct binaries — different hashes, two of them at different
lengths — produce the **identical bounded MOT17-09 probe**. G1 establishes only
the forward observation "these builds produced the same probe"; it does not
establish the reverse implication from equal probe to equal measurement-domain
behavior. A fourth artifact already on disk,
`build/h0_phase_a/` at `b064a700…`, is byte-different from `build/ci_arch214/` at
the *same* 4 065 896 length — independently reproducing the R5 parity audit's
finding without a rebuild.

The third row is the important one: it was produced by a fresh
`cmake` configure + build inside the Layer-P controller (34 s + 82 s), so the
comparison is not between pre-existing artifacts of uncertain provenance but
between a new build and the published reference.

The tool verifies which binary it loaded rather than assuming: a
site-packages `.pth` pins `<repo>/build` into `sys.path`, so an unverified
`SACCADE_BUILD_PATH` override was silently ignored and a two-build comparison can
appear to pass while testing one binary twice. The runner now fails closed unless
both the selected extension and selected TensorRT plugin are actually mapped,
and their content hashes enter the Layer-P certificate.

#### 5.1.3 Gate G2 result (2026-07-25, non-authoritative)

Three repeats on the **measurement** fixture (§ 3.3: MOT17-04-SDP, the A5 policy
target, GPU decode on, nothing pinned), capture-off:

```text
run 0 / 1 / 2   behavior = bd88260a76ceb3954318b99ecb6037b6a72dce36773f1835b16664f5f9578a5a
identical = true
```

This addresses a risk **independent of any identity mechanism**: if production
were not byte-reproducible on the policy-visible inventory, a sealed Layer-M
would select `H2_CAPTURE_PERTURBS_POLICY` or `H2_PACKET_INVALID` no matter how
provenance was established. The A7.6 comparison assumes exact equality across
four runs, and until now nothing had ever tested that assumption — H0's five
invocations all terminated before the runs stage.

Scope of the result, stated narrowly: 3 repeats, one sequence, one host, one
build, capture-off, on the four § 4.0 members. It does **not** establish
determinism of metrics, of other sequences, or of the capture-on path, and known
nondeterminism sources (GPU-decode race, relink threading) remain recorded
elsewhere as real for other quantities. What it establishes is that the specific
equality Layer M depends on is not obviously false — obtained for free, pre-seal,
which is exactly the capability H0's structure lacked.

### 5.2 Layer M — measurement, sealed, exactly-once

Scope: the four ordered runs of §3.3, the A7.6 comparison, the three capture-on
packets, and every replay predicate.

Preconditions bound into the seal:

1. a Layer-P pass certificate (`h2_layer_p_certificate_v2`) binding source
   HEAD/tree, selected base, complete changed-path verdict, decision-relevant,
   identity-semantics and plumbing-set digests;
2. exact SHA-256 of the published coordinate/probe file, probe-result file, and
   runtime-input manifest;
3. content digests of both fixtures, all configured weights/checkpoints/engines,
   extension, TensorRT plugin, sequence metadata and third-party runtime code;
4. the fixed probe fixture/mode/schema and `equivalence = unproven`;
5. the capture ABI digest.

One `I → F → S` chain, one exactly-once authorization, consumed at controller
process launch. **Retry, resume, and repaired re-run under the same `S` remain
permanently forbidden.** Retryability exists only *before* the seal.

### 5.3 Firewall against a circular oracle

Retryability is the only property of H2 that needs defending, because H0's
one-shot budget was protecting against a real hazard — the E1b circular oracle,
where implementation, schema, verifier, and tests drifted to fit each other.
H2 replaces "you get one attempt" with "the thing you must not change is
mechanically pinned":

1. **Path partition** (`scripts/tools/h2_path_partition.py`, single source).
   Classification is total and mutually exclusive; an unclassified path is
   **fail-closed** — silence is not permission, which is the assertion H0's
   enumerative admission kept getting wrong.
   - `decision_relevant` — the full production `src/` and `include/` surfaces,
     resolved eval configuration, selected preset and entrypoint. MOT17-09
     execution coverage is not used to shrink this coordinate.
   - `identity_semantics` — the ruler itself: probe producer/member definition,
     classifier, runtime-input manifest, publisher, staleness checker, terminal
     partition, Layer-P certificate producer, workflow and schema policy.
     Rejected because a retry cannot edit what decides whether it changed.
   - `identity_fixture_input` / `measurement_input` — MOT17-09 and MOT17-04
     content/metadata respectively.
   - `runtime_asset` — weights, checkpoints, engines, extension/plugin build
     products and executed third-party components.
   - `plumbing_only` — build recipe, loader environment, confinement plan,
     attestation recorder, evidence serialization, controller and verifier
     scaffolding, CI, infra.
   - `non_execution` — prose, tests and outputs not consumed by the run.

   A Layer-P retry may change `plumbing_only` and `non_execution` paths and
   nothing else. Note the asymmetry with H0's `h0_projection_path_class_v1`: a
   misclassification here does **not** gate a seal (which is what made A6
   Correction 1 necessary). Final plumbing and runtime artifacts are content-bound
   into the certificate/F rather than declared equivalent to an earlier build.
2. **Bounded probe.** Every Layer-P pass recomputes the fixed MOT17-09 probe.
   Inequality is `H2_PLUMBING_CHANGED_PROBE`. Equality means only that this probe
   observed no difference; `equivalence` stays `unproven`.
3. **Append-only retry log.** Each retry records the coordinate it resolved. No
   silent iteration.
4. **Seal binds everything.** At `F` the coordinate, bounded probe, selected base,
   authority/plumbing digests, both fixture digests, all runtime-asset digests,
   extension/plugin bytes, and exact certificate/result/manifest files are frozen
   together. This identifies one measurement without claiming cross-coordinate
   equivalence.

### 5.4 What A10's principle is retained for

A10 separated *who may write* (owner authority overlay) from *what is bound*
(runtime inputs), after the declaration's dual role invalidated `S3`. H2 keeps
that separation and extends it one step: it also separates **plumbing
verification** from **terminal measurement**. A10 cut the first seam; the second
seam is what five terminals were actually spent on.

---

## 6. Behavioral vocabulary

H2 introduces no comparison vocabulary of its own. The complete policy-visible
non-perturbation inventory is A7.6's seven members with their required
comparisons, consumed verbatim; the packet verifier's predicates are consumed
verbatim. No tolerance, similarity measure, unordered comparison, scheduling
proxy, or additional counter may be substituted for either.

---

## 7. Terminal partition (ordered; first applicable is authoritative)

Reachable outcome space: **one sealed Layer-M invocation.** Layer-P outcomes are
pre-seal and are not in this space (§5.1).

| # | Terminal | Exact condition | Mainline transition (§20.7) |
| --- | --- | --- | --- |
| 1 | `H2_INPUT_MUTATED_DURING_MEASUREMENT` | `BoundInputMonitor` observes a write to any bound input during the invocation, **or** the behavior probe at launch differs from the reference bound in `F`, **or** the Layer-P certificate/content manifest does not match `F` | closes the H2 measurement unit; `quantity.bridge_capture_provenance` unchanged; candidate set stays empty; a fresh `I → F → S` and a separate authorization would be required |
| 2 | `H2_CAPTURE_PERTURBS_POLICY` | Execution completed and any A7.6 capture-off/on equality differs | **closes the observational-capture route itself**: decision-neutral shadow capture is not achievable at this ABI, so grounding must proceed by native-side reproduction or not at all |
| 3 | `H2_PACKET_INVALID` | Non-perturbation held but any packet, exposure, overflow, native-universe, conservation, cross-repeat canonical digest, or replay predicate fails | closes this measurement; routes to a separate capture-ABI-delta charter (no ABI change is authorized here) |
| 4 | `H2_MEASUREMENT_EXECUTION_INVALID` | After the sealed launch: nonzero build, extension/plugin load failure, runner nonzero, deadline exhausted, serialization failure, missing or unreadable required artifact, or any unclassified execution failure (mandatory catch-all) | closes this measurement with no partial-capture reinterpretation; a fresh chain would be required |
| 5 | `H2_FULL_COMMIT_CAPTURE_FAITHFUL` | Every preceding condition false, all three capture-on packets and all verifications pass, **and** the frozen unlabelled seven-sequence Phase B is complete | **adds a decision capability**: a runtime-fidelity edge becomes available for owner acceptance, which is the precondition (not the activation) of `H0_ROUTE5_B1` / `GCTM_B1` / O1 |

Phase A alone may emit only terminals 1–4. Terminal 5 stays unavailable until the
seven-sequence Phase-B artifact exists; Phase A never starts Phase B.

This table is **executable**: `scripts/tools/h2_terminal_partition.py` is its
single mechanical form (`--explain` prints it; `--select` maps one observation to
one terminal). § 20.8's governing test — two independent implementers record the
bit-identical terminal — is not meetable from prose, so the selection order, the
exhaustive result mapping including the mandatory execution catch-all, the
fail-closed behavior on missing or non-boolean predicates, and the rule that **no
witness field can select a terminal** are all pinned by contract tests rather than
asserted here.

**On the difference from H0's partition.** Execution failures are *not* unmapped:
terminal 4 is the fail-closed catch-all required by §20.8 item 3. Two things
change. `provenance_invalid` disappears **as a predicate**, because enumerative
closure membership is no longer computed at all; and plumbing failures that
occur *before* the sealed launch are pre-seal engineering rather than terminals —
which is what H0's structure made impossible, since its only validation channel
was post-seal.

---

## 8. Sealability (§20.8)

### 8.1 Frozen degrees of freedom

- **Digest algorithm** — SHA-256, lowercase hex, over canonical UTF-8 JSON:
  lexicographic object keys, compact separators, finite numbers, one trailing LF
  (H0's `h0_phase_a_execution_v1` convention, consumed unchanged).
- **Equality** — byte equality throughout. No tolerance, no rounding, no
  netting of off-diagonal disagreement; there is no grey band and no weighted
  total.
- **Ordering** — `regular_files` and checksum listings sort by path bytes;
  runs execute in the §3.3 order; the §7 partition is evaluated top-to-bottom.
- **Float handling** — `final_track_rows` compares raw binary32 bit patterns,
  never decimal renderings.
- **Fixtures** — §3.2 and §3.3 are frozen by name and frame count.
- **Determinism pinning** — applies to the identity fixture only, with the exact
  settings named in §3.2; the measurement fixture pins nothing.
- **No equivalence from probe equality** — `implementation`, `environment`, or
  runtime-input drift with an equal probe is `re_attestation_required`;
  `decision_surface`, `identity_semantics`, or probe drift is `stale` (§8.4).
- **No refit** — no proxy, threshold, estimator, or classifier may be adjusted to
  close a gap; a Layer-P retry may only change `plumbing_only` paths (§5.3).

### 8.2 Mechanical decidability

Every condition in §7 and every gate in §1's validity block is a byte comparison
or a boolean over committed artifacts. No condition contains a judgment
predicate. The two-implementer test of §20.8: given this declaration, the frozen
fixtures, the Layer-P certificate, and `F`, two independent implementers record
the bit-identical terminal, because every input to §7 is a digest equality or a
process exit condition.

### 8.3 Blind→reveal

H2 has no blind phase: the capture is unlabelled and no GT/FP read is authorized.
§20.8 item 6 is therefore not applicable. The hash bindings that would carry it
are still recorded (`F` binds the runner, controller, verifier, capture ABI,
Layer-P certificate, runtime-input manifest, and probe reference).

### 8.3.1 Published coordinate/probe and the `online → research` guard

The coordinate/probe publication is
`docs/reference/runtime_identity.generated.json`; that generated file is the
authority for current digests. Its structure is:

```text
coordinate:
  decision_surface / implementation / environment
  identity_semantics / runtime_inputs
probe:
  kind: identity_probe
  sufficiency: fixture_change_detector_only
equivalence:
  state: unproven
```

| Perturbation | Result |
| --- | --- |
| any static coordinate differs from recomputed publication | hard failure, with regeneration command |
| fixture/model/engine content differs from supplied manifest | hard failure |
| `implementation` / `environment` / `runtime_inputs` differs in a binding while probe stays equal | `re_attestation_required`; consumer inadmissible |
| `decision_surface` / `identity_semantics` / probe differs | `stale`; consumer inadmissible |

`captured_under` lives in the sidecar
`docs/research/contracts/runtime_identity_bindings_v1.json`, **not** parsed out of
the registry Markdown — the same rule that made `declaration_policy_binding_v1` a
sidecar: nothing in this repository should reach a fail-closed verdict through a
partial Markdown parser. The registry stays the sole writer of object state
(C5.1); the sidecar holds only digests. No binding is populated: five spent H0
chains produced no faithful capture, so there is nothing to bind, and a contract
test pins that absence so it cannot be filled in retroactively.

The bounded probe needs a GPU, so it is re-attested by
`.github/workflows/runtime_identity.yml` on the same controlled host as H0
qualification. The workflow is triggered by implementation, environment,
identity-semantics and publication changes and also binds runtime-input content.

### 8.4 Proposed registry field (owner decision, not written here)

```yaml
captured_under:                    # proposed addition to a state record
  coordinate:
    decision_surface:   <sha256>
    implementation:     <sha256>
    environment:        <sha256>
    identity_semantics: <sha256>
    runtime_inputs:     <sha256>
  probe: <sha256>
```

Consumption rule: decision-surface, identity-semantics or probe drift is
**stale**. Implementation, environment or runtime-input drift with equal probe is
**re-attestation-required**, not behavior-preserving. V1 has no equivalence
upgrade: adding one requires a versioned verifier and accepted full
measurement-domain or explicit sufficiency evidence. Both verdicts are
version-lag accounting, not retraction of a claim about its original coordinate.

### 8.5 Scoped exhaustion

No terminal in §7 claims exhaustion of any signal family. Terminal 2 claims
closure of one route (observational shadow capture at this ABI) and nothing
wider. Terminals 1, 3, and 4 are bookkeeping closures of a single invocation and
must not be reported as evidence about the bridge, the capture surface, or
production behavior.

---

## 9. Open owner decisions (must be resolved before any seal)

```text
decision_surface: https://github.com/raylei50653/saccade/issues/286
identity:         h2_identity_mechanism_and_partition_decision_20260725
authorizes:       nothing — no I, no F/S, no capture, no exactly-once grant
```

1. §7's partition — provenance removed as a predicate; pre-seal plumbing failures
   are not terminals. **Load-bearing.**
2. §8.4's registry field addition (C5.1: the registry is the single state writer).
3. The slot name `H2`, terminal prefix `H2_*`, and evidence prefix
   `h2_measure_<I40>` (deliberately *not* `h0_phase_a_*`, so
   `check_h0_phase_a_archives.py` keeps verifying the frozen v1 corpus under the
   v1 schema).
4. The identity fixture of §3.2 (default MOT17-09-SDP).

## 10. State effect

None. No `I`, `F`, `S`, seal, authorization, guarantee, registry write, or
production change follows from this document. H0's closed history, its five spent
`S` chains, and the permanent ledger entry on
`quantity.bridge_capture_provenance` are unchanged: there is still no faithful
capture, no accepted runtime-fidelity edge, and no actual H0 guarantee envelope.

---

## Review Correction 1 — continuous runtime-input binding (2026-07-25, pre-seal)

This correction closes the manifest/monitor TOCTOU found in PR #287 review. It
changes no authority state and grants no execution.

For §4.2 predicate 2 and §5.3, "watched during execution" now has this mechanical
order:

```text
discover lexical consumer paths and symlink chains
→ start BoundInputMonitor
→ build and persist the content manifest
→ execute the bounded identity probe
→ validate every recorded file, path target, symlink chain and full membership
→ final monitor drain
→ close the monitor
→ only then admit certificate construction
```

The manifest records both the configured lexical path actually opened by the
consumer and its resolved target, plus every multi-hop symlink component and
link text (intermediate targets introduced by earlier links included; loop
detection fails closed). Layer P monitors the configured paths, resolved
targets, and multi-hop chain members before hashing begins. An equal-content
symlink retarget of a configured or intermediate link, a fixture member added
after hashing, or a content mutation in the former pre-monitor window therefore
blocks Layer P and cannot enter a pass certificate.

The controlled-host re-attestation workflow also runs against the exact head SHA
of same-repository pull requests. Fork pull requests are rejected at job scope
before any untrusted code can reach the self-hosted runner.

---

## Review Correction 2 — the four §9 decisions are resolved (2026-07-25, pre-seal)

This correction records an owner decision. It changes no authority state, grants
no execution, selects no `I`, creates no `F`/`S`, and writes no registry state.

**§9 is closed.** Its four decisions were accepted on 2026-07-25 on the decision
surface it names ([#286](https://github.com/raylei50653/saccade/issues/286),
resolved and closed). §9 above is retained as the historical statement of the
question; it is **no longer an open-decision authority**, and this correction is
its single resolution pointer.

| §9 item | Verdict |
| --- | --- |
| 1 — the §7 partition | accepted, with the §5.2 precondition below |
| 2 — §8.4's registry field | accepted as **schema only**; write condition below |
| 3 — slot `H2`, prefix `H2_*`, evidence prefix `h2_measure_<I40>` | accepted |
| 4 — §3.2's identity fixture MOT17-09-SDP | accepted |

### §5.2 gains a sixth precondition — the Phase-B chain

A fully successful Phase A selects **no terminal**: the executable partition maps
`measurement_pass` to no terminal, and §7's terminal 5 additionally requires the
frozen seven-sequence Phase B. But §2 states the seal authorizes no Phase B, §7
states Phase A never starts Phase B, and §5.2 forbids retry, resume, and repaired
re-run under the same `S`. No Phase-B chain is defined in this declaration: Phase
B appears only as *not authorized*, *required for terminal 5*, and *never started
by Phase A*.

Stated exactly: without a Phase-B chain, the best available outcome of the one
authorized Layer-M invocation is a spent authorization, an unclosed unit, and no
mainline transition under §20.7.

This is not inherited from the H0 declaration, whose seal covered sealed Phase A
and then, only if admitted, Phase B within one chain. H2 narrowed the seal to
Phase A alone without supplying the chain terminal 5 depends on.

Therefore §5.2's bound preconditions gain:

```text
6. a declared Phase-B chain form — its own I → F → S, its authorization form,
   and its precondition on a passing Phase A — published before the Phase-A seal
```

The seal's scope is unchanged: it remains one Layer-M Phase-A invocation, and §2
still authorizes no Phase B. What this precondition requires is that the success
path exist on paper before an authorization can be spent reaching it.

### §8.4's write condition (single semantics)

`captured_under` is accepted as a **schema**, not as a value to be written now.
The single mechanical rule is the one already carried by the sidecar
`docs/research/contracts/runtime_identity_bindings_v1.json`:

```text
a binding appears only if an H2 Layer-M measurement reaches terminal 5
and an owner accepts it
```

Consequences, stated so no second rule can be inferred:

- the sidecar row for `quantity.bridge_capture_provenance` already exists with
  `captured_under: null`; `null` means *no substrate-version claim*, never
  agreement, and it flips to a coordinate only under the rule above;
- evidence packets from terminals 1–4 record their own capture coordinate inside
  the packet, because a measurement must describe the coordinate it ran on; that
  packet-local record is **not** an object-level binding and never becomes one;
- no retroactive binding is written for evidence predating published identities.

### Pre-seal re-pin (and a drift this repairs)

H2 is **pre-seal**, so the binding's own rule applies: the pinned prefix is the
entire current document, and each review correction must force a conscious re-pin
rather than slipping through. The narrow prefix that ends above an appendable
region is reserved for *after* an owner seals a body.

This correction is therefore accompanied by a deliberate re-pin of
`sealed_prefix` to the full post-correction body, and by the resulting republish
of `docs/reference/runtime_identity.generated.json`, since the binding file is an
`identity_semantics` member.

That re-pin also repairs an existing drift. `Review Correction 1` was appended
without a re-pin, which is why the pinned boundary sat below it and why an append
could momentarily look free. It was not free — it was an unrecorded exception to
the pre-seal rule. The byte-level test cannot catch this on its own: it verifies
that the pinned prefix still hashes correctly and deliberately tolerates trailing
content, so a green suite is not evidence that the pre-seal re-pin rule was
honoured. Only the binding's re-pin log is.

### What this correction does not do

It does not seal, authorize, or schedule anything; does not alter §§0–8 above the
pinned prefix; does not change the capture ABI, A7.6, the packet verifier, or any
preset; and does not unblock `H0_ROUTE5_B1`, `GCTM_B1`, or O1. H0's closed
history, its five spent `S` chains, and the permanent ledger entry on
`quantity.bridge_capture_provenance` remain unchanged.
