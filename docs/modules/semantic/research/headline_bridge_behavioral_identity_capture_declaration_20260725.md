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

---

## Review Correction 3 — the Phase-B chain form (2026-07-25, pre-seal)

This correction records an owner decision and publishes the chain that §5.2
precondition 6 requires. It changes no authority state, grants no execution,
selects no `I`, creates no `F`/`S`, issues no exactly-once authorization, runs no
sequence, and writes no registry or sidecar state. Precondition 6 asks that the
success path **exist on paper before an authorization can be spent reaching it**;
this correction is that paper and nothing more.

Decision surface: [#290](https://github.com/raylei50653/saccade/issues/290),
accepted 2026-07-25 with three normative narrowings, which are incorporated
below rather than recorded as deltas. The issue remains open as the standing
surface for this chain. Its Revision 1 supersedes the original §1, §3, §4 and
Consequence D of that issue; where the issue and this correction differ, this
correction governs — an owner decision surface is not a declaration.

### C3.1 `I_B` — the Phase-B instance

```text
I_B := (I40_B, F_B)
```

`I_B` is a new instance, never a continuation or resumption of `I_A`. `I40_B` is
the exact 40-lowercase-hex head under Phase-B seal; `I40_B ≠ I40_A` is permitted
and `I40_B = I40_A` is permitted, because the head alone does not identify the
instance.

A Phase-A result is admissible into `I_B` only if:

```text
(a) a Phase-A evidence root exists whose recorded observation selects no
    terminal (result = measurement_pass under h2_terminal_partition.py), and
    whose manifest and checksum inventory verify; and
(b) all five coordinate axes and the bounded probe recorded in F_A are
    byte-equal to the publication resolved at F_B construction:
      decision_surface · implementation · environment
      identity_semantics · runtime_inputs · probe
```

Head inequality is permitted precisely because a head may differ only through
paths that move no axis — which is what allows the Phase-B controller, child,
verifier, archive checker and observation emitter to be written *after* the
Phase-A seal (`h2_path_partition.py` classifies `scripts/tools/run_h2_*`,
`verify_h2_*` and `check_h2_*` as `plumbing_only`).

If (b) fails, §8.3.1's consumption rule applies unchanged and without exception:
`decision_surface` / `identity_semantics` / probe drift is `stale`;
`implementation` / `environment` / `runtime_inputs` drift is
`re_attestation_required`. Both make the Phase-A evidence inadmissible. There is
no tolerance, no carve-out, and no inert-member exemption; inventing one would be
the equivalence upgrade §8.4 forbids in v1.

**Evidence root.**

```text
h2_measure_b_<I40_B>_<F64>
    F64 = the complete 64-lowercase-hex canonical digest of the F_B freeze record
```

The full digest is used, never a truncation: an evidence root is an identity, and
a shortened digest trades a collision probability for a cosmetic path length.
Uniqueness is mechanical rather than conventional — every successor `F_B` must
bind `prior_attempts`, the complete ordered list of preceding **consumed-attempt
records** for this Phase-A result (C3.5.1), each of which must exist and verify
in its own class — so two attempts cannot share an `F_B` digest even at a
byte-identical head. `check_h2_measure_archives.py` recomputes the digest from
the recorded freeze record, rejects any root whose name does not match it, and
rejects an incomplete `prior_attempts` list or one naming a root that does not
exist or does not verify. The accepted `h2_measure_` family (§9 item 3) is preserved,
`capture_phase` is a required manifest field, and
`check_h0_phase_a_archives.py` remains untouched.

### C3.2 `F_B` — the freeze

`F_B` binds, and a Phase-B launch is admissible only against it:

1. **the Phase-A result** — evidence root path, manifest digest, checksum
   inventory digest, the recorded six-predicate observation, and its
   `measurement_pass` selection;
2. **the coordinate** — the five axes and the bounded probe, each recorded *and*
   asserted equal to `F_A`'s value; this equality is a bound predicate, never a
   witness field (§4.1);
3. **a Layer-P v2 pass certificate** (`h2_layer_p_certificate_v2`) for the exact
   Phase-B head, `--base` given, full changed-path verdict clean;
4. **runtime inputs** — the complete runtime-input manifest of all seven
   sequences, both fixture roles, every configured weight/checkpoint/engine,
   extension, TensorRT plugin, sequence metadata, and executed third-party
   evaluator code. `F_B` binds the manifest's **`full_digest`**, never its
   `coordinate_digest`: the coordinate deliberately excludes `build_artifacts`
   (`h2_runtime_inputs.py`) so that one coordinate can span builds, and it is
   `F`/certificate scope — §4.1 — that pins which physical bytes a measurement
   was authorized to consume;
5. **the consumed-unchanged surface** — the capture ABI digest
   (`h0_bridge_decision_trace_schema_v2.json`), the packet verifier, and the A7.6
   seven-member inventory (§6: H2 introduces no comparison vocabulary of its own);
6. **the executed code** — Phase-B controller, child/recorder, verifier, archive
   checker, and observation emitter digests;
7. **`phase_a_evidence`** — the complete manifest of the bound Phase-A evidence
   root, and its membership in the `BoundInputMonitor` watch set (C3.6);
8. **`prior_attempts`** — complete and ordered, per C3.1 and C3.5.1;
9. **the exposure declaration**, in H0's own frozen vocabulary:

   ```text
   capture_phase              = phase_b
   require_candidate_exposure = true
   require_commit_exposure    = true
   ```

   Per sequence: nonzero candidate exposure required. Over the seven-sequence
   union: nonzero commit exposure required. Per-sequence commit exposure is
   recorded and may be zero — A3.1's bar is that a Phase-B artifact with no actual
   commit path cannot support terminal 5, not that every sequence must commit;
10. **the run plan** — the seven sequences in lexicographic order

    ```text
    MOT17-02-SDP  MOT17-04-SDP  MOT17-05-SDP  MOT17-09-SDP
    MOT17-10-SDP  MOT17-11-SDP  MOT17-13-SDP
    ```

    each executed as the §3.3 four-run block `00_capture_off`,
    `01/02/03_capture_on`, in that order, under the unmodified A5 policy target
    with `SACCADE_GPU_DECODE=1`, unlabelled, with no threshold sweep and no GT/FP
    read. 28 runs, one invocation, retry count zero for every step.

    Phase A's own runs are **not** reused for MOT17-04-SDP: reuse across chains is
    a resume, which §5.2 forbids.

**MOT17-09-SDP has two distinct roles and they may not be conflated.** The
bounded probe runs it in identity mode (`gpu_decode = 0`, single-threaded relink,
§3.2). Phase B also runs it as one of the seven measurement sequences under §3.3.
Neither run may be substituted for the other.

### C3.3 `S_B` — the authorization

A new owner exactly-once authorization, in the identical form as `S_A`,
introducing no new authorization vocabulary. `S_A` is spent at Phase-A launch and
its scope was Phase-A-only; it cannot be extended, re-read, or re-used. Retry,
resume, and repaired re-run under `S_B` are permanently forbidden, exactly as
§5.2 forbids them under `S_A`.

**`S_B` has exactly one consumption event, and it is not process launch.** `S_B`
is consumed by the durable write-and-flush of the `authorization_consumed` record
(C3.5.1 step 5), which happens after the C3.6 admission gate has passed and
before the measurement launches. That write *is* the consumption — not a note
about a consumption that occurs elsewhere — so the authority state has a single
definition and a single durable linearization point, and no observer can find the
authorization spent-but-unrecorded or recorded-but-unspent.

The Phase-B controller contains no downstream dispatch, import, subprocess, queue
submission, or continuation flag. It exits after Phase B. Owner acceptance is a
separate act with its own record.

### C3.4 Result mapping

Phase B emits the same six `ORDERED_PREDICATES` the partition already defines.
Each predicate's Phase-B value is the **disjunction of failure over the seven
sequences**, so the selected terminal never depends on execution order;
`select_terminal` is then applied unchanged, first-applicable, under
`phase="b"` with `phase_b_complete=True`.

| Phase-B observation | Terminal |
| --- | --- |
| a bound input — the Phase-A evidence root included — was written during the invocation | 1 `H2_INPUT_MUTATED_DURING_MEASUREMENT` |
| any sequence's A7.6 capture-off/on equality differs | 2 `H2_CAPTURE_PERTURBS_POLICY` |
| any sequence's packet, exposure, overflow, native-universe, conservation, cross-repeat canonical digest, or replay predicate fails | 3 `H2_PACKET_INVALID` |
| post-launch build / load / runner-nonzero / deadline / serialization / missing-artifact / unclassified failure | 4 `H2_MEASUREMENT_EXECUTION_INVALID` |
| every preceding false, all seven sequences complete, all 21 capture-on packets and all verifications pass | 5 `H2_FULL_COMMIT_CAPTURE_FAITHFUL` |

**Terminal 2 is reachable in Phase B, and this is load-bearing.** Every Phase-B
sequence runs `00_capture_off` and the A7.6 comparison is live on all seven: H0's
terminal-5 semantics, consumed unchanged, require the non-perturbation bars to
pass *for every sequence*. A perturbation first observable on MOT17-02 must be
able to select terminal 2; the alternative is a terminal 5 asserting
seven-sequence faithfulness on one sequence's non-perturbation evidence.

### C3.5 Re-attempt after a Phase-B terminal

```text
terminal 1 or 4
  → attempt-local. A fresh chain at the same published coordinate is
    admissible, with the prior evidence root bound into the successor F_B.

terminal 2 or 3
  → a property of the sealed F_B measurement surface — not of the attempt, and
    not of the coordinate alone. Retry against the same measurement surface is
    forbidden. A separately accepted successor (the capture-ABI-delta charter
    that terminal 3's §7 transition selects, or an owner decision on terminal
    2's route closure) becomes admissible only once it changes the bound
    measurement surface.
```

The **measurement surface** is a bound field of `F_B`:

```text
measurement_surface_digest = canonical digest of
  the five coordinate axes and the bounded probe
  the runtime-input manifest's full_digest — which is coordinate_digest
    together with build_artifacts.digest, hence the exact extension and
    TensorRT-plugin bytes (C3.2 item 4)
  capture ABI · packet verifier · A7.6 inventory
  controller · child/recorder · verifier · archive checker · observation emitter
  the run plan (C3.2 item 10) and the exposure declaration (item 9)
```

Deliberately excluded as attempt-local, which is what makes a terminal-1/4
re-attempt expressible at all: the Layer-P certificate, the bound Phase-A
evidence root, `phase_a_evidence`, `prior_attempts`, and `I40_B`.

The key is the surface rather than the coordinate because the controller, child,
verifier and archive checker are `plumbing_only` — correctly, since they move no
coordinate axis — yet they determine terminal 3. Keying the ban to the coordinate
would forbid the very ABI-delta route terminal 3 is defined to select.

**Why `full_digest` is named explicitly, and why the certificate's copy of it is
not enough.** The published `runtime_inputs` axis is the manifest's
`coordinate_digest`, and that digest excludes `build_artifacts` by construction —
`tests/contract/test_h2_runtime_inputs.py` pins exactly this: changing a build
artifact moves `full_digest` and leaves the coordinate still. The physical bytes
are otherwise bound only inside the Layer-P v2 certificate
(`run_h2_layer_p.py`'s `runtime_input_full_digest`), and the certificate is
excluded here as attempt-local. Without this line the surface would therefore
admit *different extension and plugin bytes under one identical
`measurement_surface_digest`* — while §4.2's own threat model says different
native bytes can hold the bounded probe equal. Terminals 2 and 3 are functions of
what actually executed; a permanence claim keyed on a digest that does not see
the executed binary would be a claim about the wrong object. So the surface
re-admits the one certificate field that names the executed bytes, and nothing
else from the certificate: not the head, not the changed-path verdict, not the
attempt's own pass record.

This does **not** move build artifacts into the published axis. The layering of
§4.1 is preserved exactly: the coordinate spans builds; `F`/certificate scope
pins the bytes; the measurement surface, being an `F_B` field, sits on the
`F` side of that line.

Two guards, so this is not a licence to iterate:

- a change to the measurement surface must be a **named, demonstrated defect
  repair** in H0 §6's own repair vocabulary — compilation, capacity sizing,
  serialization, or implementation bugs — with the prior terminal's evidence root
  and the named defect bound into the successor `F_B`;
- §8.1's no-refit rule applies verbatim: no bar, tolerance, proxy, threshold,
  estimator, or classifier may be adjusted to close a gap. A surface change that
  relaxes a comparison is not a repair, and a terminal 2 or 3 "fixed" by
  weakening what it checks is exactly the laundering this clause exists to
  prevent.

**A changed surface is necessary, never sufficient.** Because `full_digest` now
enters the surface, a bit-different rebuild of unchanged sources moves
`measurement_surface_digest` on its own. That does not make a successor
admissible: the two guards above are the operative bar, and a rebuild carrying no
named, demonstrated defect repair is not a repair. Re-running a terminal 2 or 3
against recompiled bytes in the hope of a different answer is a retry wearing a
new digest, and it is forbidden by the first guard, not permitted by the digest.

### C3.5.1 The consumed-attempt record — what `prior_attempts` binds

C3.1 requires every root named in `prior_attempts` to exist and verify, and C3.5
allows a terminal-4 re-attempt. Terminal 4 covers serialization failure, missing
or unreadable artifacts, and unclassified execution failure — so without this
clause an attempt could spend `S_B` and then fail *before* an archive existed,
leaving a re-attempt permanently unformable: authorization spent, no verifiable
predecessor to bind, no successor `F_B` possible. `prior_attempts` therefore binds
**consumed-attempt records**, not completed measurement archives.

```text
ordering, normative

  1. F_B is constructed and its F64 digest computed          (S_B unspent)
  2. the evidence root h2_measure_b_<I40_B>_<F64> is created and the freeze
     record written to it                                    (S_B unspent)
  3. the C3.6 admission gate is evaluated and its verdict written
  4. admission failed → the root is marked `inadmissible`, is NOT a
     consumed attempt, and is not bound by any successor's prior_attempts
     (Layer-P class, §5.1: a coordinate to retry against)
  5. admission passed → the controller writes and flushes the
     `authorization_consumed` record. THAT WRITE CONSUMES S_B (C3.3).
  6. the measurement launches
  7. every exit path — terminal, caught failure, or crash — leaves the root in
     exactly one of the three verify classes below
```

Step 5 is the consumption event itself, deliberately placed before the launch it
authorizes. A process that dies between step 5 and step 6 has therefore spent
`S_B` and produced an `unterminated` attempt; there is nothing to reconcile,
because no other event ever claimed to be the consumption. The alternative
ordering — consume at launch, record afterwards — makes the durable record lag
the act it records, which is exactly how a spent authorization goes missing.

This costs an attempt in one case, and the cost is accepted deliberately: a
crash after the flush and before the launch spends `S_B` on a measurement that
never ran. That is a real loss of one authorization, not a bookkeeping artifact,
and it is preferred to the alternative in which the same crash leaves the
authority state undecidable.

**The three verify classes.** `check_h2_measure_archives.py` classifies every
root in `prior_attempts` as exactly one, and "verifies" means the class's own
integrity condition — a terminal-4 attempt is never required to produce artifacts
its failure is defined by the absence of:

```text
complete    a full measurement archive: manifest, checksum inventory, packets,
            recorded observation and selected terminal. Verified in full.

envelope    a caught failure: freeze record, authorization_consumed record,
            the failure classification, and whatever artifacts survived, with a
            checksum inventory over exactly those. Verified as an envelope —
            completeness of the envelope, not of the measurement.

unterminated
            authorization_consumed present, no terminal recorded: the process
            did not reach an exit path it could write. It selects no terminal —
            no observation exists — the authorization is permanently spent, and
            for re-attempt admissibility it is treated as terminal 4.
```

**The kill-switch is closed.** An `unterminated` attempt is re-attemptable only
if the verifier, run over every artifact that did survive, finds no completed
sequence exhibiting a capture-off/on inequality or an invalid packet. If the
surviving evidence already shows one, the C3.5 ban on terminals 2 and 3 applies
to that surface, whether or not the attempt lived long enough to record it.
Otherwise, terminating a run at the first sign of perturbation would be a way to
convert a forbidden terminal into a re-attemptable one, which is the same
laundering §8.1 forbids in the refit direction.

The controller, the failure envelope writer and the archive checker are all
`plumbing_only` (C3.9), so this clause adds no pre-seal ruler edit; it is a
contract those Layer-M components must satisfy, and `check_h2_measure_archives.py`
is where it is mechanically enforced.

### C3.6 The admission gate is pre-terminal

Admission is an **independent gate evaluated before `S_B` is consumed**. Its
failure is an inadmissible launch: no terminal is selected, no authorization is
spent, and the outcome is Layer-P class (§5.1) — a coordinate to retry against,
not an epistemic result.

```text
admission (evaluated before the C3.5.1 step-5 write that consumes S_B)
  a. the bound Phase-A evidence root exists, verifies, and its manifest and
     checksum-inventory digests equal F_B
  b. its recorded observation selects no terminal (result = measurement_pass)
  c. the five axes and the bounded probe equal F_B, and F_B's copies equal F_A's
  d. the Layer-P v2 certificate for the Phase-B head equals F_B
  e. prior_attempts is complete and every named root exists and verifies in
     its C3.5.1 class

after that write (S_B consumed; the measurement then launches)
  terminal 1 is selected by bound-input mutation only: any write to a bound
  input, the Phase-A evidence root included, sets `bound_input_mutated`.
```

So in the Phase-B chain, terminal 1 carries exactly one meaning — a bound input
was written while the measurement ran — and every condition that is decidable
before launch is decided before launch, where it costs nothing. This is the
two-layer budget of §5 applied to its own success path.

**The asymmetry with Phase A is deliberate and is recorded, not smuggled.** §7's
terminal-1 condition also admits a launch-time probe or certificate mismatch, and
that text is untouched for Phase A: this correction is append-only and narrows
nothing above the pinned prefix. The Phase-B chain moves those two checks into
admission because Phase B has strictly more to check before launch (the Phase-A
result itself) and because the narrowing is monotone-safe — it can only *avoid*
spending an authorization, never admit a launch §7 would have refused. Aligning
Phase A would be a separate decision on a separate surface; this correction does
not make it.

### C3.7 What the partition file must change

"No new `ORDERED_PREDICATE`" holds: the Phase-A-evidence case is carried by the
admission gate of C3.6 and by the existing `bound_input_mutated` predicate. "No
partition-file change" does not hold, and is not claimed.
`scripts/tools/h2_terminal_partition.py` requires:

1. **terminal 1's condition metadata** — phase-scoped, per C3.6;
2. **terminal 5's condition metadata** — phase-aware. It currently reads "all
   three capture-on packets". Required form: `required_sequences` and
   `required_capture_on_packets` per phase — Phase A: 1 sequence, 3 capture-on
   packets; Phase B: 7 sequences, 21 capture-on packets and 7 capture-off runs —
   with `--explain --phase {a,b}` printing that phase's exact condition;
3. **an explicit `phase` argument** to `select_terminal`, with
   `phase_b_complete=True` admissible only under `phase="b"`; the inconsistent
   combination raises rather than defaulting, per the module's existing
   fail-closed intake;
4. **tests** — the existing contract tests updated, and extended to cover
   phase-awareness, the admission gate, and the rule that an admission failure
   yields no terminal.

### C3.8 `phase_a_evidence` binds in `F_B`, never in the published axis

`phase_a_evidence` enters `F_B`'s complete manifest and the `BoundInputMonitor`
watch set. It does **not** enter the published `runtime_inputs` axis.

What is frozen before the seal is the **schema and its producer**; the concrete
evidence values can only bind after Phase A passes, which is after the Phase-A
seal by construction. Admitting them to the published axis would make the axis
undefined until Phase A had already run, and would move the axis at exactly the
moment C3.9 requires it to be still.

Mechanically, and stated as the exact split the next implementer must not get
backwards — the published axis and the freeze are two different digests over two
different member sets:

```text
published runtime_inputs axis  = manifest coordinate_digest
    seven sequences + configured runtime assets + third-party runtime
    build artifacts are NOT members (h2_runtime_inputs.py builds
    coordinate_digest over exactly four sections, excluding build_artifacts)

F_B / full manifest            = manifest full_digest, plus F_B-only sections
    the published members + the exact build artifacts (extension, TensorRT
    plugin)                                            → full_digest
    + phase_a_evidence                                 → F_B field, in neither
                                                         digest
```

This is §4's own row for `runtime_inputs` and §4.1's bound-input/witness split,
unchanged; the earlier draft of this clause listed build artifacts as published
axis members, which was wrong in the direction that matters — it would have led
an implementer to publish the coordinate that binds them, or to key C3.5's
surface on the one that does not.

The `phase_a_evidence` section is an `F_B`-only manifest section, absent when no
Phase-A evidence is supplied, and a contract test pins that adding it moves
**neither** the published axis **nor** `full_digest` — the latter because the
Layer-P v2 certificate records `runtime_input_full_digest`, so a section that
moved it would invalidate certificates for reasons that have nothing to do with
what was built or run.

### C3.9 The pre-seal edit list

§C3.1(b) requires the five axes to be equal across both phases, so
`identity_semantics` must be **frozen from the Phase-A seal to the Phase-B
seal**. Every file the clauses above touch is a member of that ruler, so all of
this must land, be republished, and be re-attested on the controlled host
**before Phase A seals**:

1. this declaration's `.policy.yaml` — the `sealed_prefix` re-pin carrying this
   correction;
2. `scripts/tools/h2_runtime_inputs.py` — the seven sequences, and the
   `phase_a_evidence` schema/producer of C3.8, which must leave both
   `coordinate_digest` and `full_digest` computed over their current member sets;
3. `scripts/tools/h2_terminal_partition.py` — the four changes of C3.7;
4. `docs/reference/runtime_identity.generated.json` republished, with
   `.github/workflows/runtime_identity.yml` green at that head.

The only Phase-B work that may land *after* the Phase-A seal is the Phase-B
controller, child/recorder, verifier, archive checker and observation emitter:
they are `plumbing_only` and move no axis, while still being bound into `F_B` and
therefore into the C3.5 measurement surface.

**One trap is pinned here because nothing else would catch it.** A *new* file
named `scripts/tools/h2_*.py` classifies as `plumbing_only`: only the exact paths
in `IDENTITY_SEMANTICS_PATHS` are the ruler. If any admission or phase logic is
placed in a new `h2_` module rather than inside `h2_terminal_partition.py`, that
module must be added to `IDENTITY_SEMANTICS_PATHS` in the same change — itself a
ruler edit, hence pre-seal. Splitting ruler logic into a file that silently reads
as plumbing would let the ruler move inside the frozen window with no check
firing, which is the §5.3 circular-oracle hazard wearing a different hat.

### What this correction does not do

It does not seal, authorize, or schedule anything; issues no `S_B` and no `S_A`;
selects no `I`; creates no `F`; runs no sequence; writes no registry state and no
`captured_under` value; changes no preset, kernel constant, or capture ABI; does
not alter A7.6 or the packet verifier; does not alter §§0–8 or Corrections 1–2
above the pinned prefix; and does not unblock `H0_ROUTE5_B1`, `GCTM_B1`, or O1.
H0's closed history, its five spent `S` chains, and the permanent ledger entry on
`quantity.bridge_capture_provenance` remain unchanged. §5.2 precondition 6 is now
satisfiable in principle; the remaining gates of the charter's `Acceptance`
section are untouched.

---

## Review Correction 4 — which environment state carries authorization authority (2026-07-28, pre-seal)

This correction records an owner ruling forced by the second spent Phase-A
invocation. It changes no authority state, grants no execution, selects no `I`,
creates no `F`/`S`, issues no exactly-once authorization, runs no sequence, and
writes no registry or sidecar state. It adds no member to any comparison, moves
no axis, defines no terminal, and edits neither `EXPECTED_ENV_KEYS` nor
`STATIC_ENV` — those are H0's frozen ruler, imported here and unchanged (§6). It
names **which environment state an existing predicate is a predicate about**.

### The ruling

> The authorized environment is the immutable launch snapshot the controller
> constructs and the child captures before any third-party import. Every
> evaluation of the ingress predicate takes that snapshot as its input. The
> environment change caused by importing `cv2` is derived state of the child's
> execution: it may be observed and recorded, but it may not retroactively
> rewrite or negate a launch ingress authorization that has already passed.

### Why a correction rather than a repair note

§3.3 requires the measurement fixture to keep H0's environment exactly, and the
child enforces that as a predicate over the process environment. Nothing in
§§0–8 said *which* environment state that predicate ranges over, so re-deriving
it from live `os.environ` at any later point was a legal reading — and that is
what shipped. It cannot hold once the imported stack mutates the very object
under test.

The missing contract is not the key set. It is the authoritative observation
point: without one, a future implementation may again recompute the ingress
predicate from a live process environment after a third-party import and
reproduce this defect in a new place. Leaving that unstated in a `plumbing_only`
file is the C3.9 hazard in its usual direction — a rule that decides whether a
run is admissible would live where it can be edited without any axis moving.

### What follows, normatively

The subject of this correction is **which environment state carries authorization
authority**, not how many times a check runs or where a call sits. Those are the
repair's business; this is the contract it must satisfy.

1. **The authorized environment is an immutable launch snapshot.** It is the
   environment the controller constructs and the child captures **before** any
   third-party import, and it is a value, not a live view. `os.environ` after
   that instant is a different object with a different history.
2. **Every ingress-predicate evaluation takes that snapshot as its input.** The
   key set, the static values and the invocation's environment digest are
   properties *of the snapshot*. Re-verification is permitted and may be
   defensive — what is forbidden is re-deriving the predicate from live
   `os.environ`, because that substitutes an observation the authorization was
   never issued against.
3. **A passed ingress authorization cannot be invalidated in reverse.** Once the
   snapshot satisfies the predicate, environment change produced afterwards by
   third-party imports does not — **through the ingress predicate or the
   invocation environment digest** — unmake that decision, select an
   ingress-failure terminal, or render the launch unauthorized. The scope is
   deliberate: this closes the reverse path through the ingress predicate, and
   nothing more. A separate contract of the kind clause 4 permits keeps its own
   terminal-selecting authority over its own subject matter.
4. **The post-import delta is observable, not an ingress input.** It may be
   recorded as diagnostic, examined, or governed by a *separate and explicitly
   stated* contract with its own named baseline — for example a gate on whether
   the repository's own `configure_runtime_env` mutates anything. Such a contract
   may not borrow the ingress predicate or the invocation digest to express
   itself, and its baseline must be stated rather than inherited by proximity.
5. **The environment is not restored after import.** Undoing a dependency's
   `LD_LIBRARY_PATH` change would alter what the run loads, which is a change to
   the measured object, not a fix to a check.

What this leaves open is deliberate. An implementation may validate once at
ingress and never again, or revalidate the snapshot as often as it likes; it may
gate `configure_runtime_env` against whichever explicitly named baseline it can
defend. What it may not do is let the live process environment become the thing
the authorization is judged against, which is the single defect both the H2 child
at `:298` and the frozen H0 child at `:372` share.

### What this correction does not do

It does not seal, authorize, or schedule anything; issues no `S_A` and no `S_B`;
does not revive either spent authorization; does not alter A7.6, the packet
verifier, the terminal partition, or any of §§0–8 and Corrections 1–3; does not
edit the frozen H0 child, which carries the same latent post-import
re-application and stays byte-frozen; and does not unblock `H0_ROUTE5_B1`,
`GCTM_B1`, or O1. The two spent Phase-A invocations remain spent and produced no
capture. Evidence:
[h2_phase_a_failed_attempt_7646f421_20260728](evidence/h2_phase_a_failed_attempt_7646f421_20260728/).

---

## Review Correction 5 — execution integrity is the requirement (2026-07-30, pre-seal)

This correction narrows the requirement for the H2 successor unit. H2 requires
**execution integrity**, not environment reproducibility:

> A qualifying H2 record must prove that this execution used the resolved
> configuration it names, that no other configuration source silently changed
> that namespace, that the code, input and native-binary bytes actually used are
> the bytes recorded for this execution, that the bound inputs did not change
> during it, and that the result and verification belong to this execution.

It is not a claim that a foreign host can reproduce the same build bytes,
runtime environment or bounded observation. Source commit and tree identity
remain audit metadata; neither is a validity gate for the new record.

### The sole configuration authority

The new `run_spec.json` is the complete canonical resolved evaluator namespace
and the sole authority over execution configuration. The preset, parser
defaults, process environment and child argv are transports or projections,
never independent value sources:

1. the RunSpec resolver produces the full namespace, not a selected subset;
2. argv and the repository-owned environment projection are derived from that
   resolved object;
3. after parsing, every namespace member must equal the resolved RunSpec;
4. the runtime is checked immediately before and after execution so neither
   import-time nor execution-time mutation can silently change the declared
   configuration.

The H0 producers remain byte-frozen. Their constants may be consulted once as a
documented derivation source while the RunSpec is authored, but the new H2 path
must not import a live configuration authority from H0.

The existing adapter review truthfully recorded four accepted differences for
the then-current child. Promoting values into a complete, sole-authority RunSpec
is a new decision surface: `detector`, `max_frames`, `preset`, and
`warmup_frames` therefore require explicit owner adjudication when the RunSpec
is authored. This correction chooses none of them and does not rewrite the
historical adapter ruling.

### Content identity and the independent verifier

Diagnostic and measurement executions bridge on both
`resolved_run_spec_digest` and `execution_semantics_projection_digest`. The
execution-semantics projection is digest equality over a declared content set:
executed-surface source bytes, the capture ABI schema,
`scripts/eval/mot17_args.py`, and the RunSpec schema and resolver bytes. It does
not depend on path classification. The schema and resolver belong in the set so
unchanged RunSpec bytes cannot silently acquire new meaning.

The producer emits exactly three artifacts:

- `run_spec.json` — the complete resolved namespace and its declared execution
  semantics projection;
- `runtime_binding.json` — the code, input and build bytes actually used;
- `result.json` — this execution's result.

It must not write `verification.json`. A separate command in a separate process
reads only the emitted artifacts, archive bytes and checksum closure, fails
closed on every missing required member or field, and writes
`verification.json`. It may run on a foreign host, but its verdict must not
depend on that host's machine identity, UID, checkout, build or runtime
environment, and it may not call the producer to fill in or re-derive missing
evidence.

New archives contain those four JSON files plus `checksums.sha256`. The two
historical H2 archives keep their original schema, bytes and verifier path at
full validity. They are not migrated, rewritten, renamed or downgraded to
inventory-only.

### What is retained and what is retired

Layer P retains its six ordered stages unchanged:
`retry_admissibility`, `preflight`, `build`, `build_binding`,
`extension_load`, and `identity_run`. Preflight remains fail-closed, the
identity run still proves that the extension loads and executes, and the
runtime binding still records the build bytes actually consumed.

For the successor artifact path this correction retires:

- `layer_p_certificate` as an independent admission artifact;
- `freeze.json` and `F`;
- equality with the published coordinate or bounded probe as a measurement
  gate;
- `source_head` and `source_tree` as validity gates;
- the rule that an unrelated commit requires a controlled-host
  re-attestation, Layer-P certificate and freeze rebuild.

The identity-run probe remains a recorded observation, not an equivalence
oracle. Probe equality still establishes no behavior preservation and no
equivalence; `equivalence.state` remains `unproven`.

The retirement has a deliberate cost. The controlled-host and local Layer-P
runs were not duplicate probes: their agreement across distinct hosts and
builds made the published observation non-circular and exposed a verifier
host-dependence defect that local execution could not reveal. The narrowed
requirement no longer pays that reproducibility cost. **Reproducibility is
retired; independent verifiability is not.**

### Diagnostic and measurement authority

A diagnostic mode may run repeatedly without owner authorization, records every
independent failed predicate, and is always
`authority: non_qualifying_diagnostic`. A green diagnostic cannot complete,
qualify or authorize a measurement witness.

Measurement mode retains the real fail-fast control flow and consumes one
separately issued exactly-once authorization. Nothing in this correction, a
RunSpec, either digest, any diagnostic, or any verification record issues or
restores such authority.

### What this correction does not do

It implements no schema, resolver, producer, verifier, controller or diagnostic
mode; builds no artifact; executes no sequence; selects no `I`; creates no
`F`/`S`; issues no third authorization; seals nothing; admits nothing to the
canonical corpus; and makes no equivalence claim. The two prior owner
authorizations remain permanently spent and produced zero faithful capture.
The Item 4 and Item 5 records at `1a742765` remain truthful historical evidence;
this correction does not extend them to a descendant head or convert them into
execution authority. H0's producers, declarations, sealed history and five
spent chains remain untouched.

## Review Correction 6 — frozen RunSpec authoring profile (2026-07-30, pre-seal)

The owner adopts
`docs/research/contracts/h2_phase_a_authoring_profile_v1.json` as the
versioned, byte-frozen authoring authority for the Phase-A RunSpec. It is a
complete 454-key resolved namespace, not a partial overlay and not a runtime
preset. Its file SHA-256 is
`cadcfeb56c3ecfef1f208f901bed92f931438158b8d8c3c05779c7db7bccacfe`.
The separate append-only decision record is
`docs/research/contracts/h2_phase_a_run_spec_authoring_decision_v1.json`.

The four values left open by Correction 5 are adjudicated as:

- `detector = null`;
- `max_frames = null`;
- `preset = null`;
- `warmup_frames = 50`.

`preset = null` states the runtime truth: no live preset loader participates in
execution. The profile records
`configs/presets/mamba_whole_graph_m.yaml` and its source digest only as
authoring lineage. The ordinary preset, parser defaults and the owner decision
were resolved once to produce the reviewed complete profile; they are not
consulted again to fill or overwrite any RunSpec namespace member.

The resolver must validate the profile schema, exact 454-key inventory,
namespace digest, profile-file digest in the owner decision and all four
adjudications before it may issue a RunSpec. It must not contain Python
constants that originate those values. At runtime, the evaluator namespace,
argv and repository-owned environment projection come only from the RunSpec in
the invocation. Reading the frozen profile bytes as a member of the declared
content projection is an identity check, not a configuration-source fallback.

The interpretation closure now includes the frozen profile, its schema, the
owner decision, the RunSpec schema and resolver, `scripts/eval/mot17_args.py`,
the capture ABI and the executed surfaces. Any byte change in that declared set
moves `execution_semantics_projection_digest`.

This correction does not claim that pre/post equality detects a transient
mid-execution mutation, does not yet bind the complete evaluator/pipeline/
tracker execution-code closure across diagnostic and measurement records, and
does not add cross-verdict constraints among `valid`, predicates and terminals.
Those remain implementation obligations for the successor producer, verifier
and controller commits. This correction issues no authorization, executes no
sequence, rebuilds no native artifact, creates no `F`/`S`, seals nothing,
admits nothing to the corpus and makes no equivalence claim.

## Review Correction 7 — RunSpec canonical byte domains (2026-07-30, pre-seal)

The Phase-A RunSpec uses two distinct canonical byte domains. Object digests,
including `resolved_run_spec_digest`, are SHA-256 over finite compact JSON with
lexicographically sorted keys and **no trailing LF**. The serialized
`run_spec.json` artifact is those complete object bytes followed by **exactly
one trailing LF**.

The schema and resolver must therefore name both rules independently:

- `object_canonicalization =
  utf8_lexicographic_keys_compact_finite_no_trailing_lf_v1`;
- `artifact_serialization =
  utf8_lexicographic_keys_compact_finite_single_trailing_lf_v1`.

The former ambiguous `canonicalization =
utf8_lexicographic_keys_compact_finite_trailing_lf_v1` identifier is not a
permitted alias. An archive-only verifier hashes the no-LF object bytes for the
object digest and verifies the single-LF form only as artifact serialization.

This clarification changes no frozen authoring-profile bytes, values, lineage,
authority or interpretation closure. It issues no authorization, executes no
sequence, rebuilds no native artifact, creates no `F`/`S`, seals nothing,
admits nothing to the corpus and makes no equivalence claim.

## Review Correction 8 — controlled-host reconstruction is diagnostic only (2026-07-30, pre-seal)

Correction 5 retired environment reproducibility and equality with a published
coordinate or bounded probe as successor measurement gates. The remaining
automatic pull-request and `main` triggers for
`.github/workflows/runtime_identity.yml` would nevertheless keep paying that
retired cost: bind a controlled host's complete runtime inputs, rebuild native
artifacts and reproduce the bounded observation for every matching source or
ruler edit.

The workflow is therefore **manual diagnostic only**:

- `workflow_dispatch` is its sole trigger;
- it is not a required pull-request, merge, seal, admission or measurement gate;
- its output may report controlled-host build, input or bounded-probe drift, but
  cannot invalidate or qualify a successor execution record;
- probe equality remains only a recorded observation and cannot establish
  behavior preservation or equivalence.

This does not remove execution-time integrity. The successor producer still
retains Layer P's ordered `build`, `build_binding`, `extension_load` and
`identity_run` stages. `runtime_binding.json` records the code, input and native
bytes actually consumed by that execution, and the independent verifier judges
the resulting archive without rebuilding it or consulting the verification
host. Foreign-host-independent **verification** remains required; foreign-host
reconstruction does not.

The checked-in runtime-coordinate publication, all pre-correction workflow run
records, historical Item 4 records and both historical H2 archive verifier paths
remain truthful history. None becomes successor authority. The legacy
publication's static axes are republished once for this ruler edit from the
existing read-only probe and runtime-input records; no controlled-host or native
rebuild is performed. This correction does not dispatch the manual workflow, run
any sequence, issue or consume authorization, create `F`/`S`, seal or admit an
archive, or change `equivalence.state` from `unproven`.

---

## Review Correction 9 — the successor verdict algebra (2026-07-30, pre-seal)

Corrections 5 to 7 froze the successor artifact contract; Correction 8 removed
the reconstruction gate. What none of them fixed is that the successor artifacts
and the executable partition of § 7 now describe the same partition in two
vocabularies, and no rule joined them:

- three predicates are renamed, and one of the renames **inverts polarity**:
  `bound_input_unchanged` is true when the world is intact, while
  `bound_input_mutated` is true when it is broken;
- `runtime_binding_matches_spec` replaces `layer_p_certificate_matches_freeze`
  after Correction 5 retired the certificate, so the result token
  `certificate_mismatch` is **superseded, not deleted** — the historical archives
  that recorded it keep their meaning, and both tokens select terminal 1;
- a predicate is no longer a bool. `pass` / `fail` / `error` / `not_run` needs a
  rule the two-valued partition never had;
- `h2_execution_result_v1` constrained only its diagnostic branch, so under
  `exactly_once_measurement` a failure result with `terminal: null` validated —
  the successor spelling of the state § C3.5.1 exists to make unformable — and
  `h2_execution_verification_v1` bound `valid` to nothing at all, so `valid: true`
  validated with every check false.

### The rules

**One partition, two spellings.** The successor predicate order is the § 7 order;
the rename map, the inverted predicate and the superseded result token are
published by `h2_terminal_partition.as_payload()` so a payload-only implementer
resolves the same verdict as one calling the function (§ 20.8). A record mixing
the two spellings is refused rather than half-read.

**A decided failure outranks an undecided predicate, wherever it sits.** Not
first-applicable over raw states: an `error` on an early predicate must never
wash a later capture-perturbation or invalid-packet *finding* into terminal 4,
or killing a process on sight would launder a § C3.5-banned terminal into a
re-attemptable one. Among decided failures the § 7 order decides, unchanged.

**An undecided predicate cannot coexist with a complete execution.** If nothing
failed but something was not decided, the execution did not complete and
`execution_complete` must say so. A record claiming both is internally
contradictory and is refused, not mapped.

**A diagnostic selects no terminal.** It records every failed predicate and
resolves to `diagnostic_complete` whatever they say; that token is unavailable to
a measurement, and no diagnostic qualifies or authorizes one (Correction 5).

**The two artifacts carry the joint constraints mechanically**, so any validator
enforces them without executing the ruler: each result pins exactly the terminal
the partition selects; `measurement_pass` requires measurement authority, a null
terminal and six passing predicates; a selected terminal requires a non-passing
observation; **a named finding requires its own predicate to be `fail`** — an
`error` or `not_run` is undecided and selects nothing by name, so admitting any
non-pass state there would let two different results share one terminal for the
same observation, which is § 20.8's failure and not its test; no earlier
predicate may have decided-failed; and `valid` is exactly the conjunction of the
verification checks, with reasons empty when it holds and non-empty when it does
not.

**The runtime binding is stage-aware, because a failure archive must be
formable.** `h2_runtime_binding_v1` required two complete build artifacts, a
successful extension load, a `computed` identity probe and a zero-change input
monitor unconditionally. Under that shape the tokens this correction re-admits
could never appear in a `valid: true` archive — a genuine `build_failed` has no
loaded extension to record — and neither could terminal 1: `changed_count` was
pinned to zero, so a detected mutation was unrecordable. A contract that makes
its own truthful negatives unformable does not narrow anything; it only moves the
failure to where nothing checks it.

The binding therefore declares `failed_stage`: null, or the retained stage that
failed. Evidence is required exactly where the execution reached it — a build
failure requires no artifacts, load or probe; a `build_binding` or
`extension_load` failure requires the complete build artifacts but no load or
probe; an `identity_run` failure requires both artifacts and load but no probe;
and a null `failed_stage` requires all three, exactly as strictly as before.
Absence, never a fabricated success shape, is how an unreached stage is recorded.
`build_binding` and `identity_run` have no dedicated result token and are carried
by `unclassified_execution_failure` with the stage named in the binding: the
cause is recorded without inventing a terminal boundary.

**The cross-artifact rules the schemas cannot express** are published by
`h2_terminal_partition.as_payload()` and are **verifier obligations**, since no
JSON Schema sees two files at once. They are not unconditional: stage evidence is
subordinate to the authority boundary and to the § 7 order, and a rule that
ignores either makes truthful records unformable.

- `build_failed` requires `failed_stage` to be `build`, and
  `extension_load_failed` requires `extension_load`. This direction is
  unconditional: a named cause that does not name a failed stage is a label, which
  is the defect the two re-admitted tokens exist to avoid.
- The reverse direction holds **only when terminal 4 is the ordered winner**. A
  build that failed while a bound input moved is a real observation: terminal 1
  outranks terminal 4, so the result is `input_mutated` while `failed_stage`
  remains `build` as subordinate evidence. Under terminals 1 to 3 a non-null
  `failed_stage` is recorded and changes no verdict; demanding its token there
  would let stage evidence overturn a higher-order finding, and combined with the
  mutation rule would leave that observation with **no admissible result at all**.
- `input_mutated` holds exactly when the monitor recorded a change or an unclean
  final drain, in both directions, because terminal 1 is the highest order and a
  recorded change cannot lose to anything. Relaxing the monitor is what makes
  terminal 1 recordable; this biconditional is what keeps it honest.
- **A diagnostic demands nothing.** It records every failed predicate and every
  stage failure and still resolves to `diagnostic_complete`, so requiring a
  mutation or a failed stage to change its result would contradict the authority
  boundary this correction states three paragraphs above.

The archive-only verifier of W3 must enforce all of it, and no producer exists in
between.

**`build_failed` and `extension_load_failed` are re-admitted as result tokens.**
Correction 5 *retains* the `build` and `extension_load` stages, so their failures
must remain nameable; folding them into `runner_nonzero` would have destroyed a
cause the predecessor partition already recorded. Both map to terminal 4, exactly
as in § 7, so no terminal boundary moves.

### What this correction does not do

This correction was revised in place twice under owner review before merge, which
is why the rules above read as they do rather than as appended passes: the
correction had not merged and nothing was sealed, so revising the text is more
honest than layering a correction onto an unsealed one, and the re-pin log records
each revision. Three defects were fixed. A named finding admitted any non-pass
state. The runtime binding made every stage failure unformable. And the
cross-artifact rules were stated unconditionally, which contradicted both the
authority boundary and the § 7 order — a build failure under a moved input had no
admissible result, and a diagnostic that observed either could not stay
`diagnostic_complete`. Each was found by a case the tests did not cover: the first
by substituting a result rather than a terminal, the second by asking whether a
truthful negative can be archived at all, the third by combining two conditions
that had only ever been varied one at a time.

It implements no producer, no verifier, no diagnostic mode and no controller
change; the archive-only verifier and the producer remain the next two staged
obligations. It builds no artifact, executes no sequence, selects no `I`, creates
no `F`/`S`, issues no third authorization, seals nothing, admits nothing to the
canonical corpus and leaves `equivalence.state` at `unproven`. The two spent
owner authorizations and both historical archives are untouched, and their
verifier path keeps the legacy vocabulary it recorded. This ruler edit republishes
the legacy publication's static axes from the existing read-only probe and
runtime-input records; no controlled-host or native rebuild is performed.
