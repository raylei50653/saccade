<!-- doc-status: active -->
<!-- doc-promotion: none; terminal-review checklist artifact for the GCTM task charter (WP-A8) -->
<!-- doc-date: 2026-07-23 -->
<!-- doc-module: semantic -->

# GCTM terminal review — checklist artifact and mechanical terminal selection (WP-A8)

**用途：** the checklist artifact required by the
[GCTM task charter](../threads/closed/gap_conditioned_stochastic_transition_model_task.md)
§"Terminal partition — frozen at activation (WP-A0)" → *Decision procedure
(frozen)*, step 2, plus the mechanical terminal selection that step 1/3 defines
over it.

**Authority 邊界：**

- Task/lifecycle owner: the charter (Issue
  [#175](https://github.com/raylei50653/saccade/issues/175)). This artifact
  **implements** the charter's frozen decision procedure; it does not amend it,
  and where any wording here differs from the charter, the charter controls.
- Specification authority: frozen
  [D1](gap_conditioned_stochastic_transition_spec_v1.md) §2–§8 and
  [D2](gap_conditioned_stochastic_transition_lemmas_v1.md). Both are left
  **byte-unchanged** by this review except for D1's append-only §1 status
  correction and History (D1 §2–§8 stay byte-frozen; D2 is untouched).
- This artifact **measures no data**, fits nothing, establishes no
  runtime-fidelity edge, and grants no authority. Selecting a terminal is not
  accepting one: **owner acceptance** (decision procedure step 4) is what closes
  the task. Acceptance was recorded 2026-07-23 in the now-closed charter
  (*Final status*) and in the [claim-state registry](../contracts/claim_state_registry.md);
  the body below is kept as the review record as written at selection time.
- Lifecycle labels of D1, D2 and this artifact are **unchanged** by the closure,
  and that is a **recorded governance gap, not a C6-sanctioned exception**: C6's
  vocabulary is `proposed | active | parked | closed | archived`, it has no state
  for an *accepted canonical artifact*, and marking these files `closed` would
  force an L1 move into `closed/` that breaks the citations the closed charter
  and the registry depend on. Whether to add a `sealed` / `canonical` state is a
  separate contract decision, not something this review takes. **Do not read task
  lifecycle off these file markers** — it is owned by the closed charter and the
  [claim-state registry](../contracts/claim_state_registry.md).
  *(Wording corrected post-merge per #260 review; the review's rows, predicate
  evaluation and selected terminal are unaffected.)*

## §0 What this artifact is / is not (typed boundary)

**Does:**

1. fixes the checklist scope exactly as the charter's decision-procedure step 2
   enumerates it (§2 below), and fills one row per enumerated obligation with
   exactly one of `complete` / `incomplete` / `rejection-established`
   (+ mandatory `rejection_terminal` ∈ {1,2,3} on rejection rows only);
2. evaluates the ordered terminal predicates 1→5 and records the **first**
   applicable one (§5–§6);
3. states the judgement points where a reviewer could legitimately flip a row,
   and what the selection would become if flipped (§7);
4. records the limits the selected terminal carries and the authority it does
   **not** grant (§8–§9).

**Does not:** re-open or edit any frozen D1/D2 content; resolve, re-scope or
re-number any obligation; select a metric, threshold, parameter or data set;
activate B1/O1; assert runtime observability; change H0 state; or perform the
owner acceptance itself.

## §1 Governing procedure (quoted from the frozen charter)

The ordered partition and the selection rule below are **quoted** from the
charter (frozen at WP-A0); the charter is the authority — this section only
reproduces what the review is decided against.

| Order | Terminal | Predicate coverage |
|:--|:--|:--|
| 1 | `GCTM_OBSERVATION_INTERFACE_UNDEFINED` | state, observation, time, coordinate, covariance, or causal-availability interface remains undefined |
| 2 | `GCTM_TRANSITION_FAMILY_NOT_WELL_POSED` | the admitted transition family is dimensionally, probabilistically, or compositionally ill-posed |
| 3 | `GCTM_IDENTIFIABILITY_UNRESOLVED` | the intended claim cannot be identified under the declared observations or leakage boundary |
| 4 | `GCTM_SPECIFICATION_INCOMPLETE` | no earlier rejection applies, but required definitions, proofs, or decision items remain incomplete |
| 5 | `GCTM_MODEL_SPEC_SEALABLE` | every specification and proof obligation required for a diagnostic-only seal is satisfied |

Selection rule (charter, decision procedure steps 1–3): predicates are evaluated
**in the order above** and the first applicable predicate is the terminal, with
no later terminal substituted for an earlier applicable one; each predicate is
decided against this checklist artifact; and the mechanical rule is quoted
verbatim — *"if any `rejection-established` row exists, the terminal is the
**smallest** `rejection_terminal` value among them (first match in terminal
order); otherwise any `incomplete` row selects terminal 4; otherwise terminal
5."*

## §2 Checklist scope (which rows score)

Charter step 2 enumerates the checklist rows as *"every obligation of this
contract (the four mandatory obligations, the D1/D2 deliverable items, and the
typed B/C/D boundary)"*. That enumeration is definitional, so the **scoring**
rows are exactly:

- the mandatory activation-contract obligations (charter §"Activation-contract
  and model-seal obligations"), taken under **both** of the charter's own
  numberings — the four section headings *and* the four rows of the
  obligation-status table. These two numberings are not identical: the section
  headings are (1) exhaustive specification-incomplete outcome, (2) calibration
  vs ranking, (3) independence/\(C\), (4) affine M2, while the obligation-status
  table's row 1 is the canonical observation/time interface (the
  pre-activation seal condition). Their **union** is five distinct requirements,
  and all five are scored below (rows O-a, O-1 … O-4) so the checklist is
  exhaustive under either reading;
- the D1 deliverable items (charter §"Future deliverables" → D1, four bullets);
- the D2 deliverable items (same section → D2, five bullets);
- the typed B/C/D boundary (charter §"Bindings frozen at activation").

Items that are **not** scoring rows are listed in §4 with the reason; §4 exists
so that "not a row" is a recorded judgement rather than an omission.

## §3 Checklist (scoring rows)

Status vocabulary is the charter's: exactly one of `complete` / `incomplete` /
`rejection-established`; `rejection_terminal` is filled **only** on
`rejection-established` rows and is empty everywhere else.

| # | Obligation / deliverable item (charter wording) | Owning section or artifact | Status | `rejection_terminal` | Basis |
|:--|:--|:--|:--|:--|:--|
| **O-a** | Exhaustive, mechanically decidable specification-incomplete outcome in the terminal partition | Charter §"Terminal partition — frozen at activation (WP-A0)" | `complete` | — | Terminal 4 `GCTM_SPECIFICATION_INCOMPLETE` exists in the ordered partition; decision-procedure step 3 states exhaustiveness by construction (terminals 4/5 cover every state not captured by rejections 1–3); step 2 makes the decision mechanical over this artifact |
| **O-1** | Canonical observation/time interface (nine named fields) | [D1 §2](gap_conditioned_stochastic_transition_spec_v1.md) (+ §3) | `complete` | — | All nine fields frozen with binding classes; \(\Delta_{\mathrm{on}}=g_{\mathrm{phys}}+(\mathrm{bridge\_at}-1)\) fixed as an identity, never a default equality; operator-layer offset \(\pm(\mathrm{bridge\_at}-1)v\) kept separate from canonical drift (rows 5/8/9). Resolved at WP-A1 (active-declaration seal) |
| **O-2** | Calibration-only gain vs candidate-local ranking gain as distinct claims (different nulls, metrics, terminal consequences) | [D1 §6](gap_conditioned_stochastic_transition_spec_v1.md) + [D2 §7 (L5)](gap_conditioned_stochastic_transition_lemmas_v1.md) | `complete` | — | CAL and RANK frozen as two claims each with null / metric family / evaluation unit / consequence (§6.2), with the separation frozen in both directions (§6.3) and the shared-\(S_\Delta\) \(q\)/NLL ordering equivalence proved (L5, L5.1, L5.2). Resolved at WP-A5 |
| **O-3** | Independence declaration or explicit cross-covariance \(C\) | [D1 §5.4](gap_conditioned_stochastic_transition_spec_v1.md) (+ §5.5–§5.6) | `complete` | — | Exactly-one decision recorded: canonical \(e^-\perp\epsilon_1\Rightarrow S_\Delta=HP^-_\Delta H^\top+R_1\); dependent-error case frozen as a declared deviation with the charter's expanded form and the joint-PSD domain for \(C\). Resolved at WP-A4 |
| **O-4** | Canonical-state affine M2 transition as one complete interface (state, \(A_\Delta\), \(d_\Delta(c)\), \(Q_\Delta\), domains, units) | [D1 §4](gap_conditioned_stochastic_transition_spec_v1.md) | `complete` | — | \(K_\Delta(z_0,c)=\mathcal N(A_\Delta z_0+d_\Delta(c),Q_\Delta)\) at \(\Delta=g_{\mathrm{phys}}\), with SDE, closed-form \(Q_\Delta\), \(\gamma=0\) extension, domains, causal assumptions and units in one section. Resolved at WP-A2 |
| **D1-a** | equations and admitted parameter domains | [D1 §4](gap_conditioned_stochastic_transition_spec_v1.md) (§4.7) + §5.6 + §6.5 + §7.8 | `complete` | — | Every symbol carried by the frozen equations has a declared domain, unit and causal-availability entry; the innovation-layer and score-layer objects carry their own domain tables |
| **D1-b** | position-only and joint observation interfaces | [D1 §3](gap_conditioned_stochastic_transition_spec_v1.md) (+ §7.2) | `complete` | — | \(H_x\) and \(H_{xv}\) frozen as **different claim objects**, never interchangeable; production-corresponding instantiation bound to \(H_x\); \(H_{xv}\) admitted only under a declared causal-availability statement |
| **D1-c** | identifiability and leakage matrix | [D1 §7](gap_conditioned_stochastic_transition_spec_v1.md) (§7.6 matrix, §7.7 verdict) | `complete` | — | Target set, data-design regimes, single-event confounding, multi-gap quotient identifiability, the two structural gauges, the \(\gamma\)-unknown regime, CAL/RANK mutual non-identifiability, and the frozen leakage matrix are all specified. Verdict is **conditional and specified, not established** — see §5 (terminal-3 evaluation) and §7 (judgement point J-3) for why this is `complete` and not `rejection-established` |
| **D1-d** | schema-only interface for a separately declared future B1 input | [D1 §8](gap_conditioned_stochastic_transition_spec_v1.md) | `complete` | — | Fail-closed consumption rule, four field blocks with derived quantities computed-never-supplied (X-a computable / X-b latent), predicates W1–W9 + W5′, and the claim-restriction map translating §7.6 blocking conditions into per-atom mechanical verdicts. Resolved at WP-A7 |
| **D2-a** | \(M2\rightarrow M1\) limit | [D2 §3 (L2)](gap_conditioned_stochastic_transition_lemmas_v1.md) | `complete` | — | Nesting proved for **mean and covariance**, plus continuity of \(\gamma\mapsto(A_\Delta,d_\Delta,Q_\Delta)\) on \([0,\infty)\), upgrading D1 §4.6 from interface closure to lemma |
| **D2-b** | positive-semidefinite covariance argument | [D2 §2 (L1)](gap_conditioned_stochastic_transition_lemmas_v1.md) | `complete` | — | \(Q_\Delta\succeq0\ \forall\Delta\ge0\) with the degenerate-rank characterisation (\(Q_\Delta\) singular \(\iff D\) singular; \(\operatorname{rank}Q_\Delta=2\operatorname{rank}D\) for \(\Delta>0\)); D1 §5.5 carries the \(S_\Delta\) consequence at interface level |
| **D2-c** | \(q\)/NLL ranking equivalence under shared covariance | [D2 §7 (L5)](gap_conditioned_stochastic_transition_lemmas_v1.md) | `complete` | — | Shared-\(S\) ⇒ \(E=\tfrac12q+\kappa\) with \(\kappa\) candidate-independent ⇒ same ordering (L5); isotropic corollary (L5.1); tightness counterexample for candidate-specific covariance (L5.2) |
| **D2-d** | dimensional consistency | [D2 §6](gap_conditioned_stochastic_transition_lemmas_v1.md) → frozen [D1 §4.7](gap_conditioned_stochastic_transition_spec_v1.md) (+ §5.6) | `complete` | — | Satisfied **by cross-reference, not by a separate lemma**: D1 §4.7/§5.6 state the unit assignments; D2 §6 records that L1–L4 preserve them. Judgement point J-1 (§7) |
| **D2-e** | short- and long-gap asymptotics | [D2 §5 (L4)](gap_conditioned_stochastic_transition_lemmas_v1.md) | `complete` | — | Short-gap universal M1 limit; long-gap OU-saturated velocity covariance \(D/(2\gamma)\) and linear-in-\(\Delta\) position covariance vs M1 cubic; crossover \(\Delta\sim1/\gamma\), stated as diagnostic |
| **B-1** | typed B/C/D boundary (B/C/D downstream constructible, reached only through typed boundaries, not part of the A-layer claim identity) | Charter §"Bindings frozen at activation" + [existing-online §9.1](../../modules/semantic/research/existing_online_object_analysis_for_gctm_alignment_20260718.md) (objects A/B/C/D + the in-force conditional scope qualification) | `complete` | — | The A-layer obligation is a **typed pointer**, not a construction: the charter freezes B/C/D as downstream constructible and outside the A-layer claim identity, and the owner-accepted scope qualification (landed via PR #249) states that the A-layer seal requires A + the observation/time interface + the typed boundary toward B/C/D, and **not** event competition or full correspondence. Judgement point J-2 (§7) |

**Row count:** 15 scoring rows — 5 obligations, 4 D1 deliverable items, 5 D2
deliverable items, 1 typed B/C/D boundary. `rejection-established` rows: **none**.
`incomplete` rows: **none**.

## §4 Recorded non-scoring items (considered, with reason)

These are **not** checklist rows under the charter's step-2 enumeration. They are
recorded so that their exclusion is a stated judgement, and none of them is
treated as satisfied by this review.

| Item | Why it is not a scoring row | Standing after this review |
|:--|:--|:--|
| Pre-activation boundary-normalization obligation 2 — **L2 score-insertion contract** | Not an A-layer seal obligation: the charter states it blocks any B1/O1 **score-ranking path**, and the scope qualification / plan memo place a frozen L2 contract outside the A-layer seal conditions (a parallel B1 blocker) | **Still absent** ([registry §7](../contracts/claim_state_registry.md) architecture gap). The selected terminal does not relax it; any B1/O1 score-ranking path stays blocked |
| Pre-activation boundary-normalization obligation 3 — **typed cross-layer mapping** | Its A-layer legs are discharged inside scored rows (D1 §2 binding classes `normative` / `documented-production` / `declared-target`; D1 §3 causal availability); its **score-policy** leg is owned by the B1 charter and the registry-owned L2 contract, i.e. outside A-layer scope under the in-force qualification | A-layer legs declared; score-policy leg **not** established and not claimed |
| **Terminal review** itself (listed as an open deferral in D1 §7.9/§8.8) | It is the act this artifact performs, not an obligation it scores; scoring itself would be circular | Performed here; superseded in D1 by the WP-A8 append-only §1 note |
| **Terminal closure** write-back (charter §"Terminal closure": write terminal + lifecycle back to the charter and Issue #175) | Sequenced **after** selection by the charter's own decision procedure (step 4, owner acceptance), so it cannot be a precondition of the selection | Landed as the acceptance-step obligations listed in §10 |
| D2 §4 **L3** (semigroup / Chapman–Kolmogorov covariance composition) | Proved, but not one of the charter's five D2 bullets — extra, not required | Recorded as supporting evidence for the terminal-2 evaluation (§5) |

## §5 Predicate evaluation in charter order

**Terminal 1 — `GCTM_OBSERVATION_INTERFACE_UNDEFINED`: not applicable.**
State (D1 §4.1), observation (§3), time/gap (§2 fields 2–7), coordinate
substrate (§2 field 1), covariance objects (§5.1/§5.6) and causal availability
(§3.3, §4.7, §5.6, §6.5) are each frozen with domains and units. Nothing in the
scored rows is `incomplete` on interface grounds. Note this is an interface
**definition** verdict only: no runtime value is captured and no fidelity edge
exists — the interface's production-facing fields remain declared-target/proxy
(D1 §2 binding classes), which is a limit (§8), not an undefined interface.

**Terminal 2 — `GCTM_TRANSITION_FAMILY_NOT_WELL_POSED`: not applicable.**
Dimensional: D1 §4.7/§5.6 + D2 §6. Probabilistic: \(Q_\Delta\succeq0\) with rank
characterisation (D2 L1) and the interface-level consequence
\(S_\Delta\succeq0\), with \(R_1\succ0\Rightarrow S_\Delta\succ0\) **under
canonical \(C=0\)** (D1 §5.5). Compositional: semigroup \(A_{s+t}=A_tA_s\),
affine drift composition, covariance composition and the Chapman–Kolmogorov
measure form, valid for degenerate \(Q\) (D2 L3); \(\gamma\to0\) nesting with
proven continuity (D2 L2). The two known degeneracies are **bounded and
declared**, not ill-posedness: \(\Delta=0\) gives \(Q_0=0\) (declared boundary,
D1 §4.3 / D2 L1) and the dependent-error path guarantees only
\(S_\Delta\succeq0\), so its invertibility needs an extra nondegeneracy
assumption (D1 §5.5, with the frozen minimal counterexample) — which is exactly
why D1 §8 predicate W4 refuses to let such a row produce \(q/E/\Pi\).

**Terminal 3 — `GCTM_IDENTIFIABILITY_UNRESOLVED`: not applicable.**
The predicate is *"the intended claim cannot be identified under the declared
observations or leakage boundary"*. D1 §7 does not establish that impossibility;
it specifies a boundary with a **non-empty identifiable side**: under \(H_x\)
multi-gap populations with shared parameters, \(\ge4\) distinct \(\Delta\) and
**known \(\gamma\)**, the quotient \(\{D,P_{vv},\operatorname{sym}(P_{xv}),
P_{xx}+R_1\}\) is generically identifiable (§7.4), and the A-layer intended
claims — the transition interface, \(S_\Delta\), and the CAL/RANK claim
**definitions** — do not require the components that lie outside it. Three
things must hold for this reading to be honest, and each does:

1. **Non-identifiable components are declared, not hidden**: the
   \(P_{xx}\leftrightarrow R_1\) gauge (structural, not broken by \(H_{xv}\)),
   \(\operatorname{asym}(P_{xv})\) (\(H_x\)-invisible; observable but not thereby
   identified under \(H_{xv}\)), and \(\gamma\) unknown (an **unmet regime**:
   \(>4\) gaps + a joint-map condition, explicitly **not proven** here).
2. **The rejection is per-instantiation and is exported fail-closed**, not
   deferred: §7.7 states that a claim is non-identifiable exactly when it must
   rely on an unidentifiable component or an unmet regime, and D1 §8 turns that
   into mechanical per-atom verdicts (W6 regimes; §8.7 restriction map, union
   rule, empty-intersection test), so any future B1 input that does depend on
   such a component is rejected **before** it may cite §6/§7.
3. **No such instantiation exists at A-layer**: this charter authorises no data,
   no fitting and no identification, and D1 declares none — so there is no
   claim in scope that must rely on an unidentifiable component. §7.7 assigns
   precisely this decision to WP-A8, and the decision is: **no
   `rejection-established` row**.

The verdict remains *identifiability **specified**, not established*; that is
recorded as a carried limit (§8), not as a rejection.

**Terminal 4 — `GCTM_SPECIFICATION_INCOMPLETE`: not applicable.** No scoring row
in §3 is `incomplete`.

**Terminal 5 — `GCTM_MODEL_SPEC_SEALABLE`: applicable** by the frozen selection
rule (no `rejection-established` row, no `incomplete` row).

## §6 Selected terminal

```text
selected_terminal: GCTM_MODEL_SPEC_SEALABLE      # ordered terminal 5
selection_basis:   checklist §3 — 15/15 scoring rows `complete`;
                   0 `rejection-established`; 0 `incomplete`
procedure:         charter "Decision procedure (frozen)" steps 1–3
status:            selected by terminal review; **owner-accepted 2026-07-23**
                   (step 4) — task closed by that acceptance
```

Selection is mechanical given §3. It is **not** an acceptance: the owner
acceptance that followed (2026-07-23) is what closed the task and triggered the
charter/Issue #175 write-back plus the closure obligations in §10. Had the owner
declined, this artifact would still hold — it records the selection, not the
acceptance.

## §7 Judgement points (where a reviewer may legitimately flip a row)

The three rows below are the only ones whose status rests on a reading rather
than on a directly checkable presence of frozen text. Each is stated with the
selection that would follow if the owner reads it the other way — flipping any
one of them changes the terminal, so they are the review's load-bearing points.

| id | Row | The reading taken here | If read the other way |
|:--|:--|:--|:--|
| **J-1** | D2-d dimensional consistency | The charter requires the D2 item to be **covered**; D2 §6 covers it by cross-reference to frozen D1 §4.7/§5.6 (no unit is left unassigned, and L1–L4 preserve them) rather than by a standalone lemma | Row becomes `incomplete` ⇒ terminal **4** (`GCTM_SPECIFICATION_INCOMPLETE`) |
| **J-2** | B-1 typed B/C/D boundary | The A-layer obligation is a **typed pointer** (charter binding + owner-accepted scope qualification), and the pointer exists; the constructions themselves are explicitly downstream | If a *constructive* boundary (event lift, dependence structure, score map, claim composition written into D1) is required, the row becomes `incomplete` ⇒ terminal **4** |
| **J-3** | D1-c identifiability | §7 specifies a boundary with a non-empty identifiable side and exports per-instantiation rejection fail-closed; no A-layer claim in scope depends on an unidentifiable component | If the A-layer intended claim is read as requiring identification of \(\gamma\) and of the full \(\{P_0,R_1\}\), the row becomes `rejection-established` with `rejection_terminal: 3` ⇒ terminal **3** (`GCTM_IDENTIFIABILITY_UNRESOLVED`), which outranks any later terminal |

No other row admits a defensible flip: each of the remaining twelve is satisfied
by frozen text whose presence is directly checkable.

## §8 Limits carried by the selected terminal

The seal is **diagnostic-only** (charter `output-class: diagnostic-only`), and it
carries every limit below unchanged. None of these is a defect of the
specification; each is a declared boundary that any future consumer inherits.

1. **No runtime substrate, no fidelity edge.** H0 state is unchanged by this
   review: faithful capture = none, actual H0 guarantee = none, candidate and
   guarantee sets empty. All production-facing fields in D1 §2 are
   `declared-target` / `documented-production`, never verified runtime
   quantities. Substrate does **not** inherit (registry §4.2): consumers of this
   state on the runtime substrate remain inadmissible without a separately
   accepted transfer.
2. **Identifiability specified, not established.** \(\gamma\)-identification
   under unknown \(\gamma\) needs \(>4\) distinct gaps **and** a joint-map
   condition (global injectivity ⇒ global; full-Jacobian-rank ⇒ local only —
   the two are not equivalent) that this specification does **not** prove; the
   \(P_{xx}\leftrightarrow R_1\) gauge and (under \(H_x\))
   \(\operatorname{asym}(P_{xv})\) are structurally non-identifiable.
3. **No measured gain of any kind.** CAL and RANK are frozen **definitions**
   with nulls and metric families; no calibration or ranking gain is claimed,
   measured, or implied, and the CAL distributional statements hold only under
   the explicitly declared CAL Gaussian working null (D1 §6.1).
4. **Canonical independence is a decision, not a derivation.** \(C=0\) is the
   declared canonical case; the dependent-error path is a frozen deviation whose
   invertibility requires an extra assumption (D1 §5.5).
5. **D1 §8 states requirements, not availability.** The B1 input schema says
   what an input would have to supply; it asserts no runtime observability,
   activates no B1, and creates no fidelity edge.
6. **Operator-layer offset stays separate.** \(\pm(\mathrm{bridge\_at}-1)v\) is
   an operator-layer deterministic offset (D1 §2 rows 8–9); it must never be
   recorded as M2 drift or as unaccounted bias.
7. **A-layer only.** B/C/D remain downstream constructible; event competition,
   ranking policy, decision probability and the L2 score-layer contract are
   outside this seal and remain unresolved elsewhere.

## §9 What the terminal does not grant

Acceptance of `GCTM_MODEL_SPEC_SEALABLE` grants **no** B1, O1, online,
mainline, or production authority (charter, explicit). It authorises no data
execution, parameter fitting, capture, GT/FP reveal, tracker or CUDA/runtime
change, H0 change, threshold or policy selection, or claim-state promotion of
any production object. B1 and O1 remain separately chartered and separately
gated; the registry-owned L2 score-layer contract remains absent and continues
to block any B1/O1 score-ranking path.

## §10 Post-selection obligations (charter step 4 — owner acceptance)

Recorded here for completeness; these are the acceptance step's obligations, not
checklist rows:

1. write the selected terminal and the task lifecycle back to the charter and to
   Issue #175 (charter §"Terminal closure"), creating no empty peer document;
2. perform the closure tidy in the **same** change as acceptance
   ([doc structure contract](../../ownership/doc_structure_contract.md) C6 rule
   3): charter frontmatter + `Final status`, move to `threads/closed/`, threads
   index row, module TODO WIP projection;
3. add the first accepted-state record for this object in the
   [claim-state registry](../contracts/claim_state_registry.md) (§2/§3 schema),
   carrying the §8 limits and the substrate boundary;
4. leave D1 §2–§8 byte-frozen: any status change is an append-only correction
   note in D1 §1.

## History

- 2026-07-23 — **WP-A8**: checklist artifact created; 15 scoring rows filled;
  ordered predicates 1–5 evaluated; terminal **`GCTM_MODEL_SPEC_SEALABLE`**
  selected mechanically (no `rejection-established`, no `incomplete` row), owner
  acceptance pending. Judgement points J-1/J-2/J-3 recorded with the terminal
  each flip would select. No frozen D1/D2 content edited; no data, fitting,
  runtime, online, B1/O1 or production authority granted; H0 state unchanged.
- 2026-07-23 — terminal **owner-accepted**; the GCTM charter closed and moved to
  `threads/closed/`, the semantic WIP lock was released, and the object's first
  accepted-state record was added to the claim-state registry. This artifact's
  rows and verdicts are unchanged by that acceptance.
