<!-- doc-status: draft -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-16 -->
<!-- doc-module: semantic -->

# GCTM downstream tasks — B1 / O1 objectives and semantics

> **Shared-semantics core · not active · not sealed · no execution authority**
>
> This document owns the **cross-task** evidence semantics shared by:
>
> - **B1** — runtime-grounded offline attribution and score-ranking evaluation;
> - **O1** — online score intervention and system-efficacy evaluation.
>
> On 2026-07-17 the §37 repository split was executed: the per-task material
> (identity, activation gates, scope, frozen inputs, evidence spaces, validity
> gates, terminal families, deliverables, handoff object, checklists) moved to
> two proposed task charters:
>
> - [B1 task charter](../../../research/threads/gctm_b1_runtime_grounded_offline_attribution_task.md)
> - [O1 task charter](../../../research/threads/gctm_o1_online_intervention_efficacy_task.md)
>
> This core keeps only the shared boundary (§0), the evidence chain (§1), the
> shared semantic rules (§2), the outcome interpretation matrix (§35), the
> forbidden inference shortcuts (§36), and the final semantic summary (§39).
> Original section numbers are preserved so existing §-references stay valid.
> It does not activate B1 or O1, select a model, freeze numerical thresholds,
> authorize data access, modify runtime behavior, or promote a production
> policy.

---

## 0. Alignment boundary (anti-drift; landed 2026-07-17)

This synthesis is landed for **semantic alignment only**: it fixes the intended
B1/O1 vocabulary so future charters do not drift. It grants no lifecycle state
and re-decides nothing owned elsewhere.

- **Upstream owners (unchanged).**
  [H0 declaration](headline_bridge_full_decision_capture_declaration_20260713.md)
  owns runtime observability / evidence fidelity;
  the [GCTM parked charter](../../../research/threads/gap_conditioned_stochastic_transition_model_task.md)
  owns the transition-family specification boundary. This document consumes
  both and re-decides neither.
- **Split executed (2026-07-17).** The two proposed task charters above now
  own the per-task boundaries; this core owns only the shared semantics. The
  charters are `doc-status: proposed`, `wip-role: non-wip`; the split grants
  no activation, no seal, and changes no registry or sole-active state.
- **B1 activation prerequisites are owned by the
  [B1 task charter](../../../research/threads/gctm_b1_runtime_grounded_offline_attribution_task.md)**
  — in particular the open **B1-slot identity** question (its "Unresolved
  B1-slot identity" section) and the **registry §7 score-layer contract**
  prerequisite (its activation gate). This core only links them; it does not
  restate or re-decide them.
- **Hook scope: GPU foot-bridge only.** This synthesis is scoped end-to-end —
  B1's substrate/fidelity owner (the bridge-specific H0) and O1's intervention
  contract alike — to the **GPU foot-bridge** two-stage winner
  ([bridge decision semantics](../../../research/tracker-decision/relink_bridge.md)).
  The association (auction) stage is a different online contract and is **out
  of scope**: an association-stage variant would require its own
  substrate/fidelity owner and a separately declared synthesis/charter, not a
  reinterpretation of this one. Within the bridge, the legal insertion surface
  is the candidate-local (stage-1) lost ranking under fixed pair eligibility;
  claim arbitration, loser fallback, and commit mutation are separate
  online-contract problems.
- **Reserved-symbol renames applied at landing** (the 2026-07-16 working draft
  used the left-hand forms):

  | Draft form | Landed form | Collision avoided |
  |:--|:--|:--|
  | \(s_0(i\mid e)\) base score | \(s_{\mathrm{base}}(i\mid e)\) | repo `s0` = offline proxy of `bdist`; registry forbids "s0 represents production `bdist`" |
  | `P0_model_id` | `P0_exit_cov_id` | P0 = sealed identifiability study code |
  | `R1_model_id` | `R1_obs_cov_id` | R1 = sealed capture-replay study code |
  | "ambiguous band" (unqualified) | runtime-coordinate band, defined by the future declaration | Door 0 closed the s0-proxy band class (`T2_NO_USABLE_RANKING_POWER_IN_CLASS`, class-scoped) |

  The `GCTM_B1_*` / `GCTM_O1_*` terminal prefixes and the `gctm_b1_*` /
  `gctm_o1_*` future file names are kept precisely so they cannot be confused
  with the closed `m_b1_*` (M-B1) line.

---

## 1. Evidence chain

```text
H0
runtime observability / evidence fidelity

→ GCTM
transition-family mathematical specification

→ B1
runtime-grounded offline attribution and score-ranking evaluation

→ O1
online intervention and system-efficacy evaluation

→ production evaluation
deployment suitability and promotion
```

Every arrow is a separate evidence edge. No upstream result automatically
authorizes the next task.

### 1.1 Claim ownership

| Object | Owns | Does not establish |
|:--|:--|:--|
| H0 | Whether runtime quantities and decision events are faithfully observable | Whether a stochastic model is mathematically valid or useful |
| GCTM | Whether M0/M1/M2 and the observation interface are well-posed and sealable | Empirical calibration, ranking value, online value |
| B1 | Whether one frozen GCTM instantiation has stable runtime-grounded offline ranking value | Online retention or MOT improvement |
| O1 | Whether one frozen B1 policy survives causal online execution and improves the actual system | Production safety or default promotion |
| Production evaluation | Latency, rollback, deployment risk, broad operational acceptance | Upstream scientific claims beyond its scope |

### 1.2 Non-retroactivity

- A valid negative B1 result does not invalidate H0 or the abstract GCTM
  mathematics. It closes the declared offline score-ranking path for the tested
  model class.
- A valid negative O1 result does not invalidate a valid B1 offline claim. It
  closes the declared online transport/intervention path for the tested hook and
  policy.
- An invalid experiment closes the current execution, not the scientific
  hypothesis path.
- A downstream positive result cannot repair an upstream provenance,
  identifiability, or fidelity failure.
- Engineering merge does not equal research acceptance.
- O1 acceptance does not equal production promotion.

---

## 2. Shared semantic rules

### 2.1 Target layer

Both B1 and O1 target **score-ranking**, not a coarse reject gate.

The intended intervention has the form:

\[
s_{\mathrm{new}}(i\mid e)
=
s_{\mathrm{base}}(i\mid e)
+
\Delta s_{\mathrm{GCTM}}(i\mid e),
\]

where \(e\) is one candidate event and \(i\) is one candidate in that event.
The base score is written \(s_{\mathrm{base}}\), never `s0`: in this repository
`s0` is the reserved name of the offline proxy of `bdist`, and any use of `s0`
to stand for a production quantity is registry-inadmissible (§0).

The GCTM signal changes relative preference inside the retained ambiguous band
(a runtime-coordinate band that the future declaration must define; it is not
the closed Door 0 s0-proxy band class, §0).
It must not become a hard rejection rule unless a later task is explicitly
re-chartered for the gate layer.

### 2.2 Transition likelihood is not a GT posterior

The model quantity is of the form:

\[
p(y_1\mid z_0,\Delta,c,M),
\]

or a score derived from that conditional transition model.

It is not automatically:

\[
P(\mathrm{GT}\mid y_1,z_0,\Delta,c).
\]

A transition likelihood may rank candidates, calibrate expected motion
deviation, or expose a failure mode. It must not be described as a calibrated
identity probability without a separately specified discriminative model and
calibration claim.

### 2.3 Four distinct claim spaces

```text
calibration space
  Does q or NLL have stable probabilistic meaning across gaps and contexts?

candidate-ranking space
  Does the score order the GT candidate above competing candidates in one event?

assignment space
  Does the changed ordering alter the selected relink decision
  (bridge claim/commit outcome; §0 hook scope)?

system space
  Do altered decisions improve track-level and sequence-level MOT outcomes?
```

Improvement in one space does not imply improvement in the next.

### 2.4 Output classes

Each result must be classified as exactly one of:

- **design candidate** — purpose-aligned, interpretable, structurally simple,
  stability-validated, and above a predeclared utility bar;
- **performance upper-bound candidate** — estimates capability but carries no
  design authority;
- **diagnostic result** — calibration map, attribution, identifiability verdict,
  failure mode, or exceptional-tail description;
- **unexplained residual set** — unresolved support that must not be force-fit
  with additional conditions.

Only a B1 **design candidate** may make O1 eligible.

### 2.5 Validity versus efficacy

Both tasks must distinguish:

```text
INVALID / UNRESOLVED
  The experiment could not answer the question.

VALID NEGATIVE
  The experiment answered the question and the minimum effect was not met.

VALID POSITIVE
  The predeclared effect, stability, and mechanism bars were met.
```

There is no residual “describe more and continue” outcome in a sealed
mainline-capable declaration.

### 2.6 Dual-space and reduction typing

Every future declaration must name:

1. source space;
2. decision space;
3. reduction \(\rho\);
4. aggregation rule;
5. dependence structure;
6. conservation identities;
7. type \(\kappa\) for every decidable claim:

\[
\kappa
=
(\text{quantification space},
 \text{comparison relation},
 \text{decision rule}).
\]

Fidelity, calibration, ranking, assignment, and system metrics require separate
\(\kappa\)-objects.

---

## 3–34. Moved to the task charters (2026-07-17 split)

The per-task sections of the 2026-07-16 synthesis moved verbatim (with only
link-depth fixes and charter framing) to:

| Original sections | New owner |
|:--|:--|
| §3–§17 Part I (B1 identity, activation gate, research question, maximum claim, scope, frozen inputs, evidence spaces, model-comparison semantics, fit/blind/reveal, headline metrics, positive design bar, terminal family, mainline transitions, deliverables, conclusion form) and §34 B1→O1 handoff object | [B1 task charter](../../../research/threads/gctm_b1_runtime_grounded_offline_attribution_task.md) |
| §18–§33 Part II (O1 identity, activation gate, research question, maximum claim, scope, frozen policy identity, intervention semantics, comparison arms, evidence spaces, validity gates, headline metrics, positive bar, terminal family, mainline transitions, deliverables, conclusion form) | [O1 task charter](../../../research/threads/gctm_o1_online_intervention_efficacy_task.md) |
| §38 pre-activation checklists | split into the two charters |

The pre-split full text is preserved in git history (PR #178, `fba683b2`).

---

## 35. Outcome interpretation matrix

| B1 outcome | O1 status | Meaning |
|:--|:--|:--|
| invalid/unresolved | blocked | B1 did not answer the offline question |
| calibration-only | blocked | no ranking design candidate |
| no stable ranking gain | blocked | offline score path closed |
| score design candidate | eligible after owner acceptance | O1 may be separately declared |
| score design candidate using offline-only observation | blocked | causal variant must return through B1 |
| positive B1 + invalid O1 | unresolved online edge | B1 remains valid |
| positive B1 + harmful O1 | online path closed/default-off | B1 remains offline-valid |
| positive B1 + retention without system gain | mechanism exists but a system bottleneck remains | no production eligibility |
| positive B1 + positive O1 | production evaluation eligible | still no production promotion |

---

## 36. Forbidden inference shortcuts

\[
\text{GCTM theory sealable}
\not\Rightarrow
\text{B1 empirical value}
\]

\[
\text{better calibration}
\not\Rightarrow
\text{better candidate ranking}
\]

\[
\text{better candidate ranking}
\not\Rightarrow
\text{assignment change}
\]

\[
\text{assignment change}
\not\Rightarrow
\text{better MOT outcome}
\]

\[
\text{offline ranking gain}
\not\Rightarrow
\text{online causal availability}
\]

\[
\text{online mechanism retention}
\not\Rightarrow
\text{system gain}
\]

\[
\text{online system gain}
\not\Rightarrow
\text{production promotion}
\]

\[
\text{engineering merge}
\not\Rightarrow
\text{research acceptance}
\]

---

## 37. Repository split — executed 2026-07-17

This synthesis prescribed splitting into two independent task objects when the
tasks are scheduled. The owner chose to execute the split at charter-creation
time instead; the two proposed task charters now exist at exactly the paths
this section named:

```text
docs/research/threads/
  gctm_b1_runtime_grounded_offline_attribution_task.md

docs/research/threads/
  gctm_o1_online_intervention_efficacy_task.md
```

Each task owns its own:

```text
activation state
declaration
frozen degrees of freedom
validity gates
ordered terminals
evidence artifacts
owner terminal acceptance
```

B1 may define O1 eligibility but must not authoritatively define O1 execution.
O1 consumes the accepted B1 handoff and must not rewrite B1 semantics.
This core document is not the sealed owner for either study.

---

## 39. Final semantic summary

```text
B1 asks:
  Is the stochastic transition representation a real,
  stable, interpretable offline ranking capability
  on a runtime-grounded substrate?

O1 asks:
  Does one frozen B1 ranking policy survive causal online execution
  and improve the actual tracking system?

B1 positive:
  O1 may be declared.

O1 positive:
  production evaluation may be declared.

No result auto-promotes.
No invalid experiment becomes a negative scientific verdict.
No offline claim is silently transported online.
No online candidate becomes production behavior without a new acceptance edge.
```
