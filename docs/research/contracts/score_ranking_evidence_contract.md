<!-- doc-status: proposed -->
<!-- doc-promotion: none; draft normative contract, not yet transition authority -->
<!-- doc-date: 2026-07-23 -->
<!-- doc-module: cross -->
<!-- contract-owner: claim-state registry score layer -->

# Score-ranking evidence contract（L2）

## 0. Status, authority, and activation

This document is the proposed cross-study contract for `layer: L2 score`.
It supplies the missing semantics for event-local rank, margin, top-1,
calibration, and the score claim ladder. It is a **draft** until an owner
accepts and freezes it.

While this file remains `doc-status: proposed`:

- every registry object whose only missing transition authority is this
  contract remains `transition_semantics: unavailable`;
- this file does not activate B1 or O1, admit a score object to the candidate
  set, authorize data access, select a score, or change runtime behavior;
- a study may review this draft, but may not cite it as satisfied activation
  authority.

Promotion requires an owner review that freezes this contract, updates
[`claim_state_registry.md`](claim_state_registry.md), and re-derives affected
objects. PR merge alone is not acceptance.

## 1. Scope and precedence

This contract owns:

- score-object identity;
- candidate-event and candidate-universe identity;
- score direction, ranking, tie, margin, top-1, and top-k semantics;
- calibration claim typing;
- score-policy comparison and conservation rules;
- the score-ranking claim ladder `SR0`–`SR6`;
- fail-closed admissibility and transition rules for `layer: L2 score`.

It does not own:

- support membership, safe rejection, coverage, or prune mass (`L0 gate`);
- a study's model equations, feature extraction, or fitted parameters;
- runtime-quantity fidelity;
- assignment implementation or system metrics;
- study scheduling, terminal acceptance, or production promotion.

Precedence is:

1. [`signal_table_schema.md` §0.5](signal_table_schema.md) decides gate versus
   score;
2. this contract decides score evidence and score transitions;
3. [`runtime_quantity_fidelity_protocol.md`](runtime_quantity_fidelity_protocol.md)
   decides whether an offline quantity represents a runtime quantity;
4. a sealed study declaration instantiates, but may not redefine, these rules.

If a policy removes candidates before ranking, that operation is a gate claim
and must be declared under the gate contract. Calling the cutoff a score
threshold does not move it into L2.

## 2. Typed object and spaces

### 2.1 Score-policy object

A score-policy object is the complete tuple

\[
\mathcal P =
(U_{\mathrm{src}}, U_{\mathrm{evt}}, \rho, C, s, o, T, \tau, \pi),
\]

where:

- \(U_{\mathrm{src}}\) is the source pair space;
- \(U_{\mathrm{evt}}\) is the candidate-event space;
- \(\rho\) maps pairs to exactly one event;
- \(C_e\) is the candidate universe for event \(e\);
- \(s(e,i)\) is the pair score;
- \(o\in\{\texttt{higher\_better},\texttt{lower\_better}\}\) is orientation;
- \(T\) is the declared score transform and composition;
- \(\tau\) is the complete tie rule;
- \(\pi\) is the frozen selection or abstention rule, if the claim reaches
  assignment space.

Changing any tuple member creates a different evaluated policy. It may be
compared with the old policy, but it may not inherit the old policy's rung.

### 2.2 Required spaces

Every declaration must name:

```text
source pair space
candidate-event space
calibration space
assignment space, if any
system space, if any
reduction for every adjacent pair of spaces
trial unit and residual dependence above it
```

Ranking claims quantify over events, not pooled pair rows. Calibration claims
quantify over the declared calibration unit. Assignment and system claims
require separate claim objects; neither is a synonym for ranking.

### 2.3 Candidate-universe identity

For each event, the declaration must freeze:

```text
event_key
candidate identity key
candidate inclusion source
gate or retained-band identity
GT-present / GT-absent / ambiguous-label partition
duplicate-candidate rule
empty and singleton-event rule
drop-reason vocabulary
```

A paired policy comparison is admissible only when both policies see the same
event set and the same \(C_e\), unless candidate-universe change is itself the
predeclared intervention. In the latter case, the result is not a pure score
comparison and must report support change separately.

## 3. Canonical score semantics

### 3.1 Orientation and canonical utility

Every score field declares exactly one orientation. For comparison only, define
canonical utility

\[
u(e,i)=
\begin{cases}
s(e,i), & o=\texttt{higher\_better},\\
-s(e,i), & o=\texttt{lower\_better}.
\end{cases}
\]

Rank and top-k are computed from \(u\). Reported margins must name their native
domain as well as their sign convention. A cost margin, affinity margin, and
softmin-probability margin are different quantities.

A strictly increasing transform of \(u\) preserves strict rank but does not
preserve margin magnitude or calibration. Therefore:

```text
same rank under monotone transform
  does not imply
same margin, same cutoff, same calibration, or same assignment behavior
```

### 3.2 Rank

For an event with one declared correct candidate \(i^\star\),

\[
\operatorname{rank}_e(i^\star)
=1+\sum_{j\in C_e\setminus\{i^\star\}}
\mathbf 1[u(e,j)>u(e,i^\star)]
+\operatorname{tie\_offset}_\tau(e,i^\star).
\]

The tie rule must be deterministic and frozen before reveal. Legal rules
include stable candidate-key ordering or fractional rank for a descriptive
metric. A study may not use fractional rank for the headline and a favorable
deterministic rule for top-1 without declaring two separate claim objects.

Events with no correct candidate do not have GT rank. Events with multiple
correct candidates must predeclare whether the target is best-valid rank,
all-valid rank, or one uniquely selected label. Ambiguous labels are a separate
partition and cannot silently count as failures or successes.

### 3.3 Margin

For one correct candidate and at least one negative candidate, the canonical
GT-versus-best-negative margin is

\[
m_e =
u(e,i^\star)
-\max_{j\in C_e\setminus\{i^\star\}}u(e,j).
\]

Thus \(m_e>0\) is a strict win, \(m_e=0\) is a tie, and \(m_e<0\) is a loss.
Singleton events have no competition margin and must not receive \(+\infty\),
zero, or an imputed success in the headline denominator.

Margin claims must freeze the score domain, transform, scale, normalization,
and aggregation. Comparing margin magnitudes across domains is inadmissible
unless the declaration supplies a justified common scale.

### 3.4 Top-1 and top-k

Top-1 success means the declared correct candidate is selected by the frozen
tie-aware argmax over the unchanged \(C_e\). Top-k success uses the same
ordering and a predeclared \(k\).

Every top-1/top-k report must include:

```text
eligible event count
GT-present / GT-absent / ambiguous-label counts
singleton-event count
tie count and tie disposition
candidate-count distribution
abstention count, if a decision rule exists
```

Top-1 ranking success is not assignment retention unless the actual decision
rule, cutoff, fallback, and abstention behavior were applied.

### 3.5 Cutoff and retained-band roles

Every cutoff must declare one role:

- `support_gate_imported`: candidate membership decided by an independently
  accepted L0 object;
- `reporting_slice`: no membership or runtime decision effect;
- `assignment_rule`: an L2 decision rule evaluated only when assignment space
  is in scope.

A score study may consume a frozen support gate. It may not tune that gate on
the score test set or credit removed candidates as ranking wins.

## 4. Calibration is a separate claim axis

Calibration asks whether a declared score has a stable meaning relative to a
declared reference, not whether the correct candidate wins.

The declaration must state which claim is made:

- distributional calibration of a model residual or NLL;
- probabilistic calibration of a discriminative match probability;
- scale comparability across gaps, contexts, or components;
- no calibration claim; diagnostics only.

A transition likelihood
\(p(y_1\mid z_0,\Delta,c,M)\) is not automatically
\(P(\mathrm{GT}\mid y_1,z_0,\Delta,c)\). A proper-score or coverage improvement
does not advance a ranking rung. A rank improvement does not establish
calibration.

Calibration evidence must freeze its reference distribution or target,
calibration unit, bins or estimator, proper score if used, held-out rule, and
minimum-exposure rule. Post-reveal recalibration invalidates the held-out claim
for that fold.

## 5. Comparison contract

Every decidable score claim declares

\[
\kappa =
(\text{quantification space},
 \text{comparison relation},
 \text{decision rule}).
\]

At minimum, freeze:

```text
baseline and candidate policy identities
paired event universe
primary ranking metric
score orientation and transform
tie rule
minimum exposure
minimum effect
dependence and uncertainty treatment
folds and no-refit boundary
stability strata
short-gap or protected-stratum retention bar
missing-value and fallback behavior
futility rule
```

Legal primary ranking metrics include event-local pairwise accuracy, reciprocal
rank, top-k recall, GT-versus-best-negative margin, and correct assignment rate
under a frozen offline decision rule. Pooled AUC, raw NLL, mean pair score, or
one favorable sequence cannot substitute for an event-local headline.

## 6. Score-ranking claim ladder

The strongest supported rung is bounded by all lower-rung requirements.
Engineering delivery alone advances no rung.

### SR0 — Typed observed score

One frozen policy has well-formed pair, event, orientation, tie, and denominator
semantics on one observed sample. This supports auditability only.

### SR1 — In-sample event-local separation

The policy shows a predeclared event-local ranking effect on the data used to
select or fit it. Post-selection generalization is not established.

Required beyond SR0: policy-search space, fit/selection data, exposure, primary
metric, effect, and all candidate-universe conservation checks.

### SR2 — Held-out retained ranking

A policy frozen without the evaluated fold retains the predeclared minimum
ranking effect on held-out events.

Required beyond SR1: disjoint fit/selection/test identities, blind-to-reveal
binding, per-fold result, dependence-aware uncertainty, and no-refit behavior.

### SR3 — Stable score-design candidate

The SR2 effect also satisfies the frozen cross-sequence, gap/context,
protected-stratum, and short-gap retention bars, with no candidate-universe
drift or unexplained fallback advantage.

This is the highest rung an offline score-ranking design study can ordinarily
establish.

### SR4 — Cross-substrate portable score policy

The same policy interpretation survives the declared substrate change.

Required beyond SR3: quantity-fidelity acceptance, score and hook semantic
equivalence, runtime-causal field availability, candidate-universe parity, and
no silent transform or fallback change.

### SR5 — Online assignment-retained score policy

The portable policy is executed at the declared online hook and retains its
assignment-space effect.

Required beyond SR4: default-off A/B, applied/rejected audit, online state
provenance, actual fallback/abstention behavior, and unchanged disabled
baseline.

### SR6 — Production score-policy candidate

The SR5 policy satisfies a separately sealed system-efficacy and production
evaluation contract, including track/sequence metrics, latency, resource and
failure-mode audit, rollback, default behavior, and explicit acceptance.

SR6 is candidacy, not automatic default promotion.

## 7. Conservation and fail-closed validity

At minimum, a score packet must prove:

```text
each admitted pair belongs to exactly one event
each event belongs to exactly one sequence and fold
pair counts equal the sum of event candidate counts
GT-present / GT-absent / ambiguous partitions reconcile
fit, selection, validation, and test partitions are disjoint as declared
baseline and candidate policies have paired event and candidate identities
every dropped record has exactly one enumerated reason
all non-finite or missing scores follow the frozen rule
```

Any of the following makes the affected claim invalid rather than negative:

- unknown or drifting candidate universe;
- unstated orientation or tie behavior;
- pooled-pair inference for an event-level claim;
- post-reveal fit, weight, transform, cutoff, or fallback change;
- score-policy comparison that also changes support without typing that change;
- GT leakage into a runtime-intended score;
- missing denominator reconciliation;
- substrate claim without accepted fidelity evidence;
- claiming assignment or system efficacy from rank alone.

A valid result below the minimum effect is a valid negative. It must not be
reported as invalid merely to preserve the option to continue searching.

## 8. Required declaration and result surface

A sealed study declaration must pin all applicable fields in §§2–5, its target
rung, exhaustive validity/negative/positive terminals, and the state transition
or explicit `none` for every terminal.

A result packet must expose:

```text
policy and substrate identities
source/event/calibration spaces and reductions
candidate-universe reconciliation
orientation, transform, tie, cutoff, fallback, and missing-value rules
exposure and denominator tables
primary ranking result with dependence-aware uncertainty
per-fold and mandatory stability views
calibration results as separate claim objects
assignment/system results only when separately authorized
terminal selected by the sealed decision procedure
```

## 9. Forbidden shortcuts

The following inferences are forbidden:

```text
calibration gain        => ranking gain
rank gain               => assignment retention
assignment retention   => system efficacy
offline score          => runtime score
monotone-rank equality => equal margin or cutoff semantics
top-1 on changed C_e   => pure score improvement
pooled AUC             => event-local design candidate
PR merge               => contract acceptance or rung advancement
```

## 10. Initial consumer boundary: GCTM B1

The proposed
[`GCTM B1` charter](../threads/gctm_b1_runtime_grounded_offline_attribution_task.md)
is an intended consumer, not this contract's authority. Its future declaration
must instantiate this contract and additionally satisfy its H0 substrate,
consumer-compatibility, GCTM identity, B1-slot, seal, and scheduling gates.

For GCTM B1 specifically:

- \(q\)/NLL calibration and candidate ranking remain separate claims;
- shared-covariance \(q\)/NLL rank equivalence is an implementation invariant,
  not evidence of ranking gain;
- the GCTM score is an augmentation candidate unless a separately frozen
  replacement claim is authorized;
- the maximum ordinary offline positive state is `SR3`, bounded by the B1
  charter's own maximum supported claim.

## 11. Promotion checklist

- [ ] owner accepts the object, space, and candidate-universe definitions;
- [ ] owner accepts rank, tie, margin, top-1/top-k, and cutoff semantics;
- [ ] owner accepts calibration separation and transition-likelihood boundary;
- [ ] owner accepts `SR0`–`SR6` and non-inheritance rules;
- [ ] fail-closed validity and conservation rules are mechanically instantiable;
- [ ] registry §7 is updated from absent to the accepted contract identity;
- [ ] affected L2 objects are individually re-derived; none auto-advance;
- [ ] B1 remains non-WIP unless all of its other gates and separate scheduling
      are satisfied.
