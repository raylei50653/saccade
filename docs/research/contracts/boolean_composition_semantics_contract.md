# Boolean Composition Semantics Contract

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-10 -->
<!-- doc-module: cross -->

## 0. Role

This document is the normative Boolean-semantics patch for
[Statistical Robust Feasible-Set Estimation under Asymmetric Loss](statistical_robust_feasible_set_estimation_under_asymmetric_loss.md), especially its Boolean-composition section.

It defines the executable contract required by evaluator implementations, grammar search, RegionAsset serialization, and PR review when a policy contains `AND`, `OR`, `NOT`, necessary/support roles, missing values, or online state.

It does **not** authorize G7, NOT-based rejection, online intervention, transfer, production behavior, or any maturity/claim promotion. Existing studies remain bounded by their accepted evidence.

The governing principle is:

> Boolean set algebra is valid only after the observation space, predicate codomain, unknown-value policy, operand roles, threshold edges, grammar semantics, and execution state have been typed explicitly.

---

## 5.0 Typed spaces: candidate universe \(\Omega\) versus parameter space \(\Theta\)

Two different set types are used and must never be silently mixed.

### 5.0.1 Candidate-space predicates and reject sets

An atomic predicate is evaluated on observations or candidates:

$$
A_i : \Omega \rightarrow \{T,F,U\}
$$

where `U` means unknown or undefined under the declared predicate contract.

Its true/reject set is:

$$
R_{A_i}
=
\{x\in\Omega : A_i(x)=T\}.
$$

For a total two-valued predicate, the codomain may be declared as \(\{T,F\}\), but two-valuedness must never be inferred merely because no unknown value appeared in one sample.

### 5.0.2 Parameter-space feasible sets

A policy or parameter coordinate \(\theta\in\Theta\) induces a decision on \(\Omega\). The productive-safe feasible set is:

$$
\mathcal S_{\varepsilon,g_{\min}}
=
\left\{
\theta\in\Theta:
L_{\mathrm{GT}}(\theta)\le\varepsilon,
\quad
G_{\mathrm{FP}}(\theta)\ge g_{\min}
\right\}.
$$

`Reject-set inclusion` is a statement in \(\Omega\). `Safe-region geometry` is a statement in \(\Theta\). They are different types.

Forbidden inference:

```text
reject set grew
⇒ feasible set grew
```

For example, `OR` expands the same-state reject set in \(\Omega\), but normally makes the GT constraint harder to satisfy and can therefore shrink the feasible set in \(\Theta\).

Every equation, table, mask hash, or region statistic must declare whether its grain is:

```text
candidate / observation in Ω
reject set in Ω
policy / coordinate in Θ
feasible set in Θ
trajectory / state path
```

---

## 5.1 Predicate domain, totality, and unknown state

### 5.1.1 Unknown causes

A predicate may return `U` for reasons including:

- NaN or non-finite signal input;
- missing embedding or feature;
- insufficient history;
- zero or undefined denominator;
- uninitialized online state;
- candidate type outside the atom's declared domain;
- failed upstream observation or unavailable sensor/source.

These causes must not be silently coerced to either `T` or `F`.

### 5.1.2 Three-valued semantics

Unless a study explicitly proves total two-valued predicates, use Strong Kleene truth tables:

| A | B | `A AND B` | `A OR B` |
|:--:|:--:|:--:|:--:|
| F | F | F | F |
| F | U | F | U |
| F | T | F | T |
| U | F | F | U |
| U | U | U | U |
| U | T | U | T |
| T | F | F | T |
| T | U | U | T |
| T | T | T | T |

Negation is:

| A | `NOT A` |
|:--:|:--:|
| T | F |
| F | T |
| U | U |

Hard requirement:

$$
\neg U = U,
$$

never `T`.

### 5.1.3 Final reject action

For asymmetric GT loss, the default fail-safe action mapping is:

```text
T → reject
F → no reject
U → no reject
```

A different mapping requires a separately accepted intervention contract. In particular, `U → reject` is forbidden by default.

With the default mapping, define:

$$
R_A = \{x : A(x)=T\}.
$$

Then Strong Kleene composition still gives exact reject-set relations:

$$
R_{A\land B}=R_A\cap R_B,
\qquad
R_{A\lor B}=R_A\cup R_B.
$$

For negation under three-valued semantics:

$$
R_{\neg A}=\{x:A(x)=F\},
$$

which is **not** generally equal to \(\Omega\setminus R_A\), because unknown points are excluded from both reject sets.

The complement identity

$$
R_{\neg A}=\Omega\setminus R_A
$$

is legal only when `predicate_codomain={T,F}` is declared and verified on the entire relevant universe.

---

## 5.2 Candidate-universe identity and comparability

The candidate universe is not a generic dataset name. It is an identified pre-decision observation space:

$$
\Omega
=
\Omega(
\text{substrate},
\text{hook},
\text{candidate builder},
\text{time/state boundary},
\text{prefilters}
).
$$

A `universe_id` must bind at least:

```text
substrate identity
hook / decision location
candidate-builder contract and version
prefilters and eligibility rules
label / exposure owner
time or frame range
pre-decision state-snapshot contract
candidate primary key schema
```

A reproducible study should also seal a `universe_hash` over the ordered or canonically sorted candidate identities and the fields required to reconstruct exposure.

### 5.2.1 Same-universe requirement

Direct Boolean composition, mask equality, hurt overlap, and set inclusion require:

```text
same universe_id
same candidate primary-key semantics
same pre-decision state snapshot
same label/exposure contract
```

If:

$$
A\subseteq\Omega_1,
\qquad
B\subseteq\Omega_2,
\qquad
\Omega_1\ne\Omega_2,
$$

then `A AND B`, `A OR B`, and direct mask comparison are undefined until a declared transport/projection exists:

$$
T_{1\to2}:\Omega_1\rightarrow\Omega_2.
$$

The transport must define unmatched, duplicated, merged, and state-dependent candidates. A filename, row position, or apparent semantic similarity is not a transport contract.

### 5.2.2 Project boundary

The following are not assumed to be the same universe:

```text
offline analysis pairs
B2 candidates
online-hook candidates
post-intervention regenerated candidates
candidates from different substrates or prefilters
```

Boolean algebra is valid only on one declared universe and one declared pre-decision state.

---

## 5.3 Boolean grammar and operator precedence

### 5.3.1 Core syntax

A grammar contract must declare `grammar_version` and the legal AST forms. The core operators have fixed precedence:

```text
NOT > AND > OR
parentheses are authoritative
```

A generic typed AST is:

```text
Policy := Atom
        | NOT(Policy)
        | AND(Policy, ...)
        | OR(Policy, ...)
```

This generic syntax does not authorize every form. Each grammar version must additionally declare:

```text
maximum_nesting_depth
maximum_operands_per_node
whether n-ary AND/OR are legal
whether repeated atoms are legal
whether mixed operand roles are legal
NOT scope: atom_only | arbitrary_subtree
allowed top-level roles
```

If any bound or scope required by the grammar is absent, the policy is invalid rather than implicitly unlimited.

### 5.3.2 Current bounded grammar compatibility

A fixed-universe threshold atlas with total tail atoms, no NOT, and one binary top-level `AND` or `OR` is a valid restricted grammar instance when all threshold and universe fields in this contract are supplied.

G7, necessary/support-role composition, arbitrary NOT, and online closed-loop policies require their own declared grammar versions and role-validity contracts.

---

## 5.4 Threshold and equality boundary contract

Every threshold atom must serialize at least:

```text
signal_identity
signal_unit
comparison_direction
comparator                    # >, >=, <, <=, ==, interval, membership
endpoint_policy               # strict / inclusive per endpoint
tie_policy
threshold_value representation
threshold_index / registry identity
quantile_method and interpolation
floating_point_tolerance
clipping_domain
nan_policy
posinf_policy
neginf_policy
missing_value_policy
candidate_type applicability
```

The following are different policies and must have different semantic identities unless the registered domain proves them extensionally equal:

$$
s(x)>t
\qquad\text{and}\qquad
s(x)\ge t.
$$

A hidden comparison such as:

```python
score > threshold - 1e-6
```

is illegal unless the tolerance and arithmetic semantics are explicit policy fields.

### 5.4.1 Quantiles

A quantile threshold must declare:

```text
reference population
weighting
interpolation method
handling of duplicate values
handling of missing/non-finite values
whether threshold is recomputed or frozen
```

Quantile index equality does not imply threshold-value equality across substrates or folds.

### 5.4.2 Degenerate reject sets

If:

$$
R_\theta=\varnothing,
$$

then the policy may be empirically safe but is not productive unless the productivity contract explicitly allows zero.

If:

$$
R_\theta=\Omega,
$$

then it is maximal reject and must still be audited against actual GT exposure. It must not be labeled safe because a denominator is empty or missing.

### 5.4.3 Zero denominators

When:

$$
N_{\mathrm{GT,exposed}}=0,
$$

`gt_hurt_rate` is `NA/undefined`, not zero. Likewise for FP productivity rates with zero FP exposure.

Count-based statements may still be reported, but no rate-feasibility conclusion may be inferred without a denominator owner.

---

## 5.5 Set monotonicity and asymmetric-loss composition

For fixed predicates evaluated on the same universe, same state, same labels, and same exposure denominators:

$$
R_{A\land B}
\subseteq R_A,
\qquad
R_{A\land B}
\subseteq R_B,
$$

and:

$$
R_A
\subseteq R_{A\lor B},
\qquad
R_B
\subseteq R_{A\lor B}.
$$

These are set-theoretic results, not tendencies.

Define GT-harm and FP-removal sets:

$$
H_A=R_A\cap\Omega_{\mathrm{GT}},
\qquad
P_A=R_A\cap\Omega_{\mathrm{FP}}.
$$

Then:

$$
H_{A\land B}=H_A\cap H_B,
\qquad
P_{A\land B}=P_A\cap P_B,
$$

$$
H_{A\lor B}=H_A\cup H_B,
\qquad
P_{A\lor B}=P_A\cup P_B.
$$

Therefore:

### AND

$$
N_{\mathrm{GT,hurt}}(A\land B)
\le
\min\left(
N_{\mathrm{GT,hurt}}(A),
N_{\mathrm{GT,hurt}}(B)
\right),
$$

$$
N_{\mathrm{FP,removed}}(A\land B)
\le
\min\left(
N_{\mathrm{FP,removed}}(A),
N_{\mathrm{FP,removed}}(B)
\right).
$$

### OR

$$
N_{\mathrm{GT,hurt}}(A\lor B)
\ge
\max\left(
N_{\mathrm{GT,hurt}}(A),
N_{\mathrm{GT,hurt}}(B)
\right),
$$

$$
N_{\mathrm{FP,removed}}(A\lor B)
\ge
\max\left(
N_{\mathrm{FP,removed}}(A),
N_{\mathrm{FP,removed}}(B)
\right).
$$

The exact OR hurt is:

$$
|H_A\cup H_B|
=
|H_A|+|H_B|-|H_A\cap H_B|.
$$

Marginal hurt rates are insufficient to reconstruct joint OR safety. The audit must retain joint overlap or candidate-level membership. Without overlap, only the loose union bound is available:

$$
L_{\mathrm{GT}}(A\lor B)
\le
L_{\mathrm{GT}}(A)+L_{\mathrm{GT}}(B),
$$

capped at one when rates share the same denominator.

### 5.5.1 Exact-zero empirical closure

On the same complete observed sample, if:

$$
H_A=\varnothing,
\qquad
H_B=\varnothing,
$$

then:

$$
H_{A\lor B}=\varnothing.
$$

Thus empirical GT0 reject sets are closed under OR **only** under all of:

```text
same universe and candidate identities
same labels and exposure set
same pre-decision state
same complete observed sample
deterministic predicates or fixed U→no-reject semantics
no hidden post-action candidate regeneration
```

This is an empirical finite-sample statement. It is not population, held-out, cross-substrate, or closed-loop safety.

For \(\varepsilon>0\), operand-level feasibility does not authorize OR. A joint hurt audit is mandatory.

---

## 5.6 Operand-role validity matrix

Every atom and subtree must carry a declared semantic role. Truth-functional equivalence does not erase role provenance.

Roles:

```text
sufficient_reject
necessary_envelope
support
complement_derived
untyped_observation
```

### 5.6.1 Sufficient reject

For an empirically qualified sufficient-reject set `S`:

- `S1 AND S2` remains a sufficient-reject candidate on the same evidence, with reduced or equal coverage;
- `S1 OR S2` requires a joint hurt audit;
- under the exact-zero closure conditions in §5.5.1, `S1 OR S2` may retain **empirical** GT0 sufficiency on that sample;
- `NOT S1` does not inherit sufficient-reject semantics.

For \(\varepsilon>0\), two individually qualified sufficient operands do not imply that their OR is qualified.

### 5.6.2 Necessary envelope

Let \(R^\star\) be the target set and suppose:

$$
R^\star\subseteq N_1,
\qquad
R^\star\subseteq N_2.
$$

Then:

$$
R^\star\subseteq N_1\cap N_2,
$$

so `N1 AND N2` is a tighter necessary envelope.

Also:

$$
R^\star\subseteq N_1\cup N_2,
$$

but this is a weaker necessary envelope.

A necessary envelope is not a reject authorization. Its truth value alone must never trigger rejection unless a separate target/complement contract proves that action semantics.

### 5.6.3 Support

A support operand modifies scope, confidence, or applicability and cannot independently authorize rejection.

Allowed structural form, subject to fresh audit:

$$
S_{\mathrm{sufficient}}\land P_{\mathrm{support}}.
$$

Forbidden without a separate role authorization:

```text
support as top-level reject policy
support OR any subtree as direct reject authorization
NOT support promoted to sufficient reject
support role erased during canonicalization
```

### 5.6.4 Role-validity matrix

| Operand roles / operator | Default validity | Required evidence |
|:--|:--|:--|
| sufficient AND sufficient | conditionally valid | same universe; joint output audit |
| sufficient OR sufficient | not inherited automatically | joint hurt membership; exact-zero special case allowed |
| necessary AND necessary | valid as tighter envelope | same target-set semantics |
| necessary OR necessary | valid but weaker envelope | same target-set semantics |
| sufficient AND support | conditionally valid | sufficient branch remains action owner; joint audit |
| support alone | invalid for reject | separate authorization required |
| support OR anything | invalid for reject by default | new composite qualification required |
| NOT of any role | role becomes `complement_derived` | explicit universe, target, unknown, and action semantics |
| mixed necessary/sufficient | invalid by default | grammar-specific role theorem and audit |

Unknown or unrecognized roles fail closed at grammar validation.

---

## 5.7 NOT and complement contract

NOT is not merely a syntax operator. It requires four explicit contracts:

```text
candidate universe Ω
predicate unknown semantics
semantic target set
resulting action authorization
```

From:

$$
R^\star\subseteq N,
$$

one may derive:

$$
\Omega\setminus N
\subseteq
\Omega\setminus R^\star.
$$

This means points outside the necessary envelope are outside the target set. It does **not** determine whether those points are safe to reject until the target set's meaning and action mapping are fixed.

NOT does not automatically exchange `necessary` and `sufficient` roles.

A valid complement node must serialize:

```text
complement_universe_id
predicate_codomain
unknown_value_policy
complement_target_semantics
source_operand_role
result_role = complement_derived
final_action_authorization
```

If any field is missing, NOT may be evaluated only as a diagnostic observation, not as a reject policy.

---

## 5.8 Canonicalization and equivalence levels

### 5.8.1 Three distinct equivalences

1. **Syntactic equivalence**: canonical ASTs are identical.
2. **Logical equivalence**: policies agree for every legal input under the declared truth semantics.
3. **Observed-mask equivalence**: policies agree only on the current evaluated sample:

$$
M_{\mathcal X}(A)=M_{\mathcal X}(B).
$$

Observed-mask equality is not logical equivalence and does not establish equal extrapolation, missing-value, transport, or online behavior.

### 5.8.2 Canonical AST

Canonicalization must preserve:

```text
grammar_version
truth_semantics_version
atom semantic identity
threshold edge contract
operand roles
universe requirements
NOT scope and complement metadata
```

For commutative `AND`/`OR` nodes, children may be sorted by full semantic digest only after all role and axis metadata are bound.

Repeated identical children may be removed under idempotence:

$$
A\land A=A,
\qquad
A\lor A=A.
$$

Associative flattening is legal only when operator semantics and node-level role constraints are identical.

Absorption:

$$
A\land(A\lor B)=A,
\qquad
A\lor(A\land B)=A,
$$

and De Morgan rewrites:

$$
\neg(A\land B)=\neg A\lor\neg B,
$$

$$
\neg(A\lor B)=\neg A\land\neg B
$$

may be used only when:

```text
truth_semantics_version is declared and supports the identity
unknown semantics are identical
universe is identical
role annotations and action authorization remain equivalent
```

Strong Kleene truth functions satisfy De Morgan laws, but a rewrite may still be **policy-invalid** if it loses support/necessary-role provenance or changes where action authorization resides.

### 5.8.3 Search deduplication

Search systems must retain separate keys for:

```text
canonical_policy_ast_hash
observed_mask_hash
logical_equivalence_status
```

Mask deduplication may reduce repeated evaluation work, but must not merge semantic policy identities unless logical equivalence is separately established.

---

## 5.9 Degenerate sets and denominator behavior

Every composition audit must report:

```text
n_universe
n_true
n_false
n_unknown
n_gt_exposed
n_fp_exposed
n_gt_hurt
n_fp_removed
rate_status
```

Required statuses:

```text
VALID_RATE
ZERO_GT_DENOMINATOR
ZERO_FP_DENOMINATOR
EMPTY_UNIVERSE
UNKNOWN_CONTAMINATED
BLOCKED_BY_UNIVERSE_MISMATCH
```

Rules:

- `EMPTY_UNIVERSE` is not evidence of safe rejection;
- zero GT exposure makes the GT hurt rate undefined, not zero;
- zero FP exposure makes productivity rate undefined;
- an empty reject set may be safe but fails positive productivity unless `g_min=0` is explicitly authorized;
- an all-reject set receives no special safety shortcut;
- unknown candidates must remain counted even when final action is no-reject.

---

## 5.10 Cross-domain transport boundary

Policies, masks, and atoms from different universes cannot be combined by name alone.

A transport contract must identify:

```text
source_universe_id
target_universe_id
transport_id
candidate key mapping
one-to-zero / one-to-many / many-to-one behavior
unmatched policy
state alignment
label/exposure alignment
signal transformation or recalibration
transport uncertainty / blocked cases
```

A transported predicate is a new evaluated object:

$$
A^{(2)} = T_{1\to2}(A^{(1)}),
$$

not the same mask by default.

Cross-substrate, offline-to-online, pre-hook-to-post-hook, and pre-intervention-to-post-intervention comparisons are transport questions. They are not ordinary Boolean algebra.

---

## 5.11 Stateless observational algebra versus closed-loop intervention

### 5.11.1 Same-state observational composition

Ordinary set algebra assumes predicates are evaluated on one fixed observation and state:

$$
A(x_t,h_t),
\qquad
B(x_t,h_t).
$$

At that same state:

$$
R_{A\lor B,t}=R_{A,t}\cup R_{B,t}.
$$

This supports observational mask analysis.

### 5.11.2 Single-step intervention composition

A single-step intervention claim additionally requires proof that:

```text
the same pre-action state snapshot was used
all operands were evaluated before any action mutated state
action ordering is declared
candidate regeneration does not occur between operands
```

A replay that only scores frozen rows proves observational composition, not necessarily executable one-step composition.

### 5.11.3 Closed-loop policy composition

In online MOT, action changes state:

$$
h_{t+1}=F(h_t,x_t,a_t),
$$

which changes future candidates and signals:

$$
\Omega_{t+1},\Omega_{t+2},\ldots
$$

Therefore:

$$
\operatorname{Outcome}(A\lor B)
\ne
\operatorname{Outcome}(A)
\cup
\operatorname{Outcome}(B)
$$

in general.

Boolean set algebra does not establish:

```text
closed-loop trajectory equivalence
causal additivity
online safety
future candidate-universe equality
intervention effect retention
```

Studies must distinguish:

| Level | Meaning | Minimum evidence |
|:--|:--|:--|
| observational mask composition | same frozen rows/state | candidate-level masks and joint audit |
| single-step intervention composition | action at same pre-state | executable hook or faithful one-step replay |
| closed-loop policy composition | actions alter subsequent state | online/shadow trajectories with controls and state audit |

Offline replay that does not reconstruct intervention-mutated state cannot establish closed-loop composition.

---

## 5.12 Minimum executable audit schema

The following fields are the minimum reconstruction contract for a Boolean policy or grammar result.

### Predicate and threshold

```text
predicate_id
predicate_domain
predicate_codomain                  # {T,F} or {T,F,U}
unknown_value_policy
final_unknown_action
signal_identity
signal_unit
atom_role
comparator
endpoint_policy
tie_policy
nan_policy
posinf_policy
neginf_policy
missing_value_policy
threshold_index
threshold_value_repr
quantile_method
floating_point_tolerance
clipping_domain
```

### Universe and state

```text
universe_id
universe_hash
substrate_id
hook_id
candidate_builder_id
prefilter_contract_id
candidate_key_schema
label_exposure_contract_id
state_snapshot_identity
observation_time_range
transport_id                         # nullable only for same-universe studies
```

### Grammar and semantics

```text
grammar_version
truth_semantics_version
operator_precedence
maximum_nesting_depth
maximum_operands_per_node
not_scope
mixed_role_policy
policy_ast
canonical_policy_ast
canonical_policy_ast_hash
role_validity_result
complement_contract_id               # required for NOT
```

### Observed decisions and joint composition

```text
observed_mask_hash
n_true
n_false
n_unknown
n_gt_exposed
n_fp_exposed
n_gt_hurt
n_fp_removed
composition_audit_joint_hurt
joint_hurt_overlap_count
joint_fp_overlap_count
rate_status
logical_equivalence_status
observed_mask_equivalence_status
```

### Execution level and claims

```text
composition_level                    # observational | single_step | closed_loop
predecision_state_same
candidate_regeneration_policy
intervention_order
claim_level
maximum_claim
forbidden_promotions
production_forbidden
```

Missing required fields must produce an invalid or blocked policy record, not an inferred default.

---

## 5.13 PR and evaluator validity checks

A Boolean-composition PR is contract-valid only when checks can establish:

```text
[ ] Ω and Θ objects are typed and never joined implicitly
[ ] universe_id and pre-decision state match across operands
[ ] predicate codomain and U handling are declared
[ ] NOT never maps U to reject
[ ] threshold comparator, endpoint, tie, NaN, quantile, and tolerance policies are serialized
[ ] zero denominators produce NA/blocked status rather than zero-rate safety
[ ] AST parses under a named grammar version with explicit bounds
[ ] operand roles pass the role-validity matrix
[ ] sufficient OR operands have joint candidate-level hurt audit
[ ] necessary/support operands are not silently promoted to reject authorization
[ ] canonical AST, observed mask, and logical-equivalence identities remain separate
[ ] cross-universe comparisons name a transport contract
[ ] observational, single-step, and closed-loop claims are separated
[ ] evaluator output contains the minimum audit schema
[ ] no evidence level is promoted beyond the execution/evidence actually run
```

A PR may be engineering-ready while the research claim remains blocked or inconclusive.

---

## 5.14 Compatibility and authorization boundary

This contract is immediately applicable to studies with:

```text
fixed declared universe
frozen pre-decision state
total or explicitly three-valued tail atoms
no unauthorized NOT
reject-only observational AND/OR
candidate-level exact mask and hurt audit
```

It provides the semantics needed to design, but does not itself authorize:

```text
G7 necessary/support grammar
NOT/complement reject policies
ε>0 OR policies without joint audit
cross-substrate composition without transport
single-step or closed-loop intervention claims
production behavior
```

Those require grammar-specific evidence and later research gates.
