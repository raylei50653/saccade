# Statistical Robust Feasible-Set Estimation under Asymmetric Loss

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-10 -->
<!-- doc-module: cross -->

## 0. Role

This document defines the shared mathematical framework for studying **safe regions** in MOT decision systems.

It is a method and evidence-semantics contract. It does not introduce a new production policy, reopen a closed research line, or promote any existing candidate. Its purpose is to provide one canonical language for reasoning about:

- gate and reject rules;
- association and relink decisions;
- threshold and weight regions;
- Boolean composition;
- cross-sequence and cross-substrate robustness;
- finite-sample uncertainty;
- online retention and production promotion.

The framework name is:

> **Statistical Robust Feasible-Set Estimation under Asymmetric Loss**

The central object is not an optimal parameter point. It is an estimated feasible set whose safety, productivity, thickness, and transfer properties must be established under asymmetric error costs.

---

## 1. Motivation

Many decision-layer studies are naturally written as threshold or rule search:

$$
\max_{\theta \in \Theta} G_{\mathrm{FP}}(\theta)
\quad
\text{subject to}
\quad
L_{\mathrm{GT}}(\theta) \le \varepsilon
$$

This formulation is necessary but incomplete.

A single optimum can be:

- sample-specific;
- adjacent to an unsafe boundary;
- supported by only one sequence;
- duplicated across many threshold coordinates but equivalent as a decision mask;
- invalid after hook placement changes;
- invalid online despite looking safe offline.

Therefore the research target is not merely a best point $\theta^\star$. It is the structure of the feasible set around and across candidate decisions.

The practical question is:

> Under finite evidence and asymmetric loss, does there exist a non-trivial decision region that remains safe and productive across declared perturbations, folds, sequences, substrates, and execution contexts?

---

## 2. Asymmetric loss

### 2.1 Decision asymmetry

For upstream pruning and reject gates, GT loss and FP retention are not symmetric.

A true candidate removed upstream is often irrecoverable:

$$
\text{GT pruned upstream}
\;\Longrightarrow\;
\text{downstream recovery usually impossible}
$$

A false candidate retained upstream may still be rejected or corrected downstream:

$$
\text{FP retained upstream}
\;\Longrightarrow\;
\text{downstream correction may remain possible}
$$

The framework therefore treats GT damage as the primary feasibility constraint and FP removal as a secondary productivity objective.

### 2.2 Basic quantities

For a decision parameter or policy $\theta$, dataset slice $d$, and observation set $X_d$, define:

$$
L_{\mathrm{GT}}^{(d)}(\theta)
=
\frac{
N_{\mathrm{GT,hurt}}^{(d)}(\theta)
}{
N_{\mathrm{GT,exposed}}^{(d)}
}
$$

and:

$$
G_{\mathrm{FP}}^{(d)}(\theta)
=
\frac{
N_{\mathrm{FP,removed}}^{(d)}(\theta)
}{
N_{\mathrm{FP,exposed}}^{(d)}
}
$$

where exposure counts must be reported whenever the rates are used.

Equivalent count-based constraints may be used when the research contract requires exact zero hurt:

$$
N_{\mathrm{GT,hurt}}^{(d)}(\theta)=0
$$

The notation $L_{\mathrm{GT}}$ is generic. In individual studies it may represent:

- true relinks removed;
- GT associations blocked;
- positive candidates rejected;
- recoverable tracks made unrecoverable;
- sequence-specific identity-support loss.

The study must declare the exact operational meaning.

---

## 3. Feasible and productive-safe sets

### 3.1 Per-domain feasible set

For domain or slice $d$, define the $\varepsilon$-feasible set:

$$
\mathcal F_{\varepsilon}^{(d)}
=
\left\{
\theta \in \Theta:
L_{\mathrm{GT}}^{(d)}(\theta)\le \varepsilon
\right\}
$$

This set expresses safety only.

### 3.2 Productive-safe set

A safe point that removes no meaningful negatives is not a useful gate. Let $g_{\min}$ be a declared minimum productivity requirement:

$$
\mathcal S_{\varepsilon,g_{\min}}^{(d)}
=
\left\{
\theta \in \Theta:
L_{\mathrm{GT}}^{(d)}(\theta)\le \varepsilon,
\;
G_{\mathrm{FP}}^{(d)}(\theta)\ge g_{\min}
\right\}
$$

When $g_{\min}$ is omitted, the study must clearly distinguish:

- **safe**: satisfies the GT constraint;
- **productive-safe**: satisfies both GT and productivity constraints.

This distinction must not be collapsed in tables, atlas labels, or conclusions.

### 3.3 Robust feasible set

Let $\mathcal D$ be the declared family of domains:

$$
\mathcal D
=
\{
\text{sequences, folds, strata, substrates, hook placements, runs}
\}
$$

The robust feasible set is:

$$
\mathcal R_{\varepsilon,g_{\min}}
=
\bigcap_{d\in\mathcal D}
\mathcal S_{\varepsilon,g_{\min}}^{(d)}
$$

This is a worst-case intersection. It must not be replaced by an average safe rate unless the study explicitly defines an average-risk objective.

A relaxed robust set may use a domain-risk budget:

$$
\mathcal R_{\varepsilon,g_{\min},\delta}
=
\left\{
\theta:
P_{d\sim \mathcal D}
\left[
\theta \notin
\mathcal S_{\varepsilon,g_{\min}}^{(d)}
\right]
\le \delta
\right\}
$$

Such a relaxation is qualitatively different from exact cross-domain feasibility and must be labeled accordingly.

---

## 4. Parameter spaces and decision representations

The parameter space $\Theta$ may contain different decision forms.

### 4.1 Scalar thresholds

$$
\theta = t
$$

Example:

$$
C_t(x)=\mathbf 1[s(x)>t]
$$

### 4.2 Multi-axis thresholds

$$
\theta=(t_1,t_2,\ldots,t_k)
$$

Example:

$$
C_\theta(x)
=
\mathbf 1[
s_1(x)>t_1
\land
s_2(x)>t_2
]
$$

### 4.3 Weighted rules

$$
\theta=(w,t),\qquad
C_\theta(x)
=
\mathbf 1[
w^\top \phi(x)>t
]
$$

### 4.4 Boolean policies

$$
\theta
=
\{
A_1,\ldots,A_m,\text{composition grammar}
\}
$$

Example:

$$
C_\theta
=
(A_1\land A_2)\lor A_3
$$

### 4.5 Discrete policy identities

For a frozen portable policy, $\theta$ may be a discrete policy object rather than a continuously tunable parameter.

The study must declare whether it is estimating:

- a coordinate region;
- a rule-family region;
- a decision-mask equivalence class;
- a policy identity;
- an online execution region.

These are not interchangeable.

---

## 5. Boolean composition as set algebra

For atomic reject conditions $A$ and $B$:

$$
A\land B
\equiv
A\cap B
$$

$$
A\lor B
\equiv
A\cup B
$$

$$
\neg A
\equiv
\Omega\setminus A
$$

where $\Omega$ is the declared candidate universe.

### 5.1 Operand roles

Every Boolean operand should have a declared semantic role.

#### Sufficient reject condition

A condition $C$ is sufficient for a safe reject claim when:

$$
C\subseteq \mathcal Y_{\mathrm{safe\ reject}}
$$

Operationally, all observations captured by $C$ satisfy the declared GT constraint on the evaluated evidence.

#### Necessary envelope

A condition $N$ is necessary for a target safe region $S$ when:

$$
S\subseteq N
$$

Necessary conditions restrict where valid policies may exist. They are not themselves sufficient reject rules.

#### Support condition

A support condition modifies confidence, scope, or applicability without independently authorizing rejection.

#### Complement condition

A NOT operand requires an explicit universe $\Omega$. A complement is undefined when the candidate universe or missing-value behavior is unspecified.

### 5.2 Composition effects

Under fixed atomic predicates:

- AND usually contracts coverage and may improve purity;
- OR usually expands coverage and may increase GT exposure;
- NOT may create large complement regions whose safety depends critically on the universe definition.

These are set-theoretic tendencies, not universal empirical guarantees.

---

## 6. Region geometry

Region stability must not be summarized by area alone.

### 6.1 Measure

For continuous or normalized parameter spaces:

$$
\mu(
\mathcal S_{\varepsilon,g_{\min}}
)
$$

For a finite registered lattice:

$$
\widehat \mu
=
\frac{
|\mathcal S_{\varepsilon,g_{\min}}|
}{
|\Theta_{\mathrm{registered}}|
}
$$

The denominator must be declared. Raw coordinate counts across grids of different size or resolution are not directly comparable.

### 6.2 Boundary distance

For $\theta\in\mathcal S$:

$$
m(\theta)
=
\operatorname{dist}
\left(
\theta,
\Theta\setminus\mathcal S
\right)
$$

This is a local safety margin to the nearest known unsafe point.

A distance value is meaningful only relative to:

- the coordinate system;
- the registered lattice;
- the metric;
- the edge policy.

### 6.3 Full-neighborhood robustness radius

Define:

$$
r(\theta)
=
\sup
\left\{
r\ge 0:
B_r(\theta)\subseteq\mathcal S
\right\}
$$

For discrete lattices, $B_r$ is the registered graph neighborhood under the declared adjacency relation.

Examples:

- 1D bilateral interval;
- 2D Manhattan neighborhood;
- repeated 4-neighbor erosion;
- Hamming neighborhood for discrete policy grammars.

A point may have positive nearest-unsafe distance under a censored metric while still having zero full-neighborhood radius. The two quantities must not be conflated.

### 6.4 Connected components

Let:

$$
\mathcal S
=
\bigcup_{j=1}^{J}\mathcal C_j
$$

where each $\mathcal C_j$ is a connected component under the declared adjacency.

Report at least:

- component count;
- component size;
- axis span;
- interior-point count;
- edge-touching status;
- full-neighborhood radius distribution.

Long one-cell-wide strips are not equivalent to genuinely thick regions.

### 6.5 Interior

The discrete interior is:

$$
\operatorname{Int}(\mathcal S)
=
\left\{
\theta\in\mathcal S:
B_1(\theta)\subseteq\mathcal S
\right\}
$$

A non-empty coordinate set with empty interior should be described as thin support, isolated points, or boundary strips rather than a robust region.

---

## 7. Coordinate area, mask area, and semantic area

Threshold atlases may contain many coordinates that produce identical decisions.

Let $M(\theta)$ be the decision mask induced by $\theta$. Define the equivalence relation:

$$
\theta_i\sim\theta_j
\iff
M(\theta_i)=M(\theta_j)
$$

### 7.1 Coordinate area

Counts registered threshold coordinates.

$$
A_{\mathrm{coord}}
=
\left|
\mathcal S
\right|
$$

### 7.2 Mask area

Counts unique decision masks:

$$
A_{\mathrm{mask}}
=
\left|
\{
M(\theta):\theta\in\mathcal S
\}
\right|
$$

### 7.3 Semantic or policy area

Counts distinct rule identities or policy semantics after declared normalization.

Coordinate plateaus caused by quantization or unchanged masks must not be interpreted as evidence of decision robustness without mask-level analysis.

Recommended reporting order:

1. raw coordinate support;
2. per-grid unique-mask support;
3. cross-grid or global mask diagnostics;
4. component geometry;
5. neighborhood thickness.

Global mask collapse should be diagnostic unless the study proves that grid identity and substrate semantics are preserved by the quotient.

---

## 8. Statistical set estimation

The observed safe set is an estimate:

$$
\widehat{\mathcal S}_{\varepsilon,g_{\min}}
$$

The target population set is:

$$
\mathcal S^\star_{\varepsilon,g_{\min}}
$$

In general:

$$
\widehat{\mathcal S}_{\varepsilon,g_{\min}}
\neq
\mathcal S^\star_{\varepsilon,g_{\min}}
$$

### 8.1 Zero observed hurt is not zero population risk

If $n$ GT exposures produce zero observed hurt, this establishes:

$$
\widehat L_{\mathrm{GT}}=0
$$

It does not establish:

$$
L_{\mathrm{GT}}^\star=0
$$

A confidence upper bound or equivalent finite-sample statement should be reported when the claim extends beyond the observed sample.

For a simple binomial model, a one-sided upper bound may be written:

$$
P(
L_{\mathrm{GT}}^\star
\le u_\alpha
\mid
N_{\mathrm{GT,hurt}}=0,
N_{\mathrm{GT,exposed}}=n
)
\ge 1-\alpha
$$

The exact interval method must be declared if numerical bounds are used.

### 8.2 Exposure-aware interpretation

The following observations are not equally strong:

```text
0 hurt / 3 GT exposures
0 hurt / 300 GT exposures
0 hurt / 30,000 GT exposures
```

Every GT0 result should carry its exposure denominator or an explicit pointer to it.

### 8.3 Selection bias

If the same data are used to:

1. generate atoms;
2. search compositions;
3. choose thresholds;
4. evaluate the winning policy;

then the resulting set is in-sample and selection-biased.

No amount of coordinate thickness inside the same searched sample removes that bias.

### 8.4 Held-out set retention

For training domains $\mathcal D_{\mathrm{tr}}$ and held-out domain $d_{\mathrm{te}}$:

$$
\widehat{\mathcal S}_{\mathrm{tr}}
=
\bigcap_{d\in\mathcal D_{\mathrm{tr}}}
\widehat{\mathcal S}^{(d)}
$$

Set retention is:

$$
\rho_{\mathrm{set}}
=
\frac{
|
\widehat{\mathcal S}_{\mathrm{tr}}
\cap
\widehat{\mathcal S}^{(d_{\mathrm{te}})}
|
}{
|
\widehat{\mathcal S}_{\mathrm{tr}}
|
}
$$

When parameter spaces differ across folds, a portable coordinate, quantile, policy, or semantic normalization must be defined before computing retention.

### 8.5 Point validation versus region validation

Point validation asks:

$$
\theta^\star
\in
\widehat{\mathcal S}^{(d_{\mathrm{te}})}
\;?
$$

Region validation asks:

$$
B_r(\theta^\star)
\subseteq
\widehat{\mathcal S}^{(d_{\mathrm{te}})}
\;?
$$

or:

$$
\widehat{\mathcal S}_{\mathrm{tr}}
\cap
\widehat{\mathcal S}^{(d_{\mathrm{te}})}
\text{ has non-trivial measure and interior?}
$$

Passing a frozen point does not establish a transferable region.

---

## 9. Robustness axes

Every robust-region claim must name the axes over which robustness is evaluated.

### 9.1 Sequence robustness

$$
\mathcal R_{\mathrm{seq}}
=
\bigcap_{s\in\mathcal S_{\mathrm{seqs}}}
\mathcal S^{(s)}
$$

### 9.2 Fold robustness

$$
\mathcal R_{\mathrm{fold}}
=
\bigcap_{f\in\mathcal F}
\mathcal S^{(f)}
$$

### 9.3 Substrate robustness

A substrate includes the full observation and decision context relevant to the evaluated policy, such as:

- detector;
- model;
- candidate construction;
- score definitions;
- feature extraction;
- missing-value behavior;
- hook placement;
- online state;
- execution ordering.

$$
\mathcal R_{\mathrm{sub}}
=
\bigcap_{z\in\mathcal Z}
\mathcal S^{(z)}
$$

A policy that transfers across sequences but not across hook placement is not substrate-robust.

### 9.4 Perturbation robustness

For perturbation family $\Delta$:

$$
\mathcal R_{\Delta}
=
\left\{
\theta:
\forall \delta\in\Delta,\;
\theta+\delta\in\mathcal S
\right\}
$$

Perturbations may include:

- threshold displacement;
- quantile displacement;
- score calibration drift;
- candidate-pool changes;
- timing or online-state differences.

### 9.5 Execution robustness

Offline replay and online execution may induce different observations or state transitions.

Define:

$$
\mathcal S_{\mathrm{offline}}
\quad\text{and}\quad
\mathcal S_{\mathrm{online}}
$$

Online retention requires:

$$
\theta
\in
\mathcal S_{\mathrm{offline}}
\cap
\mathcal S_{\mathrm{online}}
$$

A correct offline result may remain online-blocked without constituting an engineering failure.

---

## 10. Safe-region claim ladder

Claims must be bounded to the strongest level actually supported.

### L0 — Observed safe point

A single evaluated $\theta$ satisfies the declared constraint on the observed sample.

Required:

- parameter or policy identity;
- substrate;
- exposure counts;
- GT hurt and productivity.

Does not establish:

- neighborhood stability;
- held-out transfer;
- online validity.

### L1 — In-sample safe region

Multiple coordinates or policies form a safe set on the searched sample.

Required:

- parameter space and denominator;
- coordinate or mask area;
- adjacency and boundary policy;
- component geometry.

Does not establish:

- post-selection generalization;
- LOO retention.

### L2 — Held-out retained point

A frozen point passes held-out, LOO, or LOSO evaluation.

Required:

- freeze before test;
- fold definitions;
- per-fold hurt and productivity;
- aggregation rule.

Does not establish:

- retained region thickness.

### L3 — Held-out retained region

A non-trivial region or neighborhood remains feasible across held-out evaluations.

Required:

- portable coordinate or policy representation;
- set-retention metric;
- non-zero shared measure;
- neighborhood or interior evidence;
- per-fold exposure.

### L4 — Cross-substrate portable region

The region survives declared substrate changes.

Required:

- substrate identity for each evaluation;
- signal and hook semantic equivalence;
- no silent score or candidate-universe drift;
- shared policy interpretation.

### L5 — Online-retained region

The policy or region survives online execution.

Required:

- actual online hook placement;
- default-off A/B;
- applied/rejected audit;
- online state provenance;
- baseline behavior unchanged when disabled.

### L6 — Production-safe candidate

A policy may be considered for production promotion.

Required in addition to L5:

- production-specific evaluation contract;
- latency and failure-mode audit;
- rollback and default behavior;
- evidence-ledger promotion;
- explicit acceptance review.

Engineering merge alone does not advance a claim up this ladder.

---

## 11. Minimum audit outputs

A safe-region study should emit enough information to reconstruct the claim.

### 11.1 Study identity

```text
study_id
code revision
data revision
substrate identity
candidate universe
signal definitions
parameter-space definition
```

### 11.2 Safety and productivity

```text
n_gt_exposed
n_gt_hurt
gt_hurt_rate
n_fp_exposed
n_fp_removed
fp_removed_rate
epsilon
g_min
```

### 11.3 Geometry

```text
registered_coordinate_count
productive_safe_coordinate_count
unique_mask_count
component_count
interior_count
edge_touching_count
nearest_unsafe_distance distribution
full_neighborhood_safe_radius distribution
```

### 11.4 Transfer

```text
per-sequence results
per-fold results
set-retention ratio
shared-region measure
shared interior or radius
portable policy identity
```

### 11.5 Execution boundary

```text
offline status
online status
hook placement
default-off status
baseline-disabled equivalence
production preset status
```

---

## 12. Decision rules for interpretation

### 12.1 Thick region

A region may be described as thick only when it has a non-trivial interior under the declared metric and edge policy.

Coordinate count alone is insufficient.

### 12.2 Stable plateau

A plateau is stable only after determining whether it represents:

- repeated threshold coordinates;
- repeated masks;
- genuinely distinct nearby decisions.

A mask-equivalent plateau is calibration-insensitive on the observed sample, but not necessarily distributionally robust.

### 12.3 Portable region

A region is portable only when the parameter or policy semantics remain invariant across domains.

Per-dataset quantiles are not automatically portable thresholds.

### 12.4 Robust point

A robust point should have:

- held-out safety;
- meaningful GT exposure;
- positive productivity;
- positive perturbation or neighborhood margin where the parameterization permits it.

### 12.5 Empty or thin interior

If:

$$
\operatorname{Int}(\mathcal S)=\varnothing
$$

the appropriate conclusion is bounded:

```text
isolated safe points
thin strips
edge-supported coordinates
mask plateaus without decision thickness
```

It is not a formal robust safe region.

---

## 13. Forbidden inference shortcuts

The following implications are invalid unless separately established:

$$
\text{high AUC}
\not\Rightarrow
\text{safe reject rule}
$$

$$
\text{in-sample GT0}
\not\Rightarrow
\text{population risk 0}
$$

$$
\text{many safe coordinates}
\not\Rightarrow
\text{many distinct safe decisions}
$$

$$
\text{coordinate plateau}
\not\Rightarrow
\text{region thickness}
$$

$$
\text{LOO-safe frozen point}
\not\Rightarrow
\text{LOO-retained region}
$$

$$
\text{offline safe}
\not\Rightarrow
\text{online effective}
$$

$$
\text{online default-off hook works}
\not\Rightarrow
\text{production promotion}
$$

$$
\text{engineering merge}
\not\Rightarrow
\text{research acceptance}
$$

---

## 14. Relationship to optimization

Optimization remains useful inside this framework.

### 14.1 Constrained objective

$$
\max_{\theta}
G_{\mathrm{FP}}(\theta)
\quad
\text{s.t.}
\quad
L_{\mathrm{GT}}(\theta)\le\varepsilon
$$

### 14.2 Robust constrained objective

$$
\max_{\theta}
\min_{d\in\mathcal D}
G_{\mathrm{FP}}^{(d)}(\theta)
\quad
\text{s.t.}
\quad
\max_{d\in\mathcal D}
L_{\mathrm{GT}}^{(d)}(\theta)
\le\varepsilon
$$

### 14.3 Region-aware objective

A region-aware score may prefer candidates with both utility and margin:

$$
J(\theta)
=
G_{\mathrm{FP}}(\theta)
+
\lambda r(\theta)
$$

subject to the GT constraint.

This does not replace direct geometry reporting. Any scalarization hides trade-offs and must not become the sole evidence.

### 14.4 Pareto frontier

When safety, productivity, and robustness conflict, report the Pareto set over:

$$
\left(
L_{\mathrm{GT}},
-G_{\mathrm{FP}},
-r,
-\mu_{\mathrm{shared}}
\right)
$$

A single chosen operating point should be justified relative to that frontier.

---

## 15. Relationship to existing project evidence

This framework is intentionally general. Existing studies serve as examples, not as definitions.

### 15.1 M-B1 repaired OR-tail

The repaired portable OR-tail line is an example of:

- asymmetric GT-first feasibility;
- atom repair after LOO hurt attribution;
- frozen point validation;
- offline region-thickness analysis;
- separation between offline acceptance and online execution.

It should be cited as a concrete study instance, not as the universal definition of a robust region.

### 15.2 Q4.5 composition atlas

The Q4.5 atlas is an example of:

- registered threshold lattices;
- coordinate versus mask support;
- component geometry;
- conservative full-neighborhood radius;
- bounded closure when no non-trivial interior exists.

Its accepted conclusion remains specific to the registered atlas and does not establish a formal portable safe region.

### 15.3 Online hook studies

Online hook studies are examples of:

- substrate and hook-placement dependence;
- default-off intervention;
- baseline-disabled equivalence;
- distinction between engineering readiness and research acceptance.

---

## 16. Review contract

A PR that uses safe-region language should be reviewed against four independent gates.

### Gate A — Mathematical identity

- Is $\Theta$ defined?
- Is the candidate universe defined?
- Are safety and productivity quantities defined?
- Is the metric or adjacency declared?
- Are coordinate and mask units distinguished?

### Gate B — Statistical validity

- Are GT exposures reported?
- Was the policy frozen before held-out evaluation?
- Is selection bias acknowledged?
- Are per-fold or per-sequence failures visible?
- Is GT0 interpreted within finite evidence?

### Gate C — Substrate validity

- Are data, score, candidate, and hook semantics unchanged?
- Is any normalization portable?
- Is offline/online drift measured?
- Is missing-value behavior fixed?

### Gate D — Claim boundary

- Which claim-ladder level is supported?
- Which stronger claims remain blocked?
- Is engineering merge separated from research acceptance?
- Is production promotion explicitly unchanged unless separately authorized?

A PR may pass engineering review while remaining statistically or scientifically inconclusive.

---

## 17. Recommended terminology

Use:

```text
feasible set
productive-safe set
robust intersection
estimated safe set
held-out retained point
held-out retained region
coordinate plateau
mask-equivalent plateau
full-neighborhood radius
boundary margin
thin component
isolated safe points
cross-substrate retention
online retention
```

Avoid unqualified use of:

```text
stable
robust
safe region
portable
production-safe
```

These terms must name the relevant axis and evidence level.

Preferred examples:

```text
LOO-retained offline point
multi-sequence productive-safe coordinate set
mask-equivalent threshold plateau
zero registered full-neighborhood thickness
online-retained default-off policy
```

---

## 18. Canonical conclusion form

A safe-region study should end with a bounded conclusion in this form:

```text
Within <declared substrate and parameter space>,
under <epsilon, productivity threshold, metric, and edge policy>,
the observed productive-safe support is <point / thin set / thick region>.

It is retained across <declared folds, sequences, or substrates> at
<point or region level>.

Maximum supported claim:
  <claim-ladder level>

Not established:
  <explicit stronger claims>

Production preset:
  unchanged | separately authorized
```

---

## 19. Summary

The framework treats safe decision research as the estimation of a constrained feasible set under asymmetric costs:

$$
\widehat{\mathcal S}_{\varepsilon,g_{\min}}
=
\left\{
\theta:
\widehat L_{\mathrm{GT}}(\theta)\le\varepsilon,
\;
\widehat G_{\mathrm{FP}}(\theta)\ge g_{\min}
\right\}
$$

Robustness is then evaluated through:

$$
\text{shared feasibility}
+
\text{region measure}
+
\text{boundary margin}
+
\text{neighborhood thickness}
+
\text{held-out retention}
+
\text{substrate retention}
$$

The key research shift is:

> Do not optimize a point and call it stable. Estimate the feasible set, measure its geometry, validate its transfer, and bound the claim to the strongest evidence actually closed.
