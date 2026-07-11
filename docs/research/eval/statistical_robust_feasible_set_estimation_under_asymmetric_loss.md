# Statistical Robust Feasible-Set Estimation under Asymmetric Loss

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-12 -->
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

When applied to estimated sets, the intersection is anti-conservative rather than conservative: a domain with little GT exposure excludes almost nothing, so the weakest-evidence domain contributes the least constraint, and

$$
\bigcap_{d\in\mathcal D}
\widehat{\mathcal S}^{(d)}
$$

is not an estimate of $\bigcap_{d}\mathcal S^{\star(d)}$. A domain's feasibility verdict may enter the robust intersection only if that domain's GT exposure meets a declared minimum $n_{\min}$. Domains below the minimum must be reported as insufficient evidence, not as passes (see §8.2).

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

The censoring policy for unevaluated lattice points must also be declared: whether a missing neighbor truncates the radius (conservative — the ball stops at the last fully evaluated shell, and radius beyond it counts as 0) or renders the radius unknown. Reported radii must state which convention is in force, and the two conventions must not be mixed within one table.

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

Any numerical bound additionally requires a declared **independence unit**. GT exposures in tracking data are typically clustered: one track contributes many frame-level exposures, so $n$ frame-level exposures may represent far fewer effectively independent trials. A binomial or rule-of-three bound computed on frame-level counts is anti-conservative under such clustering.

Required declaration:

- the exposure unit treated as an independent trial (candidate, event, track, sequence);
- the clustering structure relating the raw exposure count to that unit;
- when raw counts are clustered, either aggregate to the independence unit before bounding or use a cluster-aware method. The effective sample size must not silently equal the raw count.

Two further cautions:

- a unique-cluster count is an upper bound on the number of independent trials, not automatically an effective sample size; residual clustering above the declared unit (for example sequence-level shared scene and pipeline state) must be named;
- when observed hurt is non-zero, hurt outcomes must be aggregated to the same trial unit before the cluster count may serve as the binomial denominator.

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

The operational rule is per-fold disjointness: for each held-out fold $f$, every operation that determines the policy evaluated on $f$ — atom generation, composition search, threshold choice — must exclude $f$. Two protocols satisfy this:

- a single frozen policy whose selection data are disjoint from the union of all test folds (independent development set);
- nested per-fold selection, where each fold's policy is selected on that fold's training data only and evaluated once on its held-out fold.

Selecting atoms once on the full pool and then evaluating under LOO satisfies neither — every held-out fold has already influenced the selection.

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

Retention has no meaning without a triviality baseline. If $\widehat{\mathcal S}_{\mathrm{tr}}$ covers most of $\Theta$ because the constraints are weak, high $\rho_{\mathrm{set}}$ is guaranteed rather than evidential. Report alongside $\rho_{\mathrm{set}}$:

- the training-set fraction $|\widehat{\mathcal S}_{\mathrm{tr}}|\,/\,|\Theta_{\mathrm{registered}}|$;
- a size-matched null: the retention expected for a random coordinate set with the same cardinality as $\widehat{\mathcal S}_{\mathrm{tr}}$;
- the full-lattice baseline, reported as a trivial ceiling — it is not a size-matched null.

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
- per-fold selection disjointness (§8.3): the data determining each evaluated policy exclude its own test fold;
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
gt_exposure_unit_raw     (raw counting unit, e.g. row / candidate)
declared_trial_unit      (candidate / event / track / sequence)
n_gt_exposed_clusters    (unique clusters at the declared trial unit; null if metadata incomplete)
independence_assumption  (what is assumed independent; remaining clustering named)
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
per-domain GT exposure vs declared minimum
set-retention ratio
training-set fraction of registered space
null-reference retention baseline
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
- Is the independence unit for any numerical bound declared (§8.1)?
- Was the policy frozen before held-out evaluation?
- For each held-out fold, do the data determining its evaluated policy exclude that fold (§8.3)?
- Is selection bias acknowledged?
- Are per-fold or per-sequence failures visible?
- Is GT0 interpreted within finite evidence?
- Do all domains entering a robust intersection meet the declared minimum exposure (§3.3)?

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

### Gate E — Role alignment (§20)

- Is exactly one primary system role declared?
- Is the design objective legal for that role (§20.3)?
- Is the selection rule the ordered form of §20.4, not "best performer"?
- Are stop conditions predeclared (§20.6)?
- Is every cited result classified into a §20.4 output class?
- Does any promotion of a diagnostic or upper-bound result violate §20.5?

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

## 19. GT-support morphology on Boolean atom lattices (predeclared procedure)

### 19.0 Status

Procedure version **v1 — PROPOSED (calibrated on pooled Step-0 audit; seal = research-owner acceptance via the procedure PR merge)**. Until sealed, results computed under it are exploratory. Studies must cite the procedure version they ran under. Evidence order is normative: calibrated step-0 → procedure seal → escape-tail forensics → nested confirmation; forensic results must not enter the sealing PR, so that mechanism information cannot feed back into the boundaries.

### 19.1 When this section applies

For a Boolean atom lattice $z\in\{0,1\}^k$ over a candidate universe, the per-cell risk field

$$
E(z)=P(\text{error}\mid Z=z)
$$

is **identifiable only where cell-level GT exposure meets the §3.3 / §8.1 requirements**. A study must first run an occupancy audit (cells reaching the declared per-cell $n_{\min}$ at the declared trial unit). If fewer than a declared fraction of cells are identifiable, the study no longer estimates the risk field. The **GT placement distribution**

$$
\mu_{\mathrm{GT}}(z)=P(Z=z\mid \text{GT trial unit})
$$

together with the FP mass $\mu_{\mathrm{FP}}(z)$ remains a **descriptive placement layer** for morphology and candidate construction. Formal decision feasibility is evaluated at the declared independent trial unit. Let $Z_u$ be all valid GT cells owned by unit $u$; the decision problem becomes:

$$
\max_D \; P_{\mathrm{FP}}(Z\in D)
\quad\text{s.t.}\quad
\operatorname{UCB}_\alpha\!\left[P_u(Z_u\subseteq D)\right]\le \varepsilon
$$

with the UCB computed at the §8.1 independence unit. Equivalently, for the complementary safe closure $C=\Omega\setminus D$, the hurt indicator is $H_C(u)=\mathbf 1[Z_u\cap C=\varnothing]$. The descriptive placement mass $\mu_{\mathrm{GT}}(z)$ must not be substituted for this unit-level hurt probability. Merge-tree, barrier-height, and per-cell UCB analyses are **out of mainline** below the declared identifiability density.

### 19.2 Hard interpretation rule

$$
\text{not observed} \neq \text{unsafe}
$$

Cells with zero GT exposure are **unresolved**, never "high-risk", never "barriers". A gap in $\mu_{\mathrm{GT}}$ support is evidence about where GT mass does not go only to the extent bounded by the placement UCB; it is not evidence about conditional risk in that cell.

### 19.3 Required declarations (before computing)

```text
atom set + per-atom safe orientation (z_i = 1 == safer side)
binarization rule (sealed thresholds, or per-fold rule)
trial unit (§8.1), clustering structure, and the UCB method
  (incl. cluster handling when residual clustering is declared)
candidate universe Omega + missing-value behavior
epsilon, alpha, per-cell n_min, identifiability fraction
atom grouping for distance decomposition (e.g. structural vs motion)
morphology budget epsilon_morph + interval method (§19.5)
allowed closure complexity class (which up-set families C may range over)
closure direction convention + deterministic tie-break order
conditionable atom family for escape tails (e.g. motion)
```

For any claim above L1, atom discovery, orientation, and binarization must satisfy §8.3 per-fold disjointness.

### 19.4 Fixed morphology statistics and trial semantics

#### Set-valued trial semantics (normative)

A GT trial unit $u$ generally owns **multiple** valid GT rows. Define:

$$
Z_u = \{\,\text{cells of all valid GT rows of } u\,\}
$$

For any candidate safe closure $C$, track-level hurt is **set-valued**:

$$
H_C(u) = \mathbf 1\left[\, Z_u \cap C = \varnothing \,\right]
$$

i.e. a unit is hurt only when the closure retains **none** of its valid GT candidates. All formal core/closure feasibility statistics must use $H_C(u)$.

The **minimum-$d_H$ representative** (the unit's Hamming-closest cell to the all-safe corner $\mathbf 1$) is a **descriptive statistic only** — it may be used for shell profiles, never as the trial representation for closure validation: the Hamming-closest row need not be the safest row under the declared partial order, and ties can arbitrarily flip closure membership.

#### Descriptive statistics

```text
M_0        = P_GT(d_H = 0)                     corner mass
M_r, T_>=r = P_GT(d_H = r), P_GT(d_H >= r)     shell / tail profile
V_i        = P_GT(z_i = 0)                     atom violation profile
V_ij       = P_GT(z_i = 0, z_j = 0)            joint violation profile
d_H        = d_structural + d_motion           declared-group decomposition
```

(all computed on the min-$d_H$ representative; descriptive layer only.)

FP mass is reported on the same shells (rejectable material per shell). A tail identified by Hamming distance is a **far-Hamming descriptive tail**; the term *out-of-core GT mass* is reserved for $\{u : H_{C^\star}(u)=1\}$ after the core $C^\star$ has actually been solved.

### 19.5 Verdict typology (fixed terminals) and class boundaries

#### Morphology budget

Class boundaries bind to the asymmetric-loss budget and a confidence upper bound — **not** to raw shell fractions or fixed Hamming radii, which depend on $k$, atom correlation, and the binarization rule.

$$
\varepsilon_{\mathrm{morph}} = 5\%
$$

evaluated as a **one-sided 95% upper confidence bound at the §8.1 trial unit**. $\varepsilon_{\mathrm{morph}}$ governs morphology classification (thin tail vs diffuse) only; it is **not** a production GT-hurt budget (0 / 0.1% / 1% budgets are separate contracts).

#### UCB validity (cluster condition)

A Clopper–Pearson bound is valid **only when the declared trial units are independent Bernoulli trials**. When the study declares residual clustering above the trial unit (per §8.1 — e.g. sequence-level shared scene and pipeline state), the unique-unit count is only an upper bound on the number of independent trials, and a plain CP bound is anti-conservative. In that case the study must either:

- use a **cluster-aware bound** (e.g. aggregate to the clustering level, or a declared cluster-robust method), or
- further aggregate the trial unit until the independence assumption is defensible.

A plain CP number computed under declared residual clustering may be reported only as a **nominal diagnostic** (`nominal; not cluster-adjusted`) and **must not be used to cross the $\varepsilon_{\mathrm{morph}}$ boundary**; a terminal that would depend on it stays `UNRESOLVED`.

#### Core definition

The core is **not** a fixed Hamming radius and **not** bare set-inclusion minimality. Fix the safe-orientation partial order ($z' \ge z$ = coordinate-wise at least as safe); let $\mathcal C$ be the declared family of **upper sets (up-sets)** over the morphology-supported atoms, subject to the declared complexity cap; the reject domain $D = \Omega \setminus C$ is the complementary down-set. The core is:

$$
C^\star \in
\arg\min_{C \in \mathcal C}
P_{\mathrm{FP}}(Z \in C)
\quad
\text{s.t.}
\quad
\operatorname{UCB}\!\left[\, P\!\left(H_C(u)=1\right) \,\right]
\le \varepsilon_{\mathrm{morph}}
$$

with $H_C(u)$ the set-valued hurt of §19.4 and the UCB satisfying the validity conditions above. Minimizing retained FP mass makes $D^\star = \Omega \setminus C^\star$ the maximal-FP-removal reject domain of §19.1 under the same constraint — the two problems are complementary by construction.

**Deterministic tie-breaks** (applied in order among constraint-feasible minimizers): (1) smaller registered cell count $|C|$; (2) lexicographically smallest sorted cell-index sequence. The candidate universe $\Omega$ and missing-value behavior must be declared (§19.3). The lexicographic sequence is a total order on finite cell-index sets; if these tie-breaks do not yield a unique result, the indexing or tie-break declaration is incomplete and must be repaired before reporting a core.

Hamming shell quantities ($M_r$, $T_{\ge r}$, $R_{95}$) remain **descriptive** reporting and must not serve as cross-atom-set class boundaries.

#### Terminals

Each study must land on exactly one verdict:

```text
MONOTONE_CORE
  A predeclared monotone closure C has been SOLVED (per the core
  definition above) with a VALID (cluster-condition-satisfying)
  UCB[P(H_C(u)=1)] <= epsilon_morph, and out-of-core units
  {u : H_C(u)=1} do not form a repeatable, mechanism-consistent
  true GT regime.

CORE_PLUS_CONDITIONAL_ESCAPE_TAIL
  Such a closure C has been solved, 0 < #{u : H_C(u)=1}, and a VALID
  UCB[P(H_C(u)=1)] <= epsilon_morph, AND forensics (post-seal) confirm:
    - the tail is true GT (not annotation / signal-computation issues);
    - violations concentrate on the predeclared conditionable family
      (e.g. motion) while structural/height conditions are retained;
    - intervention = remove the violated partial-order dimension or
      regime-condition it — NEVER veto the tail (protected GT mass).
  With very small tails, do not invent precision ratios (e.g. "75%
  consistent"): enumerate per event; require no mutually conflicting
  confirmed mechanisms; mixed or unresolved forensics block promotion.

DIFFUSE_OR_NONMONOTONE
  No allowed low-complexity closure achieves the epsilon_morph UCB, or
  the required atom family / orientations flip persistently across
  nested folds.

UNRESOLVED
  Any of: no valid (cluster-condition-satisfying) UCB is available at
  the declared trial unit; the confidence bound straddles epsilon_morph;
  the core C has not been solved under the declared partial order;
  exposure for a key family is insufficient; the terminal is unstable
  under sealed thresholds; nested folds yield mutually exclusive
  verdicts; forensics leave unexcluded annotation or signal-computation
  issues.
  Verdict is "collect exposure / resolve the blocker", not "search rules".
  An UNRESOLVED study may still record a bounded DESCRIPTIVE MORPHOLOGY
  HYPOTHESIS (e.g. shell profile + violation profile) without terminal
  force.
```

Escape-tail forensics must classify each tail unit into predeclared categories only:

```text
true long-occlusion re-entry | annotation issue |
signal computation issue | threshold artifact | unresolved
```

### 19.6 Evidence order

```text
step-0 occupancy + placement audit (descriptive, L1 ceiling)
→ owner seals procedure (this section, versioned)
→ escape-tail forensics (predeclared categories)
→ nested per-fold rerun of the full chain
  (atom discovery, orientation, binarization, verdict)
```

A pooled in-sample audit may inform the choice of class boundaries; when it does, the study must say so, and the confirmatory unit is the nested per-fold rerun, not the pooled audit.

---

## 20. Role-aligned experiment contract (normative)

### 20.0 Status and scope

Contract version **v1 (2026-07-12)**. This section is the normative home of the experiment contract. Issue threads, study notes, and PR descriptions must **reference** this section; they must not restate or fork it. Every new decision-layer study that uses this framework's language or infrastructure runs under this contract. Studies opened before v1 keep their sealed procedures but must be re-classified under §20.4 before any result is cited as a design recommendation.

### 20.1 Why this contract exists

The constrained objective of §1 / §14,

$$
\max_{D} \; P_{\mathrm{FP}}(D)
\quad\text{s.t.}\quad
P_{\mathrm{GT}}(D)\le\varepsilon,
$$

is a **capability-exploration tool**: it probes signal upper bounds, maps GT-support boundaries, and surfaces failure modes. Left as the implicit default it drifts into a selection criterion, and local data optima start substituting for module responsibility. What the tooling can resolve is not what the module should do. Project evidence already shows the failure mode of boundary-hugging offline optima: probe thresholds that do not transfer across distributions, accepts that cannot be fixed as defaults across runs, and uniformly harmful monotone interventions under streaming substrates. Such optima lack structural margin and are not portable; this contract prevents them from becoming designs by default.

### 20.2 Required declarations (before running)

```text
System role      exactly one primary role:
                   coarse gate / score-ranking / assignment / calibration /
                   capability map / boundary diagnostic / performance upper bound
                 (secondary uses may be listed, but one primary role governs
                  objectives, metrics, and selection)
Design objective role-legal per §20.3; "maximize FP removed" alone is invalid
Selection rule   the ordered criteria of §20.4; "best performer" alone is invalid
Stop condition   predeclared sufficiency and futility stops per §20.6
Output class     which §20.4 classes the study's results may claim
```

### 20.3 Role-legal design objectives

**Coarse gate.** Subject to a minimum-utility threshold, find a **large-margin, monotone, structurally simple, mechanism-interpretable obvious-negative region** — candidates that are geometrically impossible, temporally implausible, scale/position-degenerate, or in extreme low-credibility tails. The gate question is *where should the gate stop*, not *how far can the gate safely reach*. Gates deliberately retain the ambiguous band for downstream layers; a gate is not required to approximate the full decision boundary.

**Score / ranking.** Within the retained ambiguous band, establish whether conditional interactions **stably and interpretably improve the relative ordering** of GT versus FP candidates. Primary metrics are event-local ranking quantities: pairwise ranking accuracy, GT rank / reciprocal rank, top-k GT recall, positive–negative score margin, assignment-flip attribution, per-sequence interaction stability, LOO interaction retention, online MOT retention. FP-removed and GT-hurt remain reportable but are not score-study objectives. Weak signals that are individually inconclusive but jointly informative (e.g. large distance ∧ long gap ∧ weak motion consistency) belong here as interaction terms $s(x)=s_0(x)+\Delta s_{\mathrm{condition}}(x)$, not as hard rejections.

**Capability map / boundary diagnostic / performance upper bound.** The §14 constrained-optimization forms are the appropriate instruments here. Outputs are bounded by §20.4 classes and §20.5: they describe what the signal family can resolve and where it stops, and they never carry design authority on their own.

The gate and score layers share one substrate — raw signals → normalized measurements → Boolean atoms and conditions → statistical validation → role-specific decision. The gate reads only the extreme, monotone, large-margin projection and emits hard reject/retain; the score layer reads interactions and conditional reliability and emits relative preference. Complex Boolean interaction structure belongs to the score/decision layer, not the coarse gate.

### 20.4 Output classes and selection rule

Every result must be classified as exactly one of:

- **design candidate** — purpose-aligned, mechanism-interpretable, structurally simple, stability-validated, utility above the declared threshold;
- **performance upper-bound candidate** — highest attainable utility under the permitted complexity class; documents capability, not design;
- **diagnostic result** — capability map, boundary morphology, identifiability verdict, failure-mode or exceptional-tail attribution;
- **unexplained residual set** — regions the declared signal family cannot stably resolve; recorded as open problems, never force-fitted with additional conditions.

Design candidates are selected in this order:

1. **purpose alignment** — the rule solves a declared module responsibility;
2. **mechanism interpretability** — each atom's reason for existing, direction consistency with tracking mechanism, clear interaction semantics, failure localizability;
3. **structural simplicity** — fewer atoms, monotone logic, no special zones, splits, or exception patches;
4. **stability** — per-sequence consistency, LOO retention, boundary margin, insensitivity to small threshold shifts, no dependence on rare samples;
5. **utility as a threshold condition** — minimum bars on ranking improvement, coverage, GT risk, and online retention; never the ranking objective among surviving candidates.

Among candidates that clear the utility bar, prefer the simplest, most stable, best-explained one. The highest-utility candidate never auto-promotes to design candidate.

### 20.5 Promotion constraint

A result classified as *diagnostic* or *performance upper bound* must not be promoted — directly or by re-labeling — into a production design recommendation. Promotion requires a new evaluation declared under the target role, passing that role's legal objective (§20.3) and the full selection order (§20.4). Re-labeling is not promotion. This constraint is what prevents the highest-utility candidate from re-entering the gate under a new name.

### 20.6 Stop conditions

Sufficiency stops (enough): a clearly large-margin region has been found; the declared minimum utility is met; additional conditions yield only marginal gains.

Futility stops (mandatory): stop when

- marginal gains come mainly from boundary hugging;
- new conditions mainly describe rare tails of a single sequence;
- LOO or cross-run margin is not retained;
- interactions fail to improve ranking (score-role studies);
- complexity growth exceeds interpretability gains.

The governing principle:

> Define the module responsibility and design purpose first, then design the experiment. Optimization serves the purpose; it does not define it.

---

## 21. Summary

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
