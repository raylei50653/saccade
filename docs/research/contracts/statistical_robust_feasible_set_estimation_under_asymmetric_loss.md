# Statistical Robust Feasible-Set Estimation under Asymmetric Loss

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-12 -->
<!-- doc-module: cross -->

## 0. Role

> **Related contracts** ([index](README.md)) — editorial cross-reference added 2026-07-12; no semantic change to this framework.
>
> - **[runtime-quantity fidelity protocol](runtime_quantity_fidelity_protocol.md)** — its core lemma (*same \(f\), different temporal reduction \(R\)*) is **the same phenomenon as § 9.3 substrate robustness**, viewed from the quantity side. **A change of coordinate provenance (offline reconstruction → runtime kernel term) is a substrate change**, so a region's \(L_{\mathrm{GT}}\) bound proved on offline coordinates does **not** transfer: it must earn rung **L4** (§ 10), and § 13's *offline safe \(\not\Rightarrow\) online effective* applies directly.
> - **[signal_table_schema § 0.5](signal_table_schema.md)** — decides whether a question is a **gate** (membership) or a **score** (ordering) before this framework's machinery is applied. A gate is not required to discriminate.
> - **[boolean_composition_semantics_contract](boolean_composition_semantics_contract.md)** — semantics for composed rules. **[safe_region_asset_contract](safe_region_asset_contract.md)** — packaging of sealed evidence into a RegionAsset.

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

- Are the target decision layer and one primary study intent declared as separate axes (§20.2)?
- Is the objective legal for the declared layer and intent (§20.3)? For assignment / calibration targets: is the intent diagnostic-only?
- Is the selection rule the ordered form of §20.4, not "best performer"?
- Is the observation-validity / identifiability gate predeclared, and is UNRESOLVED distinguished from the futility terminal (§20.7)?
- Are stop conditions predeclared (§20.6)?
- Is every cited result classified into a §20.4 output class?
- Does any promotion of a diagnostic or upper-bound result violate §20.5?
- If the study claims mainline cadence: does every predeclared terminal outcome produce a state transition (§20.7)?
- Does the declaration pass the §20.8 seal bar (frozen degrees of freedom, mechanical decidability, exhaustive terminal mapping, scoped exhaustion naming, joint headroom, blind→reveal hash binding)?

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

**Moved (2026-07-12, bookkeeping — no content or seal change).** Procedure v1
(sealed via [PR #100](https://github.com/raylei50653/saccade/pull/100) merge)
now lives at
[procedures/gt_support_morphology_procedure_v1.md](../eval/procedures/gt_support_morphology_procedure_v1.md).
The §19.x sub-section numbering (§19.0–§19.6) is preserved verbatim in that
file, so existing §19.2 / §19.4 / §19.5 citations across the repo resolve
there unchanged. This section number is retired and will not be reused; §19
is the historical exception to the §20.0 hosting rule.

---

## 20. Role-aligned experiment contract (normative)

### 20.0 Status and scope

Contract version **v1.3 (2026-07-26; append-only — v1.2 text unchanged, §20.10 added: online / research mutual exclusion, the state machine that owns the research → online direction, the default frozen axis set and its per-instance escalation, the two close dispositions, the rule that the lock stays outside every axis it freezes, and the explicit non-goals)**. Prior version note: v1.2 (2026-07-16; append-only — v1.1 2026-07-13 text unchanged, §20.9 added: substrate as a fourth declaration coordinate, dual-space accounting in owner symbols, ρ/aggregation reduction typing, conservation identities, dependence declaration, cross-space inference obligations, and typed failure semantics; no ε-bound formula is made normative)**. Prior version note: v1.1 (2026-07-13; append-only — v1 2026-07-12 text unchanged, §20.8 and the §20.2 κ line added to consolidate the declaration seal bar accrued in owner reviews). This section is the normative home of the experiment contract. Issue threads, study notes, and PR descriptions must **reference** this section; they must not restate or fork it. Every new decision-layer study that uses this framework's language or infrastructure runs under this contract. Studies opened before v1 keep their sealed procedures but must be re-classified under §20.4 before any result is cited as a design recommendation.

**Hosting rule.** This framework hosts cross-line semantics only. Line-specific predeclared procedures are hosted as standalone files under [`procedures/`](../eval/procedures/), referencing this framework for shared terms; they are not added as new framework sections. §19 (GT-support morphology) was drafted in-framework and is the historical exception — its sealed v1 body has been moved to [procedures/gt_support_morphology_procedure_v1.md](../eval/procedures/gt_support_morphology_procedure_v1.md) with §19.x numbering preserved, and the §19 slot is a tombstone.

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
Target decision layer
                 exactly one: coarse gate / score-ranking / assignment /
                 calibration / none (cross-layer substrate work)
Study intent     exactly one primary intent:
                   design evaluation / capability map /
                   boundary diagnostic / performance upper-bound probe
                 (secondary uses may be listed, but the primary intent
                  governs objectives, metrics, and selection)
Design objective role-legal per §20.3 for the declared layer and intent;
                 "maximize FP removed" alone is invalid
Selection rule   the ordered criteria of §20.4; "best performer" alone is invalid
Validity gate    predeclared observation-validity / identifiability
                 requirements (minimum exposure, trial-unit and bound
                 validity) separating UNRESOLVED from futility (§20.7)
Stop condition   predeclared sufficiency and futility stops per §20.6
Output class     which §20.4 classes the study's results may claim
Mainline transition
                 which mainline state transition each terminal outcome
                 produces (§20.7); a study with none is diagnostic and
                 must not occupy mainline cadence
Type κ           for every decidable unit:
                 κ = (quantification space, comparison relation, decision rule),
                 declared as three separate components — the space says what is
                 quantified over, the relation and rule say how it is judged.
                 Fidelity claims (e.g. exact or tolerance-bounded comparisons
                 over event/pair units) and claim-level statements (e.g.
                 ε-level bounds over trial units) route to different
                 falsification rules; sealability of the declaration is
                 governed by §20.8
```

The target layer and the study intent are independent axes: a boundary diagnostic *of the gate layer*, an upper-bound probe *of score-layer ranking*, and a capability map *of the ambiguous band* are all expressible and must be declared as such. Collapsing the two axes into one "role" is not permitted.

### 20.3 Role-legal objectives

A **design evaluation** pursues the design objective of its declared target layer. Contract v1 defines design objectives for two layers:

**Coarse gate.** Subject to a minimum-utility threshold, find a **large-margin, monotone, structurally simple, mechanism-interpretable obvious-negative region** — candidates that are geometrically impossible, temporally implausible, scale/position-degenerate, or in extreme low-credibility tails. The gate question is *where should the gate stop*, not *how far can the gate safely reach*. Gates deliberately retain the ambiguous band for downstream layers; a gate is not required to approximate the full decision boundary.

**Score / ranking.** Within the retained ambiguous band, establish whether conditional interactions **stably and interpretably improve the relative ordering** of GT versus FP candidates. Primary metrics are event-local ranking quantities: pairwise ranking accuracy, GT rank / reciprocal rank, top-k GT recall, positive–negative score margin, assignment-flip attribution, per-sequence interaction stability, LOO interaction retention, online MOT retention. FP-removed and GT-hurt remain reportable but are not score-study objectives. Weak signals that are individually inconclusive but jointly informative (e.g. large distance ∧ long gap ∧ weak motion consistency) belong here as interaction terms $s(x)=s_0(x)+\Delta s_{\mathrm{condition}}(x)$, not as hard rejections.

**Assignment / calibration.** Contract v1 does **not** define design objectives for these layers. They may be declared as target layers for diagnostic intents (capability map, boundary diagnostic, upper-bound probe); a design evaluation targeting them is blocked until a contract revision defines their role-legal objective and is sealed.

**Capability map / boundary diagnostic / performance upper-bound probe (intents).** The §14 constrained-optimization forms are the appropriate instruments here. These intents must still name the target layer they diagnose or bound. Outputs are limited to the diagnostic and upper-bound classes of §20.4 and are subject to §20.5: they describe what the signal family can resolve and where it stops, and they never carry design authority on their own.

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

A result classified as *diagnostic* or *performance upper bound* must not be promoted — directly or by re-labeling — into a production design recommendation. Promotion requires a new evaluation declared with design-evaluation intent under the target decision layer, passing that layer's legal objective (§20.3) and the full selection order (§20.4). Re-labeling is not promotion. This constraint is what prevents the highest-utility candidate from re-entering the gate under a new name.

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

### 20.7 Mainline progress accounting

Progress is counted in **mainline state transitions**, not in artifacts produced. A completed study moves the mainline only if it does at least one of:

1. **closes a core unknown** — e.g. establishes that the current signal family has no usable ranking power in the retained ambiguous band, formally terminating that path;
2. **adds a decision capability** — e.g. a validated interaction enters the score layer and changes candidate ordering;
3. **changes production behavior or metrics** — e.g. fewer ID switches, higher AssA, or the same effect from a structurally simpler gate.

Another morphology map, a tighter closure, an additional safe candidate, or a more precisely described boundary are diagnostic results (§20.4). They may proceed, but they must not occupy mainline cadence, and completing them does not count as mainline progress.

A mainline study must therefore be designed so that **every predeclared terminal outcome produces a state transition — including the negative one**. A study whose failure mode is "describe more completely and continue" is not a mainline study. Ambiguous results do not open a third door: if the predeclared minimum effect (§20.6) is not met, the futility terminal applies and the corresponding path closes.

**Observation-validity gate (precondition).** The two-door efficacy terminal applies only after the study's predeclared observation-validity / identifiability gate (§20.2) passes. A validly powered experiment that misses the minimum effect closes the path. Validity failure — insufficient exposure (§3.3), an invalid trial unit or confidence bound (§8.1; e.g. unresolved residual clustering as in §19.5), substrate corruption, or conflicting fold verdicts — yields **UNRESOLVED / INVALID-STUDY** instead. This terminal closes the current experiment, not the scientific hypothesis path, and does not count as mainline progress. This is not a third door for ambiguous efficacy: it separates "the experiment answered no" from "the experiment could not answer the question"; only the former closes a path, and the latter must not be reported as signal-family exhaustion.

The task-selection question is not *what can be validated next*, but:

> After this task terminates, which mainline state transition has occurred?

### 20.8 Declaration seal bar (v1.1)

This section consolidates the sealability requirements accrued in owner reviews
(PRs #135, #152, #154–#157). It is the single normative home for "what a
declaration must pin down before it can be sealed"; declarations, threads, and
review checklists reference it and must not restate or fork it. The governing
test for the whole bar:

> **Two independent implementers, given only the sealed declaration and its
> frozen inputs, would record the bit-identical terminal.**

A declaration is sealable only if every item below holds:

1. **Frozen degrees of freedom.** Every terminal-affecting choice is pinned in
   the declaration — tie-breaking, quantile method, refit policy, gate
   ordering, rounding/precision. A choice discovered later to affect the
   terminal is a declaration defect and is repaired by an append-only
   amendment, never an inline edit.
2. **Mechanical decidability.** Every acceptance and terminal criterion is
   decidable from frozen artifacts with no post-hoc judgment; language like
   "clearly better" or "reasonable margin" is not sealable.
3. **Exhaustive terminal mapping.** The declared terminal set partitions all
   reachable outcomes, including validity failure (§20.7 UNRESOLVED) and
   execution-invalid outcomes that produce no packet (build failure, runner
   crash, serialization failure — all fail-closed, never unmapped); there is
   no residual "describe more and continue" outcome, and every terminal names
   its mainline transition — or explicitly maps to `none / diagnostic-only`
   when the declaration claims no mainline cadence under §20.2. A diagnostic
   declaration is sealable; what it may not do is occupy mainline cadence.
4. **Scoped exhaustion and naming.** A negative terminal claims exhaustion
   only over the declared complexity class, and terminal naming must not
   exceed what the partition definitions entail — an outcome that is zero by
   construction is bookkeeping, not futility.
5. **Joint headroom.** Headroom and feasibility statements are judged jointly
   on the decision-relevant quantities, never per-quantity.
6. **Blind→reveal binding.** Where the study has a blind phase, the reveal is
   bound to the sealed runner and blind artifacts by recorded hashes (runner,
   blind artifacts, capture, manifest).

How other documents may cite a sealed boundary is doc governance, not part of
this bar: projections copy the owner document's self-designation verbatim
(doc structure contract C5.1 — link, don't relabel).

### 20.9 Dual-space accounting and reduction typing (v1.2)

This section is the normative home for how a declaration locates its claim
across quantification spaces and what a cross-space inference must declare.
The route *topology* (which spaces exist and how studies chain) is owned by
the research control plane; per-capture partitions and counts are owned by
their study documents. This section owns only the typing rules. **No specific
ε-bound formula is normative in v1.2**; the bound *interface* is fixed
(§20.9.6), the formula is deliberately deferred.

#### 20.9.1 Declaration coordinates

v1.2 **adds substrate as a fourth declaration coordinate**. This is a new
axis, not a restatement of any earlier role taxonomy. Every decidable unit is
located by four orthogonal coordinates, none inferable from another;
collapsing any two into one label is not permitted (extends the §20.2
two-axis rule):

1. **target decision layer** (§20.2);
2. **study intent** (§20.2);
3. **κ quantification space** (§20.2) — one node of the space account below;
4. **substrate** — the coordinate family the claim is proven on, **declared
   in the sealed study declaration**. When the study consumes or proposes a
   transition of a registered object, the declaration must cite and agree
   with that object's accepted `substrate` / `target_substrate` record. The
   declaration owns the current claim's substrate; the claim-state registry
   owns the accepted substrate state of production objects — the two are
   never merged, and neither a probe, a draft declaration, nor a diagnostic
   needs a registry object to declare its substrate.

Motivation: the D0 falsification shows that a shared scoring-function *form*
does not make one substrate — the offline and kernel representations of the
same quantity disagreed on the matched domain itself. A claim without a
substrate coordinate is not sealable.

#### 20.9.2 Space account (owner symbols)

The contract adopts the owner map's symbols and does not mint parallel ones:

```text
U^evt = M^evt ⊍ G^evt ⊍ E^evt
```

- **U^evt** — captured runtime event universe;
- **M^evt** — matched / joined pairs;
- **G^evt** — cohort_gap;
- **E^evt** — unemitted;
- **T_v** — trial-unit claim space (typed in §20.9.3).

Partition membership and counts for a given capture are owned by that
capture's study documents. Where matched events and joined pairs must be
distinguished, the join is a separately declared partial map **J_v**
(exact-key join); this contract introduces no separate pair-space symbol.
Fidelity-type κ quantifies over event/pair units; claim-type κ quantifies
over trial units (§20.2).

#### 20.9.3 Reduction typing: assignment ≠ aggregation ≠ judgment

A study whose claim quantifies over T_v while its evidence is produced over
event/pair units must declare the following **separately typed objects**.
Overloading one "ρ" symbol to simultaneously mean grouping, aggregation,
weighting, and decision is not sealable.

```text
S_v ⊆ U^evt                      source scope: declared union of partition cells
ρ_v : dom(ρ_v) → T_v,            assignment (quotient) map, dom(ρ_v) ⊆ S_v
X_v = S_v \ dom(ρ_v)             excluded events, each with a typed reason
a_{v,t} : Y^{ρ_v⁻¹(t)} → Z_t     per-trial aggregation of fiber observables
κ_T                              judgment over trial observables (§20.2 typed κ)
```

- **ρ_v is pure assignment**: it says only which events belong to the same
  trial unit. It is many-to-one, creates no trial units beyond its image, and
  performs no aggregation, weighting, or decision.
- **a_{v,t} is the aggregation rule**: how fiber observables become the trial
  observable. It is a separately frozen object referenced by κ_T; it is not
  itself the judgment.

All components are frozen at seal time (§20.8):

- **source scope** — `S_v` is a declared union of top-level partition cells;
- **domain** — `dom(ρ_v)` is a mechanically declared subset of S_v, selected
  by a frozen key/eligibility predicate; it is not required to equal S_v.
  Events in `S_v \ dom(ρ_v)` are excluded *and accounted* in X_v with typed
  reasons, never silently dropped;
- **codomain** — the trial-unit constructor, named before reveal. It is the
  **claim unit and candidate independence unit; independence is not granted
  by construction** and remains subject to §20.7 validity (§20.9.5);
- **version** — ρ_v is versioned; consumers bind to a version; a version bump
  is fail-closed (consumers do not silently follow);
- **computability** — ρ_v is total on its declared domain and mechanically
  computable from frozen keys alone; no post-hoc reassignment; naming must
  not exceed the partition definitions;
- **fiber accounting** — the study reports the fiber assignment and
  cardinalities `{|ρ_v⁻¹(t)| : t ∈ T_v}`.

#### 20.9.4 Conservation identities

Two identities, mechanically checkable, never merged into one equation:

```text
Σ_{t∈T_v} |ρ_v⁻¹(t)| = |dom(ρ_v)|          (assignment totality)
|dom(ρ_v)| + |X_v|   = |S_v|               (scope accounting)
```

When the study additionally claims coverage of the capture universe:

```text
|S_v| + |U^evt \ S_v| = |U^evt|            (universe accounting)
```

#### 20.9.5 Dependence declaration

T_v is the trial *observation/decision* unit. A study making trial-level
statistical claims must declare its dependence treatment: optionally a
cluster map `c_v : T_v → C_v`, and the level at which inference is claimed —
trial-weighted, cluster-weighted, sequence-blocked, or worst-case. Whether
the declared treatment suffices is judged by the §20.7 validity gate
(residual clustering, §19.5); this section only makes the treatment a
declaration obligation.

#### 20.9.6 Cross-space inference obligations

1. **No automatic transport in either direction.** An event/pair-level
   fidelity result never auto-upgrades to a trial-level claim; upward
   transport requires a sealed ρ_v, a declared aggregation rule a_{v,t},
   and a declared dependence treatment (§20.9.5). A trial-level result does
   not *by itself* refine to an event-level statement; any downward
   refinement requires a separately sealed theorem for the declared
   aggregation (e.g. a max-aggregate bound legitimately bounds every event
   in its fiber). An actual event-level counterexample continues to falsify
   a fidelity κ according to that κ's own declared quantification domain,
   comparison relation, and decision rule — trial eligibility does not
   shield a fidelity claim.
2. **Bound input interface — formula deferred.** Any cross-space bound must
   be declared as a function of a declared subset of the following interface:
   event/pair-level observables or fidelity bounds; the fiber assignment and
   cardinalities; the aggregation rule; the exclusion/missingness account;
   the weighting measure; the dependence/cluster structure; and
   reduction-specific stability certificates (e.g. margins). **Fiber
   accounting is a mandatory structural input to any cross-space bound, but
   is never presumed sufficient.** No ε-bound formula is normative in v1.2;
   nominal cluster-blind bounds remain recorded open limits and are not
   silently blessed.
3. **Substrate non-discharge.** ρ_v does not discharge substrate-equivalence
   obligations: any trial-level result derived through ρ_v remains pinned to
   the source substrate unless a separately sealed runtime-quantity fidelity
   edge authorizes transport. Reduction is never a substrate-transport proof.

#### 20.9.7 Failure semantics (not a terminal enum)

Four distinct failure semantics. This section defines the semantics only;
§20.7 owns their mapping into terminal families and §20.8 their sealability.
No terminal-slot enum values are added by this section.

| Failure semantics | Precise meaning | Repair path |
|---|---|---|
| **assignment-unresolved** | No ρ_v is mechanically computable from frozen keys on the declared S_v (constructibility/keying failure) | capture / keying |
| **not-identifiable** | The available observations do not determine the target claim on the declared domain; observational equivalence (distinct structures inducing identical observations) is one sufficient witness, not the only one — an empty decidable support is another; ρ_v may be fully constructible | new identifying evidence, substrate change, narrower model class, or weaker claim |
| **transport-noncommuting** | A declared cross-representation or cross-space transport fails to commute (e.g. offline vs runtime representations of the same quantity), or no legal expansion within the frozen class reaches the target condition | change representation/reduction, or record a class-scoped closure |
| **not-exchangeable** | The dependence structure does not support the declared nominal trial-independent inference | change cluster unit or statistical bound; owned by the §20.7 validity gate (§19.5) |

Conflating any two of these in a terminal name violates §20.8 item 4 (naming
must not exceed what the definitions entail). These semantics are **not
intrinsically assigned to one terminal family**: the mapping depends on the
study's declared target and on whether its observation-validity gate passed.
A transport-noncommuting or not-identifiable outcome can be the *valid
negative answer* of a study that targets that very question (a fidelity
study validly falsifying a proxy is a completed falsification, not an
invalid study), while the same semantics arising incidentally elsewhere is a
validity failure. §20.7 must preserve the distinction between a valid
negative answer and an experiment that could not answer.

### 20.10 Online / research mutual exclusion (v1.3)

Online modification and research measurement are **mutually exclusive states of
the repository**. §20.2 already requires a declaration to freeze its inputs
before running; this section owns the complementary direction — what the online
surface may do while that declaration is being measured against.

The direction matters because only one of the two was ever guarded. Coordinate
staleness (`check_runtime_identity_staleness.py`, run fail-closed by
`pre_push.sh`) catches an online move that was not republished, so a bound study
dies correctly. Nothing prevented the move. Evidence can therefore be collected
against a substrate that is being edited underneath it, with the loss surfacing
only after the measurement has been spent — and measurements here are spent
under exactly-once authorizations, so the cost of learning late is the study.

#### 20.10.1 The state machine

```text
ONLINE_OPEN
    ↓ open      freeze the current runtime coordinate
RESEARCH_OPEN
    ↓ close     seal the conclusion and the version binding
RESEARCH_CLOSED
    ↓ release   explicit
ONLINE_OPEN
```

The state is owned by `research_lock_v1.json` in this directory and is moved only
by `scripts/tools/research_lock.py`. The graph is **total**: the three named
transitions are the only legal moves and every other pair is refused. Enforcement
is a contract test (`tests/contract/test_research_lock.py`), so it runs under the
existing fail-closed pytest step; a **missing lock file is a deleted guard, not
`ONLINE_OPEN`**.

At most one instance is open at a time. This is the same WIP=1 shape the doc
structure contract already applies to mainline charters (C8), not a new lock.

#### 20.10.2 What is frozen, and what is deliberately not

`open` freezes, by default, the two coordinate axes that the accepted
`runtime_coordinate_bindings_v1` consumption rule already classifies as `stale`
— **conclusion-invalidating** — rather than `re_attestation_required`:
`decision_surface` and `identity_semantics`, plus the published bounded probe.
This is not a new taxonomy; forking one would create a second truth about what
invalidates evidence.

`implementation`, `environment` and `runtime_inputs` are **not** frozen by
default. Freezing `implementation` would hold every decision-relevant source file
shut for the duration of a study and buy no fail-closed guarantee that the
existing re-attestation path does not already provide. A study that genuinely
needs more freezes more, per instance, via `frozen_axes`. Only source-derived
axes are lockable at all: `environment` and `runtime_inputs` cannot be recomputed
on every host, so they cannot be enforced on every push.

While an instance is open, each frozen axis must equal the frozen digest **both**
recomputed from source and as published. A republish is an online move; it is
refused the same way, so re-publishing is not a route around the freeze.

`RESEARCH_CLOSED` retains the frozen coordinate as the study's sealed version
binding but enforces no freeze. `release` is what returns the repository to
`ONLINE_OPEN`, and it drops the instance: a released lock keeps no ghost freeze.

#### 20.10.3 Two dispositions, and the C5.1 boundary

`close` records `disposition ∈ {sealed, voided}`. Both are legal exits — an
instance that must be abandoned so the online surface can move is *voided*, and
voiding is a first-class outcome rather than a deletion. Neither disposition is
an accepted terminal: the lock records that measurement stopped, never what was
concluded. Object state remains the claim-state registry's to write (doc
structure contract C5.1), and a close that produced no accepted transition must
not manufacture a registry `last_transition` (C6).

#### 20.10.4 The lock is outside every axis it freezes

H0 re-entry #3 terminated `H0_PROVENANCE_INVALID` because its declaration was
simultaneously a frozen runtime-bound input and the target of the seal that
mutated it: the transition judged itself a provenance mismatch. Any lock whose
own transitions moved a frozen digest would reproduce that defect. Three
consequences are normative, not incidental:

1. the lock file classifies as `non_execution` and appears in no coordinate axis;
2. the lock tool is not an `identity_semantics` path, so a defect in the guard
   remains repairable while an instance is open;
3. enforcement does not edit `pre_push.sh`, which is itself an
   `identity_semantics` file.

#### 20.10.5 Explicit non-goals

The following are **not** obligations of this section and must not be added to it
by projection or by review. Each is permanent friction bought against a need that
has not been demonstrated:

| Not required | Why |
|:--|:--|
| Applicability graph across versions | A closed study states its own scope; nothing consumes a cross-version graph |
| Cross-version compatibility maintenance | `runtime_coordinate_bindings_v1` already refuses equivalence claims; `stale` is version lag, not retraction |
| Automatic migration of closed research | A closed conclusion is true of the coordinate it was captured under and makes no claim beyond it |
| Permanent preservation of historical runtimes | The `environment` axis records a recipe pointer, deliberately not a restorable environment |

A closed study is not maintained, re-validated, or re-run. Multi-version
reproducibility is established only by a separate accepted decision naming a
concrete regulatory, regression-diagnostic, benchmark, or high-value
re-verification need — never as a default, and never as a side effect of opening
a new instance.

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
