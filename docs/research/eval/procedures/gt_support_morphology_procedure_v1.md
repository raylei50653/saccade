# 19. GT-support morphology on Boolean atom lattices (predeclared procedure)

<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-12 -->
<!-- doc-module: cross -->

> **Provenance / numbering.** This file is the normative home of morphology
> procedure **v1**, sealed via [PR #100](https://github.com/raylei50653/saccade/pull/100)
> merge as §19 of the
> [framework doc](../statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)
> and moved here verbatim on 2026-07-12 (bookkeeping; no content change, seal
> unchanged). The §19.x sub-section numbering is **preserved** so that existing
> citations (§19.2, §19.4, §19.5, …) across the repo resolve here unchanged;
> the framework's §19 slot is a tombstone pointing to this file and is never
> reused. Per the framework §20.0 hosting rule, line-specific predeclared
> procedures live in standalone files like this one; §19 is the historical
> exception that was drafted in-framework. All terms (§3.3, §8.1, §8.3, §14,
> claim ladder L-levels) refer to the framework doc.


## 19.0 Status

Procedure version **v1 — PROPOSED (calibrated on pooled Step-0 audit; seal = research-owner acceptance via the procedure PR merge)**. Until sealed, results computed under it are exploratory. Studies must cite the procedure version they ran under. Evidence order is normative: calibrated step-0 → procedure seal → escape-tail forensics → nested confirmation; forensic results must not enter the sealing PR, so that mechanism information cannot feed back into the boundaries.

## 19.1 When this section applies

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

## 19.2 Hard interpretation rule

$$
\text{not observed} \neq \text{unsafe}
$$

Cells with zero GT exposure are **unresolved**, never "high-risk", never "barriers". A gap in $\mu_{\mathrm{GT}}$ support is evidence about where GT mass does not go only to the extent bounded by the placement UCB; it is not evidence about conditional risk in that cell.

## 19.3 Required declarations (before computing)

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

## 19.4 Fixed morphology statistics and trial semantics

### Set-valued trial semantics (normative)

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

### Descriptive statistics

```text
M_0        = P_GT(d_H = 0)                     corner mass
M_r, T_>=r = P_GT(d_H = r), P_GT(d_H >= r)     shell / tail profile
V_i        = P_GT(z_i = 0)                     atom violation profile
V_ij       = P_GT(z_i = 0, z_j = 0)            joint violation profile
d_H        = d_structural + d_motion           declared-group decomposition
```

(all computed on the min-$d_H$ representative; descriptive layer only.)

FP mass is reported on the same shells (rejectable material per shell). A tail identified by Hamming distance is a **far-Hamming descriptive tail**; the term *out-of-core GT mass* is reserved for $\{u : H_{C^\star}(u)=1\}$ after the core $C^\star$ has actually been solved.

## 19.5 Verdict typology (fixed terminals) and class boundaries

### Morphology budget

Class boundaries bind to the asymmetric-loss budget and a confidence upper bound — **not** to raw shell fractions or fixed Hamming radii, which depend on $k$, atom correlation, and the binarization rule.

$$
\varepsilon_{\mathrm{morph}} = 5\%
$$

evaluated as a **one-sided 95% upper confidence bound at the §8.1 trial unit**. $\varepsilon_{\mathrm{morph}}$ governs morphology classification (thin tail vs diffuse) only; it is **not** a production GT-hurt budget (0 / 0.1% / 1% budgets are separate contracts).

### UCB validity (cluster condition)

A Clopper–Pearson bound is valid **only when the declared trial units are independent Bernoulli trials**. When the study declares residual clustering above the trial unit (per §8.1 — e.g. sequence-level shared scene and pipeline state), the unique-unit count is only an upper bound on the number of independent trials, and a plain CP bound is anti-conservative. In that case the study must either:

- use a **cluster-aware bound** (e.g. aggregate to the clustering level, or a declared cluster-robust method), or
- further aggregate the trial unit until the independence assumption is defensible.

A plain CP number computed under declared residual clustering may be reported only as a **nominal diagnostic** (`nominal; not cluster-adjusted`) and **must not be used to cross the $\varepsilon_{\mathrm{morph}}$ boundary**; a terminal that would depend on it stays `UNRESOLVED`.

### Core definition

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

### Terminals

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

## 19.6 Evidence order

```text
step-0 occupancy + placement audit (descriptive, L1 ceiling)
→ owner seals procedure (this section, versioned)
→ escape-tail forensics (predeclared categories)
→ nested per-fold rerun of the full chain
  (atom discovery, orientation, binarization, verdict)
```

A pooled in-sample audit may inform the choice of class boundaries; when it does, the study must say so, and the confirmatory unit is the nested per-fold rerun, not the pooled audit.
