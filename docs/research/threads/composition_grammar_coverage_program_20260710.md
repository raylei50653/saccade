---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-10
---

# Composition Grammar Coverage Completion Program

> **One-line:** Complete the composition-grammar research map under a fixed signal family and fixed substrate. C0 is already closed by T0-A/B/R1; current sole task is **C1-A semantic-contract preflight**. No evaluator extension, grammar enumeration, LOO, online hook, production change, or ledger promotion is authorized yet.

## Status

| Item | Status |
|:--|:--|
| Program | **ACTIVE** — staged coverage completion |
| Semantic sole active | **C1-A Restricted Grammar Semantic Contract Preflight** |
| C0 G1–G3 coverage closure | **ACCEPTED / CLOSED** via T0-A/B/R1, PR #94 |
| C1 grammar semantic contract | **PREFLIGHT AUTHORIZED ONLY** |
| C2 restricted offline atlas | **NOT AUTHORIZED** |
| C3 topology | **CONDITIONAL** on non-null C2 output |
| C4 per-sequence contraction | **CONDITIONAL** on region structure |
| C5 region-level LOO | **CONDITIONAL** on transferable structure |
| C6 online retention | **CONDITIONAL** on offline + cross-seq + LOO gates |
| Occ-exit conditional modeling | **PARKED next research family** |
| Production / presets | **unchanged** |
| evidence_ledger | **not promoted** |

## Program objective

Under a **fixed signal family** and **fixed substrate**, complete bounded coverage for:

```text
composition grammar
× offline point existence
× region topology
× per-sequence contraction
× region-level LOO
× online retention
```

The research question is:

> When stable safe regions fail to appear, is the limiting factor the signal family, the grammar's expressive role structure, or contraction across validation layers?

This is a **coverage program**, not a requirement that every grammar reach online implementation.

A grammar × validation cell is complete when it has one auditable bounded status:

```text
SUPPORTED
NULL_RESULT
DERIVABLE
BLOCKED_BY_SUBSTRATE
BLOCKED_BY_PROVENANCE
NOT_APPLICABLE
NOT_JUSTIFIED
```

`CONDITIONAL` is a planning gate, not a final evidence status. It must resolve to one of the bounded statuses above when the predecessor stage closes.

## Fixed boundaries

- Frozen signal family: `score_m_bridge`, `abs_log_h`, `dist_h`, `abs_ratio_m1`, `resid_mean`.
- Frozen Stage 2 primary decision cohort and unresolved firewall.
- G1–G3 truth remains the registered Q4.5 lattices; no continuous-domain overclaim.
- Existing T0 dual units remain canonical:
  - coordinate unit = registered threshold coordinate within one grid;
  - primary mask unit = `mask_sha256` within one registered grid;
  - component unit = grammar-specific lattice adjacency within one grid.
- Dual area and dual margin must never be collapsed.
- Engineering merge and research acceptance remain separate gates.
- Research progress is measured by closed hypotheses and bounded statuses, not evaluator/tool completion.

## C0 baseline — already complete, do not rerun

C0 is the accepted T0-A/B/R1 result, not a new execution task.

Truth pointers:

1. [Closed T0 thread](composition_grammar_safe_region.md)
2. [T0 artifact preflight](../../modules/semantic/research/composition_grammar_t0_artifact_preflight_20260710.md)
3. [T0 region interpretation](../../modules/semantic/research/composition_grammar_t0_region_interpretation_20260710.md)
4. PR #94 merge `acd8e30e`

Accepted C0 closure:

```text
154 productive-safe coordinates = 1 G1 + 153 G2 + 0 G3
142 single-sequence · 12 multi-sequence
12 multi-sequence coords → 8 primary per-grid masks / 4 global strings diagnostic
143/154 coords on multi-coordinate per-grid mask plateaus
26 components · 12 single-cell-width strips
full_neighborhood_safe_radius >= 1: 0/154
G7: not_derivable_from_current_artifact_contract
Terminal B: isolated_safe_points_only
```

C0 answers G1–G3 observation coverage. It does **not** establish that all restricted composition grammars lack stable regions.

## Coverage map

| Grammar | Offline point | Region topology | Per-sequence | Region LOO | Online |
|:--|:--|:--|:--|:--|:--|
| G1 Singleton | `SUPPORTED` within registered lattice | C0: zero thickness | C0 descriptive; deeper transfer conditional | exact-clause only; region LOO unresolved | selected-policy partial; region retention unresolved |
| G2 Pairwise AND | `SUPPORTED` within registered lattice | C0: thin / edge / zero thickness | C0: 12 multi-seq coords / 8 per-grid masks | exact-clause only; region LOO unresolved | conditional; no AND region hook claim |
| G3 Hard OR | `NULL_RESULT` for productive-safe points | null topology | per-seq null semantics to be contract-mapped | `NOT_JUSTIFIED` unless future non-null parameterization exists | selected-policy freeze only; not whole-region retention |
| G4 Atom-count | C2 only after C1 | C3 if non-null | C4 if justified | C5 if justified | C6 if passed |
| G5 Family-count | blocked until principled family ownership | conditional | conditional | conditional | conditional |
| G6 Extreme / Consensus | C2 only after C1 | C3 if non-null | C4 if justified | C5 if justified | C6 if passed |
| G7 Necessary violation + support | equivalence/role contract first | conditional | conditional | conditional | conditional |

This table is a research map. No row is required to reach online if an earlier stage resolves to `NULL_RESULT`, `NOT_APPLICABLE`, or `NOT_JUSTIFIED`.

## Program stages

### C0 — G1–G3 Coverage Closure

**Status:** accepted and closed through T0-A/B/R1.

No rerun, evaluator emit, or re-interpretation is authorized by this program unless a later contract change invalidates a specific C0 assumption.

### C1 — Restricted Grammar Semantic Contract

Define G4–G7 before evaluator work.

#### G4 Atom-count

Canonical form candidate:

```text
sum_i P_i(x; theta_i) >= k
```

Must lock:

- legal atom universe;
- legal `k` range;
- same-signal / opposite-direction policy;
- threshold coordinate system;
- symmetry and canonicalization;
- combinatorial cap;
- duplicate-mask semantics;
- stop rule when no new mask or capacity appears.

#### G5 Family-count

Canonical form candidate:

```text
sum_f 1[exists i in family f: P_i(x)] >= k
```

Must first justify mechanism-family ownership. Candidate labels such as score/residual, scale/height, geometry/distance, ratio/shape are hypotheses, not accepted ownership.

A valid C1 result may be:

```text
BLOCKED_BY_SEMANTIC_OWNERSHIP
```

which maps to program status `NOT_JUSTIFIED` for C2 until family ownership is grounded.

#### G6 Extreme / Consensus

Restricted form candidate:

```text
P_extreme OR (sum_{i in M} P_i >= k)
```

No arbitrary DNF, free nested Boolean search, or unbounded atom set.

Must define:

- what qualifies as `extreme`;
- moderate set `M`;
- consensus `k`;
- overlap / subsumption handling;
- whether output adds masks beyond G1/G4;
- bounded enumeration budget.

#### G7 Necessary violation + support

Role-structured form candidate:

```text
NOT GT-envelope-safe(x; theta_N)
AND support(x; theta_P)
```

C1 must distinguish:

- logical complement identity;
- necessary-envelope operand role;
- support operand role;
- envelope parameterization;
- mask equivalence to G1/G2;
- new role semantics versus genuinely new masks / coordinates.

G7 priority comes from necessary/sufficient role structure, not generic Boolean capacity.

### C2 — Restricted Offline Atlas Extension

**Not authorized.**

Default candidate order after C1 acceptance:

```text
G7 → G4 → G6 → G5
```

The order is advisory and must be re-authorized grammar by grammar.

Each grammar first answers only:

```text
productive-safe point exists?
new per-grid productive masks beyond G1/G2?
new productive capacity?
new multi-sequence support?
```

Stop immediately for that grammar if it adds neither a new mask outcome nor new bounded capacity/support.

Do not build one giant G4–G7 evaluator PR.

### C3 — Region Topology

Only for a C2 grammar with non-zero productive-safe masks.

Required common observation language:

- coordinate safe/productive area;
- per-grid unique-mask safe/productive area;
- connected components;
- active-axis count;
- `nearest_unsafe_distance`;
- `full_neighborhood_safe_radius`;
- semantic duplicates / plateau widths;
- productive capacity distribution.

Entry gate:

```text
non-zero unique productive masks
AND non-zero productive capacity
AND not entirely coordinate duplication
```

A non-null point atlas may still close as `NULL_RESULT / no topology worth transferring`.

### C4 — Per-sequence Contraction

Only for grammars with region structure worth testing.

Must answer:

- per-sequence safe/productive area;
- productive-support sequence count;
- sequence-specific islands;
- intersection / union of productive regions;
- minimum positive-sequence capacity and sequence dominance;
- whether pooled thickness is real cross-sequence structure or overlaid sequence islands.

Do not rebrand pooled GT0 decomposition as cross-sequence productive geometry.

### C5 — Region-level LOO

Only after C3/C4 identify transferable structure.

Must remain distinct from current exact-absolute-clause repeatability.

Transport modes are separate experiments and claims:

```text
absolute threshold
train quantile
GT-envelope-relative coordinate
rank / CDF coordinate
```

Required outputs:

- train region existence;
- frozen selection unit and transport;
- holdout region retention;
- holdout GT-harm bound;
- holdout productive mass;
- component geometry retention.

No "best clause 7/7" headline may substitute for region transfer.

### C6 — Online Retention

Only for grammar/region candidates passing:

```text
non-zero offline thickness
multi-sequence support
region-level LOO harm bound
productive floor
```

Keep two questions separate:

1. selected-policy retention — one frozen representative, default-off A/B;
2. parameter-region retention — whether offline component ordering, support, boundary, and capacity survive on online substrate.

Parameter-region retention may begin as observation-only shadow evaluation. It does not automatically require a multi-parameter hook.

## Current authorized task — C1-A semantic-contract preflight

**Branch:** `research/composition-grammar-coverage-program`

### Objective

Determine whether each G4–G7 grammar contract can be defined from current trusted semantics without inventing roles, families, coordinates, or unrestricted search space.

### Inputs

- accepted C0/T0 artifacts and notes;
- Q4.5 threshold registry and evaluator contracts;
- existing signal ownership / semantic notes;
- existing nested LOSO and online-hook claim boundaries;
- current no-go and unresolved provenance constraints.

### Required deliverable

Create one research note:

```text
docs/modules/semantic/research/composition_grammar_c1_semantic_contract_preflight_20260710.md
```

For each G4–G7, report:

1. research role and canonical expression candidate;
2. trusted operand / atom sources;
3. parameter-coordinate candidate;
4. symmetry / canonicalization requirement;
5. duplicate-mask and subsumption semantics;
6. bounded enumeration budget or reason enumeration is unjustified;
7. validation-layer applicability map;
8. provenance / substrate blockers;
9. status:
   - `CONTRACT_DERIVABLE`
   - `CONTRACT_PARTIAL`
   - `BLOCKED_BY_SEMANTIC_OWNERSHIP`
   - `BLOCKED_BY_PROVENANCE`
   - `NOT_JUSTIFIED`
10. exact missing decisions required before C1-B.

### G7-specific preflight

The preflight must explicitly separate:

```text
existing mask-string overlap
existing per-grid mask equivalence
envelope complement identity
necessary operand role
support operand role
envelope-relative coordinates
```

It must not claim G7 equivalence or non-equivalence when role semantics remain absent.

### G5-specific preflight

Do not accept mechanism families by naming convenience. For every proposed family, identify:

- fact owner;
- common mechanism hypothesis;
- reason grouping changes the research question;
- falsifier or ambiguity;
- whether the family is stable enough to authorize enumeration.

### Acceptance

C1-A is accepted only if:

- all four grammars have bounded derivability statuses;
- free Boolean / arbitrary DNF remains forbidden;
- G7 role semantics are not inferred from mask overlap alone;
- G5 may legally close blocked or not justified;
- parameter coordinates are explicit and grammar-specific;
- each grammar has an early-stop rule;
- C2 remains unauthorized;
- production, hook, terminal B, and ledger remain unchanged.

### Implementation freedom

Local research-note structure and helper inspection commands are free. No new generic framework is required. Hard boundaries above are the only locked constraints.

### Must not

- Modify or rerun the evaluator.
- Enumerate G4–G7 points.
- Add threshold lattices or signals.
- Implement region LOO or online retention.
- Add hooks, flags, presets, or production behavior.
- Treat sequence names as grammar semantics.
- Promote a contract hypothesis to evidence.
- Open a C2 implementation PR.

## Review sequence

```text
C0 accepted baseline
→ C1-A semantic-contract preflight  ← current
→ chat-side review
→ C1-B final restricted grammar contract
→ chat-side review
→ authorize at most one C2 grammar slice
→ review before any topology / per-seq / LOO / online stage
```

## Occ-exit disposition

Occ-exit conditional intervention modeling remains a valid future research family, but it is parked while composition-grammar coverage is active.

The existing #55 global-audit conclusion remains intact:

```text
global audit net harmful
one local enable candidate
no production/default promotion
```

Parking does not reopen or invalidate #55 evidence.

## History

- 2026-07-10: G1–G3 T0-A/B/R1 accepted; PR #94 merged.
- 2026-07-10: existing registered-threshold interpretation line closed with terminal B.
- 2026-07-10: broader Grammar Coverage Completion Program opened; C0 mapped to accepted T0; C1-A preflight becomes semantic sole active; occ-exit parked.
