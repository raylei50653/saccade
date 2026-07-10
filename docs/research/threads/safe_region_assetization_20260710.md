---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-10
---

# Safe-Region Assetization Program

> **One-line:** R0-A preflight is **ACCEPTED** after R0-A-R1/R2 corrections. Current sole task is **R0-B Final RegionAsset Contract Draft**. R1 asset generation, grammar extension, transfer, intervention, production, and ledger promotion remain unauthorized. Maturity **A0 retained**; pack claim ceiling **L1**, with per-object claim levels preserved.

## Status

| Item | Status |
|:--|:--|
| Program | **ACTIVE** — safe-region assetization |
| Semantic sole active | **R0-B Final RegionAsset Contract Draft** |
| Existing G1–G3 T0/C0 result | **A0 descriptive baseline**; terminal B retained |
| R0-A preflight chain | `762adf9a` → R1 `136841a8` → R2 `0f5799f4` |
| R0-A research review | **ACCEPTED** — derivability/provenance/math preflight closed |
| CR1–CR9 | **PASS at preflight level**; must be frozen correctly in R0-B |
| R0-B final contract | **DRAFT AUTHORIZED ONLY**; not self-accepted |
| R1 G1–G3 asset conversion | **NOT AUTHORIZED** |
| Current maturity | **A0 retained** |
| Statistical claim | pack ceiling **L1**; G1=L0, G2=L1, G3=L0 |
| Mathematical framework | [Statistical Robust Feasible-Set Estimation under Asymmetric Loss](../eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md) |
| Production / presets | **unchanged** |
| evidence_ledger | **not promoted** |

## Current boundary

The reusable object is an estimated feasible/productive-safe region asset under a declared asymmetric-loss contract:

```text
producer semantics
→ parameter / policy space and candidate universe
→ feasibility contract
→ estimated productive-safe set
→ normalized truth identity + exact evidence seal
→ region / mask / coordinate / null identities
→ geometry + capacity + sequence applicability descriptions
→ transfer qualification
→ action contract
```

Hard separations:

```text
artifact generated
≠ engineering ready
≠ asset maturity accepted (A0–A4)
≠ statistical claim level (L0–L6)
≠ research conclusion accepted
≠ intervention qualified
≠ production approved
```

## Read first

1. [Statistical Robust Feasible-Set Estimation under Asymmetric Loss](../eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)
2. [Accepted R0-A preflight packet](../../modules/semantic/research/safe_region_r0_asset_contract_preflight_20260710.md)
3. [Closed A0/T0 thread](composition_grammar_safe_region.md)
4. [T0 artifact preflight](../../modules/semantic/research/composition_grammar_t0_artifact_preflight_20260710.md)
5. [T0 region interpretation](../../modules/semantic/research/composition_grammar_t0_region_interpretation_20260710.md)

## Accepted A0 baseline

```text
154 productive-safe coordinates = 1 G1 + 153 G2 + 0 G3
26 coordinate components · 34 productive per-grid mask units
full_neighborhood_safe_radius >= 1: 0/154
G3 Hard OR: registered-lattice NULL_RESULT
terminal B: isolated_safe_points_only
```

This is finite-sample, searched in-sample, registered-lattice evidence. It is not population safety, held-out region retention, an intervention policy, or a production candidate.

## R0-A acceptance record

### Accepted preflight findings

- raw artifact/source seals belong to `evidence_bundle_id`, not stable semantic/content identity;
- normalized truth, pack/materialization, feasibility, content, and evidence-record identities are distinct;
- component/mask/coordinate/null grains remain distinct;
- ordinal component IDs and global-mask-only PKs are forbidden;
- component capacity is non-additive; coordinate-member and per-grid-mask-member distributions are separate;
- event-union capacity is blocked without sealed event membership;
- sequence incidence, union, intersection, and member-support range are descriptive A1 fields, not A2 applicability;
- G3 uses a grammar/search-domain null plus 40 concrete OR-grid semantic members;
- coordinate FKs are authoritative for region↔mask relations;
- A0–A4 maturity and L0–L6 claim ladders are orthogonal;
- existing evidence is sufficient for a deterministic R1 conversion after an accepted R0-B contract; no evaluator rerun is required.

### R0-B normalization decisions still required

These are no longer R0-A blockers, but the final contract must resolve them before R1:

1. **Claim ownership:** `claim_level_max` is an evidence/claim outcome, not an input to `feasibility_contract_id`. The feasibility contract owns the mathematical definition; an asset/evidence claim record owns the supported L-level.
2. **Per-object claim level:** G1’s single isolated productive-safe coordinate is **L0**; G2’s multi-coordinate in-sample geometry supports **L1**; G3’s declared-domain null is **L0**. The pack-wide maximum may be L1 without assigning L1 to every object.
3. **Grid-domain authority:** add a machine authority for every `grid_domain_id` referenced by `search_domain_members` or avoid declaring it as an FK. No FK may point to an undefined object.
4. **Normalized digest freeze:** D16 must enumerate exact truth-bearing columns per input table, stable row keys, missing-value encoding, duplicate handling, and float normalization. Raw byte hashes stay in evidence seals; ordinal aliases and non-semantic formatting fields stay out of normalized identity.

## Program stages

```text
R0 Region Asset Contract
  R0-A preflight                         # ACCEPTED
  R0-B final contract draft             ← current
  chat-side contract review

R1 G1–G3 deterministic asset conversion # not authorized
R2 grammar-by-grammar asset extension
R3 transfer qualification toward A2/L2–L3
R4 intervention qualification toward A3/L4–L5
A4/L6 separate production approval
```

## Current step — R0-B Final RegionAsset Contract Draft

**Working branch:** `research/composition-grammar-coverage-program`

### Deliverable

Create the cross-cutting final contract draft:

```text
docs/research/eval/safe_region_asset_contract.md
```

Index it in `docs/research/eval/README.md`.

The R0-A preflight remains the derivability/evidence packet; R0-B must be a concise normative contract, not a copy of the 1,000-line preflight.

### Required contract sections

1. **Scope and non-claims** — method/schema contract only; no A1 acceptance, transfer, intervention, or production claim.
2. **Mathematical object** — parameter/policy space, candidate universe, asymmetric safety loss, productivity, epsilon, `g_min`, exposure denominator definitions, metric/adjacency/edge policy.
3. **Identity layers** — `truth_contract_id`, `evidence_bundle_id`, `feasibility_contract_id`, `pack_id`, semantic/search-domain/content/null/evidence-record identities.
4. **Claim ownership** — feasibility definition separate from observed evidence and `claim_level`; A0–A4 independent from L0–L6.
5. **Object grains and relations** — semantic definition, grid domain, search domain/member, region, coordinate, per-grid mask, null record, pack membership.
6. **Stable-ID canonicalization** — namespace/version, canonical JSON, content digests, collision/fail-closed policy, alias policy, reorder/rematerialization behavior.
7. **Normalized truth digest** — exact table projections, stable row keys, canonical missing values, float representation, duplicate semantics, excluded ordinal/display fields.
8. **Geometry and capacity** — dual areas/margins, components, coordinate-member and mask-member capacity distributions, explicit event-union blocker.
9. **Sequence/applicability descriptions** — incidence, union, intersection, member range, dominance/islands; no union→applicability promotion.
10. **Null-result contract** — G3 search-domain null, concrete domain membership, counts and missing-artifact distinction.
11. **Machine schemas and referential integrity** — all authority files, PK/FK rules, pack membership, validation invariants.
12. **Maturity/action/claim firewall** — A0–A4, L0–L6, allowed actions, forbidden promotions.
13. **R1 conversion boundary** — deterministic packaging inputs/outputs and what remains blocked.

### Required machine authorities

At minimum define authoritative schemas for:

```text
truth_contract.json
evidence_bundle.json
feasibility_contract.json
evidence_claims.csv|jsonl
semantic_definitions.csv|jsonl
grid_domains.csv|jsonl
search_domains.csv|jsonl
search_domain_members.csv
region_assets.csv
null_records.csv
region_masks.csv
region_coordinates.csv
pack_membership.csv
region_claim_contract.json
```

Auxiliary/derived files may be defined, but every FK must resolve to one authority and derived link tables must name their authoritative source.

### R0-B locked defaults unless falsified

```text
pack scope: one G1_G2_G3 emission
content ID scheme: region_asset_id_v2, full SHA-256 retained
AND/OR operand canonicalization: lexicographic feature,direction for symmetric leaves
primary region grain: connected productive-safe component within one registered grid
primary mask grain: truth contract + grid_id + mask_sha256
coordinate identity: truth contract + native cell_id
G3: one grammar/search-domain null + 40 concrete grid members
capacity: both coordinate-member and mask-member distributions; no sum
sequence: incidence + union + intersection; no A2 inference
region↔mask authority: coordinate FKs
current action: observation_only; production_forbidden=true
```

### Acceptance for R0-B draft review

- normative contract is concise and internally consistent;
- `feasibility_contract_id` excludes evidence outcome fields such as supported L-level;
- actual exposures, selection scope, finite-sample statement, and claim level have a declared evidence/claim owner;
- G1=L0, G2=L1, G3=L0 are preserved; pack ceiling=L1 is clearly aggregate;
- `grid_domain_id` has a machine authority and G3 has exactly 40 domain members for this sealed study;
- normalized digest column sets and float/missing-value rules are explicit enough for two implementations to agree;
- content IDs survive row reorder/rematerialization but change when normalized truth or local membership changes;
- all CR1–CR9 corrections remain intact;
- every FK resolves to one declared authority;
- no asset pack is generated;
- R1 remains unauthorized pending chat-side R0-B acceptance.

### Return packet

```text
branch
tip commit
contract path
files changed
R0-B decisions locked
remaining open decisions / blockers
validation commands/results
explicit:
  no evaluator rerun
  no asset pack generated
  no A0→A1 promotion
  no R1 authorization
  no PR opened
```

## Must not

- modify or rerun the Q4.5 evaluator;
- generate RegionAsset data files;
- implement a generic runtime/framework library;
- implement G4–G7, LOO, shadow, hook, preset, or production behavior;
- infer G7 roles;
- change terminal B, maturity A0, or accepted T0 numbers;
- promote to evidence ledger;
- self-accept R0-B or authorize R1.

## History

- 2026-07-10: T0-A/B/R1 accepted; PR #94; G1–G3 A0 baseline.
- 2026-07-10: Safe-Region Assetization opened; R0-A authorized.
- 2026-07-10: R0-A `762adf9a`; CR1–CR5 requested.
- 2026-07-10: R0-A-R1 `136841a8`; CR1–CR5 pass.
- 2026-07-10: mathematical framework added under `docs/research/eval/`; CR6–CR9 requested.
- 2026-07-10: R0-A-R2 `0f5799f4`; CR6–CR9 pass at preflight level.
- 2026-07-10: chat-side review **accepted R0-A** and authorized R0-B contract draft only; R1/A1 remain blocked.