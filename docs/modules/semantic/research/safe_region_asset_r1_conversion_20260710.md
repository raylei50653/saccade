---
doc-status: active
doc-promotion: research note only; not evidence_ledger
owner-module: semantic
created: 2026-07-10
---

# Safe-Region Asset R1 Conversion Note

> **One-line:** Deterministic packaging of sealed Q4.5 + T0-B-R1 evidence into an **A0 observation-only RegionAsset pack candidate**. Engineering validation PASS. **Not A1.** Chat-side review required.

## Authorization

| Item | Value |
|:--|:--|
| Task | `safe_region_assetization_r1_conversion` |
| Branch | `research/composition-grammar-coverage-program` |
| Reviewed R0-B tip | `f92340b7` |
| R0-B acceptance | `01a2ec37` |
| Delivery path | PR-driven engineering (direct-agent dispatch retired; no longer execution authority) |
| Contract | [safe_region_asset_contract.md](../../../research/eval/safe_region_asset_contract.md) (ACCEPTED + editorial E1) |
| Thread | [safe_region_assetization_20260710.md](../../../research/threads/closed/safe_region_assetization_20260710.md) |

## Deliverables

| Path | Role |
|:--|:--|
| `scripts/tools/convert_safe_region_asset_r1.py` | research-only converter + validator |
| `tests/unit/test_safe_region_asset_r1_conversion.py` | focused contract/unit tests |
| `out/signal_study/m_b1_5_safe_region_asset_r1_20260710/` | generated A0 pack candidate |
| this note | engineering conversion record (not ledger) |

### Pack root files (authorities + emissions)

```text
region_asset_manifest.json          # RB2 pack authority
truth_contract.json
candidate_universe_contracts.json
candidate_universe_instances.json   # RB8 instance + membership digest
threshold_registry.json
threshold_registry_entries.csv      # RB9
predicate_definitions.jsonl
policy_family_definitions.jsonl
policy_instances.jsonl
evidence_bundle.json
feasibility_contract.json
evidence_claims.jsonl
grid_domains.csv
search_domains.csv
search_domain_members.csv
region_assets.csv
null_records.csv
coordinates.csv                     # truth-level (RB5)
mask_units.csv                      # truth-level (RB5)
region_coordinate_membership.csv    # feasibility-bound (RB5)
pack_membership.csv
region_claim_contract.json
region_mask_link.csv                # derived
preflight_report.json
validation_report.json
conversion_summary.json
```

## Editorial E1

§2.4 inverted positive implications replaced with explicit non-implications:

```text
generator-contract equality ⇏ same sealed universe instance
source_event_table_sha256 ⇏ universe_membership_digest
policy family ⇏ concrete threshold-executable policy
thr_index without registry ⇏ reconstructible thr_value
```

Non-substantive; does not reopen RB8/RB9.

## Preflight seals

Runtime Q4.5 full atlases, committed Q4.5 SHA256SUMS, T0 SHA256SUMS, and `d_online_events.parquet` (`source_event_table_sha256=cfca3818…`) verified before emission. Preflight status: **OK**.

## Accepted counts (reproduced)

| Object | Count | claim_level |
|:--|--:|:--|
| G1 non-null | 1 | L0 |
| G2 isolated | 6 | L0 |
| G2 multi-coordinate | 19 | L1 |
| G3 domain null | 1 | L0 |
| PS coordinates | 154 | — |
| Productive mask units | 34 | — |
| Pack ceiling | — | L1 |

Manifest: `maturity_declared=A0`, `composition_level=observational`, `production_forbidden=true`, `terminal_letter=B`.

## Identity checks

| Check | Result |
|:--|:--|
| Two clean converter runs → identical authority content + IDs | **PASS** (fingerprint `7fdd7eba…`) |
| PK uniqueness + FK resolution | **PASS** |
| Universe contract ≠ instance; membership digest ≠ event-table SHA | **PASS** |
| Threshold entries reconstruct thr values; every coord axis binds one entry | **PASS** |
| Policy family ≠ policy instance | **PASS** |
| Pairwise leaf canonical order (RB4) | **PASS** |
| Coordinates/masks free of feasibility-bound outcome fields | **PASS** |
| G3 null `policy_family_definition_id` empty; claim L0 | **PASS** |

### Example sealed digests (this emission)

```text
pack_id:                         1a180620bc050e70…
truth_contract_id:               ba4dd82e04e0ea2e…
feasibility_contract_id:         818a157c3e5d5c52…
candidate_universe_instance_id:  d89d459a3d8a2eb3…
universe_membership_digest:      f4988a6f15582651…   # ≠ cfca3818… event table
threshold_registry_id:           a9d37931aed66356…
```

## How to reproduce

```bash
.venv/bin/python scripts/tools/convert_safe_region_asset_r1.py \
  --out out/signal_study/m_b1_5_safe_region_asset_r1_20260710

.venv/bin/python -m pytest tests/unit/test_safe_region_asset_r1_conversion.py -q --no-cov
```

## Explicit non-claims

```text
≠ A1 research acceptance (chat-side / research-owner gate still required)
≠ evaluator rerun or modification
≠ new threshold / policy / geometry search
≠ L2+ claims
≠ G4–G7 / NOT / complement reject
≠ LOO / shadow / hook / preset / production
≠ evidence_ledger promotion
≠ research acceptance or merge authorization from R1 conversion alone
≠ research verdict self-acceptance
```

R1 conversion did not itself grant research acceptance or merge authorization. Engineering delivery and review now proceed through the active implementation PR: [#95](https://github.com/raylei50653/saccade/pull/95).

## History

- 2026-07-10: R1 converter implemented; E1 applied; A0 pack candidate emitted; two-run determinism PASS; unit tests PASS.
