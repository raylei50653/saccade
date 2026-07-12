---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
work-class: maintenance
wip-role: non-wip
created: 2026-07-09
---

# association recovery registry thread

## Status

- Step maps: scripts index + info-source contract + crosswalk landed
- Registry seed: `association_tools.yaml` populated（R）
- Checker: `check_association_tools.py` path health
- Not sole active（registry hygiene only；semantic sole-active is currently none）

## Current boundary

Navigation + tooling registry only. Does not own GO/NO-GO or baseline metrics.

## Read first

- [DEVELOPMENT.md](../../../DEVELOPMENT.md) D1 routing
- [docs/modules/semantic/TODO.md](../../modules/semantic/TODO.md)（sole active 不在此 thread）
- [scripts index](../../modules/semantic/research/association_recovery_scripts_index_20260709.md)
- [info source contract](../../modules/semantic/research/association_recovery_info_source_contract_20260709.md)
- [association_tools.yaml](../../modules/semantic/research/association_tools.yaml)
- [crosswalk](../../modules/semantic/research/association_recovery_crosswalk_20260709.md)
- [offline relink hub](../../modules/semantic/research/offline_relink_candidate_analysis.md)
- [signal_table_schema](../eval/signal_table_schema.md) · [signal_analysis_ledger](../eval/signal_analysis_ledger.md)

## Artifacts

- `docs/modules/semantic/research/association_tools.yaml`
- `scripts/tools/check_association_tools.py`
- Door A signal-study tree under `out/signal_study/`（numbers master stays there）

## Current step

Keep R/H split honest: door/role/fact-owner in YAML（R）；path health via checker；conclusions stay in research/ledger.

## Acceptance

- New association script → same-PR YAML + scripts-index row（or explicit defer）
- No second truth for metrics in registry files
- Checker stays green / warn-only as designed

## Must not

- Embed long metric tables in index / YAML
- Claim production GO from registry alone
- Reopen closed NO-GO as drive-by
- Steal WIP from semantic sole active without explicit park

## History

- 2026-07-09: scripts index + info-source + crosswalk + tools YAML seed
- 2026-07-09: thread opened as continuous registry mother-line
