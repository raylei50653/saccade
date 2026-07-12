---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: ownership
work-class: governance
wip-role: non-wip
created: 2026-07-09
---

# doc structure O1.5 follow-up thread

## Status

- O1.5 contract landed（homes / index / promotion / lifecycle）
- Checker: `check_doc_structure.py` warn-only
- threads/ navigation layer added
- **TODO demoted to WIP lock**（C7 + module TODOs slimmed 2026-07-10）
- Backlog open: index debt · optional TODO-length warn · optional `--strict`

## Current boundary

Docs governance only. No runtime / preset / production path.

## Read first

- [DEVELOPMENT.md](../../../DEVELOPMENT.md)（薄入口 · D0–D4）
- [doc_structure_contract.md](../../ownership/doc_structure_contract.md)（O1.5）
- [DOC_MAINTENANCE.md](../../DOC_MAINTENANCE.md)
- [docs/README.md](../../README.md) 寫作決策樹
- [docs/research/README.md](../README.md)
- [ownership README](../../ownership/README.md)

## Artifacts

- `scripts/tools/check_doc_structure.py`
- `scripts/tools/check_doc_links.py` · `check_doc_stale_paths.py` · `check_doc_freshness.py`
- Contract backlog section in O1.5（explicit non-blocking items）

## Current step

Pay down research-index debt (e.g. semantic README S1 warnings); keep TODO thin; optional structure warn for overlong TODO.

## Acceptance

- New research note → same-PR owning README index row
- No phantom paths in active indexes
- threads/ remains navigation-only（no evidence tables）
- Strict mode only after intentional debt pass

## Must not

- Mass historical file moves without plan
- Second truth for baselines in entry docs
- Topic-hub sprawl without need
- Turn DEVELOPMENT into encyclopedia

## History

- 2026-07-09: O1.5 contract + warn-only structure checker
- 2026-07-09: DEVELOPMENT thin-entry D0–D4 routing
- 2026-07-09: threads/ mother-line layer opened
- 2026-07-10: TODO = WIP lock only；all module TODOs slimmed；C7 role split
