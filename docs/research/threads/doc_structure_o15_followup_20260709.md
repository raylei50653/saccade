---
doc-status: active-thread
doc-promotion: navigation-only; not evidence
owner-module: ownership
created: 2026-07-09
---

# doc structure O1.5 follow-up thread

## Status

- O1.5 contract landed（homes / index / promotion / lifecycle）
- Checker: `check_doc_structure.py` warn-only
- threads/ navigation layer added（this directory）
- Backlog open: index debt · semantic TODO slim · optional strict

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

Pay down index / discoverability debt; keep entry docs thin; use threads for multi-step research chains only.

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
