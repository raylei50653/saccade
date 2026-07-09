---
doc-status: active-thread
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-09
---

# occ-exit audit (#55) thread

> **One-line:** Semantic sole active. Finish audit / 條件化 on CleanFifoBank substrate; default-off only; no global production flip.

## Status

- Sole active on **semantic** (WIP=1)
- Substrate: CleanFifoBank pre-episode reference **wired**
- Remaining: Cheb-GR graph decision + 13-type sequence conditioning
- Production: not default-on

## Current boundary

RESEARCH + DEBUG. default-off flags / probes only until explicit promotion decision.

## Read first

1. [occ_exit scope](../../modules/semantic/research/occ_exit_audit_p55_scope_20260709.md)
2. [WP2 seq conditioning](../../modules/semantic/research/occ_exit_audit_p55_wp2_seq_conditioning_20260709.md)
3. [WP3 promotion decision](../../modules/semantic/research/occ_exit_audit_p55_wp3_promotion_decision_20260709.md)
4. [clean_fifo_bank substrate](../../modules/semantic/research/clean_fifo_bank_substrate_20260704.md)
5. [semantic TODO](../../modules/semantic/TODO.md)（WIP lock only）

## Artifacts

- `src/saccade/perception/eval/clean_fifo_bank.py`
- evaluator `--occ-audit-bank-reference` · `occ_exit_audit_lines_from_bank`
- `scripts/eval/diagnostics/probe_occ_audit_bank_reference.py`
- tests: `test_clean_fifo_bank.py` · `test_occ_audit_bank_reference.py`

## Current step

Cheb-GR graph decision (not cosine-only) + 13-type sequence conditioning map; keep audit path default-off.

## Acceptance

- Sequence-conditioned applicability (not global on)
- Evidence for keep-off / narrow-on / park decision on WP3 path
- No silent production preset change
- No sync ReID on critical path (#57)

## Must not

- Global on without promotion evidence
- Steal WIP narrative into TODO long prose
- Reopen closed NO-GO identity lines as drive-by

## History

- 2026-07-04: CleanFifoBank substrate + bank reference wired
- 2026-07-09: scope / WP2 / WP3 notes; thread opened as WIP mother-line
