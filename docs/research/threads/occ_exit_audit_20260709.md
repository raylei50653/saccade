---
doc-status: parked
doc-promotion: navigation-only; not evidence
owner-module: semantic
created: 2026-07-09
---

# occ-exit audit (#55) thread

> **One-line:** **PARKED next research family.** WP1–WP3 are complete: global cosine audit is net harmful, Cheb-GR remains log-only, and one local enable candidate does not justify a gate. Resume only after the composition-grammar coverage program reaches a new owner decision.

## Status

- Semantic sole active: **no**
- WP1 Cheb-GR graph decision probe: **complete** (log-only)
- WP2 sequence conditioning analysis: **complete**
- WP3 promotion decision: **complete**
- Promotion result: `split_feat_pr` recommendation, **not implemented**
- Aggregate result: cosine audit net harmful on frozen substrate
- Production / presets: unchanged
- Current disposition: **parked**

## Current boundary

The accepted #55 evidence remains:

```text
global cosine occ-exit audit: net harmful
one local enable candidate: MOT17-11
four harmful · two abstain
Cheb-GR: observation/log probe only, not metric treatment
no sequence gate
no production promotion
```

Parking preserves this bounded conclusion. It does not convert the local positive sequence into a policy and does not reopen global audit.

## Read first

1. [WP3 promotion decision](../../modules/semantic/research/occ_exit_audit_p55_wp3_promotion_decision_20260709.md)
2. [WP2 seq conditioning](../../modules/semantic/research/occ_exit_audit_p55_wp2_seq_conditioning_20260709.md)
3. [occ_exit scope](../../modules/semantic/research/occ_exit_audit_p55_scope_20260709.md)
4. [clean_fifo_bank substrate](../../modules/semantic/research/clean_fifo_bank_substrate_20260704.md)
5. [grammar coverage program](composition_grammar_coverage_program_20260710.md)

## Artifacts

- `src/saccade/perception/eval/clean_fifo_bank.py`
- evaluator `--occ-audit-bank-reference` · `occ_exit_audit_lines_from_bank`
- Cheb-GR decision-log probe columns
- `scripts/eval/diagnostics/probe_occ_audit_bank_reference.py`
- WP2 classifier / diagnostic
- WP3 frozen-substrate evidence under `results/occ_exit_p55_wp3/` (local, reproducible by documented command)

## Current step

```text
none — parked
```

Potential future family, not currently authorized:

```text
episode-level conditional intervention modeling
→ observable pre-intervention condition
→ benefit / harm attribution
→ stable applicability region
```

Sequence-name allowlists would be comparators only, not final research semantics.

## Reopen conditions

Reopen only after an explicit owner decision and one of:

- composition-grammar coverage reaches a natural pause or closure;
- new event-level counterfactual attribution becomes trustworthy;
- a separate feature proposal defines a default-off action A/B beyond the log-only Cheb-GR probe.

Reopening requires a new staged contract. It does not inherit authorization to implement an allowlist or live gate.

## Acceptance retained

- Global audit remains keep-off.
- Local positive signal remains descriptive, not promotable by itself.
- No silent production preset change.
- No synchronous ReID on critical path (#57).
- Engineering instrumentation remains reusable.

## Must not

- Treat parked as active WIP.
- Implement a sequence allowlist from WP3 alone.
- Read Cheb-GR log agreement as a Cheb-GR action benefit.
- Flip global audit or headline presets.
- Reopen closed NO-GO identity lines as drive-by work.

## History

- 2026-07-04: CleanFifoBank substrate + bank reference wired.
- 2026-07-09: WP1 probe, WP2 conditioning, and WP3 promotion evidence completed.
- 2026-07-09: WP3 concluded `split_feat_pr`, with no runtime gate or production change.
- 2026-07-10: parked while Composition Grammar Coverage Completion Program becomes semantic sole active.
