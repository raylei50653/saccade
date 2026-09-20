---
doc-status: parked
doc-promotion: navigation-only; not evidence
owner-module: semantic
work-class: mainline-study
wip-role: parked
created: 2026-08-27
---

# Observability-weighted directional likelihood task

## Status

**PARKED · mainline-study · SEALED, no runner-review scheduled.**
Owner seal issued 2026-09-20 ([#442](https://github.com/raylei50653/saccade/issues/442)):
`declaration_seal_head = 311c222580dd29fcecacd5e2cc7cf6b2b459e684`, carried in the
[seal receipt](../../modules/semantic/research/observability_weighted_directional_likelihood_seal_receipt_20260920.json).
The semantic sole-active slot is released; the authority pointer remains
[semantic TODO](../../modules/semantic/TODO.md). The seal freezes identity only —
no formal outcome access, runner, or execution is authorized, and nothing
follows automatically.

## Current boundary

Determine whether the frozen observability index separates angular-concentration
regimes in the offline B1 universe, and whether conditioning on it exposes
held-out event-ranking information hidden by raw cosine direction. The target
layer is SR2 score-ranking, **MOT17-internal**; assignment, runtime, system
efficacy, and cross-dataset generality are outside scope.

## Expected state (lease)

A seal-reviewable declaration and fail-closed implementation whose signals
distinguish:

1. low-vs-high \(q_v\) angular concentration (phenomenon); and
2. OWDL-vs-raw held-out event PWA (the existing evidence gap).

This is expected state, not accepted evidence.

## Commit point

Owner reviews the exact pre-outcome declaration, machine records, source
identities, math core, and tests and either seals them together or rejects/
revises before any formal row is loaded. A seal, if issued, is a **seal
receipt** that records the **exact merged commit SHA** as
`declaration_seal_head`. A branch name, tag, or other movable ref is not
an identity.

## Discard when

- source identity cannot be frozen or reconstructed without a new runtime run;
- candidate-event semantics cannot conserve the frozen pair universe;
- covariance estimation would require outcome labels or post-hoc regularization;
- target scope expands to score integration or MOT efficacy without a new
  declaration.

## Read first

- [study declaration](../../modules/semantic/research/observability_weighted_directional_likelihood_declaration_20260827.md)
- [score-ranking evidence contract](../contracts/score_ranking_evidence_contract.md)
- [historical offline candidate analysis](../../modules/semantic/research/offline_relink_candidate_analysis.md)

## Artifacts

- [machine study spec](../../modules/semantic/research/observability_weighted_directional_likelihood_study_v1.json)
- [`observability_weighted_directional_likelihood_study_schema_v1.json`](../../../scripts/tools/observability_weighted_directional_likelihood_study_schema_v1.json)
- [SR2 declaration record](../../modules/semantic/research/observability_weighted_directional_likelihood_declaration_20260827.score.json)
- [`observability_weighted_directional_likelihood.py`](../../../scripts/tools/observability_weighted_directional_likelihood.py)
- [`test_observability_weighted_directional_likelihood.py`](../../../tests/unit/eval/diagnostics/test_observability_weighted_directional_likelihood.py)
- [seal receipt 2026-09-20](../../modules/semantic/research/observability_weighted_directional_likelihood_seal_receipt_20260920.json)
  (`declaration_seal_head`, sealed-file digests, source hashes, custody root, open-item dispositions)

## Current step

**PARKED after seal.** The sealed object is
`311c222580dd29fcecacd5e2cc7cf6b2b459e684` (merge of PR #398, the head at which
pre-seal was confirmed complete; every sealed file is byte-identical to that
tree on `main`). No runner-review phase is scheduled. Do not load or summarize
formal B1 outcome rows, compute metrics, or add the runner.

Resume requires an explicit owner scheduling decision that names the
runner-review phase and reacquires the semantic sole-active slot; until then
this card is not executable work. Resume preconditions:

- sealed-file raw byte identity against `declaration_seal_head` still holds;
- the nine frozen sources verify 9/9 at the paths the runner reads — repo paths
  or the read-only custody root
  `/home/ray/owdl_custody/owdl_m_b1_v1_seal_311c2225_20260920/` (SHA256SUMS
  self-sealed). If neither reproduces the bytes, the first *Discard when*
  condition applies.

Any later `runner_review_head` has this **first gate**, before runner
authority is even considered:

```text
sealed declaration bytes == runner declaration bytes
```

Failure is a terminal. "Looks the same", pretty-print, JSON key reordering,
and any other re-normalization are not admissible substitutes for raw byte
identity.

## Acceptance

- declaration separates phenomenon, ranking gap, and non-claims, and separates
  the empirical hypothesis from v1's conservative modeling convention;
- source hashes and candidate/event semantics fail closed, and relational
  integrity is frozen now but executed only post-seal;
- cross-covariance and candidate-specific von Mises normalizer are implemented;
- concentration is resultant-matched, not the small-angle `1/variance` shortcut;
- exact-zero direction becomes uniform without a speed threshold;
- the positive handoff is cross-dataset confirmation, not integration design;
- formal CLI execution is unavailable before seal;
- relevant unit, declaration, document, and structure checks pass.

## Must not

- do not load or summarize formal outcome rows before seal;
- do not tune bins, covariance repair, effects, or protected strata from results;
- do not claim runtime fidelity, assignment impact, or MOT improvement;
- do not change a production preset or tracker hook;
- do not auto-continue from any terminal;
- do not name a seal by branch or other movable ref instead of exact commit SHA;
- do not pass runner-review identity by semantic equivalence or re-normalization.

## History

- 2026-08-27 — owner requested a research plan and implementation start;
  charter activated at pre-seal implementation only.
- 2026-09-12 — pre-seal implementation closed for owner seal review: math
  core, synthetic tests, check-only identity preflight (`formal_rows_read=0`),
  and document checks. Formal execution remains unauthorized.
- 2026-09-12 — owner confirmed pre-seal complete and pinned two receipt/gate
  rules without changing frozen bytes: the seal artifact records the exact
  merged commit SHA; the first runner-review gate is raw byte identity of
  the sealed declaration. Status = `WAITING_OWNER_SEAL`.
- 2026-09-20 — owner decision on [#442](https://github.com/raylei50653/saccade/issues/442):
  **seal, then park.** Seal receipt issued with
  `declaration_seal_head = 311c222580dd29fcecacd5e2cc7cf6b2b459e684` (PR #398
  merge; PR #326 merge `0e869fea` was not chosen because its test file
  predates the reviewed one, while every other sealed file is identical).
  Nine frozen sources verified 9/9 by byte identity (`formal_rows_read: 0`).
  Two open items dispositioned: (1) pre-seal label-free exposure pre-count —
  **not permitted, not taken**; an exposure shortfall at execution is an
  accepted `OWDL_INVALID_STUDY` risk; (2) frozen-source custody — **read-only
  custody copy made** at `/home/ray/owdl_custody/owdl_m_b1_v1_seal_311c2225_20260920/`.
  Pause reason: no runner-review phase scheduled; the semantic sole-active
  slot is released (`⏸️ 無 active`). No execution, runner, or registry state
  change is authorized by the seal.
