---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
work-class: mainline-study
wip-role: sole-active
created: 2026-08-27
---

# Observability-weighted directional likelihood task

## Status

**ACTIVE · mainline-study · sole-active · WAITING_OWNER_SEAL.**
Implementation complete; no formal outcome access authorized. The authority
pointer is [semantic TODO](../../modules/semantic/TODO.md). This activation
does not authorize formal study execution.

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

## Current step

**WAITING_OWNER_SEAL.** Pre-seal implementation is complete. There is no
remaining legal implementation work. Do not load or summarize formal B1
outcome rows, compute metrics, or add the runner.

Owner seal, when issued, records the exact merged commit SHA in the seal
receipt (`declaration_seal_head`), not a branch name.

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
