# H0 #209 — post-bootstrap qualification candidate

<!-- doc-status: proposed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-18 -->
<!-- doc-module: semantic -->

> **Status: task-local metadata, non-authoritative.** This document only
> declares an ordinary Repair PR as the sole H0 qualification candidate for
> Issue #209 and binds it to the later controlled-host qualification report. It
> is **not** an execution seal, an H0 terminal, an `I`/`F`/`S` identity, or an
> execution authorization. It does not add H0 research scope, touch the
> controller, schema, verifier, confinement, packets, thresholds, presets,
> production behavior, or any scientific content.

## 1. Purpose

Issue #209 (`H0 post-bootstrap qualification`) requires a fresh authority chain
over the repaired substrate that already landed through PR #205. This document
is the Stage-A task-local qualification-candidate declaration: it names the
Repair PR whose exact final head becomes the single ref submitted to the
`H0 Qualification (Non-authoritative)` workflow, and records the accepted
baseline coordinates the candidate builds on.

## 2. Accepted baseline coordinates

These are already accepted and are not re-opened by this candidate.

```text
Parent issue                     : #206
Accepted repair PR               : #205
#205 repair landing              : 2f1fc1b1826ea77e7a91008133c8be0b14f94c9d
Qualification bootstrap PR       : #208
#208 bootstrap landing           : 52fee2f8a4d720fbf52d36b183af1d44dee1f37c
#208 repair head                 : a99c2db6dedfba4c1d593ad13fbaf7b59de5908f
```

PR #208 introduced the qualification mechanism but is explicitly a
non-authoritative bootstrap. Neither its branch head nor its merge commit is
`I`, `F`, `S`, an H0 terminal, or an execution authorization.

## 3. Candidate identity (R)

The sole qualification candidate is the exact final 40-character head commit of
the Stage-A Repair PR that carries this document.

```text
R = <full repair-head SHA — recorded in Issue #209 and the Repair PR after push>
```

`R` is fixed only once the Repair PR head is stable. Movable branch names,
abbreviated SHAs, merge previews, and GitHub-generated merge refs are not
authority identities. If any new commit is added to the Repair PR (including a
Stage-B corrective batch), the prior `R` and any report tied to it are obsolete
and `R` is re-recorded against the new exact head.

## 4. Stage-A scope of this candidate

This initial Repair PR carries only this task-local declaration binding
Issue #209, the Repair PR, and the later qualification report. It makes no
speculative controller, schema, verifier, confinement, packet, threshold,
preset, production, or scientific change. Historical evidence packets remain
byte-for-byte unchanged.

A single bounded corrective batch may later be added to this same Repair PR
**only** if controlled-host qualification discovers a concrete blocker
(Issue #209 Stage-B correction budget). A second remaining blocker after that
batch yields `REPAIR_RESTART_REQUIRED`.

## 5. Bound qualification report (Stage B)

After `R` is stable, the `H0 Qualification (Non-authoritative)` workflow is
dispatched with the full literal `R` as the requested ref. Its report must
bind the same resolved head and carry:

```text
result                = passed
repository_head_sha   = R
repository_tree_sha   = <resolved 40-character tree SHA>
requested_ref         = R
authority             = non_authoritative
terminal_claim        = forbidden
phase_b               = forbidden
research_inputs       = forbidden
capture               = forbidden
```

Required qualification steps (per
`scripts/tools/h0_repair_acceptance_matrix_v1.json`): `configure`, `build`,
`build_identity`, `runtime_closure`, `extension_load`, `t1_verdict_semantics`,
`runner_launch_preflight`, `failure_envelope_serialization`. The artifact name,
workflow summary, and report must all refer to the same resolved
`repository_head_sha`, and the artifact plus its SHA-256 digest are retained for
owner review.

## 6. Non-authority

Qualification of `R` authorizes nothing further. It does not create `I`, `F`, or
`S`; it does not authorize research capture, controller execution, an H0
terminal, Phase B, or GCTM activation. The owner accepts exact `I` only through
the separate Stage-C declaration in Issue #209. GCTM #175 remains PARKED.

Terminal mapping for this candidate (Issue #209):

```text
qualification_passed                          → READY_FOR_SEAL
qualification_failed_first_bounded_blocker    → ONE_CORRECTION_BATCH_ALLOWED
qualification_failed_after_correction         → REPAIR_RESTART_REQUIRED
```
