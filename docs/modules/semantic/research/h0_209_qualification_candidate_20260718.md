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

## 7. Restart candidate (2026-07-19, append-only)

The first candidate `R = 2702c932ef0c5192d05166de0a62642e2708e742` passed
controlled-host qualification (run 29681971267, report SHA-256
`8b4b46c33c41f70a7b604f5a3a5c3f8068c1b06e9cfd3ac5402770d56806c98c`) but was
found **unsealable** at seal assembly: the pre-seal freeze assembles
`complete = false` because the #214/#216 frozen-CUDA-substrate paths and the
#217 root `DEVELOPMENT.md` edit fall outside the Amendment 6 §A6.2 frozen
admitted runtime surface. Its single Stage-B correction batch was already
spent, so per §4 and the declaration's repair gates the owner declared a
repair restart. That candidate SHA is void as an authority identity and must
not become `I`.

This section declares the restart Repair PR as the new sole candidate. Its
scope is exactly the Amendment 6 Correction 1 batch recorded in the
declaration:

- keep `h0_admitted_runtime_paths_v1` at the five §A6.2 members and add the
  content-pinned table `h0_admitted_runtime_blobs_v1` (exact after-blob
  SHA-256 for `CMakeLists.txt`, `pyproject.toml`, `uv.lock`,
  `src/perception/preprocessor.cpp`; any further change to these files
  requires a fresh admission correction), assembler and independent verifier
  in the same commit;
- classify `DEVELOPMENT.md` as the sole explicit `non_runtime_recorded`
  root-file exception (documentation only, no build/runtime-import edge);
- split the assembler into a head-static sealability half and the
  controller-input/artifact half, and add the required qualification step
  `preseal_freeze_assembly` that calls only the static half (`sealable =
  true`, empty problems, bound to the resolved head; no research input, GPU,
  or inventory access, and no file written) to the harness, the acceptance
  matrix, and its checker, with the passing report required to carry exactly
  the canonical step sequence;
- this restart declaration.

All prior sections of this document continue to bind the new candidate, with
`R` re-read as the exact final head of the restart Repair PR and the §5
required step list re-read as the canonical harness sequence — matrix
`required_steps`, its checker transcription, and the runner's `STEP_NAMES`
are the same exact ten-step tuple (`configure`, `build`, `build_identity`,
`runtime_closure`, `cuda_runtime_confinement`, `extension_load`,
`t1_verdict_semantics`, `runner_launch_preflight`,
`failure_envelope_serialization`, `preseal_freeze_assembly`). The new `R` is
recorded in Issue #209 once the restart PR head is stable and its
host-independent CI is green. Because the restart PR is cut from current `main`, the eventual
`I → F → S` chain is fast-forward-reachable from `main`, restoring the
preferred Stage-D landing.

## 8. Landing-discovery parity amendment (2026-07-20, append-only, Issue #227)

The authorized Stage-E authoritative invocation from the sealed execution
identity `S = 8970841d470371fd7c07df2bd5a367096538c228` (Issue #224) was
**rejected pre-terminal** inside the controller's `_discover_controller_input`:
the independent pre-seal verifier's landing-discovery header (`required_controller`)
was never widened for the `build_tool_binding` member that the #224 assembler and
the full-artifact verifier both carry, so the canonical current artifact was
judged to hold an unknown member and discovery aborted before `execute_controller`
and before T0. No ordered terminal was produced, no evidence packet was created,
and the exactly-once authorization under that `S` is consumed and immutable. The
defect is a controller-input member-declaration divergence, not a build-tool
provenance defect, and is tracked as the bounded repair in Issue #227.

This section extends the canonical qualification tuple for the Issue #227 repair
candidate. The §5 required-step list is re-read as the same exact **eleven-step**
tuple — matrix `required_steps`, its checker transcription, and the runner's
`STEP_NAMES` remain byte-identical to one another — appending one final
non-authoritative step after `preseal_freeze_assembly`:

`configure`, `build`, `build_identity`, `runtime_closure`,
`cuda_runtime_confinement`, `extension_load`, `t1_verdict_semantics`,
`runner_launch_preflight`, `failure_envelope_serialization`,
`preseal_freeze_assembly`, `landing_discovery_dry_run`.

The new `landing_discovery_dry_run` step makes two real, non-authoritative
passes over the checkout's mixed-version evidence tree. First it runs the
per-candidate classification corpus (`_classify_landing_candidates` → the
independent `verify_current_landing_candidate`), proving the current
`build_tool_binding`-bearing artifact and the historical artifacts that predate
it all classify without error and that no more than one current landing is seen.
Second it invokes the controller's actual entry,
`run_h0_phase_a._discover_controller_input`: on an unsealed qualification
checkout the correct fail-closed outcome is the canonical zero-current-landing
contract error (never a verifier rejection over the `build_tool_binding` member,
which was the Stage-E escape); if a real current landing exists the selected
contract must carry the canonical member set. The step performs no research
capture, creates no evidence packet or terminal, consumes no execution
authority, and leaves the candidate checkout and authoritative evidence tree
unchanged.

**Member policy (owner-clarified, Issue #227).** The "each required-member
omission rejected by both paths" rule binds the thirteen stable base members.
`build_tool_binding` is the sole owner-approved cross-version exception:

```text
landing-discovery header (all candidates): build_tool_binding optional
full / selected-current authoritative:     build_tool_binding mandatory
```

The repair collapses the controller-input member set to one canonical
declaration mirrored across the freeze assembler, the controller's landing
discovery, the full-artifact verifier, and the discovery header, with the
execution schema keeping `build_tool_binding` optional so historical evidence
still validates; equality across every runtime transcription, the schema, and an
explicit literal is pinned by a dedicated contract regression that also drives
the actual `_verify_landing_candidate_header`, `_verify_controller_input`, and
`_discover_controller_input` entry points (not only their member helpers).

**Positive-selection scope.** A *positive* `_discover_controller_input` selection
runs the full artifact verification, whose host-fidelity tail
(`_verify_host_execution_inputs`: live GPU/NVML identity, `ldd` build-tool
binding, the entire repository/model/sequence inventory, and a physical `.venv`
at the repository root) is satisfiable only in a real host-bound sealed checkout.
It cannot be reproduced in an ephemeral worktree (which lacks the git-ignored
runtime environment) nor synthesised portably, and an unsealed candidate checkout
has zero current landings by construction.

Stage B qualifies the **unsealed** repair candidate `R₂`, which by ordering
precedes owner acceptance of `I₂` and the `I₂ → F₂ → S₂` chain. Its truthful
expected `landing_discovery_dry_run` outcome is therefore:

```text
Stage B / unsealed R₂:
  mixed-version classification corpus : passes
  actual _discover_controller_input   : invoked
  current landing count               : zero
  terminal boundary                   : canonical fail-closed landing-count error
  independent-verifier rejection      : forbidden (qualification-fatal)
```

This proves the Stage-E escape does not recur; it does **not** exercise a
positive selection. A positive `_discover_controller_input` selection becomes
reachable only after an exact `S₂` exists, and is recorded here as a separate
post-seal readiness gate:

```text
After exact S₂ is landed
AND before any fresh exactly-once execution authorization:

  run a discovery-only controlled-host preflight from exact S₂
  using run_h0_phase_a._discover_controller_input

  require:
    exactly one current landing selected
    selected artifact binds exact I₂ → F₂ → S₂
    returned controller input has the exact canonical 14-member set
    build_tool_binding is present
    no execute_controller call
    no T0 entry
    no evidence packet or terminal
    no exactly-once authorization consumed
```

This preflight is not Stage B qualification, is not the authoritative controller
invocation, and authorizes no execution; it is a post-seal readiness check
required before a separate execution-authorization decision. Stage A proves the
member contract and discovery wiring through the actual entry points above.

All prior sections of this document continue to bind. The Issue #227 candidate
`R`, its exact `I → F → S` chain, and the authoritative re-execution remain gated
on separate controlled-host qualification, owner acceptance of a fresh `I`, and a
new exactly-once execution authorization; this amendment authorizes none of them.
