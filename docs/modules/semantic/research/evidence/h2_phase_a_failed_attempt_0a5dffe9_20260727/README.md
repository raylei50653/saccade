# H2 Phase-A seal attempt at `0a5dffe9` — failure evidence

<!-- doc-status: terminal -->
<!-- doc-date: 2026-07-27 -->
<!-- doc-module: semantic -->

> **No capture. Authorization spent. Controller self-mutation defect.**
>
> This directory is the durable record of the single authorized H2 Phase-A
> invocation of 2026-07-27. It is **failure evidence for a successor controller
> repair**. It is *not* measurement evidence: nothing here may be read as a
> measurement result, a seal, an equivalence statement, or a basis for any
> downstream unit.

State ownership is unchanged: the
[claim-state registry](../../../../../research/contracts/claim_state_registry.md)
is the sole writer of `quantity.bridge_capture_provenance` state (C5.1), the
[H2 declaration](../../headline_bridge_behavioral_identity_capture_declaration_20260725.md)
owns the identity mechanism and terminal partition, and the
[H2 charter](../../../../../research/threads/h2_behavioral_identity_capture_task.md)
owns navigation. This directory owns only the artifacts and their chain of
custody.

## 1. What was authorized and what was spent

| Item | Value |
| --- | --- |
| Authorization | single invocation, granted 2026-07-27, **consumed at launch** |
| `source_head` (`I40`) | `0a5dffe921d78fce8e525baf8b4b624fc9ab957c` |
| `source_tree` | `5530be2d67b8e7c83a7a858a44a2b11a1c347927` |
| `F64` | `a03fc4590ca931435fde4a93f28bec8ed156fe852718cd214e780e002d97fd8b` |
| Layer-P certificate digest | `d95859cb3cc27eeadb72b0f94fdcf45107c590058dd2c288e14bcd47c3e24802` |
| Layer-P `selected_base` | `b2f3c23f419cb03cf89eae677bdf9262a8dd3634` |
| Bounded probe | `2dabed0bc05e3bc75ec2115b3213f5c0b1aed3e837c22dd2325109339e4719b5` |
| Equivalence | `unproven`, unchanged |
| Launched | 2026-07-27T15:22:31Z |
| Terminated | 2026-07-27T15:22:37Z |
| Controller exit code | `2` |

Preconditions were independently verified **before** launch and did hold: the
controlled-host re-attestation of this exact head was green
(run [`30276844285`](https://github.com/raylei50653/saccade/actions/runs/30276844285)),
the Layer-P certificate verified 37/37 against primary sources, and the freeze
record verified 22/22 — including the controller's own terminal-1 predicate
returning zero mismatch reasons in a read-only dry run.

## 2. Outcome

```
terminal   H2_INPUT_MUTATED_DURING_MEASUREMENT   (order 1, phase a)
result     input_mutated
```

The four ordered runs `00_capture_off`, `01/02/03_capture_on` **never started**.
The archive contains no `runs/` directory. `capture_off_on_equal: true` and
`packets_valid: true` in `observation.json` are unexercised defaults and support
no conclusion of any kind.

Exit code 2 is not the terminal. The independent verifier refused the archive:

```
H2 Phase-A controller rejected: recorded Layer-P certificate match disagrees
with the archived freeze/certificate/content bindings and independent Git-tree
recomputation
```

so this attempt produced no verifier report either.

## 3. The terminal label misdescribes the root cause

No bound input was mutated by anything external. The recorded event is:

> the controller created its own evidence root inside the repository, then
> classified that same self-produced artifact as an execution-checkout mutation.

`H2_INPUT_MUTATED_DURING_MEASUREMENT` is the label the controller *recorded*; it
is not a correct semantic description of the root cause, and a successor repair
must be planned against the cause rather than the label.

## 4. Two independent defects

### 4.1 The evidence-root and clean-checkout invariants cannot both hold

Control flow, with positions at the attempted head `0a5dffe9`:

1. `scripts/tools/run_h2_measurement.py:1050-1053` — require
   `git status --porcelain --untracked-files=normal` to be empty. **Passed**: the
   checkout was clean at launch.
2. `scripts/tools/run_h2_measurement.py:1054-1057` — create the evidence root at
   `EVIDENCE_REL / h2_measure_<I40>.incomplete`, where
   `EVIDENCE_REL = "docs/modules/semantic/research/evidence"`
   (`scripts/tools/h2_measurement_evidence.py:103`). That path is **not**
   gitignored.
3. `scripts/tools/run_h2_measurement.py:1140-1143` — require the same `git
   status` to be empty again. It cannot be: step 2 made it dirty.
4. `scripts/tools/run_h2_measurement.py:1291-1293` — the stop boundary repeats
   the check and appends `execution checkout became dirty before stop
   linearization`.

There is no successful path through this control flow. The defect is independent
of host, freeze, certificate, probe and Layer-P outcome, and reproduces at every
head.

### 4.2 Predicate ownership contamination

`scripts/tools/run_h2_measurement.py:1140-1145` folds a checkout-hygiene reason
into the same `reasons` list as genuine certificate/freeze/content binding
mismatches, then sets
`predicates["layer_p_certificate_matches_freeze"] = not reasons`.

The archived `controller.json` records exactly one reason and **zero** real
certificate mismatches:

```json
"certificate_mismatch_reasons": ["execution checkout changed before monitored revalidation"]
```

The bindings themselves were correct. The archive therefore contradicts itself —
producer records `false`, independent recomputation from archived content returns
*match* — and the verifier's refusal is correct. The defect is in the predicate
semantics the producer records, not in the verifier.

This is also why `scripts/tools/check_h2_measure_archives.py` rejects a corpus
containing the raw archive, which is one reason the archive is nested under
`controller_archive/` here rather than left at the corpus-scanned
`h2_measure_*` position.

## 5. Contents

| Path | What it is |
| --- | --- |
| `controller_archive/h2_measure_0a5dffe921…/` | the controller's archive, 13 files, verbatim, no `runs/`; carries its own `checksums.sha256` (12 entries, passes) |
| `controller_stderr.txt` | the controller's raw output, byte-identical to the captured `h2_measure.log`; renamed only because `.gitignore:43` ignores `*.log` and forcing an ignored path into a governance interface is worse than a suffix |
| `controller_exit_code.txt` | `2` |
| `layer_p_retry_log.jsonl` | append-only Layer-P log: `plumbing_blocked` → `plumbing_pass` → `plumbing_pass` |
| `independent_verifiers/*.py.txt` | the pre-launch verifiers, as inert text |
| `manifest.json` | packet manifest (`h2_phase_a_failure_registration_v1`) — the adjudicated facts as machine-readable fields |
| `registration.json` | the attempt's bindings, the four defect sites, the custody incident, and the original untracked locations |
| `SHA256SUMS.json` | authoritative inventory of all 22 files, packet-relative |

The launch bundle is **not** duplicated here: the controller archived it
byte-identically. `freeze.json`, `layer_p_certificate.json`, `reference_probe.json`,
`runtime_inputs.json` and `published_identity.json` inside `controller_archive/`
are byte-equal to the untracked `out/h2_seal/freeze.json`,
`out/h2_layer_p/cert_full_base.json` and the `out/h2_layer_p/20260727T151357Z/`
records they were launched from.

The verifier scripts are stored as `.py.txt` deliberately: `scripts/pre_push.sh`
runs `ruff format .` across the repository, which would rewrite frozen evidence
in place and break these checksums.

## 6. What this attempt leaves valid, and what may not be done with it

Still valid as predecessor evidence *for `0a5dffe9` only*: the controlled-host
re-attestation, the published coordinate and probe, the Layer-P certificate, the
freeze record and the launch bundle. They establish pre-launch conditions and
nothing downstream of launch.

Not permitted as a rescue of this head: adding a `.gitignore` entry,
pre-creating or relocating the evidence root, symlinking or mounting around
`git status`, editing archived predicates, skipping the independent verifier, or
reclassifying this invocation as a preflight in order to reuse the spent
authorization. Each either changes the execution semantics under test or breaks
no-silent-iteration.

## 7. Chain-of-custody incident during registration — disclosed

While assembling this record the assistant contaminated the evidence root and
then repaired it. Disclosed because an evidence root silently modified after the
fact is worth less than one whose whole history is on the record.

A checksum verification had been run with `cd <archive>`; the shell working
directory persisted into the next command, which used relative paths — so
`out/h2_seal/failure_scene_20260727/` and five files were created **inside** the
archive instead of beside it. `evidence_files()` recurses, so the corpus checker
rejected the root with `evidence files are absent from the inventory`.

What was and was not affected:

* the controller's twelve inventoried files were **never** modified —
  `sha256sum -c checksums.sha256` returned 12/12 before the contamination, from
  inside the contaminated tree, and again after the repair;
* the contamination was purely additive: five files and two empty directories,
  none of them named in `checksums.sha256`;
* the repair was a **move**, not a delete — nothing was discarded;
* afterwards the archive holds exactly its original thirteen files, and the
  corpus checker's complaint changed to the genuine §4.2 defect.

A repository-external read-only copy taken before the repair captured the
contaminated tree and was rebuilt from the restored archive.

## 8. Successor path

1. this registration;
2. the claim-state registry and charter declaration landed with it;
3. controller repair on a **successor head**, in its own PR — at minimum §4.1's
   evidence-root placement and §4.2's predicate ownership;
4. re-review of the full launch → stop-boundary ordering, simulating every side
   effect in control-flow order;
5. rebuilt acceptance gate 2, `Acceptance` items 4 and 5, and a new `F` bound to
   the successor head — the repair moves execution-relevant code, so none of the
   `0a5dffe9` bindings transfer;
6. only then a new single-invocation authorization.

The repair PR may not describe itself as continuing this seal attempt.

## 9. Review rule this attempt establishes

> For any controller that creates evidence, locks, temporary roots or checkout
> witnesses, pre-authorization review must simulate every invariant *after* the
> side effects that precede it, in the controller's actual control-flow order.
> A dry run of a predicate in isolation does not demonstrate launchability.
