# H2 Phase-A seal attempt at `7646f421` — failure evidence

<!-- doc-status: terminal -->
<!-- doc-date: 2026-07-28 -->
<!-- doc-module: semantic -->

> **No capture. A second authorization spent. Child environment-validation
> ordering defect.**
>
> This directory is the durable record of the second authorized H2 Phase-A
> invocation, 2026-07-28. It is **failure evidence for a successor
> execution-and-archive-verifier repair** — registering it exposed a third defect
> of the same shape, in the archive verifier itself (§4.2). It is *not* measurement evidence: nothing here, and nothing in the
> controller archive it points at, may be read as a measurement result, a seal,
> an equivalence statement, or a basis for any downstream unit.

State ownership is unchanged: the
[claim-state registry](../../../../../research/contracts/claim_state_registry.md)
is the sole writer of `quantity.bridge_capture_provenance` state (C5.1), the
[H2 declaration](../../headline_bridge_behavioral_identity_capture_declaration_20260725.md)
owns the identity mechanism and terminal partition, and the
[H2 charter](../../../../../research/threads/h2_behavioral_identity_capture_task.md)
owns navigation. This directory owns only the adjudication and the chain of
custody.

The predecessor attempt is
[`h2_phase_a_failed_attempt_0a5dffe9_20260727/`](../h2_phase_a_failed_attempt_0a5dffe9_20260727/).
This attempt is its successor, not its continuation: the authorization consumed
here is a second, separately issued one.

## 1. What was authorized and what was spent

| Item | Value |
| --- | --- |
| Authorization | single invocation, issued 2026-07-28, **consumed at launch** |
| `authorization_id` | `342416678caa310ce0e73a2805ddf04767b675ffe3c5928f95d51671b8646234` |
| `invocation_id` | `4410f5e9726240cf659928be737d560a6f4dd17c856fe87b38b446509ad5aab6` |
| `source_head` (`I40`) | `7646f421a85a580e37e457def5e8ddc7c4bfa0ab` |
| `source_tree` | `79ea5ae0ca6c69d7273d558dfaae9e08d6e1a64f` |
| `F64` | `f0d1b02e5a162d4949bb2db00f30d73242e7c4a8a833400b712f378c91d31ce4` |
| Layer-P certificate | object `e60b98e6f7a2823e9921eac1b2f374d7391c686c433602b1dd41c2c04e1c1618`, file `266f4b4ca5b891639d885f795f77ef603bb0b6877990a29922054af06e63d3e2` |
| Layer-P `selected_base` | `7646f421a85a580e37e457def5e8ddc7c4bfa0ab`, `changed_count: 0` |
| Bounded probe | `2dabed0bc05e3bc75ec2115b3213f5c0b1aed3e837c22dd2325109339e4719b5` |
| Equivalence | `unproven`, unchanged |
| Launched | 2026-07-28T14:31:26Z |
| Terminated | 2026-07-28T14:32:06Z |

Preconditions were independently verified before launch and did hold: the
controlled-host re-attestation of this exact head was green (run
[`30334080842`](https://github.com/raylei50653/saccade/actions/runs/30334080842)),
the Layer-P certificate verified 65/65 against primary sources, and the freeze
record verified 51/51.

The object/file digest pair for the certificate differs by design: this unit's
canonical-object digest carries no trailing newline while the file on disk does.

## 2. Outcome

```
terminal   H2_MEASUREMENT_EXECUTION_INVALID   (order 4, phase a)
result     runner_nonzero
```

The first ordered run, `00_capture_off`, launched and exited non-zero. Runs
`01/02/03_capture_on` were never reached. **Zero faithful capture**: the archive
contains one `runs/MOT17-04-SDP/00_capture_off/` directory holding an
invocation record and two logs, and no packet, inventory or MOT output of any
kind.

The child's entire diagnostic output was one line:

```
H2 measurement child rejected: child environment keys differ from the frozen A5
execution environment
```

Unlike the 2026-07-27 attempt, the recorded terminal **is** a correct semantic
description: execution was invalid, and the partition selected the execution
catch-all for exactly that reason.

## 3. What the `0a5dffe9` repair did fix

Recording this separately because it is a real state transition and must not be
lost inside a second failure. Every defect registered against the predecessor
attempt is closed at this head, and the parts of the machinery downstream of
launch ran for the first time:

* the controller reached `child_launch` — the predecessor never started a run;
* checkout hygiene passed at launch *and* at the stop boundary
  (`checkout_clean: true`, `checkout_hygiene_reasons: []`);
* predicate ownership is clean: `certificate_mismatch_reasons: []`,
  `layer_p_certificate_matches_freeze: true`, `bound_input_mutated: false` with
  an empty event list;
* the stop boundary linearized as `clean_final_drain` with the monitor started,
  revalidated and closed;
* the evidence root finalized at its canonical corpus position with a complete
  28-entry `checksums.sha256`;
* the independent verifier **accepts** the archive —
  `valid: true`, `verify_class: complete` — and
  `check_h2_measure_archives.py` reports `PASS (1 roots; complete=1)`. Both
  results were produced on the execution host; §4.2 records why that
  qualification is currently necessary.

This is why the archive is committed at
[`h2_measure_7646f421a85a580e37e457def5e8ddc7c4bfa0ab/`](../h2_measure_7646f421a85a580e37e457def5e8ddc7c4bfa0ab/)
rather than nested inside this packet as the predecessor's had to be.

## 4. Root cause — measured, not argued

`cv2` 4.11.0 rewrites the process environment as an import side effect. Under
the child's sanitized 17-key environment it **adds** `QT_QPA_FONTDIR` and
`QT_QPA_PLATFORM_PLUGIN_PATH` and **prepends its own lib directory to**
`LD_LIBRARY_PATH`. `cv2` is reached transitively by
`h2_behavioral_identity._import_eval_stack()`.

The child validates its environment at two points:

| Site | When | Verdict |
| --- | --- | --- |
| `scripts/tools/run_h2_measurement_child.py:683` (`execute_child`) | ingress, before any import | **passes** |
| `scripts/tools/run_h2_measurement_child.py:298` (`repository_runner`) | after `_import_eval_stack()` at `:271` and `configure_runtime_env` at `:297` | **cannot pass** |

Both call sites apply the *same* predicate — `validate_environment` at `:176`,
which requires `set(environment) == h0_child.EXPECTED_ENV_KEYS` and
`_environment_digest(environment) == invocation["environment_digest"]`. The
second call therefore re-applies an ingress contract to an environment that
third-party import side effects have legitimately changed, and the result does
not depend on the host's GPU, build, dataset or model state. Any host with
OpenCV installed fails identically. This is structural self-negation, the second
consecutive one.

`root_cause_probe.py.txt` reproduces it end to end without any authorization,
and `root_cause_probe.json` is its output:

```json
{"ingress": "pass",
 "import_eval_stack": "ok",
 "cv2_version": "4.11.0",
 "added_keys": ["QT_QPA_FONTDIR", "QT_QPA_PLATFORM_PLUGIN_PATH"],
 "changed_values": ["LD_LIBRARY_PATH"],
 "post_import": "ChildError: child environment keys differ from the frozen A5 execution environment"}
```

Two consequences follow, and the second is the one that would have been missed:

* the *ingress* gate is correctly placed and correctly passed. The controller
  builds the child environment from scratch and asserts its key set before
  launch (`scripts/tools/run_h2_measurement.py:602-629`), the probe measures
  that set as exactly the 17 expected keys, and the child's own ingress check
  accepted it. The launch authorization decision was sound;
* **removing the two added keys from the comparison would not be sufficient.**
  `LD_LIBRARY_PATH` is already an expected key, so its mutation changes no key
  set — but it does change `_environment_digest`, so the digest branch of the
  same predicate fails next. The defect has two layers, and a key-set-only fix
  reaches the same terminal one line later.

### 4.1 The same shape in the frozen H0 child, latent

`scripts/tools/run_h0_phase_a_child.py:372` re-checks
`set(os.environ) != EXPECTED_ENV_KEYS` after the eval-stack import at `:346-355`
and `configure_runtime_env` at `:371` — the identical post-import re-application.
H0 escaped it only because its five invocations reached
`H0_PROVENANCE_INVALID` before that line. H0's ingress gate at `:796`
(`_initial_environment_gate`) is correctly pre-import, so the frozen child's
structure is *ingress-correct, post-hoc-defective*, exactly like H2's.

This file is frozen and hash-pinned and **must not be edited** by the H2 repair.
It is recorded here as a known latent defect, not as work.

### 4.2 A third instance of the same shape, found while registering this one

Registering the archive surfaced a defect that no previous attempt could have
exposed, because no H2 archive had ever been committed and verified anywhere but
on the machine that produced it.

`verify_h2_measurement._authorization` recomputes the authorization execution
domain by calling `h2_measurement_evidence.authorization_execution_domain`, which
reads the **verifying** host's `/etc/machine-id` and `os.getuid()`
(`h2_measurement_evidence.py:244-264`), and then requires
`expected_domain == execution_domain` against the archived record. A Phase-A
archive therefore verifies only on its own execution host. On a CI runner the
verifier refuses it with

```
Phase-A authorization grant/consumption record is absent, malformed,
digest-unreconstructable, or bound to another head/freeze/controller/invocation
```

— which is true of the runner, not of the archive. Measured: the live
recomputation on this host reproduces the archived object exactly, and the same
call on a GitHub runner does not.

Binding the *grant* to one host, uid and ledger namespace is intended and
correct: it is what stops an authorization from being carried to another machine
before launch. Carrying that live recomputation into *archive* verification is
the same structural error as the other two — a check that consults live host
state where the archived value is the thing under test — and it defeats the
verifier's own purpose, since an independent reviewer is by definition not on the
execution host. The internal consistency that archive verification can honestly
assert is already computed beside it: `receipt.execution_domain` and
`grant.execution_domain` must equal `digest(execution_domain)` of the archived
object.

This is registered, not repaired. `verify_h2_measurement.py` is an executed
surface bound into `F`, and the repair belongs with the child fix, not with the
registration of the failure that revealed it. Two consequences hold in the
meantime: the archive's verifier report in §3 is a statement about *this host*,
and the CI gate this PR installs is the host-independent inventory contract
rather than full re-verification. H0's corpus does not share the defect — its
verifier is guarded by an explicit host-independence test, which H2's never had.

## 5. Why pre-authorization review did not catch it

The 2026-07-27 review rule — simulate every invariant after the side effects
that precede it, in the controller's actual control-flow order — was applied to
the **controller** and honoured there. It was not applied across the process
boundary: the child was reviewed as source and exercised only through unit tests
with synthetic environments, never as a real process that imports the real eval
stack under a real `child_environment()`.

The launch probe is what makes this precise. It ran the full eval stack on
MOT17-09-SDP and succeeded (`launch_probe.json`, digest equal to the reference
probe), so the eval stack demonstrably imports on this host — but the probe is a
subprocess launched with `{**os.environ, ...}`
(`scripts/tools/run_h2_measurement.py:654`), i.e. the operator's inherited
environment, never the sanitized 17-key one. A green launch probe therefore
carries no information about whether the child can survive its own environment
contract, and it was read as if it did.

## 6. Contents

| Path | What it is |
| --- | --- |
| `README.md` | this adjudication |
| `registration.json` | the attempt's bindings, defect sites and custody, machine-readable |
| `root_cause_probe.py.txt` | the reproduction, as inert text |
| `root_cause_probe.json` | its output, produced on the execution host 2026-07-29 |
| `SHA256SUMS.json` | authoritative inventory of every file here except itself |

The controller archive is **not** duplicated here. It lives at its canonical
corpus position, `../h2_measure_7646f421a85a580e37e457def5e8ddc7c4bfa0ab/`, and
is committed verbatim with its own 28-entry `checksums.sha256`. CI enforces that
inventory — total in both directions, every digest recomputed, no symlinks — on
every run. Full re-verification and the corpus checker are **not** wired into CI
yet; §4.2 says why, and the repair PR is where that changes.

The probe is stored as `.py.txt` for the same reason as the predecessor's
verifiers: `scripts/pre_push.sh` runs `ruff format .` across the repository,
which would rewrite frozen evidence in place and break these checksums.

Committing the archive required a `.gitignore` change, recorded here because it
is governance-visible. Five of its 28 inventoried files matched the repository's
generic `runs/` and `*.log` rules — including
`runs/MOT17-04-SDP/00_capture_off/stderr.log`, the child's entire diagnostic
output. A plain `git add` drops them silently and yields a committed archive
that fails its own completeness check. Two negations scoped to the evidence root
were added instead of force-adding ignored paths, so the next archive cannot be
truncated by remembering or forgetting a flag. The archive itself is unchanged:
no file was renamed to dodge an ignore rule.

## 7. Chain of custody

A repository-external read-only copy of the controller archive was taken before
any git operation and verified against the archive's own inventory:

```
/home/ray/h2_phase_a_failed_attempt_7646f421_20260728   28/28 OK, mode a-w
```

No file inside the controller archive was created, modified, moved or deleted
during registration. The archive committed here is byte-identical to what the
controller finalized on 2026-07-28.

One artifact does **not** exist and is not reconstructed: the controller's own
console output was not redirected to a file for this invocation, so there is no
`controller_stderr.txt` counterpart to the predecessor's. Everything the
controller adjudicated is in `controller.json`, `observation.json`,
`terminal.json` and `lifecycle_events.jsonl`; the child's stderr survives inside
the archive.

## 8. What this attempt leaves valid, and what may not be done with it

Still valid as predecessor evidence *for `7646f421` only*: the controlled-host
re-attestation, the published coordinate and probe, the Layer-P certificate, the
freeze record and the launch bundle. They establish pre-launch conditions and
nothing downstream of launch.

`F64 f0d1b02e…` and Layer-P certificate `266f4b4c…` are **stale from the moment
the child repair lands**: the certificate binds `source_head`, the repair moves
the head, and no part of either transfers. They are not re-signed, not partially
reused, and not carried forward.

Not permitted as a rescue of this head: re-running under the spent
authorization, reclassifying this invocation as a preflight, widening
`EXPECTED_ENV_KEYS` (it is the frozen H0 ruler, imported, not H2's to change),
editing archived predicates, or skipping the independent verifier. Each either
changes the execution semantics under test or breaks no-silent-iteration.

## 9. Successor path

1. this registration;
2. the claim-state registry and charter update landed with it;
3. the execution-and-archive-verifier repair on a **successor head**, in its own
   PR — two independent commits, since §4.2's verifier defect is a second
   executed surface that also makes `F`/certificate stale and so must land at the
   same final head. The child half's owner-adopted
   shape is: keep the ingress gate, drop the post-import re-application of it,
   check `configure_runtime_env`'s repo-owned mutation against a snapshot taken
   between the import and that call, and record the `pre_import → post_import`
   delta as diagnostic that never re-enters an authorization decision;
4. a non-evidence full run — controller → child process → eval-stack import →
   environment validation → capture initialisation → first valid stop boundary —
   consuming no authorization and writing no evidence root;
5. rebuilt `Acceptance` items 4 and 5 and a new `F` at the successor head;
6. only then a third single-invocation authorization, separately issued.

The repair PR may not describe itself as continuing this seal attempt.

## 10. Review rule this attempt establishes

> A process boundary is not a review boundary. For any controller that launches
> a child under a constructed environment, pre-authorization review must run the
> child as a real process under that exact constructed environment, through its
> real imports, up to its first stop boundary. Source review, unit tests with
> synthetic environments, and a green in-controller probe running under the
> *operator's* environment are jointly insufficient — none of them observes what
> the child's own dependencies do to the contract it is required to satisfy.
