# H0-R5 Qualification-to-Authoritative Extension-Load Parity Audit

**Document type:** diagnostic parity audit (non-authoritative)  
**Date:** 2026-07-25  
**Terminal:** `H0_R5_PARITY_DIVERGENCE_IDENTIFIED`  
**Authority:** does **not** establish H0 baseline, actual guarantee, Phase B, or re-authorization  

## Scope

Read-only-first audit of the chain:

```text
qualification head 205a8117…
→ selected instrumentation I 524f7e3b…
→ freeze F 6e425dc6…
→ seal S 6fdb060c…
→ authoritative evidence h0_phase_a_524f7e3b…
→ H0_PROVENANCE_INVALID (extension_load / runtime attestation absent)
```

Out of scope: trace-v2 scoring, packet correctness, Phase B, GCTM, B1/O1, thresholds, capture experiments, new I/F/S, R6 authorization, modification of PR #284 evidence.

## Executive answer

Qualification extension-load **and** the authoritative controller call the **same** membership predicate (`assert_extension_plugin_membership` in `h0_runtime_confinement.py`, byte-identical across the qualified head and I).  

They do **not** use the same **child load vector**.

| Surface | Before `import saccade_tracking_ext` |
|--------|--------------------------------------|
| Qualification (`_qualify_confined_extension_load`) | `sys.path.insert(0, extension.parent)` |
| Authoritative (`_verify_extension_load`) | **no** insert |
| Maps dependency helper (`_runtime_maps_dependencies`) | `sys.path.insert(0, extension.parent)` (comments claim it matches the controller vector — it does not) |

Under `python -I -B`, site is still loaded (`-I` does not imply `-S`).  
`.venv/.../saccade_build.pth` injects `<repo>/build` into `sys.path`.

Without the insert, import binds:

```text
build/saccade_tracking_ext.cpython-312-x86_64-linux-gnu.so
```

not the expected:

```text
build/h0_phase_a/saccade_tracking_ext.cpython-312-x86_64-linux-gnu.so
```

Membership then fails for the expected realpath → controller raises DriftError  
`extension/plugin load is absent from runtime attestation` → `provenance_invalid`.

This was **stably reproduced** with non-authoritative host import probes (Case C / C2). Full confined re-spawn and authoritative re-launch were **not** performed.

## Phase results

### Phase 1 — Identity matrix

See `qualification_authoritative_identity_matrix.json`.

Highlights:

| Identity | Qualification | Authoritative / F | Equal |
|----------|---------------|-------------------|-------|
| Head / tree | `205a8117…` / `6d41fa03…` | I `524f7e3b…` / `c2ce782f…` | **no** |
| Membership implementation | `h0_runtime_confinement` `8e9ee434…` | same | **yes** |
| Controller implementation | `run_h0_phase_a` `205855b8…` | same at QUAL head and I | **yes** |
| Confinement plan digest | `c769a226…` | `not_recorded` (DriftError before write) | n/a |
| Runtime inputs digest | `e874fe4e…` | `not_produced` for child runs | n/a |
| Bound inputs digest | n/a (reduced inventory) | F/auth `a8e4bece…` | n/a |
| Tool runtime list | not recorded at F scale | F == auth (4518 members) | **yes** (F↔auth) |
| Extension sha256 | `f374a223…` (qual build dir) | `b064a700…` (h0_phase_a) | **no** (separate rebuilds) |

### Phase 2 — Path comparison

See `qualification_authoritative_path_comparison.json`.

Material differences: entrypoint child script, build directories, confinement inventory scale.  
Shared and equal: membership predicate, recorder backend (`landlock_seccomp_ptrace_v1`), confinement `build_plan` implementation.

### Phase 3 — Exact-head inheritance

See `qualification_head_inheritance_audit.json`.

Commits `205a8117… → 524f7e3b…`:

1. `40fa5006` — docs packet rebind (evidence only)  
2. `bddc970a` — merge PR #281  
3. `524f7e3b` — **execution-bound**: `qualify_h0_phase_a.py`, `verify_h0_preseal_freeze.py`, tests  

**Overall inheritance verdict:** `INVALIDATED_AND_BEHAVIORALLY_DIFFERENT`  

Contract forbids inheritance after post-qualification execution-bound commits. Do **not** rewrite as valid inheritance even where extension-load function bodies are equal.

**Extension-load-specific note:** load scripts and membership are byte-equal across that range. The pass/fail parity defect **already exists at the qualified head** between qualify vs controller; sealing at exact `205a8117…` would still hit the controller load-vector bug.

### Phase 4 — Reproduction (diagnostic only)

See `parity_reproduction_matrix.json`.

| Case | Result |
|------|--------|
| A | Evidence reuse: qualification passed |
| B | Not re-run (extension-load fn identical to A) |
| C | **Executed:** without insert → loads `build/`, ≠ `h0_phase_a` |
| C2 | **Executed:** with insert → loads `h0_phase_a`, equal |
| D/E | Skipped (safe isolation; not required once C/C2 stable) |

### Phase 5 — First divergence

See `qualification_authoritative_divergence_trace.json`.

**First divergence (unique `first_divergence=true`):**

```text
6_extension_load_child_vector_sys_path_bootstrap
```

Head mismatch is earlier **contractually** but is **not** the causal first difference for pass vs fail (controller already diverged at the qualified head).

### Phase 6 — Seal-gate adequacy

See `seal_gate_adequacy_audit.json`.

Seal / freeze / independent preseal verifier **do not** require:

- qualification head == I  
- qualification tree == I tree  
- qualification report digest bound into F  
- qualification confinement/runtime digests recomputable from F  

Assembler comment documents intentional exclusion of qualification harnesses from implementation bindings. Selecting I after post-qual commits is an **owner-review omission** relative to the qualification contract clause `invalidated_by = any post-qualification execution-bound commit`.

### Phase 7 — Terminal

```text
H0_R5_PARITY_DIVERGENCE_IDENTIFIED
```

## Repair recommendations (charter only — not implemented here)

### Runtime repair (justified)

- Align `_verify_extension_load` (and verifier expected vector) with qualification / maps helper: explicit `sys.path.insert(0, extension.parent)` **or** shared builder + optional `-S` to eliminate ambient `saccade_build.pth`.  
- Fix the false comment in `_runtime_maps_dependencies`.  
- Add negative diagnostic test: under site pth, missing insert binds `build/` not `h0_phase_a`.

### Governance repair (justified; separate PR)

- Fail-closed: qualification `repository_head_sha`/`repository_tree_sha` must equal I before freeze/seal.  
- Bind qualification report digest into F; assembler and independent verifier reject mismatch.  
- Do not treat qualification extension-load pass as substitute for post-build authoritative attestation without vector identity.

### Forbidden until gates exist

```text
new seal forbidden
new authorization forbidden
new authoritative launch forbidden
```

## Deliverables

```text
AUDIT_REPORT.md
VERDICT.txt
qualification_authoritative_identity_matrix.json
qualification_authoritative_path_comparison.json
qualification_head_inheritance_audit.json
parity_reproduction_matrix.json
qualification_authoritative_divergence_trace.json
seal_gate_adequacy_audit.json
```

Directory:

```text
docs/modules/semantic/research/evidence/h0_r5_qualification_authoritative_parity_audit_20260725/
```

## References (read-only)

- Qualification: `.../h0_r5_extension_plugin_attestation_closure_20260724/`  
- Freeze: `.../h0_preseal_freeze_524f7e3b88f73bc366d467d53a2c393a7d3ba937/`  
- Authoritative: `.../h0_phase_a_524f7e3b88f73bc366d467d53a2c393a7d3ba937/`  
- Execution witness: `.../h0_r5_phase_a_execution_witness_20260725/`  
- Code: `scripts/tools/{qualify_h0_phase_a,run_h0_phase_a,h0_runtime_confinement,verify_h0_phase_a}.py`  
