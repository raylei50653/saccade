# H0-R5 — extension/plugin runtime-attestation closure

<!-- doc-status: terminal -->
<!-- doc-date: 2026-07-24 -->
<!-- doc-module: semantic -->

> **Repair / qualification packet only.** This directory records the sole
> repair unit `h0_extension_plugin_runtime_attestation_closure_v1`, its
> controlled-host isolation probe, mechanical Repair terminal, and
> non-authoritative qualification report. It is **not** a Seal, I/F/S chain,
> execution authorization, H0 baseline, actual registration-v3 guarantee, or
> runtime substrate.

## Owner decision

```text
identity:  h0_r5_extension_plugin_attestation_closure_authorization_20260724
surface:   https://github.com/raylei50653/saccade/issues/280
repair_unit: h0_extension_plugin_runtime_attestation_closure_v1
```

## Spent R4 boundary (immutable)

```text
I4 = 2a233387a6a321dd43570e2e30dc718571b3b4f4
F4 = ced4a4cc6a71473dcb1225203e6d59df0437d976
S4 = a76efffa01a6fb731218150c355f5859bb8e6dd4
authorization: h0_r4_phase_a_exactly_once_authorization_20260724 (#278 closed)
terminal:      H0_PROVENANCE_INVALID (PR #279 merge 55d2da47…)
failure:       extension/plugin load is absent from runtime attestation
```

## Known gap (root cause)

Authoritative extension-load under confinement successfully executed the load
vector for:

```text
build/h0_phase_a/saccade_tracking_ext<EXT_SUFFIX>
build/h0_phase_a/libsaccade_scan_plugin.so
```

but the runtime-attestation recorder either:

1. killed the process before both top-level artifacts were observed (missing
   Python base_prefix / host tool_runtime members; missing dlopen siblings
   such as `libtbbmalloc`); and/or
2. rejected admitted physical files opened via symlink path forms or
   non-canonical `..` loader paths, so membership in
   `h0_runtime_inputs_v1.regular_files` failed.

Controller therefore selected `provenance_invalid` → `H0_PROVENANCE_INVALID`.

## Repair maximum conclusion

```text
A successfully loaded build extension and plugin must each appear as
independently observed runtime-consumed regular files in the attestation,
and controlled-host qualification must exercise the same predicate before
a head may become Seal-eligible.
```

## Files

| File | Role |
|:--|:--|
| `isolation_probe.json` | Non-authoritative controlled-host isolation probe |
| `root_cause_witness.json` | Actual root-cause witness |
| `repair_terminal.json` | Mechanical ordered Repair terminal |
| `qualification_report.json` | Non-authoritative qualification binding exact head SHA |
| `qualification_summary.json` | Controlled-host harness witness (`h0_phase_a_qualification_v1`) |
| `README.md` | This navigation note |

## Non-claims

```text
I selected / F created / S created / SEALED accepted
execution authorized / H0 baseline accepted
runtime substrate established / actual guarantee registered
runtime compatibility established / B1 or O1 activation
```
