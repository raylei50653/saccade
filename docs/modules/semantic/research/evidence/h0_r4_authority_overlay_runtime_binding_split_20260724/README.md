# H0 R4 — authority-overlay / runtime-binding split

<!-- doc-status: terminal -->
<!-- doc-date: 2026-07-24 -->
<!-- doc-module: semantic -->

> **Repair / qualification packet only.** This directory records the Amendment-10
> repair unit `h0_authority_overlay_runtime_binding_split_v1`, its mechanical
> terminal, and the non-authoritative qualification report bound to an exact
> 40-character head SHA. It is **not** a Seal, I/F/S chain, execution
> authorization, H0 baseline, actual registration-v3 guarantee, or runtime
> substrate.

## Bound authorities

```text
registration terminal:     H0_REGISTRATION_V3_CONTRACT_SEALABLE
registration owner:        h0_registration_v3_terminal_owner_acceptance_20260724
registration identity:     h0_gctm_guarantee_registration_v3
consumer universe:         gctm_runtime_native_candidate_universe_v1
previous H0 terminal:      H0_PROVENANCE_INVALID
spent chain:               I3/F3/S3 (S3 permanently spent)
```

## Known S3 defect

Declaration was simultaneously:

1. an F-frozen runtime-bound repository input; and
2. the S owner-event append target.

Repair removes declaration from runtime inventory and binds it only through
`h0_owner_authority_overlay_v1` with S-byte continuous monitoring.

## Files

| File | Role |
|:--|:--|
| `repair_terminal.json` | Mechanical ordered Repair terminal |
| `qualification_report.json` | Non-authoritative qualification binding exact head SHA |
| `README.md` | This navigation note |

## Non-claims

```text
I selected / F created / S created / SEALED accepted
execution authorized / H0 baseline accepted
runtime substrate established / actual guarantee registered
runtime compatibility established / B1 or O1 activation
```
