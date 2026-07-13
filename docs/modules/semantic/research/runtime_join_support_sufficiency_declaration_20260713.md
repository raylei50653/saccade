<!-- doc-status: sealed-for-execution -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-13 -->
<!-- doc-module: semantic -->

# RJ0 — runtime-join support sufficiency — sealed declaration

> **Independent mainline study.** RJ0 is a new L0 feasibility audit, not an S0
> continuation.  It asks only whether a finite, pre-declared runtime-join
> extension could supply enough additional *lost-track* exposure for a future,
> separately declared transfer audit.  It does not decide whether either axis
> transfers, choose a rule, or authorize an implementation or rerun.

## 1. Seal and scope

This declaration is sealed for the audit execution requested on 2026-07-13.
The outcome-blind portions (§§2–5) must be completed and their digest recorded
before any access to `gt_valid` or `gt_match`.  An RJ0 feasibility terminal is
not owner acceptance and creates no follow-on work automatically.

The only frozen inputs are the S0 packet and its D0-v2 substrate:

| Input | Frozen identity |
|---|---|
| capture | `out/signal_study/d0_runtime_shadow_fidelity_20260712T085642Z/capture.csv.gz` — `96093b9b723ed4500b389f8ad74600d75bb49a75064630dd2205cea0b0887047` |
| offline pair universe | `out/signal_study/d0_runtime_shadow_fidelity_20260712T085642Z/pairs.csv` — `ee2898a25ef7f01ed46331c49c12d667846975f25769bc4c3e6b8bad493f8e87` |
| global-ID map | `results/MOT17_eval_d0_shadow_substrate_20260712T085642Z/_global_id_map.txt` — `ae3b6441d1712bcce0826d611cee2cfdf7a01b4d37ec331336f91a0b9148f366` |
| frozen MOT substrate | concatenated sorted `MOT17-*.txt` — `4c5e322a3b8c026de584baa883e26353720837ffa2bf146dfcef2679426a670e` |
| S0 packet | `evidence/s0_safe_domain_runtime_transfer_20260713/{grid.csv,metrics.json,manifest.json}` |

J1 must reproduce the input hashes, S0 output hashes recorded in its manifest,
the partition `1684 + 539 + 354 = 2577`, and event-key version
`d0_event_key_v2_global` with fields `(seq, lost_global_id, cand_global_id)`.
Any mismatch yields `RJ0_INVALID` without a feasibility inference.

## 2. Frozen statistical reference

RJ0 uses the supplied S0 reference point, without reselecting a grid rule:

```text
trial unit = (seq, lost_global_id)
base (N, k) = (116, 3)
epsilon = 0.05; one-sided 95% Clopper–Pearson UCB
```

The best-case floor with `k = 3` is `N = 153`; hence at least 37 distinct,
eligible lost tracks with zero additional hurt would be needed.  The sensitivity
floors are `N = 181` for `k = 4` and `N = 208` for `k = 5`.  These are
feasibility bounds, not a success claim.

## 3. Outcome-blind classification (J2)

Only `cohort_gap` (539 events) and `unemitted` (354 events) are inventoried.
The blind reader projects `pairs.csv` to identity and frozen-coordinate columns
only; it never projects, parses, branches on, serializes, or counts
`gt_valid`, `gt_match`, `gt_lost`, or `gt_cand`.

Each event is assigned exactly one class using the following ordered rules:

1. **exact-key reconstructable** — a valid v2 event key agrees with its three
   global-key fields; exactly one same `(seq, lost_global_id, cand_global_id)`
   row exists in the frozen offline pair universe; and that row has finite
   `dist_h`, positive finite `h_lost_raw`, and positive finite `h_cand_raw`.
2. **deterministic auxiliary-key reconstructable** — the v2 event-key string
   is absent, but all three immutable v2 global-key fields are present and
   identify exactly one same triple with the same coordinate availability in
   the frozen pair universe.  This is a predeclared canonical-field recovery,
   not a nearest-neighbour, temporal, or score-based match.
3. **structurally unjoinable** — no same frozen offline pair exists, a global
   identity is unresolved, or the same-pair frozen coordinates do not exist.
   `unemitted` events are never rescued through `lost_local_id` or
   `cand_local_id`.
4. **provenance ambiguous** — malformed or inconsistent v2 identity,
   duplicate capture identity, non-unique offline identity, or any other
   structural contradiction.

No auxiliary key other than the redundant canonical v2-field triple is in
scope.  In particular, `(seq, lost)`, a gap/frame key, spatial proximity,
local IDs, a coordinate nearest neighbour, and any GT-derived identity are not
keys.  They would attach a runtime candidate to a different offline pair or
change the source population.

## 4. Frozen J3 reduction

Inventory aggregation is solely by `(seq, lost_global_id)`.  A lost track with
many events counts once; events with an unresolved global lost ID have no
eligible trial identity and contribute zero potential exposure.  The packet
must report each partition's event count, identified unique lost-track count,
reconstructable unique-track upper bound, sequence distribution, duplicate
rate, and class/reason distribution.

The resulting `inventory.csv` and its SHA256 are the pre-GT seal.  The J4
reader may only reveal labels for rows whose frozen classification is one of
the two reconstructable classes.  It may not delete, split, or add strata
after seeing labels.

## 5. J4 feasibility envelope and admissibility

For the frozen eligible tracks only, J4 will report the number satisfying
`gt_valid ∧ gt_match`, the additional hurt count that is observable at the
unchanged S0 reference cell, all resulting `(N, k)` combinations, and their
one-sided 95% Clopper–Pearson UCBs.  If no reconstructable track exists, the
GT label projection is empty and the only envelope is the unchanged base
point.

Any candidate wider join must preserve the same population, the exact offline
and runtime coordinate definitions, the lost-track independence unit, and the
frozen axes/grid.  It must not use a local-ID fallback, GT/outcome-conditioned
selection, a new proxy, refitting, threshold changes, or production-state
mutation.

## 6. Ordered terminal mapping

1. `RJ0_INVALID` — provenance, partition, mapping, or frozen-artifact failure.
2. `RJ0_EXPANSION_FUTILE` — even zero extra hurt cannot reach `N = 153`.
3. `RJ0_EXPANSION_INADMISSIBLE` — support might be sufficient only by changing
   population/coordinates/trial unit or by a forbidden join/selection method.
4. `RJ0_EXPANSION_ADMISSIBLE` — deterministic legal support exists and the
   frozen envelope contains at least one `(N, k)` with UCB `<= 0.05`.

An admissible result authorizes only a separately drafted wider-join
implementation/rerun declaration.  No terminal authorizes implementation,
rerunning S0, a closure solve, a score/ranking exercise, a threshold/grid
change, registry/ledger/preset change, or production modification.
