<!-- doc-status: sealed-for-execution -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-13 -->
<!-- doc-module: semantic -->

# EK0 — frozen-packet exact-key recoverability — sealed declaration (rev.2)

> **Scope.** EK0 is a bookkeeping-level audit of the frozen D0/S0 packet only.
> It asks one question: can any of the 893 unjoined runtime events be
> recovered, through the exact v2 event key or its redundant canonical-field
> triple, into additional legal lost-track exposure at the unchanged S0
> reference point?  It does not decide whether either axis transfers, choose a
> rule, or authorize an implementation or rerun.

## 0. Rev.2 rescope provenance

Rev.1 of this study executed on 2026-07-13 under the name *RJ0 runtime-join
support sufficiency* with terminal `RJ0_EXPANSION_FUTILE`.  Owner review
(PR #156) found rev.1's claim over-broad and blocked it:

1. Given the exporter's own partition definitions (`cohort_gap` = canonical
   global pair absent from the offline cohort; `unemitted` = global identity
   unresolved), both reconstructable branches are unreachable for a well-formed
   v2 packet, so a zero result is largely entailed by construction.  Rev.1
   nevertheless titled the terminal as if any wider runtime join were futile.
2. The two-phase seal did not bind the runner or the blind metrics between
   phases, nor re-verify frozen inputs at reveal.
3. The N-merge did not exclude lost tracks already present in the base
   exposure, and the terminal mapping conflated "cannot reach the floor" with
   "reaches the floor but fails the UCB".

Rev.2 therefore narrows the claim to exact-key recoverability within the
frozen packet, hardens the seal, corrects the merge and mapping, and renames
every terminal.  The outcome-blind classification rules (§3) are unchanged
from rev.1, and the sealed inventory is byte-identical to the rev.1 packet.

## 1. Seal and scope

This declaration is sealed for the audit execution requested on 2026-07-13.
The outcome-blind portions (§§2–4) must be completed and their digest recorded
before any access to `gt_valid` or `gt_match`.  An EK0 terminal is a statement
about the frozen artifacts only; it is not owner acceptance and creates no
follow-on work automatically.

**Explicit non-claims.**  No EK0 terminal says anything about:

- expanding the frozen offline pair universe (a different cohort export);
- adding identity observability so `unemitted` events resolve to global IDs;
- re-capturing the runtime side, changing exporter semantics, or any other
  "wider runtime join" that changes the frozen artifacts.

Each of those would require its own declaration with its own fidelity and
population obligations.  A negative EK0 terminal must never be cited as
evidence that such extensions are futile.

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
Any mismatch yields `EK0_INVALID` without a recoverability inference.

**Two-phase seal.**  The blind phase records the SHA256 of the declaration,
the runner, the sealed inventory, and the blind metrics in the packet
manifest.  The reveal phase must refuse to run unless *all four* still match
— in particular a runner changed between phases is `EK0_INVALID` — and must
re-verify the frozen hash of every input it reads (`pairs.csv`,
`capture.csv.gz`, S0 `grid.csv`) before any GT projection.

## 2. Frozen statistical reference

EK0 uses the supplied S0 reference point, without reselecting a grid rule:

```text
trial unit = (seq, lost_global_id)
base (N, k) = (116, 3)
epsilon = 0.05; one-sided 95% Clopper–Pearson UCB
```

The best-case floor with `k = 3` is `N = 153`; hence at least 37 distinct,
eligible **new** lost tracks with zero additional hurt would be needed.  The
sensitivity floors are `N = 181` for `k = 4` and `N = 208` for `k = 5`.
These are arithmetic reference bounds, not a success claim.

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

**Expected reachability.**  For a well-formed v2 packet, the exporter's own
partition definitions make both reconstructable classes unreachable:
`cohort_gap` asserts the canonical pair is absent from the offline cohort and
`unemitted` asserts the global identity is unresolved.  A non-empty
reconstructable stratum would therefore indicate an exporter/artifact
inconsistency that rescues events.  EK0's informative content is exactly this
consistency check plus the pinned counts — not a feasibility statement about
joins outside the frozen packet.

## 4. Frozen J3 reduction

Inventory aggregation is solely by `(seq, lost_global_id)`.  A lost track with
many events counts once; events with an unresolved global lost ID have no
eligible trial identity and contribute zero potential exposure.

**Base-overlap exclusion.**  A reconstructable lost track whose
`(seq, lost_global_id)` already appears in the joined (`matched`) partition is
counted separately and contributes **zero** new exposure: it is either one of
the base-`N` trials (counting it again would double-count) or was excluded
there by GT validity (it cannot re-enter as a new trial).  Mutating an
existing base trial's hurt status via its unjoined events is out of scope.
This exclusion is identity-level and outcome-blind.

The packet must report each partition's event count, identified unique
lost-track count, identified-not-in-joined count, reconstructable unique-track
upper bound, reconstructable **new** unique-track count, sequence
distribution, duplicate rate, and class/reason distribution.

The resulting `inventory.csv` and its SHA256 are the pre-GT seal.  The J4
reader may only reveal labels for rows whose frozen classification is one of
the two reconstructable classes and whose lost track passes the base-overlap
exclusion.  It may not delete, split, or add strata after seeing labels.

## 5. J4 envelope and semantic invariants

For the frozen eligible new tracks only, J4 will report the number satisfying
`gt_valid ∧ gt_match`, the additional hurt count that is observable at the
unchanged S0 reference cell, all resulting `(N, k)` combinations, and their
one-sided 95% Clopper–Pearson UCBs.  If no eligible new track exists, the GT
label projection is empty and the only envelope is the unchanged base point.

The following are invariants, not outcomes: same population, exact offline
and runtime coordinate definitions, the lost-track independence unit, and the
frozen axes/grid; no local-ID fallback, GT/outcome-conditioned selection, new
proxy, refitting, threshold change, or production-state mutation.  Violating
any of them makes the packet `EK0_INVALID`; there is no separate
"inadmissible" terminal, because EK0 makes no claim about joins outside the
frozen packet in the first place.

## 6. Ordered terminal mapping

The mapping is ordered and exhaustive; every run lands in exactly one:

1. `EK0_INVALID` — provenance, partition, seal, or invariant failure.
2. `EK0_NO_RECOVERABLE_SUPPORT` — the reconstructable-new-track stratum is
   empty: the frozen packet contains no exact-key-recoverable additional
   exposure at all.
3. `EK0_RECOVERABLE_SUPPORT_BELOW_FLOOR` — some new tracks are recoverable,
   but even with zero additional hurt the merged `N` cannot reach `153`.
4. `EK0_RECOVERABLE_SUPPORT_UCB_NOT_MET` — the zero-hurt bound reaches the
   floor, but no realized `(N, k)` combination has UCB `≤ 0.05`.
5. `EK0_RECOVERABLE_SUPPORT_SUFFICIENT` — deterministic legal recovery exists
   and at least one realized `(N, k)` has UCB `≤ 0.05`.

A `SUFFICIENT` result authorizes only a separately drafted follow-on
declaration.  No terminal authorizes implementation, rerunning S0, a closure
solve, a score/ranking exercise, a threshold/grid change,
registry/ledger/preset change, or production modification.
