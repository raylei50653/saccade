<!-- doc-status: sealed-for-execution -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-13 -->
<!-- doc-module: semantic -->

# EK0 — frozen-packet exact-key recoverability — sealed declaration (rev.3)

> **Scope.** EK0 is a pure consistency audit of the frozen D0 capture packet.
> It asks one question: does any of the 893 unjoined runtime events contradict
> its own partition label — i.e. is it recoverable through the exact v2 event
> key (or its redundant canonical-field triple) into the frozen offline pair
> universe, or is its identity provenance ambiguous?  It carries no
> statistical exposure machinery and never reads a GT/outcome column.

## 0. Rescope provenance (rev.1 → rev.3)

Rev.1 executed on 2026-07-13 as *RJ0 runtime-join support sufficiency*
(`RJ0_EXPANSION_FUTILE`).  Owner review (PR #156, round 1) blocked it: the
zero result is largely entailed by the exporter's partition definitions, the
two-phase seal did not bind the runner or blind metrics, the N-merge could
double-count base tracks, and the terminal mapping was not exhaustive.

Rev.2 narrowed the claim to exact-key recoverability but retained a generic
`(N, k)`/UCB feasibility envelope.  Owner review round 2 blocked that too:
the base-overlap exclusion used the wrong set (all 515 joined-partition
tracks, whereas S0's base `N = 116` counts only `gt_valid ∧ gt_match`
tracks), so the unreachable non-empty branches could misclassify terminals;
and the completed packet did not durably preserve the blind seal.

Rev.3 therefore demotes EK0 to what its evidence actually supports: a
**consistency audit**.  The exposure/UCB machinery, the base-overlap
exclusion, the S0-packet dependency, and the unreachable non-empty
feasibility branches are removed entirely.  Because no GT label is ever
read, the blind→reveal split collapses to a single outcome-blind phase.
The outcome-blind classification rules (§3) are unchanged from rev.1 and the
sealed inventory is byte-identical to the rev.1 packet.

## 1. Seal and scope

This declaration is sealed for the audit execution requested on 2026-07-13.
An EK0 terminal is a statement about the frozen artifacts only; it is not
owner acceptance and creates no follow-on work automatically.

**Explicit non-claims.**  No EK0 terminal says anything about:

- expanding the frozen offline pair universe (a different cohort export);
- adding identity observability so `unemitted` events resolve to global IDs;
- re-capturing the runtime side, changing exporter semantics, or any other
  "wider runtime join" that changes the frozen artifacts;
- statistical support, exposure, feasibility, or any `(N, k)` quantity — EK0
  computes none.

Each of those would require its own declaration.  A negative EK0 terminal
must never be cited as evidence that such extensions are futile.

The only frozen inputs are the D0-v2 capture and its substrate:

| Input | Frozen identity |
|---|---|
| capture | `out/signal_study/d0_runtime_shadow_fidelity_20260712T085642Z/capture.csv.gz` — `96093b9b723ed4500b389f8ad74600d75bb49a75064630dd2205cea0b0887047` |
| offline pair universe | `out/signal_study/d0_runtime_shadow_fidelity_20260712T085642Z/pairs.csv` — `ee2898a25ef7f01ed46331c49c12d667846975f25769bc4c3e6b8bad493f8e87` |
| global-ID map | `results/MOT17_eval_d0_shadow_substrate_20260712T085642Z/_global_id_map.txt` — `ae3b6441d1712bcce0826d611cee2cfdf7a01b4d37ec331336f91a0b9148f366` |
| frozen MOT substrate | concatenated sorted `MOT17-*.txt` — `4c5e322a3b8c026de584baa883e26353720837ffa2bf146dfcef2679426a670e` |

J1 must reproduce the input hashes, the partition `1684 + 539 + 354 = 2577`,
and event-key version `d0_event_key_v2_global` with fields
`(seq, lost_global_id, cand_global_id)`.  Any mismatch yields `EK0_INVALID`
without a consistency inference.

**Seal and immutability.**  The audit is single-phase and entirely
outcome-blind.  The packet manifest records the SHA256 of the declaration,
the runner, the sealed inventory, and the metrics.  A completed packet is
immutable: any rerun against its output directory must fail closed without
modifying it.

## 2. Outcome-blind classification (J2)

Only `cohort_gap` (539 events) and `unemitted` (354 events) are inventoried.
The reader projects `pairs.csv` to identity and frozen-coordinate columns
only; it never projects, parses, branches on, serializes, or counts
`gt_valid`, `gt_match`, `gt_lost`, or `gt_cand` — at any point in the audit.

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
keys.

**Expected reachability.**  For a well-formed v2 packet, the exporter's own
partition definitions make both reconstructable classes unreachable:
`cohort_gap` asserts the canonical pair is absent from the offline cohort and
`unemitted` asserts the global identity is unresolved.  EK0's informative
content is exactly this consistency check plus the pinned counts.

## 3. Descriptive reduction (J3)

Inventory aggregation is solely by `(seq, lost_global_id)` and is purely
descriptive: the packet reports each partition's event count, identified
unique lost-track count, reconstructable unique-track count, sequence
distribution, duplicate rate, and class/reason distribution.  No count feeds
an exposure, floor, or UCB computation — none exists in this audit.

The resulting `inventory.csv` and its SHA256 are the seal.

## 4. Ordered terminal mapping

The mapping is ordered and exhaustive; every run lands in exactly one:

1. `EK0_INVALID` — provenance, partition, hash, or immutability failure.
2. `EK0_NO_RECOVERABLE_SUPPORT` — the reconstructable stratum is empty and
   there are no provenance-ambiguous events: every unjoined event is
   structurally unjoinable, exactly as the partition definitions assert.
3. `EK0_PACKET_INCONSISTENT` — at least one reconstructable or
   provenance-ambiguous event exists: the frozen artifacts contradict their
   own partition definitions.  This is a defect finding about the packet,
   not a feasibility result; it authorizes only a separately drafted
   investigation declaration.

No terminal authorizes implementation, rerunning S0, a closure solve, a
score/ranking exercise, a threshold/grid change, registry/ledger/preset
change, or production modification.
