<!-- doc-status: active -->
<!-- doc-promotion: evidence packet; executed under a sealed rev.2 declaration -->
<!-- doc-date: 2026-07-13 -->
<!-- doc-module: semantic -->

# EK0 — frozen-packet exact-key recoverability — results

## Terminal: `EK0_NO_RECOVERABLE_SUPPORT`

Within the frozen D0/S0 packet, none of the 893 unjoined runtime events is
recoverable through the exact v2 event key or its canonical-field triple.
The reconstructable-new-track stratum is empty, so the exposure envelope is
the unchanged base point `(N, k) = (116, 3)` with one-sided 95%
Clopper–Pearson UCB `0.06548`.

**What this does and does not say.**  This is a bookkeeping statement about
the frozen artifacts.  Per the sealed declaration (§0/§3), for a well-formed
v2 packet both reconstructable classes are unreachable by the exporter's own
partition definitions, so an empty stratum is the expected outcome; the
audit's informative content is the consistency check and the pinned counts
below.  It says nothing about wider runtime joins — expanding the offline
cohort export, adding identity observability for `unemitted` events, or
re-capturing — each of which would need its own declaration.  A previous
framing of this study (RJ0, `RJ0_EXPANSION_FUTILE`) over-claimed exactly that
and was rescinded in owner review (PR #156).

Canonical packet: [manifest](evidence/ek0_frozen_packet_exact_key_recoverability_20260713/manifest.json) ·
[metrics](evidence/ek0_frozen_packet_exact_key_recoverability_20260713/metrics.json) ·
[outcome-blind inventory](evidence/ek0_frozen_packet_exact_key_recoverability_20260713/inventory.csv).

## J1 — provenance reproduction

PASS.  The four frozen source hashes reproduce exactly; S0 `grid.csv`,
`metrics.json`, and runner hashes reproduce; partition conservation is
`1,684 + 539 + 354 = 2,577`; and all capture rows carry
`d0_event_key_v2_global` with `(seq, lost_global_id, cand_global_id)`.  The
capture remains shadow provenance with zero overflow.

The rev.2 seal additionally binds the runner and blind metrics: the reveal
phase re-verified the declaration, runner, sealed inventory, and blind
metrics hashes against the blind-phase manifest, and re-checked the frozen
hashes of `pairs.csv`, `capture.csv.gz`, and S0 `grid.csv` before the (empty)
GT projection.

## J2–J3 — sealed outcome-blind inventory

Before any GT-label projection, the sealed declaration classified all 893
unjoined events using only identity, event provenance, frozen offline pair
membership, and frozen coordinate availability.  The pre-GT inventory SHA256
is `a90c424dc6a74fbbb0bdb3997e388517e56f0b237fa4f65384046510c3590d92`
(byte-identical to the rev.1 packet; classification rules are unchanged).

| Partition | Events | Identified unique lost tracks | …not in joined partition | Reconstructable (new) | Class / reason |
|---|---:|---:|---:|---:|---|
| `cohort_gap` | 539 | 169 | 57 | 0 | all structurally unjoinable: same global pair absent from frozen offline universe |
| `unemitted` | 354 | 0 | 0 | 0 | all structurally unjoinable: global identity unresolved; no local-ID fallback |

The joined (`matched`) partition covers 515 unique lost tracks; the
base-overlap exclusion (declaration §4) removes the other 112 identified
`cohort_gap` tracks from any exposure headroom.  Notably, the 57 remaining
identity-new tracks would have been numerically sufficient to reach the
`N = 153` floor (116 + 57 = 173) — the binding constraint in this packet is
recoverability, not track count.  `cohort_gap` has 370 repeat events after
lost-track reduction (68.65% among identified events).  `unemitted` has no
valid `(seq, lost_global_id)` trial identity.

| Sequence | `cohort_gap` events / tracks | `unemitted` events / tracks |
|---|---:|---:|
| MOT17-02-SDP | 99 / 25 | 21 / 0 |
| MOT17-04-SDP | 31 / 14 | 24 / 0 |
| MOT17-05-SDP | 66 / 27 | 40 / 0 |
| MOT17-09-SDP | 16 / 8 | 6 / 0 |
| MOT17-10-SDP | 131 / 43 | 56 / 0 |
| MOT17-11-SDP | 17 / 9 | 2 / 0 |
| MOT17-13-SDP | 179 / 43 | 205 / 0 |

There were no duplicate target event keys and no non-unique offline pair
identities.  The packet is internally consistent: the exporter's partition
labels agree with frozen offline universe membership for every event, and no
event is rescuable under the sealed key rules.

## J4–J5 — empty reveal and invariant check

The frozen reconstructable-new-track stratum is empty, so the permitted GT
projection is also empty: `gt_label_accessed = false`, additional
`gt_valid ∧ gt_match` tracks = 0, base-overlap exclusions = 0, and the
observable added-hurt range is `[0, 0]`.  The sole envelope point is
`(N, k) = (116, 3)`, UCB `0.06548327985535148`.

All J5 semantic invariants hold (no local-ID fallback, no
outcome-conditioned selection, no new proxy/refit, no coordinate, population,
trial-unit, or grid change, no production mutation); any violation would have
been `EK0_INVALID` rather than a terminal.

## Closure boundary

EK0 ends at its terminal.  Acceptance is recorded by the review that merges
this packet; no follow-on declaration or engineering work is opened by this
result, and no registry, ledger, or preset entry changes.
