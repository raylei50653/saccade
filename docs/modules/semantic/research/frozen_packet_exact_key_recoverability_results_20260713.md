<!-- doc-status: active -->
<!-- doc-promotion: evidence packet; executed under a sealed rev.3 declaration -->
<!-- doc-date: 2026-07-13 -->
<!-- doc-module: semantic -->

# EK0 — frozen-packet exact-key recoverability — results

## Terminal: `EK0_NO_RECOVERABLE_SUPPORT`

The frozen D0 capture packet is internally consistent.  None of the 893
unjoined runtime events is recoverable through the exact v2 event key or its
canonical-field triple, and none has ambiguous provenance: every event is
structurally unjoinable, exactly as its partition label asserts.

**What this does and does not say.**  This is a consistency statement about
the frozen artifacts.  Per the sealed declaration (§2), for a well-formed v2
packet both reconstructable classes are unreachable by the exporter's own
partition definitions, so this outcome is the expected one; the audit's
informative content is the consistency check and the pinned counts below.
EK0 computes no exposure, floor, or UCB quantity and says nothing about wider
runtime joins — expanding the offline cohort export, adding identity
observability for `unemitted` events, or re-capturing — each of which would
need its own declaration.  Two earlier framings of this study (RJ0
`RJ0_EXPANSION_FUTILE`; EK0 rev.2 with a feasibility envelope) over-claimed
and were rescinded in owner review (PR #156, declaration §0).

Canonical packet: [manifest](evidence/ek0_frozen_packet_exact_key_recoverability_20260713/manifest.json) ·
[metrics](evidence/ek0_frozen_packet_exact_key_recoverability_20260713/metrics.json) ·
[outcome-blind inventory](evidence/ek0_frozen_packet_exact_key_recoverability_20260713/inventory.csv).

## J1 — provenance reproduction

PASS.  The four frozen source hashes reproduce exactly; partition
conservation is `1,684 + 539 + 354 = 2,577`; and all capture rows carry
`d0_event_key_v2_global` with `(seq, lost_global_id, cand_global_id)`.  The
capture remains shadow provenance with zero overflow.

The audit is single-phase and never reads a GT/outcome column
(`gt_label_accessed = false` by construction).  The packet manifest seals the
declaration, runner, inventory, and metrics hashes; a completed packet is
immutable — reruns against it fail closed without modifying it.

## J2–J3 — outcome-blind inventory

All 893 unjoined events were classified using only identity, event
provenance, frozen offline pair membership, and frozen coordinate
availability.  The inventory SHA256 is
`a90c424dc6a74fbbb0bdb3997e388517e56f0b237fa4f65384046510c3590d92`
(byte-identical to the rev.1 packet; classification rules are unchanged).

| Partition | Events | Identified unique lost tracks | Reconstructable | Ambiguous | Class / reason |
|---|---:|---:|---:|---:|---|
| `cohort_gap` | 539 | 169 | 0 | 0 | all structurally unjoinable: same global pair absent from frozen offline universe |
| `unemitted` | 354 | 0 | 0 | 0 | all structurally unjoinable: global identity unresolved; no local-ID fallback |

`cohort_gap` has 370 repeat events after lost-track reduction (68.65% among
identified events).  `unemitted` has no valid `(seq, lost_global_id)` trial
identity.  These counts are descriptive only.

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
identities.  The exporter's partition labels agree with frozen offline
universe membership for every event.

## Closure boundary

EK0 ends at its terminal.  Acceptance is recorded by the review that merges
this packet; no follow-on declaration or engineering work is opened by this
result, and no registry, ledger, or preset entry changes.
