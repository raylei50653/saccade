<!-- doc-status: terminal-pending-owner-acceptance -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-13 -->
<!-- doc-module: semantic -->

# RJ0 — runtime-join support sufficiency — results

## Terminal: `RJ0_EXPANSION_FUTILE`

The frozen seven-sequence support cannot supply any additional legal
lost-track exposure.  Even under the most favourable assumption of zero new
hurt, the maximum remains `N_max = 116`, below the fixed `N = 153` floor for
`k = 3` and a one-sided 95% Clopper–Pearson UCB no greater than `0.05`.

This is a support-feasibility result only.  It neither determines axis transfer
nor authorizes wider-join implementation, an S0 rerun, a closure solve, a
threshold/grid change, or any production change.

Canonical packet: [manifest](evidence/rj0_runtime_join_support_sufficiency_20260713/manifest.json) ·
[metrics](evidence/rj0_runtime_join_support_sufficiency_20260713/metrics.json) ·
[outcome-blind inventory](evidence/rj0_runtime_join_support_sufficiency_20260713/inventory.csv).

## J1 — provenance reproduction

PASS.  The four frozen source hashes reproduce exactly; S0 `grid.csv`,
`metrics.json`, and runner hashes reproduce; partition conservation is
`1,684 + 539 + 354 = 2,577`; and all capture rows carry
`d0_event_key_v2_global` with `(seq, lost_global_id, cand_global_id)`.  The
capture remains shadow provenance with zero overflow.

## J2–J3 — sealed outcome-blind inventory

Before any GT-label projection, the sealed declaration classified all 893
unjoined events using only identity, event provenance, frozen offline pair
membership, and frozen coordinate availability.  The pre-GT inventory SHA256
is `a90c424dc6a74fbbb0bdb3997e388517e56f0b237fa4f65384046510c3590d92`.

| Partition | Events | Identified unique lost tracks | Reconstructable unique tracks | Class / reason |
|---|---:|---:|---:|---|
| `cohort_gap` | 539 | 169 | 0 | all structurally unjoinable: same global pair absent from frozen offline universe |
| `unemitted` | 354 | 0 | 0 | all structurally unjoinable: global identity unresolved; no local-ID fallback |

`cohort_gap` has 370 repeat events after lost-track reduction (68.65% among
identified events); it cannot convert those repeated events into more trials.
`unemitted` has no valid `(seq, lost_global_id)` trial identity.

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
identities.  The lack of support is therefore structural rather than a
provenance ambiguity.

## J4–J5 — empty reveal and semantic check

The frozen reconstructable stratum is empty, so the permitted GT projection is
also empty: `gt_label_accessed = false`, additional `gt_valid ∧ gt_match`
tracks = 0, and the observable added-hurt range is `[0, 0]`.  The sole merged
envelope point is `(N, k) = (116, 3)`, with UCB `0.06548327985535148`.

No local-ID fallback, outcome-conditioned selection, new proxy/refit,
coordinate change, population change, trial-unit change, frozen-grid change,
or production mutation was used.  A different key that attached an unjoined
runtime event to another offline pair would violate those retained semantics;
it is not a legal RJ0 expansion.

## Owner acceptance boundary

RJ0 stops here, awaiting owner acceptance of the single terminal above.  No
follow-on declaration or engineering work has been opened.
