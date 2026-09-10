<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-09-10 -->
<!-- doc-module: cross -->

# Asset disposal round `wa-ec3-r1-20260910`

This file is the **owner-authorized action record** for one disposal round
under [ADR 021](../decisions/021-asset-provenance-and-progress-reporting.md)
W-A exit criterion 3. It is not a candidate projection, not an inventory,
and not approval to delete any unit other than the six exact paths listed
under Disposed.

**Candidate ≠ deletion authorization.**
`scripts/provenance/asset_disposal.py` (AP-5) produces a workspace-local
review list only. Nothing it emits means safe to delete, approved, or
disposable by default. Owner approval is a separate authorized action;
absence of that action is not approval. This round's authorization does
not extend to later projections, later HEADs, wildcards, parent
directories, age-based bulk cleanup, RETAIN units, UNKNOWN units, or any
path not named below.

---

## Round identity

| Field | Value |
|:--|:--|
| `round_id` | `wa-ec3-r1-20260910` |
| bound `main` HEAD | `09984db573503f6936345b1a06a7b3e50f31c7f8` (PR #385 merge) |
| bound `--as-of` | `2026-09-10T14:17:31.179792+00:00` |
| bound candidate projection | `.provenance/asset_disposal_candidates.generated.md` (gitignored; not an authority) |
| bound projection sha256 | `a40347bd1542a0e4511d4764e5ebd46eae8e2eb91b66c4779478410cf0298e68` |
| execution | `Path.rmdir()` on each still-valid approved directory; fails closed if the directory is not empty |
| skipped-on-drift | none |

## Owner approval source

Explicit owner message on 2026-09-10, titled `OWNER APPROVAL — wa-ec3-r1-20260910`,
authorizing **exactly** the six paths below and binding that authorization to
the HEAD / as-of / projection digest in the table above.

The approval text stated that any approved unit which had drifted from the
review assumptions must not be deleted, and that a drifted HEAD or
projection must stop the round rather than extend the authorization.

This file is the durable fact-owner of that authorization after the round
closed. AP-5's generated view does not record approval.

## Disposed (exact paths)

All six remained valid at dispose time (empty directory, AP-5 candidate,
`manifest_state=absent`, not cited, no AP-4 self-attesting / covering-seal /
multi-run evidence). Each was removed with `rmdir` only.

- `runs/mamba_gt_1024_ft`
- `runs/mamba_gt_cs_n16_v14warm`
- `runs/mamba_gt_smoke_test`
- `runs/mamba_gt_vgt_mamba_v16`
- `runs/test_vgt_mamba`
- `runs/test_vgt_mamba_optb`

Bytes released: 0. GB is not an acceptance metric for this round.

## Not authorized (still present)

A tracked document that writes a living unit's directory name cites that
unit (AP-3 literal substring match). This record therefore names **only
the six disposed paths**. Those directories no longer exist as inventory
units; naming them here cannot relabel a remaining unit.

Counts from the owner-review packet, not reproduced as paths:

- RETAIN: 5 `runs/` units (review observed irreplaceable bytes: training
  checkpoints, trained weights, or demo renders)
- UNKNOWN: 69 other candidates from the bound 80-row projection
- never in scope: cited, manifested, invalid, unreadable, future-mtime,
  self-attesting, covering-seal, and multi-run-container units

The exact RETAIN / UNKNOWN paths are the bound candidate projection minus
the six disposed paths. The bound digest is in the identity table; this
file is not a second copy of that list.

## Before / after projection (same bound `--as-of`)

| | before | after |
|:--|--:|--:|
| inventory units | 1059 | 1053 |
| AP-5 candidates | 80 | 74 |
| cited | 120 | 120 |
| manifested | 0 | 0 |
| orphan | 939 | 933 |
| invalid manifests | 0 | 0 |
| AP-4 self-attesting / covering-seal | 3 | 3 |
| AP-4 multi-run containers | 25 | 25 |
| loose files (not units) | 143 | 143 |

After projection sha256 (still gitignored, not an authority):
`84311d66d309cecb6ce38f1a7352b2aa65f9228d02c046b6520da694fc48e5c2`.

After inventory sha256:
`cebc382bee95c3d15a2de7d48f04e06ad4c8936dcd70d26f530958ddca36d52c`.

The after candidate set is exactly the bound before set minus the six
disposed paths. No extra candidate appeared or disappeared.

## Verification

Revalidation immediately before `rmdir` (bound HEAD, bound projection
digest, bound `--as-of`): all six still matched the approval assumptions.
None were skipped.

After disposal, a rescan at the same `--as-of` showed:

- the six paths are absent from disk, inventory, and the AP-5 candidate list
- the remaining unit set equals the before set minus those six paths
- every RETAIN and UNKNOWN directory still exists; directory `st_mtime` unchanged
- retain sentinel files (checkpoints, a trained-head weight, a demo render,
  a debug MOT txt) unchanged in size
- cited / manifested / AP-4-protected unit sets unchanged
- no additional deletion

A first draft of this record named the RETAIN / UNKNOWN paths and, once
tracked, AP-3 cited those nine living units (candidates 74→65). That was
a documentation side-effect, not a deletion, and it violated "remaining
units were untouched." This text no longer names them. After the rewrite,
the live projection returns to 1053 / 74 / cited 120.

Official generators were re-run after disposal:

```text
python3 scripts/provenance/asset_inventory.py --emit --check
python3 scripts/provenance/asset_disposal.py --emit --as-of 2026-09-10T14:17:31.179792+00:00 --check
```

Both `--check` exits were 0.

## W-A exit criteria

This round satisfies **ADR 021 W-A EC3**: at least one owner-approved
disposal round completed.

Completing EC3 did **not** by itself close W-A. EC1 (new producers 100%
manifested) was independently unresolved at the time of this round:
`scripts/eval/mot17.py` still required controlled-host re-attestation
(ADR 021 §4.3). Completing EC3 is not authorization to start W-B, W-C,
or #368.

> **Later (2026-09-10):** EC1 was satisfied by wiring `scripts/eval/mot17.py`
> and republishing the runtime coordinate. W-A is closed. That later action
> is not this round and does not authorize further deletion.
