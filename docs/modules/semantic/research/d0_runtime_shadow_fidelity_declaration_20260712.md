# D0 — runtime shadow bridge fidelity (predeclaration)

<!-- doc-status: active -->
<!-- doc-promotion: predeclared study contract; seal = owner acceptance via PR merge -->
<!-- doc-date: 2026-07-12 -->
<!-- doc-module: semantic -->

> **One-line:** Issue #112 — decide whether the offline proxy `score_m_bridge`,
> on which every prior D0/B1 claim rests, faithfully reproduces the **live
> float32 CUDA `bdist`** actually computed by `relink_bidir_propose_kernel`,
> and over what domain that fidelity holds. Substrate validation, not a design
> search. Every terminal is a mainline state transition (§20.7).

Contract: [framework §20](../../../research/eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md) (v1, PR #133) ·
Prior packet (sealed, v1 legacy): [d0_bridge_estimator_fidelity_20260711](d0_bridge_estimator_fidelity_20260711.md)

**Seal semantics (two distinct events — do not conflate them).**

This document is the study's §20.2 declaration. Its authorization chain, as it
actually happened, was:

| Event | What it is | When |
| --- | --- | --- |
| **Execution seal** | Explicit owner confirmation of the §5 boxes (B1 ≥ 99 %, B2 ≤ 0.05, B3 ≥ 0.98) and of the frozen conventions. This is the authorization to execute. | **2026-07-12, before F1–F3 and C3 were computed.** |
| **Research acceptance** | Merge of PR #141: repository promotion of the declaration, the verifier, and the evidence packet. | at merge |

The execution seal was **owner confirmation, not PR merge**. The terminal was
computed only after that confirmation, and the boxes were fixed before any
metric was seen — which is the property the seal exists to guarantee. PR merge
constitutes research acceptance and repository promotion; it is **not** the
original authorization to execute, and this document does not claim it was.

Any deviation from this declaration (metrics, boxes, partition, terminals) voids
the run and requires a new declaration.

---

## 0. Why this study exists, and what changed (read first)

The sealed 2026-07-11 D0 packet is a **kernel-formula reconstruction**, not a
capture. It states so itself, and it fails closed with
`D0_FAIL_CLOSED_CAPTURE_UNAVAILABLE`. Issue #112 was blocked because a runtime
capture appeared impossible to join to any pair cohort. That diagnosis was
wrong in three separate ways, all now fixed:

1. **Structural.** The capture lives only inside `relink_bidir_propose_kernel`,
   which is gated on the bridge being enabled — but an enabled bridge *commits*
   (`relink_bidir_commit_kernel` is its sole write to `track_ids`/`active`),
   rewriting the very identity space the join key lives in. No commit, on any
   git revision, could have produced a joinable capture.
   Fixed by `set_research_bridge_shadow(true)`: propose and capture, skip
   commit. Output is then **bit-identical to a bridge-off run** — verified
   byte-for-byte on all 7 sequences (§2).
2. **ID universe.** The kernel records tracker-**local** ids; the evaluator
   remaps to **global** ids before writing MOT output, and the offline cohort is
   built from that MOT output. Joining on local ids produces *false* matches
   wherever the remap is the identity (observed: MOT17-02, MOT17-05 — 202
   spurious pairs). Fixed by the v2 key (§3).
3. **Impossible frame fields.** `lost_last_frame` was derived as
   `fidelity_frame - la` from a capture-local counter (the tracker has no
   absolute frame counter), underflowing to negative indices (−1, −2, −8).
   Dropped entirely; they were never needed for identity, because propose fires
   once per track life.

Nothing was lost historically: the c661adf7 substrate MOT dir still exists. The
old cohort was never detector data — it is an offline enumeration over MOT
output, and it self-describes as *"offline proxy … not bit-exact live"*.
Replacing that proxy is exactly what #112 is for, so this study rebuilds the
cohort at HEAD, where the offline pairs and the live capture come from **one
run** and join by construction.

## 1. §20.2 declaration block

```text
Target decision layer   none (cross-layer substrate work). `score_m_bridge`
                        underlies both gate-coverage and score-ranking studies;
                        this study validates the substrate, not a layer policy.
Study intent            boundary diagnostic (primary); no secondary intents.
                        Establishes where the offline proxy is a valid stand-in
                        for the runtime kernel, and where it stops being one.
Design objective        n/a (not a design evaluation). No policy, threshold, or
                        candidate may be proposed, tuned, or claimed from it.
Selection rule          none — there are no competing candidates. The terminal
                        is decided solely by the §5 boxes and the §6 coverage
                        criteria, not by choosing a best performer.
Validity gate           §7 (V1–V6), separating UNRESOLVED from a real verdict.
Stop condition          §8 — one run, no re-fit, no threshold search, no family
                        widening on failure. No third door (§20.7).
Output class            diagnostic / substrate-validity. Carries no design
                        authority; §20.5 applies to every number reported.
Mainline transition     §9 — T1 certifies the proxy and closes #112's open
                        unknown; T2 invalidates it and imposes a recorded
                        validity limit on every prior study that used it;
                        T3 certifies it only on the covered region and records
                        the cohort as non-covering; T0 = UNRESOLVED closes the
                        experiment only.
```

## 2. Substrate (frozen)

Produced at the working tree carrying shadow mode + the v2 exporter (the commit
sealing this declaration). The pipeline is deterministic under these flags; the
run was executed **twice** and produced byte-identical MOT output both times.

Common flags (both runs):
`--preset mamba_whole_graph_m --detector SDP --double-buffer
--detect-barrier event --no-interpolate-tracklets --no-gpu-decode`
(`--no-gpu-decode` is mandatory: GPU decode is a known nondeterminism source.)

| Run | Bridge | Purpose |
| --- | --- | --- |
| A | `--no-relink-bridge-enabled` | bridge-off substrate → MOT output → offline pair cohort |
| B | `--relink-bridge-enabled` + `SACCADE_RESEARCH_BRIDGE_FIDELITY_CAPTURE_SHADOW=1` | shadow capture → live CUDA `bdist` |

**Bit-exactness (the load-bearing invariant).** Run B's MOT output is
byte-identical to Run A's on all 7 sequences. This is what makes the capture
joinable: the shadow bridge observes without changing the tracking result.
Regression-tested in `test_shadow_bridge_proposes_and_captures_without_changing_identity`.

Frozen artifacts (`<S>` = `20260712T085642Z`):

| Artifact | Path | SHA256 |
| --- | --- | --- |
| Pair cohort | `out/signal_study/d0_runtime_shadow_fidelity_<S>/pairs.csv` | `ee2898a25ef7f01e…` |
| Runtime capture (v2) | `out/signal_study/d0_runtime_shadow_fidelity_<S>/capture.csv.gz` | `96093b9b723ed450…` |
| Global id map | `results/MOT17_eval_d0_shadow_substrate_<S>/_global_id_map.txt` | `ae3b6441d1712bcc…` |
| Substrate MOT (7-seq concat) | `results/MOT17_eval_d0_shadow_substrate_<S>/MOT17-*.txt` | `4c5e322a3b8c026d…` |

Cohort: 24,346 candidate pairs, 339 GT-labelled positives.
Bridge config (from capture provenance): `px=0.4, at=4, min_lost=2, ttl=120,
anchor=adaptive, anchor_rate=0.03, dir_bonus=0.0`.

Execution must re-derive these artifacts and assert the hashes reproduce
bit-for-bit (V6). A mismatch is a validity failure, not a finding.

## 3. Event key and identity (frozen — v2)

```text
event_key = seq | lost_global_id | cand_global_id     (event_key_version = d0_event_key_v2_global)
```

* Global ids only. A local-id join is a **contract error**, not a fallback.
* The v1 5-field key remains frozen for the sealed 2026-07-11 packet and is
  never reinterpreted (`EVENT_KEY_VERSION_V1_LEGACY`).
* `lost_last_frame` / `cand_first_frame` are **absent** from v2 packets.
* Uniqueness is asserted, not assumed: propose fires once per track life, so the
  three-field key is unique over all keyed rows (observed: 2,577 events, 0
  duplicates). The exporter fails closed on any duplicate.

## 4. Partition (frozen, exhaustive, mutually exclusive, conserved)

Every captured proposal falls in exactly one class. The counts are an **input**,
fixed before any fidelity quantity is computed.

| Partition | Count | Share | Research role |
| --- | ---: | ---: | --- |
| `matched` — joins the offline cohort | 1,684 | 65.35 % | **fidelity analysis set (the only one)** |
| `cohort_gap` — ids emitted, pair never enumerated | 539 | 20.92 % | cohort-support coverage gap |
| `unemitted` — id never reached MOT output | 354 | 13.74 % | observability / emission gap |
| **total captured** | **2,577** | 100 % | |

**Conservation invariant:** `1684 + 539 + 354 = 2577`. Asserted by the exporter
and by `test_v2_export_partitions_are_exhaustive_and_conserved`.

`cohort_gap` and `unemitted` are **not missing values.** They must never be
dropped, imputed, or placed in an agreement denominator. They are the limit on
how far a fidelity conclusion extrapolates, and they are answered separately in
§6.

## 5. Fidelity verdict — computed on the 1,684 `matched` pairs only

Compared quantities, per matched event:

* **Runtime:** `bdist` — float32 CUDA output of `relink_bidir_propose_kernel`.
* **Offline proxy:** `s0 = score_m_bridge`
  `= w·½(fwd_resid + bwd_resid) + (1 − w)·dist_h`,
  `w = sqrt(clip(lost_exit_speed / 0.12, 0, 1))`,
  recomputed from the frozen `pairs.csv` by `ensure_prod_proxy_scores`.

**Estimator freezes (no implementer degrees of freedom).** All arithmetic in
float64 after CSV read. Decision boundary is `score <= 0.4` (`production_safe`,
inclusive, matching the kernel). Quantiles: numpy `method="linear"` (type 7).
Spearman: average ranks for ties. No rounding beyond the exporter's 10-dp stable
serialization. No re-fit of `w`, of `0.12`, or of the `0.4` threshold — this
study measures the estimator, it does not tune it.

Primary metrics:

* **F1 — decision agreement.** Fraction of matched events where
  `1[s0 ≤ 0.4] == 1[bdist ≤ 0.4]`. Reported with the full 2×2 confusion
  (both-accept / both-reject / proxy-accept-only / runtime-accept-only). The
  two off-diagonal cells are reported separately and are **not** netted: a proxy
  that accepts what the kernel rejects is a different failure from the converse.
* **F2 — numeric error.** Δ = `s0 − bdist`. Report median, IQR, and |Δ| at
  q50 / q90 / q95, both absolute and as a fraction of the 0.4 threshold.
* **F3 — rank agreement.** Spearman ρ between `s0` and `bdist`. This is the
  metric the score-layer studies actually rest on (they rank, they do not
  threshold), so it is primary, not supporting.

Component attribution (**diagnostic only — cannot move the terminal**):
per-component agreement of `dist_h`, `fwd_r` vs `fwd_resid`, `bwd_r` vs
`bwd_resid`, and `w`, to attribute any aggregate discrepancy to the anchor /
EMA / velocity estimators rather than to the aggregation.

**Minimum-effect boxes — OWNER CONFIRMED 2026-07-12 (sealed, not adjustable).**
The proxy is used downstream both to *rank* and to *stand in for a production
accept decision*, so it must clear both:

| Box | Bar | Rationale |
| --- | --- | --- |
| B1 | F1 decision agreement ≥ **99 %** | The proxy has been used to describe a production gate's reach; a 1-in-100 mislabel is the most the gate-coverage claims can absorb. |
| B2 | \|Δ\| q95 ≤ **0.05** | 12.5 % of the 0.4 threshold — an error that cannot flip a decision that is not already marginal. |
| B3 | Spearman ρ ≥ **0.98** | Ranking is the actual downstream use; a proxy that reorders candidates invalidates the ranking probes regardless of its numeric error. |

All three must pass, **inclusive** (`≥` / `≤` as written).

**The three boxes are non-compensatory (sealed).** B1 and B2 are jointly the
necessary condition for **threshold transfer**; B3 is the necessary condition
for **rank transfer**. A strong B1/B2 does **not** rescue a failed B3, and a
strong B3 does **not** let a threshold conclusion survive a failed B1 or B2.
There is no weighted total, no grey band, and no trade-off among them. Owner
rationale (2026-07-12): 99 % on 1,684 matched pairs admits at most **16**
flipped `≤ 0.4` decisions — high interchangeability without demanding
implementation equivalence, where a handful of boundary float differences would
otherwise dominate the verdict; q95 (not mean/median/max) on |Δ| stops a mass of
mid-sized offsets being averaged away without letting a few tail values
dominate; ρ ≥ 0.98 is the floor at which existing ranking-class studies survive
as a whole, below which a substitute measure of the *same* quantity may still
permit substantial local reordering.

These bars were fixed before any of F1–F3 was computed.

## 6. Coverage verdict — answered separately, never merged into fidelity

The fidelity result speaks for 65.35 % of the bridge's actual proposals. The
coverage verdict asks whether that share is representative, and whether the
34.65 % outside it is *structurally* different.

* **C1 — share.** Report matched share of all captured (65.35 %) and of mappable
  (75.75 %). Descriptive; no bar.
* **C2 — structural bias.** Both `matched` and `cohort_gap` carry a runtime
  `bdist`, so their distributions are directly comparable. Report the two-sample
  KS statistic on `bdist` and on `gap`, plus per-sequence composition, for
  `matched` vs `cohort_gap`, and for `matched` vs `unemitted`.
* **C3 — the decision-relevant region (the real question).** Production only
  acts where `bdist ≤ 0.4`. Report the partition composition **restricted to
  `bdist ≤ 0.4`**. If a materially larger share of accept-region events falls
  outside `matched` than inside it, the offline cohort does not cover the region
  where the bridge actually fires, and no fidelity claim may be extrapolated to
  production behaviour — regardless of how well F1–F3 pass.

**Coverage passes** iff C3 shows the accept-region composition is not materially
worse than the overall composition, and C2 shows no bias that would explain the
fidelity result away. Coverage failure does not retract the fidelity number on
`matched`; it *scopes* it (→ T3).

## 7. Validity gate (V — separates UNRESOLVED from a real verdict)

* **V1** capture provenance `shadow == true`, **and** Run B's MOT output is
  byte-identical to Run A's on all 7 sequences (re-verified at execution).
* **V2** every per-sequence capture is `complete` with `overflow_events == 0`.
* **V3** the local→global map is injective per sequence, and **zero** keyed rows
  carry a global id of −1 (no silent local-id fallback).
* **V4** partition conserved: `matched + cohort_gap + unemitted == 2577`.
* **V5** `matched` N ≥ 1,000, and zero NaN in any column required by §5; if NaN
  removal would drop > 5 % of matched rows the study is invalid.
* **V6** the four frozen input hashes (§2) reproduce bit-for-bit.

Any V failure → **T0 (UNRESOLVED / INVALID-STUDY)**. It closes the experiment,
not the hypothesis path, and is not mainline progress.

## 8. Stop conditions

One run. Sufficiency: V passes and F1–F3 are computed on `matched`. Futility is
**not** a licence to widen: if the boxes fail, the study terminates at T2. No
re-fitting `w`, no re-deriving `s0`, no moving `0.4`, no adding components, no
"describe more and continue" (§20.7).

## 9. Terminals → mainline transitions (§20.7)

| Terminal | Condition | Mainline state transition |
| --- | --- | --- |
| **T1 — PROXY_FAITHFUL** | B1∧B2∧B3 pass, coverage passes | Closes #112's open unknown: `score_m_bridge` is certified as a faithful stand-in for runtime `bdist`. Prior D0/B1 claims keep their status, now resting on a *measured* rather than assumed estimator. The sealed 2026-07-11 packet may be superseded by a runtime packet — it is **not** retro-reclassified. |
| **T2 — PROXY_UNFAITHFUL** | any of B1–B3 fails | The proxy is not a valid stand-in. Every prior study that used `score_m_bridge` as a proxy for consumer-A `bdist` inherits an explicit, recorded validity limit, and estimator-shifted claims must be re-scoped. A negative terminal, but a real transition: it changes what already-accepted evidence is permitted to claim. |
| **T3 — FAITHFUL_BUT_NON_COVERING** | B1–B3 pass, coverage (C3) fails | Fidelity is certified **only on the matched region**. The offline cohort is recorded as not covering the bridge's proposal universe in the decision-relevant `bdist ≤ 0.4` region, which becomes a standing limit on all cohort-based bridge studies. |
| **T0 — UNRESOLVED / INVALID-STUDY** | any V gate fails | Closes this experiment only. Not mainline progress. Must not be reported as a fidelity finding in either direction. |

**Scope caveat (unconditional, carried verbatim by every terminal T0–T3):** the
study's scope is the 7-sequence MOT17-SDP `mamba_whole_graph_m` bridge-off
substrate. No terminal establishes any claim about a bridge that *commits*
(shadow suppresses the commit by construction), nor about any other detector,
preset, or sequence set.

## 10. Execution order

1. Regenerate Run A + Run B; assert byte-identity and the §2 hashes (V1, V6).
2. Export v2; assert V2–V4 via the exporter's fail-closed path.
3. Compute §5 on `matched` only; compute §6 on all three partitions.
4. Emit the evidence packet with the terminal.

## 11. Must not

* Compute any fidelity quantity on `cohort_gap` or `unemitted`, or admit them to
  an agreement denominator.
* Fall back to raw/local ids for any join, for any reason.
* Re-seal, re-checksum, or reinterpret the v1 legacy packet. It stays frozen.
  (`b43772b7` mutated its sealed runner in place; that mutation has been
  reverted and the fail-closed inventory checker is green again.)
* Use `lost_last_frame` / `cand_first_frame` for anything.
* Tune `0.4`, `w`, or `0.12`, or choose any box in §5 after seeing a number.

## 12. Pre-seal disclosure (what has already been observed)

Required by the seal discipline: the following were seen during the plumbing
work, before this declaration, and are disclosed so no bar can be accused of
having been chosen around them.

**Observed:** the partition counts (§4); key uniqueness (2,577 / 0 duplicates);
byte-identity of Run A vs Run B; the 202 false local-id matches and their
confinement to MOT17-02/05; a handful of individual `bdist` values seen while
debugging the capture (e.g. 5.88, 0.41); and — **coverage-relevant** — the
`gap` median of matched (13) vs unmatched (8) runtime events, max 26 in both.

**Not observed, and deliberately not computed:** any agreement, Δ, correlation,
or confusion between `s0` and `bdist`; any partition composition broken down by
`bdist` region (i.e. C3 is genuinely unknown). The terminal quantity remains
sealed at the time of writing.
