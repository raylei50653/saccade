---
doc-status: active
doc-promotion: navigation-only; not evidence
owner-module: semantic
work-class: mainline-study
wip-role: sole-active
created: 2026-07-12
---

# Runtime-faithful safe domain

## Status

**ACTIVE · mainline-study · sole-active (Amendment 1 `SEALED`; S0 execution
authorized)** (2026-07-13; PR #152). The original authority was suspended before
any run; the corrected V5 was reviewed and resealed at exact head
`70a40cf9d61eb6512b9b5096049ca59efd58aa95`.

## Expected state (lease)

S0 is expected to decide one disposable planning target: whether the accepted axes
keep the required safety characteristics on runtime coordinates. This is **not** an
accepted registry state; only the resealed declaration authorizes the bounded run.
The lease may be replaced or dropped without a registry transition if its decision
relevance becomes zero or a dependency changes.

## Commit point

Owner review of a sealed S0 evidence packet. Only that review may decide whether an
already-registered object's accepted state, substrate, limits, or transition metadata
changed. Opening, editing, or discarding this lease does not.

## Discard when

- runtime fidelity is no longer decision-relevant to the safe-domain consumer;
- a superseding substrate or contract makes this target obsolete.

## Current step

Compute the frozen declaration §§ 3–7, apply the ordered terminal mapping in § 8,
emit the canonical evidence packet, and stop for owner review. This authorizes no
closure solve, threshold selection, preset change, production hook, or score-ranking
claim.

## Design (the architecture this line serves)

```text
gate   : build a safe domain D on a mathematical structure —
         exclude the impossible region, one-sided guarantee (never drop a recoverable GT)
score  : inside the retained domain, separate GT from FP
```

The two layers fail differently and **may not compensate for each other**: a gate
error deletes a recoverable identity; a score error only misorders. Therefore the
gate is a **coarse threshold on a closed-form region with a retention guarantee** —
it does **not** fit a boundary, and it does **not** need discriminative power.
Separation is the score's job.

**This is not a new method.** The canonical mathematical framework already exists
and is normative:
[Statistical Robust Feasible-Set Estimation under Asymmetric Loss](../contracts/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md)
— \(\max_\theta G_{\mathrm{FP}}(\theta)\) s.t. \(L_{\mathrm{GT}}(\theta)\le\varepsilon\),
feasible / productive-safe / robust-feasible sets, region geometry, the mandatory
**independence-unit** declaration (§ 8.1), the **claim ladder L0–L6** (§ 10), and
the forbidden shortcuts (§ 13, including *offline safe \(\not\Rightarrow\) online
effective*). The RegionAsset packaging contract
([safe_region_asset_contract](../contracts/safe_region_asset_contract.md)) lists
**transfer as not yet authorized**. This line's units instantiate that framework;
they do not invent statistics.

## The blocking defect this line opens on

The accepted global safe axes are `{dist_h, log_h_ratio}`
([partial-order note](../../modules/semantic/research/boolean_atom_partial_order_20260711.md),
`GLOBAL_PARTIAL_ORDER_READY` / `ACCEPTED_WITH_LIMITS`, PR #107), and the
restricted-closure prototype that would actually **solve** the safe domain on those
axes **has not started**.

But those axes were computed on **offline proxy coordinates**
(`audit_relink_safe_reject.ensure_prod_proxy_scores`, whose own docstring says
"Add offline proxies of live bridge score + height ratio … Height ratio uses raw
endpoint heights (ema proxy)"):

| Axis | Accepted (offline) | Runtime (kernel) |
|---|---|---|
| `dist_h` | rebuilt from the offline pair table's reduced terms | `‖a_lost − a_cand‖ / h_ref`, from the kernel's own `bridge_anchor4` anchors |
| `log_h_ratio` | `log(h_lost_raw / h_cand_raw)` — **raw box heights** | `log(ema_lost / ema_cand)` — **EMA state**, and `h_ref = max((ema_lost+ema_cand)/2, 1)` |

[D0](../../modules/semantic/research/d0_runtime_shadow_fidelity_results_20260712.md)
already certified that this offline reconstruction is **not fidelity-aligned**
with the runtime quantity (`T2 PROXY_UNFAITHFUL`: decision agreement 95.07 % vs a
99 % bar; \(|\Delta|\) q95 = 1.417 ≈ 3.54× the 0.4 threshold; and the **GT boundary
is distorted** — 7.03 % offline-safe-but-online-unsafe).

**Consequence (the reason this line exists):** a region proved safe in offline
coordinates does **not** inherit its GT-retention guarantee in runtime
coordinates. Formula shape and field name do not transfer semantics — see the
[runtime-quantity fidelity protocol](../contracts/runtime_quantity_fidelity_protocol.md).
Solving the closure on unverified axes would build the safe domain on a substrate
D0 has already ruled unfaithful.

## Read first

- [safe-domain runtime-transfer declaration (draft)](../../modules/semantic/research/safe_domain_runtime_transfer_declaration_20260712.md) — **current unit**
- [partial-order note](../../modules/semantic/research/boolean_atom_partial_order_20260711.md) — the accepted axes, and their offline-coordinate limit
- [D0 runtime-shadow fidelity](../../modules/semantic/research/d0_runtime_shadow_fidelity_results_20260712.md) — why offline coordinates cannot carry the guarantee
- [R1 results](../../modules/semantic/research/r1_temporal_reduction_capture_results_20260712.md) — what made runtime coordinates auditable
- [gt-support morphology thread](gt_support_morphology_20260711.md) · [safe-region assetization (closed)](closed/safe_region_assetization_20260710.md)

**Framework position of this line:** the offline→runtime coordinate change is a
**substrate** change in the framework's own sense (§ 9.3 names *score definitions,
feature extraction, hook placement* as substrate), so S0 is precisely the
framework's **L4 cross-substrate signal portability** audit of the accepted axes
and partial orders — not a true region portability check of a solved region.

## Unit ladder

```text
S0  transfer audit  — do the accepted axes and partial orders keep their safety
                      characteristics (L4 signal portability) in runtime coordinates?
                                                            ← proposed unit (draft)
S1  impossible-region closed form — v_max·la / px_per_m, scale ratio;
                      coarse, no fitting, one-sided coverage proof
S2  restricted-closure solve on whichever axes survived S0  (separately declared)
S3  production hook — relink_bridge_max_speed / spatial_gate / h_lo,h_hi are
                      currently 0.0 / 0.0 / [0.75, 1.33]: the coarse layer is
                      mostly OFF and bdist is implicitly doing the gate's job
                      (separately declared; no authorization from S0–S2)
```

S0 Amendment 1 is sealed and active. S1–S3 are named so the direction is legible;
none is authorized, and each needs its own declaration.

## Must not

- Solve or tune the closure on axes whose runtime transfer is unproven.
- Treat the offline `ACCEPTED_WITH_LIMITS` boundary as a runtime-valid safe region.
- Give the gate a discrimination objective (that is score-ranking, layer L2).
- Fit a boundary, sweep a score threshold, or change a preset from this line.
- Assume an unjoined runtime event is safe (fail-closed: no coverage, no claim).

## History

- 2026-07-12: Proposed. Owner set the architecture — *gate builds a safe domain;
  score ranking separates GT* — which reclassified the discrete-\(M\) study as a
  score-ranking feature question (parked, unsealed) and proposed S0 as the next
  candidate unit. Audit then found the accepted safe axes are built on
  offline proxy coordinates that D0 already ruled unfaithful, and that the
  closure solve which would consume them has not started. S0 declared as the
  proposed draft.
- 2026-07-13: Owner seal recorded on PR #152 against reviewed head
  `55adbbbcddf52a8bf036ca0a01b3fb0ef859025a`; thread transitioned to active / sole-active
  and S0 execution authorized exactly within the frozen declaration §§ 1–8.
- 2026-07-13: Architecture review found that frozen V5 inserted raw unjoined-event
  counts into a track-level CP bound. Execution authority suspended before any run;
  Amendment 1 replaces V5 with a non-statistical adversarial coverage gate and
  returns the charter to proposed / non-WIP pending a new reviewed head and seal.
- 2026-07-13: Amendment 1 resealed on PR #152 at reviewed head
  `70a40cf9d61eb6512b9b5096049ca59efd58aa95`; thread restored to active / sole-active.
  Execution is compute §§ 3–7 and apply § 8; later units remain unauthorized.
