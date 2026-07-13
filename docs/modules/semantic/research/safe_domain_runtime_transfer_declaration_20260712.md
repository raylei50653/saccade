<!-- doc-status: active -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-07-12 -->
<!-- doc-module: semantic -->

# S0 — safe-domain axis transfer to runtime coordinates — **sealed** declaration

> **One-line:** the accepted safe axes `{dist_h, log_h_ratio}` were certified on
> **offline proxy** coordinates that D0 already ruled unfaithful. Before the
> authorized closure prototype **solves** a region on them, S0 runs the
> framework's **substrate-robustness / L4 portability** audit: **does
> \(L_{\mathrm{GT}}\) stay bounded when the same rule is evaluated on runtime
> coordinates?** Coarse thresholds only — no boundary fit, no score, no
> production change.

> **SEALED.** Execution is authorized only for the frozen scope in §§ 1–8. Results
> must be emitted to the canonical evidence packet and reviewed at the thread's
> commit point; the seal does not itself accept a terminal or authorize later units.

**Normative inputs (cited, never re-derived):**
[feasible-set framework](../../../research/contracts/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md) ·
[RegionAsset contract](../../../research/contracts/safe_region_asset_contract.md) ·
[gate-vs-score layer contract](../../../research/contracts/signal_table_schema.md#05-gate-vs-score-support--calibration--policy) ·
[runtime-quantity fidelity protocol](../../../research/contracts/runtime_quantity_fidelity_protocol.md)

Thread: [runtime-faithful safe domain](../../../research/threads/runtime_faithful_safe_domain_20260712.md) ·
Accepted axes: [partial-order note](boolean_atom_partial_order_20260711.md) ·
Fidelity fact: [D0 results](d0_runtime_shadow_fidelity_results_20260712.md)

## 1. Position in the accepted framework (binding)

This unit does **not** invent a method. It instantiates the existing one.

| Framework object | This unit |
|---|---|
| Objective (§ 1) | \(\max_\theta G_{\mathrm{FP}}(\theta)\) s.t. \(L_{\mathrm{GT}}(\theta) \le \varepsilon\) — the **gate** form: one-sided, non-compensatory |
| Robustness axis (§ 9.3 substrate) | the substrate change under test is **coordinate provenance**: offline-proxy reconstruction → runtime kernel terms. § 9.3 names *score definitions / feature extraction / hook placement* as substrate; this is exactly that |
| Claim ladder (§ 10) | targets **L4 — cross-substrate signal portability**. Since the closure has not been solved yet, no concrete region exists, so this is not a true L4 portability check of a solved region. It audits whether the signal axes and partial orders themselves can be ported to the runtime substrate, which is a prerequisite for L4. |
| Forbidden shortcut (§ 13) | *offline safe \(\not\Rightarrow\) online effective* — this unit is the direct test of the shortcut the morphology line would otherwise take |
| Layer (signal_table_schema § 0.5) | **L0 support gate**. Membership, not ordering. The gate is not required to discriminate; it is required not to delete GT. Any discriminative objective is score-layer and out of scope |

`L_GT` operational meaning is declared per framework § 2.2: **a GT-recoverable
bridge pair falling inside the pruned region** (i.e. a true relink the region
would delete).

## 2. Why the audit must precede the closure solve

The accepted global axes are `{dist_h, log_h_ratio}`
([partial-order note](boolean_atom_partial_order_20260711.md),
`GLOBAL_PARTIAL_ORDER_READY` / `ACCEPTED_WITH_LIMITS`), and the restricted-closure
prototype authorized to **solve** a region on them **has not started**.

But both axes were computed by `audit_relink_safe_reject.ensure_prod_proxy_scores`,
whose own docstring reads *"Add offline proxies of live bridge score + height
ratio … Height ratio uses raw endpoint heights (ema proxy)"*:

| Axis | Accepted (offline substrate) | Runtime substrate (kernel) |
|---|---|---|
| `dist_h` | rebuilt from the offline pair table's reduced terms | \(\|a_{\mathrm{lost}}-a_{\mathrm{cand}}\|/h_{\mathrm{ref}}\), from the kernel's own `bridge_anchor4` anchors |
| `log_h_ratio` | \(\log(h^{\mathrm{raw}}_{\mathrm{lost}}/h^{\mathrm{raw}}_{\mathrm{cand}})\) — **raw box heights** | \(\log(e_{\mathrm{lost}}/e_{\mathrm{cand}})\) — **EMA state** |
| normalizer | offline reconstruction | \(h_{\mathrm{ref}}=\max((e_{\mathrm{lost}}+e_{\mathrm{cand}})/2,\ 1)\) |

[D0](d0_runtime_shadow_fidelity_results_20260712.md) certified that substrate
**`T2 PROXY_UNFAITHFUL`**, and — decisively for a *gate* — that the offline proxy `s0`
exhibits a **7.03% offline-safe but online-unsafe inversion on GT** (where `s0 ≤ 0.4` but
`bdist > 0.4`). This is a property of the proxy `s0 ≤ 0.4` vs runtime `bdist ≤ 0.4` threshold
inversion, not a direct proof of boundary distortion in the `{dist_h, |log_h_ratio|}` rule
family itself. However, it demonstrates that a bound on \(L_{\mathrm{GT}}\) proved in offline
coordinates does not automatically transfer to runtime coordinates. Solving a closure on
unaudited axes would place the safe domain on a substrate already ruled unfaithful.

## 3. Frozen substrate (no new capture)

Both coordinate systems already exist in sealed artifacts.

| Input | Frozen source |
|---|---|
| Runtime coordinates **and** GT | sealed **D0 v2** packet — runtime `BridgeFidelityEvent` terms exact-joined to GT: `matched = 1,684`, `cohort_gap = 539`, `unemitted = 354` (partition = 2,577) — [packet](evidence/d0_runtime_shadow_fidelity_20260712/manifest.json) |
| Offline coordinates | `build_relink_candidates` pool + `ensure_prod_proxy_scores` (the morphology substrate) |
| Axes + safer directions | [partial-order note](boolean_atom_partial_order_20260711.md) — cited verbatim |
| \(\varepsilon = 0.05\) | inherited `ε_morph` — **cited, not invented** |
| \(\rho \ge 0.98\) agreement bar | D0's B3 rank bar — **cited, not invented** |

Input SHA256s are recorded pre-outcome. The D0 packet is a frozen **input**: it is
not rewritten, re-terminaled, or reinterpreted.

## 4. Frozen rule under test (coarse; closed form)

The accepted structure is a monotone partial order with a declared safer-rejection
direction per axis. The coarse region is

\[
D(\theta_d,\theta_r) = \{\,\text{pair}: \texttt{dist\_h} \ge \theta_d \ \textbf{ or }\ |\texttt{log\_h\_ratio}| \ge \theta_r \,\},
\]

evaluated **identically** in both coordinate systems. No weights, no fitted
boundary, no per-cell risk field, no interaction terms (framework § 4.2
multi-axis threshold, not § 4.3 weighted rule).

**Frozen grid:** \(\theta_d \in \{0.2,0.3,\dots,2.0\}\) (\(h\) units),
\(\theta_r \in \{0.05,0.10,\dots,0.60\}\). Every point is evaluated; **none is
selected, tuned, or recommended** — S0 chooses no \(\theta\).

## 5. Frozen safety statistics (framework § 2.2 + § 8.1)

\[
L_{\mathrm{GT}}(\theta)=\frac{N_{\mathrm{GT,hurt}}(\theta)}{N_{\mathrm{GT,exposed}}},
\qquad
G_{\mathrm{FP}}(\theta)=\frac{N_{\mathrm{FP,removed}}(\theta)}{N_{\mathrm{FP,exposed}}}
\]

Exposure counts are reported wherever a rate is reported (framework § 2.2).

**Independence unit (§ 8.1 mandatory declaration):**

| Item | Declared value |
|---|---|
| Independence unit | the **lost track** |
| Clustering structure | one lost track contributes many candidate pairs across slots and frames; pair-level counts are **not** independent trials |
| Aggregation | hurt is aggregated to the unit **before** bounding: a track counts once iff ≥ 1 of its GT pairs falls inside \(D\). Base Clopper-Pearson UCB safety calculations are performed **only** on the matched pairs that have valid track identities. The 893 unjoined events (539 cohort_gap and 354 unemitted) represent coverage ambiguity, which is evaluated on their runtime coordinates and checked as a fail-closed flip condition (see V5). |
| Bound | one-sided 95 % **Clopper–Pearson** upper bound \(\mathrm{UCB}(x,N)\) on the track-level counts |
| Residual clustering (named, not dismissed) | **sequence-level** shared scene / pipeline state remains above the declared unit; the bound is therefore not claimed to be sequence-robust, and § 9.1 sequence robustness is **not** established by this unit |

A region is **certified safe** iff \(\mathrm{UCB}(x,N) \le \varepsilon = 0.05\).
This fixes the limit the morphology note itself flagged (its track-level CP UCB was
*nominal, not cluster-adjusted*).

\(G_{\mathrm{FP}}\) is **reported, never traded** against \(L_{\mathrm{GT}}\): the
constraint is one-sided (framework § 2.1 asymmetric loss).

## 6. Frozen validity gates (non-compensatory) → failure ⇒ `S0_INVALID`

| Gate | Must pass |
|---|---|
| V1 provenance | all inputs match declared SHA256; axes, \(\varepsilon\), and \(\rho\) bar cited from their owners, never re-derived |
| V2 partition conservation | `matched + cohort_gap + unemitted = 2,577`, reproduced from the sealed packet |
| V3 join integrity | every matched pair carries **both** coordinate systems and a GT flag; keys unique |
| V4 exposure floor | \(N_{\mathrm{GT,exposed}} \ge 59\) lost tracks (mathematically required for 95% CP UCB ≤ 0.05 at zero harm) and ≥ 1,000 matched pairs, else `S0_UNDECIDABLE` |
| V5 **worst-case unjoined ambiguity (evaluated on \(D_{\mathrm{rt}}\))** | the 893 unjoined events are evaluated on their **runtime coordinates** (\(D_{\mathrm{rt}}\)). If \(M\) unjoined events fall inside \(D_{\mathrm{rt}}\) at a given grid point, they represent potential GT losses. We compute the worst-case UCB by adding \(M\) to both numerator and denominator of the matched UCB: \(\mathrm{UCB}_{\mathrm{worst}} = \mathrm{UCB}(X_{\mathrm{matched}} + M, N_{\mathrm{matched}} + M)\). If this flips a base-safe point to unsafe (\(\mathrm{UCB}_{\mathrm{worst}} > 0.05\)), the safety certification at this point is denied. If this worst-case flip occurs for any grid point certified safe offline, the terminal is `S0_UNDECIDABLE`. |
| V6 no leakage | GT is used **only** to count \(L_{\mathrm{GT}}\); it never enters a coordinate, a threshold, or the region definition |
| V7 non-empty active safe set | ∃ at least one grid point certified safe on offline coordinates (\(\mathrm{UCB}_{\mathrm{off}} \le \varepsilon\)) that is **active** (i.e. its offline reject mask is non-empty, meaning \(N_{\mathrm{FP,removed,off}} > 0\) on matched pairs), else `S0_UNDECIDABLE` (prevents vacuous transfer when the safe set is empty or rejects nothing) |

## 7. Frozen readouts

Per grid point, in **both** coordinate systems: \(N_{\mathrm{GT,exposed}}\),
\(N_{\mathrm{GT,hurt}}\), \(L_{\mathrm{GT}}\), \(\mathrm{UCB}\), \(G_{\mathrm{FP}}\)
with exposures, and the safe/unsafe certification. Plus:

| Readout | Frozen definition |
|---|---|
| axis agreement | per-axis Spearman \(\rho\) between offline and runtime values, on matched pairs |
| direction agreement | count of matched pairs whose safer-side classification flips between substrates (i.e., \(\texttt{dist\_h}_{\mathrm{off}} \ge \theta_d \iff \texttt{dist\_h}_{\mathrm{rt}} < \theta_d\), or same for \(|\texttt{log\_h\_ratio}|\)), evaluated on all points. For any offline-safe grid point, the direction flips must be exactly zero on all axes |
| region agreement | Jaccard similarity \(|D_{\mathrm{off}} \cap D_{\mathrm{rt}}|/|D_{\mathrm{off}} \cup D_{\mathrm{rt}}|\) of the reject masks. If both masks are empty (no pairs rejected by either), Jaccard is defined as 1.0; if only one is empty, it is 0.0 |
| **the decisive readout** | for every \(\theta\) the **offline** substrate certifies safe, the **runtime** \(\mathrm{UCB}\) of that same region |

## 8. Terminal mapping (ordered, disjoint)

| # | Terminal | Condition | Consequence |
|---|---|---|---|
| 0 | `S0_INVALID` | any V1–V3 / V6 gate fails | repair inputs; no conclusion |
| 1 | `S0_UNDECIDABLE` | V4 or V7 fails, **or** V5's worst-case unjoined bound flips the terminal | fail-closed; **no** portability claim in either direction; a wider runtime join is required before any closure solve |
| 2 | **`AXES_TRANSFER_BROKEN`** | ∃ a grid point certified safe on **offline** coordinates (\(\mathrm{UCB}_{\mathrm{off}}\le\varepsilon\)) whose **runtime** harm exceeds the bound (\(\mathrm{UCB}_{\mathrm{rt}}>\varepsilon\)) | the signal axes do not transfer: the offline safety certification is denied by the runtime substrate. Morphology limits amended (§ 8.1); the closure **must** be solved on runtime coordinates. This is what D0's 7.03 % offline-safe/online-unsafe inversion on GT under the s0 proxy warns us about |
| 3 | `AXES_TRANSFER_DEGRADED` | no safety inversion, but \(\rho < 0.98\) on either axis, **or** any safer-side direction flip, **or** Jaccard \(< 0.90\) | axes are **not interchangeable**: definitions may be reused, but every certification must be re-earned at runtime; **no offline number carries over** |
| 4 | `AXES_TRANSFER_HOLDS` | every point in the active safe set is runtime-certified, and all agreement bars pass (\(\rho \ge 0.98\) on both axes, zero direction flips, and Jaccard \(\ge 0.90\)) | the axes may carry into a separately declared closure solve (S2) as a prerequisite for L4, but region-level portability remains unestablished until a closure is solved and tested |

### 8.1 Pre-authorized consequence (documentation only)

On `AXES_TRANSFER_BROKEN` / `AXES_TRANSFER_DEGRADED`: append a limit to the
partial-order note recording that its axes were certified on offline coordinates
and that runtime transfer failed / degraded. This **amends a limit**; it does
**not** retract `GLOBAL_PARTIAL_ORDER_READY`, which remains true on its own
substrate.

**No terminal authorizes:** solving the closure; choosing \(\theta\); enabling
`relink_bridge_max_speed` or `relink_bridge_spatial_gate`; changing `h_lo`/`h_hi`;
any preset, ledger, no-go, or RegionAsset maturity change; any score-ranking claim.

## 9. Seal transition

```text
1. owner review comment containing the literal token:  SEALED
2. an append-only seal record in this section (date, PR, head SHA; draft → active)
3. thread + module TODO transitioned to "sealed; execution authorized"
```

PR merge alone is **not** the seal (framework § 13: *engineering merge \(\not\Rightarrow\)
research acceptance*). Before the seal: no run, no number.

### Seal candidate

- Prepared: 2026-07-13
- Scope: the frozen declaration in §§ 1–8, submitted for owner review
- Owner review: literal token `SEALED` recorded on PR #152 at reviewed head
  `55adbbbcddf52a8bf036ca0a01b3fb0ef859025a`

### Seal record

| Date | PR | Reviewed head SHA | Transition | Authority |
|---|---|---|---|---|
| 2026-07-13 | #152 | `55adbbbcddf52a8bf036ca0a01b3fb0ef859025a` | draft → active | Execute S0 exactly as frozen in §§ 1–8; no later unit or production change authorized |
