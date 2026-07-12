# Door 0 — ambiguous-band ranking-power probe: results

<!-- doc-status: active -->
<!-- doc-promotion: study results under sealed declaration; research acceptance pending owner review -->
<!-- doc-date: 2026-07-12 -->
<!-- doc-module: semantic -->

## 2026-07-12 validity amendment (normative — read before the rest)

Issue #112 measured this study's baseline ordering `s0 = score_m_bridge` against
the live CUDA `bdist` it was assumed to represent. It **failed all three fidelity
boxes** — decision agreement 95.07 % (bar 99 %), |Δ| q95 = 1.417 (bar 0.05),
**Spearman ρ = 0.9558 (bar 0.98)** — with every validity gate passing and
coverage passing. Terminal: `T2 PROXY_UNFAITHFUL`.
See [s0 proxy validity amendment](s0_proxy_validity_amendment_20260712.md).

Consequence for **this** study's terminal:

* **Still true:** the 12-member Door-0 class contains no candidate that usably
  improves ranking **over `s0`**. The class remains closed with respect to `s0`.
* **No longer supported:** reading `T2 NO_USABLE_RANKING_POWER_IN_CLASS` as a
  statement about the **production `bdist` ordering**. With ρ = 0.9558 the two
  orderings are not interchangeable, so this study does not establish the absence
  of ranking power over production's actual score.
* **Scoping limit, not a reversal.** The study remains valid; its reach is
  narrower than originally recorded.

**Owner ruling (2026-07-12, PR #141) — final:** this study **stays CLOSED**, but
its **mainline-transition status is revoked**; it is reclassified as a
**`proxy-space capability closure`**. It remains a valid completed study (the
12-member class is closed with respect to `s0`), it is not a mainline state
transition (§20.7) because it cannot change the research status of the production
`bdist` ordering, and it does **not** need to be reopened. No owner call remains
open.

Everything below is unmodified and must be read under this scope.

Everything below is unmodified and must be read under this scope.

> **Terminal: `T2 NO_USABLE_RANKING_POWER_IN_CLASS` (12-member tested
> class).** Validity V1–V5 all PASS; headroom exists (H not triggered:
> baseline PWA 0.878, top-1 0.590; 84/205 events are baseline top-1 misses);
> **no candidate passes the B1–B6 boxes** — best in-sample ΔPWA =
> **+0.001097** (`c_log_h_ratio`), ≈18× below the +0.02 bar; the motion
> conditions (`speed_mismatch`, `dir_cos`) are actively harmful.
>
> **Unconditional scope caveat (declaration §3, verbatim):** study scope is
> the gate-retained band; this terminal establishes no claim inside the
> production-reachable set (s0 ≤ 0.4; 34 events, descriptive only); any
> step-④ decision must treat the decision surface (threshold/margin
> interplay), not assume in-place reranking behavior.

Declaration (sealed, PR #135 merge = seal): [ambiguous_band_ranking_power_probe_declaration_20260712.md](ambiguous_band_ranking_power_probe_declaration_20260712.md) ·
Thread: [thread card](../../../research/threads/closed/ambiguous_band_ranking_power_probe_20260712.md) ·
Runner: `scripts/tools/run_door0_ranking_probe.py` ·
Packet: [evidence/door0_ranking_probe_20260712/](evidence/door0_ranking_probe_20260712/manifest.json) ·
Study dir: `out/signal_study/door0_ranking_probe_20260712/`

Single authorized execution per declaration §11; no deviation from the
sealed declaration (atoms, thresholds, metrics, boxes, terminals all as
sealed; a synthetic-data self-test of the metric functions preceded the run
and touched no substrate).

## 1. Validity and headroom

```text
V1 PASS  pairs.csv sha256 = 0ae38967…56a69d17 (declared value)
V2 PASS  rankable events = 205 (≥150)
V3 PASS  per-seq events: 02:50 04:12 05:32 09:12 10:69 11:13 13:17 (≥10 in ≥6/7)
V4 PASS  GT-row retention through gate = 329/340 = 96.8 % (≥90 %)
V5 PASS  NaN-dropped rows = 0 (≤5 %)

H  NOT TRIGGERED
   baseline (s0): pooled PWA = 0.8778 · MRR = 0.7327 · top-1 = 0.5902
   P3 (baseline top-1 miss events) = 84/205
```

The band is **not** already solved by baseline geometry: in 41 % of rankable
events at least one FP outranks the GT. The headroom is real; the tested
class simply cannot reach it.

## 2. Candidate table (all 12; in-sample Δ vs baseline, macro over events)

| candidate | ΔPWA | ΔMRR | Δtop-1 | B3 min-excl | LOO ΔPWA (folds ≥0) | fire FP/GT | good/bad flips | verdict |
|:--|--:|--:|--:|--:|--:|:--|:--|:--|
| `c_dist_h` | +0.0006 | 0 | 0 | +0.0003 | +0.0004 (7/7) | 0.095/0.000 | 11/0 | fail (B1,B2,B3,B5) |
| `c_log_h_ratio` | **+0.0011** | +0.0057 | +0.0049 | +0.0005 | +0.0017 (7/7) | 0.152/0.012 | 156/55 | fail (B1,B2,B3,B5) |
| `c_resid_mean` | 0.0000 | 0 | 0 | 0 | 0 (7/7) | 0.096/0.000 | 0/0 | fail |
| `c_speed_mismatch` | −0.0173 | −0.0049 | −0.0049 | −0.0227 | −0.0219 (3/7) | 0.098/0.086 | 32/216 | fail (all) |
| `c_gap` | −0.0065 | +0.0002 | 0 | −0.0133 | −0.0064 (6/7) | 0.170/0.031 | 96/141 | fail |
| `c_dir_cos` | −0.0354 | −0.0260 | −0.0244 | −0.0426 | −0.0360 (1/7) | 0.128/0.043 | 70/190 | fail (all) |
| `c_dist_h ∧ c_gap` | +0.0002 | 0 | 0 | +0.0001 | 0.0000 (7/7) | 0.020/0.000 | 3/0 | fail |
| `c_dist_h ∧ c_speed_mismatch` | 0.0000 | 0 | 0 | 0 | 0 (7/7) | 0.019/0.000 | 0/0 | fail |
| `c_gap ∧ c_speed_mismatch` | 0.0000 | 0 | 0 | 0 | 0 (7/7) | 0.023/0.000 | 0/0 | fail |
| `c_gap ∧ c_dir_cos` | +0.0003 | +0.0001 | 0 | +0.0002 | +0.0002 (7/7) | 0.019/0.000 | 6/0 | fail |
| `c_dist_h ∧ c_log_h_ratio` | +0.0002 | 0 | 0 | +0.0001 | +0.0002 (7/7) | 0.013/0.000 | 3/0 | fail |
| `c_resid_mean ∧ c_gap` | 0.0000 | 0 | 0 | 0 | 0 (7/7) | 0.048/0.000 | 0/0 | fail |

(Full precision, per-seq columns, per-fold LOO, and P3 restrictions:
`candidates.csv` in the packet. B-box bit patterns in `results.txt`.)

Reading:

- **No candidate approaches B1** (+0.02): the best is +0.001097 — an ≈18×
  shortfall. This is not a near-miss (owner acceptance: not a boundary case).
- The "clean" conditions (`dist_h`, `resid_mean`, the ANDs) fire almost
  exclusively on FPs (fire GT = 0.000) but those FPs are already ranked
  *below* the GT by `s0` — demoting them changes nothing (`good/bad = 0/0`
  for several). The tail mass the class can see is **already ordered**.
- The motion conditions are harmful exactly as the escape-tail forensics
  predicted: `speed_mismatch` and `dir_cos` fire on GT nearly as often as on
  FP inside the band (fire GT 0.086 / 0.043), so demotion flips good pairs
  to bad (216 / 190 bad flips).
- `c_log_h_ratio` is the only candidate with any positive movement on all
  three metrics and 7/7 LOO folds — but at +0.11 pp it is noise-order, and
  its fire-GT = 0.012 already shows the tail is not GT-free.

## 3. Descriptive layers (no terminal force)

- **P1 pooled separability** (rankable-event rows): `resid_mean` 0.833,
  `dist_h` 0.826, `gap` 0.734, `log_h_ratio` 0.723, `dir_cos` 0.648,
  `speed_mismatch` 0.514. The geometry atoms separate pooled GT from FP —
  but that information is already inside `s0`; conditioning tails on top of
  it adds nothing event-locally.
- **Reachable slice (34 events, descriptive only):** baseline PWA 0.588,
  top-1 0.529 — the production-reachable subset is *harder* than the band
  average, and candidate effects there are ±0.03 at most (1-event
  movements). Consistent with the caveat: nothing is established in-slice.

## 4. Terminal and mainline transition (§20.7)

```text
TERMINAL: T2 NO_USABLE_RANKING_POWER_IN_CLASS
scope:    exactly the 12 enumerated candidates (frozen q85/q15,
          λ→∞ lexicographic demotion) on frozen pairs 0ae38967…
caveat:   attached unconditionally (§3, quoted at top)
```

Mainline transition (step ⑤, class-scoped): the retained ambiguous band is
recorded as an **unexplained residual set with respect to the 12-member
Door-0 tested class**. Re-running any of the 12 members on this
family/substrate is blocked. Explicitly **not** established: anything about
the 9 untested AND pairs, other quantiles, continuous signals, finite-λ
weightings, or learned scores — probing any wider or different set requires
a new §20.2 declaration and an explicit owner charter decision.

What the capability map adds beyond the terminal: the band's residual
top-1-miss mass (84 events) is **not** reachable by unsafe-tail demotion of
the frozen atom family — the class's reject-able tail is already ordered by
`s0`, and the remaining confusions live *inside* the joint in-band
distribution where tail conditions cannot separate them. Any future ranking
attempt needs signals (or transforms) that act off the tails.

## 5. Production / promotion

```text
production preset:   unchanged (read-only probe)
promotion:           none; output class = diagnostic result (§20.4);
                     §20.5 applies to every number above
step ④:              not opened by this route
```
