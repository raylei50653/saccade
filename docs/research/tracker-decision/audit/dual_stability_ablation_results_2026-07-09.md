# Dual Stability 4-Way Ablation Results (2026-07-09)

**Status:** results report only — **production defaults unchanged**  
**Protocol:** [dual_stability_ablation_protocol.md](dual_stability_ablation_protocol.md)  
**Architecture map:** [dual_stability_cleanup.md](dual_stability_cleanup.md)

This PR does **not** flip `stability_cost_w`, `SACCADE_STABILITY_W`, checker
expected values, kernel, or setter. Decision: keep matrix **A** (both on) until a
future behavior PR reopens the question.

---

## 1. Run identity

| Field | Value |
|:--|:--|
| **git SHA** | `2bc556f2a2ae19758878fe2b0778634c9a5c2b2b` (main @ PR #68 merge) |
| **host** | `DESKTOP-0FLA6SQ` |
| **GPU** | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| **date** | 2026-07-09 (+08:00) |
| **preset** | `mamba_whole_graph` (**s** only — **m not run**) |
| **detector** | SDP |
| **scheduling** | `--double-buffer` |
| **data** | `datasets/MOT17` train-half SDP 7-seq |
| **raw outputs** | `results/dual_stability_ablation_20260709/` (local; gitignored) |

### Static guard (pre-run)

```bash
uv run python scripts/tools/check_headline_decision_contract.py
# → ✓ headline decision contract OK (YAML C1–C7/C9 + inject-map C8)
```

### Shared CLI skeleton

```bash
PRESET=mamba_whole_graph
DET=SDP
COMMON=(--preset "$PRESET" --detector "$DET" --double-buffer)
OUT_ROOT=results/dual_stability_ablation_20260709
```

### Arm commands (verbatim pattern)

| Matrix | cost (`--stability-cost-w`) | bid (`SACCADE_STABILITY_W`) | intent |
|:--|:--|:--|:--|
| **A** both on | `0.20` | `0.1` | production baseline |
| **B** cost only | `0.20` | `0` | drop bid |
| **C** bid only | `0.0` | `0.1` | drop cost |
| **D** both off | `0.0` | `0` | no height stability |

```bash
# smoke 04 (gate) then full 7-seq — same pattern per arm
export SACCADE_STABILITY_W=<bid>
uv run scripts/eval/mot17.py "${COMMON[@]}" \
  --stability-cost-w <cost> \
  --sequences MOT17-04-SDP \
  --output "$OUT_ROOT/<arm>_04"

export SACCADE_STABILITY_W=<bid>
uv run scripts/eval/mot17.py "${COMMON[@]}" \
  --stability-cost-w <cost> \
  --output "$OUT_ROOT/<arm>_7seq"
```

Env re-exported before every arm (sticky-env pitfall avoided).

**Per-seq scoring:** offline re-score of the 7-seq track files with
`saccade.perception.eval.metrics` (motmetrics + TrackEval HOTA), same protocol as
`mot17.py` overall path. JSON dump: `results/dual_stability_ablation_20260709/per_seq_metrics.json`.

---

## 2. Smoke — MOT17-04-SDP (4 arms)

| Arm | cost | bid | IDF1 | MOTA | HOTA | AssA | IDs | FP | FN |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **A** both | 0.20 | 0.1 | **93.0** | 92.0 | **84.9** | **85.2** | **24** | 156 | **3627** |
| **B** cost | 0.20 | 0 | 92.4 | **92.1** | 84.7 | 84.6 | 34 | 83 | 3644 |
| **C** bid | 0.00 | 0.1 | 91.9 | 92.0 | 84.5 | 84.3 | **24** | 134 | 3658 |
| **D** off | 0.00 | 0 | 92.3 | **92.1** | **84.9** | 84.9 | 28 | **76** | 3666 |

**Smoke PASS criteria:** no crash/NaN/empty tracks; knobs not bit-identical across
arms; no order-of-magnitude IDs/FP explosion.

| Check | Result |
|:--|:--|
| Crash / empty | none |
| Metrics move with knobs | yes (ΔIDF1 up to ~1.1; IDs 24–34; FP 76–156) |
| Pathological vs A | no |

→ Proceeded to 7-seq for all four arms.

---

## 3. 7-seq aggregate (SDP train-half)

Sequences: `02, 04, 05, 09, 10, 11, 13` (all SDP).

| Arm | IDF1 | MOTA | HOTA | DetA | AssA | IDs | FP | FN |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| **A** both | **78.4** | 78.0 | **70.1** | 70.3 | **70.0** | **425** | 3424 | 20808 |
| **B** cost | 77.4 | 78.1 | 69.6 | 70.5 | 68.9 | 490 | 3227 | 20906 |
| **C** bid | 77.5 | **78.3** | 69.6 | **70.6** | 68.8 | 430 | **3203** | **20744** |
| **D** off | 77.6 | 77.8 | 69.9 | 70.3 | 69.7 | 482 | 3437 | 21004 |

**Δ vs A (aggregate):**

| Contrast | ΔIDF1 | ΔAssA | ΔIDs | Note |
|:--|--:|--:|--:|:--|
| A − B (bid effect @ cost on) | **+1.0** | **+1.1** | **−65** | bid helps IDF1/AssA/IDs |
| A − C (cost effect @ bid on) | **+0.9** | **+1.2** | −5 | cost helps IDF1/AssA |
| A − D (any stability) | **+0.8** | +0.3 | **−57** | both-off loses IDF1/IDs |
| C − D (bid alone) | −0.1 | −0.9 | −52 | bid alone: IDs↓, IDF1 flat |
| B − D (cost alone) | −0.2 | −0.8 | +8 | cost alone: mixed / weak |

Noise floor reminder: protocol treats deltas ≲ 0.2 IDF1 as near-noise; A’s wins
over B/C/D on IDF1 are **~0.8–1.0** — above that bar.

---

## 4. Per-seq tables

### 4.1 IDF1

| Seq | A both | B cost | C bid | D off |
|:--|--:|--:|--:|--:|
| **02** | 59.5 | 59.1 | **59.8** | 59.2 |
| **04** | **93.0** | 92.4 | 91.9 | 92.3 |
| **05** | **74.0** | 72.4 | 72.9 | 71.8 |
| **09** | 68.2 | 66.1 | 66.8 | **69.4** |
| **10** | **59.6** | 57.2 | 58.7 | 57.2 |
| **11** | 78.5 | **79.0** | 78.4 | 78.9 |
| **13** | **70.6** | 68.3 | 68.8 | 70.1 |

### 4.2 AssA

| Seq | A both | B cost | C bid | D off |
|:--|--:|--:|--:|--:|
| **02** | **47.0** | 46.4 | **47.0** | 46.5 |
| **04** | **85.2** | 84.6 | 84.3 | 84.9 |
| **05** | **61.4** | 59.3 | 59.5 | 59.1 |
| **09** | 49.2 | 47.8 | 47.8 | **52.5** |
| **10** | **47.3** | 44.7 | 44.4 | 45.7 |
| **11** | 66.5 | **66.6** | 66.1 | **66.6** |
| **13** | **59.0** | 55.9 | 57.2 | **59.0** |

### 4.3 IDs

| Seq | A both | B cost | C bid | D off |
|:--|--:|--:|--:|--:|
| **02** | **61** | 73 | 63 | 71 |
| **04** | **24** | 34 | **24** | 28 |
| **05** | **55** | 67 | 58 | 66 |
| **09** | 18 | 22 | **16** | 17 |
| **10** | 140 | 151 | **132** | 149 |
| **11** | **24** | **24** | 28 | 25 |
| **13** | **103** | 119 | 109 | 126 |

### 4.4 FP / FN (for completeness)

| Seq | FP A/B/C/D | FN A/B/C/D |
|:--|:--|:--|
| 02 | 224 / 258 / 224 / 281 | 7396 / 7482 / 7390 / 7464 |
| **04** | 156 / **83** / 134 / **76** | **3627** / 3644 / 3658 / 3666 |
| **05** | 276 / 305 / **269** / 327 | **1874** / 1898 / 1884 / 1890 |
| 09 | **38** / 43 / 47 / 55 | **1266** / 1341 / 1296 / 1317 |
| 10 | 1294 / 1281 / **1108** / 1193 | 2773 / **2610** / 2646 / 2761 |
| 11 | 193 / 152 / **142** / 157 | 1540 / **1521** / 1542 / 1522 |
| 13 | 1243 / **1105** / 1279 / 1348 | 2332 / 2410 / **2328** / 2384 |

---

## 5. Focus pair: 04 vs 05 (bipolar check)

| Metric | Seq | A | B | C | D | A best? |
|:--|:--|--:|--:|--:|--:|:--|
| IDF1 | **04** | **93.0** | 92.4 | 91.9 | 92.3 | yes |
| IDF1 | **05** | **74.0** | 72.4 | 72.9 | 71.8 | yes |
| AssA | **04** | **85.2** | 84.6 | 84.3 | 84.9 | yes |
| AssA | **05** | **61.4** | 59.3 | 59.5 | 59.1 | yes |
| IDs | **04** | **24** | 34 | **24** | 28 | tied w/ C |
| IDs | **05** | **55** | 67 | 58 | 66 | yes |

**Bipolar?** **No.** Moving A→B or A→C does **not** flip sign between 04 and 05:
A wins (or ties best) on both sequences for IDF1/AssA. Protocol “do not change
default if bipolar” does not fire.

**04 FP tradeoff:** B and D cut FP on 04 (83 / 76 vs A 156) at the cost of IDs
and IDF1 — consistent with historical bid-side FP notes (#43), but not a bipolar
AssA/IDF1 flip.

---

## 6. Interpretation (not aggregate-only)

| Question | Evidence | Read |
|:--|:--|:--|
| Is **bid** doing anything? | A≫B on IDF1/AssA/IDs; C≈D on IDF1, C better IDs | Bid is **useful with cost on**; alone mainly IDs, not IDF1 |
| Is **cost** doing anything? | A≫C on IDF1/AssA; B≈D weak | Cost is **useful with bid on**; alone not a free win |
| Double-count harm? | A best, not worse than B or C | **No** — stacking is net positive on current stack |
| Either stage useless? | Neither B≈A nor C≈A | **Do not** demote either solely on this matrix |
| Both useless? | D loses IDF1/IDs vs A | **No** |
| Outliers | 09: D best IDF1; 11: B/D tiny edge | Single-seq noise; not policy-changing |
| FP side effect | A higher FP than B/C on aggregate | Real tradeoff (bid/FP); outweighed by AssA/IDs/IDF1 |

---

## 7. Recommendation

| Option | Verdict |
|:--|:--|
| **Keep both** (cleanup **option A**) | **Preferred** — A ≥ others on aggregate IDF1/HOTA/AssA and on 04+05 |
| Cost-only (demote bid env) | **Reject for now** — B loses ~1.0 IDF1 / ~1.1 AssA / +65 IDs vs A |
| Bid-only (`stability_cost_w=0`) | **Reject for now** — C loses ~0.9 IDF1 / ~1.2 AssA vs A |
| Both off | **Reject** — D loses IDF1/IDs vs A |
| Inconclusive | No — deltas above noise floor and directionally consistent |

### Production defaults (this PR)

```text
✗ SACCADE_STABILITY_W default remains 0.1 (code)
✗ stability_cost_w remains 0.20 (mamba_whole_graph*.yaml)
✗ checker expected values unchanged
✗ kernel / setter / packing unchanged
✓ results + docs only
```

### Optional follow-ups (not this PR)

1. **Docs-only polish:** rename / dual-stability narrative still ACTIVE both-on.  
2. **m capacity path:** optional `mamba_whole_graph_m` 7-seq only if a future
   behavior PR retunes m’s `kalman_r_scale` stack with stability — **not required**
   to keep A.  
3. **Behavior PR** only if new evidence (jitter floor, different detector, m
   interaction) reopens demotion — then re-run smoke → 04 → 7-seq ladder.

---

## Related

- Protocol: [dual_stability_ablation_protocol.md](dual_stability_ablation_protocol.md)
- Cleanup RFC: [dual_stability_cleanup.md](dual_stability_cleanup.md)
- Scoring: [../scoring_semantics.md](../scoring_semantics.md)
- Historical bid note: `docs/reference/no_go_registry_details.md` #43
