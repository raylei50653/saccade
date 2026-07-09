# Dual Stability 4-Way Ablation Protocol (P7)

**Status:** experiment protocol only — **do not change production defaults in this phase**  
**Date:** 2026-07-09  
**Follows:** [dual_stability_cleanup.md](dual_stability_cleanup.md) (architecture A/B/C options)  
**Results:** open a separate report PR after runs (suggested: `docs/research/tracker-decision/audit/dual_stability_ablation_results_YYYY-MM-DD.md`)

Guardrail phase (P0–P6) is closed. This document defines how to **measure** whether
cost-side and bid-side height stability are both needed, before any behavior PR.

---

## Non-goals (this phase)

```text
✗ Flip SACCADE_STABILITY_W default 0.1 → 0 in tracker_gpu.cu
✗ Change stability_cost_w in mamba_whole_graph*.yaml
✗ Merge dual knobs into one symbol
✗ Kernel / setter / packing renames
✗ Claim a production decision without 7-seq + per-seq bipolar read
```

Production remains **variant A** until a results PR + decision is merged.

---

## Knobs under test

| Knob | Stage | Control | Production (A) |
|:--|:--|:--|:--|
| `stability_cost_w` | cost Π reward (multiplicative) | YAML / CLI `--stability-cost-w` | **0.20** |
| `SACCADE_STABILITY_W` | auction bid bias | **env only** | **0.1** if unset |

Formulas and ownership: [dual_stability_cleanup.md](dual_stability_cleanup.md),  
[../scoring_semantics.md](../scoring_semantics.md) § Dual stability,  
[`math_model.md`](../../../reference/math_model.md) §7.7 + §8.2.

**Invariant for every arm:** all other headline ACTIVE knobs stay at contract
values (`check_headline_decision_contract.py` still passes for YAML; bid is env-only).

---

## 4-way matrix

| ID | Name | `stability_cost_w` | `SACCADE_STABILITY_W` | Intent |
|:--|:--|:--|:--|:--|
| **A** | both on | **0.20** | **0.1** (env unset or explicit) | Production baseline |
| **B** | cost only | **0.20** | **0** | Drop bid bias |
| **C** | bid only | **0.00** | **0.1** | Drop cost reward |
| **D** | both off | **0.00** | **0** | No height-stability preference |

Naming note: cleanup doc options A/B/C (architecture) ≠ matrix IDs A–D.
In reports always write **matrix A/B/C/D** explicitly.

---

## How to set each arm

Headline preset: `mamba_whole_graph` (s) for primary decision read.  
Optional capacity check: `mamba_whole_graph_m` **after** s 7-seq if s shows a
clear winner that might interact with m’s higher `kalman_r_scale` (not required
for first decision).

### Shared CLI skeleton

```bash
# Primary decision path (s, SDP, double-buffer)
PRESET=mamba_whole_graph
DET=SDP
COMMON=(--preset "$PRESET" --detector "$DET" --double-buffer)

# Optional: pin output root
OUT_ROOT=results/dual_stability_ablation_$(date +%Y%m%d)
mkdir -p "$OUT_ROOT"
```

### Arm A — both on (production)

```bash
# Env: leave SACCADE_STABILITY_W unset, OR
export SACCADE_STABILITY_W=0.1
unset SACCADE_STABILITY_W   # also OK — code default is 0.1

uv run scripts/eval/mot17.py "${COMMON[@]}" \
  --stability-cost-w 0.20 \
  --output "$OUT_ROOT/A_both"
```

If the preset already has `stability_cost_w: 0.20`, the CLI flag is redundant
but keeps arms self-documenting in logs.

### Arm B — cost only

```bash
export SACCADE_STABILITY_W=0
uv run scripts/eval/mot17.py "${COMMON[@]}" \
  --stability-cost-w 0.20 \
  --output "$OUT_ROOT/B_cost_only"
```

### Arm C — bid only

```bash
export SACCADE_STABILITY_W=0.1
uv run scripts/eval/mot17.py "${COMMON[@]}" \
  --stability-cost-w 0.0 \
  --output "$OUT_ROOT/C_bid_only"
```

### Arm D — both off

```bash
export SACCADE_STABILITY_W=0
uv run scripts/eval/mot17.py "${COMMON[@]}" \
  --stability-cost-w 0.0 \
  --output "$OUT_ROOT/D_both_off"
```

### Run log hygiene (required)

For every arm, record in the results doc:

```text
- git SHA
- preset + detector + double-buffer flag
- stability_cost_w (CLI/preset)
- SACCADE_STABILITY_W (env; "unset→default 0.1" vs "0")
- host / GPU / date
- command line verbatim
```

**Pitfall:** process env is sticky. Always re-export `SACCADE_STABILITY_W`
before each arm; do not rely on a previous shell state.

**Pitfall:** CUDA-graph / tracker-graph paths must match across arms (use the
same `COMMON` flags). Do not mix with debug envs (`SACCADE_ASSOC_DUMP`, jitter)
unless explicitly studying noise floors.

---

## Evaluation ladder

### Step 0 — Static guard (no GPU metrics)

```bash
uv run python scripts/tools/check_headline_decision_contract.py
```

Does **not** validate env bid weight; only YAML + inject map. Still run so
ablation tooling did not desync presets.

### Step 1 — Smoke / single-seq (MOT17-04-SDP)

Gate before 7-seq:

```bash
# Example for arm B (repeat for A/C/D)
export SACCADE_STABILITY_W=0
uv run scripts/eval/mot17.py "${COMMON[@]}" \
  --stability-cost-w 0.20 \
  --sequences MOT17-04-SDP \
  --output "$OUT_ROOT/B_cost_only_04"
```

**Smoke FAIL (stop arm / investigate):**

- Crash, NaN, empty tracks, metric script failure
- Pathological IDs/FP explosion vs A on the **same** SHA (order-of-magnitude)
- Bit-exact identity with A when knobs differ (suggests knob not wired — check env/CLI)

**Smoke PASS:** proceed to 7-seq for all arms that pass.

### Step 2 — 7-seq (full SDP train-half set)

Project-standard 7 sequences (SDP), same as other decision ablations:

```text
MOT17-02-SDP, 04, 05, 09, 10, 11, 13
```

Run all four arms A–D with identical `COMMON` and pinned SHA.

### Step 3 — Metrics to report

| Level | Metrics |
|:--|:--|
| Aggregate | IDF1, HOTA, AssA, MOTA, IDs, FP, FN |
| **Per-seq** | same columns for each of 7 seqs |
| Focus pair | **04 vs 05** (persistent crowd vs sparse crossings) |

Also note qualitative ID failure modes if available (switch / fragment) — optional.

### Step 4 — How to read (not aggregate-only)

| Question | Look at |
|:--|:--|
| Is bid doing anything? | A vs B (same cost); C vs D (no cost) |
| Is cost doing anything? | A vs C (same bid); B vs D (no bid) |
| Double-count harm? | A worse than B **or** C on AssA/IDs with no IDF1 win |
| Bipolar risk? | Sign flip on **04 vs 05** IDF1/AssA when moving A→B or A→C |
| Noise floor | Use project jitter / repeat discipline if deltas &lt; ~0.2 IDF1 aggregate |

Historical context (not a substitute for this run): no_go registry #43 noted
bid-side as a weak / binary tiebreaker with FP side effects in some setups.
Re-measure on **current** multiplicative + OAO + private continuation stack.

---

## Decision mapping (after results)

Map matrix outcomes → cleanup architecture choice:

| If evidence shows… | Prefer | Production action (later PR) |
|:--|:--|:--|
| A ≥ others; B/C each hurt something | **Keep both** (cleanup option A) | Docs/rename only; no default flip |
| B ≈ A and C ≈ D (bid useless) | **Cost-only** (cleanup option C demote bid) | Set env default `0` or require explicit env; keep YAML 0.20 |
| C ≈ A and B ≈ D (cost useless) | **Bid-only** | Set `stability_cost_w: 0` in preset + retune check; keep env 0.1 — **rare**, needs strong 7-seq |
| D ≈ A (both useless) | Drop height stability | Both off — high bar; re-check size-jitter failure modes |
| Bipolar 04/05 | **Do not** change default | Keep A; document tradeoff |

Any production change is a **behavior PR**: smoke → 04 → 7-seq again on the
chosen arm, then update:

- headline preset and/or `tracker_gpu.cu` env default
- [active contract](../README.md) + healthcheck expected values
- [dual_stability_cleanup.md](dual_stability_cleanup.md) status
- math_model §7.7 / §8.2 baseline tags

---

## Results PR template (PR #69-style)

Suggested path:

```text
docs/research/tracker-decision/audit/dual_stability_ablation_results_YYYY-MM-DD.md
```

Minimum sections:

```text
1. SHA / host / commands for A–D
2. Smoke 04 table (4 arms)
3. 7-seq aggregate table
4. Per-seq IDF1 / AssA / IDs / FP / FN (at least 04 and 05 called out)
5. Bipolar check summary
6. Recommendation → keep both | cost-only | bid-only | both off | inconclusive
7. Explicit: production defaults NOT changed in that PR unless decision + ladder done
```

---

## Related

- Architecture options: [dual_stability_cleanup.md](dual_stability_cleanup.md)
- Active contract: [../README.md](../README.md)
- Scoring semantics: [../scoring_semantics.md](../scoring_semantics.md)
- Pipeline path (s/m): [../../pipeline/](../../pipeline/)
- Historical bid note: `docs/reference/no_go_registry_details.md` #43
