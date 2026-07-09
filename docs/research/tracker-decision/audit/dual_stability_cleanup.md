# Dual Stability Cleanup (design note)

**Status:** design only — **do not merge / remove / demote knobs yet**  
**Date:** 2026-07-09  
**Scope:** document architecture options for the two height-stability preferences.  
**Not this PR:** kernel edits, preset retunes, env default flips, 7-seq.

Companion: [../scoring_semantics.md](../scoring_semantics.md) § Dual stability;  
[`docs/reference/math_model.md`](../../../reference/math_model.md) §7.7 + §8.2.

---

## Problem

Production path applies **the same geometric signal** (relative height consistency)
at **two different decision stages**, under **two different knobs**:

| Knob | Stage | Control surface | Headline value |
|:--|:--|:--|:--|
| `stability_cost_w` | association **cost** shaping (`Π` → `c_ij`) | YAML preset → `set_stability_cost_w` | **0.20** (s/m) |
| `SACCADE_STABILITY_W` | auction **bid** bias after softmin | **env only** (not in preset) | default **0.1** if unset |

They are **semantically overlapping** but **not mechanically equivalent**.
Ablating “stability” without naming the stage is undefined experiments.

```text
risk: double-count height preference; hard-to-explain ID wins
risk: λ couples only the cost-side term (reward ÷ λ); bid bias does not
risk: reviewers conflate the two when reading logs / math_model
```

---

## What each term actually does

### Cost-side — `stability_cost_w` (YAML ACTIVE)

In `stage1_cost_fused_kernel` (multiplicative path):

```text
penalty -= (stability_cost_w / max(λ, 1)) / (1 + |h_trk − h_det| / h_det)
c = clamp(1 − A · exp(−penalty), 0, 1)
```

- Enters **before** softmin (`p ∝ e^{−λc}`).
- Strengthens size-consistent pairs in the **cost matrix** and sparse top-k.
- **Coupled** with `sinkhorn_lambda`: lower λ amplifies the same `stability_cost_w`.
- Only active when `multiplicative_cost=true` and `stability_cost_w > 0`.

### Bid-side — `SACCADE_STABILITY_W` (ENV ACTIVE)

In auction after value transform (comment in `tracker_gpu.cu`):

```text
bid += stability_w / (1 + |trk_h − det_h| / det_h)   # default stability_w = 0.1
```

- Applied to the **absolute bid**, not inside `c_ij`.
- Favors height-consistent tracks when **contending** for a det (incumbent vs rival).
- Kernel comment historically claimed “IDs −42, IDF1 neutral” for this bias —
  treat as **historical note**, not a current validation claim.
- **Not** in YAML; invisible to preset diffs; easy to forget in ablations.

### Shared signal, different policy roles

| | Cost (`stability_cost_w`) | Bid (`SACCADE_STABILITY_W`) |
|:--|:--|:--|
| Changes | soft ranking of all candidates via `c` | who wins contested auctions |
| Stage order | before softmin / top-k | after softmin, at bid |
| Preset visibility | explicit | env-only |
| λ coupling | yes (`÷λ`) | no |
| Off switch | YAML `0` | `SACCADE_STABILITY_W=0` |

---

## Options (A / B / C)

### A. Keep both layers; rename / document ownership

**Idea:** Accept dual stages as intentional policy stack; fix **names and docs**
so no one confuses them.

| Work item | Behavior change? |
|:--|:--|
| Rename docs / symbols toward `stability_cost_reward_w` vs `stability_bid_w` | docs only first |
| Optional: YAML alias or env rename with deprecation window | later code |
| Checklist: every “stability ablation” must name cost vs bid | process |
| Keep math_model §3.1 dual-row (already correct) | none |

**Pros**

- Zero behavior risk; cheapest path.
- Matches reality: cost shapes pool, bid breaks ties.
- Can land entirely in docs + naming in follow-ups.

**Cons**

- Surface still dual; double-count risk remains.
- Env knob stays invisible to preset review.

**When to pick A:** Default until a controlled A/B shows one stage is redundant
after multiplicative retune.

**P7 evidence (2026-07-09):** 4-way ablation on `mamba_whole_graph` **and**
`mamba_whole_graph_m` (SDP + double-buffer) → **keep A (both on)**. Full tables:
[dual_stability_ablation_results_2026-07-09.md](dual_stability_ablation_results_2026-07-09.md).
Production defaults not flipped.

**Evidence needed to leave A:** none for “document-only A”; for rename-in-code,
need ABI/CLI migration plan only (not MOT metrics).

---

### B. Converge to a single stability policy

**Idea:** One calibrated height-consistency policy; remove the other from
production path.

Sub-choices:

| B1 | Keep **cost only**; set `SACCADE_STABILITY_W=0` (or remove env default) |
| B2 | Keep **bid only**; set `stability_cost_w=0` and retune λ if needed |

**Pros**

- Single story for reviewers and ablations.
- Removes double-count.

**Cons**

- **Behavior change** — needs smoke → MOT17-04 → 7-seq.
- Cost and bid are not substitutes: turning one off is not equal to retuning the other.
- B2 removes a YAML-visible knob; B1 removes silent env bias (may change IDs even if IDF1 flat).

**When to pick B:** Only after paired experiment:

```text
1. Baseline (both on)
2. cost-only (bid=0)
3. bid-only (cost=0)
4. both off
```

Report **per-seq AssA/IDF1** and qualitative switch/frag notes — not aggregate-only.

**Evidence bar:** 7-seq with no material AssA regression on 02/04/05/09/10/11/13;
document which identity failure modes move.

---

### C. One production; demote the other to experimental / env-only

**Idea:** Pick a **headline** stability and make the other explicitly non-production.

Recommended default **if** choosing C without full merge:

| Role | Knob | Status after C |
|:--|:--|:--|
| **Production** | `stability_cost_w` (YAML 0.20) | ACTIVE — visible, injectable, reviewed in presets |
| **Experimental** | `SACCADE_STABILITY_W` | ENV LATENT / opt-in; default **0** or documented “legacy bid bias” |

Alternative (less preferred): production bid + experimental cost — worse preset
reviewability.

**Pros**

- Shrinks accidental dual-count while keeping a research lever.
- Aligns with “ACTIVE surface small / env for ablation.”

**Cons**

- Changing env default 0.1 → 0 is **behavior change** (needs P4 smoke ladder).
- Historical runs with unset env silently used 0.1 — reproducibility note required.

**When to pick C:** Prefer as intermediate if B is too large but dual-on is
hurting explainability; still requires metrics before flipping default.

---

## Decision matrix (for a future decision PR)

| Criterion | A | B | C |
|:--|:--|:--|:--|
| Behavior risk now | none | high | medium (if default flips) |
| Explainability | medium | high | high |
| Ablation clarity | medium | high | high |
| Engineering cost | low (docs) | high (retune) | medium |
| Fits “no kernel change” phase | **yes** | no | only if default unchanged |

### Recommendation (P4 → P7)

```text
P0–P6:  Option A stance — document + healthcheck; production both on
P7:     4-way ablation protocol (measure first; do not flip defaults yet)
          → docs/research/tracker-decision/audit/dual_stability_ablation_protocol.md
Later:  Behavior PR only after results map to keep-both / cost-only / bid-only
Never:  merge casually without naming stages
```

Until architecture A/B/C is chosen **and** backed by the P7 matrix:

- Treat **both** as ACTIVE for healthchecks (cost YAML + env default on).
- Any PR that touches only one must say so in the description.
- Do **not** sweep both knobs in one “stability” hyperparam study without
  orthogonal factors.
- Run matrix **A/B/C/D** (both / cost-only / bid-only / both-off) per the
  [ablation protocol](dual_stability_ablation_protocol.md) — not architecture
  labels alone.

---

## Explicit non-goals (this note)

```text
✗ Change stability_cost_w or SACCADE_STABILITY_W values
✗ Remove env read or set_stability_cost_w
✗ Rename native symbols without a dedicated ABI PR
✗ Claim IDF1 impact without re-running eval
```

---

## Implementation sketch (future only)

If C with default flip (example):

1. Docs + healthcheck: expect bid default 0.
2. Code: change getenv fallback `0.1f` → `0.0f` **or** require explicit env.
3. Smoke MOT17-04-SDP → 7-seq.
4. Changelog: “silent bid bias removed.”

If B1 (cost-only, same default flip path as C + retire docs for bid).

If A rename only:

1. Docs / math_model labels.
2. Optional Python alias `stability_bid_w` → env bridge (no kernel rename first).

---

## Related

- [config_surface.md](config_surface.md) §1 (both knobs listed)
- [native_bridge.md](native_bridge.md) (env vs setter)
- [active_contract_healthcheck.md](active_contract_healthcheck.md) (dual still separate)
- [no_go_guardrails.md](no_go_guardrails.md) (env dual-sweep caution)
