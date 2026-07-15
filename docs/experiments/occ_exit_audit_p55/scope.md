# #55 occ-exit audit — research PR scope (2026-07-09)

**Branch:** `research/occ-exit-audit-p55`  
**Series context:** O0 WIP=1 · O1 ownership map (semantic sole active = this item)  
**Objective (O1):** **RESEARCH** + **DEBUG** (not RUNTIME promotion)

---

## Intent

```text
PR intent:
  semantic / RESEARCH + DEBUG
  occ-exit audit condition prototype
```

Finish the **audit / 條件化** path on top of substrate already landed:

| Piece | Status |
|:--|:--|
| CleanFifoBank pre-episode reference | ✅ wired (`samples_before`, `--occ-audit-bank-reference`) |
| `occ_exit_audit_lines` / `_from_bank` | ✅ cosine audit core (unit-tested) |
| Evaluator adapter + log CSV | ✅ default-off flags |
| Probe bank vs post-hoc | ✅ `probe_occ_audit_bank_reference.py` |
| **Cheb-GR graph decision** (not cosine-only) | ❌ remaining |
| **13-type sequence conditioning** (not global on) | ❌ remaining |

Substrate contracts: [clean_fifo_bank_substrate_20260704.md](../../modules/semantic/research/clean_fifo_bank_substrate_20260704.md).  
TODO entry: [semantic/TODO.md](../../modules/semantic/TODO.md) § Active.

---

## Why `research/`, not `feat/`

#55 is still described as **audit / 條件化復活**. Until there is an explicit
decision to change **production tracker defaults** or the **live critical path**,
work stays research:

- default-off flags only  
- evidence / reports / diagnostics  
- no promotion into headline preset  

If mid-flight we need production tracker behavior, **close this research PR** and
open a separate:

```text
feat/occ-exit-conditional-audit
```

Do not upgrade the same PR from probe → live default.

---

## Allowed

- Use existing CleanFifoBank / occ-audit substrate  
- Add Cheb-GR **graph** decision probe (k-reciprocal path; respect bank hard constraints: raw samples, no mean, no dupfill into graph)  
- Add **13-type** (or MOT17-seq) conditioning so flags/relabel are sequence-gated, not global  
- Produce evidence / report / diagnostic output (`_occ_audit.csv`, probe JSON/MD)  
- **default-off** flags / module YAML only  

## Not allowed

- Production default flips (`mamba_whole_graph*.yaml` headline)  
- Offline GO → live critical path without promotion bar (#57 / #56 lessons)  
- Sparse bank C++ async sidecar (parked under O0; different WIP)  
- O2 ownership notes  
- Mechanical split of `evaluator.py` / `relink.py`  

---

## O1 routing

| Objective | Rule |
|:--|:--|
| **RESEARCH** touched | Evidence / source required for any claimed metric |
| **DEBUG** touched | default-off; no metric delta claim without baseline pair |
| **RUNTIME** touched | Immediately require smoke / MOT17-04; prefer **split** to `feat/` PR |

Semantic card (should-not-own): RUNTIME critical-path ReID; dual-stability defaults; preset birth policy.  
See [docs/ownership/module_objective_map.md](../../ownership/module_objective_map.md).

---

## Current decision surface (cosine audit)

Numeric core today (`occ_audit.py`):

```text
episode plan (visclean front-occlusion geometry)
  → ref = pre-episode clean crops (or bank.samples_before)
  → audit = post-exit clean crops
  → flag if min cosine(ref, audit) < tau
  → relabel from decision frame onward (causal; no rewrite of past emit)
```

Defaults (all off unless flagged): `occ_audit`, `occ_audit_bank_reference`,
`occ_audit_log`, thresholds `tau` / `ref_n` / `window` / `min_occ` in
`scripts/eval/config/lifecycle.py`.

---

## Remaining work packages (this research line)

### WP1 — Cheb-GR graph decision probe

- Input: bank raw samples (ref) + post-exit audit set  
- Compare cosine-min gate vs Cheb-GR cost/margin (reuse offline handover signal map lessons: `best_cost` durable; do not invent live claim)  
- Output: decision log columns + per-seq flag delta vs cosine baseline  
- **Hard constraints:** no mean prototype into graph; unique samples only  

### WP2 — Sequence conditioning (13-type / MOT17 train set)

- Applicability map style: which sequences benefit / harm from audit relabel  
- Gate: enable only when seq (or scene type) passes condition; else abstain  
- Evidence: per-seq IDF1/IDs on frozen substrate; bipolar read required  

### WP3 — Research report

- Single MD under `docs/modules/semantic/research/`  
- Link probe commands + result dirs; cite numbers only from runs  

### Explicit non-goals of first PR

- C++ `occ_state` live coupling (may be **documented** as precondition; not required for probe)  
- Changing auction / dual-stability / bridge defaults  

---

## Suggested first commands (after code)

```bash
# existing bank-reference probe (substrate must exist)
.venv/bin/python scripts/eval/diagnostics/probe_occ_audit_bank_reference.py \
  --substrate results/diag_m_no_reid_current_20260704 \
  --data-root datasets/MOT17 --split train

# unit baselines
.venv/bin/pytest -q tests/unit/reid/test_occ_audit.py \
  tests/unit/eval/test_occ_audit_bank_reference.py \
  tests/unit/eval/test_clean_fifo_bank.py
```

---

## Exit criteria for research PR

- [ ] Cheb-GR graph decision path exists **behind default-off flag**  
- [ ] Sequence conditioning documented + measured (not global always-on)  
- [ ] Evidence report with per-seq read  
- [ ] No headline preset change  
- [ ] O0: sparse bank still parked; no second semantic active  

Promotion to `feat/occ-exit-conditional-audit` only if evidence supports live path
and routing upgrades to RUNTIME checks.
