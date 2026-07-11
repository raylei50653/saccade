## Summary

<!-- What changed and why (1–3 bullets). -->

## Decision-layer change?

Mark any that apply. If none, leave unchecked.

- [ ] Touched headline presets (`mamba_whole_graph*.yaml`)
- [ ] Touched schema / argparse defaults (`scripts/eval/config/**`)
- [ ] Touched pipeline inject (`pipeline.py` `set_*` / private det-set)
- [ ] Touched tracker cost / auction / relink / occ kernel
- [ ] Touched decision docs under `docs/research/tracker-decision/`

### Guardrails (when any box above is checked)

```bash
uv run python scripts/tools/check_headline_decision_contract.py
# optional focused tests:
uv run pytest tests/unit/test_headline_decision_contract.py -q
```

- [ ] `check_headline_decision_contract.py` passes (YAML C1–C7 + inject C8 + surface C9)
- [ ] Active contract / allowlist updated if intentional new ACTIVE keys
- [ ] NO-GO / LATENT not promoted without [promotion bar](docs/research/tracker-decision/audit/no_go_guardrails.md)

### Behavior change?

- [ ] **No** — docs/tools/help only (no smoke / 7-seq required)
- [ ] **Yes** — run ladder before merge:

```text
smoke → MOT17-04-SDP → 7-seq
```

Do **not** merge dual-stability (`stability_cost_w` vs `SACCADE_STABILITY_W`) or flip NO-GO defaults without a dedicated behavior PR.

### Research study? (new decision-layer experiment or result promotion)

- [ ] Role-aligned experiment contract declared per [framework §20](docs/research/eval/statistical_robust_feasible_set_estimation_under_asymmetric_loss.md), auditable at:

```text
Contract declaration location:  <issue / study doc / framework anchor>
Primary target layer:           <coarse gate / score-ranking / assignment / calibration / none>
Primary study intent:           <design evaluation / capability map / boundary diagnostic / upper-bound probe>
Mainline terminal mapping:      <terminal -> state transition per outcome, or "diagnostic-only">
```

## Test plan

- [ ] …
