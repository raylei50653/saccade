# Change routing matrix (O1)

**Goal:** review by **objective touched**, not only by file path.  
**Companion:** [module_objective_map.md](module_objective_map.md), [objective_template.md](objective_template.md)

If a PR touches multiple objectives, satisfy the **union** of rows. If one of those
objectives is on the same module’s **should-not-own** list, stop and re-scope.

---

## Matrix

| Objective touched | Typical paths | Required checks | Notes |
|:--|:--|:--|:--|
| **RUNTIME** | `eval/pipeline.py`, `eval/evaluator.py`, `eval/stages.py`, `eval/detection.py`, `tracking/*.cu`, native pipeline | **Smoke** at minimum; **MOT17-04-SDP** if association/detect branch changes; **7-seq** if identity/default path | Prefer `mamba_whole_graph` + SDP; double-buffer if that path is claimed |
| **CORRECTNESS** | match/birth/bridge/lifecycle knobs; MOT emit; metrics | Smoke + **MOT17-04** or full **7-seq** when metrics can move; unit tests for pure helpers | Cite evidence_ledger if claiming a number |
| **PERF** | graphs, double-buffer, stage timing, GMC downscale | **Perf smoke** / stage profile; do not claim FPS without method | Attribution docs are RESEARCH if no code path change |
| **CONFIG** | `scripts/eval/config/*`, `eval/config.py`, presets YAML, inject map, CLI flags | **`check_headline_decision_contract.py`** (CI contracts); config consistency / help-tag tests | Headline YAML must not enable NO-GO (C7/C9) |
| **RESEARCH** | `docs/research/*`, ablation reports, probes that are default-off | **No metric requirement**; **citation / evidence source** required for numbers | Must not flip production defaults in the same PR |
| **BRIDGE** | `tracker_gpu.py`, `tracker_gpu_python.cpp`, pybind, packing | **Native build**; pybind/unit; **smoke** if API surface changes | ABI/setter renames need migration note |
| **DEBUG** | dumps, jitter, diagnostics, env probes | **Default-off**; bit-exact or “no metric delta” vs baseline when claimed | Prefer env flags over preset defaults |
| **LEGACY** | old presets, shims, compatibility aliases | Smoke on still-supported path; document deprecation | Do not expand LEGACY without owner |

---

## Composite rules

| Situation | Route |
|:--|:--|
| CONFIG + RUNTIME (inject map) | contracts **and** smoke |
| BRIDGE + RUNTIME | native build + smoke |
| RESEARCH + any production default flip | **split PR** — research docs first or behavior second, not both |
| DEBUG probe left default-on | fail review |
| Dual-stability default change | **not O-series**; needs named decision line + evidence (status closed) |

---

## Suggested command anchors

```bash
# CONFIG / headline contract
uv run python scripts/tools/check_headline_decision_contract.py

# Doc path hygiene (ownership tree)
uv run python scripts/tools/check_doc_stale_paths.py

# RUNTIME / CORRECTNESS smoke (example)
uv run scripts/eval/mot17.py \
  --preset mamba_whole_graph \
  --detector SDP \
  --double-buffer \
  --sequences MOT17-04-SDP
```

7-seq full eval when identity defaults or association kernels change (see project eval runbooks).

---

## Review checklist (short)

```text
□ Primary objective named in PR description
□ File cards: no should-not-own expansion without extraction note
□ Checks from matrix run (or waived with reason)
□ O0 WIP=1: same module not opening a second decision-changing charter; probe / evidence / close stays non-WIP unless explicitly promoted
□ No drive-by dual-stability / preset default in “docs” PR
```
