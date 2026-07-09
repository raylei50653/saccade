# Objective template (O1)

Catalog of **objective types** and the **card schema** used in
[module_objective_map.md](module_objective_map.md).

---

## Objective types

| Code | Name | Meaning |
|:--|:--|:--|
| **RUNTIME** | Production runtime path | Detect → track → emit path that ships / is eval default |
| **CORRECTNESS** | Metric / output correctness | MOT metrics, MOT txt identity, birth/death, parity |
| **PERF** | Latency / FPS / profiling | Stage timing, graphs, double-buffer, attribution |
| **CONFIG** | Schema / preset / CLI / env | Inject map, YAML, argparse, env knobs |
| **RESEARCH** | Ablation / report / evidence | Offline probes, ledger, paper skeleton, NO-GO notes |
| **BRIDGE** | Native binding / ABI | pybind, tensor ownership, setter packing, CUDA graph capture contracts |
| **DEBUG** | Probes / dumps / diagnostics | Jitter dumps, frame dumps, default-off diagnostics |
| **LEGACY** | Retained compatibility | Old presets, shims, dead paths kept for bisect |

---

## Card schema

Every module or hot file card uses:

```text
Path / Module
Primary:          exactly 1 objective (or one short role phrase + type code)
Secondary:        0–2 objectives
Should-not-own:   explicit list of objectives or concerns this unit must not absorb
Risk:             how coupling fails today (optional but preferred for hot files)
Extraction:       candidate responsibilities to move later (O3/O4; no moves in O1)
Required checks:  default verification when this unit is touched
```

### Rules of thumb

| Rule | Detail |
|:--|:--|
| Primary ×1 | If two primaries fight, split is already justified (record under Extraction) |
| Secondary ≤2 | More than two → document extraction or park secondary under Should-not-own |
| Should-not-own | Prefer **objectives**, not file names (“RESEARCH report”, not “random.md”) |
| Extraction | Reason + target owner type; **no** move plan dates required in O1 |
| Checks | Prefer existing tools (`check_headline_decision_contract.py`, smoke, 7-seq) |

---

## Module-level vs file-level

- **Module card** (e.g. `detection`, `semantic`): maps to `docs/modules/<m>/` + primary code paths.  
- **Hot-file card**: used when one file mixes objectives even if the module is “fine.”  
  Prefer file cards for LOC or fan-in hotspots (`evaluator.py`, `pipeline.py`,
  `tracker_gpu.cu`, `tracker_gpu_python.cpp`, `stages.py`, `relink.py`,
  `scripts/eval/config/*`).

---

## Example (canonical shape)

```text
src/saccade/perception/eval/pipeline.py
Primary:     RUNTIME
Secondary:   CONFIG (injection into runtime path)
Should-not-own: RESEARCH report, PERF attribution narrative
Risk:        runtime mode and config policy coupled in one class
Extraction:  runtime mode resolver; config inject helper
Checks:      check_headline_decision_contract.py (if inject map); smoke if branch touched
```

Copy this shape into [module_objective_map.md](module_objective_map.md); do not invent
new fields without updating this template.
