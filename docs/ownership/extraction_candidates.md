# Extraction candidates (O1)

**Status:** annotate only — **no code movement** in O1.  
**Promotion:** candidates feed **O3** (plan) → **O4** (first mechanical split with tests).

Each row: **what** · **from** · **why** · **suggested owner objective after extract**.

---

## Priority band A (high coupling, high fan-in)

| Candidate | From | Why extract | After-extract objective |
|:--|:--|:--|:--|
| Runtime mode / double-buffer eligibility resolver | `eval/pipeline.py` | CONFIG + RUNTIME + DEBUG probes mixed | RUNTIME pure helper; probes → DEBUG |
| Config inject map (setters + private det-set path) | `eval/pipeline.py` | Contract C8 lives here; hard to review with frame loop | CONFIG |
| Eval CLI / env probe flags (`_detect_barrier_mode`, stream probes) | `eval/pipeline.py` / `evaluator.py` | DEBUG should be default-off and side-path | DEBUG |
| Per-frame stage graph vs report hooks | `eval/stages.py` | RUNTIME stages absorb PERF ticks & side effects | RUNTIME core; PERF collector separate |
| Result / MOT formatting & tables | `evaluator.py` → lean on `reporting.py` | Runner should not own presentation | RESEARCH/CORRECTNESS report |
| Offline Cheb-GR / bank adapters vs live relink | `eval/relink.py` | RESEARCH offline vs CORRECTNESS live wall (#57) | RESEARCH offline; live stays thin |
| Setter packing / tensor conversion / graph capture sections | `tracker_gpu_python.cpp` | BRIDGE file too multi-objective | BRIDGE submodules |
| Auction / cost / occ / bridge kernels packaging | `tracker_gpu.cu` | RUNTIME+CORRECTNESS+PERF in one TU | long-term; only with parity tests |

---

## Priority band B (clear boundary, lower urgency)

| Candidate | From | Why | After |
|:--|:--|:--|:--|
| Tile diagnostics / seam probes | `eval/detection.py` | DEBUG on legacy tile path | DEBUG |
| External FP model path | `eval/external_fp_model.py` | Already separate; keep out of default RUNTIME | RESEARCH/LEGACY |
| Concurrent eval shims | `eval/concurrent_*.py` | LEGACY / specialized | LEGACY |
| Lifecycle pure transitions | `eval/lifecycle.py` + evaluator slices | CORRECTNESS unit-testable core | CORRECTNESS |
| Preset policy “smart defaults” if any appear in facade | `tracking/tracker_gpu.py` | Facade must stay thin | CONFIG (scripts/eval/config) |

---

## Priority band C (docs / process — often already split)

| Candidate | From | Why | After |
|:--|:--|:--|:--|
| Decision semantics | was mixed with pipeline docs | Already `docs/research/tracker-decision/` | RESEARCH |
| Evidence numbers | scattered READMEs | `docs/research/evidence_ledger.md` | RESEARCH |
| WIP / objective isolation | process notes in status | This `docs/ownership/` tree | governance RESEARCH |

---

## Explicit non-candidates (do not “extract” casually)

| Item | Reason |
|:--|:--|
| Dual-stability cost vs bid merge | Closed decision (P7 keep both); not an O-series extract |
| Sync ReID onto critical path | NO-GO #57; not an extraction target for production |
| Schema knob mass-delete | Deferred; guardrails first |
| Industrial streaming into MOT eval | Keep **separated**, not merged |

---

## O3 entry criteria (for later)

A candidate may enter an extraction **plan** PR when:

1. Card exists in [module_objective_map.md](module_objective_map.md)  
2. Routing checks are known ([change_routing_matrix.md](change_routing_matrix.md))  
3. Parity / smoke owner is named  
4. O0 WIP=1: extraction does not open a second decision-changing charter; unrelated probes / evidence / close may remain non-WIP

O4 may move code only after those hold.
