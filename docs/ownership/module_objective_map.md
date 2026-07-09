# Module objective map (O1)

**Scope:** annotate only — no moves, no runtime/default changes.  
**Types:** [objective_template.md](objective_template.md)  
**Routing:** [change_routing_matrix.md](change_routing_matrix.md)  
**Extract list:** [extraction_candidates.md](extraction_candidates.md)

Paths are repo-root relative. LOC counts are approximate (2026-07 snapshot) and
are **hints**, not sole ownership criteria.

---

## 1. Modules (docs/modules + primary code)

### detection

```text
Primary:     RUNTIME (detector / Mamba head path feeding tracker)
Secondary:   RESEARCH (training protocols, score-dist notes)
Should-not-own: tracker association policy; ReID critical path; PERF attribution for whole eval
Risk:        training + eval path churn pulls same owner as production detect defaults
Extraction:  training protocol docs stay under modules/detection/research; keep preset defaults out of training scripts
Checks:      preset smoke if detect path or whole-graph flags change
```

### geometry

```text
Primary:     CORRECTNESS (association geometry / GMC / Kalman prior for matching)
Secondary:   CONFIG (geometry knobs in presets)
Should-not-own: RESEARCH paper narrative; PERF stage attribution; ReID appearance policy
Risk:        decision knobs live across geometry + lifecycle + pipeline inject
Extraction:  keep decision semantics in docs/research/tracker-decision (already split)
Checks:      contract checker if headline YAML / inject map; MOT17-04 or 7-seq if match path
```

### semantic

```text
Primary:     RESEARCH / offline identity (Cheb-GR handover, occ-exit audit substrate)
Secondary:   DEBUG (handover logs, probes)
Should-not-own: RUNTIME critical-path ReID (#57 NO-GO); dual-stability defaults; preset birth policy
Risk:        offline GO signals get “promoted” into live tracker without promotion bar
Extraction:  offline report tools vs live claim paths already separated; keep that wall
Checks:      unit tests for bank/handover tools; smoke only if evaluator flags change
Active (O0): sole active = occ-exit audit (#55) — see modules/semantic/TODO.md
```

### reid

```text
Primary:     RESEARCH (appearance quality ceiling; currently ⏸️)
Secondary:   —
Should-not-own: RUNTIME sync ReID; production preset enablement without ≥ bar
Risk:        feature work re-enters critical path under “small flag”
Extraction:  keep async sidecar design separate from GPUByteTracker hot loop
Checks:      none while ⏸️; if revived: paired A/B + FPS budget
```

### lifecycle

```text
Primary:     CORRECTNESS (tentative / confirmed / lost / birth gates)
Secondary:   CONFIG (lifecycle schema)
Should-not-own: PERF attribution; appearance bank policy; detector training
Risk:        confirm gates interact with private continuation & bridge
Extraction:  lifecycle unit slice tests (already backlog)
Checks:      lifecycle-focused unit tests; smoke if confirm/birth thresholds change
```

### motion

```text
Primary:     CORRECTNESS (motion model assumptions for matching)
Secondary:   —
Should-not-own: GMC implementation ownership (geometry/native); CONFIG surface sprawl
Risk:        motion docs drift from Kalman/GMC kernels
Extraction:  docs already lean on tracker-decision/kalman_gmc_motion.md
Checks:      smoke if R-scale / GMC mode defaults change
```

### streaming / storage / cognition / resource / trigger

```text
Primary:     RUNTIME (industrial path) or LEGACY (converged MOT-eval secondary)
Secondary:   CONFIG for RTSP/contracts where applicable
Should-not-own: MOT17 association decision contract; dual-stability
Risk:        industrial features bleed into MOT eval defaults
Extraction:  keep MOT eval entry (scripts/eval/mot17.py) free of industrial side effects
Checks:      module-specific runbooks if those services change
```

---

## 2. Hot files (eval + tracker)

### `src/saccade/perception/eval/pipeline.py` (~1.6k LOC)

```text
Primary:     RUNTIME
Secondary:   CONFIG (inject into native / tracker setters; double-buffer eligibility)
Should-not-own: RESEARCH report formatting; PERF attribution narrative; experiment taxonomy
Risk:        runtime mode resolver + config injection + barrier probes co-located
Extraction:  runtime mode resolver; config injection helper; stream/barrier probes → DEBUG-only module
Checks:      check_headline_decision_contract.py if inject map; smoke if runtime branch touched
```

### `src/saccade/perception/eval/evaluator.py` (~3.2k LOC)

```text
Primary:     RUNTIME (eval runner / frame loop orchestration)
Secondary:   CORRECTNESS (result lifecycle, MOT line emit)
Should-not-own: tracker decision semantics; PERF attribution; experiment taxonomy; CLI schema ownership
Risk:        largest Python glue; absorbs relink/audit/report branches
Extraction:  output/report formatting (partially reporting.py); profiling hooks; CLI parsing remains scripts/eval
Checks:      eval smoke; output format / unit slices for lifecycle helpers
```

### `src/saccade/perception/eval/stages.py` (~3.3k LOC)

```text
Primary:     RUNTIME (per-frame stage graph)
Secondary:   PERF (stage boundaries used by profiling)
Should-not-own: CONFIG schema definitions; RESEARCH ablation matrices
Risk:        stage bloat mixes detect/track/relink side effects
Extraction:  stage-local helpers; profile tick collection
Checks:      smoke; profile-stages smoke if PERF-sensitive
```

### `src/saccade/perception/eval/config.py` (~1.8k LOC)

```text
Primary:     CONFIG
Secondary:   LEGACY (compat aliases)
Should-not-own: RUNTIME branch logic; RESEARCH evidence tables
Risk:        mega-schema becomes default-policy owner
Extraction:  domain splits already under scripts/eval/config/* — keep python EvalConfig thin
Checks:      config consistency tests; contract checker for headline fields
```

### `src/saccade/perception/eval/relink.py` (~2.5k LOC)

```text
Primary:     CORRECTNESS (identity recovery / bridge-adjacent Python policy)
Secondary:   RESEARCH (Cheb-GR offline / semantic paths)
Should-not-own: detector training; native auction implementation
Risk:        offline and live relink share file
Extraction:  offline handover / bank adapters (partially clean_fifo_bank.py, cheb_gr_*)
Checks:      unit tests for relink helpers; smoke if production bridge flags change
```

### `src/saccade/perception/eval/detection.py` (~1.5k LOC)

```text
Primary:     RUNTIME (detect path, tiling merge legacy)
Secondary:   CORRECTNESS (box quality into tracker)
Should-not-own: tracker match cost policy; RESEARCH score-dist papers
Risk:        tiled legacy + whole-graph production coexist
Extraction:  tile diagnostics → DEBUG; keep native_640 whole-graph as primary path docs
Checks:      smoke on detect path; tile tests only if tile path touched
```

### `src/saccade/perception/eval/reporting.py` (~0.8k LOC)

```text
Primary:     RESEARCH / report formatting
Secondary:   CORRECTNESS (metric table presentation)
Should-not-own: RUNTIME control flow; CONFIG injection
Risk:        low if kept side-effect free
Extraction:  already a good isolation target for evaluator dump logic
Checks:      unit / snapshot if table schema changes
```

### `src/saccade/perception/tracking/tracker_gpu.py` (~1.6k LOC)

```text
Primary:     BRIDGE / facade (Python GPUByteTracker API)
Secondary:   CONFIG surface (safe setters)
Should-not-own: preset policy / experiment defaults; RESEARCH ablation design
Risk:        facade grows policy (“smart defaults”) instead of thin setters
Extraction:  preset policy stays in scripts/eval/config + pipeline inject
Checks:      tracker smoke; setter contract / unit if API changes
```

### `src/tracking/tracker_gpu_python.cpp` (~5.7k LOC)

```text
Primary:     BRIDGE (pybind / tensor ownership / packing)
Secondary:   DEBUG (optional dumps if any)
Should-not-own: algorithm policy explanation; preset taxonomy; paper evidence
Risk:        largest bridge file; mixes packing, graph capture, many setters
Extraction:  setter packing section; tensor conversion section; graph capture section
Checks:      native build; pybind/unit; smoke if binding surface changes
```

### `src/tracking/tracker_gpu.cu` (~5.0k LOC)

```text
Primary:     RUNTIME + CORRECTNESS (association kernels, auction, OAO, occ, Kalman update)
Secondary:   PERF (kernel fusion / graph eligibility)
Should-not-own: YAML schema; RESEARCH report text; CLI help tags
Risk:        single CU owns cost, bid, lifecycle side effects
Extraction:  long-term: cost kernel vs auction vs bridge relink vs occ_state (O3+ only)
Checks:      native build; smoke; MOT17-04 or 7-seq if association behavior changes
```

### `src/tracking/pipeline.cpp` (~1.6k LOC)

```text
Primary:     RUNTIME (native eval/pipeline facade)
Secondary:   BRIDGE (C++ orchestration)
Should-not-own: Python research probes; paper metrics ledger
Risk:        parallel ownership with Python EvalPipeline
Extraction:  clarify single orchestrator owner (Python vs C++) in O2 notes
Checks:      cpp_runner / native smoke
```

---

## 3. CLI / config entrypoints

### `scripts/eval/mot17.py` (~0.5k LOC)

```text
Primary:     CONFIG / CLI entry
Secondary:   RUNTIME (wires runner)
Should-not-own: association math; long research narratives
Risk:        flag sprawl re-exports every experiment
Extraction:  keep module YAML + presets; avoid new flags without owner module
Checks:      argparse help tags; contract if headline-related flags
```

### `scripts/eval/config/*.py` (domain schemas)

```text
Primary:     CONFIG
Secondary:   LEGACY defaults
Should-not-own: RUNTIME implementation; RESEARCH evidence files
Risk:        lifecycle.py especially large — multi-domain knobs
Extraction:  per-domain files already split; resist re-merge into config.py
Checks:      contract checker; golden config snapshots if present
```

### `scripts/tools/check_headline_decision_contract.py`

```text
Primary:     CONFIG / CORRECTNESS (guardrail)
Secondary:   —
Should-not-own: PERF; RESEARCH prose
Risk:        checker drift from actual inject map
Extraction:  —
Checks:      self (CI contracts job)
```

---

## 4. Docs surfaces

### `docs/research/tracker-decision/*`

```text
Primary:     RESEARCH (decision semantics + contract narrative)
Secondary:   CONFIG documentation (ACTIVE surface)
Should-not-own: runtime code ownership; O-series governance (that's docs/ownership)
Risk:        reopening dual-stability under “docs fix”
Extraction:  —
Checks:      doc stale paths; no metric invention without evidence_ledger
```

### `docs/ownership/*` (this tree)

```text
Primary:     RESEARCH / governance (process)
Secondary:   —
Should-not-own: tracker behavior defaults
Risk:        none if docs-only
Extraction:  —
Checks:      link/stale path checkers
```

---

## 5. How to update this map

1. New hot file or new primary responsibility → add a card **in the same PR** that introduces the coupling.  
2. Do not expand Primary to two types; instead add Extraction + O0 park if needed.  
3. Behavior extractions wait for **O3/O4** with routing-matrix checks.
