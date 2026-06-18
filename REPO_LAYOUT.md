# Repository Layout

This file documents the top-level repository layout outside `docs/` and `src/`.
Those two directories are treated as stable documentation and source-code roots
and are intentionally out of scope for this cleanup pass.

Review mark format:

- `Reviewed 2026-06-18`: role and cleanup policy checked in this pass.
- `External`: managed outside this repo cleanup policy.
- `Ignored`: local artifact/runtime path covered by `.gitignore`.

## Stable Project Roots

| Path | Role | Notes | Review |
|---|---|---|---|
| `configs/` | Runtime and evaluation configuration | Presets live in `configs/presets/`; `mamba_whole_graph.yaml` is the current frozen main-line preset. | Reviewed 2026-06-18 |
| `include/` | C++ / CUDA public headers | Native tracking, media, perception, and package-facing headers. | Reviewed 2026-06-18 |
| `scripts/` | Reusable command-line tools | Evaluation, training, native build helpers, benchmarks, and model tooling. See `scripts/README.md` for the script cleanup ledger. | Reviewed 2026-06-18 |
| `tests/` | Test suites | Unit, integration, native, benchmark, golden, and experimental tests. | Reviewed 2026-06-18 |
| `third_party/` | Vendored external code | Currently includes TrackEval. Avoid mixing local experiment output here. | Reviewed 2026-06-18 |
| `docker/`, `Dockerfile`, `docker-compose.yml` | Container entry points | Keep deployment/build container changes here. | Reviewed 2026-06-18 |
| `infra/` | Local infrastructure config | PostgreSQL/systemd and service-level support files. | Reviewed 2026-06-18 |
| `.github/`, `.githooks` | Repository automation | CI and local git hook material. | Reviewed 2026-06-18 |
| `CMakeLists.txt` | Native build entry point | Keep at repo root. | Reviewed 2026-06-18 |
| `pyproject.toml`, `uv.lock`, `.python-version` | Python project metadata | Keep at repo root. | Reviewed 2026-06-18 |
| `main.py` | Legacy/manual app entry point | Keep until confirmed unused; do not create new root scripts beside it. | Reviewed 2026-06-18 |
| `build_fpn_reid.py` | Moved native helper | Moved to `scripts/native/build_fpn_reid.py`; root copy removed. | Reviewed 2026-06-18 |

## Data, Models, And Runtime State

| Path | Role | Git policy | Review |
|---|---|---|---|
| `datasets/` | Local datasets such as MOT17, DanceTrack, SportsMOT, CrowdHuman | Ignored. Large local data only. | Reviewed 2026-06-18 |
| `models/` | Model metadata and local model binaries | Metadata and placeholders may be tracked; large weights/engines are ignored. | Reviewed 2026-06-18 |
| `storage/` | Local service state | Runtime storage such as PostgreSQL and MLflow artifacts should stay ignored. | Reviewed 2026-06-18 |
| `.venv/` | Local virtual environment | Ignored. Recreate with `uv sync`. | Reviewed 2026-06-18 |
| `.env`, `.env.example` | Local/private env and template | `.env` ignored; `.env.example` tracked as template. | Reviewed 2026-06-18 |

## Generated Outputs

These directories are local experiment/runtime output. They are not source of
truth unless a result is explicitly promoted into a tracked report or decision.

| Path | Role | Cleanup rule | Review |
|---|---|---|---|
| `out/` | Short-lived evaluation outputs | Safe to delete after important results are summarized elsewhere. | Reviewed 2026-06-18 |
| `output/` | Historical evaluation/sweep outputs | Safe to prune after preserving useful metrics. | Reviewed 2026-06-18 |
| `results/` | Evaluation result directories | Large local artifact root; promote only summaries/tables. | Reviewed 2026-06-18 |
| `runs/` | Training and experiment runs | Large local artifact root; keep only active checkpoints locally. | Reviewed 2026-06-18 |
| `logs/` | Runtime logs | Safe to rotate/delete. | Reviewed 2026-06-18 |
| `reports/` | Generated report exports | Regenerable unless manually curated. | Reviewed 2026-06-18 |
| `build/`, `build-native-coverage/`, `dist/` | Build artifacts | Regenerable. Delete when stale. | Reviewed 2026-06-18 |
| `.cache/`, `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/` | Tool caches | Regenerable. | Reviewed 2026-06-18 |
| `*.sqlite`, `*.prof`, `*.nsys-rep`, root `*.so` | Profiling/native build artifacts | Ignored. Delete or regenerate locally. | Reviewed 2026-06-18 |

## Report Material

| Path | Role | Notes | Review |
|---|---|---|---|
| `report_data/` | Tracked paper/report source material | Contains curated tables, figures, and report notes. Generated JSON is ignored. | Reviewed 2026-06-18 |
| `README.md` | Public project entry point | Keep synchronized with accepted project direction and frozen headline numbers. | Reviewed 2026-06-18 |
| `DEVELOPMENT.md` | Developer entry point | Keep workflow and engineering notes here. | Reviewed 2026-06-18 |
| `REPO_LAYOUT.md` | Top-level repository map | Update when a top-level directory changes role. | Reviewed 2026-06-18 |
| `LICENSE` | License text | Keep at repo root. | Reviewed 2026-06-18 |

## Scratch And One-Off Work

| Path | Role | Rule | Review |
|---|---|---|---|
| `scratch/` | Exploratory scripts and validation snippets | Use for one-off experiments that are not ready for `scripts/`. | Reviewed 2026-06-18 |
| root-level `debug_*.py`, `patch_*.py`, `test_*.py`, `trt_bench.py` | Temporary local scripts | Ignored unless explicitly tracked. Promote reusable code into `scripts/` or tests into `tests/`; root copies were moved to `scratch/root_artifacts_20260618/`. | Reviewed 2026-06-18 |
| `1` | Removed tracked stray file | Contained a stale log/tail fragment, not a project entry point. | Reviewed 2026-06-18 |

## Explicitly Out Of Scope

| Path | Reason | Review |
|---|---|---|
| `docs/` | User requested no cleanup in this pass. | Out of scope 2026-06-18 |
| `src/` | User requested no cleanup in this pass. | Out of scope 2026-06-18 |
| `.git/` | Git internal state. | External |
| `.agents/`, `.codex/`, `.gemini/` | Agent/tool-local metadata. | External |

## Cleanup Order

When the workspace is noisy, clean in this order:

1. Delete stale build/cache artifacts: `build/`, `build-native-coverage/`,
   `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`, `.cache/`.
2. Prune old generated outputs: `out/`, `output/`, `results/`, `reports/`,
   `logs/`.
3. Prune inactive training runs under `runs/` after checkpoint provenance is
   captured in `report_data/` or the relevant research note.
4. Move reusable root-level debug scripts into `scripts/` or `tests/`; leave
   truly one-off probes in `scratch/`.

Do not move `datasets/` or `models/` as part of cleanup. They are intentionally
large local roots and often referenced by scripts.
