<!-- doc-status: closed -->
<!-- doc-promotion: none -->
<!-- doc-date: 2026-09-09 -->

# #340 2026-09-09 incidence preregistration — seal evidence

Companion to
[capture_race_incidence_preregistration_20260909.md](capture_race_incidence_preregistration_20260909.md).
This file is the mechanical gate record. It does not restate the research
question, does not add degrees of freedom, and is not a rate result.

Incidence campaign **has not started**. #340 remains open.

## Frozen identities

| Field | Value |
|---|---|
| production target source SHA | `4afb57c33cb0f9d7ddcc57d533e87a44e0c42d7f` |
| #379 in target | yes (`4afb57c3` = merge of #379; patch `b7bdb57378d04adfbf3310b5faa8aeca41ba6e1d` is ancestor) |
| observer/control source commit | `276d8d744d7050ad272f9b69cf3c60b0c31333a5` |
| analyzer source SHA256 | `6d72a6107fedc739370489b13f6699f08e06a9e2ea4655e6836e127b9fe1fb42` |
| observer.cpp SHA256 | `4a20104185394a565014f83fd2f6993e33aa4c95a6775ebe15e91017140c592d` |
| observer.so SHA256 | `e8b83ec4866c82f4e5a6ddbb64e5b207ff65dfc031db0101078f72083da30f29` |
| control_owner.cpp SHA256 | `a2e97433d16c9cb516f0d8c808d6afd595ad5f1e4f26533e174c714f9cb471c5` |
| control_owner.so SHA256 | `a7c9f01c3500ed1d08e6aed9b5c2bd29ffb8f1f6ff23eeb669f3e5eeb5288d0e` |
| decision_surface | `9b7faeb0f76a43483a924ac4028361bb155373027e9fe44e4cffbd5f1a2b0369` |
| environment | `df4c89b6aae2c555aaac108bb8fbaf835aa21f1057c03e25ea7e26d009d5b091` |
| identity_semantics | `8fc9bd85dd651791ec552dd2d6ab0244e849d213dcee7a2b77dfb9246a7969ca` |
| implementation | `2f69ac56e8cbfeb41300d479b2910ad6d750391b7099bb1af649956fc209d05d` |
| runtime_inputs | `0b839df0b89141959a4ae4762c727d446a2292016832ab468ce48334fff1a3d5` |
| identity probe | `2dabed0bc05e3bc75ec2115b3213f5c0b1aed3e837c22dd2325109339e4719b5` |
| N | 100 effective 7-sequence runs / path |
| 0/100 one-sided 95% exact binomial upper bound | `1 - 0.05^(1/100) = 0.029513 ≈ 2.95%` |

Coordinate values copied from
`docs/reference/runtime_identity.generated.json` at the production target SHA.
Observer hashes copied from
`~/.local/state/saccade/perf/capture-attribution-340-20260909-parentage/build/build.json`
and the freeze-commit blobs.

## Freeze commit vs smoke-tested worktree

`git diff 4afb57c3..276d8d74` is only the five observer/analyzer files. `src/`,
`include/`, `scripts/eval/`, and `configs/` are empty. Production capture /
runtime semantics were not changed.

Relative to the already-qualified dirty worktree
(`source.diff` SHA256 `fa8cbf776813e59a1ad8624e5ab5042c36d4bf1aa030d9a122eca9507ce875dc`):

- `observer.cpp` / `observer.so` / `run.py` / `qualify.py` / `control.py` /
  `control_owner.cpp` remain byte-identical.
- `analyze.py` and `test_capture_attribution_nested_driver.py` differ only by
  pre-commit `ruff format` line wrapping. `ast.dump` and `ast.unparse` are
  identical to the smoke-tested bytes.

Because the committed analyzer is not byte-identical, synthetic qualification
and both bounded topology smokes were re-run on commit `276d8d74` (still not
incidence). Output:
`~/.local/state/saccade/perf/capture-attribution-340-20260909-parentage/requalify-ruff-276d8d74/`.

## Tests that qualified the freeze

- `tests/unit/eval/test_capture_attribution_nested_driver.py`
- `tests/unit/eval/test_capture_attribution_harness.py` (29 passed together)
- `scripts/tools/capture_attribution/qualify.py` six cases, all `passed: true`
- bounded topology smoke, production GPU decode, `trace_structure_ok=true`
- bounded topology smoke, `--no-gpu-decode`, `trace_structure_ok=true`

## Pre-execution gate

| Item | Result |
|---|---|
| #379 in target | pass |
| target SHA frozen | pass |
| production coordinate frozen | pass |
| observer/control implementation frozen | pass |
| observer qualification passed | pass |
| production topology smoke `trace_structure_ok=true` | pass |
| DALI topology smoke `trace_structure_ok=true` | pass |
| `ownership_evidence_ok=true` | pass (both paths, original and requalify) |
| `evidence_gaps=[]` | pass |
| no unexplained participant class | pass; every owner class is `resolved:*` |
| assets frozen | pass (`runtime_inputs` above) |
| primary predicate frozen | pass (CUDA 900/901/906 + listed wrappers only) |
| denominator/validity rules frozen | pass |
| stopping rule frozen | pass |
| N frozen | pass |
| first rate run started | **no** |
| #340 closed | **no** |

Requalify smoke owner classes (commit `276d8d74`, same structure as the
2026-09-09 parentage readiness smokes):

- GPU decode: nvinfer `getBuilderSafePluginRegistry` (nested-driver 16 +
  driver-direct 1), c10 `initSingleStream` 128, `saccade::GMC::GMC` 2,
  nvjpeg `nvjpegDecodeJpegDevice` nested-driver 5.
- `--no-gpu-decode`: same nvinfer / c10 / GMC classes; DALI
  `libdali_core.so:dali::CUDAStream::Create` 2 instead of nvjpeg.

## What this seal authorizes

Harness implementation and a spec-equivalence check against the
preregistration. It does **not** authorize:

- starting rate run 1
- a "just look once" trial
- changing N
- closing #340
