# scripts/

Reusable command-line tools live here. This tree is intentionally separate from
`src/`: scripts may orchestrate experiments, training, export, profiling, or
maintenance, but reusable library logic should move into `src/saccade/...`.

## Review Status

First inventory pass: 2026-06-18. Count refresh: 2026-06-28.

| Area | Files | Current role | Review status |
|---|---:|---|---|
| `eval/` | 134 tracked files / 123 Python | MOT17 eval, ablations, diagnostics, one-off analysis | Classified; first unused archive/prototype removal pass done 2026-06-18; 28 duplicate Python basenames still need triage |
| `tools/` | 68 tracked files / 64 Python | Mixed maintenance checks, analysis probes, render/remap helpers | Classified 2026-06-18; unused tools removal passes in progress |
| `model/` | 21 | Model export/build/calibration helpers | Classified 2026-06-18; no files moved |
| `benchmarks/` | 17 | Latency, stress, and profiler scripts | Classified 2026-06-18; no files moved |
| `train/` | 5 + 39 in `temporal_yolo/` | Training entrypoints and research run scripts | Classified 2026-06-18; `temporal_yolo/` reviewed against v14 protocol |
| `ops/` | 5 | Local service and RTSP demo operations | Classified 2026-06-18; no files moved |
| `native/` | 4 | Native build/coverage helpers | Reviewed 2026-06-18 |
| root files | 10 | Stable hooks, one historical training runner, and dataset downloaders | Downloader role needs a separate manifest/runbook decision |

## Triage Labels

Use these labels in README tables and file headers while cleaning:

- `stable`: supported workflow entrypoint; keep path stable.
- `diagnostic`: reusable analysis/debug tool, but not a main workflow.
- `experiment`: tied to a named experiment or historical sweep.
- `archive-candidate`: no longer active; keep only if referenced by docs/results.
- `generated`: output or cache; should be ignored, not tracked.

## Stable Entrypoints

These paths are treated as stable unless a dedicated migration updates callers
and docs in the same change.

| Path | Purpose |
|---|---|
| `scripts/eval/mot17.py` | Main MOT17 evaluation entrypoint |
| `scripts/eval/mot17_all_sdp.py` | Per-sequence MOT17-SDP dispatch and merge helper |
| `scripts/eval/_perseq_extract.py` | Per-sequence metric extraction for completed runs |
| `scripts/eval/ablation_mot17.py` | Grouped MOT17 ablation runner |
| `scripts/eval/summarize_ablation_mot17.py` | Ablation result summarizer |
| `scripts/eval/detection_map.py` | Detector-only mAP evaluation |
| `scripts/eval/calculate_mota.py` | Metric recomputation from MOT output files |
| `scripts/eval/latency_report.py` | Post-run latency analysis |
| `scripts/eval/dancetrack.py` | DanceTrack evaluation |
| `scripts/eval/sportsmot.py` | SportsMOT evaluation |
| `scripts/eval/module_benchmark.sh` | Multi-step eval/profile/contribution wrapper |
| `scripts/pre_push.sh` | Local pre-push validation |
| `scripts/test_native.sh` | Stable native test entrypoint |
| `scripts/native/rebuild.sh` | Native extension rebuild helper |
| `scripts/native/coverage_native.sh` | Native coverage helper |

## Root Scripts

| Script | Role | Status |
|---|---|---|
| `README.md` | This script-tree cleanup ledger | stable |
| `__init__.py` | Package marker for scripts imports/static analysis | stable |
| `download_crowdhuman_hf.py` | CrowdHuman dataset downloader | diagnostic; root-surface cleanup candidate |
| `download_external_datasets.py` | CrowdHuman / CityPersons downloader | diagnostic; root-surface cleanup candidate |
| `download_kitti_tracking.py` | KITTI tracking downloader | diagnostic; root-surface cleanup candidate |
| `download_market1501.py` | Market1501 downloader | diagnostic; root-surface cleanup candidate |
| `download_motsynth.py` | MOTSynth downloader | diagnostic; root-surface cleanup candidate |
| `pre_push.sh` | Local pre-push CI mirror | stable |
| `test_native.sh` | Native C++/CUDA coverage test entrypoint | stable |
| `train_option_d.sh` | Historical Option-D two-phase training runner | archive-candidate; uses old `train/temporal_yolo/...` path |

## Dataset Downloaders

These root-level downloader scripts are currently tracked. They mix data
acquisition into the root script surface; keep the source references here so a
future cleanup can rebuild them as a manifest or runbook before moving/removing
the scripts.

| Script | Dataset/source references |
|---|---|
| `download_crowdhuman_hf.py` | Hugging Face dataset repo `Carles208AVL/CrowdHuman` |
| `download_external_datasets.py` | CrowdHuman: `https://data.vision.ee.ethz.ch/cvl/rro/CrowdHuman/CrowdHuman.tar`, `https://data.vision.ee.ethz.ch/cvl/rro/CrowdHuman/CrowdHuman.zip`; CityPersons: `https://data.vision.ee.ethz.ch/CD/684804/data/CityPersons.zip`, `https://data.vision.ee.ethz.ch/CD/684804/data/CityPersonsAnnotations.zip` |
| `download_kitti_tracking.py` | KITTI Tracking page: `https://www.cvlibs.net/datasets/kitti/eval_tracking.php`; login endpoint: `https://www.cvlibs.net/datasets/kitti/user_login_check.php`; components: `https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip`, `https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_3.zip`, `https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_velodyne.zip`, `https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_oxts.zip`, `https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_calib.zip`, `https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip` |
| `download_market1501.py` | Hugging Face dataset repo `aveocr/Market-1501-v15.09.15.zip`, file `Market-1501-v15.09.15.zip` |
| `download_motsynth.py` | `https://motchallenge.net/data/MOTSynth_1.zip`, `https://motchallenge.net/data/MOTSynth_2.zip`, `https://motchallenge.net/data/MOTSynth_3.zip`, `https://motchallenge.net/data/MOTSynth_mot_annotations.zip` |

## Cleanup Order

1. **Do not move stable entrypoints first.**
   Create or update references before any path migration.
2. **Archive experiment runners by family.**
   Good first targets are `scripts/train/temporal_yolo/run_*.sh` and
   `scripts/eval/experiments/`, because they usually encode one historical
   sweep and should keep only documented provenance.
3. **Split diagnostics from entrypoints.**
   `scripts/eval/diagnostics/`, `scripts/eval/detector/`, and
   `scripts/eval/appearance/` should stay documented as provenance-bearing
   diagnostics unless the related docs/results are removed.
4. **Consolidate maintenance checks.**
   Keep `scripts/tools/check_*.py` and service helpers easy to find; move
   research probes only after confirming no docs rely on exact paths.
5. **Delete only after provenance is captured.**
   If a script supports a documented NO-GO, report table, or decision record,
   preserve it or archive it with a note rather than deleting it.

## Directory Policy

| Directory | Policy |
|---|---|
| `eval/` | Main evaluation plus layered diagnostics, baselines, detector analysis, appearance/ReID work, and experiment runners. |
| `model/` | Model build/export/calibration only. Do not put training loops here. |
| `train/` | Training loops and reproducible run scripts. Historical one-off runs should be grouped by experiment family. |
| `benchmarks/` | Performance measurement only. Debug-only scripts need file-header scope notes. |
| `tools/` | Maintenance, conversion, rendering, and research probes that are not eval entrypoints. This is the broadest bucket and needs gradual splitting. |
| `ops/` | Local runtime/service/demo operations. |
| `native/` | C++/CUDA build and coverage helpers. |

## Next Review Targets

Recommended next passes:

1. `scripts/eval/experiments/`
   - first deletion pass is done; remaining files are doc-cited or need
     provenance confirmation before removal.
2. `scripts/train/temporal_yolo/run_*.sh`
   - reviewed in `scripts/train/temporal_yolo/README.md`; keep protocol runners
     stable and archive attribution runners only with doc updates.
3. `scripts/benchmarks/`
   - classified in `scripts/benchmarks/README.md`; only delete after deciding
     whether historical stress results still matter.
4. `scripts/model/`
   - classified in `scripts/model/README.md`; embedding/ReID legacy and
     Mamba export/build scripts should be moved or removed as families.
5. `scripts/tools/`
   - classified in `scripts/tools/README.md`; cleanup should happen by family,
     starting with local ops legacy or unreferenced manual probes.
6. `scripts/tools/test_*.py`
   - these are not pytest tests; rename/move or document as manual probes.
