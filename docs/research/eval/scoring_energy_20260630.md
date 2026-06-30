# Association Energy Scoring Notes - 2026-06-30

## Source Material

- User note: `/mnt/c/Users/Ray Lei/Downloads/ChordEdit 儲存庫分析.md`
- Original paper: ChordEdit: One-Step Low-Energy Transport for Image Editing, arXiv:2602.19083, https://arxiv.org/abs/2602.19083
- Follow-up: Rethinking One-Step Image Editing through ChordEdit, arXiv:2606.14042, https://arxiv.org/abs/2606.14042

## Transferable Ideas

ChordEdit is useful here as a scoring-design analogy, not as a direct vision-editing algorithm. The relevant ideas are:

- Replace brittle one-shot arithmetic with lower-energy control terms.
- Keep the core predictor mostly black-box and put experimental control logic in a thin outer layer.
- Split coarse/global decisions from fine/local refinement.
- Treat fixed hyperparameters as a baseline, then expose small candidate-selection knobs for sweeps.

For Saccade association scoring this maps to:

- Preserve the old additive association score as `baseline`.
- Add an opt-in `energy` mode that augments association cost with score and height-consistency penalties.
- Add a private-continuation `energy` selector that ranks candidate/prior pairs by low energy and rejects ambiguous pairs by margin.
- Record the scoring configuration beside eval outputs so result directories can be audited later.

## Implemented Scope

1. Native association scoring
   - New mode: `--association-scoring-mode {baseline,energy}`.
   - New weights: `--assoc-score-cost-w`, `--assoc-height-cost-w`.
   - Energy mode enables multiplicative cost form and passes score/height penalties into the native tracker.
   - Baseline leaves existing cost behavior unchanged.

2. Private continuation selection
   - New mode: `--private-selection-mode energy`.
   - New margin: `--private-energy-margin`.
   - Pair energy uses prior IoU, normalized center distance, detector score, height ratio, and sparse symmetric support.
   - Positive margin keeps only mutual-best candidate/prior pairs whose best-vs-second score gap is large enough.

3. Diagnostics
   - `--assoc-energy-diagnostics` writes `_association_scoring_profile.json` under the eval output root.
   - The file records mode, weights, sinkhorn/stability settings, private mode, and private margin.

## Suggested Experiment Matrix

Keep the preset unchanged and pass overrides from CLI:

```bash
python scripts/eval/mot17.py \
  --config configs/presets/mamba_whole_graph.yaml \
  --association-scoring-mode baseline
```

```bash
python scripts/eval/mot17.py \
  --config configs/presets/mamba_whole_graph.yaml \
  --association-scoring-mode energy \
  --assoc-score-cost-w 0.05 \
  --assoc-height-cost-w 0.05 \
  --assoc-energy-diagnostics
```

```bash
python scripts/eval/mot17.py \
  --config configs/presets/mamba_whole_graph.yaml \
  --private-continuation-enabled \
  --private-selection-mode energy \
  --private-energy-margin 0.03 \
  --association-scoring-mode energy \
  --assoc-score-cost-w 0.05 \
  --assoc-height-cost-w 0.05 \
  --assoc-energy-diagnostics
```

Sweep candidates:

- `assoc_score_cost_w`: `0.00`, `0.03`, `0.05`, `0.08`
- `assoc_height_cost_w`: `0.00`, `0.03`, `0.05`, `0.08`
- `private_energy_margin`: `0.00`, `0.02`, `0.04`, `0.06`

Primary readout:

- IDF1, HOTA AssA, IDs
- FN/FP drift, especially in crowded frames
- private continuation accepted count
- latency impact relative to baseline

## Process Log

- Read the ChordEdit analysis note and checked the arXiv pages for the original and follow-up papers.
- Implemented the new scoring knobs as opt-in flags instead of changing `mamba_whole_graph` defaults.
- Added native tracker API plumbing for optional score and height energy terms.
- Added Python private-continuation energy ranking with mutual-best margin rejection.
- Added config parsing and CLI support for all new flags.
- Added focused unit coverage for config parsing and private energy helper behavior.
- Ran a 3-sequence MOT17 SDP slice smoke test on 2026-06-30:
  `MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP`, first 300 frames each.
  Outputs are under `report_data/scoring_energy_20260630/`.
- Ran full-frame validation for the promising slice candidates:
  first on `MOT17-02/04/05-SDP`, then on all 7 MOT17 train SDP sequences.

## Slice 300 Results

These are partial-frame runs. MOTMetrics still compares against full sequence
ground truth, so absolute values are not valid as final MOT scores. The table is
only for relative screening across variants with the same frame cap.

| run | IDF1 | MOTA | IDs | FP | FN | Rcll | Prcn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_slice300 | 38.9% | 25.5% | 46 | 243 | 54136 | 25.9% | 98.7% |
| assoc_e003_h003_slice300 | 39.1% | 25.5% | 45 | 253 | 54149 | 25.9% | 98.7% |
| assoc_e005_h005_slice300 | 39.1% | 25.5% | 46 | 258 | 54152 | 25.9% | 98.7% |
| private_energy_m006_slice300 | 39.1% | 25.6% | 44 | 281 | 54030 | 26.0% | 98.5% |
| assoc_private_e003_h003_m003_slice300 | 39.2% | 25.6% | 45 | 240 | 54085 | 26.0% | 98.8% |
| assoc_private_e003_h003_m006_slice300 | 39.2% | 25.6% | 45 | 235 | 54091 | 26.0% | 98.8% |
| assoc_private_e005_h005_m003_slice300 | 39.2% | 25.6% | 45 | 257 | 54079 | 26.0% | 98.7% |

Initial read:

- Native association energy alone has weak positive IDF1/ID-switch signal, but
  the tested weights increased FP/FN slightly.
- Private energy selection carries most of the recall/ID-switch signal, but on
  its own it adds too many FP.
- `assoc_score_cost_w=0.03`, `assoc_height_cost_w=0.03`,
  `private_energy_margin=0.06` is the best slice candidate so far: +0.3 IDF1,
  +0.1 MOTA, -1 ID, -8 FP, and -45 FN relative to baseline slice.
- Private energy mode currently disables native private postprocess, reducing
  throughput from about 124 FPS baseline to about 100 FPS on this slice.

## Full-Frame Results

### MOT17-02/04/05-SDP

| run | IDF1 | MOTA | IDs | FP | FN | Rcll | Prcn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_full3 | 83.8% | 81.3% | 140 | 656 | 12897 | 82.3% | 98.9% |
| assoc_score000_h003_full3 | 83.8% | 81.3% | 140 | 658 | 12885 | 82.4% | 98.9% |
| assoc_score003_h000_full3 | 83.8% | 81.3% | 141 | 639 | 12909 | 82.3% | 98.9% |
| assoc_e003_h003_full3 | 83.6% | 81.1% | 144 | 677 | 12981 | 82.2% | 98.9% |
| assoc_private_e003_h003_m006_full3 | 83.2% | 81.1% | 144 | 796 | 12903 | 82.3% | 98.7% |

### All 7 MOT17 Train SDP Sequences

| run | IDF1 | MOTA | IDs | FP | FN | Rcll | Prcn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_full7_sdp | 78.4% | 78.0% | 425 | 3424 | 20808 | 81.5% | 96.4% |
| assoc_score002_h000_full7_sdp | 78.2% | 78.0% | 430 | 3446 | 20792 | 81.5% | 96.4% |
| assoc_score003_h000_full7_sdp | 78.4% | 78.4% | 428 | 3076 | 20731 | 81.5% | 96.7% |
| assoc_score004_h000_full7_sdp | 78.3% | 78.3% | 428 | 3138 | 20837 | 81.4% | 96.7% |

Full-frame read:

- The slice winner `assoc_private_e003_h003_m006` does not hold up on full frames
  (`full3` IDF1 -0.6, FP +140, IDs +4). Treat private energy selection as
  no-go until it has native telemetry or a different ambiguity rule.
- Height-only `assoc_height_cost_w=0.03` is nearly neutral on full3: it recovers
  a few FN but does not move aggregate IDF1/MOTA.
- Score-only `assoc_score_cost_w=0.03` is the only full7 candidate with a useful
  aggregate trade: IDF1 flat, MOTA +0.4, FP -348, FN -77, precision +0.3, but
  IDs +3. This is a MOTA/FP cleanup candidate, not an IDF1 improvement.
- `assoc_score_cost_w=0.02` under-shoots and `0.04` starts losing IDF1/recall,
  so `0.03` is the current narrow operating point.

## Open Follow-Ups

- If accepting a MOTA/FP tradeoff with flat IDF1, test
  `association_scoring_mode=energy`, `assoc_score_cost_w=0.03`,
  `assoc_height_cost_w=0.0` on another split/dataset before changing a preset.
- Add per-frame private energy telemetry only if the aggregate result changes enough to justify deeper attribution.
- Consider a learned or scene-adaptive margin after the static sweep identifies a stable operating region.

## HOTA Pipeline Fix

During the `mamba_whole_graph_m` rerun, the terminal output only contained
IDF1/MOTA/CLEAR counts. Root cause: the vendored TrackEval copy was missing
`trackeval/datasets/`, and the root `.gitignore` rule `datasets/` ignored that
package path. `run_motmetrics_evaluation()` still called `_calculate_hota()`,
but TrackEval import failed and returned `None`, silently dropping HOTA/DetA/AssA.

Fix:

- Restored `third_party/TrackEval/trackeval/datasets/` from official TrackEval
  commit `12c8791b303e0a0b50f753af204249e622d0281a`.
- Added `.gitignore` exceptions for the vendored package path.
- Changed `_calculate_hota()` to warn on TrackEval import/evaluation/parsing
  failure instead of silently omitting HOTA.
- Added a unit test that imports `trackeval.datasets.MotChallenge2DBox`.

Verification:

```text
IDF1: 79.5%
MOTA: 81.3%
HOTA: 73.3%
DetA: 74.9%
AssA: 71.9%
IDs: 335
FP: 2315
FN: 18374
Rcll: 83.6%
Prcn: 97.6%
```

## mamba_whole_graph_m Full7 Results

These use `--preset mamba_whole_graph_m`, not the earlier
`configs/presets/mamba_whole_graph.yaml` run.

| run | IDF1 | MOTA | HOTA | DetA | AssA | IDs | FP | FN | Rcll | Prcn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| m_preset_baseline_full7_sdp | 79.5% | 81.3% | 73.3% | 74.9% | 71.9% | 335 | 2315 | 18374 | 83.6% | 97.6% |
| m_preset_assoc_score002_h000_full7_sdp | 80.1% | 81.4% | 73.7% | 74.9% | 72.7% | 345 | 2066 | 18468 | 83.6% | 97.8% |
| m_preset_assoc_score003_h000_full7_sdp | 80.0% | 81.1% | 73.5% | 74.7% | 72.5% | 342 | 2282 | 18550 | 83.5% | 97.6% |

Current m-preset read:

- `assoc_score_cost_w=0.02`, `assoc_height_cost_w=0.0` is the strongest tested
  m-preset candidate so far: IDF1 +0.6, HOTA +0.4, AssA +0.8, MOTA +0.1, FP
  -249, with IDs +10 and FN +94.
- `assoc_score_cost_w=0.03` still improves IDF1/HOTA/AssA but loses MOTA and FN
  relative to 0.02. Do not carry over the s-preset `0.03` conclusion to m.
