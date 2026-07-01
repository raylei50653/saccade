# PP22 as a training-pipeline stress test — findings & handoff (2026-07-01)

> **Purpose of the PP22 detour**: not to optimize PP22, but to (a) test how well the
> current training pipeline adapts to *complex scenes*, and (b) surface latent
> problems. PP22 (PersonPath22) is small-object-heavy, occlusion-heavy, and
> scene-heterogeneous — a deliberate hard case. Main line remains **MOT17**
> (headline IDF1 78.6 s / 79.5 m). This doc captures what the stress test exposed
> and what carries back to the MOT17 line.

## Verdict: pipeline adapts mechanically; the real value is the latent problems it exposed

The GT1→T3→T1 curriculum trains, converges, and is stable on the complex domain
(after one bug fix). It was also extended to handle complex scenes: GPU-decode
training (1080p full-cadence, data I/O fully overlaps) + keyframe-aware
train/eval. **Mechanical fit is fine.** The findings below are the payload.

## The 5 latent problems

| # | Problem | Type | Blast radius |
|---|---|---|---|
| 1 | **GO/selection judged at @0.001 instead of the deployable @0.25** — the whole head-architecture investigation chased a 30pp @0.001 gap that is only **+4.5pp at @0.25** (the tracker's `new_track_thresh` ~0.28). The 30pp is low-confidence detections no tracker uses. | **process / methodology** | **any domain incl. MOT17** |
| 2 | `clip_len` > keyframe spacing → a whole batch can be all-interpolated (no keyframe) → `batch_loss` has no grad_fn → `backward()` crashes | **real bug (FIXED, `5ab711cc`)** | any keyframe-annotated dataset |
| 3 | PP22→MOT converter writes `conf/cls/vis=1` → eval's `vis<0.1` GT filter is a no-op → FN denominator inflated with near-invisible people (occluded/background/tiny) → recall cratered at *every* size bin (even ≥16px: 60% PP22 vs 93% MOT17) | **eval fidelity** | any new dataset ingest |
| 4 | Temporal (T3) training does **not transfer** to the T=1 non-temporal deploy path; cadence/interp only shapes T3 | **capability boundary** | any temporal-training idea |
| 5 | Small-object recall is **resolution-bound**, occlusion is **association-bound**; head knobs (DFL reg_max, sr=2, head-depth) can't move deployable recall (+0.5 / +2.5 / ~0 pp) | **capability ceiling** | detection-head architecture choices |

## Supporting evidence (all measured this session)

- **Head is not the deploy bottleneck**: same gated FPN features — teacher YOLO head @0.001=75.2 vs mamba head 45.4 (−30pp), but **@0.25 teacher 39.4 vs mamba 34.9 (only −4.5pp)**. `raw_yolo_recall.py` / `teacher_recall.py` (reuse `mamba_size_binned_recall` helpers).
- **Head knobs NO-GO** (single-variable GT1 vs baseline GT1, PP22 held-out `mot_test_kf` 8 seqs, @0.001): DFL reg_max=16 → flat (45.8); sr=2 → 47.7 (+2.5, weak, 3× head latency); head-depth 2 (YOLO-aligned 2 conv blocks + BN) → 45.1 (flat).
- **Stage probe**: gap is present from GT1 (45.2), not introduced by temporal (T3 47.3, T1 45.4) — temporal curriculum is innocent.
- **Full-cadence + interp NO-GO** on both MOT17 transfer and PP22 held-out self-domain (see `pp22_full_cadence_interp_training_plan.md` §5–6).
- **Model does better on transfer (MOT17 75.8) than its own domain (PP22 34.9)** — PP22 is a genuinely hard recall target (occlusion + inclusive annotation + heterogeneity; cf. PASTA, arXiv 2411.00553).

## What carries back to the MOT17 main line

- **#1 is the big one**: MOT17 selection/GO must use **deployable thresholds** (@0.25 / tracker `new_track_thresh`), never @0.001. Nearly cost half a day of misdirected head work here.
- **#4 / #5**: do **not** spend MOT17 effort on temporal-training or detection-head architecture to chase recall — same walls. Toward 80, go **association recovery / calibration** (consistent with the main-line framing).
- **#2 fixed**; **#3** is PP22-specific (MOT17 vis filter works).

## Infra kept (all default-off / opt-in, bit-exact when unused)

- GPU-decode training: `train_mamba_gt.py --gpu-decode` + `dataset.py gpu_decode_clip_batch` (workers return raw JPEG bytes, nvJPEG batch-decode in-loop).
- `--interpolate-gt` keyframe-loss-mask + no-keyframe backward guard.
- `--reg-max` (DFL), `--spatial-reduction` (sr), `--head-depth` (+ YOLO bias_init) — all configurable, reinit + warm-start filtering + inference DFL decode wired.
- `eval_gated_bytetrack.py --score-on-gt-frames` (keyframe-aware original-model tracking).
- Drivers: `run_pp22_full_cadence_chain.sh`, `run_pp22_dfl16_chain.sh`, `run_pp22_sr2_chain.sh`. Ckpts: `runs/mamba_gt_pp22_{full,dfl16,sr2,hd2}_*`.

## Related memories

`project_mamba_head_compresses_recall_go` (head gap is @0.001 not @0.25; three knobs NO-GO),
`project_pp22_full_cadence_interp_nogo`, `project_recall_lever_separability` (recall is occlusion-bound),
`project_pp22_conversion_label_policy` (vis=1 artifact), `project_main_line_framing`.

**Status: PP22 stress test CLOSED.** Return to MOT17 main line.
