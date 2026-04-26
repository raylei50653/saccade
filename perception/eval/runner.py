# mypy: ignore-errors
import configparser
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms, nms

# MUST IMPORT THIS BEFORE torchvision (already guaranteed by import order above being after perception)
from perception.cropper import ZeroCopyCropper
from perception.detector_trt import TRTYoloDetector
from perception.feature_extractor import TRTFeatureExtractor

from perception.eval.detection import _box_iou_single, detect_adaptive_960_tiled, merge_cross_tile_duplicates_fast
from perception.eval.pool import AdaptiveFramePool
from perception.eval.preprocess import GeometryScaleState, apply_frame_preprocess, geometry_mid_thresh_scale, parse_preprocess
from perception.eval.relink import SemanticRelinker
from perception.eval.streaming import DALIStreamerStream
from perception.eval.tracking import GlobalTrackIdMapper


def run_eval(engine, output, data_root, split, sequences, max_frames, conf_threshold, reid_mode="off", reid_model="siglip2", **kwargs):
    output_root = Path(output)
    output_root.mkdir(parents=True, exist_ok=True)
    fps_summary_lines = []
    overall_latency_ms = []
    profile_stages = bool(kwargs.get("profile_stages", False))
    stage_summary_lines = []
    global_id_mapper = GlobalTrackIdMapper()

    detector = TRTYoloDetector(engine_path=engine)

    if reid_mode not in {"off", "tracker", "semantic", "hybrid"}:
        raise ValueError(f"Unsupported reid_mode: {reid_mode}")
    reid_enabled = reid_mode != "off"
    profile_lazy_reid_embeddings = bool(kwargs.get("profile_lazy_reid_embeddings", False))
    profile_lazy_reid_candidates = bool(kwargs.get("profile_lazy_reid_candidates", False)) or profile_lazy_reid_embeddings
    reid_work_enabled = reid_enabled or profile_lazy_reid_embeddings
    _reid_engine = kwargs.pop("reid_engine_path", "") or ""
    _crop_hw: tuple[int, int] = (256, 128) if reid_model == "transreid" else (224, 224)
    extractor = TRTFeatureExtractor(
        engine_path=_reid_engine,
        model_type=reid_model,
    ) if reid_work_enabled else None
    cropper = ZeroCopyCropper(
        output_size=_crop_hw,
        mode=kwargs.get("reid_crop_mode", "tight"),
        padding=float(kwargs.get("reid_crop_padding", 0.0)),
    ) if reid_work_enabled else None

    reid_interval = max(1, int(kwargs.get("reid_interval", 4)))
    reid_crop_layout = kwargs.get("reid_crop_layout", "full")
    if reid_crop_layout not in {"full", "parts"}:
        raise ValueError(f"Unsupported reid_crop_layout: {reid_crop_layout}")

    use_semantic_mode = reid_mode in {"semantic", "hybrid"}
    use_tracker_reid = reid_mode in {"tracker", "hybrid"}
    person_class = int(kwargs.get("person_class", 0))
    track_person_only = bool(kwargs.get("track_person_only", True))
    track_thresh = float(kwargs.get("track_thresh", 0.1))
    high_thresh = float(kwargs.get("high_thresh", 0.5))
    match_thresh = float(kwargs.get("match_thresh", 0.8))
    cross_tile_merge = bool(kwargs.get("cross_tile_merge", False))
    geometry_mid_scale = bool(kwargs.get("geometry_mid_scale", False))
    geometry_ref_height_ratio = float(kwargs.get("geometry_ref_height_ratio", 0.12))
    geometry_min_scale = float(kwargs.get("geometry_min_scale", 0.875))
    geometry_max_scale = float(kwargs.get("geometry_max_scale", 1.20))
    geometry_ema_beta = float(kwargs.get("geometry_ema_beta", 0.80))
    geometry_loosen_step = float(kwargs.get("geometry_loosen_step", 0.08))
    geometry_tighten_step = float(kwargs.get("geometry_tighten_step", 0.03))
    geometry_min_samples = int(kwargs.get("geometry_min_samples", 5))
    lazy_reid_min_hit_streak = int(kwargs.get("lazy_reid_min_hit_streak", 2))
    lazy_reid_self_threshold = float(kwargs.get("lazy_reid_self_threshold", 0.85))
    preprocess_modes = parse_preprocess(kwargs.get("preprocess", "letterbox"))
    gamma = float(kwargs.get("gamma", 0.9))
    gamma_luma_threshold = float(kwargs.get("gamma_luma_threshold", 0.35))
    contrast = float(kwargs.get("contrast", 1.08))
    seqs = sequences.split(",") if sequences else [
        d.name for d in (Path(data_root) / split).iterdir() if d.is_dir()
    ]

    def time_stage(stage_totals, stage_name, fn, sync_cuda=False):
        if profile_stages and sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        if profile_stages and sync_cuda:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if profile_stages:
            stage_totals[stage_name] += elapsed_ms
        return result, elapsed_ms

    overall_stage_totals = OrderedDict(
        (name, 0.0)
        for name in (
            "fetch",
            "ingest_preprocess",
            "detect",
            "postprocess",
            "post_filter",
            "post_nms",
            "post_merge",
            "reid",
            "lazy_reid",
            "track",
            "relink_write",
            "frame_total",
        )
    )
    overall_profiled_frames = 0
    overall_post_counts = OrderedDict(
        (name, 0)
        for name in (
            "raw_boxes",
            "after_filter",
            "after_nms",
            "after_merge",
        )
    )
    overall_lazy_reid_candidates = 0
    overall_lazy_reid_frames = 0
    overall_lazy_reid_crops = 0
    overall_lazy_reid_self_pairs = 0
    overall_lazy_reid_self_pass = 0
    overall_lazy_reid_self_sim_sum = 0.0
    overall_lazy_reid_arbiter_checks = 0
    overall_lazy_reid_arbiter_approve = 0

    for seq in seqs:
        detector.reset_tracker()
        geometry_scale_state = GeometryScaleState()
        detector.tracker.set_params(
            track_thresh=track_thresh,
            high_thresh=high_thresh,
            match_thresh=match_thresh,
            track_buffer=30,
            mid_thresh=float(kwargs.get("mid_thresh", 0.40)),
            confirm_streak=int(kwargs.get("confirm_streak", 3)),
            confirm_score_thresh=float(kwargs.get("confirm_score_thresh", 0.50)),
            adaptive_confirmation=bool(kwargs.get("adaptive_confirmation", False)),
        )
        detector.tracker.set_reid_params(
            cos_threshold=float(kwargs.get("reid_cos_threshold", 0.90)),
            iou_low=float(kwargs.get("reid_iou_low", 0.30)),
            iou_high=float(kwargs.get("reid_iou_high", 0.60)),
            weight=float(kwargs.get("reid_weight", 0.40)),
        )

        relinker = SemanticRelinker(
            sim_threshold=kwargs.get("semantic_threshold", 0.95),
            ttl=kwargs.get("semantic_ttl", 45),
            ema_beta=kwargs.get("semantic_ema", 0.83),
            spatial_gate=kwargs.get("semantic_spatial_gate", 0.20),
            min_lost_frames=kwargs.get("semantic_min_lost_frames", 2),
            min_iou=kwargs.get("semantic_min_iou", 0.20),
            mahalanobis_threshold=kwargs.get("semantic_mahalanobis_threshold", 0.0),
            debug=kwargs.get("semantic_debug", False),
        ) if use_semantic_mode else None

        seq_path = Path(data_root) / split / seq
        if not (seq_path / "seqinfo.ini").exists():
            continue
        config = configparser.ConfigParser()
        config.read(seq_path / "seqinfo.ini")
        w_orig = config.getint("Sequence", "imWidth")
        h_orig = config.getint("Sequence", "imHeight")
        frame_end = min(max_frames or int(1e9), config.getint("Sequence", "seqLength"))

        pool = AdaptiveFramePool(h_orig, w_orig)
        streamer = DALIStreamerStream(seq_path / "img1")
        stream_iter = iter(streamer)
        results_lines, frame_latencies = [], []
        start_time = time.time()
        warmup_frames = int(kwargs.get("warmup_frames", 50))
        seq_stage_totals = OrderedDict((name, 0.0) for name in overall_stage_totals.keys())
        seq_post_counts = OrderedDict(
            (name, 0)
            for name in ("raw_boxes", "after_filter", "after_nms", "after_merge")
        )
        seq_lazy_reid_candidates = 0
        seq_lazy_reid_frames = 0
        seq_lazy_reid_crops = 0
        seq_lazy_reid_self_pairs = 0
        seq_lazy_reid_self_pass = 0
        seq_lazy_reid_self_sim_sum = 0.0
        seq_lazy_reid_arbiter_checks = 0
        seq_lazy_reid_arbiter_approve = 0
        lazy_reid_prev_embeddings: dict[int, torch.Tensor] = {}
        seq_profiled_frames = 0

        for frame_id in range(1, frame_end + 1):
            t_e2e_start = time.perf_counter()
            try:
                frame_gpu, _fetch_ms = time_stage(
                    seq_stage_totals, "fetch", lambda: next(stream_iter), sync_cuda=False
                )
            except StopIteration:
                break
            t_frame_start = time.perf_counter()

            _, _ = time_stage(
                seq_stage_totals,
                "ingest_preprocess",
                lambda: (
                    pool.frame_buffer.copy_(frame_gpu.permute(2, 0, 1).float() / 255.0),
                    apply_frame_preprocess(pool.frame_buffer, preprocess_modes, gamma, gamma_luma_threshold, contrast),
                ),
                sync_cuda=True,
            )

            (fused_boxes, fused_scores, fused_classes, is_tiled), _ = time_stage(
                seq_stage_totals,
                "detect",
                lambda: detect_adaptive_960_tiled(detector, pool, h_orig, w_orig, preprocess_modes),
                sync_cuda=True,
            )

            if fused_boxes.numel() == 0:
                if frame_id > warmup_frames:
                    frame_latencies.append((time.perf_counter() - t_frame_start) * 1000)
                if profile_stages and frame_id > warmup_frames:
                    seq_stage_totals["frame_total"] += (time.perf_counter() - t_e2e_start) * 1000
                    seq_profiled_frames += 1
                if frame_id % 100 == 0:
                    print(f"🎬 {seq} [{frame_id}/{frame_end}]")
                continue

            # Keep low-score boxes down to track_thresh so ByteTrack's
            # second-stage association can actually use them.
            if profile_stages:
                torch.cuda.synchronize()
                t_post_start = time.perf_counter()
            raw_box_count = int(fused_scores.numel())
            t_sub_start = time.perf_counter()
            keep_mask = fused_scores > min(conf_threshold, track_thresh)
            if track_person_only:
                keep_mask = keep_mask & (fused_classes == person_class)
            fused_boxes = fused_boxes[keep_mask]
            fused_scores = fused_scores[keep_mask]
            fused_classes = fused_classes[keep_mask]
            if profile_stages:
                torch.cuda.synchronize()
                seq_stage_totals["post_filter"] += (time.perf_counter() - t_sub_start) * 1000
            after_filter_count = int(fused_scores.numel())

            if fused_boxes.numel() == 0:
                if frame_id > warmup_frames:
                    frame_latencies.append((time.perf_counter() - t_frame_start) * 1000)
                if profile_stages and frame_id > warmup_frames:
                    seq_stage_totals["frame_total"] += (time.perf_counter() - t_e2e_start) * 1000
                    seq_profiled_frames += 1
                if frame_id % 100 == 0:
                    print(f"🎬 {seq} [{frame_id}/{frame_end}]")
                continue

            if is_tiled and fused_boxes.numel() > 0:
                if profile_stages:
                    torch.cuda.synchronize()
                    t_sub_start = time.perf_counter()
                if track_person_only:
                    keep = nms(fused_boxes, fused_scores, 0.5)
                else:
                    keep = batched_nms(fused_boxes, fused_scores, fused_classes, 0.5)
                fused_boxes = fused_boxes[keep]
                fused_scores = fused_scores[keep]
                fused_classes = fused_classes[keep]
                if profile_stages:
                    torch.cuda.synchronize()
                    seq_stage_totals["post_nms"] += (time.perf_counter() - t_sub_start) * 1000
            after_nms_count = int(fused_scores.numel())

            if cross_tile_merge and is_tiled and fused_boxes.numel() > 1:
                if profile_stages:
                    torch.cuda.synchronize()
                    t_sub_start = time.perf_counter()
                fused_boxes, fused_scores, fused_classes = merge_cross_tile_duplicates_fast(
                    fused_boxes, fused_scores, fused_classes
                )
                if profile_stages:
                    torch.cuda.synchronize()
                    seq_stage_totals["post_merge"] += (time.perf_counter() - t_sub_start) * 1000
            after_merge_count = int(fused_scores.numel())
            if profile_stages:
                torch.cuda.synchronize()
                seq_stage_totals["postprocess"] += (time.perf_counter() - t_post_start) * 1000
                if frame_id > warmup_frames:
                    seq_post_counts["raw_boxes"] += raw_box_count
                    seq_post_counts["after_filter"] += after_filter_count
                    seq_post_counts["after_nms"] += after_nms_count
                    seq_post_counts["after_merge"] += after_merge_count

            embeddings = None
            if reid_enabled and extractor and cropper and fused_boxes.numel() > 0 and frame_id % reid_interval == 0:
                if profile_stages:
                    torch.cuda.synchronize()
                    t_reid_start = time.perf_counter()
                frame_batch = pool.frame_buffer.unsqueeze(0)
                if reid_crop_layout == "parts":
                    crops = cropper.process_parts(frame_batch, fused_boxes)
                    if crops.numel() > 0:
                        part_embeddings = extractor.extract(crops).view(3, fused_boxes.shape[0], -1)
                        weights = torch.tensor(
                            [0.5, 0.3, 0.2],
                            device=part_embeddings.device,
                            dtype=part_embeddings.dtype,
                        ).view(3, 1, 1)
                        embeddings = F.normalize((part_embeddings * weights).sum(dim=0), dim=-1)
                else:
                    crops = cropper.process(frame_batch, fused_boxes)
                    if crops.numel() > 0:
                        embeddings = extractor.extract(crops)
                if profile_stages:
                    torch.cuda.synchronize()
                    seq_stage_totals["reid"] += (time.perf_counter() - t_reid_start) * 1000

            mid_thresh_scale = geometry_mid_thresh_scale(
                fused_boxes, fused_classes, h_orig,
                enabled=geometry_mid_scale,
                person_class=person_class,
                track_person_only=track_person_only,
                ref_height_ratio=geometry_ref_height_ratio,
                min_scale=geometry_min_scale,
                max_scale=geometry_max_scale,
                ema_beta=geometry_ema_beta,
                loosen_step=geometry_loosen_step,
                tighten_step=geometry_tighten_step,
                min_samples=geometry_min_samples,
                state=geometry_scale_state,
            )
            tracks, _ = time_stage(
                seq_stage_totals,
                "track",
                lambda: detector.tracker.update(
                    fused_boxes,
                    fused_scores,
                    fused_classes.to(torch.int32),
                    embeddings=embeddings if use_tracker_reid else None,
                    mid_thresh_scale=mid_thresh_scale,
                ),
                sync_cuda=True,
            )

            if profile_lazy_reid_candidates:
                candidates = detector.tracker.get_tentative_candidates()
                ready_candidates = [
                    c for c in candidates
                    if c.hit_streak >= lazy_reid_min_hit_streak
                    and c.hit_streak < c.required_confirm_streak
                ]
                seq_lazy_reid_candidates += len(ready_candidates)
                seq_lazy_reid_frames += 1
                if profile_lazy_reid_embeddings and extractor and cropper and candidates:
                    ready_ids = {int(c.obj_id) for c in ready_candidates}

                    def _profile_lazy_reid_embeddings() -> tuple[int, int, int, float, int, int, set[int]]:
                        embed_candidates = [
                            c for c in candidates
                            if int(c.class_id) == person_class and c.hit_streak >= 1
                        ]
                        if not embed_candidates:
                            return 0, 0, 0, 0.0, 0, 0, set()
                        cand_boxes = torch.tensor(
                            [[c.x1, c.y1, c.x2, c.y2] for c in embed_candidates],
                            device=pool.frame_buffer.device,
                            dtype=torch.float32,
                        )
                        crops = cropper.process(pool.frame_buffer.unsqueeze(0), cand_boxes)
                        if crops.numel() == 0:
                            return 0, 0, 0, 0.0, 0, 0, set()
                        cand_embeddings = extractor.extract(crops)
                        pairs, passed, sim_sum = 0, 0, 0.0
                        arbiter_checks, arbiter_approve = 0, 0
                        seen_ids: set[int] = set()
                        for cand, emb in zip(embed_candidates, cand_embeddings):
                            tid = int(cand.obj_id)
                            seen_ids.add(tid)
                            prev = lazy_reid_prev_embeddings.get(tid)
                            if prev is not None:
                                sim = float(torch.dot(prev, emb).item())
                                pairs += 1
                                sim_sum += sim
                                if sim >= lazy_reid_self_threshold:
                                    passed += 1
                                if tid in ready_ids:
                                    arbiter_checks += 1
                                    if sim >= lazy_reid_self_threshold:
                                        arbiter_approve += 1
                            lazy_reid_prev_embeddings[tid] = emb.detach()
                        return len(embed_candidates), pairs, passed, sim_sum, arbiter_checks, arbiter_approve, seen_ids

                    (crop_count, pair_count, pass_count, sim_sum, arbiter_checks, arbiter_approve, seen_ids), _ = time_stage(
                        seq_stage_totals, "lazy_reid", _profile_lazy_reid_embeddings, sync_cuda=True,
                    )
                    seq_lazy_reid_crops += crop_count
                    seq_lazy_reid_self_pairs += pair_count
                    seq_lazy_reid_self_pass += pass_count
                    seq_lazy_reid_self_sim_sum += sim_sum
                    seq_lazy_reid_arbiter_checks += arbiter_checks
                    seq_lazy_reid_arbiter_approve += arbiter_approve
                    if seen_ids:
                        for stale_id in set(lazy_reid_prev_embeddings.keys()) - seen_ids:
                            lazy_reid_prev_embeddings.pop(stale_id, None)

            if profile_stages:
                torch.cuda.synchronize()
                t_relink_write_start = time.perf_counter()
            if relinker:
                relinker.update_motion_snapshots(detector.tracker.get_state_snapshots())

            assigned_ids: set = set()
            for t in tracks:
                if int(t.class_id) != person_class:
                    continue
                tid = t.obj_id
                if relinker:
                    emb = None
                    if embeddings is not None:
                        tb = torch.tensor([t.x1, t.y1, t.x2, t.y2], device=fused_boxes.device)
                        ious = _box_iou_single(tb, fused_boxes)
                        best = int(ious.argmax())
                        if float(ious[best]) > 0.5:
                            emb = embeddings[best]
                    tid = relinker.resolve(
                        t.obj_id, emb,
                        (t.x1, t.y1, t.x2, t.y2),
                        t.score, frame_id, w_orig, h_orig, assigned_ids,
                    )
                global_tid = global_id_mapper.map(seq, tid)
                x1, y1, x2, y2 = t.x1, t.y1, t.x2, t.y2
                results_lines.append(
                    f"{frame_id},{global_tid},{max(0,x1):.2f},{max(0,y1):.2f},"
                    f"{min(w_orig,x2)-max(0,x1):.2f},{min(h_orig,y2)-max(0,y1):.2f},"
                    f"{t.score:.4f},-1,-1,-1"
                )
            if profile_stages:
                torch.cuda.synchronize()
                seq_stage_totals["relink_write"] += (time.perf_counter() - t_relink_write_start) * 1000

            if frame_id > warmup_frames:
                frame_latencies.append((time.perf_counter() - t_frame_start) * 1000)
            if profile_stages and frame_id > warmup_frames:
                seq_stage_totals["frame_total"] += (time.perf_counter() - t_e2e_start) * 1000
                seq_profiled_frames += 1
            if frame_id % 100 == 0:
                print(f"🎬 {seq} [{frame_id}/{frame_end}]")

        if frame_latencies:
            lats = np.array(frame_latencies)
            mean_ms = float(np.mean(lats))
            fps = 1000.0 / mean_ms
            print(f"\n📊 Production Latency Report for {seq}:")
            print(f"  - FPS:  {fps:.2f}")
            print(f"  - Mean latency: {mean_ms:.2f} ms")
            fps_summary_lines.append(
                f"{seq}\tfps={fps:.2f}\tmean_ms={mean_ms:.2f}\tframes={len(frame_latencies)}"
            )
            overall_latency_ms.extend(frame_latencies)
        else:
            fps_summary_lines.append(f"{seq}\tfps=n/a\tmean_ms=n/a\tframes=0")

        Path(output_root / f"{seq}.txt").write_text("\n".join(results_lines))
        print(f"✅ Finished {seq} (Total Time: {time.time()-start_time:.2f}s)")
        if relinker:
            relinker.report()
        if profile_stages and seq_profiled_frames > 0:
            print(f"\n🧪 Stage Profile for {seq}:")
            for stage_name, total_ms in seq_stage_totals.items():
                mean_ms = total_ms / seq_profiled_frames
                share = total_ms / max(seq_stage_totals["frame_total"], 1e-6) * 100.0
                print(f"  - {stage_name}: {mean_ms:.2f} ms/frame ({share:.1f}%)")
                overall_stage_totals[stage_name] += total_ms
            if any(seq_post_counts.values()):
                print("  - post_counts:")
                for count_name, total_count in seq_post_counts.items():
                    mean_count = total_count / seq_profiled_frames
                    print(f"    - {count_name}: {mean_count:.1f} boxes/frame")
                    overall_post_counts[count_name] += total_count
            if profile_lazy_reid_candidates and seq_lazy_reid_frames > 0:
                mean_lazy = seq_lazy_reid_candidates / seq_lazy_reid_frames
                print(f"  - lazy_reid_candidates: {mean_lazy:.2f}/frame ({seq_lazy_reid_candidates} total)")
                overall_lazy_reid_candidates += seq_lazy_reid_candidates
                overall_lazy_reid_frames += seq_lazy_reid_frames
                overall_lazy_reid_crops += seq_lazy_reid_crops
                overall_lazy_reid_self_pairs += seq_lazy_reid_self_pairs
                overall_lazy_reid_self_pass += seq_lazy_reid_self_pass
                overall_lazy_reid_self_sim_sum += seq_lazy_reid_self_sim_sum
                overall_lazy_reid_arbiter_checks += seq_lazy_reid_arbiter_checks
                overall_lazy_reid_arbiter_approve += seq_lazy_reid_arbiter_approve
                if profile_lazy_reid_embeddings:
                    mean_crops = seq_lazy_reid_crops / seq_lazy_reid_frames
                    mean_sim = seq_lazy_reid_self_sim_sum / max(seq_lazy_reid_self_pairs, 1)
                    pass_rate = seq_lazy_reid_self_pass / max(seq_lazy_reid_self_pairs, 1) * 100.0
                    print(
                        f"  - lazy_reid_embeddings: {mean_crops:.2f} crops/frame, "
                        f"self_pairs={seq_lazy_reid_self_pairs}, mean_cos={mean_sim:.3f}, "
                        f"pass@{lazy_reid_self_threshold:.2f}={pass_rate:.1f}%"
                    )
                    arbiter_rate = seq_lazy_reid_arbiter_approve / max(seq_lazy_reid_arbiter_checks, 1) * 100.0
                    print(
                        f"  - lazy_reid_arbiter_dry_run: checks={seq_lazy_reid_arbiter_checks}, "
                        f"approve={seq_lazy_reid_arbiter_approve} ({arbiter_rate:.1f}%)"
                    )
            overall_profiled_frames += seq_profiled_frames
            stage_summary_lines.append(f"[{seq}] frames={seq_profiled_frames}")
            for stage_name, total_ms in seq_stage_totals.items():
                mean_ms = total_ms / seq_profiled_frames
                share = total_ms / max(seq_stage_totals["frame_total"], 1e-6) * 100.0
                stage_summary_lines.append(
                    f"{stage_name}\tmean_ms={mean_ms:.2f}\ttotal_ms={total_ms:.2f}\tshare={share:.1f}%"
                )
            for count_name, total_count in seq_post_counts.items():
                mean_count = total_count / seq_profiled_frames
                stage_summary_lines.append(f"{count_name}\tmean={mean_count:.1f}\ttotal={total_count}")
            if profile_lazy_reid_candidates and seq_lazy_reid_frames > 0:
                mean_lazy = seq_lazy_reid_candidates / seq_lazy_reid_frames
                stage_summary_lines.append(
                    f"lazy_reid_candidates\tmean={mean_lazy:.2f}\ttotal={seq_lazy_reid_candidates}"
                )
                if profile_lazy_reid_embeddings:
                    mean_crops = seq_lazy_reid_crops / seq_lazy_reid_frames
                    mean_sim = seq_lazy_reid_self_sim_sum / max(seq_lazy_reid_self_pairs, 1)
                    pass_rate = seq_lazy_reid_self_pass / max(seq_lazy_reid_self_pairs, 1) * 100.0
                    stage_summary_lines.append(
                        f"lazy_reid_embeddings\tmean_crops={mean_crops:.2f}\t"
                        f"self_pairs={seq_lazy_reid_self_pairs}\tmean_cos={mean_sim:.3f}\t"
                        f"pass_rate={pass_rate:.1f}%"
                    )
                    arbiter_rate = seq_lazy_reid_arbiter_approve / max(seq_lazy_reid_arbiter_checks, 1) * 100.0
                    stage_summary_lines.append(
                        f"lazy_reid_arbiter_dry_run\tchecks={seq_lazy_reid_arbiter_checks}\t"
                        f"approve={seq_lazy_reid_arbiter_approve}\tapprove_rate={arbiter_rate:.1f}%"
                    )
            stage_summary_lines.append("")

    if fps_summary_lines:
        if overall_latency_ms:
            overall_mean_ms = float(np.mean(np.array(overall_latency_ms)))
            overall_fps = 1000.0 / overall_mean_ms
            fps_summary_lines.append(
                f"OVERALL\tfps={overall_fps:.2f}\tmean_ms={overall_mean_ms:.2f}\tframes={len(overall_latency_ms)}"
            )
            print(f"\n📈 Overall throughput: {overall_fps:.2f} FPS ({overall_mean_ms:.2f} ms)")
        (output_root / "_fps_summary.txt").write_text("\n".join(fps_summary_lines) + "\n")
    mapping_lines = global_id_mapper.dump_lines()
    if mapping_lines:
        (output_root / "_global_id_map.txt").write_text("\n".join(mapping_lines) + "\n")
    if profile_stages and overall_profiled_frames > 0:
        print(f"\n🧪 Overall Stage Profile ({overall_profiled_frames} frames):")
        stage_summary_lines.append(f"[OVERALL] frames={overall_profiled_frames}")
        for stage_name, total_ms in overall_stage_totals.items():
            mean_ms = total_ms / overall_profiled_frames
            share = total_ms / max(overall_stage_totals["frame_total"], 1e-6) * 100.0
            print(f"  - {stage_name}: {mean_ms:.2f} ms/frame ({share:.1f}%)")
            stage_summary_lines.append(
                f"{stage_name}\tmean_ms={mean_ms:.2f}\ttotal_ms={total_ms:.2f}\tshare={share:.1f}%"
            )
        if any(overall_post_counts.values()):
            print("  - post_counts:")
            for count_name, total_count in overall_post_counts.items():
                mean_count = total_count / overall_profiled_frames
                print(f"    - {count_name}: {mean_count:.1f} boxes/frame")
                stage_summary_lines.append(f"{count_name}\tmean={mean_count:.1f}\ttotal={total_count}")
        if profile_lazy_reid_candidates and overall_lazy_reid_frames > 0:
            mean_lazy = overall_lazy_reid_candidates / overall_lazy_reid_frames
            print(f"  - lazy_reid_candidates: {mean_lazy:.2f}/frame ({overall_lazy_reid_candidates} total)")
            stage_summary_lines.append(
                f"lazy_reid_candidates\tmean={mean_lazy:.2f}\ttotal={overall_lazy_reid_candidates}"
            )
            if profile_lazy_reid_embeddings:
                mean_crops = overall_lazy_reid_crops / overall_lazy_reid_frames
                mean_sim = overall_lazy_reid_self_sim_sum / max(overall_lazy_reid_self_pairs, 1)
                pass_rate = overall_lazy_reid_self_pass / max(overall_lazy_reid_self_pairs, 1) * 100.0
                print(
                    f"  - lazy_reid_embeddings: {mean_crops:.2f} crops/frame, "
                    f"self_pairs={overall_lazy_reid_self_pairs}, mean_cos={mean_sim:.3f}, "
                    f"pass@{lazy_reid_self_threshold:.2f}={pass_rate:.1f}%"
                )
                arbiter_rate = overall_lazy_reid_arbiter_approve / max(overall_lazy_reid_arbiter_checks, 1) * 100.0
                print(
                    f"  - lazy_reid_arbiter_dry_run: checks={overall_lazy_reid_arbiter_checks}, "
                    f"approve={overall_lazy_reid_arbiter_approve} ({arbiter_rate:.1f}%)"
                )
                stage_summary_lines.append(
                    f"lazy_reid_embeddings\tmean_crops={mean_crops:.2f}\t"
                    f"self_pairs={overall_lazy_reid_self_pairs}\tmean_cos={mean_sim:.3f}\t"
                    f"pass_rate={pass_rate:.1f}%"
                )
                stage_summary_lines.append(
                    f"lazy_reid_arbiter_dry_run\tchecks={overall_lazy_reid_arbiter_checks}\t"
                    f"approve={overall_lazy_reid_arbiter_approve}\tapprove_rate={arbiter_rate:.1f}%"
                )
        (output_root / "_stage_profile.txt").write_text("\n".join(stage_summary_lines) + "\n")
