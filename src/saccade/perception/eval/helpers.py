import torch
from typing import Any
from .types import (
    HostTrackResultView,
    HostTrackBatch,
    PreparedTrackCandidate,
    CandidateAppearanceUpdate,
    ResolvedTrack,
)
from .quality import (
    compute_bank_quality_score as _compute_bank_quality_score,
)
from .utils import (
    mot_result_line as _mot_result_line,
)
from saccade.perception.eval.detection import (
    _box_iou_matrix,
    _box_iou_pairwise_diag,
)


_CUDA_COPY_STREAM: "torch.cuda.Stream | None" = None


def _get_copy_stream() -> torch.cuda.Stream:
    global _CUDA_COPY_STREAM
    if _CUDA_COPY_STREAM is None:
        _CUDA_COPY_STREAM = torch.cuda.Stream()
    return _CUDA_COPY_STREAM


def materialize_gpu_track_results(
    result_buffers: Any,
    *,
    default_class_id: int | None = None,
    include_det_idx: bool = True,
) -> HostTrackResultView:
    count = int(result_buffers["count"].cpu().item())
    if count <= 0:
        return {
            "count": 0,
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "scores": torch.empty((0,), dtype=torch.float32),
            "ids": torch.empty((0,), dtype=torch.int32),
            "classes": (
                None
                if default_class_id is not None
                else torch.empty((0,), dtype=torch.int32)
            ),
            "det_idx": (
                torch.empty((0,), dtype=torch.int32) if include_det_idx else None
            ),
        }

    boxes = result_buffers["boxes"][:count].cpu()
    scores = result_buffers["scores"][:count].cpu()
    ids = result_buffers["ids"][:count].cpu()
    classes = (
        None
        if default_class_id is not None
        else result_buffers["classes"][:count].cpu()
    )
    det_idx = result_buffers["det_idx"][:count].cpu() if include_det_idx else None

    if default_class_id is not None:
        classes = torch.full_like(ids, int(default_class_id), dtype=torch.int32)

    return {
        "count": count,
        "boxes": boxes,
        "scores": scores,
        "ids": ids,
        "classes": classes,
        "det_idx": det_idx,
    }


def materialize_gpu_track_results_async(
    result_buffers: Any,
    pinned: dict[str, torch.Tensor],
    *,
    default_class_id: int | None = None,
    include_det_idx: bool = True,
) -> tuple[torch.cuda.Event, dict[str, torch.Tensor]]:
    """Async D2H: launch non-blocking copies on a dedicated copy stream.

    Records a CUDA event that fires when all copies land in host memory.
    Call ``read_deferred_result(event, pinned, ...)`` after the event completes.

    Copies full MAX_TRACKS (pinned buffer capacity) so no count-dependent
    slicing is needed. The copy stream is isolated from the default (compute)
    stream so the GPU can start the next frame while DMA is still in flight.
    """
    compute_done = torch.cuda.Event()
    compute_done.record(torch.cuda.current_stream())

    copy_stream = _get_copy_stream()
    copy_stream.wait_event(compute_done)
    with torch.cuda.stream(copy_stream):
        pinned["count"].copy_(result_buffers["count"], non_blocking=True)
        pinned["boxes"].copy_(result_buffers["boxes"], non_blocking=True)
        pinned["scores"].copy_(result_buffers["scores"], non_blocking=True)
        pinned["ids"].copy_(result_buffers["ids"], non_blocking=True)
        if default_class_id is None:
            pinned["classes"].copy_(result_buffers["classes"], non_blocking=True)
        if include_det_idx:
            pinned["det_idx"].copy_(result_buffers["det_idx"], non_blocking=True)
        event = torch.cuda.Event()
        event.record(copy_stream)
    return event, pinned


def read_deferred_result(
    event: torch.cuda.Event,
    pinned: dict[str, torch.Tensor],
    *,
    default_class_id: int | None = None,
    include_det_idx: bool = True,
) -> HostTrackResultView:
    event.synchronize()
    count = int(pinned["count"].item())
    if count <= 0:
        return {
            "count": 0,
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "scores": torch.empty((0,), dtype=torch.float32),
            "ids": torch.empty((0,), dtype=torch.int32),
            "classes": (
                None
                if default_class_id is not None
                else torch.empty((0,), dtype=torch.int32)
            ),
            "det_idx": (
                torch.empty((0,), dtype=torch.int32) if include_det_idx else None
            ),
        }

    boxes = pinned["boxes"][:count]
    scores = pinned["scores"][:count]
    ids = pinned["ids"][:count]

    if default_class_id is not None:
        classes = torch.full_like(ids, int(default_class_id), dtype=torch.int32)
    else:
        classes = pinned["classes"][:count]

    det_idx = pinned["det_idx"][:count] if include_det_idx else None

    return {
        "count": count,
        "boxes": boxes,
        "scores": scores,
        "ids": ids,
        "classes": classes,
        "det_idx": det_idx,
    }


def materialize_gpu_track_results_pinned(
    result_buffers: Any,
    pinned: dict[str, torch.Tensor],
    *,
    default_class_id: int | None = None,
    include_det_idx: bool = True,
) -> HostTrackResultView:
    """Blocking read of pinned D2H results via a dedicated copy stream.

    Launches copies on a non-default stream so the default (compute) stream
    is free to start the next frame before DMA completes.  Only the copy
    stream's event is synchronized — ``torch.cuda.synchronize()`` is avoided.
    """
    compute_done = torch.cuda.Event()
    compute_done.record(torch.cuda.current_stream())

    copy_stream = _get_copy_stream()
    copy_stream.wait_event(compute_done)
    with torch.cuda.stream(copy_stream):
        pinned["count"].copy_(result_buffers["count"], non_blocking=True)
        pinned["boxes"].copy_(result_buffers["boxes"], non_blocking=True)
        pinned["scores"].copy_(result_buffers["scores"], non_blocking=True)
        pinned["ids"].copy_(result_buffers["ids"], non_blocking=True)
        if default_class_id is None:
            pinned["classes"].copy_(result_buffers["classes"], non_blocking=True)
        if include_det_idx:
            pinned["det_idx"].copy_(result_buffers["det_idx"], non_blocking=True)
        copy_done = torch.cuda.Event()
        copy_done.record(copy_stream)

    copy_done.synchronize()
    count = int(pinned["count"].item())
    if count <= 0:
        return {
            "count": 0,
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "scores": torch.empty((0,), dtype=torch.float32),
            "ids": torch.empty((0,), dtype=torch.int32),
            "classes": (
                None
                if default_class_id is not None
                else torch.empty((0,), dtype=torch.int32)
            ),
            "det_idx": (
                torch.empty((0,), dtype=torch.int32) if include_det_idx else None
            ),
        }

    boxes = pinned["boxes"][:count]
    scores = pinned["scores"][:count]
    ids = pinned["ids"][:count]

    if default_class_id is not None:
        classes = torch.full_like(ids, int(default_class_id), dtype=torch.int32)
    else:
        classes = pinned["classes"][:count]

    det_idx = pinned["det_idx"][:count] if include_det_idx else None

    return {
        "count": count,
        "boxes": boxes,
        "scores": scores,
        "ids": ids,
        "classes": classes,
        "det_idx": det_idx,
    }


def fast_emit_mot_lines(
    *,
    track_results: HostTrackResultView,
    global_id_mapper: Any,
    seq: str,
    frame_id: int,
    frame_w: int,
    frame_h: int,
) -> list[str]:
    count = track_results["count"]
    if count <= 0:
        return []

    boxes_np = track_results["boxes"][:count].numpy()
    scores_np = track_results["scores"][:count].numpy()
    ids_np = track_results["ids"][:count].numpy()

    lines = [""] * count
    for i in range(count):
        gid = global_id_mapper.map(seq, int(ids_np[i]))
        x1, y1, x2, y2 = boxes_np[i]
        s = float(scores_np[i])
        w = x2 - x1
        h = y2 - y1
        lines[i] = (
            f"{frame_id},{gid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{s:.4f},-1,-1,-1"
        )
    return lines


def build_dynamic_reid_observations(
    track_ids: list[int],
    track_boxes: list[tuple[float, float, float, float]],
    track_scores: list[float],
    track_classes: list[int] | None,
    *,
    person_class: int,
) -> dict[int, Any]:
    if track_classes is None:
        return {}
    # ReIDTrackObservation imported locally to avoid circularity if needed,
    # but it's already in the workspace.
    from saccade.perception.tracking.tracker_gpu import ReIDTrackObservation

    observations: dict[int, ReIDTrackObservation] = {}
    for obj_id, box, score, class_id in zip(
        track_ids, track_boxes, track_scores, track_classes
    ):
        if class_id != person_class:
            continue
        observations[obj_id] = ReIDTrackObservation(box=box, det_score=score)
    return observations


def prepare_host_track_batch(
    track_results: HostTrackResultView,
    tracker_result_buffers: Any,
    *,
    dynamic_reid_enabled: bool,
    person_class: int,
) -> HostTrackBatch:
    count = track_results["count"]
    boxes_np = track_results["boxes"][:count].numpy()
    scores_np = track_results["scores"][:count].numpy()
    ids_np = track_results["ids"][:count].numpy()
    boxes = [tuple(row.astype(float)) for row in boxes_np]
    scores = [float(s) for s in scores_np.tolist()]
    ids = [int(i) for i in ids_np.tolist()]
    classes = (
        [int(c) for c in track_results["classes"][:count].numpy().tolist()]
        if track_results["classes"] is not None
        else None
    )
    det_idx = (
        [int(d) for d in track_results["det_idx"][:count].numpy().tolist()]
        if track_results["det_idx"] is not None
        else None
    )
    person_observations = (
        build_dynamic_reid_observations(
            ids,
            boxes,
            scores,
            classes,
            person_class=person_class,
        )
        if dynamic_reid_enabled
        else None
    )
    return HostTrackBatch(
        boxes_gpu=tracker_result_buffers["boxes"][:count],
        boxes=boxes,
        scores=scores,
        ids=ids,
        classes=classes,
        det_idx=det_idx,
        person_observations=person_observations,
    )


def resolve_frame_tracks(
    *,
    frame_id: int,
    frame_w: int,
    frame_h: int,
    prepared_candidates: list[PreparedTrackCandidate],
    lifecycle_merger: Any,
    identity_resolver: Any = None,
) -> list[ResolvedTrack]:
    if identity_resolver is not None and prepared_candidates:
        resolved_ids = identity_resolver.resolve_pass(
            [c.local_track_id for c in prepared_candidates],
            [c.embedding for c in prepared_candidates],
            [c.box for c in prepared_candidates],
            [c.score for c in prepared_candidates],
            frame_id=frame_id,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        return [
            ResolvedTrack(
                local_track_id=candidate.local_track_id,
                resolved_track_id=int(resolved_id),
                box=candidate.box,
                score=candidate.score,
                embedding=candidate.embedding,
            )
            for candidate, resolved_id in zip(prepared_candidates, resolved_ids)
        ]

    if not prepared_candidates:
        return []
    resolve_many_packed = getattr(lifecycle_merger, "resolve_many_packed", None)
    if callable(resolve_many_packed):
        resolved_ids = resolve_many_packed(
            [c.local_track_id for c in prepared_candidates],
            [c.box for c in prepared_candidates],
            [c.score for c in prepared_candidates],
            [None] * len(prepared_candidates),
            frame_id=frame_id,
            frame_w=frame_w,
            frame_h=frame_h,
        )
    else:
        resolved_ids = lifecycle_merger.resolve_many(
            [(c.local_track_id, c.box, c.score, None) for c in prepared_candidates],
            frame_id=frame_id,
            frame_w=frame_w,
            frame_h=frame_h,
        )
    return [
        ResolvedTrack(
            local_track_id=candidate.local_track_id,
            resolved_track_id=int(resolved_id),
            box=candidate.box,
            score=candidate.score,
            embedding=candidate.embedding,
        )
        for candidate, resolved_id in zip(prepared_candidates, resolved_ids)
    ]


def collect_stability_candidates(
    *,
    track_results: HostTrackResultView,
    host_batch: HostTrackBatch,
    person_class: int,
    track_person_only: bool,
    fused_boxes: torch.Tensor,
    geometry_suspect_mask: torch.Tensor,
    geometry_suspect_support: bool,
    geometry_suspect_support_score: float,
) -> tuple[list[int], list[tuple[int, tuple[float, float, float, float], float]]]:
    count = track_results["count"]
    person_indices: list[int] = []
    for i in range(count):
        class_id = host_batch.classes[i] if host_batch.classes is not None else -1
        if not track_person_only and class_id != person_class:
            continue
        person_indices.append(i)

    if not person_indices:
        return [], []

    excluded: set[int] = set()
    if geometry_suspect_support and geometry_suspect_mask.any():
        # Guard: external_fp_filter / fp_hard_filter may have removed detections
        # from fused_boxes without updating geometry_suspect_mask (bug). Re-sync.
        if geometry_suspect_mask.shape[0] != fused_boxes.shape[0]:
            geometry_suspect_mask = torch.zeros(
                fused_boxes.shape[0], dtype=torch.bool, device=fused_boxes.device
            )
        low_i = [
            i
            for i in person_indices
            if host_batch.scores[i] <= geometry_suspect_support_score + 1e-4
        ]
        if low_i:
            track_boxes = host_batch.boxes_gpu[low_i]
            suspect_boxes = fused_boxes[geometry_suspect_mask]
            if suspect_boxes.numel() > 0:
                max_ious = _box_iou_matrix(track_boxes, suspect_boxes).max(dim=1).values
                excluded = {
                    low_i[j] for j, v in enumerate(max_ious.cpu().tolist()) if v > 0.5
                }

    candidate_indices: list[int] = []
    stability_candidates: list[
        tuple[int, tuple[float, float, float, float], float]
    ] = []
    for i in person_indices:
        if i in excluded:
            continue
        candidate_indices.append(i)
        stability_candidates.append(
            (host_batch.ids[i], host_batch.boxes[i], host_batch.scores[i])
        )
    return candidate_indices, stability_candidates


def build_prepared_candidates(
    *,
    candidate_indices: list[int],
    stability_accepts: list[bool],
    host_batch: HostTrackBatch,
    embeddings: torch.Tensor | None,
    fused_boxes: torch.Tensor,
    fused_scores: torch.Tensor,
    geometry_suspect_mask: torch.Tensor,
    frame_id: int,
    frame_w: int = 0,
    frame_h: int = 0,
    bank_quality_v2: bool = False,
    bank_quality_w_det: float = 0.45,
    bank_quality_w_iou: float = 0.20,
    bank_quality_w_aspect: float = 0.15,
    bank_quality_w_center: float = 0.10,
    bank_quality_w_area: float = 0.10,
) -> tuple[list[PreparedTrackCandidate], list[CandidateAppearanceUpdate]]:
    pairs: list[tuple[int, int, int]] = []
    accepted_flat: list[tuple[int, int, int, float]] = []
    for loop_idx, (i, accepted) in enumerate(zip(candidate_indices, stability_accepts)):
        if not accepted:
            continue
        obj_id = host_batch.ids[i]
        score = host_batch.scores[i]
        accepted_flat.append((loop_idx, i, obj_id, score))
        det_idx = host_batch.det_idx[i] if host_batch.det_idx is not None else -1
        if embeddings is not None and 0 <= det_idx < fused_boxes.shape[0]:
            pairs.append((loop_idx, i, det_idx))

    precomp: dict[
        int, tuple[float, float, float, bool, tuple[float, float, float, float]]
    ] = {}
    if pairs:
        li_list = [p[0] for p in pairs]
        ti_list = [p[1] for p in pairs]
        di_list = [p[2] for p in pairs]
        track_boxes = host_batch.boxes_gpu[ti_list]
        det_boxes_gpu = fused_boxes[di_list]
        ious_gpu = _box_iou_pairwise_diag(track_boxes, det_boxes_gpu)
        bw_gpu = (det_boxes_gpu[:, 2] - det_boxes_gpu[:, 0]).clamp(min=1e-6)
        bh_gpu = (det_boxes_gpu[:, 3] - det_boxes_gpu[:, 1]).clamp(min=1e-6)
        aspects_gpu = bh_gpu / bw_gpu
        det_scores_gpu = fused_scores[di_list]
        batch_cpu = (
            torch.stack(
                [
                    ious_gpu,
                    aspects_gpu,
                    det_scores_gpu,
                    det_boxes_gpu[:, 0],
                    det_boxes_gpu[:, 1],
                    det_boxes_gpu[:, 2],
                    det_boxes_gpu[:, 3],
                ],
                dim=1,
            )
            .cpu()
            .tolist()
        )
        has_suspect = geometry_suspect_mask.numel() > 0
        for li, di, row in zip(li_list, di_list, batch_cpu):
            iou, asp, ds, x1, y1, x2, y2 = row
            suspect = (
                bool(geometry_suspect_mask[di])
                if has_suspect and di < geometry_suspect_mask.numel()
                else False
            )
            precomp[li] = (iou, asp, ds, suspect, (x1, y1, x2, y2))

    prepared: list[PreparedTrackCandidate] = []
    appearance_updates: list[CandidateAppearanceUpdate] = []
    for loop_idx, i, obj_id, score in accepted_flat:
        raw_box = host_batch.boxes[i]
        det_idx = host_batch.det_idx[i] if host_batch.det_idx is not None else -1
        emb = (
            embeddings[det_idx]
            if (embeddings is not None and 0 <= det_idx < fused_boxes.shape[0])
            else None
        )
        if emb is not None and loop_idx in precomp:
            match_iou, aspect_ratio, _det_score, _suspect, _det_box = precomp[loop_idx]
            if match_iou >= 0.0:
                if bank_quality_v2 and frame_w > 0 and frame_h > 0:
                    _bq = _compute_bank_quality_score(
                        _det_score,
                        match_iou,
                        aspect_ratio,
                        _det_box,
                        frame_w,
                        frame_h,
                        w_det=bank_quality_w_det,
                        w_iou=bank_quality_w_iou,
                        w_aspect=bank_quality_w_aspect,
                        w_center=bank_quality_w_center,
                        w_area=bank_quality_w_area,
                    )
                else:
                    _bq = 0.0
                appearance_updates.append(
                    CandidateAppearanceUpdate(
                        track_id=obj_id,
                        embedding=emb,
                        det_score=_det_score,
                        iou=match_iou,
                        frame_id=frame_id,
                        geometry_clean=True,
                        suspect_box=_suspect,
                        aspect_ratio=aspect_ratio,
                        bank_quality_score=_bq,
                        box=_det_box,
                    )
                )
        prepared.append(
            PreparedTrackCandidate(
                local_track_id=obj_id,
                box=raw_box,
                score=score,
                embedding=emb,
            )
        )
    return prepared, appearance_updates


def apply_consistency_gate(
    prepared: list[PreparedTrackCandidate],
    primary_appearance_bank: Any,
) -> list[PreparedTrackCandidate]:
    return [
        PreparedTrackCandidate(
            local_track_id=candidate.local_track_id,
            box=candidate.box,
            score=candidate.score,
            embedding=(
                None
                if not primary_appearance_bank.is_consistent(candidate.local_track_id)
                else candidate.embedding
            ),
        )
        for candidate in prepared
    ]


def emit_resolved_tracks(
    *,
    seq: str,
    frame_id: int,
    frame_w: int,
    frame_h: int,
    resolved_tracks: list[ResolvedTrack],
    global_id_mapper: Any,
    output_appearance_bank: Any,
) -> list[str]:
    mapped_tracks = [
        (
            global_id_mapper.map(seq, track.resolved_track_id),
            track.box,
            track.score,
            track.embedding,
        )
        for track in resolved_tracks
    ]
    if output_appearance_bank is not None and mapped_tracks:
        output_appearance_bank.update_many(
            [
                (int(global_tid), embedding, score, frame_id)
                for global_tid, _, score, embedding in mapped_tracks
            ]
        )
    return [
        _mot_result_line(frame_id, global_tid, box, score, frame_w, frame_h)
        for global_tid, box, score, _ in mapped_tracks
    ]


def inject_lost_track_references(
    *,
    relinker: Any,
    primary_appearance_bank: Any,
    prev_track_ids: set[int],
    curr_track_ids: set[int],
) -> None:
    references_to_inject: list[tuple[int, torch.Tensor]] = []
    for lost_tid in prev_track_ids - curr_track_ids:
        canonical_id = relinker.canonical_id(lost_tid)
        if not relinker.has_feature(canonical_id):
            continue
        if primary_appearance_bank.is_high_quality(lost_tid):
            representative = primary_appearance_bank.high_quality_representative(
                lost_tid
            )
        elif primary_appearance_bank.is_consistent(lost_tid):
            representative = primary_appearance_bank.representative(lost_tid)
        else:
            continue
        if representative is None:
            continue
        references_to_inject.append((canonical_id, representative))

    if not references_to_inject:
        return
    if hasattr(relinker, "inject_references_batch"):
        ids = [cid for cid, _ in references_to_inject]
        reps = torch.stack([rep for _, rep in references_to_inject])
        relinker.inject_references_batch(ids, reps)
    elif hasattr(relinker, "inject_references_many"):
        relinker.inject_references_many(references_to_inject)
    else:
        for canonical_id, embedding in references_to_inject:
            relinker.inject_reference(canonical_id, embedding)


def finalize_frame_side_effects(
    *,
    curr_track_ids: set[int],
    prev_track_ids: set[int],
    relinker: Any,
    semantic_bank_inject: bool,
    primary_appearance_bank: Any,
    dynamic_reid: Any,
    person_observations: dict[int, Any] | None,
    gmc_warp: torch.Tensor | None,
    gmc_enabled: bool,
) -> set[int]:
    if (
        relinker is not None
        and semantic_bank_inject
        and primary_appearance_bank is not None
    ):
        inject_lost_track_references(
            relinker=relinker,
            primary_appearance_bank=primary_appearance_bank,
            prev_track_ids=prev_track_ids,
            curr_track_ids=curr_track_ids,
        )
    if dynamic_reid is not None and person_observations is not None:
        dynamic_reid.observe(
            person_observations,
            gmc=gmc_warp if gmc_enabled else None,
        )
    if primary_appearance_bank is not None:
        primary_appearance_bank.prune(curr_track_ids)
    return curr_track_ids


def budget_reid_candidates(
    fused_boxes: torch.Tensor,
    fused_scores: torch.Tensor,
    budget: int,
    dynamic_reid: Any = None,
    gmc_warp: torch.Tensor | None = None,
    gmc_uncertain: bool = False,
) -> torch.Tensor:
    num_dets = fused_boxes.shape[0]
    if budget <= 0 or num_dets <= budget:
        return torch.arange(num_dets, device=fused_boxes.device)

    if dynamic_reid is None:
        _, top_idx = torch.topk(fused_scores, budget)
        return torch.sort(top_idx).values

    track_priorities = dynamic_reid.get_priorities()
    track_boxes = dynamic_reid.get_last_boxes()

    if gmc_uncertain and track_priorities:
        track_priorities = {tid: p * 1.5 for tid, p in track_priorities.items()}

    if not track_priorities:
        _, top_idx = torch.topk(fused_scores, budget)
        return torch.sort(top_idx).values

    h00, h01, h02, h10, h11, h12 = 1.0, 0.0, 0.0, 0.0, 1.0, 0.0
    if gmc_warp is not None:
        gmc_cpu = gmc_warp.detach().cpu().view(-1).tolist()
        if len(gmc_cpu) >= 6:
            h00, h01, h02, h10, h11, h12 = gmc_cpu[:6]

    warped_track_boxes = []
    priorities_list = []
    for tid, p in track_priorities.items():
        box = track_boxes.get(tid)
        if box is None:
            continue

        if gmc_warp is not None:
            x1, y1, x2, y2 = box
            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            tx = [h00 * x + h01 * y + h02 for x, y in corners]
            ty = [h10 * x + h11 * y + h12 for x, y in corners]
            box = (min(tx), min(ty), max(tx), max(ty))

        warped_track_boxes.append(box)
        priorities_list.append(p)

    if not warped_track_boxes:
        _, top_idx = torch.topk(fused_scores, budget)
        return torch.sort(top_idx).values

    t_boxes = torch.tensor(warped_track_boxes, device=fused_boxes.device)
    t_priorities = torch.tensor(priorities_list, device=fused_boxes.device)

    lt = torch.max(fused_boxes[:, None, :2], t_boxes[None, :, :2])
    rb = torch.min(fused_boxes[:, None, 2:], t_boxes[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (fused_boxes[:, 2] - fused_boxes[:, 0]) * (
        fused_boxes[:, 3] - fused_boxes[:, 1]
    )
    area_b = (t_boxes[:, 2] - t_boxes[:, 0]) * (t_boxes[:, 3] - t_boxes[:, 1])
    iou = inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=1e-6)

    track_contribution, _ = torch.max(t_priorities[None, :] * iou, dim=1)
    det_priorities = fused_scores + track_contribution

    _, top_idx = torch.topk(det_priorities, budget)
    return torch.sort(top_idx).values


def prepare_track_candidates(
    *,
    frame_id: int,
    track_results: HostTrackResultView,
    host_batch: HostTrackBatch,
    person_class: int,
    track_person_only: bool,
    geometry_suspect_support: bool,
    geometry_suspect_support_score: float,
    id_stability_filter: Any,
    embeddings: torch.Tensor | None,
    fused_boxes: torch.Tensor,
    fused_scores: torch.Tensor,
    geometry_suspect_mask: torch.Tensor,
    primary_appearance_bank: Any,
    frame_w: int = 0,
    frame_h: int = 0,
    bank_quality_v2: bool = False,
    bank_quality_w_det: float = 0.45,
    bank_quality_w_iou: float = 0.20,
    bank_quality_w_aspect: float = 0.15,
    bank_quality_w_center: float = 0.10,
    bank_quality_w_area: float = 0.10,
) -> list[PreparedTrackCandidate]:
    candidate_indices, stability_candidates = collect_stability_candidates(
        track_results=track_results,
        host_batch=host_batch,
        person_class=person_class,
        track_person_only=track_person_only,
        fused_boxes=fused_boxes,
        geometry_suspect_mask=geometry_suspect_mask,
        geometry_suspect_support=geometry_suspect_support,
        geometry_suspect_support_score=geometry_suspect_support_score,
    )

    stability_accepts = (
        id_stability_filter.accept_many(stability_candidates, frame_id)
        if id_stability_filter
        else [True] * len(stability_candidates)
    )

    prepared, appearance_updates = build_prepared_candidates(
        candidate_indices=candidate_indices,
        stability_accepts=stability_accepts,
        host_batch=host_batch,
        embeddings=embeddings,
        fused_boxes=fused_boxes,
        fused_scores=fused_scores,
        geometry_suspect_mask=geometry_suspect_mask,
        frame_id=frame_id,
        frame_w=frame_w,
        frame_h=frame_h,
        bank_quality_v2=bank_quality_v2,
        bank_quality_w_det=bank_quality_w_det,
        bank_quality_w_iou=bank_quality_w_iou,
        bank_quality_w_aspect=bank_quality_w_aspect,
        bank_quality_w_center=bank_quality_w_center,
        bank_quality_w_area=bank_quality_w_area,
    )
    if primary_appearance_bank is not None and appearance_updates:
        primary_appearance_bank.update_many(
            [
                (
                    update.track_id,
                    update.embedding,
                    update.det_score,
                    update.iou,
                    update.frame_id,
                    update.geometry_clean,
                    update.suspect_box,
                    update.aspect_ratio,
                    update.bank_quality_score,
                    update.box,
                )
                for update in appearance_updates
            ]
        )
    if primary_appearance_bank is not None:
        prepared = apply_consistency_gate(prepared, primary_appearance_bank)
    return prepared
