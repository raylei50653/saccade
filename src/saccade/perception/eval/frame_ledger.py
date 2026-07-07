"""Lightweight per-frame CSV ledger. Zero CUDA sync, zero behavior change.

Field semantics
---------------

Timing columns (wall-clock, perf_counter-based, no GPU sync):

  total_ms              End-to-end frame latency from t_frame_start to frame end.
  fetch_ms              Frame fetch + NV12/CHW preprocess (combined in non-workbench path).
  preprocess_ms         Zero placeholder.  Preprocess is currently bundled into fetch_ms.
  detect_ms             YOLO TRT detection wall time (non-workbench) or wb.process_frame
                        (workbench path, also includes post + track).
  post_ms               Postprocess wall time (filter + NMS + quality + merge).
  gmc_ms                GMC wall time (phase-correlation PCR or optical flow).
                        Measured separately from ReID starting Phase 2.5.
  reid_ms               ReID wall time (budget + crop + extract + async sync).
                        Measured separately from GMC starting Phase 2.5.
  track_ms              Track + materialize wall time.
  handover_ms           Background relink-write wait (bg_relink_wait).
  output_ms             MOT-format emit wall time.

  handover_snapshot_ms  [Phase 2.5 placeholder — requires C++ exposure]
  handover_score_ms     [Phase 2.5 placeholder — requires C++ exposure]
  handover_apply_ms     [Phase 2.5 placeholder — requires C++ exposure]

  reid_crop_stash_ms    ReID crop submit (async) or full crop+extract (sync) wall time.
  reid_extract_submit_ms Full extract wall time (sync path only; 0 for async path).
  reid_gpu_ms           [Phase 2.5 placeholder — would require CUDA event sync]
  reid_d2h_ms           [Phase 2.5 placeholder — would require D2H sync measurement]

Count columns (no-op when the corresponding feature is disabled):

  n_dets_raw            Raw YOLO output count (per tile batch for tiled paths).
  n_dets_after_filter   After geometry/quality filter (buffer allocation for native path).
  n_dets_after_nms      After NMS (buffer allocation for native path).
  n_dets_final          Final detection count after all postprocessing.
  n_tracks              [Phase 2.5 placeholder]
  n_alive               [Phase 2.5 placeholder]
  n_dead_archive        [Phase 2.5 placeholder]
  n_newborn             [Phase 2.5 placeholder]
  n_ho_candidates       [Phase 2.5 placeholder]
  n_aliases             [Phase 2.5 placeholder]
  n_crops               Number of ReID crops this frame (nonzero when _do_reid triggers).
  n_requery             [Phase 2.5 placeholder]

ReID booleans / timing:

  reid_submitted        1 if this frame triggered ReID (embeddings were computed).
  reid_waited_this_frame 1 if this frame waited on async ReID side-stream completion.
  reid_blocking_wait_ms  Wall time spent in reid_side_event.synchronize().

Handover:

  handover_score_matrix_n [Phase 2.5 placeholder — requires C++ exposure]

Data volume:

  h2d_bytes             [Phase 2.5 placeholder]
  d2h_bytes             [Phase 2.5 placeholder]

Stream identity (Phase 6C/7.0):

  detect_stream_handle      CUDA stream handle used for TRT enqueue (hex int).
  post_capture_stream_handle CUDA stream handle used for NMS graph capture (hex int).
  post_replay_stream_handle  CUDA stream handle used for NMS graph replay (hex int).
  trt_enqueue_stream_handle  Stream handle passed to TRT execute_async_v3 (hex int).
  dets_hash                  hashlib.md5 of per-frame detection count & box-sum for determinism.
  explicit_stream_dispatch_ms Host time for explicit-stream stream switch + event fence + post replay.
  post_count_wait_ms       Entire postprocessing after graph replay (sum of below).
  post_tensor_prep_ms      Tensor cast + contiguous conversion + narrow bonus (pre-NMS).
  post_nms_replay_ms       NMS graph replay wall time (= post_graph_replay_ms).
  post_finalize_ms         Filter, quality gate, suspect mask, private continuation.
  post_any_sync_ms         Wall time for .any()/.all() implicit D2H syncs.
  post_item_sync_ms        Wall time for .item() implicit D2H reads.
  post_filter_d2h_ms       Time from after NMS to after filter_detections_fast (includes .item() D2H sync).
  post_quality_filters_ms   Time in detection filters + cross-tile merge + quality gate.
  post_tail_ms             Time from after detection filters to end of post (private continuation, tracker prep).

Usage
-----
    from .frame_ledger import FrameLedger

    ledger = FrameLedger()
    ledger.add(seq="MOT17-04-SDP", frame=10, total_ms=11.5, fetch_ms=2.8, detect_ms=5.3)
    ledger.write_csv("_frame_ledger_MOT17-04-SDP.csv")
"""

import csv
from pathlib import Path
from typing import Any

FRAME_LEDGER_FIELDS = [
    "seq",
    "frame",
    "total_ms",
    "fetch_ms",
    "preprocess_ms",
    "detect_ms",
    "detect_ingest_barrier_ms",
    "detect_trt_enqueue_ms",
    "detect_postproc_barrier_ms",
    "post_ms",
    "gmc_ms",
    "post_filter_count_sync_ms",
    "post_nms_count_sync_ms",
    "post_final_count_sync_ms",
    "post_graph_replay_ms",
    "post_graph_count_wait_ms",
    "post_pre_nms_ms",
    "reid_ms",
    "track_ms",
    "handover_ms",
    "output_ms",
    "handover_snapshot_ms",
    "handover_score_ms",
    "handover_apply_ms",
    "reid_crop_stash_ms",
    "reid_extract_submit_ms",
    "reid_gpu_ms",
    "reid_d2h_ms",
    "n_dets_raw",
    "n_dets_after_filter",
    "n_dets_after_nms",
    "n_dets_final",
    "n_tracks",
    "n_alive",
    "n_dead_archive",
    "n_newborn",
    "n_ho_candidates",
    "n_aliases",
    "n_crops",
    "n_requery",
    "reid_submitted",
    "reid_waited_this_frame",
    "reid_blocking_wait_ms",
    "handover_score_matrix_n",
    "h2d_bytes",
    "d2h_bytes",
    "detect_stream_handle",
    "post_capture_stream_handle",
    "post_replay_stream_handle",
    "trt_enqueue_stream_handle",
    "dets_hash",
    "explicit_stream_dispatch_ms",
    "trt_enqueue_host_ms",
    "event_record_ms",
    "post_graph_launch_host_ms",
    "post_tensor_prep_ms",
    "post_nms_replay_ms",
    "post_finalize_ms",
    "post_any_sync_ms",
    "post_item_sync_ms",
    "post_filter_d2h_ms",
    "post_quality_filters_ms",
    "post_tail_ms",
]


class FrameLedger:
    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def add(self, **kwargs: Any) -> None:
        row = {f: kwargs.get(f, 0) for f in FRAME_LEDGER_FIELDS}
        self._rows.append(row)

    def write_csv(self, path: Path | str) -> None:
        if not self._rows:
            return
        with open(str(path), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FRAME_LEDGER_FIELDS)
            w.writeheader()
            w.writerows(self._rows)

    def __len__(self) -> int:
        return len(self._rows)
