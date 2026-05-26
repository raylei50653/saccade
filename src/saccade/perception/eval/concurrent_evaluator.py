"""
Concurrent Evaluator — 支援多序列併發的 MOT17 評估

設計原則：
1. 共享 TRT engine 例項（只讀）
2. 每個工作執行緒建立獨立 TRT context + cudaStream
3. 每個工作執行緒獨立 tracker + relinker 狀態
4. ReID：SharedExtractorService 用鎖序列化 GPU 呼叫，省去 N 份 SigLIP2 VRAM

使用方式：
    from saccade.perception.eval.concurrent_evaluator import (
        run_eval_concurrent, SharedExtractorService,
    )

    metrics = run_eval_concurrent(
        engine="models/yolo/yolo26s_960_batch1.engine",
        sequences="MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP",
        max_workers=2,   # sweet spot: 2.5× batch speedup, 83 FPS/stream
    )
"""

# mypy: ignore-errors
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from saccade.perception.detector_trt import (
    TRTYoloDetector,
    ConcurrentDetectorProxy,
)
from saccade.perception.eval.evaluator import run_eval as run_eval_single
from saccade.perception.eval.metrics import run_motmetrics_evaluation


# ============================================================================
# Shared ReID Extractor Service
# ============================================================================


class SharedExtractorService:
    """Thread-safe shared ReID extractor for concurrent workers.

    Wraps a single TRTFeatureExtractor with a mutex so N worker threads share
    one SigLIP2 engine — eliminates N-way VRAM duplication and serialises GPU
    calls to avoid SigLIP2 SM contention.  Lock overhead is <0.01 ms vs 4 ms
    per extraction call (negligible).

    cpp_ptr is intentionally absent so the evaluator uses the Python ReID path,
    which routes through this lock-protected service.
    """

    def __init__(
        self,
        extractor: Any,
        max_batch: int = 64,
        drain_ms: float = 0.0,  # kept for API compatibility; unused
    ) -> None:
        self._ext = extractor
        self._lock = threading.Lock()

        # Mirror read-only attributes expected by the evaluator
        self.feature_dim: int = extractor.feature_dim
        self.input_hw: Tuple[int, int] = extractor.input_hw
        self.max_batch: int = max_batch
        self.device: str = extractor.device
        self.model_type: str = getattr(extractor, "model_type", "siglip2")
        # No cpp_ptr → evaluator uses the Python ReID path

        print("🔗 [SharedExtractorService] started (lock-based, 1 SigLIP2 engine)")

    # ── public API (mirrors TRTFeatureExtractor) ─────────────────────────────

    def extract(
        self,
        input_tensor: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        if input_tensor.size(0) == 0:
            return torch.empty((0, self.feature_dim), device=self.device)
        with self._lock:
            return self._ext.extract(input_tensor, stream)

    def extract_parts_fused(
        self,
        input_tensor: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        with self._lock:
            return self._ext.extract_parts_fused(input_tensor, stream)

    def extract_with_stability(self, input_tensor: torch.Tensor, **kwargs: Any) -> Any:
        with self._lock:
            return self._ext.extract_with_stability(input_tensor, **kwargs)

    def extract_to_cpu(self, input_tensor: torch.Tensor) -> torch.Tensor:
        with self._lock:
            return self._ext.extract_to_cpu(input_tensor)

    def shutdown(self) -> None:
        pass  # nothing to clean up


# ============================================================================
# Concurrent Evaluation Entry Point
# ============================================================================


def run_eval_concurrent(
    engine: str,
    output: str,
    data_root: str,
    split: str,
    sequences: str,
    max_frames: Optional[int],
    conf_threshold: float,
    max_workers: int = 2,
    reid_mode: str = "semantic",
    reid_model: str = "siglip2",
    detector: Any = None,
    extractor: Any = None,
    pose_engine: Optional[str] = None,
    profile_stages: bool = False,
    latency_only: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run MOT17 evaluation concurrently across multiple sequences.

    Args:
        max_workers: Number of concurrent sequence workers.
                     Sweet spot is 2: 2.5× wall-clock speedup, ~83 FPS/stream.
        Other args match run_eval().

    Returns:
        Merged metrics dict.
    """
    seqs = [s.strip() for s in sequences.split(",") if s.strip()]
    if not seqs:
        raise ValueError("No sequences provided")

    print(f"🚀 [ConcurrentEval] {len(seqs)} sequences × {max_workers} workers")
    print(f"   Sequences: {', '.join(seqs)}")

    # Shared YOLO detector
    if detector is None:
        detector = TRTYoloDetector(
            engine_path=engine, device="cuda:0", enable_concurrent=True
        )

    # Shared ReID extractor service (one SigLIP2 engine for all workers)
    extractor_service: Optional[SharedExtractorService] = None
    if reid_mode != "off" and extractor is None:
        from saccade.perception.feature_extractor import TRTFeatureExtractor

        base_extractor = TRTFeatureExtractor(
            engine_path="", model_type=reid_model, max_batch=64
        )
        extractor_service = SharedExtractorService(base_extractor, max_batch=64)
        extractor = extractor_service
    elif extractor is not None and not isinstance(extractor, SharedExtractorService):
        extractor_service = SharedExtractorService(extractor, max_batch=64)
        extractor = extractor_service

    seq_results: Dict[str, Dict[str, Any]] = {}
    seq_times: Dict[str, float] = {}
    futures: Dict[Any, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for seq in seqs:
            future = executor.submit(
                _run_sequence_concurrent,
                seq=seq,
                engine=engine,
                output_root=output,
                data_root=data_root,
                split=split,
                max_frames=max_frames,
                conf_threshold=conf_threshold,
                detector=detector,
                extractor=extractor,
                reid_mode=reid_mode,
                reid_model=reid_model,
                profile_stages=profile_stages,
                latency_only=latency_only,
                kwargs=kwargs,
            )
            futures[future] = seq

        for future in as_completed(futures):
            seq = futures[future]
            try:
                result = future.result()
                seq_results[seq] = result
                seq_times[seq] = result.get("wall_sec", 0)
                status = "✅" if result.get("success") else "❌"
                print(f"   {status} {seq}: {result.get('mota', 'N/A')} MOTA")
            except Exception as e:
                print(f"   ❌ {seq}: Exception - {e}")
                seq_results[seq] = {"success": False, "error": str(e)}

    print("\n" + "=" * 60)
    print("📊 Concurrent Evaluation Summary")
    print("=" * 60)

    overall_metrics = run_motmetrics_evaluation(
        data_root=data_root,
        split=split,
        output=output,
        sequences=sequences,
        detector=kwargs.get("detector", "SDP"),
    )

    if overall_metrics:
        print("\n=== OVERALL METRICS ===")
        for k, v in overall_metrics.items():
            print(f"  {k}: {v}")

    total_frames = sum(
        r.get("result", {}).get("frames", 0) if isinstance(r.get("result"), dict) else 0
        for r in seq_results.values()
        if r.get("success")
    )
    total_wall = sum(seq_times.values())
    if total_frames > 0 and total_wall > 0:
        print(f"\n⏱️  Total wall time: {total_wall:.2f}s")
        print(f"   Total frames: {total_frames}")
        print(f"   Average FPS: {total_frames / total_wall:.2f}")

    return overall_metrics or {}


def _run_sequence_concurrent(
    seq: str,
    engine: str,
    output_root: str,
    data_root: str,
    split: str,
    max_frames: Optional[int],
    conf_threshold: float,
    detector: TRTYoloDetector,
    extractor: Any,
    reid_mode: str,
    reid_model: str,
    profile_stages: bool,
    latency_only: bool,
    kwargs: dict,
) -> Dict[str, Any]:
    """Run a single sequence using a per-sequence ConcurrentDetectorProxy
    and the shared extractor service."""
    start_time = time.perf_counter()
    seq_detector = ConcurrentDetectorProxy(detector, stream_id=seq)
    try:
        result = run_eval_single(
            engine=engine,
            output=f"{output_root}/{seq}",
            data_root=data_root,
            split=split,
            sequences=seq,
            max_frames=max_frames,
            conf_threshold=conf_threshold,
            reid_mode=reid_mode,
            reid_model=reid_model,
            detector=seq_detector,
            extractor=extractor,
            profile_stages=profile_stages,
            latency_only=latency_only,
            **kwargs,
        )
        wall_sec = time.perf_counter() - start_time
        return {"success": True, "wall_sec": wall_sec, "result": result}
    except Exception as e:
        wall_sec = time.perf_counter() - start_time
        return {"success": False, "error": str(e), "wall_sec": wall_sec}
    finally:
        if seq_detector.context is not None:
            seq_detector.context.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Concurrent MOT17 Eval")
    parser.add_argument("--sequences", default="MOT17-02-SDP,MOT17-04-SDP")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--engine", default="models/yolo/yolo26s_960_batch1.engine")
    parser.add_argument("--output", default="results/MOT17_concurrent")
    parser.add_argument("--data-root", default="datasets/MOT17")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    metrics = run_eval_concurrent(
        engine=args.engine,
        output=args.output,
        data_root=args.data_root,
        split=args.split,
        sequences=args.sequences,
        max_frames=None,
        conf_threshold=0.25,
        max_workers=args.max_workers,
    )
