#!/usr/bin/env python3
"""
Concurrent MOT17 Evaluator — batch-fused TRT (threading + batch4 engine)

GPU 資源配置：
  單一 batch4 YOLO engine 共享；每個 worker thread 各有獨立 tracker + ReID
  N 個 worker frame 合成一個 batch4 TRT call → GPU SM 利用率最大化

Worker 甜蜜點（RTX 3090 / 24 GB，yolo26s batch4）：
  workers=1   ~155 FPS（無法 fill batch，drain_ms 超時後單幀送出）
  workers=2   ~150 FPS/路  ← 推薦；batch 常常填滿 2 幀，detect latency 略升
  workers=4   ~140 FPS/路  ← batch 穩定填滿 4 幀，最大化 TRT 效率
  workers=7   ~110 FPS/路  ← ReID lock 開始成為瓶頸

使用方式：
    # 7 個 SDP 序列（推薦）
    uv run python scripts/eval/concurrent_mot17.py \\
        --preset speed --detector SDP

    # 比較 sequential vs concurrent
    uv run python scripts/eval/concurrent_mot17.py \\
        --preset speed --detector SDP --mode compare

    # 手動指定序列
    uv run python scripts/eval/concurrent_mot17.py \\
        --sequences MOT17-02-SDP,MOT17-04-SDP --max-workers 2
"""
# status: stable

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import torch

from saccade.perception.detector_trt import BatchingTRTDetector
from saccade.perception.eval.concurrent_evaluator import SharedExtractorService
from saccade.perception.eval.metrics import run_motmetrics_evaluation


# ── helpers ──────────────────────────────────────────────────────────────────


def _pct(v: object, default: float = 0.0) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return float(v.rstrip("%"))
    return default


# ── worker functions (must be top-level for pickling) ────────────────────────


def run_sequence_single(
    seq: str,
    engine: str,
    output: str,
    data_root: str,
    split: str,
    conf_threshold: float,
    reid_mode: str = "semantic",
    reid_model: str = "siglip2",
    profile_stages: bool = False,
    latency_only: bool = False,
    **kwargs,
) -> dict:
    """Sequential mode — single process, no parallelism."""
    from saccade.perception.eval.evaluator import run_eval

    start = time.perf_counter()
    try:
        result = run_eval(
            engine=engine,
            output=output,
            data_root=data_root,
            split=split,
            sequences=seq,
            max_frames=None,
            conf_threshold=conf_threshold,
            reid_mode=reid_mode,
            reid_model=reid_model,
            profile_stages=profile_stages,
            latency_only=latency_only,
            **kwargs,
        )
        wall = time.perf_counter() - start
        return {
            "seq": seq,
            "success": True,
            "mota": _pct(result.get("MOTA")) if result else 0.0,
            "idf1": _pct(result.get("IDF1")) if result else 0.0,
            "wall_sec": wall,
            "result": result,
        }
    except Exception as e:
        return {
            "seq": seq,
            "success": False,
            "error": str(e),
            "wall_sec": time.perf_counter() - start,
        }


def run_sequence_batched(
    seq: str,
    engine: str,
    output: str,
    data_root: str,
    split: str,
    conf_threshold: float,
    batcher: BatchingTRTDetector,
    shared_extractor: Optional[SharedExtractorService] = None,
    reid_mode: str = "semantic",
    reid_model: str = "siglip2",
    profile_stages: bool = False,
    latency_only: bool = False,
    **kwargs: Any,
) -> dict:
    """Thread worker — uses a per-sequence BatchedDetectorProxy backed by the
    shared batch-fused TRT detector.  Each thread owns its own tracker."""
    from saccade.perception.eval.evaluator import run_eval

    start = time.perf_counter()
    proxy = batcher.make_proxy()
    # Per-sequence CUDA stream. All worker GPU work (preproc, postprocess, ReID)
    # runs on this stream so sibling workers don't serialize through the default
    # stream. The batcher reads torch.cuda.current_stream() inside detect_raw to
    # set up the frame_event / infer_event handshake on the right stream.
    worker_stream = torch.cuda.Stream(device=batcher._device)
    try:
        with torch.cuda.stream(worker_stream):
            result = run_eval(
                engine=engine,
                output=output,
                data_root=data_root,
                split=split,
                sequences=seq,
                max_frames=None,
                conf_threshold=conf_threshold,
                reid_mode=reid_mode,
                reid_model=reid_model,
                detector=proxy,
                extractor=shared_extractor,
                profile_stages=profile_stages,
                latency_only=latency_only,
                **kwargs,
            )
        wall = time.perf_counter() - start
        return {
            "seq": seq,
            "success": True,
            "mota": _pct(result.get("MOTA")) if result else 0.0,
            "idf1": _pct(result.get("IDF1")) if result else 0.0,
            "wall_sec": wall,
            "result": result,
        }
    except Exception as e:
        import traceback

        return {
            "seq": seq,
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "wall_sec": time.perf_counter() - start,
        }


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concurrent MOT17 Evaluation (multiprocessing)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sequences",
        default="",
        help="Comma-separated sequence names (empty = all in split)",
    )
    parser.add_argument(
        "--preset",
        choices=["speed", "baseline", "accuracy", "mamba_whole_graph"],
        default=None,
        help="Built-in preset config",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help=(
            "Number of concurrent sequence threads. "
            "All share one batch4 YOLO engine. "
            "4 fills the batch optimally; try 7 for max throughput."
        ),
    )
    parser.add_argument(
        "--batch-engine",
        default="models/yolo/yolo26s_960_batch4.engine",
        help="Path to dynamic-batch TRT engine used for batched inference.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for the batch-fused TRT detector.",
    )
    parser.add_argument(
        "--drain-ms",
        type=float,
        default=2.0,
        help="Max wait (ms) before firing a partial batch.",
    )
    parser.add_argument(
        "--mode",
        choices=["sequential", "concurrent", "compare"],
        default="concurrent",
        help="Execution mode",
    )
    parser.add_argument("--output", default="results/MOT17_concurrent")
    parser.add_argument("--data-root", default="datasets/MOT17")
    parser.add_argument("--split", default="train")
    parser.add_argument("--detector", choices=["SDP", "DPM", "FRCNN"], default=None)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--reid-mode", default="semantic")
    parser.add_argument("--profile-stages", action="store_true")

    args = parser.parse_args()

    import yaml

    # Load config: preset > mot17_baseline.yaml fallback
    preset_config: dict = {}
    if args.preset:
        preset_path = Path("configs/presets") / f"{args.preset}.yaml"
        if preset_path.exists():
            with open(preset_path) as f:
                preset_config = yaml.safe_load(f) or {}
            print(f"📋 Loaded preset: {args.preset}")
    else:
        fallback = Path("configs/mot17_baseline.yaml")
        if fallback.exists():
            with open(fallback) as f:
                preset_config = yaml.safe_load(f) or {}
            print(f"📋 Loaded default config: {fallback}")

    # Auto-discover sequences
    if not args.sequences:
        data_dir = Path(args.data_root) / args.split
        if data_dir.exists():
            suffix = f"-{args.detector}" if args.detector else ""
            args.sequences = ",".join(
                sorted(
                    d.name
                    for d in data_dir.iterdir()
                    if d.is_dir() and (not suffix or d.name.endswith(suffix))
                )
            )

    if not args.sequences:
        print("❌ No sequences found")
        sys.exit(1)

    seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]
    print(
        f"📊 Sequences: {len(seqs)} ({', '.join(seqs[:3])}{'...' if len(seqs) > 3 else ''})"
    )
    print(f"🚀 Max workers: {args.max_workers}")
    print(f"🎯 Mode: {args.mode}")

    # Pop keys that are passed as explicit kwargs to avoid duplicate conflicts
    engine = preset_config.pop("engine", "models/yolo/yolo26s_960_batch1.engine")
    reid_model = preset_config.pop("reid_model", "siglip2")
    preset_config.pop("conf_threshold", None)
    preset_config.pop("reid_mode", None)
    print(f"🔧 Engine: {engine}")

    # ── shared batch-fused detector (concurrent modes only) ───────────────────
    batcher: Optional[BatchingTRTDetector] = None
    shared_extractor: Optional[SharedExtractorService] = None
    if args.mode in ("concurrent", "compare"):
        batcher = BatchingTRTDetector(
            engine_path=args.batch_engine,
            batch_size=args.batch_size,
            drain_ms=args.drain_ms,
        )
        if args.reid_mode != "off":
            from saccade.perception.feature_extractor import TRTFeatureExtractor

            base_ext = TRTFeatureExtractor(
                engine_path="", model_type=reid_model, max_batch=64
            )
            shared_extractor = SharedExtractorService(base_ext, max_batch=64)

    # ── sequential ────────────────────────────────────────────────────────────
    results: dict = {}

    if args.mode == "sequential":
        print("\n" + "=" * 60)
        print("🔄 Sequential Mode")
        print("=" * 60)
        for seq in seqs:
            result = run_sequence_single(
                seq=seq,
                engine=engine,
                output=args.output,
                data_root=args.data_root,
                split=args.split,
                conf_threshold=args.conf_threshold,
                reid_mode=args.reid_mode,
                reid_model=reid_model,
                profile_stages=args.profile_stages,
                **preset_config,
            )
            results[seq] = result
            status = "✅" if result["success"] else "❌"
            print(
                f"   {status} {seq}: MOTA={result.get('mota', 0):.1f}% wall={result['wall_sec']:.2f}s"
            )

    # ── concurrent (batch-fused threading) ────────────────────────────────────
    elif args.mode == "concurrent":
        print("\n" + "=" * 60)
        print("⚡ Concurrent Mode (batch-fused TRT)")
        print("=" * 60)
        assert batcher is not None
        _run_concurrent(
            seqs, engine, args, batcher, shared_extractor, preset_config, results
        )

    # ── compare ───────────────────────────────────────────────────────────────
    elif args.mode == "compare":
        print("\n" + "=" * 60)
        print("📊 Sequential Mode")
        print("=" * 60)
        for seq in seqs:
            result = run_sequence_single(
                seq=seq,
                engine=engine,
                output=args.output,
                data_root=args.data_root,
                split=args.split,
                conf_threshold=args.conf_threshold,
                reid_mode=args.reid_mode,
                reid_model=reid_model,
                profile_stages=args.profile_stages,
                **preset_config,
            )
            results[f"{seq}_seq"] = result
            status = "✅" if result["success"] else "❌"
            print(
                f"   {status} {seq}: MOTA={result.get('mota', 0):.1f}% wall={result['wall_sec']:.2f}s"
            )

        print("\n" + "=" * 60)
        print("⚡ Concurrent Mode (batch-fused TRT)")
        print("=" * 60)
        assert batcher is not None
        conc: dict = {}
        _run_concurrent(
            seqs, engine, args, batcher, shared_extractor, preset_config, conc
        )
        for seq, r in conc.items():
            results[f"{seq}_concurrent"] = r

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)

    if args.mode == "compare":
        print(f"{'Sequence':<20} {'Mode':<14} {'MOTA':<8} {'IDF1':<8} {'Wall (s)':<10}")
        print("-" * 62)
        for seq in seqs:
            sr = results.get(f"{seq}_seq", {})
            cr = results.get(f"{seq}_concurrent", {})
            if sr.get("success"):
                print(
                    f"{seq:<20} {'sequential':<14} "
                    f"{sr.get('mota', 0):<8.1f} {sr.get('idf1', 0):<8.1f} "
                    f"{sr['wall_sec']:<10.2f}"
                )
            if cr.get("success"):
                speedup = (
                    sr.get("wall_sec", 0) / cr["wall_sec"]
                    if sr.get("wall_sec", 0) > 0 and cr.get("wall_sec", 0) > 0
                    else 0
                )
                print(
                    f"{seq:<20} {'concurrent':<14} "
                    f"{cr.get('mota', 0):<8.1f} {cr.get('idf1', 0):<8.1f} "
                    f"{cr['wall_sec']:<10.2f} (×{speedup:.2f})"
                )
    else:
        succ = [r for r in results.values() if r.get("success")]
        total_wall = sum(r.get("wall_sec", 0) for r in succ)
        total_frames = sum(
            r.get("result", {}).get("frames", 0)
            if isinstance(r.get("result"), dict)
            else 0
            for r in succ
        )
        print(f"Total sequences: {len(succ)}")
        print(f"Total wall time: {total_wall:.2f}s")
        if total_frames > 0 and total_wall > 0:
            print(f"Average FPS: {total_frames / total_wall:.2f}")

    # Cleanup shared resources
    if shared_extractor is not None:
        shared_extractor.shutdown()
    if batcher is not None:
        batcher.shutdown()

    # Overall MOT metrics
    if results:
        output_dir = f"{args.output}/merged"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print("\n📈 Computing overall MOT metrics...")
        overall = run_motmetrics_evaluation(
            data_root=args.data_root,
            split=args.split,
            output=output_dir,
            sequences=args.sequences,
            detector=args.detector or "SDP",
        )
        if overall:
            print("\n=== OVERALL METRICS ===")
            for k, v in overall.items():
                print(f"  {k}: {v}")


def _run_concurrent(
    seqs: list,
    engine: str,
    args: argparse.Namespace,
    batcher: BatchingTRTDetector,
    shared_extractor: Optional[SharedExtractorService],
    extra_kwargs: dict,
    results: dict,
) -> None:
    """Submit all sequences to ThreadPoolExecutor; each thread gets its own
    BatchedDetectorProxy backed by the shared batch-fused TRT detector."""
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                run_sequence_batched,
                seq=seq,
                engine=engine,
                output=args.output,
                data_root=args.data_root,
                split=args.split,
                conf_threshold=args.conf_threshold,
                batcher=batcher,
                shared_extractor=shared_extractor,
                reid_mode=args.reid_mode,
                profile_stages=args.profile_stages,
                **extra_kwargs,
            ): seq
            for seq in seqs
        }
        for future in as_completed(futures):
            seq = futures[future]
            result = future.result()
            results[seq] = result
            status = "✅" if result["success"] else "❌"
            if result["success"]:
                print(
                    f"   {status} {seq}: MOTA={result['mota']:.1f}% wall={result['wall_sec']:.2f}s"
                )
            else:
                print(f"   {status} {seq}: {result.get('error', '?')}")
                if "traceback" in result:
                    print(result["traceback"])


if __name__ == "__main__":
    main()
