"""
Multi-stream Mamba benchmark.

Treats N MOT17 sequences as N concurrent video streams sharing ONE
MultiStreamMambaServer (cross-stream batched backbone + head). Each stream runs
the real `run_eval` via a MambaStreamProxy, so MOTA / IDF1 / coordinate handling
/ tracker params are identical to the single-stream `mamba_optimal` path — the
only difference is that detection is coalesced across streams.

Reports wall-clock aggregate FPS, the realized batch-size histogram, and the
full MOT metrics. Compare batched (--max-batch 4) vs no-batch (--max-batch 1)
at the same --max-workers to isolate the cross-stream batching speedup.

Usage:
    uv run python scripts/benchmarks/multistream_mamba.py \
        --sequences MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP \
        --max-workers 4 --max-batch 4
    # baseline (no coalescing):
    uv run python scripts/benchmarks/multistream_mamba.py ... --max-batch 1
"""
# status: diagnostic

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch  # noqa: F401  # load cuBLAS etc under normal dlopen flags first

# The TorchScript C++ head dispatches selective_scan_fwd back to the
# saccade_tracking_ext CUDA op; load it (RTLD_GLOBAL) before torchvision/PIL so
# that lazy import does not hit a libtiff symbol clash. Restore the dlopen flags
# afterwards so libraries loaded later (e.g. cuBLAS) initialise normally.
_old_dlflags = sys.getdlopenflags()
sys.setdlopenflags(_old_dlflags | ctypes.RTLD_GLOBAL)
import saccade_tracking_ext  # noqa: F401, E402  # must load before torchvision

sys.setdlopenflags(_old_dlflags)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import yaml  # noqa: E402

from saccade.perception.eval.evaluator import run_eval as run_eval_single  # noqa: E402
from saccade.perception.eval.metrics import run_motmetrics_evaluation
from saccade.perception.multistream_mamba_server import (
    MambaStreamProxy,
    MultiStreamMambaServer,
)

# Preset keys consumed when building the detector/server, not by run_eval.
_DETECTOR_KEYS = {
    "mamba_ckpt",
    "mamba_teacher_ckpt",
    "fpn_backbone_engine",
    "use_cuda_graph",
}


def load_preset(name: str) -> dict[str, Any]:
    path = Path("configs/presets") / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-stream Mamba benchmark")
    ap.add_argument(
        "--sequences", default="MOT17-02-SDP,MOT17-04-SDP,MOT17-05-SDP,MOT17-09-SDP"
    )
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument(
        "--max-batch", type=int, default=4, help="1 = no cross-stream coalescing"
    )
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--preset", default="mamba_optimal")
    ap.add_argument(
        "--backbone-engine",
        default="models/yolo/yolo26s_backbone_640_batch4.engine",
        help="Batch-N TRT backbone engine for the shared server",
    )
    ap.add_argument("--output", default="results/MOT17_multistream")
    ap.add_argument("--data-root", default="datasets/MOT17")
    ap.add_argument("--split", default="train")
    ap.add_argument("--flush-us", type=float, default=2000.0)
    ap.add_argument(
        "--py-head",
        action="store_true",
        help="Use the GIL-bound Python Mamba head instead of the GIL-free C++ head.",
    )
    ap.add_argument(
        "--event-handoff",
        action="store_true",
        help="Server does backbone only; each stream runs the head + postprocess "
        "+ tracker on its own CUDA stream (event handoff). Needs the C++ head.",
    )
    args = ap.parse_args()

    preset = load_preset(args.preset)
    eval_kwargs = {k: v for k, v in preset.items() if k not in _DETECTOR_KEYS}
    eval_kwargs["engine"] = "mamba"
    eval_kwargs.setdefault("tiling", "native_640")
    eval_kwargs.setdefault("reid_mode", "off")
    eval_kwargs.setdefault("conf_threshold", 0.001)

    mamba_ckpt = preset.get("mamba_ckpt", "runs/mamba_gt_vgt_mamba_v14/best.ckpt")

    seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]
    print(
        f"🚀 [MultiStreamMamba] {len(seqs)} streams × {args.max_workers} workers "
        f"| max_batch={args.max_batch}"
    )
    print(f"   backbone={args.backbone_engine}")
    print(f"   sequences: {', '.join(seqs)}")

    print(f"   head: {'Python (GIL-bound)' if args.py_head else 'C++ (GIL-free)'}")
    print(f"   event-handoff (per-stream CUDA stream): {args.event_handoff}")
    server = MultiStreamMambaServer(
        backbone_engine=args.backbone_engine,
        mamba_ckpt=mamba_ckpt,
        img_size=640,
        conf_thr=0.001,
        max_det=300,
        max_batch=args.max_batch,
        flush_timeout_s=args.flush_us / 1e6,
        use_cpp_head=not args.py_head,
        event_handoff=args.event_handoff,
    )

    def count_frames(seq: str) -> int:
        n = len(list((Path(args.data_root) / args.split / seq / "img1").glob("*.jpg")))
        return min(n, args.max_frames) if args.max_frames else n

    def run_one(seq: str) -> dict[str, Any]:
        proxy = MambaStreamProxy(server, seq)
        # In event-handoff mode, route this worker thread's GPU work (head +
        # postprocess + tracker) onto the proxy's dedicated CUDA stream so it
        # overlaps with other streams instead of serializing on the default one.
        if proxy.stream is not None:
            torch.cuda.set_stream(proxy.stream)
        t0 = time.perf_counter()
        result = run_eval_single(
            output=f"{args.output}/{seq}",
            data_root=args.data_root,
            split=args.split,
            sequences=seq,
            max_frames=args.max_frames,
            detector=proxy,
            **eval_kwargs,
        )
        return {"seq": seq, "wall": time.perf_counter() - t0, "result": result}

    wall_start = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(run_one, s): s for s in seqs}
        for fut in as_completed(futs):
            seq = futs[fut]
            try:
                results.append(fut.result())
                print(f"   ✅ {seq} done")
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                print(f"   ❌ {seq}: {e}")
    wall_total = time.perf_counter() - wall_start
    total_frames = sum(count_frames(s) for s in seqs)

    print("\n" + "=" * 60)
    print("📊 Multi-Stream Mamba Benchmark")
    print("=" * 60)
    print(f"batch-size histogram (index=size): {server.batch_hist}")
    coalesced = sum(i * c for i, c in enumerate(server.batch_hist))
    calls = sum(server.batch_hist)
    if calls:
        print(f"mean batch size: {coalesced / calls:.2f}  ({calls} backbone calls)")
    print(f"wall time (concurrent): {wall_total:.2f}s")
    print(f"total frames: {total_frames}")
    if total_frames and wall_total:
        print(f"aggregate throughput: {total_frames / wall_total:.1f} FPS")
        print(
            f"per-stream throughput:  {total_frames / wall_total / max(len(seqs), 1):.1f} FPS"
        )

    print("\n=== PER-STREAM METRICS (parity vs single-stream baseline) ===")
    keys = ("MOTA", "IDF1", "HOTA", "DetA", "AssA", "IDs")
    for r in sorted(results, key=lambda x: x["seq"]):
        m = r.get("result")
        if isinstance(m, dict) and m:
            line = "  ".join(f"{k}={m[k]}" for k in keys if k in m)
            print(f"  {r['seq']:16s} {line}  (wall {r['wall']:.1f}s)")
        else:
            print(f"  {r['seq']:16s} (no metrics returned)")

    server.shutdown()

    print("\n=== AGGREGATE MOT METRICS ===")
    metrics = run_motmetrics_evaluation(
        data_root=args.data_root,
        split=args.split,
        output=args.output,
        sequences=args.sequences,
        detector="SDP",
    )
    if metrics:
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    else:
        print("  (aggregate unavailable; rely on per-stream metrics above)")


if __name__ == "__main__":
    main()
