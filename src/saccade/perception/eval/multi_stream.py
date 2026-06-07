"""
MultiStreamRunner — 多路並行 MOT17 評估

架構:
    MultiStreamMambaServer (shared):
        BatchedBackbone (C++, batch-N TRT)
        → per-stream Mamba head (event-handoff 到 worker)
        → 每路獨立 StreamState（temporal buffer + GMC warp）

    StreamWorker (per-sequence, ThreadPoolExecutor):
        TorchvisionGpuStreamer → preprocess → submit to server
        → NMS graph (per-stream) → tracker graph (per-stream) → write results

    MultiStreamRunner (orchestrator):
        建立 server + worker pool
        分派序列到獨立執行緒（每路獨立 CUDA stream）
        彙總 metrics
        N=1 fallback: 直接走 run_eval()（不建立 server）
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch

from saccade.perception.eval.evaluator import run_eval as run_eval_single
from saccade.perception.eval.metrics import run_motmetrics_evaluation

_graph_capture_lock = threading.RLock()


def _wave_chunks(items: list[str], size: int) -> list[list[str]]:
    step = max(1, size)
    return [items[i : i + step] for i in range(0, len(items), step)]


def _build_multi_stream_server(
    backbone_engine: str,
    mamba_ckpt: str,
    *,
    mamba_head_script: str = "models/yolo/mamba_head_best.pt",
    yolo_pt_path: str = "models/yolo/yolo26s.pt",
    teacher_ckpt: str = "",
    img_size: int = 640,
    conf_thr: float = 0.001,
    max_det: int = 300,
    max_batch: int = 4,
    flush_timeout_s: float = 0.002,
    device: str = "cuda",
    temporal_T_override: int | None = None,
) -> Any:
    from saccade.perception.multistream_mamba_server import MultiStreamMambaServer

    server = MultiStreamMambaServer(
        backbone_engine=backbone_engine,
        mamba_ckpt=mamba_ckpt,
        yolo_pt_path=yolo_pt_path,
        teacher_ckpt=teacher_ckpt,
        img_size=img_size,
        conf_thr=conf_thr,
        max_det=max_det,
        max_batch=max_batch,
        flush_timeout_s=flush_timeout_s,
        device=device,
        mamba_head_script=mamba_head_script,
        use_cpp_head=True,
        event_handoff=True,
    )
    return server


def _run_sequence_worker(
    seq: str,
    server: Any,
    eval_kwargs: dict[str, Any],
    profile_stages: bool = False,
) -> dict[str, Any]:
    from saccade.perception.multistream_mamba_server import MambaStreamProxy

    proxy = MambaStreamProxy(server, stream_id=seq)
    worker_stream = proxy.stream or torch.cuda.Stream()

    import saccade.perception.eval.evaluator as _eval_mod

    if _eval_mod._worker_init_lock is not None:
        _eval_mod._worker_init_lock.acquire()

    try:
        with torch.cuda.stream(worker_stream):
            result = run_eval_single(
                detector=proxy,
                sequences=seq,
                profile_stages=profile_stages,
                **{
                    k: v
                    for k, v in eval_kwargs.items()
                    if k not in ("detector", "sequences", "profile_stages")
                },
            )
    except BaseException:
        if _eval_mod._worker_init_barrier is not None:
            try:
                _eval_mod._worker_init_barrier.abort()
            except threading.BrokenBarrierError:
                pass
        raise
    finally:
        _eval_mod._release_worker_init_lock()
    return result or {}


class MultiStreamRunner:
    def __init__(
        self,
        *,
        backbone_engine: str = "models/yolo/yolo26s_backbone_640_batch16.engine",
        mamba_ckpt: str = "runs/mamba_gt_vgt_mamba_v14/best.ckpt",
        mamba_head_script: str = "models/yolo/mamba_head_best.pt",
        yolo_pt_path: str = "models/yolo/yolo26s.pt",
        teacher_ckpt: str = "",
        img_size: int = 640,
        conf_thr: float = 0.001,
        max_det: int = 300,
        max_streams: int = 4,
        flush_timeout_s: float = 0.002,
        device: str = "cuda",
        temporal_T_override: int | None = None,
    ) -> None:
        self._max_streams = max(1, max_streams)
        self._server = None
        if self._max_streams > 1:
            self._server = _build_multi_stream_server(
                backbone_engine=backbone_engine,
                mamba_ckpt=mamba_ckpt,
                mamba_head_script=mamba_head_script,
                yolo_pt_path=yolo_pt_path,
                teacher_ckpt=teacher_ckpt,
                img_size=img_size,
                conf_thr=conf_thr,
                max_det=max_det,
                max_batch=self._max_streams,
                flush_timeout_s=flush_timeout_s,
                device=device,
                temporal_T_override=temporal_T_override,
            )
        self._backbone_engine = backbone_engine
        self._mamba_ckpt = mamba_ckpt
        self._img_size = img_size

    @property
    def max_streams(self) -> int:
        return self._max_streams

    @property
    def is_multi(self) -> bool:
        return self._server is not None

    def run_sequences(
        self,
        sequences: list[str],
        eval_kwargs: dict[str, Any],
        *,
        profile_stages: bool = False,
    ) -> dict[str, Any] | None:
        # Drop sequences/detector from eval_kwargs if present (they come from the
        # original argparse namespace and conflict with the explicit args).
        _ek = {
            k: v for k, v in eval_kwargs.items() if k not in ("sequences", "detector")
        }
        if not self.is_multi:
            return run_eval_single(
                sequences=",".join(sequences),
                profile_stages=profile_stages,
                **_ek,
            )

        seq_list = [s.strip() for s in sequences if s.strip()]
        if len(seq_list) <= 1:
            return run_eval_single(
                sequences=",".join(seq_list),
                profile_stages=profile_stages,
                **_ek,
            )

        print(
            f"\n🔀 [MultiStreamRunner] {len(seq_list)} sequences "
            f"on {self._max_streams} concurrent streams "
            f"(backbone={self._backbone_engine}, batch={self._max_streams})"
        )

        total_frames = 0
        for seq in seq_list:
            import configparser

            data_root = _ek.get("data_root", "datasets/MOT17")
            split = _ek.get("split", "train")
            seq_path = Path(data_root) / split / seq
            ini = seq_path / "seqinfo.ini"
            if ini.exists():
                c = configparser.ConfigParser()
                c.read(ini)
                sl = c.getint("Sequence", "seqLength", fallback=0)
                mf = _ek.get("max_frames")
                if mf is not None:
                    sl = min(int(mf), sl)
                total_frames += sl

        start_t = time.perf_counter()
        results: dict[str, dict[str, Any]] = {}
        server = self._server

        import saccade.perception.eval.evaluator as _eval_mod

        _eval_mod._graph_capture_lock = _graph_capture_lock
        _eval_mod._worker_init_lock = threading.Lock()
        try:
            for wave_idx, wave in enumerate(
                _wave_chunks(seq_list, self._max_streams), start=1
            ):
                if len(seq_list) > self._max_streams:
                    print(f"🌊 [MultiStreamRunner] Wave {wave_idx}: {', '.join(wave)}")
                _eval_mod._worker_init_lock = threading.Lock()
                _eval_mod._worker_init_barrier = (
                    threading.Barrier(len(wave)) if len(wave) > 1 else None
                )
                with ThreadPoolExecutor(max_workers=len(wave)) as executor:
                    futures = {
                        executor.submit(
                            _run_sequence_worker,
                            seq,
                            server,
                            _ek,
                            profile_stages,
                        ): seq
                        for seq in wave
                    }
                    for fut in as_completed(futures):
                        seq = futures[fut]
                        try:
                            results[seq] = fut.result()
                            print(f"✅ {seq} done")
                        except Exception as exc:
                            print(f"❌ {seq} failed: {exc}")
                            results[seq] = {"error": str(exc)}
                _eval_mod._worker_init_lock = None
                _eval_mod._worker_init_barrier = None
        finally:
            _eval_mod._graph_capture_lock = None
            _eval_mod._worker_init_lock = None
            _eval_mod._worker_init_barrier = None

        elapsed = time.perf_counter() - start_t
        fps_str = ""
        if total_frames > 0 and elapsed > 0:
            fps_str = (
                f" (Aggregated Throughput: {total_frames / elapsed:.2f} FPS "
                f"across {total_frames} frames)"
            )
        print(
            f"\n📈 [MultiStreamRunner] Completed {len(seq_list)} sequences "
            f"in {elapsed:.2f}s{fps_str}"
        )

        all_seqs = ",".join(seq_list)
        metrics = run_motmetrics_evaluation(
            data_root=_ek.get("data_root", "datasets/MOT17"),
            split=_ek.get("split", "train"),
            output=str(_ek.get("output", "results/MOT17_eval")),
            sequences=all_seqs,
            max_frames=_ek.get("max_frames"),
        )
        if metrics:
            print("\n=== OVERALL METRICS ===")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
        return metrics

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.shutdown()

    def __del__(self) -> None:
        self.shutdown()
