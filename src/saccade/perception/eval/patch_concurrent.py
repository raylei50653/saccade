"""
Patch for Concurrent Evaluation in Evaluator

這個模組提供對 evaluator.py 的 patch 支援，使其能夠在現有流程中加入
per-sequence concurrent 執行能力，而無需大幅重構 evaluator.py。

Patch 策略：
1. 修改 TRTYoloDetector 增加 per-sequence context 管理
2. 修改 run_eval 接受 per-sequence detector（可選）
3. 提供 wrapper 自動為每個序列建立獨立 context

使用方式：
    from saccade.perception.eval.patch_concurrent import enable_concurrent_eval

    # 啟用後，run_eval 自動支援 per-sequence concurrent
    enable_concurrent_eval(max_workers=8)

    # 或手動為每個序列建立 context
    from saccade.perception.detector_trt import TRTYoloDetector
    detector = TRTYoloDetector("models/yolo/xxx.engine", enable_concurrent=True)
    seq_det = detector.get_seq_context("MOT17-02-SDP")
# mypy: ignore-errors
"""

from __future__ import annotations

import threading
from typing import Any

# 全域 concurrent 配置
_concurrent_config = {
    "enabled": False,
    "max_workers": 8,
    "shared_detector": None,
    "context_lock": None,
    "seq_contexts": None,
}


def enable_concurrent_eval(max_workers: int = 8) -> None:
    """
    啟用 concurrent evaluation 模式。

    呼叫後，所有使用 TRTYoloDetector 的地方將支援 per-sequence context。
    """
    _concurrent_config["enabled"] = True
    _concurrent_config["max_workers"] = max_workers
    _concurrent_config["context_lock"] = threading.Lock()
    _concurrent_config["seq_contexts"] = {}
    print(f"🚀 [ConcurrentEval] Enabled with max_workers={max_workers}")


def disable_concurrent_eval() -> None:
    """停用 concurrent evaluation 模式"""
    _concurrent_config["enabled"] = False
    _concurrent_config["seq_contexts"] = {}
    print("⏹️  [ConcurrentEval] Disabled")


def get_seq_context(
    stream_id: str,
    engine_path: str,
) -> Any:
    """
    為指定序列取得獨立 context 的 detector proxy。

    如果已存在則重用，否則建立新的。
    """
    if not _concurrent_config["enabled"]:
        raise RuntimeError(
            "Concurrent eval not enabled. Call enable_concurrent_eval() first."
        )

    with _concurrent_config["context_lock"]:
        if stream_id not in _concurrent_config["seq_contexts"]:
            from saccade.perception.detector_trt import (
                TRTYoloDetector,
                ConcurrentDetectorProxy,
            )

            # 建立共享 detector
            if _concurrent_config["shared_detector"] is None:
                _concurrent_config["shared_detector"] = TRTYoloDetector(
                    engine_path=engine_path,
                    enable_concurrent=True,
                )

            # 建立序列專屬 context
            _concurrent_config["seq_contexts"][stream_id] = ConcurrentDetectorProxy(
                _concurrent_config["shared_detector"],
                stream_id=stream_id,
            )

        return _concurrent_config["seq_contexts"][stream_id]


def release_seq_context(stream_id: str) -> None:
    """釋放指定序列的 context"""
    if not _concurrent_config["enabled"]:
        return

    with _concurrent_config["context_lock"]:
        if stream_id in _concurrent_config["seq_contexts"]:
            del _concurrent_config["seq_contexts"][stream_id]
            print(f"🔄 [ConcurrentEval] Released context for {stream_id}")


def get_shared_detector() -> Any:
    """取得共享的 TRT detector（只讀）"""
    return _concurrent_config["shared_detector"]


def is_enabled() -> bool:
    return _concurrent_config["enabled"]


# ============================================================================
# 自動 patch run_eval 支援 concurrent
# ============================================================================

_original_run_eval = None


def _patch_run_eval() -> None:
    """
    Patch evaluator.run_eval 以支援 concurrent execution。

    Patch 策略：
    - 在 run_eval 開頭檢查是否為 concurrent 模式
    - 如果是，則將 sequences 拆分為 per-sequence 任務
    - 使用 ThreadPoolExecutor 執行
    - 合併結果
    """
    global _original_run_eval

    try:
        from saccade.perception.eval.evaluator import run_eval as _orig

        _original_run_eval = _orig

        def run_eval_concurrent_wrapper(*args, **kwargs):
            """Concurrent wrapper for run_eval"""
            sequences = kwargs.get("sequences", "")
            if not sequences:
                # 非 concurrent 模式，直接呼叫原版
                return _orig(*args, **kwargs)

            seqs = [s.strip() for s in sequences.split(",") if s.strip()]
            if len(seqs) == 1:
                return _orig(*args, **kwargs)

            # 取得 max_workers
            max_workers = kwargs.pop(
                "max_workers", _concurrent_config.get("max_workers", 8)
            )

            print(
                f"🚀 [ConcurrentPatch] Running {len(seqs)} sequences with {max_workers} workers"
            )

            from concurrent.futures import ThreadPoolExecutor, as_completed
            from saccade.perception.eval.metrics import run_motmetrics_evaluation

            results = {}

            # 準備每個序列的參數
            base_kwargs = {
                k: v for k, v in kwargs.items() if k not in ("max_workers", "sequences")
            }

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for seq in seqs:
                    # 為每個序列建立獨立 detector proxy
                    seq_kwargs = base_kwargs.copy()
                    seq_kwargs["sequences"] = seq
                    seq_kwargs["max_workers"] = 1  # 單序列模式

                    # 建立 per-sequence detector
                    engine = base_kwargs.get(
                        "engine", "models/yolo/yolo26s_960_batch1.engine"
                    )
                    seq_detector = get_seq_context(seq, engine)
                    seq_kwargs["detector"] = seq_detector

                    future = executor.submit(_orig, *args, **seq_kwargs)
                    futures[future] = seq

                for future in as_completed(futures):
                    seq = futures[future]
                    try:
                        result = future.result()
                        results[seq] = result
                        print(f"   ✅ {seq} completed")
                    except Exception as e:
                        print(f"   ❌ {seq} failed: {e}")
                        results[seq] = {"error": str(e)}

            # 合併結果：計算整體 metrics
            output = base_kwargs.get("output", "results/MOT17_eval")
            data_root = base_kwargs.get("data_root", "datasets/MOT17")
            split = base_kwargs.get("split", "train")

            overall = run_motmetrics_evaluation(
                data_root=data_root,
                split=split,
                output=output,
                sequences=sequences,
                detector=kwargs.get("detector_suffix", "SDP"),
            )

            return overall

        # 應用 patch
        import saccade.perception.eval.evaluator

        saccade.perception.eval.evaluator.run_eval = run_eval_concurrent_wrapper
        print("🔧 [ConcurrentPatch] run_eval patched successfully")

    except ImportError as e:
        print(f"⚠️  [ConcurrentPatch] Failed to patch: {e}")


# 自動 patch（可選）
# _patch_run_eval()
