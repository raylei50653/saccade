#!/usr/bin/env python3
# mypy: ignore-errors
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
build_path = project_root / "build"
if build_path.exists():
    sys.path.insert(0, str(build_path))

# MUST IMPORT THIS BEFORE torchvision TO AVOID LIBJPEG CONFLICT
from saccade.perception.detector_trt import TRTYoloDetector  # noqa: F401, E402
from saccade.perception.eval.runner import run_eval  # noqa: E402
from saccade.perception.eval.evaluator import run_eval_cpp  # noqa: E402

import yaml  # noqa: E402
from mot17_args import build_parser  # noqa: E402


def _load_config_defaults(project_root: Path) -> dict:
    """Pre-parse --config / --preset / --module-* and return merged defaults dict.

    Priority (lowest → highest):
      argparse defaults < config file < module YAMLs < preset < CLI flags.

    Only yaml.safe_load is used; no dynamic import or eval.
    """
    _MODULE_FLAGS = [
        "module_detection",
        "module_geometry",
        "module_motion",
        "module_reid",
        "module_semantic",
        "module_trigger",
        "module_lifecycle",
    ]

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre.add_argument("--preset", default=None)
    for flag in _MODULE_FLAGS:
        pre.add_argument(f"--{flag.replace('_', '-')}", default=None, dest=flag)
    pre_args, _ = pre.parse_known_args()

    defaults: dict = {}

    # 1. Core config file (or fallback baseline)
    if pre_args.config:
        config_path = Path(pre_args.config)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        with config_path.open() as f:
            loaded = yaml.safe_load(f) or {}
        defaults.update(loaded)
    elif not pre_args.preset:
        fallback = project_root / "configs" / "mot17_baseline.yaml"
        if fallback.exists():
            with fallback.open() as f:
                loaded = yaml.safe_load(f) or {}
            defaults.update(loaded)

    # 2. Per-module YAML files (opt-in)
    for flag in _MODULE_FLAGS:
        path_str = getattr(pre_args, flag, None)
        if path_str:
            module_path = Path(path_str)
            if not module_path.is_absolute():
                module_path = Path.cwd() / module_path
            with module_path.open() as f:
                loaded = yaml.safe_load(f) or {}
            defaults.update(loaded)

    # 3. Preset (highest priority among file-based configs)
    if pre_args.preset:
        preset_path = project_root / "configs" / "presets" / f"{pre_args.preset}.yaml"
        with preset_path.open() as f:
            loaded = yaml.safe_load(f) or {}
        defaults.update(loaded)

    return defaults


if __name__ == "__main__":
    parser = build_parser()
    config_defaults = _load_config_defaults(project_root)
    if config_defaults:
        parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    if args.detector and not args.sequences:
        from pathlib import Path as _Path

        data_root = _Path(args.data_root)
        split_dir = data_root / args.split
        if split_dir.exists():
            args.sequences = ",".join(
                sorted(
                    d.name
                    for d in split_dir.iterdir()
                    if d.is_dir() and d.name.endswith(f"-{args.detector}")
                )
            )
    _MODULE_KEYS = {
        "module_detection",
        "module_geometry",
        "module_motion",
        "module_reid",
        "module_semantic",
        "module_trigger",
        "module_lifecycle",
    }
    eval_kwargs = {k: v for k, v in vars(args).items() if k not in _MODULE_KEYS}

    if getattr(args, "cpp_threads", 0) > 0:
        # ── C++ multi-threaded path ───────────────────────────────────────────
        n = args.cpp_threads
        print(f"🚀 [C++ EvaluatorPool] Running with {n} threads")
        metrics = run_eval_cpp(
            engine=args.engine,
            output=args.output,
            data_root=args.data_root,
            split=args.split,
            sequences=args.sequences or "",
            n_threads=n,
            **{
                k: v
                for k, v in eval_kwargs.items()
                if k
                not in {
                    "engine",
                    "output",
                    "data_root",
                    "split",
                    "sequences",
                    "workbench",
                    "threads",
                    "cpp_threads",
                }
            },
        )
        if metrics:
            print("\n=== OVERALL METRICS ===")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
    elif args.workbench and args.threads > 1:
        from saccade.perception.detector_trt import BatchingTRTDetector
        from concurrent.futures import ThreadPoolExecutor

        from saccade.perception.eval.config import parse_eval_config as _pec

        _cfg_tmp = _pec(
            output=args.output,
            data_root=args.data_root,
            split=args.split,
            sequences=args.sequences or "",
            conf_threshold=args.conf_threshold,
            reid_mode="off",
            reid_model="siglip2",
            profile_stages=False,
            kwargs={},
        )
        seqs = _cfg_tmp.seqs
        print(
            f"🚀 [Workbench] Running {len(seqs)} sequences with {args.threads} threads using BatchingTRTDetector"
        )

        # Auto-select a batch engine compatible with the requested thread count.
        # Priority: exact batchN match → largest available batch engine → batch_size=1.
        # _eff_batch is always derived from the selected engine filename so passing
        # --engine .../yolo26s_960_batch4.engine directly works correctly.
        from pathlib import Path as _Path
        import re as _re

        def _engine_batch(path: "_Path") -> int:
            m = _re.search(r"_batch(\d+)", path.name)
            return int(m.group(1)) if m else 1

        _ep = _Path(args.engine)
        _exact = _ep.parent / _ep.name.replace("_batch1", f"_batch{args.threads}")
        if _exact.exists():
            _engine_path = str(_exact)
        elif _engine_batch(_ep) > 1:
            # User passed a batch engine directly (e.g. yolo26s_960_batch4.engine)
            _engine_path = str(_ep)
        else:
            # Scan siblings for any batch engine; pick the largest one available.
            _candidates = []
            for _f in _ep.parent.glob(f"{_ep.stem.split('_batch')[0]}*.engine"):
                _b = _engine_batch(_f)
                if _b > 1:
                    _candidates.append((_b, _f))
            if _candidates:
                _, _f = max(_candidates)
                _engine_path = str(_f)
                print(
                    f"  No batch-{args.threads} engine; using batch-{_engine_batch(_f)} engine"
                )
            else:
                _engine_path = args.engine
                print(
                    "  No compatible batch engine found; running batch_size=1 (sequential)"
                )
        _eff_batch = min(_engine_batch(_Path(_engine_path)), args.threads)
        print(f"  Engine: {_engine_path}  batch_size={_eff_batch}")
        batcher = BatchingTRTDetector(_engine_path, batch_size=_eff_batch)

        def run_single_seq(seq_name):
            proxy = batcher.make_proxy()
            kwargs = eval_kwargs.copy()
            kwargs["sequences"] = seq_name
            kwargs["detector"] = proxy
            kwargs["workbench"] = True  # ensure evaluator uses the workbench path
            return run_eval(**kwargs)

        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            results = list(executor.map(run_single_seq, seqs))

        # We don't have a good way to merge overall metrics here since motmetrics
        # aggregation is complex, but for parity check of per-sequence results it's fine.
        # Actually, evaluator.py prints per-sequence results anyway.
        metrics = (
            None  # Overall metrics print won't happen, but sequences already printed.
        )
        print("\n✅ Concurrent evaluation finished.")
    else:
        if args.preset == "motr" or args.detector == "MOTR":
            print("\n🚀 [MOTR] Running Temporal YOLO Hybrid Tracking Pipeline")
            from saccade.perception.eval.config import parse_eval_config
            from saccade.perception.temporal_yolo import MOTREvaluator

            cfg = parse_eval_config(
                output=args.output,
                data_root=args.data_root,
                split=args.split,
                sequences=args.sequences or "",
                conf_threshold=args.conf_threshold,
                reid_mode="off",
                reid_model="",
                profile_stages=False,
                kwargs=eval_kwargs,
            )
            for seq in cfg.seqs:
                print(f"  Evaluating MOTR on sequence: {seq}")
                seq_dir = Path(args.data_root) / args.split / seq
                evaluator = MOTREvaluator(cfg, seq, seq_dir)
                try:
                    evaluator.evaluate()
                except NotImplementedError as e:
                    print(f"  [MOTR] {e}")
            metrics = None
        else:
            metrics = run_eval(**eval_kwargs)

        if metrics:
            print("\n=== OVERALL METRICS ===")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
