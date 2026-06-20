#!/usr/bin/env python3
# mypy: ignore-errors
import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
build_path = Path(os.environ.get("SACCADE_BUILD_PATH", project_root / "build"))
if build_path.exists():
    sys.path.insert(0, str(build_path))

# MUST IMPORT THIS BEFORE torchvision TO AVOID LIBJPEG CONFLICT
from saccade.perception.detector_trt import TRTYoloDetector  # noqa: F401, E402
from saccade.perception.eval.runner import run_eval  # noqa: E402
from saccade.perception.eval.evaluator import run_eval_cpp  # noqa: E402
from mlflow_logger import log_eval_run  # noqa: E402

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
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile on detection head and postprocess",
    )
    config_defaults = _load_config_defaults(project_root)
    if config_defaults:
        parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    if args.double_buffer:
        if args.detect_barrier not in {None, "event"}:
            parser.error("--double-buffer requires --detect-barrier event")
        os.environ["SACCADE_DOUBLE_BUFFER"] = "1"
        os.environ["SACCADE_DETECT_BARRIER"] = "event"
    elif args.detect_barrier is not None:
        os.environ["SACCADE_DETECT_BARRIER"] = args.detect_barrier
    if not getattr(args, "no_gpu_decode", False):
        os.environ["SACCADE_GPU_DECODE"] = "1"
    if os.environ.get("SACCADE_NV12_BUFFER") == "1":
        _nv_lib = project_root / ".venv/lib/python3.12/site-packages/nvidia/cu13/lib"
        _ld_preload = os.environ.get("LD_PRELOAD", "")
        _cublas = str(_nv_lib / "libcublas.so.13")
        _cublaslt = str(_nv_lib / "libcublasLt.so.13")
        _preloads = [p for p in _ld_preload.split(":") if p]
        _need_restart = False
        if _cublas not in _preloads:
            _preloads.insert(0, _cublas)
            _need_restart = True
        if _cublaslt not in _preloads:
            _preloads.insert(0, _cublaslt)
            _need_restart = True
        if _need_restart and not os.environ.get("_SACCADE_NV12_RESTARTED"):
            os.environ["LD_PRELOAD"] = ":".join(_preloads)
            os.environ["_SACCADE_NV12_RESTARTED"] = "1"
            os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)
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

    if getattr(args, "processes", 0) > 0 and args.sequences:
        import subprocess
        import time
        import configparser
        from concurrent.futures import ThreadPoolExecutor

        seq_list = [s.strip() for s in args.sequences.split(",") if s.strip()]
        if len(seq_list) > 1:
            print(
                f"\n🔥 [Multi-Process Launcher] Spawning {len(seq_list)} processes (max {args.processes} concurrent)"
            )

            # Parse frame counts to calculate aggregated FPS later
            total_frames = 0
            for seq in seq_list:
                seq_path = Path(args.data_root) / args.split / seq
                if (seq_path / "seqinfo.ini").exists():
                    config = configparser.ConfigParser()
                    config.read(seq_path / "seqinfo.ini")
                    seq_len = config.getint("Sequence", "seqLength", fallback=0)
                    if getattr(args, "max_frames", None):
                        seq_len = min(args.max_frames, seq_len)
                    total_frames += seq_len

            start_t = time.perf_counter()

            # Remove --processes and --sequences from sys.argv to build child commands
            cmd_base = [sys.executable, __file__]
            i = 1
            while i < len(sys.argv):
                arg = sys.argv[i]
                if arg == "--processes":
                    i += 2
                elif arg.startswith("--processes="):
                    i += 1
                elif arg == "--sequences":
                    i += 2
                elif arg.startswith("--sequences="):
                    i += 1
                else:
                    cmd_base.append(arg)
                    i += 1

            def run_single_seq(seq_name):
                cmd = cmd_base + ["--sequences", seq_name, "--processes", "0"]
                print(f" 🚀 Spawning process for {seq_name}...")
                res = subprocess.run(cmd, capture_output=True, text=True)
                return seq_name, res.returncode, res.stdout, res.stderr

            with ThreadPoolExecutor(max_workers=args.processes) as executor:
                futures = [executor.submit(run_single_seq, seq) for seq in seq_list]
                for fut in futures:
                    seq_name, code, out, err = fut.result()
                    if code == 0:
                        print(f"✅ Process for {seq_name} finished successfully")
                    else:
                        print(
                            f"❌ Process for {seq_name} failed with code {code}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
                        )

            elapsed = time.perf_counter() - start_t
            fps_str = ""
            if total_frames > 0 and elapsed > 0:
                fps_str = f" (Aggregated Throughput: {total_frames / elapsed:.2f} FPS across {total_frames} frames)"
            print(
                f"\n📈 [Multi-Process Launcher] Completed all sequences in {elapsed:.2f}s{fps_str}"
            )

            # Run metrics in the parent process
            from saccade.perception.eval.metrics import run_motmetrics_evaluation

            metrics = run_motmetrics_evaluation(
                data_root=args.data_root,
                split=args.split,
                output=str(args.output),
                sequences=args.sequences,
            )
            if metrics:
                print("\n=== OVERALL METRICS ===")
                for k, v in metrics.items():
                    print(f"  {k}: {v}")
            sys.exit(0)

    _MODULE_KEYS = {
        "module_detection",
        "module_geometry",
        "module_motion",
        "module_reid",
        "module_semantic",
        "module_trigger",
        "module_lifecycle",
    }
    _MAMBA_KEYS = {"mamba_ckpt", "mamba_teacher_ckpt", "mamba_yolo_weights"}
    _VIS_KEYS = {
        "visualize",
        "visualize_scale",
        "visualize_fps",
        "no_visualize_score",
        "visualize_trail_len",
    }
    eval_kwargs = {
        k: v
        for k, v in vars(args).items()
        if k not in _MODULE_KEYS and k not in _MAMBA_KEYS and k not in _VIS_KEYS
    }

    if getattr(args, "mamba_ckpt", None):
        print(f"\n🧠 [Mamba] Loading Mamba head from {args.mamba_ckpt}")

        _tiling = getattr(args, "tiling", "native_640")
        if "1280" in _tiling:
            _img_sz = 1280
        elif "1024" in _tiling:
            _img_sz = 1024
        elif "960" in _tiling:
            _img_sz = 960
        else:
            _img_sz = 640

        # Single-stream MambaGatedDetector (Python, batch-1). For concurrent
        # multi-stream serving see scripts/benchmarks/multistream_mamba.py.
        from saccade.perception.temporal_yolo.mamba_gated_detector import (
            build_mamba_gated_detector,
        )

        mamba_detector = build_mamba_gated_detector(
            yolo_pt_path=args.mamba_yolo_weights,
            teacher_ckpt=args.mamba_teacher_ckpt,
            mamba_ckpt=args.mamba_ckpt,
            img_size=_img_sz,
            device="cuda",
            conf_thr=0.001,
            max_det=getattr(args, "max_det", 300),
            trt_backbone_engine=getattr(args, "fpn_backbone_engine", ""),
            temporal_T_override=0 if getattr(args, "no_temporal", False) else None,
            use_cuda_graph=getattr(args, "use_cuda_graph", False)
            and not (getattr(args, "cpp_threads", 0) > 0),
            use_whole_graph=getattr(args, "use_whole_graph", False)
            and not (getattr(args, "cpp_threads", 0) > 0),
            small_p3_max_threshold=getattr(args, "mamba_small_p3_max_threshold", 0.0),
        )

        from saccade.perception.temporal_yolo.mamba_gated_detector import (
            set_postprocess_compile,
        )

        if not getattr(args, "no_compile", False):
            set_postprocess_compile(True)
            mamba_detector.mamba_head.set_head_compile(True)
            mamba_detector.mamba_head.set_block_compile(True)
        else:
            set_postprocess_compile(False)

        eval_kwargs["detector"] = mamba_detector
        eval_kwargs["tiling"] = _tiling
        eval_kwargs["engine"] = "mamba"

    if getattr(args, "cpp_threads", 0) > 0:
        n = args.cpp_threads
        print(f"🚀 [C++ EvaluatorPool] Running with {n} threads")
        metrics = run_eval_cpp(
            engine=eval_kwargs.get("engine", args.engine),
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
    else:
        metrics = run_eval(**eval_kwargs)
        if metrics:
            print("\n=== OVERALL METRICS ===")
            for k, v in metrics.items():
                print(f"  {k}: {v}")

    if metrics:
        try:
            tags = {}
            if getattr(args, "preset", None):
                tags["preset"] = args.preset
            if getattr(args, "detector", None):
                tags["detector"] = args.detector
            # Capture the env-var escape hatches too — they bypass config and
            # would otherwise be invisible to the run record (some default on).
            from saccade.perception.eval.assoc_basis import resolved_env_overrides

            _params = {
                k: v
                for k, v in vars(args).items()
                if k not in _MODULE_KEYS and k not in _VIS_KEYS
            }
            _params.update({f"env.{k}": v for k, v in resolved_env_overrides().items()})
            log_eval_run(
                uri=args.mlflow_uri,
                experiment_name=args.mlflow_experiment,
                run_name=args.mlflow_run_name,
                params=_params,
                metrics=metrics,
                tags=tags,
            )
        except Exception as e:
            print(f"[mlflow] ERROR: {e}")

    if getattr(args, "visualize", False):
        print("\n🎬 [Visualization] Generating tracking videos...")
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "render_mot_result",
            project_root / "scripts" / "tools" / "render_mot_result.py",
        )
        render_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(render_mod)
        render_fn = render_mod.render

        # Determine which sequences were evaluated
        seq_list = []
        if args.sequences:
            seq_list = [s.strip() for s in args.sequences.split(",") if s.strip()]
        else:
            from pathlib import Path as _Path

            split_dir = _Path(args.data_root) / args.split
            if split_dir.exists():
                seq_list = sorted(d.name for d in split_dir.iterdir() if d.is_dir())

        for seq in seq_list:
            result_txt = Path(args.output) / f"{seq}.txt"
            seq_dir = Path(args.data_root) / args.split / seq
            output_mp4 = Path(args.output) / "renders" / f"{seq}_visualized.mp4"
            gt_txt = seq_dir / "gt" / "gt.txt"
            if not gt_txt.exists():
                gt_txt = None

            if not result_txt.exists():
                print(
                    f"⚠️ Tracking result not found for sequence {seq} at {result_txt}, skipping visualization."
                )
                continue

            print(f"🎥 Rendering {seq} ...")
            try:
                render_fn(
                    result_txt=result_txt,
                    seq_dir=seq_dir,
                    output_mp4=output_mp4,
                    gt_txt=gt_txt,
                    scale=args.visualize_scale,
                    fps=args.visualize_fps,
                    show_score=not args.no_visualize_score,
                    trail_len=args.visualize_trail_len,
                )
                print(f"✅ Saved video to: {output_mp4}")
            except Exception as e:
                print(f"❌ Failed to render {seq}: {e}")
