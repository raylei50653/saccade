#!/usr/bin/env python3
# mypy: ignore-errors
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
build_path = project_root / "build"
if build_path.exists():
    sys.path.insert(0, str(build_path))

from mot17 import _load_config_defaults  # noqa: E402
from mot17_args import build_parser  # noqa: E402
from saccade.perception.eval.runner import run_eval  # noqa: E402


def main() -> int:
    parser = build_parser()
    parser.add_argument(
        "--profiler-output",
        default="runs/torch_profiler",
        help="Directory for chrome trace and summarized tables.",
    )
    parser.add_argument(
        "--profiler-topk",
        type=int,
        default=40,
        help="Number of rows to print from profiler summaries.",
    )
    config_defaults = _load_config_defaults(project_root)
    if config_defaults:
        parser.set_defaults(**config_defaults)
    args = parser.parse_args()

    _module_keys = {
        "module_detection",
        "module_geometry",
        "module_reid",
        "module_semantic",
        "module_trigger",
        "module_lifecycle",
    }
    eval_kwargs = {k: v for k, v in vars(args).items() if k not in _module_keys}
    profiler_output = Path(eval_kwargs.pop("profiler_output"))
    profiler_topk = int(eval_kwargs.pop("profiler_topk"))
    profiler_output.mkdir(parents=True, exist_ok=True)

    trace_path = profiler_output / "trace.json"
    cpu_table_path = profiler_output / "cpu_table.txt"
    cuda_table_path = profiler_output / "cuda_table.txt"

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        run_eval(**eval_kwargs)

    cpu_table = prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=profiler_topk,
    )
    cuda_table = prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=profiler_topk,
    )
    prof.export_chrome_trace(str(trace_path))
    cpu_table_path.write_text(cpu_table)
    cuda_table_path.write_text(cuda_table)

    print(f"\n[torch.profiler] Chrome trace: {trace_path}")
    print(f"[torch.profiler] CPU table: {cpu_table_path}")
    print(f"[torch.profiler] CUDA table: {cuda_table_path}\n")
    print(cuda_table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
