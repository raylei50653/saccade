#!/usr/bin/env python3
# mypy: ignore-errors
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


import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from mot17_args import build_parser  # noqa: E402

if __name__ == "__main__":
    parser = build_parser()
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
    metrics = run_eval(**vars(args))
    if metrics:
        print("\n=== OVERALL METRICS ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
