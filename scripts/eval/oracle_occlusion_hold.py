#!/usr/bin/env python3
"""Compatibility wrapper for scripts/eval/experiments/oracle_occlusion_hold.py."""
# status: stable

try:
    from scripts.eval._redirect import run_eval_script
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from _redirect import run_eval_script

if __name__ != "__main__":
    from scripts.eval.experiments.oracle_occlusion_hold import *  # noqa: F403


if __name__ == "__main__":
    run_eval_script("experiments/oracle_occlusion_hold.py")
