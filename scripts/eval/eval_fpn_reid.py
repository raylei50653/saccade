#!/usr/bin/env python3
"""Compatibility wrapper for scripts/eval/appearance/eval_fpn_reid.py."""
# status: stable

try:
    from scripts.eval._redirect import run_eval_script
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from _redirect import run_eval_script

if __name__ != "__main__":
    from scripts.eval.appearance.eval_fpn_reid import *  # noqa: F403


if __name__ == "__main__":
    run_eval_script("appearance/eval_fpn_reid.py")
