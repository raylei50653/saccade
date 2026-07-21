#!/usr/bin/env python3
"""Compatibility wrapper for scripts/eval/baselines/mot17_public.py."""
# status: stable

try:
    from scripts.eval._redirect import run_eval_script
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from _redirect import run_eval_script


if __name__ == "__main__":
    run_eval_script("baselines/mot17_public.py")
