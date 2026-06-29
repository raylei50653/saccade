#!/usr/bin/env python3
"""Compatibility wrapper for scripts/eval/appearance/jde_market1501.py."""

try:
    from scripts.eval._redirect import run_eval_script
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from _redirect import run_eval_script

if __name__ != "__main__":
    from scripts.eval.appearance.jde_market1501 import *  # noqa: F403


if __name__ == "__main__":
    run_eval_script("appearance/jde_market1501.py")
