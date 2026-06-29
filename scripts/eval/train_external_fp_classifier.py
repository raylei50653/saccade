#!/usr/bin/env python3
"""Compatibility wrapper for appearance/train_external_fp_classifier.py."""

try:
    from scripts.eval._redirect import run_eval_script
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from _redirect import run_eval_script

if __name__ != "__main__":
    from scripts.eval.appearance.train_external_fp_classifier import *  # noqa: F403


if __name__ == "__main__":
    run_eval_script("appearance/train_external_fp_classifier.py")
