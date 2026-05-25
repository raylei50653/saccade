from __future__ import annotations

import os
import re
import subprocess
import time
from typing import Any


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.path.dirname(__file__),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _git_branch() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=os.path.dirname(__file__),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _parse_percent(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.match(r"([\d.]+)%", value)
        if m:
            return float(m.group(1))
    return None


def log_eval_run(
    uri: str,
    experiment_name: str,
    run_name: str | None,
    params: dict[str, Any],
    metrics: dict[str, Any],
    tags: dict[str, str] | None = None,
    nested: bool = False,
) -> bool:
    try:
        import mlflow
    except ImportError:
        print("[mlflow] mlflow not installed, skipping.")
        return False

    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)

        run_name = run_name or f"run-{time.strftime('%Y%m%d-%H%M%S')}"

        mlflow.start_run(run_name=run_name, nested=nested)

        safe_params = {
            k: v for k, v in params.items() if v is not None and k != "kwargs"
        }
        try:
            mlflow.log_params(safe_params)
        except Exception as e:
            print(f"[mlflow] WARNING: log_params failed: {e}")

        float_metrics: dict[str, float] = {}
        for k, v in metrics.items():
            pct = _parse_percent(v)
            if pct is not None:
                float_metrics[k] = pct
            elif isinstance(v, (int, float)):
                float_metrics[k] = float(v)

        try:
            mlflow.log_metrics(float_metrics)
        except Exception as e:
            print(f"[mlflow] WARNING: log_metrics failed: {e}")

        all_tags = {
            "git_commit": _git_commit(),
            "git_branch": _git_branch(),
        }
        if tags:
            all_tags.update(tags)
        try:
            mlflow.set_tags(all_tags)
        except Exception as e:
            print(f"[mlflow] WARNING: set_tags failed: {e}")

        mlflow.end_run()
        print(
            f"[mlflow] Logged to {uri} — experiment={experiment_name}, run={run_name}"
        )
        return True
    except Exception as e:
        print(f"[mlflow] Skipping (server unavailable): {e}")
        return False
