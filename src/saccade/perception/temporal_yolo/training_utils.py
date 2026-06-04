"""
Shared training utilities for temporal YOLO scripts.

Import:
    from saccade.perception.temporal_yolo.training_utils import (
        save_checkpoint,
        load_checkpoint,
        strip_compiled_keys,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def strip_compiled_keys(sd: dict[str, Any]) -> dict[str, Any]:
    """Remove _orig_mod. prefix inserted by torch.compile, making
    checkpoints compile-agnostic."""
    return {k.replace("._orig_mod.", "."): v for k, v in sd.items()}


def save_checkpoint(
    state: dict[str, Any],
    run_dir: Path,
    epoch: int,
    is_best: bool = False,
) -> None:
    """Save state dict as latest.ckpt + epoch_N.ckpt, plus best.ckpt if is_best."""
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, run_dir / "latest.ckpt")
    torch.save(state, run_dir / f"epoch_{epoch:04d}.ckpt")
    if is_best:
        torch.save(state, run_dir / "best.ckpt")
    tag = " [BEST]" if is_best else ""
    print(f"  Saved epoch_{epoch:04d}.ckpt{tag}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = False,
) -> int:
    """Load checkpoint into model and optionally optimizer.

    Returns the next epoch to start from (resumed epoch + 1).
    Always strips _orig_mod. keys via strip_compiled_keys.
    Optimizer loading is best-effort — failure prints a warning.
    """
    print(f"[Resume] {path}  strict={strict}")
    state = torch.load(path, map_location="cpu", weights_only=False)

    sd = strip_compiled_keys(state.get("model", state.get("student", {})))
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing:
        print(
            f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)})")

    if optimizer is not None:
        if "optimizer" in state:
            try:
                optimizer.load_state_dict(state["optimizer"])
            except Exception:
                print("  [Warn] Optimizer state not loaded")
        else:
            print("  [Warn] No optimizer state in checkpoint")

    prev_epoch = state.get("epoch", 0)
    best_loss = state.get("best_loss", float("inf"))
    print(f"  Resumed from epoch {prev_epoch}  best_loss={best_loss:.4f}")
    return prev_epoch + 1
